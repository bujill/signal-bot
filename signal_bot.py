#!/usr/bin/env python3
"""
บอทส่งสัญญาณซื้อแบบ "ตามเทรนด์" สำหรับหุ้น SET50 + คริปโต Top 10 → Discord

กลยุทธ์ (ดูกราฟรายวัน):
  สัญญาณซื้อ เมื่อเกิดครบทุกข้อในวันเดียวกัน
    1) EMA20 ตัดขึ้นเหนือ EMA50          (เทรนด์ระยะกลางเริ่มเป็นขาขึ้น)
    2) ราคาปิด > EMA200                 (เทรนด์ใหญ่ยังเป็นขาขึ้น)
    3) วอลุ่ม > 1.2 เท่าของค่าเฉลี่ย 20 วัน (มีแรงซื้อยืนยัน)
    4) RSI(14) < 75                     (ยังไม่ร้อนแรงเกินไป)
  จุดตัดขาดทุน  = ราคาซื้อ − 2 × ATR(14)        (ระยะนี้เรียกว่า 1R)
  เป้าขาย 1     = ราคาซื้อ + 2R
  เป้าขาย 2     = ราคาซื้อ + 3R
  สัญญาณออก     = EMA20 ตัดลงใต้ EMA50

วิธีใช้:
  python signal_bot.py --market set              # เช็กหุ้นไทยวันนี้
  python signal_bot.py --market crypto           # เช็กคริปโต (แท่งเมื่อวานที่ปิดแล้ว)
  python signal_bot.py --market all --days 10 --ignore-date   # ดูสัญญาณย้อนหลัง 10 แท่ง (ไว้ทดสอบ)
  python signal_bot.py --market all --backtest   # ทดสอบแบบพอร์ตจริงย้อนหลัง 10 ปี

ถ้าไม่ได้ตั้งค่า DISCORD_WEBHOOK_URL บอทจะพิมพ์ผลออกหน้าจอแทน
ข้อมูลราคามาจาก Yahoo Finance (ฟรี) — ไม่ใช่คำแนะนำการลงทุน
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

# ============================================================
# ตั้งค่า (แก้ได้ตามต้องการ)
# ============================================================

# SET50 รอบ 1 ก.ค. – 31 ธ.ค. 2026 (ตลาดหลักทรัพย์ปรับรายชื่อทุก ม.ค. และ ก.ค.)
SET50 = [
    "ADVANC", "AOT", "AWC", "BANPU", "BBL", "BCP", "BDMS", "BEM", "BH", "BJC",
    "CCET", "COM7", "CPALL", "CPF", "CPN", "CRC", "DELTA", "EGCO", "GPSC", "GULF",
    "HMPRO", "ITC", "IVL", "KBANK", "KKP", "KTB", "KTC", "LH", "MINT", "MRDIYT",
    "MTC", "OR", "PTT", "PTTEP", "PTTGC", "RATCH", "SCB", "SCC", "SCGP", "TCAP",
    "TFG", "THAI", "TIDLOR", "TISCO", "TLI", "TOP", "TRUE", "TTB", "TU", "WHA",
]

# คริปโตมูลค่าตลาดสูงสุด (ไม่รวม stablecoin) — อันดับเปลี่ยนได้ แก้รายชื่อได้ตามใจ
CRYPTO = ["BTC", "ETH", "XRP", "BNB", "SOL", "DOGE", "ADA", "TRX", "LINK", "AVAX"]

PARAMS = {
    "fast": 20,        # EMA เร็ว
    "slow": 50,        # EMA ช้า
    "trend": 200,      # EMA เทรนด์ใหญ่
    "vol_window": 20,  # ค่าเฉลี่ยวอลุ่ม
    "vol_mult": 1.2,   # วอลุ่มต้องมากกว่ากี่เท่าของค่าเฉลี่ย
    "rsi_len": 14,
    "rsi_max": 75,
    "atr_len": 14,
    "atr_stop": 2.0,   # จุดตัดขาดทุน = ราคา − 2×ATR  (= 1R)
    "target1_r": 2.0,  # เป้าขาย 1 = ราคา + 2R
    "target2_r": 3.0,  # เป้าขาย 2 = ราคา + 3R
}

# ค่าธรรมเนียม + slippage โดยประมาณต่อขา (ซื้อ หรือ ขาย) ใช้ใน backtest
FEES = {"set": 0.0020, "crypto": 0.0015}

# backtest แบบพอร์ต: ถือพร้อมกันได้สูงสุดกี่ตัว (แต่ละตัวใช้เงิน 1/N ของพอร์ต)
SLOTS = {"set": 10, "crypto": 5}
BACKTEST_YEARS = 10
START_CAPITAL = 100_000

# ดัชนีอ้างอิงสำหรับเทียบผล
BENCHMARK = {"set": ("^SET.BK", "ดัชนี SET"), "crypto": ("BTC-USD", "ซื้อ BTC ถือไว้")}

# วิธีขายที่ใช้เทียบใน backtest
EXIT_MODES = {
    "trend": "ถือจนมีสัญญาณออก (EMA ตัดลง)",
    "half":  "ขายครึ่งที่เป้า 1 · ที่เหลือถือจนสัญญาณออก",
    "t1":    "ขายทั้งหมดที่เป้า 1 (2R)",
    "t2":    "ขายทั้งหมดที่เป้า 2 (3R)",
}

# แผนขายที่แสดงในข้อความสัญญาณ — หลังดูผล backtest แล้ว เปลี่ยนเป็นวิธีที่ได้ผลดีที่สุดได้
SELL_PLAN = "half"  # เลือกจาก: trend, half, t1, t2
SELL_PLAN_TEXT = {
    "trend": "ถือไปเรื่อย ๆ จนมีสัญญาณออก (เป้าขายใช้ดูเป็นระยะทางเท่านั้น)",
    "half":  "ขายครึ่งที่เป้า 1 แล้วเลื่อนจุดตัดขาดทุนมาที่ราคาทุน · ที่เหลือถือจนมีสัญญาณออก",
    "t1":    "ขายทั้งหมดเมื่อถึงเป้า 1",
    "t2":    "ขายทั้งหมดเมื่อถึงเป้า 2",
}

BKK = ZoneInfo("Asia/Bangkok")
MARKET_NAME = {"set": "หุ้นไทย SET50", "crypto": "คริปโต"}
GREEN, RED, BLUE, GREY = 0x2ECC71, 0xE74C3C, 0x3498DB, 0x95A5A6
DISCLAIMER = "สัญญาณจากสูตรเทคนิคอัตโนมัติ ไม่ใช่คำแนะนำการลงทุน — ตัดสินใจและบริหารความเสี่ยงเอง"


# ============================================================
# ข้อมูลราคา
# ============================================================

def to_yf(symbol: str, market: str) -> str:
    return f"{symbol}.BK" if market == "set" else f"{symbol}-USD"


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Open", "High", "Low", "Close"]).copy()
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df = df[~df.index.duplicated(keep="last")]
    # กันข้อมูลเสีย: ราคา 0 หรือ High < Low
    df = df[(df["Close"] > 0) & (df["Low"] > 0) & (df["High"] >= df["Low"])]
    return df


def load_data(symbols: list[str], market: str, years: float) -> dict[str, pd.DataFrame]:
    """ดึงราคารายวันจาก Yahoo Finance คืนค่า {symbol: DataFrame(Open,High,Low,Close,Volume)}"""
    import yfinance as yf

    start = (dt.date.today() - dt.timedelta(days=int(years * 365.25) + 5)).isoformat()
    yf_map = {to_yf(s, market): s for s in symbols}
    raw = yf.download(
        list(yf_map), start=start, interval="1d", group_by="ticker",
        auto_adjust=True, threads=True, progress=False,
    )
    out: dict[str, pd.DataFrame] = {}
    for yft, sym in yf_map.items():
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                if yft not in raw.columns.get_level_values(0):
                    continue
                df = raw[yft]
            else:
                df = raw
            df = _clean(df)
            if not df.empty:
                out[sym] = df
        except Exception as e:  # ตัวไหนพังก็ข้ามไป ไม่ให้ทั้งบอทล่ม
            print(f"[warn] โหลด {sym} ไม่ได้: {e}", file=sys.stderr)
    return out


def load_benchmark(market: str, years: float) -> pd.Series | None:
    import yfinance as yf

    ticker = BENCHMARK[market][0]
    try:
        start = (dt.date.today() - dt.timedelta(days=int(years * 365.25) + 5)).isoformat()
        df = yf.download(ticker, start=start, interval="1d", auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        s = _clean(df)["Close"]
        return s if len(s) > 50 else None
    except Exception as e:
        print(f"[warn] โหลดดัชนี {ticker} ไม่ได้: {e}", file=sys.stderr)
        return None


def drop_unfinished_crypto_bar(df: pd.DataFrame) -> pd.DataFrame:
    """แท่งรายวันของคริปโตปิดตอน 00:00 UTC ตัดแท่งของวันนี้ (ยังไม่ปิด) ทิ้ง"""
    today_utc = dt.datetime.now(dt.timezone.utc).date()
    return df[df.index.date < today_utc]


# ============================================================
# อินดิเคเตอร์ + สัญญาณ
# ============================================================

def add_indicators(df: pd.DataFrame, p: dict = PARAMS) -> pd.DataFrame:
    d = df.copy()
    c = d["Close"]
    d["ema_fast"] = c.ewm(span=p["fast"], adjust=False).mean()
    d["ema_slow"] = c.ewm(span=p["slow"], adjust=False).mean()
    d["ema_trend"] = c.ewm(span=p["trend"], adjust=False).mean()

    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / p["rsi_len"], adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / p["rsi_len"], adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    d["rsi"] = (100 - 100 / (1 + rs)).fillna(100)

    prev_c = c.shift(1)
    tr = pd.concat([d["High"] - d["Low"], (d["High"] - prev_c).abs(), (d["Low"] - prev_c).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1 / p["atr_len"], adjust=False).mean()

    d["vol_ma"] = d["Volume"].rolling(p["vol_window"]).mean()
    d["vol_x"] = d["Volume"] / d["vol_ma"].replace(0, np.nan)

    above = d["ema_fast"] > d["ema_slow"]
    cross_up = above & ~above.shift(1, fill_value=True)
    cross_down = ~above & above.shift(1, fill_value=False)
    # ถ้าไม่มีข้อมูลวอลุ่ม (เป็น 0/NaN) ให้ผ่านเงื่อนไขวอลุ่มไป
    vol_ok = d["vol_x"].isna() | (d["vol_x"] > p["vol_mult"])

    warm = np.arange(len(d)) >= p["trend"]  # ต้องมีข้อมูลพอให้ EMA200 นิ่งก่อน
    d["buy"] = warm & cross_up & (c > d["ema_trend"]) & vol_ok & (d["rsi"] < p["rsi_max"])
    d["exit"] = warm & cross_down
    return d


def levels(price: float, atr: float, p: dict = PARAMS) -> dict:
    """จุดตัดขาดทุนและเป้าขาย จากราคาซื้อ"""
    r = p["atr_stop"] * atr
    return {
        "stop": price - r,
        "t1": price + p["target1_r"] * r,
        "t2": price + p["target2_r"] * r,
        "r": r,
    }


def target_history(ind: pd.DataFrame, max_bars: int = 250) -> dict:
    """ดูสัญญาณซื้อในอดีตของหุ้นตัวนี้ (ไม่รวมแท่งสุดท้าย):
    ซื้อราคาเปิดวันถัดไป แล้วนับว่าราคาไปถึงเป้า 1 / เป้า 2 ก่อนโดนจุดตัดขาดทุนกี่ครั้ง"""
    o, h, l = (ind[k].to_numpy() for k in ("Open", "High", "Low"))
    atr, buy = ind["atr"].to_numpy(), ind["buy"].to_numpy()
    res = {"n": 0, "t1": 0, "t2": 0, "days_t1": []}
    for i in np.flatnonzero(buy[:-1]):
        e = i + 1
        lv = levels(o[e], atr[i])
        hit1 = hit2 = False
        for j in range(e, min(e + max_bars, len(o))):
            if l[j] <= lv["stop"]:  # ถ้าวันเดียวกันโดนทั้งสองอย่าง นับว่าโดนตัดขาดทุนก่อน (มองแง่ร้าย)
                break
            if not hit1 and h[j] >= lv["t1"]:
                hit1 = True
                res["days_t1"].append(j - e + 1)
            if h[j] >= lv["t2"]:
                hit2 = True
                break
        else:
            if e + max_bars > len(o) and not hit1:  # ยังไม่รู้ผล ไม่นับ
                continue
        res["n"] += 1
        res["t1"] += hit1
        res["t2"] += hit2
    return res


def scan(data: dict[str, pd.DataFrame], market: str, days: int, ignore_date: bool):
    """หาสัญญาณใน N แท่งล่าสุด คืนค่า (signals, info)"""
    today_bkk = dt.datetime.now(BKK).date()
    signals, too_short, stale = [], [], 0
    for sym, df in data.items():
        if market == "crypto":
            df = drop_unfinished_crypto_bar(df)
        if len(df) < PARAMS["trend"] + 10:
            too_short.append(sym)
            continue
        if market == "set" and not ignore_date and df.index[-1].date() != today_bkk:
            stale += 1  # วันนี้ยังไม่มีแท่งใหม่ (วันหยุด หรือข้อมูลยังไม่อัปเดต)
            continue
        ind = add_indicators(df)
        tail = ind.tail(days)
        if not (tail["buy"].any() or tail["exit"].any()):
            continue
        for ts, row in tail.iterrows():
            if not (row["buy"] or row["exit"]):
                continue
            close, atr = float(row["Close"]), float(row["atr"])
            sig = {
                "symbol": sym, "market": market, "date": ts.date(),
                "kind": "buy" if row["buy"] else "exit",
                "close": close, "rsi": float(row["rsi"]),
                "vol_x": None if pd.isna(row["vol_x"]) else float(row["vol_x"]),
                **levels(close, atr),
            }
            if sig["kind"] == "buy":
                sig["hist"] = target_history(ind.loc[:ts])
            signals.append(sig)
    signals.sort(key=lambda s: (s["kind"] != "buy", s["date"], s["symbol"]))
    info = {"checked": len(data), "too_short": too_short, "stale": stale}
    return signals, info


# ============================================================
# Backtest แบบพอร์ตจริง
# ============================================================

def _prepare(frames: dict[str, pd.DataFrame], market: str):
    """จัดข้อมูลทุกตัวให้อยู่บนปฏิทินเดียวกัน"""
    inds = {}
    for sym, df in frames.items():
        if market == "crypto":
            df = drop_unfinished_crypto_bar(df)
        if len(df) < PARAMS["trend"] + 20:
            continue
        ind = add_indicators(df)
        # สัญญาณเกิดตอนปิดตลาดวันนี้ → ซื้อได้ราคาเปิด "วันทำการถัดไปของหุ้นตัวนั้น"
        ind["enter"] = ind["buy"].shift(1, fill_value=False)
        ind["atr_sig"] = ind["atr"].shift(1)
        ind["rank_sig"] = ind["vol_x"].shift(1).fillna(1.0)
        inds[sym] = ind
    if not inds:
        return None
    dates = sorted(set().union(*[set(i.index) for i in inds.values()]))
    dates = pd.DatetimeIndex(dates)
    arr = {}
    for sym, ind in inds.items():
        r = ind.reindex(dates)
        arr[sym] = {
            "o": r["Open"].to_numpy(float), "h": r["High"].to_numpy(float),
            "l": r["Low"].to_numpy(float), "c": r["Close"].to_numpy(float),
            "c_ff": r["Close"].ffill().to_numpy(float),
            "enter": r["enter"].fillna(False).to_numpy(bool),
            "exit": r["exit"].fillna(False).to_numpy(bool),
            "atr": r["atr_sig"].to_numpy(float),
            "rank": r["rank_sig"].fillna(0).to_numpy(float),
        }
    return dates, arr


def simulate(prepared, market: str, mode: str, start: pd.Timestamp) -> dict:
    dates, arr = prepared
    fee, slots = FEES[market], SLOTS[market]
    i0 = int(np.searchsorted(dates, start))
    cash, equity_prev = 1.0, 1.0
    pos: dict[str, dict] = {}
    trades, curve, invested = [], [], []

    def close_part(sym, frac, px, i):
        nonlocal cash
        p = pos[sym]
        sh = p["shares"] * frac
        cash += sh * px * (1 - fee)
        p["proceeds"] += sh * px * (1 - fee)
        p["shares"] -= sh
        if p["shares"] <= 1e-12:
            trades.append({"ret": p["proceeds"] / p["cost"] - 1, "days": i - p["i"], "sym": sym})
            del pos[sym]

    for i in range(i0, len(dates)):
        # 1) ซื้อที่ราคาเปิด ตามสัญญาณของเมื่อวาน
        cands = [s for s, a in arr.items()
                 if a["enter"][i] and s not in pos and not np.isnan(a["o"][i]) and not np.isnan(a["atr"][i])]
        cands.sort(key=lambda s: -arr[s]["rank"][i])
        slot_value = equity_prev / slots
        for s in cands:
            if len(pos) >= slots:
                break
            amount = min(slot_value, cash)
            if amount < slot_value * 0.5:
                break
            a = arr[s]
            px = a["o"][i]
            lv = levels(px, a["atr"][i])
            cash -= amount
            pos[s] = {"shares": amount * (1 - fee) / px, "cost": amount, "proceeds": 0.0, "i": i,
                      "entry": px, "stop": lv["stop"], "t1": lv["t1"], "t2": lv["t2"], "half": False}

        # 2) ระหว่างวัน: จุดตัดขาดทุน / เป้าขาย  3) ตอนปิด: สัญญาณออก
        for s in list(pos):
            a, p = arr[s], pos[s]
            o, h, l, c = a["o"][i], a["h"][i], a["l"][i], a["c"][i]
            if np.isnan(o):
                continue
            if l <= p["stop"]:  # ถ้าวันเดียวกันโดนทั้งจุดตัดขาดทุนและเป้า นับว่าโดนตัดขาดทุนก่อน
                close_part(s, 1.0, min(o, p["stop"]), i)
                continue
            if mode == "t1" and h >= p["t1"]:
                close_part(s, 1.0, max(o, p["t1"]), i)
                continue
            if mode == "t2" and h >= p["t2"]:
                close_part(s, 1.0, max(o, p["t2"]), i)
                continue
            if mode == "half" and not p["half"] and h >= p["t1"]:
                close_part(s, 0.5, max(o, p["t1"]), i)
                p["half"] = True
                p["stop"] = max(p["stop"], p["entry"])  # เลื่อนจุดตัดขาดทุนมาที่ทุน
            if a["exit"][i]:
                close_part(s, 1.0, c, i)

        equity = cash + sum(p["shares"] * arr[s]["c_ff"][i] for s, p in pos.items())
        curve.append(equity)
        invested.append(1 - cash / equity if equity > 0 else 0)
        equity_prev = equity

    open_trades = len(pos)
    curve = pd.Series(curve, index=dates[i0:])
    return {"curve": curve, "trades": trades, "open": open_trades, "exposure": float(np.mean(invested)) if invested else 0}


def curve_stats(curve: pd.Series) -> dict:
    curve = curve.dropna()
    years = max((curve.index[-1] - curve.index[0]).days / 365.25, 1e-9)
    total = curve.iloc[-1] / curve.iloc[0] - 1
    cagr = (curve.iloc[-1] / curve.iloc[0]) ** (1 / years) - 1
    dd = (curve / curve.cummax() - 1).min()
    yearly = curve.resample("YE").last()
    yearly = pd.concat([pd.Series([curve.iloc[0]], index=[curve.index[0]]), yearly])
    yr = yearly.pct_change().dropna()
    return {"total": total, "cagr": cagr, "maxdd": float(dd), "years": years,
            "yearly": {d.year: float(v) for d, v in yr.items()}}


def equal_weight_curve(prepared, start) -> pd.Series:
    """ซื้อทุกตัวเท่า ๆ กันแล้วถือ (ปรับน้ำหนักเท่ากันทุกวัน) — ไว้เทียบ"""
    dates, arr = prepared
    closes = pd.DataFrame({s: a["c_ff"] for s, a in arr.items()}, index=dates)
    rets = closes.pct_change().loc[start:]
    rets = rets.where(closes.shift(1).loc[start:].notna())
    daily = rets.mean(axis=1).fillna(0)
    return (1 + daily).cumprod()


def run_backtest(frames, market: str, bench: pd.Series | None, years: float = BACKTEST_YEARS) -> dict:
    prepared = _prepare(frames, market)
    if prepared is None:
        return {}
    dates = prepared[0]
    start = max(dates[0] + pd.Timedelta(days=1), pd.Timestamp(dt.date.today()) - pd.DateOffset(years=years))
    # ต้องมีอย่างน้อย 1 ตัวที่ผ่านช่วงอุ่นเครื่อง EMA200 แล้ว
    start = max(start, dates[min(PARAMS["trend"], len(dates) - 1)])
    out = {"market": market, "symbols": len(prepared[1]), "start": start, "end": dates[-1], "modes": {}}
    for mode in EXIT_MODES:
        sim = simulate(prepared, market, mode, start)
        t = sim["trades"]
        rets = [x["ret"] for x in t]
        out["modes"][mode] = {
            **curve_stats(sim["curve"]),
            "trades": len(t), "open": sim["open"],
            "win": float(np.mean([r > 0 for r in rets])) if rets else 0.0,
            "avg": float(np.mean(rets)) if rets else 0.0,
            "avg_win": float(np.mean([r for r in rets if r > 0])) if any(r > 0 for r in rets) else 0.0,
            "avg_loss": float(np.mean([r for r in rets if r <= 0])) if any(r <= 0 for r in rets) else 0.0,
            "days": float(np.median([x["days"] for x in t])) if t else 0.0,
            "exposure": sim["exposure"],
            "best": max(t, key=lambda x: x["ret"]) if t else None,
            "worst": min(t, key=lambda x: x["ret"]) if t else None,
        }
    out["equal_weight"] = curve_stats(equal_weight_curve(prepared, start))
    if bench is not None and len(bench.loc[start:]) > 50:
        out["bench"] = curve_stats(bench.loc[start:])
    return out


# ============================================================
# Discord
# ============================================================

def fmt_price(x: float, market: str) -> str:
    if market == "set":
        return f"฿{x:,.2f}"
    return f"${x:,.4f}" if x < 1 else f"${x:,.2f}"


def pct(x: float, digits: int = 0) -> str:
    return f"{x * 100:+.{digits}f}%"


def chart_url(sym: str, market: str) -> str:
    tv = f"SET:{sym}" if market == "set" else f"BINANCE:{sym}USDT"
    return f"https://www.tradingview.com/chart/?symbol={tv}"


def signal_embed(s: dict) -> dict:
    m = s["market"]
    if s["kind"] == "buy":
        c = s["close"]
        title, color = f"🟢 สัญญาณซื้อ {s['symbol']}", GREEN
        fields = [
            {"name": "ราคาปิด", "value": fmt_price(c, m), "inline": True},
            {"name": "🎯 เป้าขาย 1 (2R)", "value": f"{fmt_price(s['t1'], m)} ({pct(s['t1'] / c - 1, 1)})", "inline": True},
            {"name": "🎯 เป้าขาย 2 (3R)", "value": f"{fmt_price(s['t2'], m)} ({pct(s['t2'] / c - 1, 1)})", "inline": True},
            {"name": "🛑 จุดตัดขาดทุน", "value": f"{fmt_price(s['stop'], m)} ({pct(s['stop'] / c - 1, 1)})", "inline": True},
            {"name": "RSI(14)", "value": f"{s['rsi']:.0f}", "inline": True},
            {"name": "วอลุ่ม", "value": "—" if s["vol_x"] is None else f"{s['vol_x']:.1f}× ค่าเฉลี่ย", "inline": True},
        ]
        h = s.get("hist")
        if h and h["n"] > 0:
            med = f" · ใช้เวลาประมาณ {int(np.median(h['days_t1']))} วัน" if h["days_t1"] else ""
            fields.append({
                "name": f"สถิติ {s['symbol']} ในอดีต ({h['n']} สัญญาณ)",
                "value": (f"ถึงเป้า 1 ก่อนโดนตัดขาดทุน {h['t1'] / h['n'] * 100:.0f}%{med}\n"
                          f"ถึงเป้า 2 ก่อนโดนตัดขาดทุน {h['t2'] / h['n'] * 100:.0f}%"
                          + ("\n⚠️ จำนวนครั้งน้อย ใช้อ้างอิงได้จำกัด" if h["n"] < 5 else "")),
                "inline": False,
            })
        fields.append({"name": "แผนขาย", "value": SELL_PLAN_TEXT[SELL_PLAN], "inline": False})
    else:
        title, color = f"🔴 สัญญาณออก {s['symbol']}", RED
        fields = [
            {"name": "ราคาปิด", "value": fmt_price(s["close"], m), "inline": True},
            {"name": "เหตุผล", "value": "EMA20 ตัดลงใต้ EMA50 — ถ้ายังถืออยู่ ควรพิจารณาขาย", "inline": False},
        ]
    return {
        "title": title, "url": chart_url(s["symbol"], m), "color": color,
        "description": f"{MARKET_NAME[m]} · แท่งวันที่ {s['date']:%d/%m/%Y}",
        "fields": fields,
    }


def backtest_embed(bt: dict) -> dict:
    m = bt["market"]
    cap = START_CAPITAL
    cur = "฿" if m == "set" else "$"
    fields = []
    best_mode = max(bt["modes"], key=lambda k: bt["modes"][k]["cagr"])
    for mode, label in EXIT_MODES.items():
        r = bt["modes"][mode]
        star = " ⭐" if mode == best_mode else ""
        fields.append({
            "name": f"{label}{star}",
            "value": (f"ผลตอบแทนรวม **{pct(r['total'])}** · เฉลี่ยต่อปี **{pct(r['cagr'], 1)}**\n"
                      f"ทุน {cur}{cap:,.0f} → {cur}{cap * (1 + r['total']):,.0f} · ติดลบหนักสุด {pct(r['maxdd'])}\n"
                      f"{r['trades']} เทรด · ชนะ {r['win'] * 100:.0f}% · กำไรเฉลี่ยตอนชนะ {pct(r['avg_win'], 1)} "
                      f"/ ขาดทุนเฉลี่ยตอนแพ้ {pct(r['avg_loss'], 1)} · ถือราว {r['days']:.0f} วัน"),
            "inline": False,
        })
    ew = bt["equal_weight"]
    ref = f"ซื้อทุกตัวเท่ากันแล้วถือ: รวม **{pct(ew['total'])}** · ต่อปี **{pct(ew['cagr'], 1)}** · ติดลบหนักสุด {pct(ew['maxdd'])}"
    if "bench" in bt:
        b = bt["bench"]
        ref += f"\n{BENCHMARK[m][1]}: รวม **{pct(b['total'])}** · ต่อปี **{pct(b['cagr'], 1)}** · ติดลบหนักสุด {pct(b['maxdd'])}"
    fields.append({"name": "📏 เทียบกับการซื้อแล้วถือ", "value": ref, "inline": False})

    yrs = bt["modes"][best_mode]["yearly"]
    if yrs:
        fields.append({"name": f"รายปี (วิธี ⭐)", "value": " · ".join(f"{y}: {pct(v)}" for y, v in yrs.items())[:1024],
                       "inline": False})
    return {
        "title": f"🧪 Backtest {MARKET_NAME[m]} · {bt['start']:%m/%Y} – {bt['end']:%m/%Y}",
        "color": BLUE,
        "description": (f"จำลองพอร์ตจริง: ทุนเริ่ม {cur}{cap:,.0f} · ถือพร้อมกันสูงสุด {SLOTS[m]} ตัว (ตัวละ 1/{SLOTS[m]} ของพอร์ต)\n"
                        f"ซื้อราคาเปิดวันถัดจากสัญญาณ · หักค่าธรรมเนียม+slippage {FEES[m] * 100:.2f}% ต่อขา · {bt['symbols']} ตัว"),
        "fields": fields,
        "footer": {"text": "ใช้รายชื่อปัจจุบันย้อนอดีต ผลจึงดูดีกว่าจริง (survivorship bias) · ไม่รวมปันผล/ภาษี · ผลในอดีตไม่รับประกันอนาคต"},
    }


def backtest_text(bt: dict) -> str:
    """สรุปแบบตัวอักษร (พิมพ์ลง log ของ GitHub Actions ด้วย)"""
    lines = [f"BACKTEST {bt['market']} {bt['start']:%Y-%m-%d} → {bt['end']:%Y-%m-%d} symbols={bt['symbols']}"]
    for mode, r in bt["modes"].items():
        lines.append(f"  {mode:<6} total={pct(r['total']):>7} cagr={pct(r['cagr'], 1):>7} maxdd={pct(r['maxdd']):>5} "
                     f"trades={r['trades']:>4} win={r['win'] * 100:.0f}% avg={pct(r['avg'], 2)} days={r['days']:.0f} "
                     f"expo={r['exposure'] * 100:.0f}%")
    ew = bt["equal_weight"]
    lines.append(f"  equal-weight buy&hold total={pct(ew['total'])} cagr={pct(ew['cagr'], 1)} maxdd={pct(ew['maxdd'])}")
    if "bench" in bt:
        b = bt["bench"]
        lines.append(f"  {BENCHMARK[bt['market']][0]} total={pct(b['total'])} cagr={pct(b['cagr'], 1)} maxdd={pct(b['maxdd'])}")
    return "\n".join(lines)


def post_discord(webhook: str | None, content: str, embeds: list[dict]) -> None:
    if not webhook:
        print("\n" + "=" * 60 + "\n" + content)
        for e in embeds:
            print(json.dumps(e, ensure_ascii=False, indent=1))
        return
    chunks = [embeds[i:i + 10] for i in range(0, len(embeds), 10)] or [[]]  # Discord รับได้ 10 embed/ข้อความ
    for n, chunk in enumerate(chunks):
        payload = {"content": (content or None) if n == 0 else None, "embeds": chunk, "username": "Signal Bot"}
        for _ in range(3):
            r = requests.post(webhook, json=payload, timeout=20)
            if r.status_code == 429:  # โดนจำกัดความถี่ รอแล้วลองใหม่
                time.sleep(float(r.json().get("retry_after", 2)))
                continue
            r.raise_for_status()
            break
        time.sleep(0.5)


# ============================================================
# main
# ============================================================

def run_signals(markets, days, ignore_date, webhook):
    now = dt.datetime.now(BKK)
    for market in markets:
        symbols = SET50 if market == "set" else CRYPTO
        data = load_data(symbols, market, years=BACKTEST_YEARS)  # ใช้ 10 ปีเพื่อคำนวณสถิติเป้าขายในอดีต
        if not data:
            post_discord(webhook, f"⚠️ {MARKET_NAME[market]}: ดึงข้อมูลราคาไม่ได้ ลองใหม่ภายหลัง", [])
            continue
        signals, info = scan(data, market, days, ignore_date)

        if market == "set" and not ignore_date and info["stale"] > 0 and info["stale"] == info["checked"] - len(info["too_short"]):
            post_discord(webhook, f"😴 {MARKET_NAME[market]} · {now:%d/%m/%Y}: ไม่มีข้อมูลแท่งใหม่วันนี้ (ตลาดปิดหรือข้อมูลยังไม่อัปเดต)", [])
            continue

        buys = sum(s["kind"] == "buy" for s in signals)
        exits = len(signals) - buys
        span = "วันนี้" if days == 1 else f"ใน {days} แท่งล่าสุด"
        head = f"📊 **{MARKET_NAME[market]}** · {now:%d/%m/%Y %H:%M} · เช็ก {info['checked']} ตัว{span}\n"
        head += f"ซื้อ {buys} · ออก {exits}" if signals else "✅ ไม่มีสัญญาณใหม่"
        if info["too_short"]:
            head += f"\nข้อมูลย้อนหลังยังไม่พอ: {', '.join(info['too_short'])}"
        embeds = [signal_embed(s) for s in signals]
        if embeds:
            embeds[-1]["footer"] = {"text": DISCLAIMER}
        post_discord(webhook, head, embeds)


def run_backtests(markets, webhook):
    for market in markets:
        symbols = SET50 if market == "set" else CRYPTO
        # โหลดเกิน 10 ปีไป 1 ปี เพื่อใช้อุ่นเครื่อง EMA200
        data = load_data(symbols, market, years=BACKTEST_YEARS + 1.2)
        if not data:
            post_discord(webhook, f"⚠️ Backtest {MARKET_NAME[market]}: ดึงข้อมูลไม่ได้", [])
            continue
        bench = load_benchmark(market, BACKTEST_YEARS + 0.2)
        bt = run_backtest(data, market, bench)
        if not bt:
            post_discord(webhook, f"⚠️ Backtest {MARKET_NAME[market]}: ข้อมูลไม่พอ", [])
            continue
        print(backtest_text(bt))
        post_discord(webhook, "", [backtest_embed(bt)])


def main(argv=None):
    ap = argparse.ArgumentParser(description="บอทสัญญาณเทรนด์ SET50 + คริปโต → Discord")
    ap.add_argument("--market", choices=["set", "crypto", "all"], default="all")
    ap.add_argument("--days", type=int, default=1, help="ดูสัญญาณกี่แท่งล่าสุด (ค่าเริ่มต้น 1 = วันล่าสุด)")
    ap.add_argument("--ignore-date", action="store_true", help="ไม่ต้องเช็กว่าหุ้นมีแท่งของวันนี้ (ใช้ตอนทดสอบ)")
    ap.add_argument("--backtest", action="store_true", help="ทดสอบแบบพอร์ตจริงย้อนหลัง 10 ปี")
    args = ap.parse_args(argv)

    webhook = os.environ.get("DISCORD_WEBHOOK_URL") or None
    markets = ["set", "crypto"] if args.market == "all" else [args.market]
    if args.backtest:
        run_backtests(markets, webhook)
    else:
        run_signals(markets, max(1, args.days), args.ignore_date or args.days > 1, webhook)


if __name__ == "__main__":
    main()
