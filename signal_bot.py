#!/usr/bin/env python3
"""
บอทส่งสัญญาณซื้อแบบ "ตามเทรนด์" สำหรับหุ้น SET50 + คริปโต Top 10 → Discord

กลยุทธ์ (ดูกราฟรายวัน):
  สัญญาณซื้อ เมื่อเกิดครบทุกข้อในวันเดียวกัน
    1) EMA20 ตัดขึ้นเหนือ EMA50          (เทรนด์ระยะกลางเริ่มเป็นขาขึ้น)
    2) ราคาปิด > EMA200                 (เทรนด์ใหญ่ยังเป็นขาขึ้น)
    3) วอลุ่ม > 1.2 เท่าของค่าเฉลี่ย 20 วัน (มีแรงซื้อยืนยัน)
    4) RSI(14) < 75                     (ยังไม่ร้อนแรงเกินไป)
  จุดตัดขาดทุนแนะนำ = ราคาปิด − 2 × ATR(14)
  สัญญาณออก (แจ้งเพื่อทราบ) เมื่อ EMA20 ตัดลงใต้ EMA50

วิธีใช้:
  python signal_bot.py --market set              # เช็กหุ้นไทยวันนี้
  python signal_bot.py --market crypto           # เช็กคริปโต (แท่งเมื่อวานที่ปิดแล้ว)
  python signal_bot.py --market all --days 10 --ignore-date   # ดูสัญญาณย้อนหลัง 10 แท่ง (ไว้ทดสอบ)
  python signal_bot.py --market all --backtest   # ทดสอบย้อนหลัง ~10 ปี

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
    "atr_stop": 2.0,   # จุดตัดขาดทุน = ราคา − 2×ATR
}

# ค่าธรรมเนียมโดยประมาณต่อขา (ซื้อ หรือ ขาย) ใช้ใน backtest
FEES = {"set": 0.0017, "crypto": 0.001}

BKK = ZoneInfo("Asia/Bangkok")
MARKET_NAME = {"set": "หุ้นไทย SET50", "crypto": "คริปโต"}
GREEN, RED, BLUE, GREY = 0x2ECC71, 0xE74C3C, 0x3498DB, 0x95A5A6
DISCLAIMER = "สัญญาณจากสูตรเทคนิคอัตโนมัติ ไม่ใช่คำแนะนำการลงทุน — ตัดสินใจและบริหารความเสี่ยงเอง"


# ============================================================
# ข้อมูลราคา
# ============================================================

def to_yf(symbol: str, market: str) -> str:
    return f"{symbol}.BK" if market == "set" else f"{symbol}-USD"


def load_data(symbols: list[str], market: str, period: str) -> dict[str, pd.DataFrame]:
    """ดึงราคารายวันจาก Yahoo Finance คืนค่า {symbol: DataFrame(Open,High,Low,Close,Volume)}"""
    import yfinance as yf

    yf_map = {to_yf(s, market): s for s in symbols}
    raw = yf.download(
        list(yf_map), period=period, interval="1d", group_by="ticker",
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
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"]).copy()
            idx = pd.to_datetime(df.index)
            if getattr(idx, "tz", None) is not None:
                idx = idx.tz_localize(None)
            df.index = idx
            if not df.empty:
                out[sym] = df
        except Exception as e:  # ตัวไหนพังก็ข้ามไป ไม่ให้ทั้งบอทล่ม
            print(f"[warn] โหลด {sym} ไม่ได้: {e}", file=sys.stderr)
    return out


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
        for ts, row in ind.tail(days).iterrows():
            if not (row["buy"] or row["exit"]):
                continue
            close, atr = float(row["Close"]), float(row["atr"])
            stop = close - PARAMS["atr_stop"] * atr
            signals.append({
                "symbol": sym, "market": market, "date": ts.date(),
                "kind": "buy" if row["buy"] else "exit",
                "close": close, "stop": stop, "risk_pct": (close - stop) / close * 100,
                "rsi": float(row["rsi"]),
                "vol_x": None if pd.isna(row["vol_x"]) else float(row["vol_x"]),
                "ema_trend": float(row["ema_trend"]),
            })
    signals.sort(key=lambda s: (s["kind"] != "buy", s["date"], s["symbol"]))
    info = {"checked": len(data), "too_short": too_short, "stale": stale}
    return signals, info


# ============================================================
# Backtest
# ============================================================

def backtest_symbol(df: pd.DataFrame, fee: float) -> dict | None:
    ind = add_indicators(df)
    if len(ind) < PARAMS["trend"] + 50:
        return None
    o, h, l, c = (ind[k].to_numpy() for k in ("Open", "High", "Low", "Close"))
    buy, ext, atr = ind["buy"].to_numpy(), ind["exit"].to_numpy(), ind["atr"].to_numpy()

    trades, equity = [], 1.0
    in_pos, entry, stop = False, 0.0, 0.0
    for i in range(len(ind)):
        if in_pos:
            exit_px = None
            if l[i] <= stop:                      # โดนจุดตัดขาดทุน (ถ้าเปิดกระโดดต่ำกว่า ใช้ราคาเปิด)
                exit_px = min(o[i], stop)
            elif ext[i]:                          # EMA20 ตัดลง
                exit_px = c[i]
            if exit_px is not None:
                r = (exit_px * (1 - fee)) / (entry * (1 + fee)) - 1
                trades.append(r)
                equity *= 1 + r
                in_pos = False
        elif buy[i]:
            in_pos, entry, stop = True, c[i], c[i] - PARAMS["atr_stop"] * atr[i]
    if in_pos:  # ปิดสถานะที่ค้างอยู่ด้วยราคาล่าสุด
        r = (c[-1] * (1 - fee)) / (entry * (1 + fee)) - 1
        trades.append(r)
        equity *= 1 + r

    start = PARAMS["trend"]
    years = (ind.index[-1] - ind.index[start]).days / 365.25
    bh = c[-1] / c[start] - 1
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    return {
        "trades": len(trades), "wins": len(wins),
        "avg_win": float(np.mean(wins)) if wins else 0.0,
        "avg_loss": float(np.mean(losses)) if losses else 0.0,
        "gross_win": float(sum(wins)), "gross_loss": float(-sum(losses)),
        "strategy": equity - 1, "buy_hold": float(bh), "years": years,
        "trade_returns": trades,
    }


def run_backtest(data: dict[str, pd.DataFrame], market: str) -> dict:
    fee = FEES[market]
    rows = {}
    for sym, df in data.items():
        if market == "crypto":
            df = drop_unfinished_crypto_bar(df)
        res = backtest_symbol(df, fee)
        if res:
            rows[sym] = res
    all_trades = [t for r in rows.values() for t in r["trade_returns"]]
    n = len(all_trades)
    wins = sum(1 for t in all_trades if t > 0)
    gross_win = sum(t for t in all_trades if t > 0)
    gross_loss = -sum(t for t in all_trades if t <= 0)
    beat = sum(1 for r in rows.values() if r["strategy"] > r["buy_hold"])
    return {
        "market": market, "symbols": len(rows), "trades": n,
        "win_rate": wins / n if n else 0.0,
        "avg_trade": float(np.mean(all_trades)) if n else 0.0,
        "profit_factor": gross_win / gross_loss if gross_loss else float("inf"),
        "median_strategy": float(np.median([r["strategy"] for r in rows.values()])) if rows else 0.0,
        "median_buy_hold": float(np.median([r["buy_hold"] for r in rows.values()])) if rows else 0.0,
        "beat_bh": beat,
        "avg_years": float(np.mean([r["years"] for r in rows.values()])) if rows else 0.0,
        "per_symbol": rows,
    }


# ============================================================
# Discord
# ============================================================

def fmt_price(x: float, market: str) -> str:
    if market == "set":
        return f"฿{x:,.2f}"
    return f"${x:,.4f}" if x < 1 else f"${x:,.2f}"


def chart_url(sym: str, market: str) -> str:
    tv = f"SET:{sym}" if market == "set" else f"BINANCE:{sym}USDT"
    return f"https://www.tradingview.com/chart/?symbol={tv}"


def signal_embed(s: dict) -> dict:
    m = s["market"]
    if s["kind"] == "buy":
        title, color = f"🟢 สัญญาณซื้อ {s['symbol']}", GREEN
        fields = [
            {"name": "ราคาปิด", "value": fmt_price(s["close"], m), "inline": True},
            {"name": "จุดตัดขาดทุนแนะนำ", "value": f"{fmt_price(s['stop'], m)} (−{s['risk_pct']:.1f}%)", "inline": True},
            {"name": "RSI(14)", "value": f"{s['rsi']:.0f}", "inline": True},
            {"name": "วอลุ่ม", "value": "—" if s["vol_x"] is None else f"{s['vol_x']:.1f}× ค่าเฉลี่ย", "inline": True},
            {"name": "เหตุผล", "value": "EMA20 ตัดขึ้น EMA50 · ราคาอยู่เหนือ EMA200", "inline": False},
        ]
    else:
        title, color = f"🔴 สัญญาณออก {s['symbol']}", RED
        fields = [
            {"name": "ราคาปิด", "value": fmt_price(s["close"], m), "inline": True},
            {"name": "เหตุผล", "value": "EMA20 ตัดลงใต้ EMA50 (ถ้าถืออยู่ ควรพิจารณา)", "inline": False},
        ]
    return {
        "title": title, "url": chart_url(s["symbol"], m), "color": color,
        "description": f"{MARKET_NAME[m]} · แท่งวันที่ {s['date']:%d/%m/%Y}",
        "fields": fields,
    }


def post_discord(webhook: str | None, content: str, embeds: list[dict]) -> None:
    if not webhook:
        print("\n" + "=" * 60 + "\n" + content)
        for e in embeds:
            print(json.dumps(e, ensure_ascii=False, indent=1))
        return
    chunks = [embeds[i:i + 10] for i in range(0, len(embeds), 10)] or [[]]  # Discord รับได้ 10 embed/ข้อความ
    for n, chunk in enumerate(chunks):
        payload = {"content": content if n == 0 else None, "embeds": chunk, "username": "Signal Bot"}
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
        data = load_data(symbols, market, period="2y")
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
        if signals:
            head += f"ซื้อ {buys} · ออก {exits}"
        else:
            head += "✅ ไม่มีสัญญาณใหม่"
        if info["too_short"]:
            head += f"\nข้อมูลย้อนหลังยังไม่พอ: {', '.join(info['too_short'])}"
        embeds = [signal_embed(s) for s in signals]
        if embeds:
            embeds[-1]["footer"] = {"text": DISCLAIMER}
        post_discord(webhook, head, embeds)


def run_backtests(markets, webhook):
    for market in markets:
        symbols = SET50 if market == "set" else CRYPTO
        data = load_data(symbols, market, period="10y")
        if not data:
            post_discord(webhook, f"⚠️ Backtest {MARKET_NAME[market]}: ดึงข้อมูลไม่ได้", [])
            continue
        bt = run_backtest(data, market)
        ranked = sorted(bt["per_symbol"].items(), key=lambda kv: kv[1]["strategy"], reverse=True)

        def line(sym, r):
            return f"`{sym:<7}` กลยุทธ์ {r['strategy']*100:+.0f}% · ซื้อถือ {r['buy_hold']*100:+.0f}% · {r['trades']} เทรด"

        top = "\n".join(line(s, r) for s, r in ranked[:5]) or "—"
        bottom = "\n".join(line(s, r) for s, r in ranked[5:][-5:]) or "—"
        pf = "∞" if bt["profit_factor"] == float("inf") else f"{bt['profit_factor']:.2f}"
        embed = {
            "title": f"🧪 Backtest {MARKET_NAME[market]} (~{bt['avg_years']:.1f} ปี)",
            "color": BLUE,
            "description": (
                f"หักค่าธรรมเนียม {FEES[market]*100:.2f}% ต่อขาแล้ว\n"
                f"จำนวนตัว {bt['symbols']} · จำนวนเทรด {bt['trades']}\n"
                f"อัตราชนะ **{bt['win_rate']*100:.0f}%** · กำไรเฉลี่ย/เทรด **{bt['avg_trade']*100:+.2f}%** · Profit factor **{pf}**\n"
                f"ผลตอบแทนรวม (ค่ามัธยฐาน): กลยุทธ์ **{bt['median_strategy']*100:+.0f}%** vs ซื้อแล้วถือ **{bt['median_buy_hold']*100:+.0f}%**\n"
                f"ชนะการซื้อแล้วถือ {bt['beat_bh']}/{bt['symbols']} ตัว"
            ),
            "fields": [
                {"name": "ดีที่สุด 5 ตัว", "value": top[:1024], "inline": False},
                {"name": "แย่ที่สุด 5 ตัว", "value": bottom[:1024], "inline": False},
            ],
            "footer": {"text": "หมายเหตุ: ใช้รายชื่อหุ้นปัจจุบันทดสอบย้อนหลัง ผลจึงดูดีกว่าความจริง (survivorship bias) · ผลในอดีตไม่รับประกันอนาคต"},
        }
        post_discord(webhook, "", [embed])


def main(argv=None):
    ap = argparse.ArgumentParser(description="บอทสัญญาณเทรนด์ SET50 + คริปโต → Discord")
    ap.add_argument("--market", choices=["set", "crypto", "all"], default="all")
    ap.add_argument("--days", type=int, default=1, help="ดูสัญญาณกี่แท่งล่าสุด (ค่าเริ่มต้น 1 = วันล่าสุด)")
    ap.add_argument("--ignore-date", action="store_true", help="ไม่ต้องเช็กว่าหุ้นมีแท่งของวันนี้ (ใช้ตอนทดสอบ)")
    ap.add_argument("--backtest", action="store_true", help="ทดสอบกลยุทธ์ย้อนหลัง ~10 ปี")
    args = ap.parse_args(argv)

    webhook = os.environ.get("DISCORD_WEBHOOK_URL") or None
    markets = ["set", "crypto"] if args.market == "all" else [args.market]
    if args.backtest:
        run_backtests(markets, webhook)
    else:
        run_signals(markets, max(1, args.days), args.ignore_date or args.days > 1, webhook)


if __name__ == "__main__":
    main()
