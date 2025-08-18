#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Yahoo Finance Ultra Conservative Screener — shard runner
(now/y1/y2 + Annual fallback + fixed CSV schema)

- 입력: --tickers_csv, --shard_idx, --num_shards, --out_csv, [--resume]
- 출력: artifacts/shard_*.csv (항상 동일한 열 순서/스키마)
"""

import os, math, argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

DATE_FMT = "%Y-%m-%d"
TODAY = datetime.utcnow().date()

SNAP_OFFSETS = {"now": 0, "y1": 365, "y2": 365*2}

# ===== 고정 스키마(열 순서) =====
SCHEMA = (
    ["ticker", "sector", "industry", "shares_outstanding"] +
    [f"price_{t}" for t in ["now","y1","y2"]] +
    [f"mktCap_{t}" for t in ["now","y1","y2"]] +
    [f"EV_{t}" for t in ["now","y1","y2"]] +
    [f"PE_TTM_{t}" for t in ["now","y1","y2"]] +
    [f"PB_TTM_{t}" for t in ["now","y1","y2"]] +
    [f"EBITDA_TTM_{t}" for t in ["now","y1","y2"]] +
    [f"EV_EBITDA_TTM_{t}" for t in ["now","y1","y2"]] +
    [f"FCF_TTM_{t}" for t in ["now","y1","y2"]] +
    [f"FCF_Yield_{t}" for t in ["now","y1","y2"]] +
    [f"{m}_chg_0_1y" for m in ["price","mktCap","PE_TTM","PB_TTM","EV_EBITDA_TTM","EV","FCF_TTM","FCF_Yield"]] +
    [f"{m}_chg_1_2y" for m in ["price","mktCap","PE_TTM","PB_TTM","EV_EBITDA_TTM","EV","FCF_TTM","FCF_Yield"]] +
    ["error"]
)

def nearest_trading_close(tkr: yf.Ticker, target_date):
    start = (target_date - timedelta(days=20)).strftime(DATE_FMT)
    end   = (target_date + timedelta(days=20)).strftime(DATE_FMT)
    hist = tkr.history(start=start, end=end, auto_adjust=False)
    if hist is None or hist.empty:
        return np.nan
    hist = hist.reset_index()
    hist["date"] = pd.to_datetime(hist["Date"]).dt.date
    hist["dist"] = (hist["date"] - target_date).abs()
    row = hist.sort_values("dist").head(1)
    return float(row["Close"].iloc[0]) if len(row) > 0 else np.nan

def _latest_on_or_before(df: pd.DataFrame, cutoff_ts: pd.Timestamp):
    if df is None or df.empty:
        return None
    idx = df.index[df.index <= cutoff_ts]
    if len(idx) == 0:
        return None
    return idx.max()

def ttm_sum(df: pd.DataFrame, col: str, asof_date):
    if df is None or df.empty or (col not in df.columns):
        return np.nan
    df = df.copy().sort_index()
    cutoff = pd.Timestamp(asof_date)
    last_idx = _latest_on_or_before(df, cutoff)
    if last_idx is None:
        return np.nan
    last_pos = df.index.get_loc(last_idx)
    start_pos = max(0, last_pos - 3)
    window = df.iloc[start_pos:last_pos+1]
    return float(pd.to_numeric(window[col], errors="coerce").dropna().sum()) if len(window) > 0 else np.nan

def build_ttm_snapshots(tkr: yf.Ticker):
    # 분기
    q_is = tkr.quarterly_financials.T
    q_bs = tkr.quarterly_balance_sheet.T
    q_cf = tkr.quarterly_cashflow.T
    for df in (q_is, q_bs, q_cf):
        if df is not None and not df.empty:
            df.index = pd.to_datetime(df.index)
    # 연간(백업)
    a_is = tkr.financials.T
    a_bs = tkr.balance_sheet.T
    a_cf = tkr.cashflow.T
    for df in (a_is, a_bs, a_cf):
        if df is not None and not df.empty:
            df.index = pd.to_datetime(df.index)

    KEYS = {
        "net_income": ["Net Income","NetIncome","Net Income Common Stockholders"],
        "ebitda": ["EBITDA"],
        "total_debt": ["Total Debt","TotalDebt"],
        "cash": ["Cash And Cash Equivalents","Cash And Cash Equivalents Including Restricted Cash","Cash"],
        "equity": ["Total Stockholder Equity","Total Stockholders' Equity","Total Equity Gross Minority Interest"],
        "ocf": ["Operating Cash Flow","Total Cash From Operating Activities"],
        "capex": ["Capital Expenditure","Capital Expenditures"],
    }
    def pick(df, names):
        if df is None or df.empty: return None
        for n in names:
            if n in df.columns: return n
        return None

    ni_col     = pick(q_is, KEYS["net_income"]) or pick(a_is, KEYS["net_income"])
    ebitda_col = pick(q_is, KEYS["ebitda"]) or pick(a_is, KEYS["ebitda"])
    debt_col   = pick(q_bs, KEYS["total_debt"]) or pick(a_bs, KEYS["total_debt"])
    cash_col   = pick(q_bs, KEYS["cash"]) or pick(a_bs, KEYS["cash"])
    equity_col = pick(q_bs, KEYS["equity"]) or pick(a_bs, KEYS["equity"])
    ocf_col    = pick(q_cf, KEYS["ocf"]) or pick(a_cf, KEYS["ocf"])
    capex_col  = pick(q_cf, KEYS["capex"]) or pick(a_cf, KEYS["capex"])

    def latest_val(df, col, asof_date):
        if df is None or df.empty or not col: return np.nan
        cutoff = pd.Timestamp(asof_date)
        last_idx = _latest_on_or_before(df, cutoff)
        if last_idx is None: return np.nan
        return pd.to_numeric(df.loc[last_idx, col], errors="coerce")

    def latest_annual(df, col, asof_date):
        if df is None or df.empty or not col: return np.nan
        cutoff = pd.Timestamp(asof_date)
        idx = df.index[df.index <= cutoff]
        if len(idx) == 0: return np.nan
        return pd.to_numeric(df.loc[idx.max(), col], errors="coerce")

    return {
        "NI_TTM":      lambda d: ttm_sum(q_is, ni_col, d) if ni_col else np.nan,
        "EBITDA_TTM":  lambda d: ttm_sum(q_is, ebitda_col, d) if ebitda_col else np.nan,
        "Debt":        lambda d: (latest_val(q_bs, debt_col, d) if debt_col else np.nan),
        "Cash":        lambda d: (latest_val(q_bs, cash_col, d) if cash_col else np.nan),
        "Equity":      lambda d: (latest_val(q_bs, equity_col, d) if equity_col else np.nan),
        "OCF_TTM":     lambda d: ttm_sum(q_cf, ocf_col, d) if ocf_col else np.nan,
        "Capex_TTM":   lambda d: ttm_sum(q_cf, capex_col, d) if capex_col else np.nan,
        "_annual_refs": {
            "a_is": a_is, "a_cf": a_cf,
            "ni_col": ni_col, "ebitda_col": ebitda_col,
            "ocf_col": ocf_col, "capex_col": capex_col,
            "latest_annual": latest_annual,
        }
    }

def compute_snapshots_for_ticker(ticker: str):
    try:
        tkr = yf.Ticker(ticker)
        info = tkr.get_info() or {}

        # shares 보강
        shares = info.get("sharesOutstanding") or np.nan
        if not shares or shares != shares:
            try:
                fi = getattr(tkr, "fast_info", None)
                if fi and isinstance(fi, dict):
                    shares = fi.get("shares", np.nan)
            except Exception:
                pass

        # sector/industry
        sector   = info.get("sector") or np.nan
        industry = info.get("industry") or info.get("industryDisp") or np.nan

        snaps = {}
        ttm = build_ttm_snapshots(tkr)
        A   = ttm["_annual_refs"]
        a_is, a_cf = A["a_is"], A["a_cf"]
        LA  = A["latest_annual"]
        ni_col, ebitda_col = A["ni_col"], A["ebitda_col"]
        ocf_col, capex_col = A["ocf_col"], A["capex_col"]

        for tag, days in SNAP_OFFSETS.items():
            d = TODAY - timedelta(days=days)
            p = nearest_trading_close(tkr, d)
            snaps[f"price_{tag}"] = p

            NI      = ttm["NI_TTM"](d);      NI      = NI      if NI==NI      else LA(a_is, ni_col, d)
            EBITDA  = ttm["EBITDA_TTM"](d);  EBITDA  = EBITDA  if EBITDA==EBITDA else LA(a_is, ebitda_col, d)
            OCF     = ttm["OCF_TTM"](d);     OCF     = OCF     if OCF==OCF     else LA(a_cf, ocf_col, d)
            Capex   = ttm["Capex_TTM"](d);   Capex   = Capex   if Capex==Capex else LA(a_cf, capex_col, d)

            Debt    = ttm["Debt"](d)
            Cash    = ttm["Cash"](d)
            Equity  = ttm["Equity"](d)

            FCF = (OCF - Capex) if (not math.isnan(OCF) and not math.isnan(Capex)) else np.nan
            mktcap = (p * shares) if (shares == shares and p == p) else np.nan
            EV = (mktcap + (0 if math.isnan(Debt) else Debt) - (0 if math.isnan(Cash) else Cash)) if mktcap == mktcap else np.nan

            EPS_TTM  = (NI / shares) if (shares == shares and NI == NI and shares != 0) else np.nan
            BVPS_TTM = (Equity / shares) if (shares == shares and Equity == Equity and shares != 0) else np.nan

            PE        = (p / EPS_TTM) if (p == p and EPS_TTM and EPS_TTM != 0) else np.nan
            PB        = (p / BVPS_TTM) if (p == p and BVPS_TTM and BVPS_TTM != 0) else np.nan
            EV_EBITDA = (EV / EBITDA) if (EV == EV and EBITDA and EBITDA != 0) else np.nan
            FCF_Yield = (FCF / EV) if (EV == EV and FCF == FCF and EV != 0) else np.nan

            snaps.update({
                "shares_outstanding": shares,
                f"mktCap_{tag}": mktcap,
                f"EV_{tag}": EV,
                f"PE_TTM_{tag}": PE,
                f"PB_TTM_{tag}": PB,
                f"EBITDA_TTM_{tag}": EBITDA,
                f"EV_EBITDA_TTM_{tag}": EV_EBITDA,
                f"FCF_TTM_{tag}": FCF,
                f"FCF_Yield_{tag}": FCF_Yield,
            })

        # 변화율 (now vs y1, y1 vs y2)
        def ratio(a, b):
            if a != a or b != b or b == 0: return np.nan
            return (a / b) - 1.0

        seg = {}
        for metric in ["price","mktCap","PE_TTM","PB_TTM","EV_EBITDA_TTM","EV","FCF_TTM","FCF_Yield"]:
            now = snaps.get(f"{metric}_now", np.nan)
            y1  = snaps.get(f"{metric}_y1",  np.nan)
            y2  = snaps.get(f"{metric}_y2",  np.nan)
            seg[f"{metric}_chg_0_1y"] = ratio(now, y1)
            seg[f"{metric}_chg_1_2y"] = ratio(y1, y2)
        snaps.update(seg)

        # 루프 밖 메타
        snaps["ticker"] = ticker
        snaps["sector"] = sector
        snaps["industry"] = industry
        return snaps

    except Exception as e:
        return {"ticker": ticker, "error": str(e).replace(",", ";")}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers_csv", required=True)
    ap.add_argument("--shard_idx", type=int, required=True)
    ap.add_argument("--num_shards", type=int, default=8)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    # 헤더 유무 자동 처리
    df_t = pd.read_csv(args.tickers_csv)
    if 'ticker' not in df_t.columns:
        df_t = pd.read_csv(args.tickers_csv, header=None, names=['ticker'])
    tickers = [t.strip().upper() for t in df_t['ticker'].dropna().astype(str).tolist() if t.strip()]

    shard = [t for i, t in enumerate(tickers) if (i % args.num_shards) == args.shard_idx]

    out_path = args.out_csv
    done = set()
    if args.resume and os.path.exists(out_path):
        try:
            prev = pd.read_csv(out_path)
            done = set(prev["ticker"].astype(str))
        except Exception:
            pass

    rows = []
    def flush():
        nonlocal rows
        if not rows: return
        df = pd.DataFrame(rows)
        # 누락 칼럼 보강 + 스키마 순서로 고정
        for c in SCHEMA:
            if c not in df.columns:
                df[c] = np.nan
        df = df.reindex(columns=SCHEMA)
        header_needed = not os.path.exists(out_path)
        df.to_csv(out_path, index=False,
                  mode=("w" if header_needed else "a"),
                  header=header_needed)
        rows = []

    for t in shard:
        if t in done: continue
        rows.append(compute_snapshots_for_ticker(t))
        if len(rows) % 20 == 0: flush()
    flush()

if __name__ == "__main__":
    main()
