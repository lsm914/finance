#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, glob, argparse
import pandas as pd
import numpy as np

def robust_read_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    except Exception:
        df = pd.read_csv(path, engine="python", on_bad_lines="skip", sep=None)
    if "ticker" in df.columns:
        df = df[df["ticker"] != "ticker"]  # 중간 헤더 제거
    # 중복/공백 컬럼 정리
    cols, seen = [], {}
    for c in df.columns:
        c2 = str(c).strip()
        if c2 in seen:
            seen[c2] += 1
            c2 = f"{c2}__dup{seen[c2]}"
        else:
            seen[c2] = 0
        cols.append(c2)
    df.columns = cols
    return df

def load_all_csv(input_dir: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(input_dir, "**", "shard_*.csv"), recursive=True))
    if not paths:
        raise SystemExit(f"No shard csvs found under: {input_dir}")
    dfs = [robust_read_csv(p) for p in paths]
    dfs = [d for d in dfs if not d.empty]
    if not dfs:
        raise SystemExit("No valid rows after reading shard csvs")
    return pd.concat(dfs, ignore_index=True, sort=False)

def get_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)

def pctrank_col(df: pd.DataFrame, col: str, invert: bool=False) -> pd.Series:
    s = get_series(df, col)
    r = s.rank(pct=True, method="max")
    return (1 - r) if invert else r

def safe_mean(series_list):
    cols = [s for s in series_list if isinstance(s, pd.Series)]
    if not cols:
        return pd.Series(np.nan)
    return pd.concat(cols, axis=1).mean(axis=1)

def to_100(x: pd.Series) -> pd.Series:
    return (x*100).clip(lower=0, upper=100)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="artifacts")
    ap.add_argument("--out_all", default="all_nasdaq_value_screen.csv")
    args = ap.parse_args()

    df = load_all_csv(args.input_dir)

    # y3/2_3y 잔재 제거
    drop_pat = df.columns.str.contains(r"(_y3$)|(_chg_2_3y$)")
    if drop_pat.any():
        df = df.loc[:, ~drop_pat]

    # sector/industry 앞으로
    front = [c for c in ["ticker","sector","industry"] if c in df.columns]
    others = [c for c in df.columns if c not in front]
    df = df[front + others]

    # ===== 지수 계산 =====
    uv = safe_mean([
        pctrank_col(df, "PE_TTM_now", invert=True),
        pctrank_col(df, "PB_TTM_now", invert=True),
        pctrank_col(df, "EV_EBITDA_TTM_now", invert=True),
        pctrank_col(df, "FCF_Yield_now"),
    ])
    df["UndervaluationIndex"] = to_100(uv.fillna(0))

    w0, w1 = 0.7, 0.3
    def g2(prefix):
        a = pctrank_col(df, f"{prefix}_chg_0_1y")
        b = pctrank_col(df, f"{prefix}_chg_1_2y")
        return (a.fillna(0)*w0 + b.fillna(0)*w1)

    g_price   = g2("price")
    g_mcap    = g2("mktCap")
    g_ebitda  = g2("EBITDA_TTM")
    df["GrowthIndex"] = to_100(safe_mean([g_price, g_mcap, g_ebitda]).fillna(0))

    a_level = pctrank_col(df, "FCF_Yield_now").fillna(0)
    f0 = get_series(df, "FCF_TTM_chg_0_1y")
    f1 = get_series(df, "FCF_TTM_chg_1_2y")
    pos_ratio = ((f0 > 0).astype(int) + (f1 > 0).astype(int)) / 2.0
    vol = pd.concat([f0.abs(), f1.abs()], axis=1).std(axis=1)
    vol_pen = vol.rank(pct=True, method="max").fillna(0)
    cash_safe = (a_level*0.6 + pos_ratio.fillna(0)*0.3 - vol_pen*0.1)
    df["CashSafetyIndex"] = to_100(cash_safe.fillna(0))

    sort_cols = [c for c in ["UndervaluationIndex","CashSafetyIndex","GrowthIndex"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=False)

    df.to_csv(args.out_all, index=False)

if __name__ == "__main__":
    main()
