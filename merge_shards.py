#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, glob, argparse
import pandas as pd
import numpy as np



def get_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")

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

def load_all_csv(input_dir: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(input_dir, "**", "shard_*.csv"), recursive=True))
    if not paths:
        raise SystemExit(f"No shard csvs found under: {input_dir}")
    dfs = [pd.read_csv(p) for p in paths]
    return pd.concat(dfs, ignore_index=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="artifacts")
    ap.add_argument("--out_all", default="all_nasdaq_value_screen.csv")
    args = ap.parse_args()

    df = load_all_csv(args.input_dir)
    
    # 🔒 안전 필터: 혹시 섞여 들어온 y3/2_3y 컬럼은 제거
    drop_pat = df.columns.str.contains(r'(_y3$)|(_chg_2_3y$)')
    if drop_pat.any():
        df = df.loc[:, ~drop_pat]

    # 1) 저평가 지수
    uv = safe_mean([
        pctrank_col(df, "PE_TTM_now", invert=True),
        pctrank_col(df, "PB_TTM_now", invert=True),
        pctrank_col(df, "EV_EBITDA_TTM_now", invert=True),
        pctrank_col(df, "FCF_Yield_now", invert=False),
    ])
    df["UndervaluationIndex"] = to_100(uv)

    # 2) 성장 지수 (분절 2개: 0_1y, 1_2y)
    w0, w1 = 0.7, 0.3  # 최근 구간 가중치 ↑
    def g2(prefix):
        a = pctrank_col(df, f"{prefix}_chg_0_1y")
        b = pctrank_col(df, f"{prefix}_chg_1_2y")
        return a*w0 + b*w1

    g_price   = g2("price")
    g_mcap    = g2("mktCap")
    g_ebitda  = g2("EBITDA_TTM")
    df["GrowthIndex"] = to_100(safe_mean([g_price, g_mcap, g_ebitda]))

    # 3) 현금 안정성 지수 (분절 2개)
    a_level = pctrank_col(df, "FCF_Yield_now")
    f0 = get_series(df, "FCF_TTM_chg_0_1y")
    f1 = get_series(df, "FCF_TTM_chg_1_2y")

    # 양수 비율: 0, 0.5, 1.0
    pos_ratio = ((f0 > 0).astype(int) + (f1 > 0).astype(int)) / 2.0

    # 변동성 패널티: 두 구간 |변화율|의 표준편차 → 퍼센타일
    vol = pd.concat([f0.abs(), f1.abs()], axis=1).std(axis=1)
    vol_pen = vol.rank(pct=True, method="max")

    cash_safe = (a_level*0.6 + pos_ratio*0.3 - vol_pen*0.1)
    df["CashSafetyIndex"] = to_100(cash_safe)

    # 정렬
    df = df.sort_values(["UndervaluationIndex", "CashSafetyIndex", "GrowthIndex"], ascending=False)
    df.to_csv(args.out_all, index=False)

if __name__ == "__main__":
    main()
