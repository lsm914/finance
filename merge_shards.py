#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Merge shards & compute indices (v2)
- 입력: artifacts/shard_*.csv
- 출력: all_nasdaq_value_screen.csv, sector_summary.csv (선택)

지수 정의(0~100 점수, 높을수록 좋음):
1) 저평가지수(UndervaluationIndex):
   - 후보 지표: PE_TTM_now, PB_TTM_now, EV_EBITDA_TTM_now, FCF_Yield_now
   - 방향성: 낮을수록(PE, PB, EV/EBITDA), 높을수록(FCF_Yield)
   - 방법: 각 지표를 유니버스 내 퍼센타일로 변환 후 평균 (역방향은 100-퍼센타일)

2) 성장지수(GrowthIndex):
   - 후보 지표: price, mktCap, EBITDA_TTM — 각 분절(chg_0_1y, 1_2y, 2_3y)
   - 방법: 세 분절별로 퍼센타일을 구하고 가중합(최근 구간 가중치 ↑: w=0.5, 0.3, 0.2)
            price와 mktCap, EBITDA를 평균하여 최종 점수 산출

3) 현금안정성지수(CashSafetyIndex):
   - 후보 지표: FCF_TTM_now, FCF_Yield_now, FCF_TTM의 분절(chg_0_1y, 1_2y, 2_3y)
   - 구성: (a) 현재 절대수준(FCF_Yield_now 퍼센타일)
           (b) 일관성: 분절별 FCF_TTM 성장률의 **음수 비율 감소** (양수일수록 가점)
           (c) 변동성 패널티: |FCF_TTM 분절|의 표준편차가 클수록 감점
           -> 100*(a*0.6 + consistency*0.3 - volatility_penalty*0.1)

결과에 세 지수를 컬럼으로 추가합니다.
"""

import os, glob, math
import pandas as pd
import numpy as np

IN_DIR = 'artifacts'
OUT_ALL = 'all_nasdaq_value_screen.csv'
OUT_SECTOR = 'sector_summary.csv'

def pct_rank(s: pd.Series, invert=False):
    s = s.astype(float)
    r = s.rank(pct=True, method='max')
    if invert:
        return 1 - r
    return r

def safe_mean(cols):
    arr = [c for c in cols if c.notna().any()]
    if not arr: return pd.Series(np.nan, index=cols[0].index)
    return pd.concat(arr, axis=1).mean(axis=1)

def to_100(x):
    return (x*100).clip(lower=0, upper=100)

def main():
    paths = sorted(glob.glob(os.path.join(IN_DIR, 'shard_*.csv')))
    dfs = [pd.read_csv(p) for p in paths if os.path.exists(p)]
    if not dfs:
        raise SystemExit('No shard csvs found')
    df = pd.concat(dfs, ignore_index=True)

    uv = safe_mean([
        pct_rank(df['PE_TTM_now'], invert=True),
        pct_rank(df['PB_TTM_now'], invert=True),
        pct_rank(df['EV_EBITDA_TTM_now'], invert=True),
        pct_rank(df['FCF_Yield_now'], invert=False),
    ])
    df['UndervaluationIndex'] = to_100(uv)

    w0, w1, w2 = 0.5, 0.3, 0.2
    def growth_triplet(prefix):
        a = pct_rank(df[f'{prefix}_chg_0_1y'])
        b = pct_rank(df[f'{prefix}_chg_1_2y'])
        c = pct_rank(df[f'{prefix}_chg_2_3y'])
        return (a*w0 + b*w1 + c*w2)

    g_price = growth_triplet('price')
    g_mcap = growth_triplet('mktCap')
    g_ebitda = growth_triplet('EBITDA_TTM')
    df['GrowthIndex'] = to_100(safe_mean([g_price, g_mcap, g_ebitda]))

    a_level = pct_rank(df['FCF_Yield_now'])
    pos_ratio = (
        (df['FCF_TTM_chg_0_1y']>0).astype(int) +
        (df['FCF_TTM_chg_1_2y']>0).astype(int) +
        (df['FCF_TTM_chg_2_3y']>0).astype(int)
    ) / 3.0
    vol = pd.concat([
        df['FCF_TTM_chg_0_1y'].abs(),
        df['FCF_TTM_chg_1_2y'].abs(),
        df['FCF_TTM_chg_2_3y'].abs(),
    ], axis=1).std(axis=1)
    vol_pen = pct_rank(vol)
    cash_safe = (a_level*0.6 + pos_ratio*0.3 - vol_pen*0.1)
    df['CashSafetyIndex'] = to_100(cash_safe)

    df = df.sort_values(['UndervaluationIndex','CashSafetyIndex','GrowthIndex'], ascending=False)
    df.to_csv(OUT_ALL, index=False)

if __name__ == '__main__':
    main()
