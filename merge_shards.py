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

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, glob, argparse
import pandas as pd
import numpy as np

def pct_rank(s: pd.Series, invert=False):
    s = s.astype(float)
    r = s.rank(pct=True, method='max')
    return (1 - r) if invert else r

def safe_mean(series_list):
    cols = [s for s in series_list if isinstance(s, pd.Series)]
    if not cols: return None
    df = pd.concat(cols, axis=1)
    return df.mean(axis=1)

def to_100(x):
    return (x*100).clip(lower=0, upper=100)

def load_all_csv(input_dir: str) -> pd.DataFrame:
    # 하위 폴더까지 재귀로 모든 shard_*.csv 수집
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

    # ===== 저평가 지수 =====
    uv = safe_mean([
        pct_rank(df.get('PE_TTM_now'), invert=True),
        pct_rank(df.get('PB_TTM_now'), invert=True),
        pct_rank(df.get('EV_EBITDA_TTM_now'), invert=True),
        pct_rank(df.get('FCF_Yield_now')),
    ])
    df['UndervaluationIndex'] = to_100(uv)

    # ===== 성장 지수 =====
    w0, w1, w2 = 0.5, 0.3, 0.2
    def gtrip(prefix):
        a = pct_rank(df.get(f'{prefix}_chg_0_1y'))
        b = pct_rank(df.get(f'{prefix}_chg_1_2y'))
        c = pct_rank(df.get(f'{prefix}_chg_2_3y'))
        return a*w0 + b*w1 + c*w2

    g_price   = gtrip('price')
    g_mcap    = gtrip('mktCap')
    g_ebitda  = gtrip('EBITDA_TTM')
    df['GrowthIndex'] = to_100(safe_mean([g_price, g_mcap, g_ebitda]))

    # ===== 현금 안정성 지수 =====
    a_level = pct_rank(df.get('FCF_Yield_now'))  # 높을수록 좋음
    pos_ratio = (
        (df.get('FCF_TTM_chg_0_1y')>0).astype(int) +
        (df.get('FCF_TTM_chg_1_2y')>0).astype(int) +
        (df.get('FCF_TTM_chg_2_3y')>0).astype(int)
    ) / 3.0

    vol = pd.concat([
        df.get('FCF_TTM_chg_0_1y').abs(),
        df.get('FCF_TTM_chg_1_2y').abs(),
        df.get('FCF_TTM_chg_2_3y').abs(),
    ], axis=1).std(axis=1)
    vol_pen = pct_rank(vol)  # 높을수록 변동성 큼 → 감점

    cash_safe = (a_level*0.6 + pos_ratio*0.3 - vol_pen*0.1)
    df['CashSafetyIndex'] = to_100(cash_safe)

    # 선호 정렬
    df = df.sort_values(['UndervaluationIndex', 'CashSafetyIndex', 'GrowthIndex'], ascending=False)
    df.to_csv(args.out_all, index=False)

if __name__ == '__main__':
    main()
