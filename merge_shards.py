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

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, glob, argparse
import pandas as pd
import numpy as np

# ---- 안전 헬퍼들 ----
def get_series(df: pd.DataFrame, col: str) -> pd.Series:
    """컬럼이 없으면 길이 맞는 NaN Series 반환, 있으면 숫자형으로 정규화"""
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors='coerce')

def pctrank_col(df: pd.DataFrame, col: str, invert: bool=False) -> pd.Series:
    s = get_series(df, col)
    r = s.rank(pct=True, method='max')  # NaN은 그대로 유지
    return (1 - r) if invert else r

def safe_mean(series_list):
    cols = [s for s in series_list if isinstance(s, pd.Series)]
    if not cols:
        return pd.Series(np.nan)
    return pd.concat(cols, axis=1).mean(axis=1)  # NaN 자동 제외 평균

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

    # ===== 저평가 지수 =====
    uv = safe_mean([
        pctrank_col(df, 'PE_TTM_now', invert=True),
        pctrank_col(df, 'PB_TTM_now', invert=True),
        pctrank_col(df, 'EV_EBITDA_TTM_now', invert=True),
        pctrank_col(df, 'FCF_Yield_now', invert=False),
    ])
    df['UndervaluationIndex'] = to_100(uv)

    # ===== 성장 지수 =====
    w0, w1, w2 = 0.5, 0.3, 0.2
    def gtrip(prefix: str) -> pd.Series:
        a = pctrank_col(df, f'{prefix}_chg_0_1y')
        b = pctrank_col(df, f'{prefix}_chg_1_2y')
        c = pctrank_col(df, f'{prefix}_chg_2_3y')
        return a*w0 + b*w1 + c*w2

    g_price   = gtrip('price')
    g_mcap    = gtrip('mktCap')
    g_ebitda  = gtrip('EBITDA_TTM')
    df['GrowthIndex'] = to_100(safe_mean([g_price, g_mcap, g_ebitda]))

    # ===== 현금 안정성 지수 =====
    a_level = pctrank_col(df, 'FCF_Yield_now')  # 높을수록 좋음

    f0 = get_series(df, 'FCF_TTM_chg_0_1y')
    f1 = get_series(df, 'FCF_TTM_chg_1_2y')
    f2 = get_series(df, 'FCF_TTM_chg_2_3y')

    # NaN>0 은 False로 처리되므로 일관성 계산 안전
    pos_ratio = ((f0 > 0).astype(int) + (f1 > 0).astype(int) + (f2 > 0).astype(int)) / 3.0

    vol = pd.concat([f0.abs(), f1.abs(), f2.abs()], axis=1).std(axis=1)
    vol_pen = vol.rank(pct=True, method='max')  # 높을수록 변동 큼 → 감점

    cash_safe = (a_level*0.6 + pos_ratio*0.3 - vol_pen*0.1)
    df['CashSafetyIndex'] = to_100(cash_safe)

    df = df.sort_values(['UndervaluationIndex', 'CashSafetyIndex', 'GrowthIndex'], ascending=False)
    df.to_csv(args.out_all, index=False)

if __name__ == '__main__':
    main()
