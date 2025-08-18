# YF Ultra Conservative Screener v2

## 구조
```
.
├─ value_screen_yf_ultra_conservative.py   # 샤드 실행 스크립트
├─ merge_shards.py                          # 병합 & 3가지 지수 계산
├─ data/
│  └─ nasdaq_tickers.csv                    # 샘플 티커 목록(컬럼명: ticker)
├─ artifacts/                               # 샤드 출력 디렉토리
└─ .github/workflows/
   └─ yf_ultra_conservative.yml             # 8샤드 + 머지 워크플로우
```

## 로컬 실행
```bash
pip install yfinance pandas numpy
python value_screen_yf_ultra_conservative.py --tickers_csv data/nasdaq_tickers.csv --shard_idx 0 --num_shards 1 --out_csv artifacts/shard_0.csv --resume
python merge_shards.py
```

## GitHub Actions
- 수동 실행 또는 매주 월요일 01:00 KST 자동 실행
- 8개 샤드 병렬 → 머지 후 `all_nasdaq_value_screen.csv` 산출
