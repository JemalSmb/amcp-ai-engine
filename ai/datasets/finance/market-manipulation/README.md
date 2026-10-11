# Market manipulation datasets

Windows of market activity for the trading-behaviour BiLSTM.

**Model:** `finance_trading_behavior` | **Reference data:** LOBSTER order book sample, Binance klines

## Data format
- **Files:** `sequences.npy` and `labels.npy` in the folder passed to the pipeline (`processed/`).
- **`sequences.npy`:** float array of shape `(N, 30, 16)`.
- **`labels.npy`:** integer array of shape `(N,)`; `0` = normal, `1` = suspicious.
- **Features per step** (see `ai/models/finance_trading_behavior/config.yaml`): 16 values per tick: price, price change, volume, volume imbalance, order flow, bid-ask spread, depth imbalance, inter-trade time, cancel rate, order-to-trade ratio, layering score.

## What to look for in a dataset
- **Order book or trade tick data** with timestamps. Candlesticks alone lack depth and cancel information.
- Labelled manipulation examples are rare and mostly private. Most public data is unlabelled, so expect to label by rules or simulate patterns (spoofing, layering, wash trading).
- Use fixed windows of 30 ticks with the same 16 features in the same order.
- Split by time, never randomly, to avoid look-ahead leakage.
## Train on your own data
1. Put your files in `raw/` (keep the original download as-is).
2. Convert them to the format above and write the result to `processed/`. Start from `ai/training/preprocessing/preprocess_finance.py` and adapt the column mapping.
3. Train: `python ai/training/pipelines/finance_trading_behavior.py --config ai/models/finance_trading_behavior/config.yaml --data ai/datasets/finance/market-manipulation/processed --output <out_dir>`
4. If no data is found the pipeline silently falls back to **synthetic** data, so check the log line `Loading real data from ...` to confirm yours was used.
