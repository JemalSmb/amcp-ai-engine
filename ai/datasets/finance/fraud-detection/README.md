# Fraud detection datasets

Transaction records for the fraud anomaly model (Isolation Forest).

**Model:** `finance_fraud_anomaly` | **Reference data:** Kaggle Credit Card Fraud, PaySim

## Data format
- **File:** a CSV placed in `raw/` (preprocessed output goes in `processed/`). The pipeline reads the **first `*.csv`** in the folder it is given.
- **Columns:** numeric feature columns plus one label column. Non-numeric columns are dropped, and `NaN` or infinite values are replaced.
- **Label column:** `isFraud`, `label`, `fraud` or `is_fraud`. `0` = legitimate, `1` = fraud. The model trains on legitimate rows only.
- **Features:** the serving model expects **20** numeric features (see `ai/models/finance_fraud_anomaly/config.yaml` for names and order). Your columns should map to that count, or update the config and retrain.

## What to look for in a dataset
- One row per **transaction** with amount, time of day or day of week, merchant or payment type, and ideally account history (transaction counts and velocity, deviation from the average amount).
- A fraud label. Fraud is rare (about 0.1 to 2%), so expect heavy imbalance and judge with recall, precision and AUC, not accuracy.
- Anonymised PCA columns (as in the credit card set) work but carry no meaning; named features give explainable alerts.
- Do not include features computed after the fraud is known (for example a chargeback flag).
## Train on your own data
1. Put your files in `raw/` (keep the original download as-is).
2. Convert them to the format above and write the result to `processed/`. Start from `ai/training/preprocessing/preprocess_finance.py` and adapt the column mapping.
3. Train: `python ai/training/pipelines/finance_fraud_anomaly.py --config ai/models/finance_fraud_anomaly/config.yaml --data ai/datasets/finance/fraud-detection/processed --output <out_dir>`
4. If no data is found the pipeline silently falls back to **synthetic** data, so check the log line `Loading real data from ...` to confirm yours was used.
