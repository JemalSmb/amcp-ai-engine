# AI Training

Preprocessing scripts and training pipelines for the AMCP models.

```text
training/
├── preprocessing/   # raw datasets -> processed CSV/npy files
├── pipelines/       # one training script per model
├── utils.py         # shared helpers (config, CSV loading, synthetic data, MLflow registration)
└── requirements.txt
```

## Pipelines

Each pipeline takes a model config from `ai/models/<model_id>/config.yaml`, trains, evaluates on a held-out split, logs to MLflow, and writes artifacts to `--output`.

| Pipeline | Model | Algorithm | Config |
|---|---|---|---|
| `anomaly_detection.py` | `anomaly_detection` | Isolation Forest (benign-only) | `models/anomaly_detection` |
| `phishing_detection.py` | `phishing_detection` | Random Forest | `models/phishing_detection` |
| `threat_detection.py` | `threat_detection` | Random Forest (3-class) | `models/threat_detection` |
| `ransomware_detection.py` | `ransomware_detection` | BiLSTM + attention | `models/ransomware_detection` |
| `finance_fraud_anomaly.py` | `finance_fraud_anomaly` | Isolation Forest | `models/finance_fraud_anomaly` |
| `finance_trading_behavior.py` | `finance_trading_behavior` | BiLSTM + attention | `models/finance_trading_behavior` |
| `manufacturing_process_anomaly.py` | `manufacturing_process_anomaly` | Autoencoder | `models/manufacturing_process_anomaly` |
| `manufacturing_maintenance_forecast.py` | `manufacturing_maintenance_forecast` | Transformer | `models/manufacturing_maintenance_forecast` |

## Setup

```bash
cd ai/training
pip install -r requirements.txt
```

Credentials are read from `ai/inference/.env` (see `ai/inference/.env.example`), so training and serving share one set.

## Preprocess

Raw data lives outside git (see `ai/datasets/PROJECT_DATASETS.md`).

```bash
python preprocessing/preprocess_all.py \
  --datasets-dir /path/to/Project_Datasets \
  --output-dir ./processed

# one domain only, or a quick test with limited EMBER records
python preprocessing/preprocess_all.py ... --only cybersecurity
python preprocessing/preprocess_all.py ... --ember-limit 1000
```

Use `--skip-ransomware` to leave out the large EMBER set.

## Train

All pipelines share the same arguments:

```bash
python pipelines/phishing_detection.py \
  --config ../models/phishing_detection/config.yaml \
  --data   ../datasets/cybersecurity/phishing/processed \
  --output ../models/phishing_detection/artifacts \
  --mlflow-uri sqlite:///pipelines/mlflow.db   # optional
```

| Argument | Meaning |
|---|---|
| `--config` | model config YAML |
| `--data` | directory holding the processed CSV (first `*.csv` is used) |
| `--output` | where the `.joblib` and `.onnx` artifacts are written |
| `--mlflow-uri` | MLflow tracking URI (optional) |
| `--experiment-name` | MLflow experiment (defaults per pipeline) |

> **Synthetic fallback:** if `--data` contains no CSV, most pipelines silently generate synthetic data so the code path still runs. Check the log for `Loading real data from ...` to confirm real data was used. Models trained on synthetic data have no predictive value.

## Registering a model

`utils.log_and_register_model()` bundles the fitted scaler and estimator as one MLflow pyfunc model and registers it. The inference service polls the registry and hot-swaps new versions without a restart, see `ai/inference/README.md`. `pipelines/phishing_detection.py` shows the pattern.
