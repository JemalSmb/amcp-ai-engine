# AMCP AI

The AI layer of the AI Modular Cybersecurity Platform: datasets, training, evaluation, model definitions and inference serving.

| Directory | Purpose |
|---|---|
| [datasets/](datasets/README.md) | Dataset layout and documentation (raw data is kept out of git) |
| [training/](training/README.md) | Preprocessing scripts and per-model training pipelines |
| [evaluation/](evaluation/README.md) | Evaluation approach and metrics |
| [models/](models/) | Per-model `config.yaml`, shared base types, trained artifacts |
| [inference/](inference/README.md) | FastAPI service that serves the models |

## Models

| model_id | Domain | Type |
|---|---|---|
| `anomaly_detection` | cybersecurity | Isolation Forest |
| `phishing_detection` | cybersecurity | Random Forest |
| `threat_detection` | cybersecurity | Random Forest (3-class) |
| `ransomware_detection` | cybersecurity | BiLSTM + attention |
| `finance_fraud_anomaly` | finance | Isolation Forest |
| `finance_trading_behavior` | finance | BiLSTM + attention |
| `manufacturing_process_anomaly` | manufacturing | Autoencoder |
| `manufacturing_maintenance_forecast` | manufacturing | Transformer |

Shared base types are defined in `models/bases/model_types.yaml`. Currently the inference service serves a subset of these, with the rest on the mock fallback (see `inference/README.md`).

## Flow

```text
datasets -> training/preprocessing -> training/pipelines -> MLflow registry + models/*/artifacts -> inference
```
