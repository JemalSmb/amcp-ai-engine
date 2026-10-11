# AI Evaluation

How the AMCP models are evaluated.

There is no standalone evaluation code in this directory yet. Each training pipeline in `ai/training/pipelines/` evaluates its model on a held-out split (`training.test_size` in the model config, default 0.2, stratified where labels exist) right after training, and logs the results to MLflow. This directory is reserved for shared evaluation tooling, such as cross-model comparison or regression checks against a registered version.

## Metrics logged per model

| Model | Logged metrics |
|---|---|
| `anomaly_detection` | accuracy, precision, recall, f1, auc_roc (decision threshold tuned on test data and saved with the model) |
| `phishing_detection` | accuracy, precision, recall, f1, auc_roc, feature importances |
| `threat_detection` | accuracy, precision_macro, recall_macro, f1_macro, plus a printed confusion matrix and per-class report |
| `ransomware_detection` | accuracy, precision, recall, f1, auc_roc |
| `finance_fraud_anomaly` | accuracy, precision, recall, f1, auc_roc, threshold |
| `finance_trading_behavior` | per-epoch `train_loss` and `val_*` metrics |
| `manufacturing_process_anomaly` | per-epoch `train_loss`, `val_f1` |
| `manufacturing_maintenance_forecast` | per-epoch `train_loss`, `val_mse` |

The `evaluation.metrics` and `evaluation.thresholds` sections in each `config.yaml` list the intended metrics and operating points (for example `default`, `high_precision`, `high_recall`). Not every pipeline computes every metric listed there.

## Reading results

Point MLflow at the tracking store used for training:

```bash
mlflow ui --backend-store-uri sqlite:///ai/training/pipelines/mlflow.db
```

## Caveats

- Results from a run that fell back to synthetic data are not meaningful. Check the pipeline log for `Loading real data from ...`.
- Evaluate on data from different days or captures than the training set, otherwise network-flow models will look better than they are.
- Attack classes are usually a small share of the data, so look at precision, recall and the confusion matrix rather than accuracy alone.
