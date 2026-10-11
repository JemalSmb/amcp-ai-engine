# Project datasets

What data the AMCP models are trained on. The raw data lives on the external
drive at `Project_Datasets/` (about 315 GB) and is **not** in git. This repo
only keeps the folder layout (`.gitkeep`) and this description. For download
links per model see [DATASET_GUIDE.md](DATASET_GUIDE.md).

Layout on the drive: `<domain>/<task>/{raw,processed}/`. Copy or symlink a
task folder into `ai/datasets/<domain>/<task>/` before training.

## Cybersecurity

### anomaly-detection (about 27 GB) -> `anomaly_detection`
Network intrusion flow data for the unsupervised anomaly model.

| Dataset | Size | Notes |
| --- | --- | --- |
| CICIDS 2017 | 19 GB | CIC flow CSVs, labelled benign and attack days |
| CICIDS 2018 (`CICIDS2018_CSVs`) | 6.5 GB | CIC flow CSVs; `CICIDS2018` folder is empty |
| UNSW-NB15 | 1.9 GB | Mixed modern attack and normal traffic |
| NSL-KDD | 107 MB | Classic KDD intrusion benchmark |

### phishing (about 147 MB) -> `phishing_detection`
URL and page-feature data for the phishing classifier.

| Dataset | Size | Notes |
| --- | --- | --- |
| `Phishing_Legitimate_full.csv` | 1.4 MB | Labelled phishing and legitimate feature table |
| `phishtank_verified_online.csv` | 11 MB | Verified phishing URL feed |
| `urlhaus.abuse.ch` | 2.5 MB | Malicious URL feed |
| `tranco_6GPNX.csv` | 133 MB | Tranco top sites, used as legitimate URLs |

### ransomware (about 34 GB) -> `ransomware_detection`
| Dataset | Size | Notes |
| --- | --- | --- |
| EMBER | 34 GB | PE-file features for malware vs benign |
| MalwareBazaar | 467 MB | Malware sample metadata |
| CIC Ransomware 2022 | 364 KB | Ransomware behaviour records |

### threat_detection (about 245 GB) -> `threat_detection`
Multi-class threat classifier; shares CICIDS 2017 with anomaly detection.

| Dataset | Size | Notes |
| --- | --- | --- |
| CICDDoS2019 | 153 GB | DDoS attack classes |
| CTU-13 | 75 GB | Botnet netflow captures |
| CICIDS 2017 | 19 GB | Same data as in anomaly-detection |
| API Call Sequences | 18 MB | Malware API-call sequences |

## Finance

### fraud-detection (about 615 MB) -> `finance_fraud_anomaly`
| Dataset | Size | Notes |
| --- | --- | --- |
| Credit Card Fraud | 144 MB | 284k anonymised card transactions, highly imbalanced |
| Synthetic Financial Dataset (PaySim) | 471 MB | Simulated mobile-money transactions with fraud labels |

### market-manipulation (about 3.9 GB) -> `finance_trading_behavior`
| Dataset | Size | Notes |
| --- | --- | --- |
| Binance klines | 3.0 GB | Historical candlesticks for spot pairs |
| LOBSTER | 921 MB | Limit order book sample |

## Manufacturing

### production-anomaly (about 1.4 GB) -> `manufacturing_process_anomaly`
| Dataset | Size | Notes |
| --- | --- | --- |
| Tennessee Eastman Process | 1.4 GB | Simulated chemical-process sensor data with faults |
| NASA Turbofan (C-MAPSS) | 12 MB | Engine degradation simulation (also fits maintenance forecasting) |

### predictive-maintenance (about 512 KB) -> `manufacturing_maintenance_forecast`
| Dataset | Size | Notes |
| --- | --- | --- |
| AI4I 2020 Predictive Maintenance | 512 KB | Small tabular machine-failure dataset |

## Agriculture (not yet backed by a model)

| Folder | Size | Contents |
| --- | --- | --- |
| path-planning | 964 KB | `grid_robot_navigation_dataset.csv` |
| yield-prediction | 7 MB | `Agri_yield_prediction.csv` |
| vineyard-lidar | 2.7 GB | FLEXIGROBOTS LAZ point clouds (2021 to 2022 flights) |
| field-anomaly | empty | placeholder |

## Common
`geolocation`, `time-series-benchmarks` and `weather` are empty placeholders.

## Status and gaps
- **No `processed/` data yet:** every `processed/` folder holds only a placeholder, so preprocessing in `ai/training/preprocessing` still has to run.
- **Folder names differ:** the repo uses `threat-detection` and `agriculture`; the drive uses `threat_detection` and `Agriculture`.
- **Licences:** Some sets (for example the CIC datasets) need registration or have terms; check before redistributing anything.
