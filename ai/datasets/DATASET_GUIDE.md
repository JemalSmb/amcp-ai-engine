# 📦 Dataset Guide — AI Modular Platform

> All datasets are **publicly downloadable** — no hardware, sensors, or special access required

---

## 🛡️ Cybersecurity (4 Models)

### 1. `anomaly_detection` — Network Anomaly (Isolation Forest)
| Dataset | Download |
|---------|----------|
| **CICIDS 2017** (~2.8 GB CSV) | https://www.unb.ca/cic/datasets/ids-2017.html |
| **CICIDS 2018** | https://www.unb.ca/cic/datasets/ids-2018.html |
| **UNSW-NB15** | https://research.unsw.edu.au/projects/unsw-nb15-dataset |
| **NSL-KDD** | https://www.unb.ca/cic/datasets/nsl.html |

Place in: `ai/datasets/cybersecurity/threat-detection/raw/`

---

### 2. `phishing_detection` — Phishing URL/Email (Random Forest)
| Dataset | Download |
|---------|----------|
| **Kaggle Phishing URL Dataset** | https://www.kaggle.com/datasets/shashwatwork/phishing-dataset-for-machine-learning |
| **PhishTank** (CSV feed) | https://phishtank.org/developer_info.php |
| **URLhaus** (malicious URLs CSV) | https://urlhaus.abuse.ch/downloads/csv_recent/ |
| **Tranco Top 1M** (legitimate URLs) | https://tranco-list.eu/ |

Place in: `ai/datasets/cybersecurity/phishing/raw/`

---

### 3. `ransomware_detection` — File Behaviour (BiLSTM)
| Dataset | Download |
|---------|----------|
| **EMBER 2018** (PE features, ~8 GB) | https://github.com/elastic/ember |
| **MalwareBazaar** (metadata CSV) | https://bazaar.abuse.ch/export/ |
| **CIC Ransomware 2022** | https://www.unb.ca/cic/datasets/ → search ransomware |
| **Cuckoo API call sequences** | https://www.kaggle.com/datasets/ang3loliveira/malware-analysis-datasets-api-call-sequences |

Place in: `ai/datasets/cybersecurity/ransomware/raw/`

---

### 4. `threat_detection` — Multi-class Threat Classifier
| Dataset | Download |
|---------|----------|
| **CICIDS 2017/2018** (shared with anomaly_detection) | https://www.unb.ca/cic/datasets/ |
| **CICDDoS2019** | https://www.unb.ca/cic/datasets/ddos-2019.html |
| **CTU-13** (botnet netflow) | https://www.stratosphereips.org/datasets-ctu13 |

Place in: `ai/datasets/cybersecurity/threat-detection/raw/`

---

## 💰 Finance (2 Models)

### 5. `finance_fraud_anomaly` — Transaction Fraud (Isolation Forest)
| Dataset | Download |
|---------|----------|
| **Kaggle Credit Card Fraud** (284k rows) | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud |
| **IEEE-CIS Fraud Detection** | https://www.kaggle.com/c/ieee-fraud-detection/data |
| **PaySim** (synthetic mobile fraud) | https://www.kaggle.com/datasets/ealaxi/paysim1 |

Place in: `ai/datasets/finance/fraud-detection/raw/`

---

### 6. `finance_trading_behavior` — Market Manipulation (BiLSTM)
| Dataset | Download |
|---------|----------|
| **LOBSTER Order Book** (free sample) | https://lobsterdata.com/ |
| **Binance Historical Klines** (free, all pairs) | https://data.binance.vision/?prefix=data/spot/daily/klines/ |
| **Synthetic sequences** (pipeline built-in fallback) | `ai/training/pipelines/finance_trading_behavior.py` |

Place in: `ai/datasets/finance/market-manipulation/raw/`

---

## 🌿 Agriculture (4 Preprocessing Tracks)

### 7. `vineyard_lidar` — Vineyard LiDAR Perception
| Dataset | Download |
|---------|----------|
| **Vineyard LiDAR point clouds** | (project-specific / public vineyard scans) |
| **Canopy geometry exports (CSV/Parquet)** | (derived from LiDAR processing tools) |

Place in: `ai/datasets/agriculture/raw/vineyard-lidar/`

---

### 8. `yield_prediction` — Yield Forecasting
| Dataset | Download |
|---------|----------|
| **Yield records (block/row level)** | (project-specific farm records) |
| **Weather time-series** | (local station exports / public weather APIs) |
| **Management events (irrigation/fertilization)** | (optional) |

Place in: `ai/datasets/agriculture/raw/yield-prediction/`

---

### 9. `path_planning` — Navigation / Coverage Optimization
| Dataset | Download |
|---------|----------|
| **Trajectory logs (GPS/odometry)** | (robot/tractor runs) |
| **Obstacle maps / row geometry** | (GIS/CSV exports) |

Place in: `ai/datasets/agriculture/raw/path-planning/`

---

### 10. `anomaly_detection` — Field Anomaly Monitoring
| Dataset | Download |
|---------|----------|
| **Sensor telemetry (soil, moisture, temperature)** | (IoT exports) |
| **Disease/stress labels (if available)** | (manual scouting annotations) |

Place in: `ai/datasets/agriculture/raw/anomaly-detection/`

---

## 🏭 Manufacturing (2 Models)

### 9. `manufacturing_process_anomaly` — Process Anomaly (Autoencoder)
| Dataset | Download |
|---------|----------|
| **AI4I 2020 Predictive Maintenance** (UCI, ~1 MB) | https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset |
| **Tennessee Eastman Process** | https://www.kaggle.com/datasets/averkij/tennessee-eastman-process-simulation-dataset |
| **Bosch Production Line Performance** | https://www.kaggle.com/c/bosch-production-line-performance/data |

Place in: `ai/datasets/manufacturing/production-anomaly/raw/`

---

### 10. `manufacturing_maintenance_forecast` — Predictive Maintenance (Transformer)
| Dataset | Download |
|---------|----------|
| **NASA C-MAPSS Turbofan** (~3 MB) | https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository |
| **CWRU Bearing Dataset** | https://engineering.case.edu/bearingdatacenter/download-data-file |
| **Azure Predictive Maintenance (Kaggle)** | https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance |

Place in: `ai/datasets/manufacturing/predictive-maintenance/raw/`

---

## 🗂️ Folder Structure

```
ai/datasets/
├── cybersecurity/
│   ├── threat-detection/
│   │   ├── raw/     ← CICIDS 2017/18, UNSW-NB15, NSL-KDD, CTU-13, CICDDoS2019
│   │   └── processed/
│   ├── phishing/
│   │   ├── raw/     ← PhishTank, URLhaus, Kaggle Phishing URL, Tranco Top 1M
│   │   └── processed/
│   └── ransomware/
│       ├── raw/     ← EMBER 2018, MalwareBazaar, CIC Ransomware, Cuckoo traces
│       └── processed/
├── finance/
│   ├── fraud-detection/
│   │   ├── raw/     ← Kaggle CC Fraud, IEEE-CIS, PaySim
│   │   └── processed/
│   └── market-manipulation/
│       ├── raw/     ← LOBSTER, Binance Klines, synthetic
│       └── processed/
├── agriculture/
│   ├── raw/
│   │   ├── vineyard-lidar/
│   │   ├── yield-prediction/
│   │   ├── path-planning/
│   │   └── anomaly-detection/
│   └── processed/
└── manufacturing/
    ├── production-anomaly/
    │   ├── raw/     ← AI4I UCI, Tennessee Eastman, Bosch
    │   └── processed/
    └── predictive-maintenance/
        ├── raw/     ← NASA C-MAPSS, CWRU Bearing, Azure PM
        └── processed/
```

---

## ✅ No-Registration Quick Start

These can be downloaded **immediately** with no account:

| Model | Dataset | Approx. Size |
|-------|---------|-------------|
| `anomaly_detection` | CICIDS 2017 | ~2.8 GB |
| `phishing_detection` | Kaggle Phishing URL | ~5 MB |
| `ransomware_detection` | EMBER 2018 | ~8 GB |
| `finance_fraud_anomaly` | Kaggle CC Fraud | ~150 MB |
| `finance_trading_behavior` | Binance Klines | configurable |
| `manufacturing_process_anomaly` | AI4I UCI | ~1 MB |
| `manufacturing_maintenance_forecast` | NASA C-MAPSS | ~3 MB |

> 💡 All training pipelines include **synthetic data generation** as a fallback.
> You can train and test every model without downloading anything first.
> Replace synthetic data with real datasets for production-quality accuracy.

---

## 🔄 Training After Downloading

```bash
cd ai/training
pip install -r requirements.txt

# Cybersecurity
python pipelines/anomaly_detection.py
python pipelines/phishing_detection.py
python pipelines/ransomware_detection.py
python pipelines/threat_detection.py

# Finance
python pipelines/finance_fraud_anomaly.py
python pipelines/finance_trading_behavior.py

# Manufacturing
python pipelines/manufacturing_process_anomaly.py
python pipelines/manufacturing_maintenance_forecast.py

# Agriculture preprocessing
python preprocessing/preprocess_agriculture.py \
    --raw-dir ../datasets/agriculture/raw \
    --output-dir ../datasets/agriculture/processed
```

Trained models are exported to `ai/models/<model_name>/model.onnx`.
