# Threat detection datasets

Network-flow records for the multi-class threat classifier. The `anomaly_detection` model (unsupervised Isolation Forest) reads the same kind of flow data, so its data can sit here too.

**Model:** `threat_detection` | **Reference data:** CICIDS 2017/2018, CICDDoS2019, CTU-13, UNSW-NB15, NSL-KDD

## Data format
- **File:** a CSV placed in `raw/` (preprocessed output goes in `processed/`). The pipeline reads the **first `*.csv`** in the folder it is given.
- **Columns:** numeric feature columns plus one label column. Non-numeric columns are dropped, and `NaN` or infinite values are replaced.
- **Label column:** `label` (required for threat detection). `0` = background, `1` = normal, `2` = botnet. For the anomaly model use `0` = benign, `1` = attack. Without a label the anomaly pipeline treats everything as benign.
- **Features:** the serving model expects **45 (anomaly) or as set in the threat config** numeric features (see `ai/models/threat_detection/config.yaml` for names and order). Your columns should map to that count, or update the config and retrain.

## What to look for in a dataset
- One row per **network flow** (not per packet), with duration, packet and byte counts per direction, rates and inter-arrival times (CICFlowMeter-style, like CICIDS or CICDDoS).
- A trustworthy ground-truth label per flow, not only per capture file.
- Drop identifiers such as IP addresses, ports used as IDs, flow IDs and timestamps, since they leak the label and do not generalise.
- Check class balance. Attack classes are usually a small share; the anomaly model trains on benign rows only (`use_only_benign`).
- Make sure train and test come from different days or captures, so you measure generalisation and not memorisation.
## Train on your own data
1. Put your files in `raw/` (keep the original download as-is).
2. Convert them to the format above and write the result to `processed/`. Start from `ai/training/preprocessing/preprocess_threat_detection.py` and adapt the column mapping.
3. Train: `python ai/training/pipelines/threat_detection.py --config ai/models/threat_detection/config.yaml --data ai/datasets/cybersecurity/threat-detection/processed --output <out_dir>`
4. If no data is found the pipeline silently falls back to **synthetic** data, so check the log line `Loading real data from ...` to confirm yours was used.
