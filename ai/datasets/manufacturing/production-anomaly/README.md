# Production anomaly datasets

Machine and process sensor readings for the process anomaly autoencoder.

**Model:** `manufacturing_process_anomaly` | **Reference data:** Tennessee Eastman Process, AI4I 2020, NASA C-MAPSS

## Data format
- **File:** a CSV placed in `raw/` (preprocessed output goes in `processed/`). The pipeline reads the **first `*.csv`** in the folder it is given.
- **Columns:** numeric feature columns plus one label column. Non-numeric columns are dropped, and `NaN` or infinite values are replaced.
- **Label column:** `fault`, `label`, `anomaly` or `failure`. `0` = normal, `1` = fault. Training uses normal rows to learn what normal looks like.
- **Features:** the serving model expects **14** numeric features (see `ai/models/manufacturing_process_anomaly/config.yaml` for names and order). Your columns should map to that count, or update the config and retrain.

## What to look for in a dataset
- **Multivariate sensor snapshots** (vibration, temperature, current, pressure, speed, efficiency, power, torque and similar) from a running machine or process.
- A good amount of **normal** operation data. Faults can be few or missing.
- Consistent units and sampling interval across machines; rescale per machine if ranges differ.
- Fault labels, if present, should mark where the fault actually starts, not a whole run.
## Train on your own data
1. Put your files in `raw/` (keep the original download as-is).
2. Convert them to the format above and write the result to `processed/`. Start from `ai/training/preprocessing/preprocess_manufacturing.py` and adapt the column mapping.
3. Train: `python ai/training/pipelines/manufacturing_process_anomaly.py --config ai/models/manufacturing_process_anomaly/config.yaml --data ai/datasets/manufacturing/production-anomaly/processed --output <out_dir>`
4. If no data is found the pipeline silently falls back to **synthetic** data, so check the log line `Loading real data from ...` to confirm yours was used.
