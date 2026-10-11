# Phishing datasets

URL-level features for the phishing classifier (Random Forest).

**Model:** `phishing_detection` | **Reference data:** Phishing_Legitimate_full.csv, PhishTank, URLhaus, Tranco

## Data format
- **File:** a CSV placed in `raw/` (preprocessed output goes in `processed/`). The pipeline reads the **first `*.csv`** in the folder it is given.
- **Columns:** numeric feature columns plus one label column. Non-numeric columns are dropped, and `NaN` or infinite values are replaced.
- **Label column:** `label`, `result`, `CLASS_LABEL`, `class_label`, `target` or `Target` (first match is used). `0` = legitimate, `1` = phishing; `-1` is mapped to `0`.
- **Features:** the serving model expects **30** numeric features (see `ai/models/phishing_detection/config.yaml` for names and order). Your columns should map to that count, or update the config and retrain.

## What to look for in a dataset
- Already **feature-engineered** URL data (length, number of dots, hyphens, `@`, IP-in-URL, HTTPS, subdomains, suspicious TLD, domain age) or raw URL lists you can extract those features from.
- Both classes from comparable sources. Phishing URLs from a feed paired with legitimate URLs from a top-sites list can make the model learn the feed, not phishing.
- Raw URL lists (PhishTank, URLhaus, Tranco) need feature extraction first; `preprocess_phishing.py` does this.
- Remove duplicate URLs and keep each domain in only one of train or test.
## Train on your own data
1. Put your files in `raw/` (keep the original download as-is).
2. Convert them to the format above and write the result to `processed/`. Start from `ai/training/preprocessing/preprocess_phishing.py` and adapt the column mapping.
3. Train: `python ai/training/pipelines/phishing_detection.py --config ai/models/phishing_detection/config.yaml --data ai/datasets/cybersecurity/phishing/processed --output <out_dir>`
4. If no data is found the pipeline silently falls back to **synthetic** data, so check the log line `Loading real data from ...` to confirm yours was used.
