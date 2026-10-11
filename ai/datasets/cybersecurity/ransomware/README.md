# Ransomware datasets

Sequences of behaviour records for the ransomware BiLSTM.

**Model:** `ransomware_detection` | **Reference data:** EMBER 2018, MalwareBazaar, CIC Ransomware 2022

## Data format
- **Files:** `sequences.npy` and `labels.npy` in the folder passed to the pipeline (`processed/`).
- **`sequences.npy`:** float array of shape `(N, 64, 32)`.
- **`labels.npy`:** integer array of shape `(N,)`; `0` = benign, `1` = ransomware.
- **Features per step** (see `ai/models/ransomware_detection/config.yaml`): 32 values per event window, such as operation type, file size, bytes written, entropy before and after, operations per second, directory type and process info.

## What to look for in a dataset
- Data with a **time order**: file-system or API events from one process or sample, which you can cut into windows of 64 events.
- Entropy and write-burst signals matter most (encryption raises entropy and rewrites many files quickly).
- Static feature sets like EMBER are not true behaviour traces; the bundled preprocessor flattens them into a fixed 32-value vector and slides a window over samples. Real behaviour logs will do better.
- Keep windows from one sample or machine in the same split.
## Train on your own data
1. Put your files in `raw/` (keep the original download as-is).
2. Convert them to the format above and write the result to `processed/`. Start from `ai/training/preprocessing/preprocess_ransomware.py` and adapt the column mapping.
3. Train: `python ai/training/pipelines/ransomware_detection.py --config ai/models/ransomware_detection/config.yaml --data ai/datasets/cybersecurity/ransomware/processed --output <out_dir>`
4. If no data is found the pipeline silently falls back to **synthetic** data, so check the log line `Loading real data from ...` to confirm yours was used.
