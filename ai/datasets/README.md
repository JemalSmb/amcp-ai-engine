# AI Datasets

Folder layout and documentation for the datasets used to train the AMCP models. The raw data itself is **not** in git (about 315 GB on an external drive).

```text
datasets/
├── cybersecurity/   # anomaly-detection, phishing, ransomware, threat-detection
├── finance/         # fraud-detection, market-manipulation
├── manufacturing/   # predictive-maintenance, production-anomaly
├── agriculture/
└── common/          # time-series-benchmarks, geolocation, weather
```

Each task folder holds `raw/` (original downloads, untouched) and `processed/` (output of `ai/training/preprocessing/`), plus its own `README.md` describing the expected format.

## Documents

- [PROJECT_DATASETS.md](PROJECT_DATASETS.md): which datasets each model uses, with sizes
- [DATASET_GUIDE.md](DATASET_GUIDE.md): download links and per-dataset notes

## Workflow

1. Copy or symlink a task folder from the external drive into `ai/datasets/<domain>/<task>/`.
2. Run the preprocessing script (see `ai/training/README.md`) to produce CSVs in `processed/`.
3. Train with `--data` pointing at the `processed/` folder.

## Conventions

- Pipelines read the **first `*.csv`** in the folder they are given.
- Numeric feature columns plus a `label` column; non-numeric columns are dropped and `NaN`/infinite values replaced.
- Drop identifiers (IP addresses, flow IDs, timestamps) since they leak the label.
- `.gitkeep` files only preserve the empty folder structure. Never commit raw data.
