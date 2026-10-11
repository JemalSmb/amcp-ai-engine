# Predictive maintenance datasets

Equipment degradation time series for the maintenance forecaster (Transformer).

**Model:** `manufacturing_maintenance_forecast` | **Reference data:** AI4I 2020, NASA C-MAPSS turbofan

## Data format
- **Files:** `sequences.npy` and `targets.npy` in the folder passed to the pipeline (`processed/`).
- **`sequences.npy`:** float array of shape `(N, 30, 8)`.
- **`targets.npy`:** float array of shape `(N, 5, 8)`: the next 5 cycles of the same 8 features.
- **Features per step** (see `ai/models/manufacturing_maintenance_forecast/config.yaml`): 8 values per cycle: vibration, temperature, current, pressure, rpm, efficiency, operating hours, cycle count.

## What to look for in a dataset
- **Run-to-failure or long monitoring series** per unit, ordered by cycle or time, so a unit's degradation can be followed.
- At least a few dozen units, each with enough cycles for a 30-cycle lookback plus the 5-cycle horizon.
- Constant features (flat in the whole dataset) carry no signal and should be dropped or replaced.
- Split by unit, not by window, so the same machine is not in both train and test.
## Train on your own data
1. Put your files in `raw/` (keep the original download as-is).
2. Convert them to the format above and write the result to `processed/`. Start from `ai/training/preprocessing/preprocess_manufacturing.py` and adapt the column mapping.
3. Train: `python ai/training/pipelines/manufacturing_maintenance_forecast.py --config ai/models/manufacturing_maintenance_forecast/config.yaml --data ai/datasets/manufacturing/predictive-maintenance/processed --output <out_dir>`
4. If no data is found the pipeline silently falls back to **synthetic** data, so check the log line `Loading real data from ...` to confirm yours was used.
