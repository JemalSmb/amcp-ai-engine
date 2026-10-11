"""
Manufacturing — Raw Data Preprocessor

Converts manufacturing datasets into training-ready CSV/npy files.

Input sources:
  1. AI4I 2020 Predictive Maintenance (ai4i2020.csv)
     → CSV with sensor readings + Machine failure label
     → Used for: process_anomaly (Autoencoder) pipeline

  2. NASA C-MAPSS Turbofan (CMAPSSData.zip)
     → Text files: train_FD001..4, test_FD001..4, RUL_FD001..4
     → Used for: maintenance_forecast (Transformer) pipeline

  3. Tennessee Eastman Process (TEP_*.RData)
     → R data files with 52 process variables + 20 fault types
     → Used for: process_anomaly (Autoencoder) pipeline

Output:
  → process_anomaly/processed/ai4i_process.csv
  → process_anomaly/processed/tep_process.csv       (if pyreadr available)
  → predictive_maintenance/processed/sequences.npy  (C-MAPSS time-series)
  → predictive_maintenance/processed/targets.npy
"""

import argparse
import logging
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ── AI4I 2020

def process_ai4i(raw_dir: Path, output_dir: Path) -> bool:
    """
    Process AI4I 2020 Predictive Maintenance → CSV for process anomaly detector.

    Columns: UDI, Product ID, Type, Air temperature [K], Process temperature [K],
             Rotational speed [rpm], Torque [Nm], Tool wear [min],
             Machine failure, TWF, HDF, PWF, OSF, RNF

    Maps to model features: temperature, rpm, torque, tool_wear, etc.
    """
    csv_path = None
    for pattern in ["ai4i*/*.csv", "ai4i*.csv", "*.csv"]:
        matches = list(raw_dir.glob(pattern))
        if matches:
            csv_path = matches[0]
            break

    if csv_path is None:
        logger.warning(f"AI4I CSV not found in {raw_dir}")
        return False

    logger.info(f"Processing AI4I 2020 from {csv_path}")
    df = pd.read_csv(csv_path)

    # Encode product type (L/M/H)
    if "Type" in df.columns:
        le = LabelEncoder()
        df["type_enc"] = le.fit_transform(df["Type"])

    # Rename to model-friendly names and select features
    rename_map = {
        "Air temperature [K]": "air_temperature_k",
        "Process temperature [K]": "process_temperature_k",
        "Rotational speed [rpm]": "rpm",
        "Torque [Nm]": "torque_nm",
        "Tool wear [min]": "tool_wear_min",
        "Machine failure": "label",
    }
    df = df.rename(columns=rename_map)

    # Derived features to better match the 14-feature process_anomaly model
    df["temp_diff"] = df["process_temperature_k"] - df["air_temperature_k"]
    df["power_kw"] = df["rpm"] * df["torque_nm"] / 9550  # mechanical power approx
    df["torque_per_rpm"] = df["torque_nm"] / (df["rpm"] + 1e-8)

    # Include failure type columns as features
    feature_cols = [
        "air_temperature_k", "process_temperature_k", "rpm",
        "torque_nm", "tool_wear_min", "temp_diff", "power_kw",
        "torque_per_rpm",
        "TWF", "HDF", "PWF", "OSF", "RNF",
    ]
    if "type_enc" in df.columns:
        feature_cols.append("type_enc")

    out_df = df[feature_cols + ["label"]].copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "ai4i_process.csv"
    out_df.to_csv(out, index=False)
    logger.info(f"  → {out} ({len(out_df)} rows, failures={int(out_df['label'].sum())})")
    return True


# ── NASA C-MAPSS

CMAPSS_COLS = (
    ["unit", "cycle"] +
    ["op_setting_1", "op_setting_2", "op_setting_3"] +
    [f"sensor_{i}" for i in range(1, 22)]
)

# Sensors with near-zero variance in FD001 (typically dropped)
CMAPSS_DROP_SENSORS = ["sensor_1", "sensor_5", "sensor_6", "sensor_10",
                       "sensor_16", "sensor_18", "sensor_19"]


def process_cmapss(raw_dir: Path, output_dir: Path, seq_length: int = 30, forecast_horizon: int = 5) -> bool:
    """
    Process NASA C-MAPSS Turbofan → sliding-window sequences for Transformer.

    Steps:
      1. Extract CMAPSSData.zip if needed
      2. Parse train_FD001.txt (simplest sub-dataset)
      3. Compute RUL for each unit
      4. Normalise sensor features
      5. Create sliding windows → sequences.npy + targets.npy
    """
    # Find the zip file
    zip_candidates = list(raw_dir.glob("**/CMAPSSData.zip"))
    if not zip_candidates:
        logger.warning("CMAPSSData.zip not found")
        return False

    zip_path = zip_candidates[0]
    extract_dir = zip_path.parent / "CMAPSSData"

    # Extract if not already done
    if not extract_dir.exists():
        logger.info(f"Extracting {zip_path}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

    # Find train_FD001.txt
    train_file = None
    for f in extract_dir.rglob("train_FD001.txt"):
        train_file = f
        break

    if train_file is None:
        logger.warning("train_FD001.txt not found in extracted C-MAPSS data")
        return False

    logger.info(f"Processing C-MAPSS from {train_file}")

    # Parse whitespace-separated file
    df = pd.read_csv(train_file, sep=r"\s+", header=None, names=CMAPSS_COLS)

    # Compute RUL: max_cycle_per_unit - current_cycle
    max_cycles = df.groupby("unit")["cycle"].max().to_dict()
    df["RUL"] = df.apply(lambda r: max_cycles[r["unit"]] - r["cycle"], axis=1)

    # Drop low-variance sensors and operational settings
    drop_cols = CMAPSS_DROP_SENSORS + ["op_setting_3"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Select sensor features
    sensor_cols = [c for c in df.columns if c.startswith("sensor_") or c.startswith("op_setting")]
    n_features = len(sensor_cols)

    # Normalise per column
    for col in sensor_cols:
        mean = df[col].mean()
        std = df[col].std() + 1e-8
        df[col] = (df[col] - mean) / std

    # Build sliding windows per unit
    sequences, targets = [], []
    for unit_id in sorted(df["unit"].unique()):
        unit_df = df[df["unit"] == unit_id].sort_values("cycle")
        feats = unit_df[sensor_cols].values
        ruls = unit_df["RUL"].values

        total_len = seq_length + forecast_horizon
        for i in range(len(feats) - total_len + 1):
            seq = feats[i : i + seq_length]
            target = feats[i + seq_length : i + total_len]
            sequences.append(seq)
            targets.append(target)

    sequences = np.array(sequences, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)

    # Trim to match model's expected feature dim (8)
    # Use first 8 sensor columns if more are available
    if sequences.shape[2] > 8:
        sequences = sequences[:, :, :8]
        targets = targets[:, :, :8]

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "sequences.npy", sequences)
    np.save(output_dir / "targets.npy", targets)
    logger.info(f"  → sequences.npy {sequences.shape}, targets.npy {targets.shape}")
    logger.info(f"    ({len(sequences)} windows from {len(df['unit'].unique())} engine units)")
    return True


# ── Tennessee Eastman Process 

def process_tep(raw_dir: Path, output_dir: Path) -> bool:
    """
    Process Tennessee Eastman Process .RData → CSV for process anomaly detector.

    TEP has 52 variables (XMEAS 1-41 + XMV 1-11) across 20 fault types + fault-free.
    """
    tep_dir = None
    for d in raw_dir.iterdir():
        if d.is_dir() and "Tennessee" in d.name:
            tep_dir = d
            break

    if tep_dir is None:
        logger.warning("Tennessee Eastman directory not found")
        return False

    rdata_files = list(tep_dir.glob("*.RData"))
    if not rdata_files:
        logger.warning("No .RData files found in TEP directory")
        return False

    try:
        import pyreadr
    except ImportError:
        logger.warning("pyreadr not installed — skipping TEP. Install with: pip install pyreadr")
        return False

    logger.info(f"Processing Tennessee Eastman from {tep_dir}")

    frames = []

    # Process fault-free files
    for rdata_file in sorted(rdata_files):
        logger.info(f"  Reading {rdata_file.name}...")
        try:
            result = pyreadr.read_r(str(rdata_file))
            for key, df in result.items():
                if "FaultFree" in rdata_file.name:
                    df["label"] = 0
                else:
                    # Faulty data — each simulation run is a different fault
                    # Use faultNumber column if present, otherwise label as 1
                    if "faultNumber" in df.columns:
                        df["label"] = (df["faultNumber"] != 0).astype(int)
                    else:
                        df["label"] = 1
                frames.append(df)
        except Exception as e:
            logger.warning(f"  Error reading {rdata_file.name}: {e}")

    if not frames:
        return False

    df = pd.concat(frames, ignore_index=True)

    # Keep only numeric columns + label
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "label" not in numeric_cols:
        numeric_cols.append("label")

    # Drop simulation metadata columns if present
    drop_meta = ["simulationRun", "sample", "faultNumber"]
    for col in drop_meta:
        if col in numeric_cols:
            numeric_cols.remove(col)

    out_df = df[numeric_cols].replace([np.inf, -np.inf], 0).fillna(0)

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "tep_process.csv"
    out_df.to_csv(out, index=False)
    logger.info(f"  → {out} ({len(out_df)} rows, {len(numeric_cols)-1} features, faults={int(out_df['label'].sum())})")
    return True


def main():
    parser = argparse.ArgumentParser(description="Preprocess manufacturing datasets")
    parser.add_argument("--raw-dir", type=str, required=True,
                        help="Base path containing predictive-maintenance/raw/ and production-anomaly/raw/")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Base path for processed output (will create subdirectories)")
    parser.add_argument("--seq-length", type=int, default=30,
                        help="Sequence length for C-MAPSS windows (default: 30)")
    parser.add_argument("--forecast-horizon", type=int, default=5,
                        help="Forecast horizon for C-MAPSS (default: 5)")
    args = parser.parse_args()

    raw_base = Path(args.raw_dir)
    out_base = Path(args.output_dir)

    # AI4I is in predictive-maintenance/raw/ on the external drive
    # but logically belongs to process_anomaly
    pm_raw = raw_base / "predictive-maintenance" / "raw"
    pa_raw = raw_base / "production-anomaly" / "raw"

    # If given a flat path, try both structures
    if not pm_raw.exists() and not pa_raw.exists():
        pm_raw = raw_base
        pa_raw = raw_base

    ok1 = process_ai4i(pm_raw, out_base / "process_anomaly")
    ok2 = process_cmapss(pa_raw, out_base / "predictive_maintenance", args.seq_length, args.forecast_horizon)
    ok3 = process_tep(pa_raw, out_base / "process_anomaly")

    if ok1 or ok2 or ok3:
        logger.info("Manufacturing preprocessing complete.")
    else:
        logger.error("No manufacturing datasets found. Check --raw-dir path.")


if __name__ == "__main__":
    main()
