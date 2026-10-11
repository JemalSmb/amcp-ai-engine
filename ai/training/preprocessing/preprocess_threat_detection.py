"""
Threat Detection — Raw Data Preprocessor

Converts CTU-13 binetflow files and API Call Sequences into CSV
with numeric features + multi-class 'label' column.

Input sources (from Project_Datasets/cybersecurity/threat_detection/raw/):
  - CTU-13-Dataset/{1..13}/capture*.binetflow
  - API Call Sequences/dynamic_api_call_sequence_per_malware_100_0_306.csv

Output:
  → processed/ctu13_flows.csv          (network flow features + label)
  → processed/api_call_sequences.csv   (already CSV, cleaned)
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def process_ctu13(raw_dir: Path, output_dir: Path) -> bool:
    """
    Parse CTU-13 binetflow files → unified CSV.

    Binetflow columns:
    StartTime, Dur, Proto, SrcAddr, Sport, Dir, DstAddr, Dport,
    State, sTos, dTos, TotPkts, TotBytes, SrcBytes, Label

    Label contains: 'flow=Background-*', 'flow=From-Botnet-*', 'flow=Normal-*'
    We map these to: Background=0, Normal=1, Botnet=2
    """
    ctu_dir = raw_dir / "CTU-13-Dataset"
    if not ctu_dir.exists():
        logger.warning(f"CTU-13-Dataset not found at {ctu_dir}")
        return False

    logger.info(f"Processing CTU-13 from {ctu_dir}")

    frames = []
    for scenario_dir in sorted(ctu_dir.iterdir()):
        if not scenario_dir.is_dir():
            continue
        for bf in scenario_dir.glob("*.binetflow"):
            try:
                df = pd.read_csv(bf, low_memory=False)
                df["scenario"] = scenario_dir.name
                frames.append(df)
                logger.info(f"  {bf.parent.name}/{bf.name}: {len(df)} flows")
            except Exception as e:
                logger.warning(f"  Skipping {bf}: {e}")

    if not frames:
        logger.warning("No binetflow files parsed")
        return False

    df = pd.concat(frames, ignore_index=True)

    # Standardise column names (strip whitespace)
    df.columns = [c.strip() for c in df.columns]

    # Parse Label column into threat class
    def _map_label(raw_label: str) -> str:
        raw_label = str(raw_label).lower().strip()
        if "botnet" in raw_label:
            return "botnet"
        elif "normal" in raw_label:
            return "normal"
        else:
            return "background"

    df["threat_class"] = df["Label"].apply(_map_label)

    # Drop rows with parsing issues
    df = df.dropna(subset=["Dur", "TotPkts", "TotBytes", "SrcBytes"])

    # Numeric conversion
    for col in ["Dur", "TotPkts", "TotBytes", "SrcBytes"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Dur", "TotPkts", "TotBytes", "SrcBytes"])

    # Encode categoricals
    le_proto = LabelEncoder()
    df["Proto_enc"] = le_proto.fit_transform(df["Proto"].astype(str))

    le_state = LabelEncoder()
    df["State_enc"] = le_state.fit_transform(df["State"].astype(str))

    le_dir = LabelEncoder()
    df["Dir_enc"] = le_dir.fit_transform(df["Dir"].astype(str))

    # Encode sport/dport as numeric (some are service names)
    for col in ["Sport", "Dport"]:
        df[col + "_enc"] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(int)

    # Derived features
    df["bytes_per_pkt"] = df["TotBytes"] / (df["TotPkts"] + 1e-8)
    df["src_byte_ratio"] = df["SrcBytes"] / (df["TotBytes"] + 1e-8)
    df["pkts_per_sec"] = df["TotPkts"] / (df["Dur"] + 1e-8)
    df["bytes_per_sec"] = df["TotBytes"] / (df["Dur"] + 1e-8)

    # Encode sTos, dTos as numeric
    df["sTos"] = pd.to_numeric(df.get("sTos", 0), errors="coerce").fillna(0)
    df["dTos"] = pd.to_numeric(df.get("dTos", 0), errors="coerce").fillna(0)

    # Build output DataFrame
    feature_cols = [
        "Dur", "Proto_enc", "Dir_enc", "State_enc", "Sport_enc", "Dport_enc",
        "sTos", "dTos", "TotPkts", "TotBytes", "SrcBytes",
        "bytes_per_pkt", "src_byte_ratio", "pkts_per_sec", "bytes_per_sec",
    ]

    # Encode label: background=0, normal=1, botnet=2
    label_map = {"background": 0, "normal": 1, "botnet": 2}
    df["label"] = df["threat_class"].map(label_map)

    out_df = df[feature_cols + ["label"]].copy()
    out_df = out_df.replace([np.inf, -np.inf], 0).fillna(0)

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "ctu13_flows.csv"
    out_df.to_csv(out, index=False)

    for cls_name, cls_val in label_map.items():
        cnt = int((out_df["label"] == cls_val).sum())
        logger.info(f"  {cls_name}: {cnt} flows")
    logger.info(f"  → {out} ({len(out_df)} total flows)")
    return True


def process_api_calls(raw_dir: Path, output_dir: Path) -> bool:
    """Clean API Call Sequences CSV for downstream use."""
    api_dir = raw_dir / "API Call Sequences"
    if not api_dir.exists():
        logger.warning(f"API Call Sequences not found at {api_dir}")
        return False

    csv_files = list(api_dir.glob("*.csv"))
    if not csv_files:
        logger.warning("No CSV files in API Call Sequences/")
        return False

    logger.info(f"Processing API Call Sequences from {api_dir}")

    df = pd.read_csv(csv_files[0], low_memory=False)

    # Columns: hash, t_0..t_99, malware
    # Encode API call names at each timestep to integers
    # First, collect all unique API names
    timestep_cols = [c for c in df.columns if c.startswith("t_")]
    all_apis = set()
    for col in timestep_cols:
        all_apis.update(df[col].dropna().unique())

    api_to_idx = {api: idx for idx, api in enumerate(sorted(all_apis))}

    for col in timestep_cols:
        df[col] = df[col].map(api_to_idx).fillna(0).astype(int)

    # Rename malware → label
    if "malware" in df.columns:
        df = df.rename(columns={"malware": "label"})

    # Drop hash column
    if "hash" in df.columns:
        df = df.drop(columns=["hash"])

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "api_call_sequences.csv"
    df.to_csv(out, index=False)
    logger.info(f"  → {out} ({len(df)} samples, {len(timestep_cols)} timesteps, {len(api_to_idx)} unique API calls)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Preprocess threat detection datasets")
    parser.add_argument("--raw-dir", type=str, required=True,
                        help="Path to threat_detection/raw/ directory")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Path to processed/ output directory")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)

    ok1 = process_ctu13(raw_dir, output_dir)
    ok2 = process_api_calls(raw_dir, output_dir)

    if ok1 or ok2:
        logger.info("Threat detection preprocessing complete.")
    else:
        logger.error("No datasets found. Check --raw-dir path.")


if __name__ == "__main__":
    main()
