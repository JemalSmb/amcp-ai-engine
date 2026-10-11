"""
Anomaly Detection — Raw Data Preprocessor

Converts NSL-KDD (.txt) and UNSW-NB15 (.csv) into a unified CSV
with numeric features + binary 'label' column for the Isolation Forest pipeline.

Input sources (from Project_Datasets/cybersecurity/anomaly-detection/raw/):
  - nsl-kdd/KDDTrain+.txt    (no header, 43 columns)
  - nsl-kdd/KDDTest+.txt
  - UNSW-NB15/*.csv           (if present)

Output:
  → processed/anomaly_detection_train.csv
  → processed/anomaly_detection_test.csv
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# NSL-KDD column names (41 features + attack_type + difficulty_level)
NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "attack_type", "difficulty_level",
]

# Categorical columns that need encoding
NSL_KDD_CATEGORICAL = ["protocol_type", "service", "flag"]

# UNSW-NB15 columns to drop (identifiers, not features)
UNSW_DROP_COLS = ["srcip", "dstip", "sport", "dsport", "Stime", "Ltime"]


def process_nsl_kdd(raw_dir: Path, output_dir: Path) -> bool:
    """Process NSL-KDD .txt files → CSV with numeric features + label."""
    train_path = raw_dir / "nsl-kdd" / "KDDTrain+.txt"
    test_path = raw_dir / "nsl-kdd" / "KDDTest+.txt"

    if not train_path.exists():
        logger.warning(f"NSL-KDD not found at {train_path}")
        return False

    logger.info(f"Processing NSL-KDD from {raw_dir / 'nsl-kdd'}")

    # Load without header
    df_train = pd.read_csv(train_path, header=None, names=NSL_KDD_COLUMNS)
    if test_path.exists():
        df_test = pd.read_csv(test_path, header=None, names=NSL_KDD_COLUMNS)
    else:
        df_test = None

    def _transform(df: pd.DataFrame) -> pd.DataFrame:
        # Binary label: 'normal' → 0, everything else → 1 (anomaly)
        df["label"] = (df["attack_type"].str.strip() != "normal").astype(int)
        df = df.drop(columns=["attack_type", "difficulty_level"])

        # Encode categorical features
        for col in NSL_KDD_CATEGORICAL:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

        return df

    df_train = _transform(df_train)
    if df_test is not None:
        df_test = _transform(df_test)

    output_dir.mkdir(parents=True, exist_ok=True)

    train_out = output_dir / "nsl_kdd_train.csv"
    df_train.to_csv(train_out, index=False)
    logger.info(f"  NSL-KDD train → {train_out} ({len(df_train)} rows, {df_train['label'].sum()} anomalies)")

    if df_test is not None:
        test_out = output_dir / "nsl_kdd_test.csv"
        df_test.to_csv(test_out, index=False)
        logger.info(f"  NSL-KDD test  → {test_out} ({len(df_test)} rows, {df_test['label'].sum()} anomalies)")

    return True


def process_unsw_nb15(raw_dir: Path, output_dir: Path) -> bool:
    """Process UNSW-NB15 CSV files → unified CSV with label column."""
    unsw_dir = raw_dir / "UNSW-NB15"
    if not unsw_dir.exists():
        logger.warning(f"UNSW-NB15 not found at {unsw_dir}")
        return False

    csv_files = sorted(unsw_dir.glob("*.csv"))
    if not csv_files:
        logger.warning("No CSV files found in UNSW-NB15 directory")
        return False

    logger.info(f"Processing UNSW-NB15 from {unsw_dir}")

    frames = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, low_memory=False)
            frames.append(df)
        except Exception as e:
            logger.warning(f"  Skipping {f.name}: {e}")

    if not frames:
        return False

    df = pd.concat(frames, ignore_index=True)

    # Standardise label column
    label_candidates = ["Label", "label", "attack_cat", "Attack"]
    label_col = None
    for c in label_candidates:
        if c in df.columns:
            label_col = c
            break

    if label_col:
        if df[label_col].dtype == object:
            # Binary: 'Normal' → 0, everything else → 1
            df["label"] = (df[label_col].str.strip().str.lower() != "normal").astype(int)
        else:
            df["label"] = (df[label_col] != 0).astype(int)
        if label_col != "label":
            df = df.drop(columns=[label_col])
    else:
        df["label"] = 0

    # Drop identifier columns if present
    for col in UNSW_DROP_COLS:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Drop attack_cat if still present (we already made binary label)
    if "attack_cat" in df.columns:
        df = df.drop(columns=["attack_cat"])

    # Encode remaining string columns
    for col in df.select_dtypes(include=["object"]).columns:
        if col != "label":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "unsw_nb15.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"  UNSW-NB15 → {out_path} ({len(df)} rows, {df['label'].sum()} anomalies)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Preprocess anomaly detection datasets")
    parser.add_argument("--raw-dir", type=str, required=True,
                        help="Path to raw/ directory containing nsl-kdd/ and/or UNSW-NB15/")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Path to processed/ output directory")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)

    ok_kdd = process_nsl_kdd(raw_dir, output_dir)
    ok_unsw = process_unsw_nb15(raw_dir, output_dir)

    if ok_kdd or ok_unsw:
        logger.info("Anomaly detection preprocessing complete.")
    else:
        logger.error("No datasets found to process. Check --raw-dir path.")


if __name__ == "__main__":
    main()
