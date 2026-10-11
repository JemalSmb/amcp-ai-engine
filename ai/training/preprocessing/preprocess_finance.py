"""
Finance Fraud — Raw Data Preprocessor

Converts Kaggle Credit Card Fraud and PaySim datasets into
CSV with numeric features + 'label' column (0=legit, 1=fraud).

Input sources (from Project_Datasets/finance/fraud-detection/raw/):
  - CreditCard/creditcard.csv              (V1..V28 PCA features + Amount + Class)
  - Synthetic Financial Dataset/PS_*.csv   (PaySim: step, type, amount, ... + isFraud)

Output:
  → processed/creditcard_fraud.csv
  → processed/paysim_fraud.csv
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def process_creditcard(raw_dir: Path, output_dir: Path) -> bool:
    """
    Process Kaggle Credit Card Fraud dataset.

    Columns: Time, V1..V28, Amount, Class
    Class: 0=legit, 1=fraud → rename to 'label'
    """
    cc_dir = raw_dir / "CreditCard"
    csv_path = cc_dir / "creditcard.csv"
    if not csv_path.exists():
        logger.warning(f"creditcard.csv not found at {csv_path}")
        return False

    logger.info(f"Processing Credit Card Fraud from {csv_path}")
    df = pd.read_csv(csv_path)

    # Rename Class → label
    df = df.rename(columns={"Class": "label"})

    # All columns are already numeric (PCA features)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "creditcard_fraud.csv"
    df.to_csv(out, index=False)
    logger.info(f"  → {out} ({len(df)} rows, fraud={int(df['label'].sum())}, ratio={df['label'].mean():.4%})")
    return True


def process_paysim(raw_dir: Path, output_dir: Path) -> bool:
    """
    Process PaySim synthetic financial dataset.

    Columns: step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
             nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud

    Feature engineering:
      - Encode transaction type (CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER)
      - Compute balance deltas and error flags
      - Drop string ID columns
    """
    syn_dir = raw_dir / "Synthetic Financial Dataset"
    csv_files = list(syn_dir.glob("PS_*.csv")) if syn_dir.exists() else []
    if not csv_files:
        logger.warning(f"PaySim CSV not found in {syn_dir}")
        return False

    logger.info(f"Processing PaySim from {csv_files[0]}")
    df = pd.read_csv(csv_files[0])

    # Encode transaction type
    le = LabelEncoder()
    df["type_enc"] = le.fit_transform(df["type"])

    # Feature engineering
    df["balance_delta_orig"] = df["newbalanceOrig"] - df["oldbalanceOrg"]
    df["balance_delta_dest"] = df["newbalanceDest"] - df["oldbalanceDest"]
    df["amount_to_balance_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1)

    # Error flags: amount doesn't match balance change
    df["orig_balance_error"] = (
        np.abs(df["balance_delta_orig"] + df["amount"]) > 1
    ).astype(int)
    df["dest_balance_error"] = (
        np.abs(df["balance_delta_dest"] - df["amount"]) > 1
    ).astype(int)

    # Hour of day proxy from step (each step = 1 hour in simulation)
    df["hour_of_day"] = df["step"] % 24
    df["day_of_week"] = (df["step"] // 24) % 7

    # Rename isFraud → label
    df = df.rename(columns={"isFraud": "label"})

    # Select numeric features
    feature_cols = [
        "step", "type_enc", "amount",
        "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest",
        "balance_delta_orig", "balance_delta_dest",
        "amount_to_balance_ratio",
        "orig_balance_error", "dest_balance_error",
        "hour_of_day", "day_of_week",
        "isFlaggedFraud",
        "label",
    ]

    out_df = df[feature_cols].copy()
    out_df = out_df.replace([np.inf, -np.inf], 0).fillna(0)

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "paysim_fraud.csv"
    out_df.to_csv(out, index=False)
    logger.info(f"  → {out} ({len(out_df)} rows, fraud={int(out_df['label'].sum())}, ratio={out_df['label'].mean():.4%})")
    return True


def main():
    parser = argparse.ArgumentParser(description="Preprocess finance fraud datasets")
    parser.add_argument("--raw-dir", type=str, required=True,
                        help="Path to fraud-detection/raw/ directory")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Path to processed/ output directory")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)

    ok1 = process_creditcard(raw_dir, output_dir)
    ok2 = process_paysim(raw_dir, output_dir)

    if ok1 or ok2:
        logger.info("Finance fraud preprocessing complete.")
    else:
        logger.error("No datasets found. Check --raw-dir path.")


if __name__ == "__main__":
    main()
