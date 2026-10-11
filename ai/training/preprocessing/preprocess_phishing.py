"""
Phishing Detection — Raw Data Preprocessor

Converts phishing datasets into a unified CSV with numeric features + 'label'.

Input sources (from Project_Datasets/cybersecurity/phishing/raw/):
  - Phishing_Legitimate_full.csv   (48 features + CLASS_LABEL)
  - urlhaus.abuse.ch               (comment-headed CSV of malicious URLs)
  - tranco_6GPNX.csv               (rank, domain — legitimate URLs)

The Kaggle Phishing_Legitimate_full.csv is already feature-engineered.
URLhaus + Tranco are raw URL lists — we extract URL-level features
and combine them as an additional dataset.

Output:
  → processed/phishing_kaggle.csv      (from Phishing_Legitimate_full.csv)
  → processed/phishing_urls.csv        (from URLhaus + Tranco)
"""

import argparse
import logging
import math
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ── Kaggle Phishing_Legitimate_full.csv 

def process_kaggle_phishing(raw_dir: Path, output_dir: Path) -> bool:
    """Rename CLASS_LABEL → label, drop id column."""
    csv_path = raw_dir / "Phishing_Legitimate_full.csv"
    if not csv_path.exists():
        logger.warning(f"Phishing_Legitimate_full.csv not found at {csv_path}")
        return False

    logger.info(f"Processing Kaggle phishing dataset from {csv_path}")
    df = pd.read_csv(csv_path)

    # Rename label column
    if "CLASS_LABEL" in df.columns:
        df = df.rename(columns={"CLASS_LABEL": "label"})

    # Drop id column if present
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "phishing_kaggle.csv"
    df.to_csv(out, index=False)
    logger.info(f"  → {out} ({len(df)} rows, phishing ratio: {df['label'].mean():.2%})")
    return True


# ── URL Feature Extraction

def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    freq = Counter(s)
    length = len(s)
    return -sum((c / length) * math.log2(c / length) for c in freq.values())


def extract_url_features(url: str) -> dict:
    """Extract 20 numeric features from a raw URL string."""
    try:
        parsed = urlparse(url if "://" in url else f"http://{url}")
    except Exception:
        return {}

    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""

    # Check if hostname is an IP address
    ip_pattern = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
    has_ip = int(bool(ip_pattern.match(hostname)))

    return {
        "url_length": len(url),
        "hostname_length": len(hostname),
        "path_length": len(path),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_underscores": url.count("_"),
        "num_slashes": url.count("/"),
        "num_query_params": len(parse_qs(query)),
        "num_digits_in_url": sum(c.isdigit() for c in url),
        "has_ip_address": has_ip,
        "has_at_symbol": int("@" in url),
        "has_double_slash_redirect": int("//" in path),
        "has_https": int(parsed.scheme == "https"),
        "num_subdomains": max(0, hostname.count(".") - 1),
        "url_entropy": round(_shannon_entropy(url), 4),
        "path_entropy": round(_shannon_entropy(path), 4),
        "hostname_entropy": round(_shannon_entropy(hostname), 4),
        "digit_ratio": round(sum(c.isdigit() for c in url) / max(len(url), 1), 4),
        "special_char_ratio": round(
            sum(not c.isalnum() for c in url) / max(len(url), 1), 4
        ),
        "path_depth": path.strip("/").count("/") + 1 if path.strip("/") else 0,
    }


def process_url_lists(raw_dir: Path, output_dir: Path) -> bool:
    """Combine URLhaus (malicious) + Tranco (legitimate) → feature CSV."""
    urlhaus_path = raw_dir / "urlhaus.abuse.ch"
    tranco_path = None
    for f in raw_dir.glob("tranco_*.csv"):
        tranco_path = f
        break

    if not urlhaus_path.exists() or tranco_path is None:
        if not urlhaus_path.exists():
            logger.warning("urlhaus.abuse.ch not found")
        if tranco_path is None:
            logger.warning("No tranco_*.csv found")
        return False

    logger.info("Processing URLhaus + Tranco → URL features CSV")

    # Parse URLhaus (skip comment lines starting with #)
    mal_urls = []
    with open(urlhaus_path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) >= 3:
                url = parts[2].strip('"')
                if url.startswith("http"):
                    mal_urls.append(url)

    # Parse Tranco (rank, domain)
    legit_urls = []
    tranco_df = pd.read_csv(tranco_path, header=None, names=["rank", "domain"])
    # Sample up to len(mal_urls) * 2 legitimate domains for balance
    sample_n = min(len(tranco_df), max(len(mal_urls) * 2, 5000))
    sampled = tranco_df.sample(n=sample_n, random_state=42)
    legit_urls = [f"https://{d}" for d in sampled["domain"].values]

    logger.info(f"  URLhaus: {len(mal_urls)} malicious URLs")
    logger.info(f"  Tranco:  {len(legit_urls)} legitimate URLs (sampled)")

    # Extract features
    records = []
    for url in mal_urls:
        feats = extract_url_features(url)
        if feats:
            feats["label"] = 1
            records.append(feats)
    for url in legit_urls:
        feats = extract_url_features(url)
        if feats:
            feats["label"] = 0
            records.append(feats)

    df = pd.DataFrame(records)

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "phishing_urls.csv"
    df.to_csv(out, index=False)
    logger.info(f"  → {out} ({len(df)} rows, phishing ratio: {df['label'].mean():.2%})")
    return True


def main():
    parser = argparse.ArgumentParser(description="Preprocess phishing detection datasets")
    parser.add_argument("--raw-dir", type=str, required=True,
                        help="Path to phishing/raw/ directory")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Path to processed/ output directory")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)

    ok1 = process_kaggle_phishing(raw_dir, output_dir)
    ok2 = process_url_lists(raw_dir, output_dir)

    if ok1 or ok2:
        logger.info("Phishing preprocessing complete.")
    else:
        logger.error("No phishing datasets found. Check --raw-dir path.")


if __name__ == "__main__":
    main()
