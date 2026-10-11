#!/usr/bin/env python3
"""
Master Preprocessing Orchestrator

Runs all dataset preprocessors, reading raw data from Project_Datasets
and writing processed CSVs/npys to the output directory structure.

Usage:
  python preprocess_all.py --datasets-dir /media/.../Project_Datasets --output-dir ./processed

  # Or process only specific domains:
  python preprocess_all.py --datasets-dir /media/.../Project_Datasets --output-dir ./processed --only cybersecurity

  # Quick test with limited EMBER records:
  python preprocess_all.py --datasets-dir /media/.../Project_Datasets --output-dir ./processed --ember-limit 1000
"""

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Add parent to path so preprocessing modules can be imported
sys.path.insert(0, str(Path(__file__).parent))


def run_anomaly_detection(datasets_dir: Path, output_dir: Path):
    from preprocess_anomaly_detection import process_nsl_kdd, process_unsw_nb15

    raw = datasets_dir / "cybersecurity" / "anomaly-detection" / "raw"
    out = output_dir / "cybersecurity" / "anomaly-detection"

    ok1 = process_nsl_kdd(raw, out)
    ok2 = process_unsw_nb15(raw, out)
    return ok1 or ok2


def run_phishing(datasets_dir: Path, output_dir: Path):
    from preprocess_phishing import process_kaggle_phishing, process_url_lists

    raw = datasets_dir / "cybersecurity" / "phishing" / "raw"
    out = output_dir / "cybersecurity" / "phishing"

    ok1 = process_kaggle_phishing(raw, out)
    ok2 = process_url_lists(raw, out)
    return ok1 or ok2


def run_ransomware(datasets_dir: Path, output_dir: Path, ember_limit: int = 0):
    from preprocess_ransomware import process_ember

    raw = datasets_dir / "cybersecurity" / "ransomware" / "raw"
    out = output_dir / "cybersecurity" / "ransomware"
    return process_ember(raw, out, max_per_file=ember_limit)


def run_threat_detection(datasets_dir: Path, output_dir: Path):
    from preprocess_threat_detection import process_ctu13, process_api_calls

    raw = datasets_dir / "cybersecurity" / "threat_detection" / "raw"
    out = output_dir / "cybersecurity" / "threat-detection"

    ok1 = process_ctu13(raw, out)
    ok2 = process_api_calls(raw, out)
    return ok1 or ok2


def run_finance(datasets_dir: Path, output_dir: Path):
    from preprocess_finance import process_creditcard, process_paysim

    raw = datasets_dir / "finance" / "fraud-detection" / "raw"
    out = output_dir / "finance" / "fraud-detection"

    ok1 = process_creditcard(raw, out)
    ok2 = process_paysim(raw, out)
    return ok1 or ok2


def run_manufacturing(datasets_dir: Path, output_dir: Path):
    from preprocess_manufacturing import process_ai4i, process_cmapss, process_tep

    pm_raw = datasets_dir / "manufacturing" / "predictive-maintenance" / "raw"
    pa_raw = datasets_dir / "manufacturing" / "production-anomaly" / "raw"
    out = output_dir / "manufacturing"

    ok1 = process_ai4i(pm_raw, out / "process-anomaly")
    ok2 = process_cmapss(pa_raw, out / "predictive-maintenance", seq_length=30, forecast_horizon=5)
    ok3 = process_tep(pa_raw, out / "process-anomaly")
    return ok1 or ok2 or ok3


def run_agriculture(datasets_dir: Path, output_dir: Path):
    from preprocess_agriculture import (
        process_vineyard_lidar,
        process_yield_prediction,
        process_path_planning,
        process_field_anomaly,
    )

    raw = datasets_dir / "agriculture" / "raw"
    out = output_dir / "agriculture"

    ok1 = process_vineyard_lidar(raw, out / "vineyard-lidar")
    ok2 = process_yield_prediction(raw, out / "yield-prediction")
    ok3 = process_path_planning(raw, out / "path-planning")
    ok4 = process_field_anomaly(raw, out / "anomaly-detection")
    return ok1 or ok2 or ok3 or ok4


PIPELINES = {
    "cybersecurity": [
        ("anomaly_detection", run_anomaly_detection),
        ("phishing", run_phishing),
        ("threat_detection", run_threat_detection),
    ],
    "finance": [
        ("fraud", run_finance),
    ],
    "manufacturing": [
        ("manufacturing", run_manufacturing),
    ],
    "agriculture": [
        ("agriculture", run_agriculture),
    ],
}

# Ransomware handled separately due to extra arg
RANSOMWARE = ("ransomware", run_ransomware)


def main():
    parser = argparse.ArgumentParser(description="Run all dataset preprocessing pipelines")
    parser.add_argument("--datasets-dir", type=str, required=True,
                        help="Path to Project_Datasets root directory")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Path to write processed output (creates subdirectories)")
    parser.add_argument("--only", type=str, default=None,
                        choices=["cybersecurity", "finance", "manufacturing", "agriculture"],
                        help="Process only one domain")
    parser.add_argument("--skip-ransomware", action="store_true",
                        help="Skip EMBER ransomware processing (very large)")
    parser.add_argument("--ember-limit", type=int, default=0,
                        help="Limit EMBER records per file (0 = all)")
    args = parser.parse_args()

    datasets_dir = Path(args.datasets_dir)
    output_dir = Path(args.output_dir)

    if not datasets_dir.exists():
        logger.error(f"Datasets directory not found: {datasets_dir}")
        sys.exit(1)

    results = {}
    start = time.time()

    # Determine which domains to process
    domains = [args.only] if args.only else list(PIPELINES.keys())

    for domain in domains:
        for name, func in PIPELINES.get(domain, []):
            logger.info(f"{'='*60}")
            logger.info(f"Processing: {domain}/{name}")
            logger.info(f"{'='*60}")
            try:
                ok = func(datasets_dir, output_dir)
                results[f"{domain}/{name}"] = "OK" if ok else "NO DATA"
            except Exception as e:
                logger.error(f"FAILED: {e}", exc_info=True)
                results[f"{domain}/{name}"] = f"ERROR: {e}"

    # Ransomware (separate due to ember_limit arg)
    if not args.skip_ransomware and (args.only is None or args.only == "cybersecurity"):
        name, func = RANSOMWARE
        logger.info(f"{'='*60}")
        logger.info(f"Processing: cybersecurity/{name}")
        logger.info(f"{'='*60}")
        try:
            ok = func(datasets_dir, output_dir, args.ember_limit)
            results[f"cybersecurity/{name}"] = "OK" if ok else "NO DATA"
        except Exception as e:
            logger.error(f"FAILED: {e}", exc_info=True)
            results[f"cybersecurity/{name}"] = f"ERROR: {e}"

    elapsed = time.time() - start

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"PREPROCESSING SUMMARY ({elapsed:.1f}s)")
    logger.info(f"{'='*60}")
    for pipeline, status in results.items():
        icon = "✓" if status == "OK" else ("○" if status == "NO DATA" else "✗")
        logger.info(f"  {icon} {pipeline}: {status}")

    # List output files
    if output_dir.exists():
        logger.info(f"\nOutput files:")
        for f in sorted(output_dir.rglob("*")):
            if f.is_file():
                size_mb = f.stat().st_size / (1024 * 1024)
                logger.info(f"  {f.relative_to(output_dir)} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
