#!/usr/bin/env python3
"""
Cybersecurity Stack Orchestrator
================================
Runs preprocessing and the remaining cybersecurity training pipelines in one pass.

By default this executes:
  - cybersecurity preprocessing
  - phishing detection training
  - ransomware detection training
  - threat detection training

Anomaly detection is optional because that pipeline is already handled separately.
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _run_command(command: list[str], label: str) -> None:
    logger.info("%s", "=" * 72)
    logger.info("Running %s", label)
    logger.info("%s", "=" * 72)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the cybersecurity training stack")
    parser.add_argument("--datasets-dir", type=str, required=True,
                        help="Path to Project_Datasets root directory")
    parser.add_argument("--processed-dir", type=str, default=None,
                        help="Directory for processed cybersecurity datasets")
    parser.add_argument("--skip-preprocess", action="store_true",
                        help="Skip preprocessing and use existing processed datasets")
    parser.add_argument("--skip-ransomware", action="store_true",
                        help="Skip ransomware preprocessing and training")
    parser.add_argument("--ember-limit", type=int, default=0,
                        help="Limit EMBER records per file during preprocessing")
    parser.add_argument("--include-anomaly", action="store_true",
                        help="Also run anomaly detection training")
    parser.add_argument("--require-cuda", action="store_true",
                        help="Pass --require-cuda to training pipelines")
    parser.add_argument("--mlflow-uri", type=str, default=None,
                        help="Optional MLflow tracking URI for all pipelines")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    datasets_dir = Path(args.datasets_dir)
    processed_dir = Path(args.processed_dir) if args.processed_dir else repo_root / "artifacts" / "cybersecurity" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    preprocessing_script = repo_root / "ai" / "training" / "preprocessing" / "preprocess_all.py"
    pipeline_dir = repo_root / "ai" / "training" / "pipelines"
    model_dir = repo_root / "ai" / "models"

    results: list[tuple[str, str]] = []

    if not args.skip_preprocess:
        preprocess_command = [
            sys.executable,
            str(preprocessing_script),
            "--datasets-dir",
            str(datasets_dir),
            "--output-dir",
            str(processed_dir),
            "--only",
            "cybersecurity",
        ]
        if args.skip_ransomware:
            preprocess_command.append("--skip-ransomware")
        if args.ember_limit:
            preprocess_command.extend(["--ember-limit", str(args.ember_limit)])

        try:
            _run_command(preprocess_command, "cybersecurity preprocessing")
            results.append(("preprocessing", "OK"))
        except subprocess.CalledProcessError as exc:
            results.append(("preprocessing", f"ERROR ({exc.returncode})"))
            logger.error("Preprocessing failed with exit code %s", exc.returncode)
            raise

    pipeline_specs = [
        {
            "name": "phishing",
            "script": pipeline_dir / "phishing_detection.py",
            "config": model_dir / "phishing_detection" / "config.yaml",
            "data": processed_dir / "cybersecurity" / "phishing",
            "output": model_dir / "phishing_detection" / "artifacts",
            "dataset": "all",
        },
        {
            "name": "ransomware",
            "script": pipeline_dir / "ransomware_detection.py",
            "config": model_dir / "ransomware_detection" / "config.yaml",
            "data": processed_dir / "cybersecurity" / "ransomware",
            "output": model_dir / "ransomware_detection" / "artifacts",
        },
        {
            "name": "threat_detection",
            "script": pipeline_dir / "threat_detection.py",
            "config": model_dir / "threat_detection" / "config.yaml",
            "data": processed_dir / "cybersecurity" / "threat_detection",
            "output": model_dir / "threat_detection" / "artifacts",
            "dataset": "ctu13",
        },
    ]

    if args.include_anomaly:
        pipeline_specs.insert(
            0,
            {
                "name": "anomaly_detection",
                "script": pipeline_dir / "anomaly_detection.py",
                "config": model_dir / "anomaly_detection" / "config.yaml",
                "data": processed_dir / "cybersecurity" / "anomaly-detection",
                "output": model_dir / "anomaly_detection" / "artifacts",
            },
        )

    for spec in pipeline_specs:
        if args.skip_ransomware and spec["name"] == "ransomware":
            results.append((spec["name"], "SKIPPED"))
            continue

        if not spec["script"].exists():
            results.append((spec["name"], "MISSING_SCRIPT"))
            logger.warning("Skipping %s because %s is missing", spec["name"], spec["script"])
            continue

        if not spec["config"].exists():
            results.append((spec["name"], "MISSING_CONFIG"))
            logger.warning("Skipping %s because %s is missing", spec["name"], spec["config"])
            continue

        if not spec["data"].exists():
            results.append((spec["name"], "MISSING_DATA"))
            logger.warning("Skipping %s because %s is missing", spec["name"], spec["data"])
            continue

        command = [
            sys.executable,
            str(spec["script"]),
            "--config",
            str(spec["config"]),
            "--data",
            str(spec["data"]),
            "--output",
            str(spec["output"]),
        ]
        if "dataset" in spec:
            command.extend(["--dataset", spec["dataset"]])
        if args.require_cuda:
            command.append("--require-cuda")
        if args.mlflow_uri:
            command.extend(["--mlflow-uri", args.mlflow_uri])

        try:
            _run_command(command, f"{spec['name']} training")
            results.append((spec["name"], "OK"))
        except subprocess.CalledProcessError as exc:
            results.append((spec["name"], f"ERROR ({exc.returncode})"))
            logger.error("%s training failed with exit code %s", spec["name"], exc.returncode)

    logger.info("%s", "=" * 72)
    logger.info("Cybersecurity stack summary")
    logger.info("%s", "=" * 72)
    for name, status in results:
        logger.info("%s: %s", name, status)

    if any(status.startswith("ERROR") for _, status in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()