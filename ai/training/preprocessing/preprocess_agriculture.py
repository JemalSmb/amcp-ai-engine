"""
Agriculture — Raw Data Preprocessor

Prepares agriculture datasets for four model tasks:
  1) Vineyard LiDAR perception
  2) Yield prediction
  3) Path planning
  4) Field anomaly detection

Expected raw layout (flexible, first matching file is used):
  agriculture/raw/
    vineyard-lidar/      (*.csv, *.parquet, optional *.las/*.laz)
    yield-prediction/    (*.csv, *.parquet)
    path-planning/       (*.csv, *.parquet)
    anomaly-detection/   (*.csv, *.parquet)

Outputs:
  agriculture/<task>/*.csv
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _find_first_table(folder: Path) -> Optional[Path]:
    if not folder.exists():
        return None
    for pattern in ("*.csv", "*.parquet"):
        files = sorted(folder.glob(pattern))
        if files:
            return files[0]
    return None


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def _save(df: pd.DataFrame, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / name
    df.to_csv(out, index=False)
    return out


def process_vineyard_lidar(raw_dir: Path, output_dir: Path) -> bool:
    """
    Prepare LiDAR/point-cloud derived table for modeling.

    If CSV/Parquet exists, keeps numeric fields, sanitizes inf/nan, and creates
    a binary label when a likely anomaly/class column is present.
    """
    base = raw_dir / "vineyard-lidar"
    table = _find_first_table(base)

    if table is None:
        # Optional support for LAS/LAZ if laspy is installed.
        las_files = sorted([*base.glob("*.las"), *base.glob("*.laz")]) if base.exists() else []
        if las_files:
            try:
                import laspy  # type: ignore
            except ImportError:
                logger.warning("LiDAR LAS/LAZ found but laspy is not installed. Skipping vineyard-lidar.")
                return False

            logger.info(f"Processing LiDAR LAS file: {las_files[0]}")
            las = laspy.read(str(las_files[0]))
            df = pd.DataFrame(
                {
                    "x": np.asarray(las.x),
                    "y": np.asarray(las.y),
                    "z": np.asarray(las.z),
                    "intensity": np.asarray(getattr(las, "intensity", np.zeros_like(las.x))),
                    "return_num": np.asarray(getattr(las, "return_number", np.ones_like(las.x))),
                    "num_returns": np.asarray(getattr(las, "number_of_returns", np.ones_like(las.x))),
                }
            )
            # Simple unsupervised label placeholder: top 1% height as potential anomaly.
            z_thr = df["z"].quantile(0.99)
            df["label"] = (df["z"] >= z_thr).astype(int)
        else:
            logger.warning(f"No vineyard-lidar source table found in {base}")
            return False
    else:
        logger.info(f"Processing vineyard-lidar table: {table}")
        df = _read_table(table)

    # Standard cleaning
    df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]

    label_candidates = ["label", "class", "target", "anomaly", "disease", "status"]
    label_col = next((c for c in label_candidates if c in df.columns), None)

    if label_col is None:
        # fallback pseudo-label from extreme z/intensity for proof-of-concept
        if "z" in df.columns:
            df["label"] = (df["z"] >= df["z"].quantile(0.99)).astype(int)
        elif "intensity" in df.columns:
            df["label"] = (df["intensity"] >= df["intensity"].quantile(0.99)).astype(int)
        else:
            df["label"] = 0
    elif label_col != "label":
        if df[label_col].dtype == object:
            le = LabelEncoder()
            df["label"] = le.fit_transform(df[label_col].astype(str))
        else:
            df["label"] = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)

    numeric = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], 0).fillna(0)
    if "label" not in numeric.columns:
        numeric["label"] = df["label"].astype(int)

    out = _save(numeric, output_dir, "vineyard_lidar.csv")
    logger.info(f"  -> {out} ({len(numeric)} rows)")
    return True


def process_yield_prediction(raw_dir: Path, output_dir: Path) -> bool:
    """Prepare yield data for forecasting/regression tasks."""
    base = raw_dir / "yield-prediction"
    table = _find_first_table(base)
    if table is None:
        logger.warning(f"No yield-prediction source table found in {base}")
        return False

    logger.info(f"Processing yield-prediction table: {table}")
    df = _read_table(table)
    df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]

    target_candidates = ["yield", "yield_t_ha", "yield_kg_ha", "target"]
    target_col = next((c for c in target_candidates if c in df.columns), None)

    # Encode low-cardinality categoricals (e.g., cultivar/block)
    for c in df.select_dtypes(include=["object"]).columns:
        if c in ("date", "timestamp"):
            continue
        if df[c].nunique(dropna=True) <= 200:
            le = LabelEncoder()
            df[c] = le.fit_transform(df[c].astype(str))

    if target_col and target_col != "target":
        df = df.rename(columns={target_col: "target"})
    elif "target" not in df.columns:
        logger.warning("Yield table has no explicit target column; creating placeholder target=0")
        df["target"] = 0.0

    # Date features if available
    if "date" in df.columns:
        dtv = pd.to_datetime(df["date"], errors="coerce")
        df["year"] = dtv.dt.year.fillna(0).astype(int)
        df["month"] = dtv.dt.month.fillna(0).astype(int)
        df["dayofyear"] = dtv.dt.dayofyear.fillna(0).astype(int)

    out_df = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], 0).fillna(0)
    out = _save(out_df, output_dir, "yield_prediction.csv")
    logger.info(f"  -> {out} ({len(out_df)} rows)")
    return True


def process_path_planning(raw_dir: Path, output_dir: Path) -> bool:
    """
    Prepare path-planning trajectories.

    Expected fields (any subset): x,y,heading,speed,curvature,obstacle_distance,reward,action
    """
    base = raw_dir / "path-planning"
    table = _find_first_table(base)
    if table is None:
        logger.warning(f"No path-planning source table found in {base}")
        return False

    logger.info(f"Processing path-planning table: {table}")
    df = _read_table(table)
    df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]

    for c in df.select_dtypes(include=["object"]).columns:
        if c in ("action", "mode"):
            le = LabelEncoder()
            df[c] = le.fit_transform(df[c].astype(str))

    # RL-friendly fallback columns
    if "reward" not in df.columns:
        # Penalize distance/curvature heuristically if present.
        dist_pen = df.get("obstacle_distance", pd.Series(np.zeros(len(df))))
        curv_pen = df.get("curvature", pd.Series(np.zeros(len(df))))
        df["reward"] = -0.1 * pd.to_numeric(curv_pen, errors="coerce").fillna(0) + 0.01 * pd.to_numeric(dist_pen, errors="coerce").fillna(0)

    out_df = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], 0).fillna(0)
    out = _save(out_df, output_dir, "path_planning.csv")
    logger.info(f"  -> {out} ({len(out_df)} rows)")
    return True


def process_field_anomaly(raw_dir: Path, output_dir: Path) -> bool:
    """Prepare agriculture field anomaly table with binary label."""
    base = raw_dir / "anomaly-detection"
    table = _find_first_table(base)
    if table is None:
        logger.warning(f"No anomaly-detection source table found in {base}")
        return False

    logger.info(f"Processing field-anomaly table: {table}")
    df = _read_table(table)
    df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]

    label_candidates = ["label", "anomaly", "fault", "disease", "target", "status"]
    label_col = next((c for c in label_candidates if c in df.columns), None)

    if label_col is None:
        # fallback label from extreme sensor values (PoC)
        num = df.select_dtypes(include=[np.number])
        if num.shape[1] > 0:
            z = (num - num.mean()) / (num.std() + 1e-8)
            score = np.abs(z).mean(axis=1)
            df["label"] = (score >= np.quantile(score, 0.99)).astype(int)
        else:
            df["label"] = 0
    elif label_col != "label":
        if df[label_col].dtype == object:
            df["label"] = (df[label_col].astype(str).str.lower().isin(["anomaly", "fault", "disease", "1", "true"])).astype(int)
        else:
            df["label"] = (pd.to_numeric(df[label_col], errors="coerce").fillna(0) != 0).astype(int)

    out_df = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], 0).fillna(0)
    if "label" not in out_df.columns:
        out_df["label"] = df["label"].astype(int)

    out = _save(out_df, output_dir, "field_anomaly.csv")
    logger.info(f"  -> {out} ({len(out_df)} rows, anomalies={int(out_df['label'].sum())})")
    return True


def main():
    parser = argparse.ArgumentParser(description="Preprocess agriculture datasets")
    parser.add_argument("--raw-dir", type=str, required=True,
                        help="Path to agriculture/raw directory")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Path to agriculture output directory")
    args = parser.parse_args()

    raw = Path(args.raw_dir)
    out = Path(args.output_dir)

    ok1 = process_vineyard_lidar(raw, out / "vineyard-lidar")
    ok2 = process_yield_prediction(raw, out / "yield-prediction")
    ok3 = process_path_planning(raw, out / "path-planning")
    ok4 = process_field_anomaly(raw, out / "anomaly-detection")

    if ok1 or ok2 or ok3 or ok4:
        logger.info("Agriculture preprocessing complete.")
    else:
        logger.error("No agriculture datasets found. Check raw folder layout.")


if __name__ == "__main__":
    main()
