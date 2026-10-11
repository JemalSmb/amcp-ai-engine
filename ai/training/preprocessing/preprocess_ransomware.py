"""
Ransomware Detection — Raw Data Preprocessor

Converts EMBER JSONL (PE feature vectors) into sequences for the BiLSTM pipeline.

Input sources (from Project_Datasets/cybersecurity/ransomware/raw/):
  - Ember/ember_dataset_2018_2/ember2018/train_features_*.jsonl
  - Ember/ember_dataset_2018_2/ember2018/test_features.jsonl

EMBER provides per-sample PE feature records with:
  - histogram (256 byte counts)
  - byteentropy (256 entropy bins)
  - strings (numstrings, avlength, printabledist[96], printables, entropy)
  - general (size, vsize, has_debug, exports, imports, has_relocations, ...)
  - header.coff / header.optional subsections
  - section info, imports, exports, datadirectories

Strategy:
  We flatten the most discriminative features into a fixed 32-dim vector
  (matching the ransomware detector's input_dim=32), then package sliding
  windows of 64 samples as sequences → .npy for the BiLSTM pipeline.

Output:
  → processed/sequences.npy   (N, 64, 32)
  → processed/labels.npy      (N,)
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Target feature dimension must match ransomware config input_dim
FEATURE_DIM = 32
SEQ_LENGTH = 64


def _flatten_ember_record(record: dict) -> Tuple[np.ndarray, int]:
    """Extract a fixed 32-dim feature vector from one EMBER JSONL record."""
    label = record.get("label", -1)  # -1 = unlabeled
    feats = []

    # General features (8)
    gen = record.get("general", {})
    feats.append(gen.get("size", 0))
    feats.append(gen.get("vsize", 0))
    feats.append(gen.get("has_debug", 0))
    feats.append(gen.get("exports", 0))
    feats.append(gen.get("imports", 0))
    feats.append(gen.get("has_relocations", 0))
    feats.append(gen.get("has_resources", 0))
    feats.append(gen.get("has_signature", 0))

    # String features (5)
    strings = record.get("strings", {})
    feats.append(strings.get("numstrings", 0))
    feats.append(strings.get("avlength", 0))
    feats.append(strings.get("printables", 0))
    feats.append(strings.get("entropy", 0))
    feats.append(len(strings.get("printabledist", [])))

    # Byte histogram summary (4): mean, std, max, entropy-of-histogram
    hist = record.get("histogram", [0] * 256)
    hist = np.array(hist, dtype=np.float32)
    if hist.sum() > 0:
        hist_norm = hist / hist.sum()
        hist_ent = -np.sum(hist_norm[hist_norm > 0] * np.log2(hist_norm[hist_norm > 0]))
    else:
        hist_ent = 0.0
    feats.append(float(hist.mean()))
    feats.append(float(hist.std()))
    feats.append(float(hist.max()))
    feats.append(float(hist_ent))

    # Byte entropy summary (4)
    bent = record.get("byteentropy", [0] * 256)
    bent = np.array(bent, dtype=np.float32)
    feats.append(float(bent.mean()))
    feats.append(float(bent.std()))
    feats.append(float(bent.max()))
    feats.append(float(np.median(bent)))

    # Section info (4)
    sec = record.get("section", {})
    sections = sec.get("sections", [])
    feats.append(len(sections))
    if sections:
        sizes = [s.get("size", 0) for s in sections]
        entropies = [s.get("entropy", 0) for s in sections]
        feats.append(float(np.mean(sizes)))
        feats.append(float(np.mean(entropies)))
        feats.append(float(np.max(entropies)))
    else:
        feats.extend([0.0, 0.0, 0.0])

    # Data directories summary (3)
    dd = record.get("datadirectories", [])
    dd_sizes = [d.get("size", 0) if isinstance(d, dict) else 0 for d in dd]
    feats.append(len(dd))
    feats.append(float(np.sum(dd_sizes)) if dd_sizes else 0.0)
    feats.append(float(np.max(dd_sizes)) if dd_sizes else 0.0)

    # Import diversity (2)
    imports = record.get("imports", {})
    feats.append(len(imports))  # number of DLLs imported
    feats.append(sum(len(v) if isinstance(v, list) else 0 for v in imports.values()))

    # Exports (1)
    feats.append(len(record.get("exports", [])))

    # Header features (1)
    hdr = record.get("header", {})
    opt = hdr.get("optional", {})
    feats.append(opt.get("sizeof_code", 0))

    # Pad/truncate to FEATURE_DIM
    feats = feats[:FEATURE_DIM]
    while len(feats) < FEATURE_DIM:
        feats.append(0.0)

    return np.array(feats, dtype=np.float32), label


def _read_jsonl_features(jsonl_path: Path, max_records: int = 0) -> Tuple[List[np.ndarray], List[int]]:
    """Read EMBER JSONL file, return feature vectors and labels."""
    features, labels = [], []
    with open(jsonl_path, "r") as f:
        for i, line in enumerate(f):
            if max_records and i >= max_records:
                break
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            feat, label = _flatten_ember_record(record)
            if label in (0, 1):  # skip unlabeled (-1)
                features.append(feat)
                labels.append(label)
    return features, labels


def process_ember(raw_dir: Path, output_dir: Path, max_per_file: int = 0) -> bool:
    """
    Process EMBER JSONL → sequences.npy + labels.npy.

    Groups consecutive same-label samples into windows of SEQ_LENGTH
    to form sequences for the BiLSTM.
    """
    # Find the ember2018 directory (preferred) or fall back to any ember directory
    ember_dirs = [
        raw_dir / "Ember" / "ember_dataset_2018_2" / "ember2018",
        raw_dir / "Ember" / "ember_dataset_2017_2" / "ember_2017_2",
        raw_dir / "Ember" / "ember_dataset" / "ember",
    ]

    ember_dir = None
    for d in ember_dirs:
        if d.exists():
            ember_dir = d
            break

    if ember_dir is None:
        logger.warning("No EMBER dataset directory found")
        return False

    logger.info(f"Processing EMBER from {ember_dir}")

    # Collect features from train files
    all_features, all_labels = [], []
    train_files = sorted(ember_dir.glob("train_features_*.jsonl"))
    test_files = sorted(ember_dir.glob("test_features*.jsonl"))

    for jf in train_files + test_files:
        logger.info(f"  Reading {jf.name}...")
        feats, labs = _read_jsonl_features(jf, max_records=max_per_file)
        all_features.extend(feats)
        all_labels.extend(labs)
        logger.info(f"    → {len(feats)} labeled samples (benign={labs.count(0)}, malware={labs.count(1)})")

    if not all_features:
        logger.warning("No labeled EMBER samples found")
        return False

    features = np.array(all_features, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int32)

    logger.info(f"  Total: {len(features)} samples (benign={int((labels==0).sum())}, malware={int((labels==1).sum())})")

    # Normalise features per-column
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-8
    features = (features - mean) / std

    # Build sequences: sliding window over each class, then merge
    sequences, seq_labels = [], []

    for cls in [0, 1]:
        cls_feats = features[labels == cls]
        n_seqs = len(cls_feats) // SEQ_LENGTH
        for i in range(n_seqs):
            seq = cls_feats[i * SEQ_LENGTH : (i + 1) * SEQ_LENGTH]
            sequences.append(seq)
            seq_labels.append(cls)

    sequences = np.array(sequences, dtype=np.float32)
    seq_labels = np.array(seq_labels, dtype=np.int32)

    # Shuffle
    idx = np.random.RandomState(42).permutation(len(sequences))
    sequences = sequences[idx]
    seq_labels = seq_labels[idx]

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "sequences.npy", sequences)
    np.save(output_dir / "labels.npy", seq_labels)
    logger.info(f"  → sequences.npy {sequences.shape}, labels.npy {seq_labels.shape}")
    logger.info(f"    benign seqs={int((seq_labels==0).sum())}, malware seqs={int((seq_labels==1).sum())}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Preprocess EMBER ransomware dataset")
    parser.add_argument("--raw-dir", type=str, required=True,
                        help="Path to ransomware/raw/ directory")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Path to processed/ output directory")
    parser.add_argument("--max-per-file", type=int, default=0,
                        help="Max records per JSONL file (0 = all, use for testing)")
    args = parser.parse_args()

    if process_ember(Path(args.raw_dir), Path(args.output_dir), args.max_per_file):
        logger.info("Ransomware preprocessing complete.")
    else:
        logger.error("Failed to process EMBER data. Check --raw-dir path.")


if __name__ == "__main__":
    main()
