#!/usr/bin/env python3
"""
Build Augmented Dataset for Signformer Training.

This script generalizes build_phoenix14t_ext.py to work with any SLT dataset.
It takes augmented annotations (CSV/TSV) and creates the .pami0.* files
needed for Signformer training.

The script:
1. Loads original .pami0.* pickle files
2. Matches samples with augmented text variants
3. Creates new pickle files with augmented samples
4. Preserves original pose data while expanding text variants

Usage:
    python scripts/build_augmented_dataset.py \\
        --dataset RWTH_PHOENIX_2014T \\
        --data_dir /mnt/disk3Tb/exported-slt-datasets \\
        --aug_dir /mnt/disk3Tb/augmented-slt-datasets \\
        --out_prefix RWTH_PHOENIX_2014T-aug.pami0
"""

import argparse
import gzip
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("build_augmented_dataset")


# Dataset configurations
DATASET_CONFIGS = {
    "RWTH_PHOENIX_2014T": {
        "file_prefix": "RWTH_PHOENIX_2014T.pami0",
        "aug_file": "train_aug.tsv",
        "id_column": "id",
    },
    "phoenix14t": {
        "file_prefix": "phoenix14t.pami0",
        "aug_file": "train_aug.tsv",
        "id_column": "id",
    },
    "lsat": {
        "file_prefix": "LSAT.pami0",
        "aug_file": "train_aug.tsv",
        "id_column": "id",
    },
    "LSA-T": {
        "file_prefix": "LSAT.pami0",
        "aug_file": "train_aug.tsv",
        "id_column": "id",
    },
    "How2Sign": {
        "file_prefix": "How2Sign.pami0",
        "aug_file": "train_aug.tsv",
        "id_column": "id",
    },
    "ISL": {
        "file_prefix": "ISL.pami0",
        "aug_file": "train_aug.tsv",
        "id_column": "id",
    },
    "LSFB-CONT": {
        "file_prefix": "LSFB-CONT.pami0",
        "aug_file": "train_aug.tsv",
        "id_column": "id",
    },
    "GSL": {
        "file_prefix": "GSL.pami0",
        "aug_file": "train_aug.tsv",
        "id_column": "id",
    },
}


def load_pickle_list(path: str) -> List[dict]:
    """Load gzip-pickled list of samples."""
    logger.debug(f"Loading pickle: {path}")
    with gzip.open(path, "rb") as f:
        obj = pickle.load(f)
        return obj if isinstance(obj, list) else [obj]


def write_pickle_list(path: str, samples: List[dict]) -> None:
    """Write samples to gzip-pickled file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    logger.debug(f"Writing pickle: {path} ({len(samples)} samples)")
    with gzip.open(path, "wb") as f:
        pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)


def build_indices(
    df: pd.DataFrame,
    id_column: str = "id",
) -> Tuple[
    Dict[str, pd.core.groupby.generic.DataFrameGroupBy],
    Dict[str, pd.DataFrame]
]:
    """
    Build lookup indices for efficient sample matching.
    
    Returns:
        Tuple of (by_split_id, by_split_text) indices
    """
    df = df.copy()
    df[id_column] = df[id_column].astype(str)
    
    # Ensure split column exists
    if "split" not in df.columns:
        df["split"] = "train"
    df["split"] = df["split"].astype(str)
    
    # Normalize text for fallback matching
    df["text_norm"] = df["text"].astype(str).str.strip().str.lower()
    
    # Handle optional columns
    if "gloss" in df.columns:
        df["gloss"] = df["gloss"].astype(str)
    if "signer" in df.columns:
        df["signer"] = df["signer"].astype(str)
    
    # Build indices by split
    by_split = {s: g for s, g in df.groupby("split")}
    by_split_id = {s: g.groupby(id_column) for s, g in by_split.items()}
    by_split_text = {s: g.set_index("text_norm") for s, g in by_split.items()}
    
    return by_split_id, by_split_text


def extract_sample_id(sample: dict) -> str:
    """Extract ID from sample, handling different naming conventions."""
    name = sample.get("name", "")
    
    # Handle "split/id" format
    if "/" in name:
        _, id_name = name.split("/", 1)
        return id_name
    
    return name


def make_augmented_for_split(
    split_name: str,
    in_path: str,
    out_path: str,
    by_split_id: Dict[str, pd.core.groupby.generic.DataFrameGroupBy],
    by_split_text: Dict[str, pd.DataFrame],
    id_column: str = "id",
    preserve_original_on_no_match: bool = True,
) -> Tuple[int, int, int, int, int]:
    """
    Create augmented dataset for a single split.
    
    Args:
        split_name: Name of the split (train, dev, test)
        in_path: Path to input pickle file
        out_path: Path to output pickle file
        by_split_id: ID-based lookup index
        by_split_text: Text-based lookup index
        id_column: Column name for sample IDs
        preserve_original_on_no_match: Keep original if no augmentation found
    
    Returns:
        Tuple of (n_in, n_out, n_id_match, n_text_fallback, n_no_match)
    """
    if not os.path.exists(in_path):
        logger.warning(f"Input file not found: {in_path}")
        return 0, 0, 0, 0, 0
    
    base_samples = load_pickle_list(in_path)
    out_samples: List[dict] = []
    
    grp_by_id = by_split_id.get(split_name)
    text_index = by_split_text.get(split_name)
    
    n_in = len(base_samples)
    n_id_match = 0
    n_text_fallback = 0
    n_no_match = 0
    
    for sample in base_samples:
        id_name = extract_sample_id(sample)
        
        rows = None
        
        # Try ID-based matching first
        if grp_by_id is not None and id_name in grp_by_id.groups:
            rows = grp_by_id.get_group(id_name)
            n_id_match += 1
        else:
            # Fallback to text-based matching
            txt = str(sample.get("text", "")).strip().lower()
            if text_index is not None and txt in text_index.index:
                loc = text_index.loc[txt]
                rows = loc.to_frame().T if isinstance(loc, pd.Series) else loc
                n_text_fallback += 1
            else:
                # No match found
                if preserve_original_on_no_match:
                    out_samples.append(sample)
                n_no_match += 1
                continue
        
        if isinstance(rows, pd.Series):
            rows = rows.to_frame().T
        
        # Create augmented samples
        for _, r in rows.iterrows():
            dup = dict(sample)
            dup["text"] = str(r["text"]).strip()
            
            # Update optional fields if present in augmentation data
            if "gloss" in r and pd.notna(r["gloss"]):
                dup["gloss"] = str(r["gloss"]).strip()
            if "signer" in r and pd.notna(r["signer"]):
                dup["signer"] = str(r["signer"]).strip()
            
            # Track augmentation source
            if "augmentation_pivot" in r:
                dup["augmentation_pivot"] = str(r["augmentation_pivot"])
            if "augmentation_method" in r:
                dup["augmentation_method"] = str(r["augmentation_method"])
            
            out_samples.append(dup)
    
    write_pickle_list(out_path, out_samples)
    
    logger.info(
        f"{split_name}: in={n_in:,} out={len(out_samples):,} "
        f"id_matches={n_id_match:,} text_fallbacks={n_text_fallback:,} "
        f"no_match={n_no_match:,}"
    )
    
    return n_in, len(out_samples), n_id_match, n_text_fallback, n_no_match


def build_augmented_dataset(
    dataset: str,
    data_dir: str,
    aug_dir: str,
    out_prefix: str,
    out_dir: Optional[str] = None,
    aug_file: Optional[str] = None,
    splits: Optional[List[str]] = None,
) -> Dict[str, Tuple[int, int]]:
    """
    Build augmented dataset files for Signformer.
    
    Args:
        dataset: Dataset identifier
        data_dir: Directory containing original .pami0.* files
        aug_dir: Directory containing augmented annotations
        out_prefix: Prefix for output files
        out_dir: Output directory (defaults to data_dir)
        aug_file: Augmentation file name (auto-detected if None)
        splits: Splits to process (defaults to train, val, test)
    
    Returns:
        Dict mapping split names to (in_count, out_count) tuples
    """
    # Get dataset configuration
    config = DATASET_CONFIGS.get(dataset, {})
    file_prefix = config.get("file_prefix", f"{dataset}.pami0")
    default_aug_file = config.get("aug_file", "train_aug.tsv")
    id_column = config.get("id_column", "id")
    
    # Setup paths
    aug_file = aug_file or default_aug_file
    out_dir = out_dir or data_dir
    splits = splits or ["train", "val", "test"]
    
    # Find augmentation file
    aug_path = None
    search_paths = [
        os.path.join(aug_dir, dataset, aug_file),
        os.path.join(aug_dir, dataset, "annotations_train_augmented.csv"),
        os.path.join(aug_dir, aug_file),
        os.path.join(data_dir, aug_file),
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            aug_path = path
            break
    
    if aug_path is None:
        raise FileNotFoundError(
            f"Could not find augmentation file. Searched:\n" +
            "\n".join(f"  - {p}" for p in search_paths)
        )
    
    logger.info(f"Using augmentation file: {aug_path}")
    
    # Load augmentation data
    if aug_path.endswith(".tsv"):
        df = pd.read_csv(aug_path, sep="\t")
    else:
        df = pd.read_csv(aug_path)
    
    logger.info(f"Loaded {len(df):,} augmented annotations")
    
    # Build lookup indices
    by_split_id, by_split_text = build_indices(df, id_column)
    
    # Process each split
    results = {}
    total_in = 0
    total_out = 0
    
    for split in splits:
        # Map split names
        split_file = split
        if split == "val":
            # Try both 'val' and 'dev' naming conventions
            split_file = "val" if os.path.exists(
                os.path.join(data_dir, f"{file_prefix}.val")
            ) else "dev"
        
        in_path = os.path.join(data_dir, f"{file_prefix}.{split_file}")
        out_path = os.path.join(out_dir, f"{out_prefix}.{split_file}")
        
        # Also check for .dev files
        if not os.path.exists(in_path) and split == "val":
            in_path = os.path.join(data_dir, f"{file_prefix}.dev")
            out_path = os.path.join(out_dir, f"{out_prefix}.dev")
        
        n_in, n_out, _, _, _ = make_augmented_for_split(
            split_name=split,
            in_path=in_path,
            out_path=out_path,
            by_split_id=by_split_id,
            by_split_text=by_split_text,
            id_column=id_column,
        )
        
        results[split] = (n_in, n_out)
        total_in += n_in
        total_out += n_out
    
    # Print summary
    logger.info("=" * 60)
    logger.info("BUILD SUMMARY")
    logger.info("=" * 60)
    for split, (n_in, n_out) in results.items():
        expansion = n_out / n_in if n_in > 0 else 0
        logger.info(f"  {split}: {n_in:,} → {n_out:,} ({expansion:.1f}x)")
    logger.info("-" * 60)
    expansion = total_out / total_in if total_in > 0 else 0
    logger.info(f"  TOTAL: {total_in:,} → {total_out:,} ({expansion:.1f}x)")
    logger.info("=" * 60)
    
    return results


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build augmented dataset pickles for Signformer training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build augmented RWTH-Phoenix dataset
  python scripts/build_augmented_dataset.py \\
      --dataset RWTH_PHOENIX_2014T \\
      --data_dir /mnt/disk3Tb/exported-slt-datasets \\
      --aug_dir /mnt/disk3Tb/augmented-slt-datasets

  # Build with custom output prefix
  python scripts/build_augmented_dataset.py \\
      --dataset GSL \\
      --out_prefix GSL-aug.pami0

  # Process only training split
  python scripts/build_augmented_dataset.py \\
      --dataset lsat \\
      --splits train
        """
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list(DATASET_CONFIGS.keys()),
        help="Dataset to process"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.getenv("EXPORTED_DATASETS_DIR", "/mnt/disk3Tb/exported-slt-datasets"),
        help="Directory containing original .pami0.* files (default: $EXPORTED_DATASETS_DIR)"
    )
    parser.add_argument(
        "--aug_dir",
        type=str,
        default=os.getenv("AUGMENTED_DATASETS_DIR", "/mnt/disk3Tb/augmented-slt-datasets"),
        help="Directory containing augmented annotations (default: $AUGMENTED_DATASETS_DIR)"
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default=None,
        help="Output file prefix (default: {dataset}-aug.pami0)"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory (default: same as data_dir)"
    )
    parser.add_argument(
        "--aug_file",
        type=str,
        default=None,
        help="Augmentation file name (auto-detected if not specified)"
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "val", "test"],
        help="Splits to process"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set default output prefix
    out_prefix = args.out_prefix or f"{args.dataset}-aug.pami0"
    
    logger.info(f"Building augmented dataset: {args.dataset}")
    logger.info(f"  Data directory: {args.data_dir}")
    logger.info(f"  Augmentation directory: {args.aug_dir}")
    logger.info(f"  Output prefix: {out_prefix}")
    logger.info(f"  Splits: {args.splits}")
    
    try:
        build_augmented_dataset(
            dataset=args.dataset,
            data_dir=args.data_dir,
            aug_dir=args.aug_dir,
            out_prefix=out_prefix,
            out_dir=args.out_dir,
            aug_file=args.aug_file,
            splits=args.splits,
        )
        logger.info("Build completed successfully!")
        
    except Exception as e:
        logger.error(f"Build failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
