#!/usr/bin/env python3
"""
Dataset Augmentation CLI with Back-Translation.

This script provides a command-line interface for augmenting SLT datasets
using back-translation. Features:
- Support for all 6 datasets
- Checkpoint/resume capability
- Comprehensive structured logging
- Azure OpenAI rate limiting
- Multi-dataset batch processing

Usage:
    python -m src.augmentation.augment_dataset --dataset RWTH_PHOENIX_2014T
    python -m src.augmentation.augment_dataset --all-datasets --summary-report
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

# Add lib to path for imports
_lib_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "lib")
if _lib_path not in sys.path:
    sys.path.insert(0, _lib_path)

from dotenv import load_dotenv

# Try to import SLTDataset, with graceful fallback to simple CSV loading
try:
    from slt_datasets.SLTDataset import SLTDataset
    HAS_SLT_DATASETS = True
except ImportError as e:
    HAS_SLT_DATASETS = False
    SLTDataset = None
    _slt_import_error = str(e)


class SimpleDatasetLoader:
    """
    Simple dataset loader that reads annotations directly from CSV.
    Used as fallback when SLTDataset is not available (e.g., missing torchvision).
    """
    
    def __init__(self, data_dir: str, split: str = "train"):
        self.data_dir = data_dir
        self.split = split
        
        # Load annotations
        annotations_file = os.path.join(data_dir, "annotations.csv")
        if not os.path.exists(annotations_file):
            raise FileNotFoundError(f"Annotations file not found: {annotations_file}")
        
        self.annotations = pd.read_csv(annotations_file)
        
        # Filter by split
        if "split" in self.annotations.columns:
            self.annotations = self.annotations[self.annotations["split"] == split].reset_index(drop=True)
        
        # Ensure required columns exist
        if "text" not in self.annotations.columns:
            raise ValueError("Annotations must have a 'text' column")
        if "id" not in self.annotations.columns:
            self.annotations["id"] = self.annotations.index.astype(str)
        
        # Filter out rows with NaN or empty text values
        initial_count = len(self.annotations)
        self.annotations = self.annotations[self.annotations["text"].notna()].reset_index(drop=True)
        self.annotations = self.annotations[self.annotations["text"].str.strip() != ""].reset_index(drop=True)
        filtered_count = initial_count - len(self.annotations)
        if filtered_count > 0:
            print(f"Warning: Filtered out {filtered_count} rows with empty/NaN text values")
    
    def __len__(self):
        return len(self.annotations)

from .back_translate import BackTranslator, create_back_translator_for_dataset
from .prompts import DATASET_LANGUAGES, get_source_language, PIVOT_LANGUAGES
from .rate_limiter import AzureRateLimiter

# Load environment variables
load_dotenv()

# Default directories from environment
DEFAULT_SLT_DIR = os.getenv("SLT_DATASETS_DIR", "/mnt/disk3Tb/slt-datasets")
DEFAULT_AUG_DIR = os.getenv("AUGMENTED_DATASETS_DIR", "/mnt/disk3Tb/augmented-slt-datasets")
DEFAULT_EXPORT_DIR = os.getenv("EXPORTED_DATASETS_DIR", "/mnt/disk3Tb/exported-slt-datasets")

# Dataset configurations - paths are relative to SLT_DATASETS_DIR
DATASET_CONFIGS = {
    "RWTH_PHOENIX_2014T": {
        "subdir": "RWTH_PHOENIX_2014T",
        "language": "de",
    },
    "lsat": {
        "subdir": "lsat",
        "language": "es",
    },
    "LSAT": {
        "subdir": "LSAT",
        "language": "es",
    },
    "How2Sign": {
        "subdir": "How2Sign",
        "language": "en",
    },
    "ISL": {
        "subdir": "ISL",
        "language": "en",
    },
    "LSFB-CONT": {
        "subdir": "LSFB-CONT",
        "language": "fr",
    },
    "GSL": {
        "subdir": "GSL",
        "language": "el",
    },
}

# Color codes for console output
YELLOW = '\033[1;33m'
BLUE = '\033[1;34m'
CYAN = '\033[1;36m'
GREEN = '\033[1;32m'
RED = '\033[1;31m'
BOLD = '\033[1m'
NC = '\033[0m'  # No Color

def get_dataset_path(dataset_id: str) -> str:
    """Get full path to dataset directory."""
    config = DATASET_CONFIGS.get(dataset_id, {})
    subdir = config.get("subdir", dataset_id)
    return os.path.join(DEFAULT_SLT_DIR, subdir)


class JSONLogFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        
        # Add extra fields if present
        if hasattr(record, "dataset"):
            log_data["dataset"] = record.dataset
        if hasattr(record, "split"):
            log_data["split"] = record.split
        if hasattr(record, "batch"):
            log_data["batch"] = record.batch
        if hasattr(record, "metrics"):
            log_data["metrics"] = record.metrics
            
        return json.dumps(log_data)


# Model Pricing (per 1M tokens) - Update these based on current Azure rates
MODEL_PRICING = {
    "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60, "cached": 0.075},
    "gpt-4o": {"prompt": 5.00, "completion": 15.00, "cached": 2.50},
    "gpt-4-turbo": {"prompt": 10.00, "completion": 30.00, "cached": 5.00},
    "gpt-4.1-mini": {"prompt": 0.40, "completion": 1.60, "cached": 0.10},
    "gpt-35-turbo": {"prompt": 0.50, "completion": 1.50, "cached": 0.25},
    "gpt-5.2-global": {"prompt": 2.50, "completion": 10.00, "cached": 1.25},
    "gpt-5-mini": {"prompt": 0.25, "completion": 2.00, "cached": 0.025},
    "default": {"prompt": 0.40, "completion": 1.60, "cached": 0.10},
}


def get_model_cost(model_name: str, prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0) -> float:
    """Calculate cost for a given model and token counts."""
    pricing = MODEL_PRICING.get(model_name, MODEL_PRICING["default"])
    
    # Prompt tokens from Azure/OpenAI usage object includes cached tokens
    # We subtract cached tokens from the prompt rate and apply the cached rate
    regular_prompt_tokens = max(0, prompt_tokens - cached_tokens)
    
    cost = (regular_prompt_tokens / 1_000_000 * pricing["prompt"]) + \
           (cached_tokens / 1_000_000 * pricing.get("cached", pricing["prompt"] * 0.5)) + \
           (completion_tokens / 1_000_000 * pricing["completion"])
    return cost


class AugmentationLogger:
    """Structured logger for augmentation runs."""
    
    def __init__(
        self,
        dataset: str,
        log_dir: Path,
        log_level: str = "INFO",
        quiet: bool = False,
    ):
        self.dataset = dataset
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup main logger
        self.logger = logging.getLogger(f"augmentation.{dataset}")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        self.logger.handlers = []  # Clear existing handlers
        
        # Also set level for sub-modules
        logging.getLogger("src.augmentation").setLevel(getattr(logging, log_level.upper()))
        logging.getLogger("openai").setLevel(logging.WARNING)  # Keep openai quiet unless error
        
        # File handler - human readable
        file_handler = logging.FileHandler(log_dir / "augmentation.log")
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s"
        ))
        self.logger.addHandler(file_handler)
        
        # JSON file handler
        json_handler = logging.FileHandler(log_dir / "augmentation.jsonl")
        json_handler.setFormatter(JSONLogFormatter())
        self.logger.addHandler(json_handler)
        
        # Console handler (unless quiet)
        if not quiet:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(message)s",
                datefmt="%H:%M:%S"
            ))
            self.logger.addHandler(console_handler)
        
        # Metrics tracking
        self.metrics = {
            "dataset": dataset,
            "start_time": datetime.utcnow().isoformat(),
            "samples_processed": 0,
            "samples_total": 0,
            "variants_generated": 0,
            "api_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "errors": 0,
        }
    
    def info(self, msg: str, **kwargs):
        record = self.logger.makeRecord(
            self.logger.name, logging.INFO, "", 0, msg, (), None
        )
        for k, v in kwargs.items():
            setattr(record, k, v)
        self.logger.handle(record)
    
    def warning(self, msg: str, **kwargs):
        record = self.logger.makeRecord(
            self.logger.name, logging.WARNING, "", 0, msg, (), None
        )
        for k, v in kwargs.items():
            setattr(record, k, v)
        self.logger.handle(record)
    
    def error(self, msg: str, **kwargs):
        record = self.logger.makeRecord(
            self.logger.name, logging.ERROR, "", 0, msg, (), None
        )
        for k, v in kwargs.items():
            setattr(record, k, v)
        self.logger.handle(record)
        self.metrics["errors"] += 1
    
    def update_metrics(self, **kwargs):
        self.metrics.update(kwargs)
    
    def save_metrics(self):
        self.metrics["end_time"] = datetime.utcnow().isoformat()
        with open(self.log_dir / "metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=2)


class CheckpointManager:
    """Manages checkpoints for resumable augmentation runs."""
    
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = checkpoint_dir / "progress.json"
        self.data_file = checkpoint_dir / "partial_data.csv"
        
    def load(self) -> Optional[Dict[str, Any]]:
        """Load checkpoint if exists."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file) as f:
                return json.load(f)
        return None
    
    def save(
        self,
        processed_ids: List[str],
        current_batch: int,
        total_batches: int,
        partial_df: Optional[pd.DataFrame] = None,
    ):
        """Save checkpoint state."""
        checkpoint = {
            "processed_ids": processed_ids,
            "current_batch": current_batch,
            "total_batches": total_batches,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.checkpoint_file, "w") as f:
            json.dump(checkpoint, f, indent=2)
        
        if partial_df is not None:
            partial_df.to_csv(self.data_file, index=False)
    
    def clear(self):
        """Clear checkpoint files."""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
        if self.data_file.exists():
            self.data_file.unlink()
    
    def load_partial_data(self) -> Optional[pd.DataFrame]:
        """Load partial data from checkpoint."""
        if self.data_file.exists():
            return pd.read_csv(self.data_file)
        return None


def predict_augmentation_cost(
    dataset_id: str,
    split: str,
    num_variants: int,
    model_name: str,
    avg_tokens_per_sentence: int = 25
) -> Dict[str, Any]:
    """
    Predict the cost of augmenting a dataset.
    
    With batched structured output, we make 1 API call per (sample, pivot) combination,
    and each call returns num_variants back-translations in JSON format.
    
    Args:
        dataset_id: Dataset ID
        split: Dataset split
        num_variants: Number of variants per pivot language per request
        model_name: Azure deployment name
        avg_tokens_per_sentence: Estimated tokens per translation
        
    Returns:
        Dictionary with prediction results
    """
    # 1. Get sample count
    data_path = get_dataset_path(dataset_id)
    try:
        if os.path.exists(os.path.join(data_path, "annotations.csv")):
            df = pd.read_csv(os.path.join(data_path, "annotations.csv"))
            if "split" in df.columns:
                num_samples = len(df[df["split"] == split])
            else:
                num_samples = len(df)
        else:
            return {"error": f"Annotations not found at {data_path}"}
    except Exception as e:
        return {"error": f"Error reading dataset: {e}"}

    # 2. Get pivot count
    lang = DATASET_CONFIGS.get(dataset_id, {}).get("language", "en")
    pivots = PIVOT_LANGUAGES.get(lang, ["en"])
    num_pivots = len(pivots)
    
    # 3. Calculate total API requests and variants
    # With batched output: 1 request per (sample, pivot) returns num_variants variants
    total_requests = num_samples * num_pivots
    total_variants = total_requests * num_variants
    
    # 4. Estimate tokens
    # Input: ~50 tokens for prompt + ~25 tokens for input sentence
    # Output: JSON with pivot translation + num_variants back-translations (~30 tokens each)
    est_prompt_tokens = int(total_requests * (50 + avg_tokens_per_sentence))
    est_cached_tokens = int(est_prompt_tokens * 0.25)  # System prompts get cached
    est_completion_tokens = int(total_requests * (30 + num_variants * avg_tokens_per_sentence))
    est_total_tokens = est_prompt_tokens + est_completion_tokens
    
    # 5. Calculate cost
    est_cost = get_model_cost(model_name, est_prompt_tokens, est_completion_tokens, est_cached_tokens)
    
    return {
        "dataset": dataset_id,
        "split": split,
        "samples": num_samples,
        "pivots": pivots,
        "num_pivots": num_pivots,
        "num_variants": num_variants,
        "total_requests": total_requests,
        "total_variants": total_variants,
        "est_prompt_tokens": est_prompt_tokens,
        "est_completion_tokens": est_completion_tokens,
        "est_total_tokens": est_total_tokens,
        "est_cost": est_cost,
        "model": model_name
    }


async def augment_dataset(
    dataset_id: str,
    output_dir: Path,
    rate_limiter: AzureRateLimiter,
    num_variants: int = 2,
    batch_size: int = 8,
    limit: Optional[int] = None,
    split: str = "train",
    resume: bool = False,
    log_level: str = "INFO",
    quiet: bool = False,
    progress_bar: bool = True,
) -> Dict[str, Any]:
    """
    Augment a single dataset with back-translation.
    
    Args:
        dataset_id: Dataset identifier
        output_dir: Output directory for augmented data
        rate_limiter: Rate limiter instance
        num_variants: Number of back-translation variants per pivot language
        batch_size: Batch size for API calls
        split: Dataset split to augment
        resume: Whether to resume from checkpoint
        log_level: Logging level
        quiet: Suppress console output
        progress_bar: Show progress bar
    
    Returns:
        Dictionary with augmentation statistics
    """
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = output_dir / "logs" / f"{dataset_id}_{timestamp}"
    aug_logger = AugmentationLogger(dataset_id, log_dir, log_level, quiet)
    
    aug_logger.info(f"Starting augmentation for {dataset_id}")
    
    # Setup checkpoint manager
    checkpoint_mgr = CheckpointManager(log_dir)
    
    # Check for existing checkpoint if resuming
    checkpoint = None
    processed_ids = []
    partial_df = None
    
    if resume:
        checkpoint = checkpoint_mgr.load()
        if checkpoint:
            processed_ids = checkpoint.get("processed_ids", [])
            partial_df = checkpoint_mgr.load_partial_data()
            aug_logger.info(
                f"Resuming from checkpoint: {len(processed_ids)} samples already processed"
            )
    
    # Load dataset
    try:
        data_path = get_dataset_path(dataset_id)
        
        if HAS_SLT_DATASETS:
            dataset = SLTDataset(
                data_dir=data_path,
                input_mode="pose",
                output_mode="text",
                split=split,
            )
            annotations = dataset.annotations.copy()
            aug_logger.info(f"Loaded {len(annotations)} samples from {data_path}/{split} (SLTDataset)")
        else:
            # Use simple CSV loader as fallback
            dataset = SimpleDatasetLoader(data_dir=data_path, split=split)
            annotations = dataset.annotations.copy()
            aug_logger.info(f"Loaded {len(annotations)} samples from {data_path}/{split} (CSV fallback)")
        
    except Exception as e:
        aug_logger.error(f"Failed to load dataset: {e}")
        raise
    
    # Filter out already processed samples
    if processed_ids:
        annotations = annotations[~annotations["id"].isin(processed_ids)]
        aug_logger.info(f"{len(annotations)} samples remaining to process")
    
    aug_logger.update_metrics(samples_total=len(dataset.annotations))
    
    # Create back-translator
    translator = create_back_translator_for_dataset(
        dataset_id,
        rate_limiter=rate_limiter,
        batch_size=batch_size,
        num_variants=num_variants,
    )
    
    # Limit samples if requested
    if limit:
        annotations = annotations.head(limit)
        aug_logger.info(f"Limited to first {limit} samples")
    
    # Process in batches
    all_rows = []
    if partial_df is not None:
        all_rows = partial_df.to_dict("records")
    
    texts = annotations["text"].tolist()
    ids = annotations["id"].tolist()
    total_samples = len(texts)
    
    # Setup progress bar with enhanced metrics
    pbar = None
    num_pivots = len(translator.pivot_langs)
    variants_generated = [0]  # Use list for closure mutability
    
    if progress_bar:
        # Total = samples × pivots (1 API call per combination)
        full_total_samples = len(dataset.annotations)
        total_operations = full_total_samples * num_pivots
        initial_operations = len(processed_ids) * num_pivots
        
        pbar = tqdm(
            total=total_operations,
            initial=initial_operations,
            desc=f"Augmenting {dataset_id}",
            unit="req",
            dynamic_ncols=True,
            smoothing=0.1,  # Use more smoothing for stable ETA
        )
    
    def progress_callback(increment: int, status: str):
        if pbar:
            pbar.update(increment)
            
            # Get metrics from rate limiter
            max_concurrent = rate_limiter.max_concurrent
            avg_duration = rate_limiter.avg_call_duration
            
            # Build clean status: batch size, variants generated, avg call time
            pbar.set_postfix(
                batch=max_concurrent if max_concurrent > 0 else batch_size,
                variants=variants_generated[0],
                avg_call=f"{avg_duration:.1f}s" if avg_duration > 0 else "-",
                refresh=False
            )
    
    # Process batches
    batch_num = 0
    for i in range(0, total_samples, batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        batch_annotations = annotations.iloc[i:i + batch_size]
        
        try:
            # Run back-translation (don't reset stats between batches)
            results = await translator.back_translate(
                batch_texts,
                progress_callback=progress_callback,
                reset_statistics=(batch_num == 0),  # Only reset on first batch
            )
            
            # Create new rows for each sample and its variants
            for j, result in enumerate(results):
                row = batch_annotations.iloc[j].to_dict()
                
                # Original row
                all_rows.append(row)
                processed_ids.append(batch_ids[j])
                
                # Variant rows (only non-duplicates)
                # Now back_translations is Dict[str, List[str]] with multiple variants per pivot
                for pivot, variants in result.back_translations.items():
                    dup_flags = result.is_duplicate.get(pivot, [])
                    scores_list = result.similarity_scores.get(pivot, [])
                    
                    for var_idx, variant in enumerate(variants):
                        # Skip duplicates
                        is_dup = dup_flags[var_idx] if var_idx < len(dup_flags) else False
                        if is_dup:
                            aug_logger.metrics["duplicates_removed"] = aug_logger.metrics.get("duplicates_removed", 0) + 1
                            continue
                        
                        variant_row = row.copy()
                        variant_row["text"] = variant
                        variant_row["augmentation_pivot"] = pivot
                        variant_row["augmentation_variant_idx"] = var_idx
                        variant_row["augmentation_method"] = "back_translate"
                        
                        # Add similarity scores to the row
                        if var_idx < len(scores_list):
                            scores = scores_list[var_idx]
                            variant_row["similarity_jaccard_char"] = round(scores.jaccard_char, 4)
                            variant_row["similarity_jaccard_word"] = round(scores.jaccard_word, 4)
                            variant_row["similarity_levenshtein"] = round(scores.levenshtein, 4)
                            variant_row["similarity_ngram"] = round(scores.ngram_3, 4)
                            variant_row["similarity_average"] = round(scores.average, 4)
                        
                        all_rows.append(variant_row)
                        aug_logger.metrics["variants_generated"] += 1
                        variants_generated[0] += 1
            
            aug_logger.metrics["samples_processed"] += len(batch_texts)
            
            # Save checkpoint every 10 batches
            batch_num += 1
            if batch_num % 10 == 0:
                checkpoint_mgr.save(
                    processed_ids,
                    batch_num,
                    (total_samples + batch_size - 1) // batch_size,
                    pd.DataFrame(all_rows),
                )
                aug_logger.info(
                    f"Checkpoint saved: {len(processed_ids)}/{len(dataset.annotations)} samples",
                    batch=batch_num,
                )
            
        except Exception as e:
            aug_logger.error(f"Batch {batch_num} failed: {e}", batch=batch_num)
            # Save checkpoint on error
            checkpoint_mgr.save(
                processed_ids,
                batch_num,
                (total_samples + batch_size - 1) // batch_size,
                pd.DataFrame(all_rows),
            )
            raise
    
    if pbar:
        pbar.close()
    
    # Create final DataFrame
    augmented_df = pd.DataFrame(all_rows)
    
    # Save augmented dataset
    output_path = output_dir / dataset_id
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save annotations
    annotations_file = output_path / f"annotations_{split}_augmented.csv"
    augmented_df.to_csv(annotations_file, index=False)
    aug_logger.info(f"Saved augmented annotations to {annotations_file}")
    
    # Also save TSV for Signformer compatibility
    tsv_file = output_path / f"{split}_aug.tsv"
    augmented_df.to_csv(tsv_file, sep="\t", index=False)
    
    # Get detailed statistics from translator
    translator_stats = translator.get_statistics()
    rate_limiter_stats = rate_limiter.metrics
    
    # Compile comprehensive statistics
    comprehensive_stats = {
        "run_info": {
            "dataset": dataset_id,
            "split": split,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "output_dir": str(output_path),
            "azure_deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT", "unknown"),
        },
        "augmentation_stats": translator_stats.to_dict(),
        "rate_limiter_stats": rate_limiter_stats,
        "sample_counts": {
            "original_samples": len(dataset.annotations),
            "output_samples": len(augmented_df),
            "expansion_ratio": round(len(augmented_df) / max(len(dataset.annotations), 1), 2),
        },
    }
    
    # Update and save metrics
    aug_logger.update_metrics(
        api_calls=rate_limiter_stats.get("total_requests", 0),
        prompt_tokens=rate_limiter_stats.get("prompt_tokens", 0),
        completion_tokens=rate_limiter_stats.get("completion_tokens", 0),
        total_tokens=rate_limiter_stats.get("total_tokens", 0),
        exact_duplicates_removed=translator_stats.exact_duplicates_removed,
        near_duplicates_removed=translator_stats.near_duplicates_removed,
        failed_translations=translator_stats.failed_translations,
        similarity_distribution=translator_stats.to_dict().get("similarity_distribution", {}),
        per_pivot_stats=translator_stats.pivot_stats,
    )
    aug_logger.save_metrics()
    
    # Save detailed statistics to multiple files
    # 1. Comprehensive stats JSON
    stats_file = output_path / f"{split}_augmentation_stats.json"
    with open(stats_file, "w") as f:
        json.dump(comprehensive_stats, f, indent=2)
    aug_logger.info(f"Saved detailed statistics to {stats_file}")
    
    # 2. Similarity scores CSV (for analysis)
    if translator_stats.similarity_scores:
        similarity_df = pd.DataFrame({
            "similarity_score": translator_stats.similarity_scores
        })
        similarity_file = output_path / f"{split}_similarity_scores.csv"
        similarity_df.to_csv(similarity_file, index=False)
        aug_logger.info(f"Saved similarity scores to {similarity_file}")
    
    # 3. Per-pivot statistics CSV
    if translator_stats.pivot_stats:
        pivot_rows = []
        for pivot, pstats in translator_stats.pivot_stats.items():
            pivot_rows.append({
                "pivot_language": pivot,
                **pstats
            })
        pivot_df = pd.DataFrame(pivot_rows)
        pivot_file = output_path / f"{split}_pivot_stats.csv"
        pivot_df.to_csv(pivot_file, index=False)
        aug_logger.info(f"Saved pivot statistics to {pivot_file}")
    
    # 4. Human-readable summary
    summary_text = translator_stats.print_summary()
    summary_file = output_path / f"{split}_augmentation_summary.txt"
    model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", "unknown")
    p_tokens = rate_limiter_stats.get("prompt_tokens", 0)
    ca_tokens = rate_limiter_stats.get("cached_prompt_tokens", 0)
    c_tokens = rate_limiter_stats.get("completion_tokens", 0)
    total_cost = get_model_cost(model_name, p_tokens, c_tokens, ca_tokens)
    
    with open(summary_file, "w") as f:
        f.write(f"Augmentation Run: {dataset_id}/{split}\n")
        f.write(f"Timestamp: {datetime.utcnow().isoformat()}Z\n")
        f.write(f"Azure Deployment: {model_name}\n\n")
        f.write(summary_text)
        f.write(f"\n\nRate Limiter Stats:\n")
        f.write(f"  Total Requests:    {rate_limiter_stats.get('total_requests', 0):,}\n")
        f.write(f"  Prompt Tokens:     {p_tokens:,}\n")
        f.write(f"  Cached Tokens:     {ca_tokens:,}\n")
        f.write(f"  Completion Tokens: {c_tokens:,}\n")
        f.write(f"  Total Tokens:      {rate_limiter_stats.get('total_tokens', 0):,}\n")
        f.write(f"  Estimated Cost:    ${total_cost:.4f}\n")
        f.write(f"  Retries:           {rate_limiter_stats.get('retries', 0):,}\n")
        f.write(f"  Total Wait Time:   {rate_limiter_stats.get('total_wait_time_sec', 0):.2f}s\n")
    aug_logger.info(f"Saved summary to {summary_file}")
    
    # 5. Copy stats to log directory as well
    log_stats_file = log_dir / "comprehensive_stats.json"
    with open(log_stats_file, "w") as f:
        json.dump(comprehensive_stats, f, indent=2)
    
    # Print statistics summary to console
    print("\n" + summary_text)
    
    # Clear checkpoint on success
    checkpoint_mgr.clear()
    
    aug_logger.info(
        f"Augmentation complete: {len(dataset.annotations)} → {len(augmented_df)} samples"
    )
    aug_logger.info(f"Statistics saved to: {output_path}")
    
    return aug_logger.metrics


def print_summary_report(results: Dict[str, Dict[str, Any]]):
    """Print a summary report for all datasets."""
    print("\n" + "=" * 110)
    print(f"AUGMENTATION SUMMARY - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 110)
    print()
    print(f"{'Dataset':<20} | {'Samples':>8} | {'Variants':>10} | {'Dups':>6} | {'Tokens (P/C)':>22} | {'Cost':>10}")
    print("-" * 110)
    
    total_samples = 0
    total_augmented = 0
    total_prompt_tokens = 0
    total_cached_tokens = 0
    total_completion_tokens = 0
    total_duplicates = 0
    total_cost = 0.0
    
    model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", "default")
    
    for dataset, metrics in results.items():
        samples = metrics.get("samples_total", 0)
        augmented = metrics.get("samples_processed", 0) + metrics.get("variants_generated", 0)
        p_tokens = metrics.get("prompt_tokens", 0)
        ca_tokens = metrics.get("cached_prompt_tokens", 0)
        c_tokens = metrics.get("completion_tokens", 0)
        exact_dups = metrics.get("exact_duplicates_removed", 0)
        near_dups = metrics.get("near_duplicates_removed", 0)
        duplicates = exact_dups + near_dups
        
        cost = get_model_cost(model_name, p_tokens, c_tokens, ca_tokens)
        
        total_samples += samples
        total_augmented += augmented
        total_prompt_tokens += p_tokens
        total_cached_tokens += ca_tokens
        total_completion_tokens += c_tokens
        total_duplicates += duplicates
        total_cost += cost
        
        token_str = f"{p_tokens:,}/{c_tokens:,}"
        print(f"{dataset:<20} | {samples:>8,} | {augmented:>10,} | {duplicates:>6,} | {token_str:>22} | ${cost:>9.2f}")
    
    print("-" * 110)
    total_token_str = f"{total_prompt_tokens:,}/{total_completion_tokens:,}"
    print(f"{'TOTAL':<20} | {total_samples:>8,} | {total_augmented:>10,} | {total_duplicates:>6,} | {total_token_str:>22} | ${total_cost:>9.2f}")
    print()
    
    print(f"Total Prompt Tokens:     {total_prompt_tokens:,}")
    print(f"Total Cached Tokens:     {total_cached_tokens:,}")
    print(f"Total Completion Tokens: {total_completion_tokens:,}")
    print(f"Total Combined Tokens:   {total_prompt_tokens + total_completion_tokens:,}")
    print(f"Duplicates Removed:      {total_duplicates:,}")
    print(f"Total Estimated Cost:    ${total_cost:.2f}")
    print("=" * 110)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Augment SLT datasets using back-translation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Augment a single dataset
  python -m src.augmentation.augment_dataset --dataset RWTH_PHOENIX_2014T

  # Augment all datasets with summary
  python -m src.augmentation.augment_dataset --all-datasets --summary-report

  # Resume an interrupted run
  python -m src.augmentation.augment_dataset --dataset How2Sign --resume

  # Custom rate limits
  python -m src.augmentation.augment_dataset --dataset GSL --rpm 30 --tpm 60000
        """
    )
    
    # Dataset selection
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument(
        "--dataset",
        type=str,
        choices=list(DATASET_CONFIGS.keys()),
        help="Dataset to augment"
    )
    dataset_group.add_argument(
        "--all-datasets",
        action="store_true",
        help="Augment all available datasets"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,  # Will use AUGMENTED_DATASETS_DIR env var
        help="Output directory for augmented data (default: $AUGMENTED_DATASETS_DIR)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split to augment"
    )
    
    # Augmentation options
    parser.add_argument(
        "--num-variants",
        type=int,
        default=2,
        help="Number of back-translation variants per pivot language (default: 2)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for API calls"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to process (for testing)"
    )
    
    # Rate limiting
    parser.add_argument(
        "--rpm",
        type=int,
        default=None,
        help="Requests per minute limit (default: from env or 60)"
    )
    parser.add_argument(
        "--tpm",
        type=int,
        default=None,
        help="Tokens per minute limit (default: from env or 90000)"
    )
    
    # Checkpoint/resume
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available"
    )
    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="Only predict costs and statistics, don't run augmentation"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory containing checkpoint to resume from"
    )
    
    # Logging options
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Custom log directory"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output"
    )
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="Disable progress bar"
    )
    parser.add_argument(
        "--summary-report",
        action="store_true",
        help="Print summary report after completion"
    )
    
    return parser.parse_args()


# Color codes for console output
YELLOW = '\033[1;33m'
NC = '\033[0m'  # No Color


async def main_async(args: argparse.Namespace):
    """Async main function."""
    # Check SLTDataset availability and warn if using fallback (unless quiet)
    if not HAS_SLT_DATASETS and not args.quiet:
        print(f"{YELLOW}Warning: Could not import SLTDataset: {_slt_import_error}{NC}")
        print(f"{YELLOW}Using simple CSV loader as fallback.{NC}")
        print()
    
    # Check if using OpenAI direct or Azure
    use_openai_direct = os.getenv("USE_OPENAI_DIRECT", "false").lower() == "true"
    
    # Validate environment based on which API we're using
    if use_openai_direct:
        required_env = ["OPENAI_API_KEY"]
    else:
        required_env = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY"]
    
    missing = [v for v in required_env if not os.getenv(v)]
    if missing:
        print(f"Error: Missing required environment variables: {missing}")
        print("Please set them in your .env file or environment.")
        sys.exit(1)
    
    # Print configuration (if not quiet)
    if not args.quiet:
        print("\n" + "=" * 60)
        print("CONFIGURATION")
        print("=" * 60)
        if use_openai_direct:
            print(f"  API: OpenAI Direct")
            print(f"  Model: {os.getenv('OPENAI_MODEL', 'gpt-4.1-mini')}")
        else:
            print(f"  API: Azure OpenAI")
            print(f"  Azure Endpoint: {os.getenv('AZURE_OPENAI_ENDPOINT', 'not set')[:50]}...")
            print(f"  Azure Deployment: {os.getenv('AZURE_OPENAI_DEPLOYMENT', 'not set')}")
            print(f"  API Version: {os.getenv('AZURE_OPENAI_API_VERSION', 'not set')}")
        print(f"  SLT Datasets Dir: {DEFAULT_SLT_DIR}")
        print(f"  Output Dir: {args.output_dir or DEFAULT_AUG_DIR}")
        print("=" * 60)
    
    # Create rate limiter
    rate_limiter = AzureRateLimiter(
        requests_per_minute=args.rpm,
        tokens_per_minute=args.tpm,
    )
    
    # Handle prediction only
    if args.predict_only:
        datasets = list(DATASET_CONFIGS.keys()) if args.all_datasets else [args.dataset]
        model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini")
        
        print(f"\n{BOLD}{BLUE}=== COST PREDICTION ({model_name}) ==={NC}")
        print(f"{BOLD}{'Dataset':<20} | {'Samples':>8} | {'Pivots':>6} | {'Requests':>10} | {'Variants':>10} | {'Tokens (P/Ca/Co)':>20} | {'Est. Cost':>10}{NC}")
        print("-" * 120)
        
        total_cost = 0.0
        total_samples = 0
        total_requests = 0
        total_variants = 0
        
        for ds in datasets:
            prediction = predict_augmentation_cost(ds, args.split, args.num_variants, model_name)
            if "error" in prediction:
                print(f"{ds:<20} | Error: {prediction['error']}")
                continue
            
            total_cost += prediction["est_cost"]
            total_samples += prediction["samples"]
            total_requests += prediction["total_requests"]
            total_variants += prediction["total_variants"]
            
            token_str = f"{prediction['est_prompt_tokens'] // 1000}k/{prediction.get('est_cached_tokens', 0) // 1000}k/{prediction['est_completion_tokens'] // 1000}k"
            print(f"{ds:<20} | {prediction['samples']:>8,} | {prediction['num_pivots']:>6} | {prediction['total_requests']:>10,} | {prediction['total_variants']:>10,} | {token_str:>20} | ${prediction['est_cost']:>9.2f}")
        
        print("-" * 120)
        print(f"{BOLD}{'TOTAL':<20} | {total_samples:>8,} | {'':>6} | {total_requests:>10,} | {total_variants:>10,} | {'':>20} | ${total_cost:>9.2f}{NC}")
        print("=" * 120)
        print(f"\n{YELLOW}Note: Predictions are estimates (P=Prompt, Ca=Cached, Co=Completion).{NC}\n")
        return
    
    # Use default output directory from env if not specified
    output_dir = Path(args.output_dir) if args.output_dir else Path(DEFAULT_AUG_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine datasets to process
    if args.all_datasets:
        datasets = list(DATASET_CONFIGS.keys())
    else:
        datasets = [args.dataset]
    
    results = {}
    
    for dataset_id in datasets:
        print(f"\n{'='*60}")
        print(f"Processing: {dataset_id}")
        print(f"{'='*60}")
        
        try:
            metrics = await augment_dataset(
                dataset_id=dataset_id,
                output_dir=output_dir,
                rate_limiter=rate_limiter,
                num_variants=args.num_variants,
                batch_size=args.batch_size,
                limit=args.limit,
                split=args.split,
                resume=args.resume,
                log_level=args.log_level,
                quiet=args.quiet,
                progress_bar=not args.no_progress_bar,
            )
            results[dataset_id] = metrics
            
        except Exception as e:
            print(f"Error processing {dataset_id}: {e}")
            results[dataset_id] = {"error": str(e), "errors": 1}
    
    # Print summary if requested
    if args.summary_report or args.all_datasets:
        print_summary_report(results)
    
    return results


def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        # Ensure nest_asyncio for Jupyter compatibility
        try:
            import nest_asyncio
            nest_asyncio.apply()
        except ImportError:
            pass
        
        asyncio.run(main_async(args))
        
    except KeyboardInterrupt:
        print("\nInterrupted by user. Checkpoint saved if available.")
        sys.exit(130)


if __name__ == "__main__":
    main()
