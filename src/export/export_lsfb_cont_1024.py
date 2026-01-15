#!/usr/bin/env python3
"""
Export LSFB-CONT dataset to Signformer .pami0 format with 1024 features.

=============================================================================
PROBLEM:
LSFB-CONT uses a custom pose format with 75 keypoints (150 features after
flattening x,y coords), while other datasets use MediaPipe with 512 keypoints
(1024 features). This script pads LSFB-CONT to 1024 features for compatibility.

INPUT DATA STRUCTURE:
    /mnt/disk3Tb/slt-datasets/LSFB-CONT/
    ├── annotations.csv          # Columns: id, text, video_id, start, end, split
    └── poses/
        ├── pose/{video_id}.npy       # Body keypoints (33 kp, frames x 33 x 3)
        ├── left_hand/{video_id}.npy  # Left hand (21 kp, frames x 21 x 3)
        └── right_hand/{video_id}.npy # Right hand (21 kp, frames x 21 x 3)

    /mnt/disk3Tb/augmented-slt-datasets/LSFB-CONT/
    └── annotations_train_augmented.csv  # Augmented text annotations

OUTPUT FORMAT:
    /mnt/disk3Tb/exported-slt-datasets/LSFB-CONT{-aug}-1024.pami0.{split}.part{N}
    
    Each part is a gzipped pickle with list of samples:
    [{"sign": Tensor(frames, 1024), "text": str, "gloss": "", "signer": str, "name": str}]

FEATURE LAYOUT (1024-dim):
    [0:66]    = Body pose (33 keypoints * 2 coords)
    [66:108]  = Left hand (21 keypoints * 2 coords)
    [108:150] = Right hand (21 keypoints * 2 coords)
    [150:1024] = Zero padding

MEMORY MANAGEMENT:
    - Streams data in chunks of 200 samples to avoid OOM
    - Each chunk saved as separate .part file (no final merge needed)
    - Recommended Docker memory limit: 16GB

=============================================================================
Usage:
    python export_lsfb_cont_1024.py              # Vanilla (train/val/test)
    python export_lsfb_cont_1024.py --augmented  # Augmented train + vanilla val/test
"""

import os
import gc
import sys
import gzip
import pickle
import argparse
import tempfile
import shutil
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Configuration
DATA_DIR = "/mnt/disk3Tb/slt-datasets/LSFB-CONT"
AUG_DATA_DIR = "/mnt/disk3Tb/augmented-slt-datasets/LSFB-CONT"
OUTPUT_DIR = "/mnt/disk3Tb/exported-slt-datasets"
FPS = 50
TARGET_FEATURES = 1024
CHUNK_SIZE = 200  # Save to disk every N samples

POSE_TYPES = ["pose", "left_hand", "right_hand"]


def load_video_poses(video_id: str) -> Optional[np.ndarray]:
    """Load and concatenate pose keypoints for a video."""
    poses = []
    for pose_type in POSE_TYPES:
        path = os.path.join(DATA_DIR, "poses", pose_type, f"{video_id}.npy")
        if not os.path.exists(path):
            return None
        poses.append(np.load(path))
    return np.concatenate(poses, axis=1)


def timestamp_to_frame(timestamp_ms: int) -> int:
    """Convert millisecond timestamp to frame index."""
    return int(timestamp_ms * FPS / 1000)


def format_pose(pose: np.ndarray) -> torch.Tensor:
    """Format pose with padding to 1024 features."""
    frames = pose.shape[0]
    pose_2d = pose[:, :, :2]
    pose_flat = pose_2d.reshape(frames, -1)
    
    padded = np.zeros((frames, TARGET_FEATURES), dtype=np.float32)
    padded[:, :150] = pose_flat
    
    tensor = torch.from_numpy(padded)
    tensor = torch.nan_to_num(tensor, nan=0.0)
    return tensor


def save_chunk(samples: List[dict], chunk_path: str):
    """Save a chunk of samples."""
    with gzip.open(chunk_path, "wb") as f:
        pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)


def finalize_parts(chunk_paths: List[str], output_base: str):
    """
    Rename chunks to final part files for Signformer streaming.
    Format: {output_base}.part{N} for N=0,1,2...
    """
    os.makedirs(os.path.dirname(output_base), exist_ok=True)
    
    total = 0
    for i, chunk_path in enumerate(chunk_paths):
        # Count samples in chunk
        with gzip.open(chunk_path, "rb") as f:
            chunk_samples = pickle.load(f)
        count = len(chunk_samples)
        total += count
        del chunk_samples
        gc.collect()
        
        # Rename to final part file
        part_path = f"{output_base}.part{i}"
        shutil.move(chunk_path, part_path)
        print(f"    Part {i}: {count} samples -> {os.path.basename(part_path)}")
    
    return total, len(chunk_paths)


def export_split_streaming(annotations: pd.DataFrame, split: str, output_path: str, tmp_dir: str) -> int:
    """Export a split with streaming to disk."""
    
    split_ann = annotations[annotations['split'] == split].reset_index(drop=True)
    if len(split_ann) == 0:
        print(f"  No samples for split '{split}'")
        return 0
    
    print(f"  {split}: {len(split_ann)} samples to process (chunk size: {CHUNK_SIZE})")
    
    pose_cache: Dict[str, np.ndarray] = {}
    current_chunk = []
    chunk_paths = []
    skipped = 0
    chunk_num = 0
    
    for idx, row in tqdm(split_ann.iterrows(), total=len(split_ann), desc=f"  {split}"):
        video_id = row['video_id']
        
        # Load poses with limited cache
        if video_id in pose_cache:
            poses = pose_cache[video_id]
        else:
            poses = load_video_poses(video_id)
            if poses is None:
                skipped += 1
                continue
            # Keep only 5 videos in cache
            if len(pose_cache) >= 5:
                oldest_key = next(iter(pose_cache))
                del pose_cache[oldest_key]
            pose_cache[video_id] = poses
        
        # Extract segment
        start_frame = max(0, timestamp_to_frame(row['start']))
        end_frame = min(len(poses), timestamp_to_frame(row['end']))
        
        if end_frame <= start_frame:
            skipped += 1
            continue
        
        segment_pose = poses[start_frame:end_frame]
        sign_tensor = format_pose(segment_pose)
        
        current_chunk.append({
            "sign": sign_tensor,
            "text": row['text'],
            "gloss": "",
            "signer": str(row.get('video_id', '')),
            "name": row['id']
        })
        
        # Save chunk when full
        if len(current_chunk) >= CHUNK_SIZE:
            chunk_path = os.path.join(tmp_dir, f"{split}_chunk_{chunk_num:04d}.pkl.gz")
            save_chunk(current_chunk, chunk_path)
            chunk_paths.append(chunk_path)
            print(f"    Chunk {chunk_num} saved ({len(current_chunk)} samples)")
            current_chunk = []
            chunk_num += 1
            
            # Clear cache and collect garbage
            pose_cache.clear()
            gc.collect()
    
    # Save remaining samples
    if current_chunk:
        chunk_path = os.path.join(tmp_dir, f"{split}_chunk_{chunk_num:04d}.pkl.gz")
        save_chunk(current_chunk, chunk_path)
        chunk_paths.append(chunk_path)
        print(f"    Chunk {chunk_num} saved ({len(current_chunk)} samples)")
    
    # Clear memory
    del pose_cache
    del current_chunk
    gc.collect()
    
    # Finalize chunks as parts (no merge needed)
    print(f"  Finalizing {len(chunk_paths)} parts...")
    total, num_parts = finalize_parts(chunk_paths, output_path)
    print(f"  {split}: {total} samples in {num_parts} parts (skipped {skipped})")
    
    gc.collect()
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--augmented", action="store_true", help="Export augmented version")
    args = parser.parse_args()
    
    suffix = "-aug-1024" if args.augmented else "-1024"
    print("=" * 60)
    print(f"Exporting LSFB-CONT{suffix} to Signformer format")
    print(f"Target features: {TARGET_FEATURES}")
    print(f"Chunk size: {CHUNK_SIZE} (streaming to disk)")
    print("=" * 60)
    
    # Create temp directory for chunks
    tmp_dir = os.path.join(OUTPUT_DIR, f".tmp_lsfb{suffix}")
    os.makedirs(tmp_dir, exist_ok=True)
    
    try:
        if args.augmented:
            aug_ann_path = os.path.join(AUG_DATA_DIR, "annotations_train_augmented.csv")
            vanilla_ann_path = os.path.join(DATA_DIR, "annotations.csv")
            
            aug_annotations = pd.read_csv(aug_ann_path)
            vanilla_annotations = pd.read_csv(vanilla_ann_path)
            
            print(f"Augmented train samples: {len(aug_annotations)}")
            
            # Export augmented train
            train_path = os.path.join(OUTPUT_DIR, f"LSFB-CONT{suffix}.pami0.train")
            export_split_streaming(aug_annotations, "train", train_path, tmp_dir)
            
            # Export vanilla val/test
            for split in ["val", "test"]:
                output_path = os.path.join(OUTPUT_DIR, f"LSFB-CONT{suffix}.pami0.{split}")
                export_split_streaming(vanilla_annotations, split, output_path, tmp_dir)
        else:
            ann_path = os.path.join(DATA_DIR, "annotations.csv")
            annotations = pd.read_csv(ann_path)
            print(f"Loaded {len(annotations)} annotations")
            
            for split in ["train", "val", "test"]:
                output_path = os.path.join(OUTPUT_DIR, f"LSFB-CONT{suffix}.pami0.{split}")
                export_split_streaming(annotations, split, output_path, tmp_dir)
    finally:
        # Clean up temp directory
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
    
    print("=" * 60)
    print("Export complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
