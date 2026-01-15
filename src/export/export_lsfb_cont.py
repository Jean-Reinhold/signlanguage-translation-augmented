#!/usr/bin/env python3
"""
Export LSFB-CONT dataset to Signformer .pami0 format.

LSFB-CONT has video-level pose files with segment annotations (start/end timestamps).
This script slices the poses and exports individual segments.

Usage:
    python src/export/export_lsfb_cont.py
"""

import os
import gzip
import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Configuration
DATA_DIR = "/mnt/disk3Tb/slt-datasets/LSFB-CONT"
OUTPUT_DIR = "/mnt/disk3Tb/exported-slt-datasets"
FPS = 50  # Calculated from annotations: ~50.5 fps

# Pose directories
POSE_TYPES = ["pose", "left_hand", "right_hand"]  # Skip face for efficiency (like other datasets)


def load_video_poses(video_id: str) -> Optional[np.ndarray]:
    """
    Load and concatenate pose keypoints for a video.
    
    Returns:
        Combined pose array of shape (frames, total_keypoints, 3)
        where total_keypoints = 33 (body) + 21 (left) + 21 (right) = 75
    """
    poses = []
    
    for pose_type in POSE_TYPES:
        path = os.path.join(DATA_DIR, "poses", pose_type, f"{video_id}.npy")
        if not os.path.exists(path):
            return None
        poses.append(np.load(path))
    
    # Concatenate along keypoint dimension: (frames, 33+21+21, 3)
    return np.concatenate(poses, axis=1)


def timestamp_to_frame(timestamp_ms: int) -> int:
    """Convert millisecond timestamp to frame index."""
    return int(timestamp_ms * FPS / 1000)


def format_pose(pose: np.ndarray) -> torch.Tensor:
    """
    Format pose for Signformer.
    
    Input: (frames, 75, 3) - 75 keypoints with x,y,z
    Output: (frames, 150) - 75 keypoints with x,y (drop z, flatten)
    """
    # Take only x,y coordinates
    pose_2d = pose[:, :, :2]  # (frames, 75, 2)
    
    # Flatten to (frames, 150)
    pose_flat = pose_2d.reshape(pose_2d.shape[0], -1)
    
    # Convert to tensor and handle NaN
    tensor = torch.from_numpy(pose_flat.astype(np.float32))
    tensor = torch.nan_to_num(tensor, nan=0.0)
    
    return tensor


def export_split(annotations: pd.DataFrame, split: str, output_path: str) -> int:
    """Export a single split to .pami0 format."""
    
    split_ann = annotations[annotations['split'] == split]
    if len(split_ann) == 0:
        print(f"  No samples for split '{split}'")
        return 0
    
    # Cache loaded video poses
    pose_cache: Dict[str, np.ndarray] = {}
    
    samples = []
    skipped = 0
    
    for _, row in tqdm(split_ann.iterrows(), total=len(split_ann), desc=f"  {split}"):
        video_id = row['video_id']
        
        # Load poses (with caching)
        if video_id not in pose_cache:
            poses = load_video_poses(video_id)
            if poses is None:
                skipped += 1
                continue
            pose_cache[video_id] = poses
        
        poses = pose_cache[video_id]
        
        # Extract segment
        start_frame = timestamp_to_frame(row['start'])
        end_frame = timestamp_to_frame(row['end'])
        
        # Clamp to valid range
        start_frame = max(0, start_frame)
        end_frame = min(len(poses), end_frame)
        
        if end_frame <= start_frame:
            skipped += 1
            continue
        
        segment_pose = poses[start_frame:end_frame]
        
        # Format for Signformer
        sign_tensor = format_pose(segment_pose)
        
        samples.append({
            "sign": sign_tensor,
            "text": row['text'],
            "gloss": "",
            "signer": str(row.get('video_id', '')),
            "name": row['id']
        })
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with gzip.open(output_path, "wb") as f:
        pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"  {split}: {len(samples)} samples saved (skipped {skipped})")
    return len(samples)


def main():
    print("=" * 60)
    print("Exporting LSFB-CONT to Signformer format")
    print("=" * 60)
    
    # Load annotations
    ann_path = os.path.join(DATA_DIR, "annotations.csv")
    annotations = pd.read_csv(ann_path)
    print(f"Loaded {len(annotations)} annotations")
    
    # Export each split
    total = 0
    for split in ["train", "val", "test"]:
        output_path = os.path.join(OUTPUT_DIR, f"LSFB-CONT.pami0.{split}")
        count = export_split(annotations, split, output_path)
        total += count
    
    print("=" * 60)
    print(f"Total exported: {total} samples")
    print("=" * 60)


if __name__ == "__main__":
    main()
