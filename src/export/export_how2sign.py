#!/usr/bin/env python3
"""
Export How2Sign dataset to Signformer .pami0 format.

How2Sign has OpenPose keypoints in JSON format (per-frame files).
This script converts them to numpy arrays and exports to .pami0 format.

Usage:
    python src/export/export_how2sign.py
"""

import os
import json
import gzip
import pickle
from pathlib import Path
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Configuration
DATA_DIR = "/mnt/disk3Tb/slt-datasets/How2Sign"
OUTPUT_DIR = "/mnt/disk3Tb/exported-slt-datasets"

# OpenPose keypoint counts
POSE_KEYPOINTS = 25  # Body keypoints
FACE_KEYPOINTS = 70  # Face keypoints (we'll skip these for efficiency)
HAND_KEYPOINTS = 21  # Per hand

# We'll use: pose (25) + left_hand (21) + right_hand (21) = 67 keypoints
# Each has x, y, confidence -> we take x, y only


def load_openpose_sequence(json_dir: str) -> Optional[np.ndarray]:
    """
    Load all OpenPose JSON files for a video and return as numpy array.
    
    Returns:
        Array of shape (frames, 67, 2) for pose + hands keypoints
        or None if no valid frames
    """
    if not os.path.exists(json_dir):
        return None
    
    json_files = sorted([f for f in os.listdir(json_dir) if f.endswith('_keypoints.json')])
    
    if not json_files:
        return None
    
    frames = []
    
    for json_file in json_files:
        json_path = os.path.join(json_dir, json_file)
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except:
            continue
        
        if not data.get('people'):
            # No person detected, add zeros
            frames.append(np.zeros((67, 2), dtype=np.float32))
            continue
        
        # Take first person
        person = data['people'][0]
        
        # Extract keypoints (format: x, y, confidence, x, y, confidence, ...)
        pose_kp = person.get('pose_keypoints_2d', [0] * (POSE_KEYPOINTS * 3))
        left_hand = person.get('hand_left_keypoints_2d', [0] * (HAND_KEYPOINTS * 3))
        right_hand = person.get('hand_right_keypoints_2d', [0] * (HAND_KEYPOINTS * 3))
        
        # Reshape to (N, 3) and take only x, y
        pose_arr = np.array(pose_kp, dtype=np.float32).reshape(-1, 3)[:POSE_KEYPOINTS, :2]
        left_arr = np.array(left_hand, dtype=np.float32).reshape(-1, 3)[:HAND_KEYPOINTS, :2]
        right_arr = np.array(right_hand, dtype=np.float32).reshape(-1, 3)[:HAND_KEYPOINTS, :2]
        
        # Concatenate: pose (25) + left_hand (21) + right_hand (21) = 67 keypoints
        frame_kp = np.concatenate([pose_arr, left_arr, right_arr], axis=0)
        frames.append(frame_kp)
    
    if not frames:
        return None
    
    return np.stack(frames, axis=0)  # (frames, 67, 2)


def format_pose(pose: np.ndarray) -> torch.Tensor:
    """
    Format pose for Signformer.
    
    Input: (frames, 67, 2) - 67 keypoints with x, y
    Output: (frames, 134) - flattened
    """
    # Flatten to (frames, 134)
    pose_flat = pose.reshape(pose.shape[0], -1)
    
    # Convert to tensor and handle NaN/inf
    tensor = torch.from_numpy(pose_flat.astype(np.float32))
    tensor = torch.nan_to_num(tensor, nan=0.0)
    
    return tensor


def get_video_id_mapping(annotations: pd.DataFrame, split: str) -> dict:
    """
    Create mapping from annotation ID to video file path.
    
    Annotation IDs are like: --7E2sU6zP4_10
    Video files are like: --7E2sU6zP4_10-5-rgb_front
    """
    split_dir = os.path.join(DATA_DIR, "sentence_level", split, "rgb_front", "features", "openpose_output", "json")
    
    if not os.path.exists(split_dir):
        return {}
    
    video_dirs = os.listdir(split_dir)
    
    # Create mapping: annotation_id -> video_dir_path
    mapping = {}
    for video_dir in video_dirs:
        # Extract ID from video directory name
        # Format: {youtube_id}_{segment}-5-rgb_front
        parts = video_dir.rsplit('-', 2)
        if len(parts) >= 3:
            ann_id = parts[0]  # e.g., --7E2sU6zP4_10
            mapping[ann_id] = os.path.join(split_dir, video_dir)
    
    return mapping


def export_split(annotations: pd.DataFrame, split: str, output_path: str) -> int:
    """Export a single split to .pami0 format."""
    
    split_ann = annotations[annotations['split'] == split]
    if len(split_ann) == 0:
        print(f"  No samples for split '{split}'")
        return 0
    
    # Get video ID mapping
    id_mapping = get_video_id_mapping(annotations, split)
    print(f"  Found {len(id_mapping)} video directories for {split}")
    
    samples = []
    skipped = 0
    
    for _, row in tqdm(split_ann.iterrows(), total=len(split_ann), desc=f"  {split}"):
        ann_id = row['id']
        
        if ann_id not in id_mapping:
            skipped += 1
            continue
        
        json_dir = id_mapping[ann_id]
        
        # Load OpenPose sequence
        pose_sequence = load_openpose_sequence(json_dir)
        
        if pose_sequence is None or len(pose_sequence) == 0:
            skipped += 1
            continue
        
        # Format for Signformer
        sign_tensor = format_pose(pose_sequence)
        
        samples.append({
            "sign": sign_tensor,
            "text": row['text'],
            "gloss": "",
            "signer": "",
            "name": ann_id
        })
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with gzip.open(output_path, "wb") as f:
        pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"  {split}: {len(samples)} samples saved (skipped {skipped})")
    return len(samples)


def main():
    print("=" * 60)
    print("Exporting How2Sign to Signformer format")
    print("=" * 60)
    
    # Load annotations
    ann_path = os.path.join(DATA_DIR, "annotations.csv")
    annotations = pd.read_csv(ann_path)
    print(f"Loaded {len(annotations)} annotations")
    print(f"Splits: {annotations['split'].value_counts().to_dict()}")
    
    # Export each split
    total = 0
    for split in ["train", "val", "test"]:
        output_path = os.path.join(OUTPUT_DIR, f"How2Sign.pami0.{split}")
        count = export_split(annotations, split, output_path)
        total += count
    
    print("=" * 60)
    print(f"Total exported: {total} samples")
    print("=" * 60)


if __name__ == "__main__":
    main()
