#!/usr/bin/env python3
"""
Export How2Sign dataset to Signformer .pami0 format with 1024 features.

=============================================================================
PROBLEM:
How2Sign uses OpenPose with 67 keypoints (134 features after flattening x,y),
while other datasets use MediaPipe with 512 keypoints (1024 features).
This script pads How2Sign to 1024 features for compatibility.

INPUT DATA STRUCTURE:
    /mnt/disk3Tb/slt-datasets/How2Sign/
    ├── annotations.csv                    # Columns: id, text, split
    └── sentence_level/
        └── {train,val,test}/
            └── rgb_front/features/openpose_output/json/
                └── {video_id}-5-rgb_front/    # Per-video directory
                    └── *_keypoints.json       # Per-frame OpenPose output
    
    OpenPose JSON structure per frame:
    {"people": [{"pose_keypoints_2d": [...], "hand_left_keypoints_2d": [...], 
                 "hand_right_keypoints_2d": [...]}]}

    /mnt/disk3Tb/augmented-slt-datasets/How2Sign/
    └── train_aug.tsv                      # Augmented text annotations

OUTPUT FORMAT:
    /mnt/disk3Tb/exported-slt-datasets/How2Sign{-aug}-1024.pami0.{split}.part{N}
    
    Each part is a gzipped pickle with list of samples:
    [{"sign": Tensor(frames, 1024), "text": str, "gloss": "", "signer": "", "name": str}]

FEATURE LAYOUT (1024-dim):
    [0:50]    = Body pose (25 keypoints * 2 coords)
    [50:92]   = Left hand (21 keypoints * 2 coords)
    [92:134]  = Right hand (21 keypoints * 2 coords)
    [134:1024] = Zero padding

MEMORY MANAGEMENT:
    - Streams data in chunks of 100 samples to avoid OOM
    - Each chunk saved as separate .part file (no final merge needed)
    - Recommended Docker memory limit: 4GB

=============================================================================
Usage:
    python export_how2sign_1024.py              # Vanilla (train/val/test)
    python export_how2sign_1024.py --augmented  # Augmented train + vanilla val/test
"""

import os
import gc
import sys
import json
import gzip
import pickle
import argparse
import shutil
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Configuration
DATA_DIR = "/mnt/disk3Tb/slt-datasets/How2Sign"
AUG_DATA_DIR = "/mnt/disk3Tb/augmented-slt-datasets/How2Sign"
OUTPUT_DIR = "/mnt/disk3Tb/exported-slt-datasets"
TARGET_FEATURES = 1024
CHUNK_SIZE = 100  # How2Sign has heavier per-sample processing

# OpenPose keypoint counts
POSE_KEYPOINTS = 25
HAND_KEYPOINTS = 21


def load_openpose_sequence(json_dir: str) -> Optional[np.ndarray]:
    """Load OpenPose JSON files and return as numpy array."""
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
            frames.append(np.zeros((67, 2), dtype=np.float32))
            continue
        
        person = data['people'][0]
        pose_kp = person.get('pose_keypoints_2d', [0] * (POSE_KEYPOINTS * 3))
        left_hand = person.get('hand_left_keypoints_2d', [0] * (HAND_KEYPOINTS * 3))
        right_hand = person.get('hand_right_keypoints_2d', [0] * (HAND_KEYPOINTS * 3))
        
        pose_arr = np.array(pose_kp, dtype=np.float32).reshape(-1, 3)[:POSE_KEYPOINTS, :2]
        left_arr = np.array(left_hand, dtype=np.float32).reshape(-1, 3)[:HAND_KEYPOINTS, :2]
        right_arr = np.array(right_hand, dtype=np.float32).reshape(-1, 3)[:HAND_KEYPOINTS, :2]
        
        frame_kp = np.concatenate([pose_arr, left_arr, right_arr], axis=0)
        frames.append(frame_kp)
    
    if not frames:
        return None
    return np.stack(frames, axis=0)


def format_pose(pose: np.ndarray) -> torch.Tensor:
    """Format pose with padding to 1024 features."""
    frames = pose.shape[0]
    pose_flat = pose.reshape(frames, -1)
    
    padded = np.zeros((frames, TARGET_FEATURES), dtype=np.float32)
    padded[:, :134] = pose_flat
    
    tensor = torch.from_numpy(padded)
    tensor = torch.nan_to_num(tensor, nan=0.0)
    return tensor


def get_video_id_mapping(split: str) -> dict:
    """Create mapping from annotation ID to video file path."""
    split_dir = os.path.join(DATA_DIR, "sentence_level", split, "rgb_front", "features", "openpose_output", "json")
    
    if not os.path.exists(split_dir):
        return {}
    
    video_dirs = os.listdir(split_dir)
    mapping = {}
    for video_dir in video_dirs:
        parts = video_dir.rsplit('-', 2)
        if len(parts) >= 3:
            ann_id = parts[0]
            mapping[ann_id] = os.path.join(split_dir, video_dir)
    return mapping


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
        with gzip.open(chunk_path, "rb") as f:
            chunk_samples = pickle.load(f)
        count = len(chunk_samples)
        total += count
        del chunk_samples
        gc.collect()
        
        part_path = f"{output_base}.part{i}"
        shutil.move(chunk_path, part_path)
        print(f"    Part {i}: {count} samples -> {os.path.basename(part_path)}")
    
    return total, len(chunk_paths)


def export_split_streaming(annotations: pd.DataFrame, split: str, output_path: str, 
                           tmp_dir: str, id_mapping: dict = None) -> int:
    """Export a split with streaming to disk."""
    
    split_ann = annotations[annotations['split'] == split].reset_index(drop=True)
    if len(split_ann) == 0:
        print(f"  No samples for split '{split}'")
        return 0
    
    print(f"  {split}: {len(split_ann)} samples to process (chunk size: {CHUNK_SIZE})")
    
    # Get mapping if not provided
    if id_mapping is None:
        id_mapping = get_video_id_mapping(split)
    print(f"  Found {len(id_mapping)} video directories")
    
    current_chunk = []
    chunk_paths = []
    skipped = 0
    chunk_num = 0
    
    for idx, row in tqdm(split_ann.iterrows(), total=len(split_ann), desc=f"  {split}"):
        ann_id = row['id']
        
        if ann_id not in id_mapping:
            skipped += 1
            continue
        
        json_dir = id_mapping[ann_id]
        pose_sequence = load_openpose_sequence(json_dir)
        
        if pose_sequence is None or len(pose_sequence) == 0:
            skipped += 1
            continue
        
        sign_tensor = format_pose(pose_sequence)
        
        current_chunk.append({
            "sign": sign_tensor,
            "text": row['text'],
            "gloss": "",
            "signer": "",
            "name": ann_id
        })
        
        # Save chunk when full
        if len(current_chunk) >= CHUNK_SIZE:
            chunk_path = os.path.join(tmp_dir, f"{split}_chunk_{chunk_num:04d}.pkl.gz")
            save_chunk(current_chunk, chunk_path)
            chunk_paths.append(chunk_path)
            print(f"    Chunk {chunk_num} saved ({len(current_chunk)} samples)")
            current_chunk = []
            chunk_num += 1
            gc.collect()
    
    # Save remaining samples
    if current_chunk:
        chunk_path = os.path.join(tmp_dir, f"{split}_chunk_{chunk_num:04d}.pkl.gz")
        save_chunk(current_chunk, chunk_path)
        chunk_paths.append(chunk_path)
        print(f"    Chunk {chunk_num} saved ({len(current_chunk)} samples)")
    
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
    print(f"Exporting How2Sign{suffix} to Signformer format")
    print(f"Target features: {TARGET_FEATURES}")
    print(f"Chunk size: {CHUNK_SIZE} (streaming to disk)")
    print("=" * 60)
    
    # Create temp directory
    tmp_dir = os.path.join(OUTPUT_DIR, f".tmp_how2sign{suffix}")
    os.makedirs(tmp_dir, exist_ok=True)
    
    try:
        if args.augmented:
            aug_ann_path = os.path.join(AUG_DATA_DIR, "train_aug.tsv")
            vanilla_ann_path = os.path.join(DATA_DIR, "annotations.csv")
            
            aug_annotations = pd.read_csv(aug_ann_path, sep='\t')
            aug_annotations['split'] = 'train'
            vanilla_annotations = pd.read_csv(vanilla_ann_path)
            
            print(f"Augmented train samples: {len(aug_annotations)}")
            
            # For augmented, we use train split's video mapping
            train_mapping = get_video_id_mapping("train")
            
            # Export augmented train
            train_path = os.path.join(OUTPUT_DIR, f"How2Sign{suffix}.pami0.train")
            export_split_streaming(aug_annotations, "train", train_path, tmp_dir, train_mapping)
            
            # Export vanilla val/test
            for split in ["val", "test"]:
                output_path = os.path.join(OUTPUT_DIR, f"How2Sign{suffix}.pami0.{split}")
                export_split_streaming(vanilla_annotations, split, output_path, tmp_dir)
        else:
            ann_path = os.path.join(DATA_DIR, "annotations.csv")
            annotations = pd.read_csv(ann_path)
            print(f"Loaded {len(annotations)} annotations")
            print(f"Splits: {annotations['split'].value_counts().to_dict()}")
            
            for split in ["train", "val", "test"]:
                output_path = os.path.join(OUTPUT_DIR, f"How2Sign{suffix}.pami0.{split}")
                export_split_streaming(annotations, split, output_path, tmp_dir)
    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
    
    print("=" * 60)
    print("Export complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
