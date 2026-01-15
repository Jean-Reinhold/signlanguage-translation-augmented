# Dataset Export to 1024 Features

This document describes the process of exporting How2Sign and LSFB-CONT datasets to Signformer's `.pami0` format with unified 1024 features.

## Table of Contents

1. [Overview](#overview)
2. [Problem Statement](#problem-statement)
3. [Solution Architecture](#solution-architecture)
4. [Input Data Requirements](#input-data-requirements)
5. [Output Format](#output-format)
6. [Running the Export](#running-the-export)
7. [Memory Management](#memory-management)
8. [Troubleshooting](#troubleshooting)

---

## Overview

The Sign Language Translation (SLT) project uses multiple datasets with different pose estimation formats:

| Dataset | Pose Format | Keypoints | Features |
|---------|-------------|-----------|----------|
| RWTH-PHOENIX-2014T | MediaPipe Holistic | 512 | **1024** |
| GSL | MediaPipe Holistic | 512 | **1024** |
| LSAT | MediaPipe Holistic | 512 | **1024** |
| How2Sign | OpenPose | 67 | 134 |
| LSFB-CONT | Custom (body+hands) | 75 | 150 |

To train a unified model on all datasets, How2Sign and LSFB-CONT must be padded to 1024 features.

---

## Problem Statement

### Feature Size Mismatch

When combining datasets for multi-corpus training, Signformer's data loader expects all samples to have the same feature dimension. The error manifests as:

```
RuntimeError: stack expects each tensor to be equal size, 
but got [134] at entry 0 and [1024] at entry 189
```

### Memory Exhaustion

Large datasets (15k+ samples with long video sequences) exceed available RAM when loaded entirely into memory. The system would freeze with:

```
Mem: 31Gi total, 30Gi used, 48Mi available
Swap: 8Gi total, 8Gi used, 5Mi free
```

---

## Solution Architecture

### Feature Padding

Zero-pad smaller feature vectors to 1024 dimensions:

```
How2Sign (134 features):
┌─────────────────────────────────────────────────────────────┐
│ Body (50) │ L.Hand (42) │ R.Hand (42) │ Zero Padding (890)  │
└─────────────────────────────────────────────────────────────┘
     0          50            92           134              1024

LSFB-CONT (150 features):
┌─────────────────────────────────────────────────────────────┐
│ Body (66) │ L.Hand (42) │ R.Hand (42) │ Zero Padding (874)  │
└─────────────────────────────────────────────────────────────┘
     0          66           108           150              1024
```

### Streaming Export

Instead of loading all samples into memory:

1. Process samples in chunks of 100-200
2. Save each chunk to a separate `.part{N}` file
3. No final merge required - Signformer loads parts via streaming

```
Traditional (OOM):
┌──────────────────────────────────────┐
│     Load ALL 15k samples to RAM      │ → OOM Kill
└──────────────────────────────────────┘

Streaming (Safe):
┌────────┐ ┌────────┐ ┌────────┐
│ Part 0 │ │ Part 1 │ │ Part N │ → Save each to disk
│ 200    │ │ 200    │ │  46    │
└────────┘ └────────┘ └────────┘
```

---

## Input Data Requirements

### Directory Structure

```
/mnt/disk3Tb/
├── slt-datasets/                          # Raw datasets (read-only)
│   ├── How2Sign/
│   │   ├── annotations.csv                # id, text, split
│   │   └── sentence_level/
│   │       ├── train/
│   │       ├── val/
│   │       └── test/
│   │           └── rgb_front/features/openpose_output/json/
│   │               └── {video_id}-5-rgb_front/
│   │                   └── frame_000000_keypoints.json
│   │                   └── frame_000001_keypoints.json
│   │                   └── ...
│   │
│   └── LSFB-CONT/
│       ├── annotations.csv                # id, text, video_id, start, end, split
│       └── poses/
│           ├── pose/{video_id}.npy        # (frames, 33, 3)
│           ├── left_hand/{video_id}.npy   # (frames, 21, 3)
│           └── right_hand/{video_id}.npy  # (frames, 21, 3)
│
├── augmented-slt-datasets/                # Augmented annotations (read-only)
│   ├── How2Sign/
│   │   └── train_aug.tsv                  # id, text (augmented)
│   └── LSFB-CONT/
│       └── annotations_train_augmented.csv
│
└── exported-slt-datasets/                 # Output directory (read-write)
    └── {Dataset}-{aug?}-1024.pami0.{split}.part{N}
```

### Annotation File Formats

**How2Sign annotations.csv:**
```csv
id,text,split
--7E2sU6zP4_10,"The weather is nice today",train
--7E2sU6zP4_11,"I went to the store",train
```

**LSFB-CONT annotations.csv:**
```csv
id,text,video_id,start,end,split
CLSFBI0103A_S001_B_0,"Bonjour",CLSFBI0103A_S001_B,251198,252692,train
```

**OpenPose JSON (per frame):**
```json
{
  "people": [{
    "pose_keypoints_2d": [x1,y1,c1, x2,y2,c2, ...],  // 25 keypoints
    "hand_left_keypoints_2d": [...],                  // 21 keypoints
    "hand_right_keypoints_2d": [...]                  // 21 keypoints
  }]
}
```

---

## Output Format

### File Naming

```
{Dataset}-{suffix}.pami0.{split}.part{N}

Examples:
  How2Sign-1024.pami0.train.part0
  How2Sign-1024.pami0.train.part1
  How2Sign-aug-1024.pami0.train.part0
  LSFB-CONT-aug-1024.pami0.val.part0
```

### File Contents

Each `.part{N}` file is a gzip-compressed pickle containing a list of samples:

```python
import gzip
import pickle

with gzip.open("How2Sign-1024.pami0.train.part0", "rb") as f:
    samples = pickle.load(f)

# samples = [
#     {
#         "sign": torch.Tensor,  # Shape: (num_frames, 1024)
#         "text": str,           # Target translation
#         "gloss": str,          # Empty string (not used)
#         "signer": str,         # Video/signer identifier
#         "name": str            # Sample ID
#     },
#     ...
# ]
```

### Export Statistics

| Dataset | Split | Samples | Parts |
|---------|-------|---------|-------|
| LSFB-CONT-1024 | train | 4,430 | 1 |
| LSFB-CONT-1024 | val | 155 | 1 |
| LSFB-CONT-1024 | test | 804 | 1 |
| LSFB-CONT-aug-1024 | train | 14,446 | 73 |
| LSFB-CONT-aug-1024 | val | 155 | 1 |
| LSFB-CONT-aug-1024 | test | 804 | 5 |
| How2Sign-1024 | train | ~31,000 | 285 |
| How2Sign-1024 | val | ~1,700 | ~17 |
| How2Sign-1024 | test | 2,344 | 24 |
| How2Sign-aug-1024 | train | ~87,000 | 858 |
| How2Sign-aug-1024 | val | ~1,700 | ~17 |
| How2Sign-aug-1024 | test | 2,344 | 24 |

---

## Running the Export

### Prerequisites

- Docker and Docker Compose installed
- At least 16GB RAM available
- Input data mounted at `/mnt/disk3Tb/`

### Build the Export Image

```bash
cd signlanguage-translation-augmented
docker compose -f docker/docker-compose.export.yml build
```

### Run Exports (One at a Time!)

⚠️ **Important:** Run only ONE export at a time to avoid memory exhaustion.

```bash
# LSFB-CONT Augmented (largest, needs 16GB)
docker compose -f docker/docker-compose.export.yml run --rm export-lsfb-aug

# How2Sign Vanilla
docker compose -f docker/docker-compose.export.yml run --rm export-how2sign

# How2Sign Augmented
docker compose -f docker/docker-compose.export.yml run --rm export-how2sign-aug
```

### Monitor Progress

In a separate terminal:
```bash
# Watch memory usage
watch -n 1 free -h

# Check Docker container stats
docker stats
```

---

## Memory Management

### Configuration

| Service | Memory Limit | Chunk Size | Rationale |
|---------|-------------|------------|-----------|
| export-lsfb-aug | 16GB | 200 | Large video poses, many augmented samples |
| export-how2sign | 4GB | 100 | Per-frame JSON parsing is slower |
| export-how2sign-aug | 4GB | 100 | Same as vanilla |

### Tuning

If you encounter OOM errors, reduce `CHUNK_SIZE` in the export scripts:

```python
# In export_lsfb_cont_1024.py or export_how2sign_1024.py
CHUNK_SIZE = 100  # Reduce from 200 to 100
```

---

## Troubleshooting

### Error: Exit Code 137 (OOM Killed)

**Cause:** Container exceeded memory limit during merge phase.

**Solution:** The scripts now use part files (no merge). If still occurring:
1. Reduce `CHUNK_SIZE` in the export script
2. Increase `mem_limit` in docker-compose.export.yml

### Error: "No samples for split"

**Cause:** Annotation file missing or wrong format.

**Solution:** Check annotation file exists and has correct columns:
- How2Sign: `id`, `text`, `split`
- LSFB-CONT: `id`, `text`, `video_id`, `start`, `end`, `split`

### Error: "Pose file not found"

**Cause:** Missing pose data for a video.

**Solution:** These samples are skipped automatically. Check the "skipped" count in output.

### Slow Export

**Cause:** How2Sign reads hundreds of JSON files per sample.

**Solution:** This is expected. How2Sign exports take 1-2 hours. LSFB-CONT is faster (~10 min).

---

## Integration with Signformer

After export, configure Signformer to load the new datasets:

```yaml
# configs/combined_all.yaml
data:
    train:
      - RWTH_PHOENIX_2014T.pami0.train
      - GSL.pami0.train
      - LSAT.pami0.train
      - How2Sign-1024.pami0.train      # New padded version
      - LSFB-CONT-1024.pami0.train     # New padded version
    feature_size: 1024
    stream_train_parts: true           # Enable streaming for part files
```

**Note:** Signformer's data loader may need modification to glob `.part*` files. See the Signformer documentation for streaming configuration.

---

## Files Reference

| File | Description |
|------|-------------|
| `docker/docker-compose.export.yml` | Docker Compose configuration |
| `docker/Dockerfile.export` | Docker image definition |
| `src/export/export_how2sign_1024.py` | How2Sign export script |
| `src/export/export_lsfb_cont_1024.py` | LSFB-CONT export script |

---

*Last updated: January 2026*
