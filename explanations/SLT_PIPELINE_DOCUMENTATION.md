# Sign Language Translation (SLT) Pipeline Documentation

## Overview

This document describes the complete machine learning pipeline for Sign Language Translation (SLT), covering data storage, preprocessing, augmentation, training, and evaluation across two main repositories:

1. **`slt_models_tryout`** - Dataset preparation, augmentation, and custom transformer training
2. **`Signformer`** - Production-ready Sign Language Translation model training

---

## 1. Data Storage & Locations

### 1.1 Raw Datasets

Raw SLT datasets are stored in multiple locations on the disk:

| Location | Purpose |
|----------|---------|
| `/mnt/disk3Tb/slt-datasets/` | Primary storage for formatted SLT datasets |
| `/mnt/disk3Tb/augmented-slt-datasets/` | Augmented dataset variants |
| `/mnt/disk3Tb/exported-slt-datasets/` | Exported datasets in Signformer format (`.pami0.*`) |
| `/mnt/data3/tfds_slt/` | TensorFlow Datasets source data |

### 1.2 Available Datasets

The following datasets are available (from `datasets.json`):

| Dataset ID | Path | Size (GB) |
|------------|------|-----------|
| `GSL` (Greek SL) | `/mnt/disk3Tb/slt-datasets/GSL` | 5.42 |
| `ISL` (Indian SL) | `/mnt/disk3Tb/slt-datasets/ISL` | 422.51 |
| `LSAT` (LSA-T) | `/mnt/disk3Tb/slt-datasets/lsat` | 45.59 |
| `RWTH_PHOENIX_2014T` | `/mnt/disk3Tb/slt-datasets/RWTH_PHOENIX_2014T` | 5.75 |
| `How2Sign` | `/mnt/disk3Tb/slt-datasets/How2Sign` | 69.47 |
| `LSFB-CONT` | `/mnt/disk3Tb/slt-datasets/LSFB-CONT` | 28.05 |
| `Content4All` variants | `/mnt/disk3Tb/slt-datasets/Content4All/...` | varies |
| `WMT-SLT23` | `/mnt/disk3Tb/slt-datasets/WMT-SLT23*` | 463+ |

### 1.3 Dataset Structure

Each dataset in `/mnt/disk3Tb/slt-datasets/{DATASET}/` follows this structure:

```
{DATASET}/
├── metadata.json         # Dataset metadata (name, languages, input/output types)
├── annotations.csv       # Sample annotations with columns: id, text, gloss, signer, split
└── poses/                # Pose data files
    └── {id}.npy          # NumPy arrays of pose keypoints (frames, people, keypoints, coords)
```

#### Pose Data Format
- Shape: `(num_frames, num_people, num_keypoints, coords)` 
- Example: `(47, 1, 543, 3)` for MediaPipe Holistic (543 keypoints with x, y, z)
- Keypoints include: face (468), body/pose (33), left hand (21), right hand (21)

---

## 2. Dataset Formatting Pipeline

### 2.1 Source: TensorFlow Datasets (TFDS)

Raw datasets are initially obtained from TensorFlow Datasets using the `sign_language_datasets` package.

**Location**: `slt_models_tryout/dataset_formatting/rwth/format_db.ipynb`

```python
import tensorflow_datasets as tfds
from sign_language_datasets.datasets.config import SignDatasetConfig

config = SignDatasetConfig(
    name="rwth_phoenix2014_t_poses", 
    version="3.0.0", 
    include_video=False, 
    include_pose="holistic"
)
rwth_phoenix2014_t = tfds.load(
    name='rwth_phoenix2014_t', 
    builder_kwargs=dict(config=config), 
    data_dir="/mnt/data3/tfds_slt"
)
```

### 2.2 SLTDataset Class (slt_datasets package)

The `slt_datasets` package provides a unified interface for loading datasets:

**Package Location**: `/home/pdalbianco/.local/lib/python3.10/site-packages/slt_datasets/`

**Key Components**:
- `SLTDataset` - Main dataset class
- `Pose` class from `posecraft` - Pose data handling
- `WordLevelTokenizer` - Text tokenization

```python
from slt_datasets.SLTDataset import SLTDataset

dataset = SLTDataset(
    data_dir="/mnt/disk3Tb/slt-datasets/RWTH_PHOENIX_2014T",
    input_mode="pose",      # "pose" or "video"
    output_mode="text",     # "text" or "gloss"
    split="train",          # "train", "val", "test", or None
    transforms=transforms,  # Optional transforms
    max_tokens=40           # Optional max token length
)
```

### 2.3 Dataset Formatting Notebooks

Each dataset has its own formatting notebook in `slt_models_tryout/dataset_formatting/`:

| Dataset | Notebook |
|---------|----------|
| RWTH-Phoenix | `rwth/format_db.ipynb`, `rwth/rwth.ipynb` |
| GSL | `gsl/gsl.ipynb` |
| LSAT | `lsat/format_keys.ipynb` |
| ISL | `isl/pose_processing.ipynb` |
| How2Sign | `howtosign/check.ipynb` |
| LSFB-CONT | `lsfb-cont/process.ipynb` |
| Content4All | `content4all/format_vrt_news.py` |

---

## 3. Data Augmentation

### 3.1 Text Augmentation with LLM Paraphrasing

**Location**: `slt_models_tryout/src/expand_db.ipynb`

Text augmentation uses GPT-4o-mini to generate paraphrases of German sentences:

```python
SYSTEM_PROMPT = """
You are a helpful assistant rewriting German sentences.
For every user input, produce {n} paraphrases that:
• Preserve the exact meaning, tense and register.
• Reuse at least 70% of the original words.
• Vary mainly through word order or minor synonym substitutions.
• Keep length within ±3 tokens of the original.
• Do NOT add or omit information.
"""

# Configuration
AUG_FACTOR = 2          # Paraphrases per original
BATCH_SIZE = 8          # Concurrent API requests
MODEL_NAME = "gpt-4o-mini"
```

**Result**: Original 7,096 samples → 21,288 augmented samples (3x)

### 3.2 Augmented Data Integration

The augmented annotations are saved and used to create extended datasets:

**Script**: `Signformer/scripts/build_phoenix14t_ext.py`

```bash
python scripts/build_phoenix14t_ext.py \
    --data_dir PHOENIX2014T \
    --tsv train_aug.tsv \
    --out_prefix phoenix14t-ext.pami0
```

This creates `phoenix14t-ext.pami0.{train,dev,test}` files with duplicated pose samples for each paraphrased text.

---

## 4. Data Export for Signformer

### 4.1 Export Process

**Location**: `slt_models_tryout/src/export_db.ipynb`

Converts SLTDataset format to Signformer's gzip-pickled format:

```python
def format_pose(pose):
    # Take first person, 2D coordinates only
    pose = pose[:, 0, :, :2]  
    return pose.reshape(pose.shape[0], -1)  # Flatten to (frames, keypoints*2)

def export_dataset(dataset: SLTDataset, output_path: str):
    samples = []
    for i in range(len(dataset)):
        pose = format_pose(dataset[i][0])
        # Remove face keypoints (34-95) to get 1024 features
        pose = torch.cat((pose[:, :34], pose[:, 96:]), dim=1)
        pose = torch.nan_to_num(pose, nan=0.0)
        samples.append({
            "sign": pose,          # Tensor (frames, 1024)
            "text": text,
            "gloss": "",
            "signer": "",
            "name": name
        })
    with gzip.open(output_path, "wb") as f:
        pickle.dump(samples, f)
```

### 4.2 Exported Dataset Format

Output files: `/mnt/disk3Tb/exported-slt-datasets/{DATASET}.pami0.{train,val,test}`

Each sample is a dictionary:
```python
{
    "sign": torch.Tensor,  # Shape (num_frames, 1024)
    "text": str,           # Target text
    "gloss": str,          # Gloss sequence (may be empty)
    "signer": str,         # Signer ID (may be empty)
    "name": str            # Sample identifier
}
```

### 4.3 Dataset Merging

Multiple datasets can be merged into a single training set:

```python
DATASETS = ["GSL", "ISL", "LSAT", "RWTH_PHOENIX_2014T"]
# Creates: MERGE.pami0.{train,val,test}
```

---

## 5. Training Pipeline: slt_models_tryout

### 5.1 Configuration

**Location**: `slt_models_tryout/src/config/{DATASET}.json`

```json
{
    "INPUT_MODE": "pose",
    "OUTPUT_MODE": "text",
    "BATCH_SIZE": 64,
    "MAX_FRAMES": 170,
    "MAX_TOKENS": 40,
    "LANDMARKS_USED": ["body", "lhand", "rhand"],
    "TRANSFORMS": [
        "FilterLandmarks",
        "RandomSampleFrames", 
        "PadTruncateFrames",
        "ReplaceNansWithZeros",
        "FlattenKeypoints"
    ],
    "D_MODEL": 128,
    "DROPOUT": 0.2,
    "NUM_ENCODER_LAYERS": 4,
    "NUM_DECODER_LAYERS": 1,
    "LR": 0.0001
}
```

### 5.2 Training Script

**Location**: `slt_models_tryout/src/train.ipynb`

```python
from LightningKeypointsTransformer import LKeypointsTransformer

# Model
l_model = LKeypointsTransformer(hp, device, train_dataset.tokenizer)

# Training with PyTorch Lightning
trainer = L.Trainer(
    logger=WandbLogger(project=DATASET),
    callbacks=[
        EarlyStopping(monitor="val_accuracy", mode="max", patience=30),
        ModelCheckpoint(monitor="val_loss", mode="min"),
    ],
)
trainer.fit(model=l_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
```

### 5.3 Results Storage

```
slt_models_tryout/src/results/{DATASET}/{experiment_name}/
├── hp.json              # Hyperparameters
├── best-epoch=*.ckpt    # Best model checkpoint
└── translations.csv     # Test predictions
```

---

## 6. Training Pipeline: Signformer

### 6.1 Configuration

**Location**: `Signformer/configs/sign.yaml`

```yaml
name: sign_experiment
data:
    data_path: ../PHOENIX2014T
    train: ISL-MERGED.pami0.train
    dev: ISL-MERGED.pami0.val
    test: ISL-MERGED.pami0.test
    feature_size: 1024
    level: word
    max_sent_length: 400
    stream_train_parts: true
    stream_chunk_size: 5000

training:
    model_dir: "/app/Signformer/sign_sample_isl"
    recognition_loss_weight: 0.0    # 0 = translation only
    translation_loss_weight: 1.0
    eval_metric: bleu
    optimizer: sophiag
    learning_rate: 0.0004
    batch_size: 32
    epochs: 1000
    validation_freq: 100

model:
    encoder:
        type: transformer
        num_layers: 1
        num_heads: 8
        hidden_size: 256
        ff_size: 1024
    decoder:
        type: transformer
        num_layers: 1
        num_heads: 8
        hidden_size: 256
        ff_size: 1024
```

### 6.2 Training Commands

```bash
# Train from scratch
python -m main train configs/sign.yaml

# Test with checkpoint
python -m main test configs/sign.yaml --ckpt path/to/best.ckpt
```

### 6.3 Data Loading (Streaming)

Signformer supports streaming large datasets to reduce memory usage:

**Location**: `Signformer/main/data.py`, `Signformer/main/training.py`

```python
# Streaming is enabled via config:
stream_train_parts: true
stream_chunk_size: 5000

# Data is loaded in chunks during training
for chunk in iter_dataset_file(train_path):
    # Process chunk of 5000 samples
```

### 6.4 Results Storage

```
Signformer/sign_finetune/
├── best.ckpt            # Best checkpoint
├── config.yaml          # Training config
├── gls.vocab            # Gloss vocabulary
├── txt.vocab            # Text vocabulary
├── train.log            # Training log
├── validations.txt      # Validation metrics over time
├── tensorboard/         # TensorBoard logs
└── txt/                 # Hypothesis outputs
    └── {step}.dev.hyp.txt
```

---

## 7. Evaluation Metrics

Both pipelines use these metrics:

| Metric | Description |
|--------|-------------|
| **BLEU-4** | Bilingual Evaluation Understudy (primary metric) |
| **BLEU-1/2/3** | N-gram precision variants |
| **CHRF** | Character F-score |
| **ROUGE** | Recall-Oriented Understudy |
| **WER** | Word Error Rate (for recognition) |

### Results Analysis

**Location**: `slt_models_tryout/src/results_analysis.ipynb`

```python
translations = "results/GSL/firm-frog-32/translations.csv"
df = pd.read_csv(translations)

# Perfect translations
perfect = df[df["bleu_4_greedy"] == 1.0]
```

---

## 8. Complete Workflow Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA FLOW DIAGRAM                                  │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌──────────────────────┐
                    │   Raw Video/Poses    │
                    │  (TensorFlow Datasets)│
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  Format to SLTDataset │
                    │  (format_db.ipynb)    │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │ /mnt/disk3Tb/        │
                    │   slt-datasets/      │
                    │  ├── annotations.csv │
                    │  └── poses/*.npy     │
                    └──────────┬───────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
          ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  Text Augment   │  │  Direct Train   │  │  Export to      │
│  (expand_db)    │  │ (slt_models_    │  │  Signformer     │
│  GPT-4o-mini    │  │  tryout)        │  │  (export_db)    │
└────────┬────────┘  └─────────────────┘  └────────┬────────┘
         │                                         │
         ▼                                         ▼
┌─────────────────┐                     ┌─────────────────┐
│  train_aug.tsv  │                     │ .pami0.* files  │
└────────┬────────┘                     │ (gzip+pickle)   │
         │                              └────────┬────────┘
         ▼                                       │
┌─────────────────┐                              │
│build_phoenix14t │                              │
│    _ext.py      │                              │
└────────┬────────┘                              │
         │                                       │
         └───────────────┬───────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   Signformer Train  │
              │  python -m main     │
              │      train          │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   Model Checkpoints │
              │   + Vocabularies    │
              │   + Eval Results    │
              └─────────────────────┘
```

---

## 9. Key Files Reference

### slt_models_tryout
| File | Purpose |
|------|---------|
| `src/expand_db.ipynb` | LLM-based text augmentation |
| `src/export_db.ipynb` | Export to Signformer format |
| `src/train.ipynb` | Custom transformer training |
| `src/config/*.json` | Hyperparameter configs |
| `dataset_formatting/*/` | Dataset-specific formatting |

### Signformer
| File | Purpose |
|------|---------|
| `main/training.py` | Training loop and TrainManager |
| `main/data.py` | Data loading and vocabulary building |
| `main/dataset.py` | SignTranslationDataset class |
| `main/model.py` | Signformer model architecture |
| `configs/*.yaml` | Training configurations |
| `scripts/build_phoenix14t_ext.py` | Augmented dataset builder |

---

## 10. Quick Start Commands

### Format a new dataset
```bash
cd slt_models_tryout/dataset_formatting/{dataset}/
jupyter notebook {dataset}.ipynb
```

### Augment text data
```bash
cd slt_models_tryout/src/
jupyter notebook expand_db.ipynb
# Set DATASET, run cells
```

### Export for Signformer
```bash
cd slt_models_tryout/src/
jupyter notebook export_db.ipynb
# Set DATASET and OUTPUT_DIR, run cells
```

### Train Signformer
```bash
cd Signformer/
python -m main train configs/sign.yaml --gpu_id 0
```

### Evaluate model
```bash
python -m main test configs/sign.yaml --ckpt sign_finetune/best.ckpt
```

---

*Document generated: January 2026*
*Author: Auto-generated from codebase exploration*
