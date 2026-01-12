# agents.md — Sign Language Translation with Data Augmentation

> **Project**: Sign Language Translation Pipeline with LLM-based Data Augmentation  
> **Author**: Pedro Dal Bianco  
> **Status**: Active Development (Master's Thesis Research)

---

## 🎯 Project Goal

This repository implements an **augmented training pipeline for Sign Language Translation (SLT)** based on the **Signformer** architecture. The core hypothesis being tested is:

> *It is possible to improve **generalization** in SLT by training a single keypoint-based model across multiple languages and corpora, supported by carefully controlled **textual and kinematic data augmentation**.*

The project advances three integrated fronts:

1. **Multi-corpora Integration** — Standardizing heterogeneous SLT datasets (PHOENIX-2014T, LSA-T, How2Sign, LSFB-CONT, ISLTranslate, GSL) into a unified manifest format
2. **Multilingual Gloss-free Training** — Training a single Signformer model on multiple sign languages using pose keypoints (no intermediate gloss annotations)
3. **Data Augmentation for Generalization** — Evaluating the impact of LLM-based text augmentation (paraphrases, back-translation) and kinematic pose perturbations on cross-domain/cross-signer performance

---

## 🏗️ Architecture Overview

### Why Signformer?

The **Signformer** model operates on **pose keypoints** (extracted via MediaPipe Holistic) rather than raw video frames. This design choice provides:

- **Dimensionality Reduction**: Keypoints compress visual input to essential geometry/motion
- **Invariance**: Robust to lighting, background, camera angle, and appearance variations
- **Portability**: Enables training across heterogeneous datasets with different capture conditions
- **Interpretability**: Attention maps over keypoints reveal which body regions contribute to predictions
- **Efficiency**: Lower compute requirements than video-based models (larger batch sizes, faster convergence)

### Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA FLOW DIAGRAM                                  │
└─────────────────────────────────────────────────────────────────────────────┘

              ┌──────────────────────┐
              │   Raw Video/Poses    │
              │  (Multiple Datasets) │
              └──────────┬───────────┘
                         │
              ┌──────────▼───────────┐
              │  Format to SLTDataset │  ← lib/slt_datasets/
              │  (Unified Manifest)   │
              └──────────┬───────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
┌───────────────┐ ┌─────────────┐ ┌───────────────┐
│ Text Augment  │ │  Pose Aug   │ │ Export to     │
│ (LLM-based)   │ │ (Kinematic) │ │ Signformer    │
│ GPT-4o-mini   │ │             │ │ .pami0 format │
└───────┬───────┘ └──────┬──────┘ └───────┬───────┘
        │                │                │
        └────────────────┼────────────────┘
                         │
              ┌──────────▼───────────┐
              │   Signformer Train   │
              │  (Transformer Enc/Dec)│
              └──────────┬───────────┘
                         │
              ┌──────────▼───────────┐
              │   Evaluation         │
              │ (BLEU, chrF, ROUGE)  │
              └─────────────────────┘
```

---

## 📁 Repository Structure

```
signlanguage-translation-augmented/
├── agents.md                          # This file
├── README.md                          # Quick start guide
├── pyproject.toml                     # Dependencies & package config
├── requirements.txt                   # Pip dependencies
├── .env.example                       # Environment variables template
├── .env                               # Local environment (gitignored)
│
├── lib/                               # LOCAL LIBRARY PACKAGES
│   ├── posecraft/                     # Pose keypoint manipulation
│   │   ├── Pose.py                    # Pose class & operations
│   │   ├── transforms.py              # Data transforms (filter, pad, flatten)
│   │   └── interpolate.py             # Frame interpolation
│   └── slt_datasets/                  # SLT dataset utilities
│       ├── SLTDataset.py              # Unified dataset class
│       └── WordLevelTokenizer.py      # Text tokenization
│
├── src/                               # Main source code
│   ├── augmentation/                  # TEXT AUGMENTATION
│   │   ├── expand_db.ipynb            # GPT-4o-mini paraphrasing
│   │   └── train_aug.tsv              # Sample augmented data
│   ├── export/                        # Dataset export utilities
│   │   ├── export_db.ipynb            # Export to Signformer format
│   │   └── export_db.py               # Export script
│   ├── training/                      # Custom transformer training
│   │   ├── train.ipynb                # PyTorch Lightning training
│   │   ├── KeypointsTransformer.py    # Transformer model
│   │   ├── LightningKeypointsTransformer.py  # Lightning wrapper
│   │   ├── Translator.py              # Translation utilities
│   │   └── results_analysis.ipynb     # Analyze results
│   ├── config/                        # Hyperparameter configs
│   │   ├── GSL.json
│   │   ├── LSAT.json
│   │   └── RWTH_PHOENIX_2014T.json
│   └── interp/                        # Interpolation experiments
│
├── signformer/                        # SIGNFORMER MODEL (Production)
│   ├── main/                          # Core modules
│   │   ├── training.py                # Training loop
│   │   ├── data.py                    # Data loading
│   │   ├── dataset.py                 # SignTranslationDataset
│   │   ├── model.py                   # Model architecture
│   │   ├── encoders.py                # Transformer encoder
│   │   ├── decoders.py                # Transformer decoder
│   │   └── prediction.py              # Inference
│   ├── configs/                       # YAML configurations
│   │   ├── sign.yaml                  # Main config
│   │   └── sign_finetune.yaml         # Finetuning config
│   ├── scripts/
│   │   └── build_phoenix14t_ext.py    # Build augmented dataset
│   └── requirements.txt               # Signformer-specific deps
│
├── dataset_formatting/                # Dataset-specific formatting
│   ├── rwth/                          # RWTH-Phoenix-2014T (DGS→DE)
│   ├── gsl/                           # Greek Sign Language (GSL→EL)
│   ├── lsat/                          # LSA-T Argentinian (LSA→ES)
│   ├── isl/                           # Indian Sign Language (ISL→EN)
│   ├── howtosign/                     # How2Sign (ASL→EN)
│   └── lsfb-cont/                     # LSFB Continuous (LSFB→FR)
│
├── explanations/                      # Documentation
│   └── SLT_PIPELINE_DOCUMENTATION.md  # Complete pipeline docs
│
└── docs/                              # Thesis and papers
    └── thesis/
        └── exemplo-tcc.tex            # Master's thesis (Portuguese)
```

---

## 🔧 Dependencies & Setup

### Core Dependencies

| Category | Packages |
|----------|----------|
| **ML Frameworks** | `torch>=2.0`, `lightning>=2.0`, `torchmetrics` |
| **Data Processing** | `numpy<2.0`, `pandas`, `scikit-learn`, `h5py` |
| **NLP & Metrics** | `sacrebleu`, `nltk`, `rouge-score` |
| **Visualization** | `matplotlib`, `seaborn`, `tqdm` |
| **Experiment Tracking** | `wandb` |

### Optional Dependencies

```bash
# LLM augmentation (Azure OpenAI API)
pip install openai nest-asyncio python-dotenv

# Signformer training (SophiaG optimizer)
pip install sophia-optimizer

# Dataset formatting (TensorFlow Datasets)
pip install tensorflow tensorflow-datasets sign-language-datasets
```

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
# Edit .env with your Azure OpenAI credentials
```

| Variable | Description |
|----------|-------------|
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI resource endpoint |
| `AZURE_OPENAI_API_KEY` | API key for authentication |
| `AZURE_OPENAI_API_VERSION` | API version (e.g., `2025-04-01-preview`) |
| `AZURE_OPENAI_DEPLOYMENT` | Model deployment name (e.g., `gpt-5-mini`) |

### Quick Setup

```bash
# Using uv (recommended)
./scripts/setup.sh --all

# Manual setup
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[all]"
```

---

## 📊 Available Datasets

| Dataset | Sign Language | Text Language | Samples | Domain |
|---------|---------------|---------------|---------|--------|
| **RWTH-PHOENIX-2014T** | German (DGS) | German | 8,257 | TV Weather |
| **LSA-T** | Argentine (LSA) | Spanish | 14,880 | News |
| **How2Sign** | American (ASL) | English | 35,191 | How-to tutorials |
| **LSFB-CONT** | Belgian French (LSFB) | French | 27,500 | Narratives |
| **ISLTranslate** | Indian (ISL) | English | 31,222 | Educational |
| **GSL Continuous** | Greek (GSL) | Greek | 40,826 | Daily phrases |

### Data Format

Each dataset follows the unified `SLTDataset` structure:

```
{DATASET}/
├── metadata.json         # Dataset metadata
├── annotations.csv       # id, text, gloss, signer, split
└── poses/
    └── {id}.npy          # Shape: (frames, people, keypoints, coords)
```

**Pose keypoints**: MediaPipe Holistic (543 landmarks: 468 face + 33 body + 21×2 hands)

---

## 🚀 Data Augmentation Methodology

### 1. Text Augmentation with LLMs

**Location**: `src/augmentation/expand_db.ipynb`

The pipeline uses **Azure OpenAI GPT-5-mini** to generate semantically equivalent paraphrases:

```python
# Load from .env file
from dotenv import load_dotenv
import os

load_dotenv()

# Azure OpenAI Configuration
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT")  # gpt-5-mini

SYSTEM_PROMPT = """
You are a helpful assistant rewriting sentences.
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
TEMPERATURE = 0.8       # Controlled diversity
```

**Result**: RWTH-Phoenix training set expanded from 7,096 → 21,288 samples (3×)

### 2. Back-Translation Strategy (Thesis Plan)

**Pivot Language Policy**:
- For non-EN/DE targets → Two pivots: `t→EN→t` and `t→DE→t`
- For EN/DE targets → Spanish pivot: `EN→ES→EN` or `DE→ES→DE`

This creates systematic cross-lingual coupling to improve generalization.

### 3. Validation & Filtering

Generated variants are validated using:

| Metric | Purpose |
|--------|---------|
| **SBERT Cosine** | Semantic similarity threshold |
| **chrF/chrF++** | Surface-level distance control |
| **BERTScore** | Semantic verification |
| **Deduplication** | n-gram/minhash filtering |

### 4. Kinematic Pose Augmentation (Planned)

Perturbations applied to keypoint sequences:

- **Geometric**: Translation, scaling, rotation (plane)
- **Temporal**: Time warping, random sampling, padding
- **Dropout**: Keypoint occlusion with interpolation
- **Constraints**: Bone-length consistency validation

```python
# Plausibility check
φ(K) = (1/|B|T) Σ_t Σ_(p,q)∈B |‖p̂_t - q̂_t‖₂ - ℓ̄_pq|
# Accept if φ(K) ≤ τ (bone-length deviation threshold)
```

---

## 🏋️ Training Pipeline

### Using Signformer

```bash
cd signformer/

# 1. Build augmented dataset
python scripts/build_phoenix14t_ext.py \
    --data_dir PHOENIX2014T \
    --tsv train_aug.tsv \
    --out_prefix phoenix14t-ext.pami0

# 2. Train
python -m main train configs/sign.yaml --gpu_id 0

# 3. Evaluate
python -m main test configs/sign.yaml --ckpt path/to/best.ckpt
```

### Configuration (sign.yaml)

```yaml
data:
    data_path: ../PHOENIX2014T
    train: phoenix14t-ext.pami0.train    # Augmented
    dev: phoenix14t.pami0.dev
    test: phoenix14t.pami0.test
    feature_size: 1024                    # 512 keypoints × 2 coords

training:
    optimizer: sophiag
    learning_rate: 0.0004
    batch_size: 32
    epochs: 1000
    validation_freq: 100
    recognition_loss_weight: 0.0          # Translation-only (gloss-free)
    translation_loss_weight: 1.0

model:
    encoder:
        type: transformer
        num_layers: 1
        num_heads: 8
        hidden_size: 256
    decoder:
        type: transformer
        num_layers: 1
        num_heads: 8
```

### Using Custom Transformer (Alternative)

```bash
cd src/training/
jupyter notebook train.ipynb
```

---

## 📈 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **BLEU-4** | Primary metric (n-gram precision with brevity penalty) |
| **chrF/chrF++** | Character-level F-score (morphologically rich languages) |
| **ROUGE** | Recall-oriented summary evaluation |
| **BERTScore** | Semantic similarity via embeddings |

### Evaluation Protocols

1. **In-domain**: Train/test on same corpus
2. **Cross-domain**: Train on corpus A, test on corpus B
3. **Signer-held-out**: Exclude specific signers from training

---

## 📚 Key References

### Architecture
- **Signformer**: Yang, E. (2024). *Signformer is all you need: Towards Edge AI for Sign Language*. arXiv:2411.12901

### Datasets
- **PHOENIX-2014T**: Forster et al. (2014), Koller et al. (2015)
- **LSA-T**: Dal Bianco et al. (2022)
- **How2Sign**: Duarte et al. (CVPR 2021)
- **LSFB-CONT**: Fink et al. (2021)
- **ISLTranslate**: Joshi et al. (ACL Findings 2023)
- **GSL**: Adaloglou et al. (2020)

### Data Augmentation
- **Back-translation**: Sennrich et al. (2016), Edunov et al. (2018)
- **ParaNMT**: Wieting & Gimpel (2018)
- **LLM Data Aug**: Ding et al. (2024), ChatGPT DA (2023)
- **PoseAug**: Gong et al. (2021)

### Pose Extraction
- **MediaPipe Holistic**: Lugaresi et al. (2019)

---

## 🗂️ External Dependencies

### Required Repositories

| Repository | Purpose | Location |
|------------|---------|----------|
| `posecraft` | Pose manipulation library | `lib/posecraft/` (local copy) |
| `slt_datasets` | Unified dataset interface | `lib/slt_datasets/` (local copy) |

### Data Locations (Local Machine)

| Path | Content |
|------|---------|
| `/mnt/disk3Tb/slt-datasets/` | Raw formatted datasets |
| `/mnt/disk3Tb/augmented-slt-datasets/` | Text-augmented datasets |
| `/mnt/disk3Tb/exported-slt-datasets/` | Signformer format (`.pami0.*`) |

---

## 🔬 Research Questions

This project aims to answer:

1. **Does LLM-based text augmentation improve SLT performance?**
   - Compare original vs. augmented training sets
   - Measure BLEU/chrF improvements

2. **Can cross-lingual coupling via back-translation improve generalization?**
   - Test pivot language strategies (EN/DE/ES)
   - Evaluate on cross-domain protocols

3. **Do kinematic augmentations complement textual ones?**
   - Ablation studies with/without pose perturbations
   - Analyze robustness to occlusions and tracking noise

4. **Is a unified multilingual model viable?**
   - Train single model on 6 sign languages
   - Compare to language-specific baselines

---

## 📝 Notes for AI Agents

- **Pose data**: Always check for NaN values (tracking failures) before processing
- **Text normalization**: Apply Unicode NFKC + lowercase before augmentation
- **Signformer format**: Uses gzip-pickled lists; feature size is 1024 (512 keypoints × 2D)
- **Face keypoints**: Often removed for efficiency (reduces 543 → 75 landmarks)
- **Streaming**: Large datasets use `stream_train_parts: true` to avoid OOM
- **SophiaG optimizer**: After install, edit `sophia/__init__.py` to remove problematic import

---

*Last updated: January 2026*
