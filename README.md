# Sign Language Translation with Data Augmentation

This repository contains a complete pipeline for Sign Language Translation (SLT) with text augmentation using LLMs.

## Important: Local Libraries

This repository includes local copies of two essential packages in `lib/`:
- **posecraft** - Library for manipulating pose keypoints
- **slt_datasets** - Multilanguage datasets for Sign Language Translation

These are automatically added to the Python path in all source files.

## Repository Structure

```
signlanguage-translation-augmented/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── lib/                               # LOCAL LIBRARY PACKAGES (backups)
│   ├── posecraft/                     # Pose keypoint manipulation
│   │   ├── Pose.py                    # Pose class
│   │   ├── transforms.py              # Data transforms
│   │   └── interpolate.py             # Frame interpolation
│   └── slt_datasets/                  # SLT dataset utilities
│       ├── SLTDataset.py              # Dataset class
│       └── WordLevelTokenizer.py      # Tokenization
│
├── src/                               # Main source code
│   ├── augmentation/                  # Text augmentation with LLM
│   │   ├── expand_db.ipynb            # GPT-4o-mini paraphrasing
│   │   └── train_aug.tsv              # Sample augmented data
│   ├── export/                        # Dataset export utilities
│   │   ├── export_db.ipynb            # Export to Signformer format
│   │   └── export_db.py               # Export script
│   ├── training/                      # Custom transformer training
│   │   ├── train.ipynb                # PyTorch Lightning training
│   │   ├── KeypointsTransformer.py    # Transformer model
│   │   ├── LightningKeypointsTransformer.py
│   │   ├── Translator.py              # Translation utilities
│   │   ├── WordLevelTokenizer.py      # Tokenization
│   │   └── results_analysis.ipynb     # Analyze results
│   ├── config/                        # Hyperparameter configs
│   │   ├── GSL.json
│   │   ├── LSAT.json
│   │   └── RWTH_PHOENIX_2014T.json
│   ├── interp/                        # Interpolation modules
│   ├── helpers.py                     # Utility functions
│   ├── hyperparameters.py             # Hyperparameter loading
│   └── datasets.json                  # Dataset registry
│
├── signformer/                        # Signformer model (production)
│   ├── main/                          # Core modules
│   │   ├── training.py                # Training loop
│   │   ├── data.py                    # Data loading
│   │   ├── dataset.py                 # SignTranslationDataset
│   │   ├── model.py                   # Model architecture
│   │   ├── prediction.py              # Inference
│   │   └── ...                        # Other modules
│   ├── configs/                       # YAML configurations
│   │   ├── sign.yaml                  # Main config
│   │   ├── sign_finetune.yaml         # Finetuning config
│   │   └── ...
│   ├── scripts/                       # Utility scripts
│   │   └── build_phoenix14t_ext.py    # Build augmented dataset
│   ├── requirements.txt               # Signformer-specific deps
│   ├── Dockerfile                     # Docker setup
│   └── docker-compose.yml
│
├── dataset_formatting/                # Dataset-specific formatting
│   ├── rwth/                          # RWTH-Phoenix-2014T
│   ├── gsl/                           # Greek Sign Language
│   ├── lsat/                          # LSA-T (Argentinian)
│   ├── isl/                           # Indian Sign Language
│   ├── howtosign/                     # How2Sign
│   ├── lsfb-cont/                     # LSFB Continuous
│   └── content4all/                   # Content4All dataset
│
├── explanations/                      # Documentation
│   └── SLT_PIPELINE_DOCUMENTATION.md  # Complete pipeline docs
│
├── plans/                             # Experiment plans
│
└── docs/                              # Thesis and papers
    └── thesis/
```

## Quick Start

### 1. Setup with uv (Recommended)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Quick setup with the setup script
./scripts/setup.sh --all

# OR manual setup:
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[all]"
```

### Optional Dependency Groups

```bash
# Base only (core ML dependencies)
uv pip install -e .

# With LLM augmentation (OpenAI)
uv pip install -e ".[augmentation]"

# With Signformer training (SophiaG optimizer)
uv pip install -e ".[signformer]"

# With Jupyter notebooks
uv pip install -e ".[notebooks]"

# With TensorFlow for dataset loading
uv pip install -e ".[tfds]"

# All optional dependencies
uv pip install -e ".[all]"

# Development dependencies
uv pip install -e ".[dev]"
```

### Alternative: pip install

```bash
# Create virtualenv manually
python3.11 -m venv .venv
source .venv/bin/activate

# Install with pip
pip install -r requirements.txt

# For Signformer specifically:
cd signformer
pip install -r requirements.txt
```

### Import Structure

The `lib/` directory contains local copies of the `posecraft` and `slt_datasets` packages.
All Python files in `src/` automatically add `lib/` to `sys.path`:

```python
# Example from src/hyperparameters.py
import sys, os
_lib_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "lib")
if _lib_path not in sys.path:
    sys.path.insert(0, _lib_path)

# Now imports work:
from slt_datasets.SLTDataset import SLTDataset
from posecraft.Pose import Pose
from posecraft.transforms import FilterLandmarks, PadTruncateFrames
```

### 2. Data Augmentation

```bash
cd src/augmentation
jupyter notebook expand_db.ipynb
```

Configure:
- `DATASET`: Target dataset (e.g., "RWTH_PHOENIX_2014T")
- `OPENAI_API_KEY`: Your OpenAI API key
- `AUG_FACTOR`: Number of paraphrases per sample (default: 2)

### 3. Export Dataset for Signformer

```bash
cd src/export
jupyter notebook export_db.ipynb
```

### 4. Build Augmented Dataset

```bash
cd signformer
python scripts/build_phoenix14t_ext.py \
    --data_dir /path/to/PHOENIX2014T \
    --tsv train_aug.tsv \
    --out_prefix phoenix14t-ext.pami0
```

### 5. Train Signformer

```bash
cd signformer
python -m main train configs/sign.yaml --gpu_id 0
```

### 6. Evaluate

```bash
python -m main test configs/sign.yaml --ckpt path/to/best.ckpt
```

## Data Locations

| Purpose | Default Path |
|---------|--------------|
| Raw datasets | `/mnt/disk3Tb/slt-datasets/` |
| Augmented datasets | `/mnt/disk3Tb/augmented-slt-datasets/` |
| Exported datasets | `/mnt/disk3Tb/exported-slt-datasets/` |

## Available Datasets

- **RWTH-Phoenix-2014T** (German weather news)
- **GSL** (Greek Sign Language)
- **LSAT/LSA-T** (Argentinian Sign Language)
- **ISL** (Indian Sign Language)
- **How2Sign** (American Sign Language)
- **LSFB-CONT** (French Belgian Sign Language)
- **Content4All** (Swiss/Flemish Sign Languages)

## Pipeline Overview

```
Raw Video/Poses → Format to SLTDataset → Text Augmentation (LLM)
                                              ↓
                                    Export to Signformer format
                                              ↓
                                    Train Signformer model
                                              ↓
                                    Evaluate (BLEU, CHRF, ROUGE)
```

## Key References

- [Signformer Paper](https://arxiv.org/abs/2411.12901v1)
- Full pipeline documentation: `explanations/SLT_PIPELINE_DOCUMENTATION.md`

## Citation

```bibtex
@article{eta2024signformer,
    title={Signformer is all you need: Towards Edge AI for Sign Language}, 
    author={Eta Yang},
    year={2024},
    journal={arXiv preprint arXiv:2411.12901}, 
}
```
