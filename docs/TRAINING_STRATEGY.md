# Training Strategy for Combined Sign Language Translation

This document describes the training strategy for the combined multi-corpus Sign Language Translation (SLT) experiments.

## Table of Contents

1. [Overview](#overview)
2. [Hardware Configuration](#hardware-configuration)
3. [Experiment Design](#experiment-design)
4. [Model Architecture](#model-architecture)
5. [Training Configuration](#training-configuration)
6. [Dataset Statistics](#dataset-statistics)
7. [Running the Training](#running-the-training)
8. [Monitoring](#monitoring)
9. [Expected Results](#expected-results)
10. [Troubleshooting](#troubleshooting)

---

## Overview

### Research Question

> Does data augmentation improve Sign Language Translation performance when training on multiple diverse datasets?

### Experimental Approach

1. **Vanilla Baseline**: Train on 5 datasets without augmentation
2. **Augmented Model**: Train on same 5 datasets with augmentation
3. **Comparison**: Evaluate both on held-out test sets

### Key Innovations

- **Multi-corpus training**: First time combining PHOENIX, GSL, LSAT, How2Sign, and LSFB-CONT
- **Feature unification**: Padding smaller datasets (134/150) to 1024 features
- **Streaming data loading**: Memory-efficient training on combined ~100k samples

---

## Hardware Configuration

### Target System

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA RTX 3090 (24GB VRAM) |
| RAM | 31GB |
| CPU | 4 cores |
| Storage | 3TB HDD (mounted at `/mnt/disk3Tb`) |

### Memory Budget

```
Total VRAM: 24GB
├── Model weights:      ~2GB
├── Optimizer states:   ~4GB
├── Activations:        ~8GB (batch_size=32)
├── Gradients:          ~2GB
└── Buffer:             ~8GB
```

### Recommended Batch Size

| VRAM | Batch Size | Batch Multiplier | Effective Batch |
|------|------------|------------------|-----------------|
| 24GB | 32 | 2 | 64 |
| 16GB | 16 | 4 | 64 |
| 12GB | 8 | 8 | 64 |

---

## Experiment Design

### Experiments

| Experiment | Config File | Description |
|------------|-------------|-------------|
| `combined-vanilla` | `combined_vanilla.yaml` | Baseline without augmentation |
| `combined-augmented` | `combined_augmented.yaml` | With text + pose augmentation |

### Control Variables

To ensure fair comparison, both experiments use:
- Same model architecture (3-layer encoder/decoder)
- Same optimizer (AdamW, lr=0.0003)
- Same batch size (32 × 2 = 64 effective)
- Same validation set (PHOENIX val)
- Same test sets (vanilla versions)

### Independent Variable

- **Vanilla**: Original training text
- **Augmented**: Original + paraphrases + back-translations

---

## Model Architecture

### Signformer Configuration

```yaml
encoder:
    type: transformer
    num_layers: 3
    num_heads: 8
    hidden_size: 512
    ff_size: 2048
    dropout: 0.2

decoder:
    type: transformer
    num_layers: 3
    num_heads: 8
    hidden_size: 512
    ff_size: 2048
    dropout: 0.2
```

### Architecture Rationale

| Choice | Rationale |
|--------|-----------|
| 3 layers | Balance between capacity and training speed |
| 512 hidden | Sufficient for multi-lingual vocabulary |
| 2048 FFN | Standard 4x hidden size ratio |
| 0.2 dropout | Higher than default (0.1) for multi-dataset regularization |

### Parameter Count

```
Encoder:
  - Embedding projection: 1024 × 512 = 524K
  - Self-attention (×3): 3 × (512² × 4) = 3.1M
  - FFN (×3): 3 × (512 × 2048 × 2) = 6.3M
  
Decoder:
  - Embedding: vocab_size × 512 ≈ 5M
  - Self-attention (×3): 3.1M
  - Cross-attention (×3): 3.1M
  - FFN (×3): 6.3M

Total: ~25-30M parameters (depending on vocabulary)
```

---

## Training Configuration

### Optimizer Settings

```yaml
optimizer: adamw
learning_rate: 0.0003
betas: [0.9, 0.999]
weight_decay: 0.01
```

**Why AdamW over SophiaG?**
- More stable for diverse multi-lingual data
- Better studied in translation literature
- SophiaG optional for fine-tuning later

### Learning Rate Schedule

```
Warmup: Linear increase over 2000-3000 steps
Main: Plateau scheduling
  - Patience: 15 validations without improvement
  - Decay factor: 0.7
  - Minimum LR: 1e-7
```

### Regularization

| Technique | Value | Purpose |
|-----------|-------|---------|
| Label smoothing | 0.1 | Prevent overconfidence |
| Dropout | 0.2 | Prevent overfitting |
| Weight decay | 0.01 | L2 regularization |

---

## Dataset Statistics

### Training Data (Vanilla)

| Dataset | Language | Samples | Features |
|---------|----------|---------|----------|
| RWTH-PHOENIX-2014T | German | ~7,000 | 1024 (native) |
| GSL | Greek | ~10,000 | 1024 (native) |
| LSAT | Spanish | ~15,000 | 1024 (native) |
| How2Sign-1024 | English | ~31,000 | 1024 (padded) |
| LSFB-CONT-1024 | French | ~4,400 | 1024 (padded) |
| **Total** | - | **~67,000** | - |

### Training Data (Augmented)

| Dataset | Original | Augmented | Multiplier |
|---------|----------|-----------|------------|
| RWTH-PHOENIX-2014T | 7,000 | ~21,000 | 3x |
| GSL | 10,000 | ~30,000 | 3x |
| LSAT | 15,000 | ~45,000 | 3x |
| How2Sign | 31,000 | ~87,000 | 2.8x |
| LSFB-CONT | 4,400 | ~14,400 | 3.3x |
| **Total** | 67,000 | **~197,000** | 2.9x |

### Validation/Test Sets

Always use vanilla (non-augmented) for fair evaluation:

| Split | Dataset | Samples |
|-------|---------|---------|
| Val | PHOENIX | ~500 |
| Test | All 5 datasets | ~5,000 total |

---

## Running the Training

### Prerequisites

```bash
# Ensure experiments directory exists
mkdir -p /mnt/disk3Tb/signformer-experiments/combined-vanilla
mkdir -p /mnt/disk3Tb/signformer-experiments/combined-augmented

# Check GPU
nvidia-smi
```

### Build and Start

```bash
cd ~/Github/signlanguage-translation-augmented

# Build training image
docker compose -f docker/docker-compose.training.yml build

# Start TensorBoard (background)
docker compose -f docker/docker-compose.training.yml up -d tensorboard

# Run vanilla baseline
docker compose -f docker/docker-compose.training.yml run --rm train-vanilla
```

### Training Timeline

| Phase | Duration | Notes |
|-------|----------|-------|
| Vanilla training | 2-3 days | ~67k samples |
| Augmented training | 3-4 days | ~197k samples |
| Total | 5-7 days | Sequential (single GPU) |

### Resume Training

If training is interrupted:

```bash
# Edit config to set:
#   overwrite: false
#   reset_best_ckpt: false
#   reset_scheduler: false
#   reset_optimizer: false

# Re-run the same command
docker compose -f docker/docker-compose.training.yml run --rm train-vanilla
```

---

## Monitoring

### TensorBoard

Access at: **http://localhost:6006**

Metrics tracked:
- `train/loss` - Training loss per step
- `valid/bleu` - BLEU score on validation set
- `valid/loss` - Validation loss
- `learning_rate` - Current learning rate

### Log Files

```bash
# Real-time training log
tail -f /mnt/disk3Tb/signformer-experiments/combined-vanilla/train.log

# GPU utilization
watch -n 1 nvidia-smi

# Memory usage
watch -n 1 free -h
```

### Key Metrics to Watch

| Metric | Healthy Range | Action if Outside |
|--------|---------------|-------------------|
| GPU Memory | 80-95% | Reduce batch size |
| GPU Utilization | 80-100% | Check data loading |
| Training Loss | Decreasing | Check learning rate |
| Validation BLEU | Increasing | Normal |

---

## Expected Results

### Baseline Expectations

Based on Signformer paper and prior experiments:

| Dataset | Expected BLEU (Vanilla) |
|---------|------------------------|
| PHOENIX-2014T | 20-24 |
| GSL | 15-20 |
| LSAT | 10-15 |
| How2Sign | 8-12 |
| LSFB-CONT | 5-10 |

### Augmentation Impact

Literature suggests 2-5 BLEU improvement from augmentation:

| Dataset | Vanilla | Augmented (Expected) |
|---------|---------|---------------------|
| PHOENIX-2014T | 22 | 24-26 |
| GSL | 17 | 19-21 |
| LSAT | 12 | 14-16 |

---

## Troubleshooting

### Out of Memory (OOM)

```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Reduce `batch_size` in config (e.g., 32 → 16)
2. Increase `batch_multiplier` proportionally (2 → 4)
3. Reduce `stream_chunk_size` (3000 → 2000)

### Slow Training

**Symptoms:** GPU utilization < 50%

**Solutions:**
1. Increase `stream_chunk_size` to reduce I/O
2. Check disk I/O: `iostat -x 1`
3. Ensure data is on fast storage

### NaN Loss

```
Loss: nan
```

**Solutions:**
1. Reduce learning rate (0.0003 → 0.0001)
2. Increase warmup steps (2000 → 4000)
3. Check for corrupted data files

### Validation BLEU Stuck at 0

**Cause:** Vocabulary mismatch or data loading issue

**Solutions:**
1. Check `txt.vocab` is being built correctly
2. Verify validation data paths
3. Try with single dataset first

---

## File Reference

| File | Purpose |
|------|---------|
| `docker/docker-compose.training.yml` | Training orchestration |
| `signformer/Dockerfile` | Training image |
| `signformer/configs/combined_vanilla.yaml` | Vanilla baseline config |
| `signformer/configs/combined_augmented.yaml` | Augmented config |
| `docs/TRAINING_STRATEGY.md` | This document |

---

*Last updated: January 2026*
