# Resource Constraints and Optimization

This document details the hardware limitations and optimization strategies for the SLT training environment.

## Critical Status

```
┌─────────────────────────────────────────────────────────────┐
│  ⚠️  DISK: 99% FULL - ONLY 29GB FREE OF 2.7TB             │
│  ✅  GPU: RTX 3090 24GB - Adequate                         │
│  ✅  RAM: 31GB - Adequate                                  │
│  ⚠️  CPU: 4 cores (i5-7400) - Limited                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. Disk Space (CRITICAL)

### Current Usage

| Path | Size | Purpose |
|------|------|---------|
| `/mnt/disk3Tb/slt-datasets/` | **1.9TB** | Raw video/pose data |
| `/mnt/disk3Tb/medical-imaging-datasets/` | 403GB | Unrelated project |
| `/mnt/disk3Tb/exported-slt-datasets/` | 206GB | Processed .pami0 files |
| `/mnt/disk3Tb/uv-cache/` | 40GB | Python package cache |
| **Free space** | **29GB** | ⚠️ CRITICAL |

### Disk Space Breakdown - Exported Datasets

| File | Size | Status |
|------|------|--------|
| `MERGE.pami0.*` | ~86GB | ❌ OLD - Can delete |
| `ISL-MERGED.pami0.*` | ~78GB | ❌ OLD - Can delete |
| `RWTH_PHOENIX_2014T_EXT.pami0.train` | 7.8GB | ❓ May be redundant |
| Individual datasets | ~35GB | ✅ NEEDED |
| **Potential savings** | **~170GB** | If old merges deleted |

### Immediate Actions Required

```bash
# Check what's using space
du -sh /mnt/disk3Tb/exported-slt-datasets/MERGE* /mnt/disk3Tb/exported-slt-datasets/ISL-MERGED*

# If safe to delete (verify first!):
# rm /mnt/disk3Tb/exported-slt-datasets/MERGE.pami0.*
# rm /mnt/disk3Tb/exported-slt-datasets/ISL-MERGED.pami0.*
# This would free ~164GB
```

### Training Space Requirements

| Item | Space Needed |
|------|--------------|
| Checkpoints (3 kept) | ~2GB |
| TensorBoard logs | ~500MB |
| Vocabulary files | ~10MB |
| Training logs | ~100MB |
| **Total per experiment** | **~3GB** |

With 29GB free, we can run experiments but:
- Cannot store many checkpoint histories
- May need to clean up between experiments

---

## 2. GPU Memory (24GB VRAM)

### RTX 3090 Specifications

| Spec | Value |
|------|-------|
| VRAM | 24GB GDDR6X |
| CUDA Cores | 10,496 |
| Memory Bandwidth | 936 GB/s |
| TDP | 350W |

### Memory Budget During Training

```
┌─────────────────────────────────────────────────────────┐
│                    24GB VRAM Budget                      │
├─────────────────────────────────────────────────────────┤
│ Model weights           │  ~2GB   │ 3-layer transformer │
│ Optimizer states        │  ~4GB   │ AdamW momentum      │
│ Gradients               │  ~2GB   │ Backward pass       │
│ Activations (batch=32)  │  ~8GB   │ Forward pass        │
│ CUDA workspace          │  ~4GB   │ cuDNN, etc.         │
├─────────────────────────────────────────────────────────┤
│ TOTAL                   │ ~20GB   │ 4GB headroom        │
└─────────────────────────────────────────────────────────┘
```

### Batch Size Recommendations

| Batch Size | VRAM Usage | Safe? |
|------------|------------|-------|
| 64 | ~28GB | ❌ OOM risk |
| 48 | ~24GB | ⚠️ Borderline |
| **32** | **~20GB** | **✅ Recommended** |
| 16 | ~14GB | ✅ Conservative |

### If OOM Occurs

```yaml
# In config, reduce batch_size and increase batch_multiplier:
batch_size: 16           # Reduced from 32
batch_multiplier: 4      # Increased from 2
# Effective batch stays 64, but uses less VRAM per step
```

---

## 3. System RAM (31GB)

### RAM Budget

```
┌─────────────────────────────────────────────────────────┐
│                    31GB RAM Budget                       │
├─────────────────────────────────────────────────────────┤
│ OS + System             │  ~2GB   │ Linux overhead      │
│ Docker overhead         │  ~1GB   │ Container runtime   │
│ Data loading buffer     │  ~8GB   │ stream_chunk_size   │
│ Python process          │  ~4GB   │ Signformer code     │
│ TensorBoard             │  ~1GB   │ Monitoring          │
├─────────────────────────────────────────────────────────┤
│ TOTAL                   │ ~16GB   │ 15GB headroom       │
└─────────────────────────────────────────────────────────┘
```

### Streaming Configuration

The `stream_chunk_size` parameter controls RAM usage during data loading:

| Chunk Size | RAM for Data | Recommendation |
|------------|--------------|----------------|
| 5000 | ~10GB | ⚠️ High |
| **3000** | **~6GB** | **✅ Default** |
| 2000 | ~4GB | Conservative |
| 1000 | ~2GB | Very conservative |

### Why Streaming is Critical

Without streaming, loading all training data would require:

```
67,000 samples × ~2MB avg = ~134GB RAM  ❌ Impossible
```

With streaming (chunk_size=3000):
```
3,000 samples × ~2MB avg = ~6GB RAM     ✅ Feasible
```

---

## 4. CPU (4 Cores)

### i5-7400 Specifications

| Spec | Value |
|------|-------|
| Cores | 4 |
| Threads | 4 (no hyperthreading) |
| Base Clock | 3.0 GHz |
| Generation | 7th Gen (2017) |

### CPU Bottlenecks

| Task | CPU Impact | Mitigation |
|------|------------|------------|
| Data loading | High | Pre-fetch with streaming |
| Tokenization | Medium | Batch processing |
| Logging | Low | Reduce logging_freq |
| Checkpointing | Medium | Reduce keep_last_ckpts |

### Optimization for 4 Cores

```yaml
# In training config, reduce CPU-intensive operations:
logging_freq: 100        # Less frequent logging
validation_freq: 1500    # Less frequent validation
num_valid_log: 5         # Fewer examples to log
```

### Data Loading Workers

PyTorch DataLoader workers (not directly in Signformer, but if modified):

```python
# With 4 cores, use 2-3 workers max
num_workers: 2  # Leave cores for training
```

---

## 5. Optimization Summary

### Configuration for Resource-Constrained Training

```yaml
# Optimized for: RTX 3090, 31GB RAM, 4 cores, 29GB disk

data:
    stream_train_parts: true
    stream_chunk_size: 2500      # Reduced for RAM
    
training:
    batch_size: 32               # Safe for 24GB VRAM
    batch_multiplier: 2          # Effective batch = 64
    keep_last_ckpts: 2           # Save disk space
    logging_freq: 100            # Reduce CPU load
    validation_freq: 1500        # Reduce CPU load
```

### Before Training Checklist

```bash
# 1. Check disk space
df -h /mnt/disk3Tb
# Must have > 10GB free

# 2. Check GPU
nvidia-smi
# Must show RTX 3090 with ~24GB free

# 3. Check RAM
free -h
# Must show > 20GB available

# 4. Clean old experiments if needed
rm -rf /mnt/disk3Tb/signformer-experiments/old-experiment/
```

---

## 6. Cleanup Recommendations

### Safe to Delete (After Verification)

| Path | Size | Reason |
|------|------|--------|
| `MERGE.pami0.*` | 86GB | Old merged dataset |
| `ISL-MERGED.pami0.*` | 78GB | Old merged dataset |
| `/mnt/disk3Tb/uv-cache/` | 40GB | Package cache (regenerable) |

### Keep (Required)

| Path | Size | Reason |
|------|------|--------|
| `RWTH_PHOENIX_2014T*.pami0.*` | ~6GB | PHOENIX dataset |
| `GSL*.pami0.*` | ~3GB | GSL dataset |
| `LSAT*.pami0.*` | ~5GB | LSAT dataset |
| `How2Sign-1024*.pami0.*` | ~5GB | How2Sign (padded) |
| `LSFB-CONT-1024*.pami0.*` | ~2GB | LSFB-CONT (padded) |

### Cleanup Command (USE WITH CAUTION)

```bash
# Free ~164GB by removing old merged files
# VERIFY THESE ARE NOT NEEDED FIRST!

# Check what would be deleted
ls -lh /mnt/disk3Tb/exported-slt-datasets/MERGE.pami0.*
ls -lh /mnt/disk3Tb/exported-slt-datasets/ISL-MERGED.pami0.*

# If confirmed safe:
# rm /mnt/disk3Tb/exported-slt-datasets/MERGE.pami0.*
# rm /mnt/disk3Tb/exported-slt-datasets/ISL-MERGED.pami0.*
```

---

## 7. Resource Monitoring During Training

### Real-time Monitoring Commands

```bash
# Terminal 1: GPU
watch -n 1 nvidia-smi

# Terminal 2: RAM and CPU
watch -n 1 'free -h && echo "" && top -bn1 | head -15'

# Terminal 3: Disk
watch -n 60 'df -h /mnt/disk3Tb'

# Terminal 4: Training logs
tail -f /mnt/disk3Tb/signformer-experiments/combined-vanilla/train.log
```

### Alert Thresholds

| Resource | Warning | Critical | Action |
|----------|---------|----------|--------|
| Disk | < 10GB | < 5GB | Stop training, cleanup |
| GPU Mem | > 22GB | > 23GB | Reduce batch size |
| RAM | > 28GB | > 30GB | Reduce chunk_size |
| CPU | 100% sustained | - | Normal during training |

---

*Last updated: January 2026*
