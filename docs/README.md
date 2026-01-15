# Documentation

This folder contains documentation for the Sign Language Translation (SLT) Augmentation project.

## Contents

### Technical Documentation

| Document | Description |
|----------|-------------|
| [DATASET_EXPORT_1024.md](./DATASET_EXPORT_1024.md) | Exporting How2Sign and LSFB-CONT datasets with 1024-feature padding |

### Thesis

The `thesis/` folder contains LaTeX files for the academic thesis associated with this project.

## Related Documentation

- **[../explanations/SLT_PIPELINE_DOCUMENTATION.md](../explanations/SLT_PIPELINE_DOCUMENTATION.md)** - Full pipeline documentation
- **[../README.md](../README.md)** - Project overview
- **[../agents.md](../agents.md)** - Agent context for AI assistants

## Quick Links

### Data Flow

```
Raw Datasets          →    Augmentation    →    Export (.pami0)    →    Training
/mnt/disk3Tb/              docker/              exported-slt-        Signformer/
slt-datasets/              augmentation/        datasets/
```

### Key Directories

| Path | Purpose |
|------|---------|
| `src/augmentation/` | Text augmentation (paraphrasing, back-translation) |
| `src/export/` | Dataset export scripts |
| `docker/` | Docker configurations for isolated execution |
| `signformer/` | Training configurations and scripts |

### Dataset Feature Sizes

| Dataset | Original Features | Exported Features |
|---------|------------------|-------------------|
| RWTH-PHOENIX-2014T | 1024 | 1024 (native) |
| GSL | 1024 | 1024 (native) |
| LSAT | 1024 | 1024 (native) |
| How2Sign | 134 | **1024** (padded) |
| LSFB-CONT | 150 | **1024** (padded) |
