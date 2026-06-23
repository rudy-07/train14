# Train14: Advanced Weather Forecasting Architecture Exploration

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)

> **Note:** This repository is part of a larger, two-part research endeavor in weather AI. While Part 1 focuses on fine-tuning Huawei's original Pangu-Weather model on regional datasets, **this repository (Part 2)** is dedicated to the **Train14** architecture—a Pangu-Lite inspired implementation focused on architecture experimentation, distributed training, and model scaling.

## 📖 Overview

Train14 is a research and engineering showcase for advanced data-driven weather forecasting architectures. It stems from the motivation to understand, replicate, and extend the capabilities of state-of-the-art transformer-based weather models while optimizing for HPC workflows and distributed training paradigms.

The core implementation is heavily inspired by publicly available implementations from **[zhaoshan2](https://github.com/zhaoshan2/Pangu-Weather)** and **[lizhouq](https://github.com/lizhouq/Pangu-Weather-lite)**. We acknowledge their foundational work and build upon it to introduce novel architectural variants and engineering improvements.

### Why Train14?

1. **Research Motivation**: To push the boundaries of current weather AI by experimenting with different attention mechanisms, U-Net hybridizations, and dataset adaptations.
2. **Engineering Excellence**: To build a scalable, distributed, and clean codebase capable of running on multi-node GPU clusters using PyTorch DDP.
3. **Architecture Exploration**: To systematically evaluate the performance trade-offs of various architectural modifications.

---

## 🏗 Architecture & Variants

Train14 is not a single model but a family of experimental architectures designed for different datasets and research goals. 

### Core Variants

- **Train14 ERA5** (`src/train14_era5.py`): The baseline Pangu-Lite implementation adapted for the global ERA5 reanalysis dataset.
- **Train14 GFS** (`src/train14_gfs.py`): Adapted to ingest and train on the GFS (Global Forecast System) dataset.
- **Train14 IMDAA** (`src/train14_imdaa.py`): A localized variant tailored for the IMDAA high-resolution regional reanalysis dataset.

### Experimental Variants

- **Cross-Attention Variant** (`src/train14_cross_attention.py`): Introduces cross-attention mechanisms to better fuse surface and upper-air atmospheric variables, breaking the standard self-attention bottleneck.
- **Train14-Innovate** (`src/train14_innovate_era5.py`): An experimental architecture that heavily modifies the embedding and downsampling layers for improved gradient flow and feature extraction.
- **TransFusion UNet Variant** (`src/train14_transfusion_unet.py`): A hybrid architecture that merges Transformer-based attention blocks with a U-Net style skip-connection hierarchy to preserve fine-grained spatial details.

For detailed architectural diagrams and design choices, please refer to [docs/architecture.md](docs/architecture.md).

---

## ⚡ Distributed Training & Scalability

Train14 is engineered from the ground up for High-Performance Computing (HPC) environments.

- **DistributedDataParallel (DDP)**: Full support for multi-node, multi-GPU training.
- **Automatic Mixed Precision (AMP)**: Optimizes VRAM usage and accelerates training without sacrificing numerical stability.
- **Efficient Data Loading**: Custom `xarray` and `netCDF4` based data loaders designed to minimize NFS/Lustre I/O bottlenecks using intelligent caching and worker prefetching.

Example SLURM/PBS scripts for distributed training can be found in the `examples/` directory.

---



---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- PyTorch 2.0+ (with CUDA support)
- xarray, netCDF4, numpy, pandas, timm

### Repository Structure

```
├── configs/             # Training and model configurations
├── docs/                # Detailed documentation and architecture diagrams
├── examples/            # Example PBS/SLURM batch scripts
├── images/              # Generated architecture and workflow diagrams
├── src/                 # Core Train14 source code (ERA5, GFS, IMDAA, etc.)
├── README.md            # Repository overview (this file)
└── CITATION.cff         # Academic citation information
```

---

## 🔮 Future Work

- Expanding the TransFusion UNet variant to higher resolutions.
- Incorporating temporal sequence modeling (e.g., auto-regressive training loops) directly into the DDP pipeline.
- Ablation studies on the Cross-Attention feature fusion.

## 🙌 Acknowledgements

We extend our deep gratitude to the open-source weather AI community, specifically:
- **zhaoshan2** for the initial Pangu-Weather PyTorch insights.
- **lizhouq** for the structural foundations of Pangu-Weather-Lite.

*Note: This repository does not claim ownership of their original work but builds upon it for academic and architectural exploration.*
