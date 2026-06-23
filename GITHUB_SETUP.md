# GitHub Repository Metadata

Use the following information to setup your GitHub repository.

## Repository Description
`A Pangu-Lite inspired weather forecasting architecture built for distributed HPC training, featuring cross-attention and U-Net hybrid experiments.`

## GitHub Topics (Tags)
`weather-forecasting` `deep-learning` `pytorch` `transformers` `distributed-training` `hpc` `climate-tech` `neural-networks` `xarray` `earth-science`

## Project Highlights (For Resume / Portfolio)
- **Engineered an Advanced Weather Forecasting Model:** Replicated and extended the Pangu-Lite architecture, introducing cross-attention mechanisms and U-Net skip connections to enhance fine-grained meteorological predictions.
- **Optimized for High-Performance Computing (HPC):** Implemented multi-node, multi-GPU distributed training using PyTorch `DistributedDataParallel` (DDP) and Automatic Mixed Precision (AMP).
- **Efficient Data Pipeline:** Designed a robust dataset ingestion pipeline using `xarray` capable of efficiently handling terabytes of monthly ERA5, GFS, and IMDAA reanalysis data without I/O bottlenecks.
- **Architecture Exploration:** Conducted extensive ablation studies comparing standard self-attention against hybrid TransFusion models to improve predictive skill scores on surface and upper-air variables.

## Suggested Release Notes (v1.0.0)

**Title:** Train14: Initial Release - Core Architecture & HPC Workflows

**Description:**
We are excited to announce the initial release of **Train14**, a research repository dedicated to advanced data-driven weather forecasting architectures. 

**Key Features:**
- **Baseline Implementations:** Includes ERA5, GFS, and IMDAA dataset adaptations of the Pangu-Lite architecture.
- **Novel Architectures:** Introduces the Cross-Attention variant and the TransFusion-UNet hybrid for improved spatial feature retention.
- **Distributed Training Ready:** Full PyTorch DDP integration with example PBS/SLURM scripts for immediate HPC deployment.
- **Clean, Sanitized Codebase:** Refactored for readability, removing environment-specific hardcoded paths to ensure reproducibility across different clusters.

*Note: Pre-trained weights and final evaluation metrics will be released in subsequent updates following the conclusion of ongoing experiments.*
