# Train14: Advanced Weather Forecasting Architecture Exploration

This repository contains the codebase and architectural implementation details for the Train14 project. Train14 is an advanced, exploratory weather forecasting model inspired by the Pangu-Weather architecture. 

## Repository Scope

This repository documents the architectural evolution, engineering implementations, and distributed training infrastructure of the Train14 model variants. It represents the second major phase of our weather AI research effort. While Phase 1 focused on finetuning an existing global model on regional datasets, this Phase 2 repository focuses on building and modifying the underlying neural network architectures from the ground up to solve specific forecasting bottlenecks.

## Table of Contents
1. [Project Highlights](#1-project-highlights)
2. [Key Contributions](#2-key-contributions)
3. [Research Motivation](#3-research-motivation)
4. [What Problem Train14 Solves](#4-what-problem-train14-solves)
5. [Architecture Evolution Timeline](#5-architecture-evolution-timeline)
6. [Engineering Challenges](#6-engineering-challenges)
7. [Distributed Training and HPC](#7-distributed-training-and-hpc)
8. [Experimental Variants](#8-experimental-variants)
9. [Architecture Comparison Table](#9-architecture-comparison-table)
10. [Future Work](#10-future-work)
11. [Acknowledgements](#11-acknowledgements)

## 1. Project Highlights
* **Architectural Innovation:** Developed multiple experimental variants of the Pangu-Lite architecture, including Cross-Attention and TransFusion-UNet models.
* **Dataset Adaptability:** Successfully adapted the core architecture to ingest and train on diverse atmospheric datasets including global ERA5, NCEP GFS, and the high-resolution regional IMDAA dataset.
* **HPC Optimization:** Engineered a highly scalable distributed training pipeline utilizing PyTorch DistributedDataParallel (DDP) across multi-node GPU clusters.
* **Efficient I/O:** Built custom xarray and netCDF4 data loaders designed to eliminate Lustre and NFS storage bottlenecks during distributed training.

## 2. Key Contributions
Key contributions made during the development of Train14 include:
* Implemented a fully functional, lightweight Earth-Specific Transformer architecture capable of running on accessible academic hardware.
* Designed a novel Cross-Attention mechanism to fuse surface and upper-air atmospheric states, preventing deep self-attention layers from washing out localized features.
* Engineered the TransFusion-UNet, a hybrid architecture combining U-Net skip connections with transformer blocks to preserve high-frequency spatial details like sharp weather fronts.
* Developed a robust preprocessing and training pipeline that normalizes and dynamically loads massive gridded datasets on the fly.
* Resolved critical memory constraints and gradient flow issues inherent to training deep 3D-transformer models on high-resolution meteorological data.

## 3. Research Motivation
The primary motivation behind Train14 was to push beyond the limitations of standard global forecasting models. While existing models perform exceptionally well on global scales, they often struggle with high-resolution, localized weather phenomena. 

We recognized that standard self-attention mechanisms might not be optimal for capturing the complex, multi-scale dynamics of the atmosphere, particularly the interactions between surface variables and upper-air pressure levels. Train14 was conceived as a modular testbed to systematically explore these architectural bottlenecks and develop new neural network components tailored specifically for atmospheric physics.

## 4. What Problem Train14 Solves
Train14 addresses several critical challenges in AI-driven weather forecasting:
* **The Resolution Bottleneck:** Standard transformers scale quadratically with grid resolution. Train14 explores convolutional embeddings and U-Net hierarchies to handle higher resolutions without catastrophic memory scaling.
* **Feature Washing:** In deep transformer layers, distinct surface features can be smoothed out by dominant upper-air patterns. The Cross-Attention variant explicitly solves this by maintaining separate query pathways.
* **Domain Adaptation:** The repository provides a generalized framework that allows a single core architecture to be trained on completely different grid structures (ERA5 vs GFS vs IMDAA) with minimal friction.

## 5. Architecture Evolution Timeline
The development of Train14 proceeded through several distinct stages:
1. **Baseline Replication:** Implemented the base Pangu-Lite architecture to establish a working, reproducible pipeline on the global ERA5 reanalysis dataset.
2. **Dataset Adaptation:** Modified the data ingestion layers to support the NCEP GFS dataset, proving the model's generalizability across different coordinate systems and variable conventions.
3. **Regional High-Resolution:** Adapted the model for the IMDAA regional dataset, necessitating significant memory optimizations and structural adjustments to handle the localized high-resolution grid.
4. **Architectural Branching:** Developed the Train14-Innovate and Cross-Attention variants to address gradient flow and feature fusion limitations observed during baseline training.
5. **Hybridization:** Designed the TransFusion-UNet variant, representing the culmination of the project by merging the best spatial properties of convolutions with the global receptive fields of transformers.

## 6. Engineering Challenges
Building Train14 required overcoming significant engineering hurdles:
* **I/O Bottlenecks:** Loading terabytes of NetCDF files during distributed training caused severe NFS storms, bringing cluster file systems to a halt. We solved this by implementing intelligent rank-0 pre-scanning, aggressive local caching, and optimized worker prefetching.
* **GPU Memory Constraints:** 3D atmospheric tensors are exceptionally large. We implemented Automatic Mixed Precision (AMP) and heavily optimized our patchification and un-patchification layers to fit the models within standard VRAM limits.
* **Distributed Synchronization:** Ensuring stable gradient all-reduce operations across multiple nodes required careful tuning of PyTorch DDP parameters and network interfaces to prevent collective timeouts during massive epoch iterations.

## 7. Distributed Training and HPC
Train14 is engineered for execution on High-Performance Computing clusters. 
* **DistributedDataParallel (DDP):** The codebase fully supports multi-node, multi-GPU training.
* **Cluster Integration:** Provided PBS and SLURM scripts demonstrate how to automatically detect network interfaces, allocate master ports, and launch robust torchrun instances across diverse node topologies.
* **Scalability:** The training loop is designed to scale linearly with available GPUs, automatically adjusting batch distribution and sampler synchronization without user intervention.

## 8. Experimental Variants
The repository contains several distinct architectural branches, each designed to test a specific research hypothesis.

### Train14 ERA5
The foundational baseline model. It utilizes standard 3D Earth-Specific Positional Bias and self-attention over the global ERA5 grid.

### Train14 GFS
An adaptation of the baseline model designed to ingest GFS data. This required rewriting the data loading mechanisms to handle the specific variable packings and coordinate structures of the GFS format.

### Train14 IMDAA
A regional variant tailored for the high-resolution IMDAA dataset. This variant required architectural adjustments to the patch embedding layers to accommodate the non-global geographic boundaries and higher grid density.

### Cross-Attention Variant
This architecture breaks the traditional self-attention mold. It separates surface variables and upper-air variables into distinct processing streams, using cross-attention modules to allow the surface states to selectively query information from the upper atmosphere.

### Train14 Innovate
An experimental approach that introduces convolutional downsampling layers before the transformer blocks. This acts as a strong spatial inductive bias, allowing the model to capture local high-frequency details before applying global attention.

### TransFusion UNet Variant
A hybrid architecture combining a U-Net style encoder-decoder structure with transformer bottleneck layers. Skip connections pass high-resolution spatial information directly to the output layers, preserving sharp weather features that are often blurred by pure transformer networks.

## 9. Architecture Comparison Table

| Variant | Dataset | Key Modification | Research Hypothesis |
|---------|---------|------------------|---------------------|
| Train14 ERA5 | Global ERA5 | None (Baseline) | Establish a strong, reproducible baseline. |
| Train14 GFS | NCEP GFS | Adapted coordinate loading | Validate architecture generalizes beyond ECMWF datasets. |
| Train14 IMDAA | IMDAA Regional | High-resolution grid adaptation | Higher spatial resolution yields sharper gradient predictions. |
| Cross-Attention | ERA5 / IMDAA | Separate surface/upper-air queries | Explicit cross-attention prevents features from mutually washing out. |
| Train14 Innovate | ERA5 | Convolutional embeddings | Convolutions act as better spatial inductive biases before sequence modeling. |
| TransFusion UNet | ERA5 / IMDAA | U-Net skip connections | Multi-scale feature fusion preserves high-frequency meteorological features. |

## 10. Future Work
* Expanding the TransFusion UNet variant to process even higher resolution grids.
* Incorporating temporal sequence modeling (autoregressive training loops) directly into the DDP pipeline to optimize for multi-day stability during training rather than just 24-hour step accuracy.
* Conducting detailed ablation studies on the Cross-Attention feature fusion to determine the optimal depth for cross-stream interaction.

## 11. Acknowledgements
This research heavily utilized concepts and structural foundations from the following open-source projects. We deeply thank the original authors:
* Pangu-Weather (Original Architecture)
* pangu-pytorch (PyTorch implementation by zhaoshan2)
* Pangu-Weather-lite (Structural foundations by lizhouq)
