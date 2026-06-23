# Train14 Architecture

Train14 represents a series of exploratory architectural variants inspired by Pangu-Lite but customized for distributed High-Performance Computing workflows and novel attention mechanisms. This document outlines the core architectures implemented in this repository.

## 1. Base Train14 Architecture

The baseline Train14 models (`train14_era5.py`, `train14_gfs.py`, `train14_imdaa.py`) utilize an Earth-Specific Transformer layout, employing 3D Earth-Specific Positional Bias (3D-ESPB) and standard self-attention mechanisms over atmospheric states.

```mermaid
graph TD
    A[Atmospheric State t] --> B[Patch Embedding]
    B --> C[Transformer Block 1]
    C --> D[Transformer Block 2]
    D --> E[... Transformer Block N]
    E --> F[Patch Unembedding]
    F --> G[Atmospheric State t+1]
```

## 2. Data Pipeline

Train14 features a high-throughput, DDP-compatible data pipeline tailored for loading massive `netCDF4` / `grib` datasets efficiently across nodes without NFS congestion.

```mermaid
flowchart LR
    A[(NFS / Lustre)] -->|xarray open| B(Worker 0 Cache)
    A -->|xarray open| C(Worker 1 Cache)
    B --> D[Distributed Sampler]
    C --> D
    D --> E[GPU DataLoader Prefetch]
    E --> F[PyTorch DDP Model]
```

## 3. Distributed Training Workflow

```mermaid
sequenceDiagram
    participant SLURM as SLURM / PBS
    participant Master as Master Node (Rank 0)
    participant Worker as Worker Nodes
    SLURM->>Master: Allocate Nodes
    SLURM->>Worker: Allocate Nodes
    Master->>Master: Initialize Process Group (NCCL)
    Worker->>Master: Connect via Master IP
    Master->>Master: Broadcast Dataset Keys (Avoid NFS storm)
    Master->>Worker: Sync Models & Optimizer states
    loop Training
        Master->>Worker: DDP Gradient All-Reduce
    end
```

## 4. Cross-Attention Workflow

The Cross-Attention variant (`train14_cross_attention.py`) modifies standard self-attention by explicitly fusing features from surface level observations with upper-air states using cross-attention modules.

```mermaid
graph TD
    S[Surface Patches] --> C[Cross-Attention Block]
    U[Upper-Air Patches] --> C
    C --> O[Fused Features]
    O --> T[Self-Attention Blocks]
```

## 5. Train14-Innovate Architecture

The `train14_innovate_era5.py` variant experiments with convolutional embeddings prior to patchification, allowing the model to capture local high-frequency details before applying global transformer-based attention.

```mermaid
graph TD
    In[Input] --> Conv[Convolutional Downsampling]
    Conv --> Patch[Patch Embedding]
    Patch --> Trans[Transformer Blocks]
    Trans --> Up[Deconvolutional Upsampling]
    Up --> Out[Output Prediction]
```

## 6. TransFusion-UNet Architecture

The TransFusion-UNet variant (`train14_transfusion_unet.py`) combines the spatial hierarchy and skip connections of a U-Net with the global receptive field of Transformer layers.

```mermaid
graph TD
    In[Input] --> E1[Encoder Block 1]
    E1 --> E2[Encoder Block 2]
    E2 --> E3[Encoder Block 3]
    E3 --> B[Bottleneck Transformer]
    B --> D3[Decoder Block 3]
    E2 -->|Skip Connection| D3
    D3 --> D2[Decoder Block 2]
    E1 -->|Skip Connection| D2
    D2 --> D1[Decoder Block 1]
    In -->|Skip Connection| D1
    D1 --> Out[Output Prediction]
```
