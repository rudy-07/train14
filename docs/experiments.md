# Experimental Configurations and Research Goals

This repository tracks multiple experimental branches of the Train14 architecture. Our core research goals revolve around assessing how localized high-resolution weather datasets and advanced neural network components influence forecast accuracy compared to standard global baselines.

## Architecture Comparisons

| Variant | Dataset | Key Modification | Research Hypothesis |
|---------|---------|------------------|---------------------|
| **Train14-ERA5** | Global ERA5 | None (Baseline) | Establish a strong, reproducible baseline matching Pangu-Lite. |
| **Train14-GFS** | NCEP GFS | Adapted coordinate/variable loading | Validate that the architecture generalizes beyond ECMWF datasets. |
| **Train14-IMDAA** | IMDAA Regional | High-resolution grid adaptation | Determine if higher spatial resolution in localized domains yields sharper gradient predictions. |
| **Cross-Attention** | ERA5 / IMDAA | Separate surface/upper-air queries | Hypothesis: Explicit cross-attention prevents features from mutually washing out during deep self-attention. |
| **Train14-Innovate** | ERA5 | Convolutional patch embeddings | Hypothesis: Convolutions act as better spatial inductive biases before sequence modeling. |
| **TransFusion-UNet** | ERA5 / IMDAA | U-Net skip connections | Hypothesis: Multi-scale feature fusion preserves high-frequency meteorological features (e.g., sharp fronts) often smoothed by pure transformers. |

## Experiment Tracking

All distributed training runs utilize customized logging scripts. Due to the high computational cost, runs are performed incrementally on HPC clusters. 


