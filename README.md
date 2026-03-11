# Project Aletheia

> A 1-Bit Non-Autoregressive Foundation Model with a Semantic Interlingua

## Overview

Aletheia is a research foundation model that combines:

- **BitNet b1.58** ternary quantization ({-1, 0, +1} weights) for extreme efficiency
- **VQ-VAE Latent Interlingua** — discrete semantic bottleneck grounded in Arabic morphology and Sanskrit grammar
- **Non-autoregressive diffusion** — continuous-time text generation via Energy-Based Diffusion Language Models
- **Epistemological data quarantine** — training exclusively on pre-1930 classical texts and post-1930 primary sources

## Quick Start

```bash
# Install in development mode
pip install -e ".[dev]"

# Run unit tests with the tiny debug config
pytest tests/ -v

# Smoke-test the tiny model on CPU
python -m aletheia.training.pretrain --config configs/model/bitlinear_tiny.yaml
```

## Configs

| Config | Params | Device | Purpose |
|--------|--------|--------|---------|
| `bitlinear_tiny.yaml` | ~1M | CPU (laptop) | Local dev & unit tests |
| `bitlinear_300m.yaml` | 300M | 1× A100 | Primary training target |
| `bitlinear_2b.yaml` | 2B | 8× A100 | Production scale-up |

## License

MIT
