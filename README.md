# Project Aletheia

> A 1-Bit Non-Autoregressive Foundation Model with a Semantic Interlingua

## Overview

Aletheia is a research foundation model that combines:

- **BitNet b1.58** ternary quantization ({-1, 0, +1} weights) for extreme efficiency
- **VQ-VAE Latent Interlingua** — discrete semantic bottleneck grounded in Arabic morphology and Sanskrit grammar
- **Non-autoregressive diffusion** — continuous-time text generation via Energy-Based Diffusion Language Models
- **Epistemological data quarantine** — training exclusively on pre-1930 classical texts and post-1930 primary sources
- **Equal Global Representation** — mandatory inclusion of Eastern, South Asian, African, and Indigenous sources in original scripts

## Quick Start

```bash
# Install in development mode
pip install -e ".[dev]"

# Regenerate the "Museum Grade" corpus (Original language ONLY, no translations)
make data
# On Windows (if 'make' is not installed):
# python -c "from aletheia.data.classical_scraper import scrape_all_classical; scrape_all_classical()"
```

## Data Philosophy (Museum Grade)

Aletheia adheres to a strict **Museum Grade** data policy:
- **No Translations**: Texts are collected in their original language and script (e.g., Devanagari Sanskrit, Classical Chinese, Hieroglyphs, Cuneiform).
- **Global Equality**: Every region of the world (East Asia, South Asia, Africa, Middle East, Americas, Europe) must have equal weight in the final corpus.
- **Temporal Cutoff**: No data published after **2021** is used, strictly preventing AI inbreeding.
- **Primary Sources**: Prioritizes declassified documents, legal opinions, and foundational STEM manuscripts.

## Automation & Storage

- **Automated Regeneration**: A GitHub Action (`.github/workflows/data_generation.yml`) regenerates the corpus weekly to maintain fresh, verified datasets.
- **Tracked Datasets**: High-purity `.jsonl` files are tracked in the repository for immediate reproducibility, with automated purification filters enforced at commit-time.

## Configs

| Config | Params | Device | Purpose |
|--------|--------|--------|---------|
| `bitlinear_tiny.yaml` | ~1M | CPU (laptop) | Local dev & unit tests |
| `bitlinear_300m.yaml` | 300M | 1× A100 | Primary training target |
| `bitlinear_2b.yaml` | 2B | 8× A100 | Production scale-up |

## License

MIT
