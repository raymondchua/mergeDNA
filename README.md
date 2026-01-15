# mergeDNA

**MergeDNA** is a lightweight transformer-based autoencoder designed for modeling long DNA sequences efficiently.  
This repository is a work-in-progress implementation of the model described in the paper:
"MergeDNA: Context-aware Genome Modeling with Dynamic Tokenization through Token Merging" by Siyuan Li et al. (arXiv:2511.14806)

## Overview

The model follows a two-stage design:

1. **Local Encoder / Decoder**  
   Applies attention over short windows and adaptively merges nearby tokens to compress the input sequence.

2. **Latent Encoder / Decoder**  
   Processes the compressed representation using standard Transformer blocks (no merging) to learn global, high-level representations.

## Implemented Components

- Token Merging
- Token Unmerging
- Local Transformer Encoder / Decoder
- Latent Transformer Encoder / Decoder 
- Reconstruction

## Components not yet implemented

- Dynamic Merging Strategy (currently uses fixed merging for simplicity)
- Adaptive Masked Token Modeling

## Dataset

For simplicity, this implementation uses a publicly available DNA dataset from Hugging Face:

- **Dataset**: `katarinagresova/Genomic_Benchmarks_human_nontata_promoters`

This dataset consists of human non-TATA promoter sequences and is commonly used as a benchmark for genomic sequence modeling. Sequences are composed of the standard DNA alphabet (A, C, G, T), with optional ambiguous bases handled during preprocessing.

The dataset is used here purely for **representation learning and reconstruction**, rather than supervised classification, making it suitable for evaluating compression and information preservation in long DNA sequences.

## Running the Code

To run training locally using Hydra with the Submitit launcher:

```bash
python train.py hydra/launcher=submitit_local
```
## References

If you use or build upon this code, please consider citing the original MergeDNA paper:

```bibtex
@article{li2025mergedna,
  title={MergeDNA: Context-aware Genome Modeling with Dynamic Tokenization through Token Merging},
  author={Li, Siyuan and others},
  journal={arXiv preprint arXiv:2511.14806},
  year={2025}
}
