<div align="center">

# VQ-Transplant

### Efficient VQ-Module Integration for Pre-trained Visual Tokenizers

[![ICLR 2026](https://img.shields.io/badge/ICLR-2026-8c1c13.svg)](https://iclr.cc/virtual/2026/poster/10008338)
[![Project Page](https://img.shields.io/badge/Project-Page-3b6ea8.svg)](https://vq-research.github.io/VQ-Transplant/)
[![Paper](https://img.shields.io/badge/Paper-PDF-b31b1b.svg)](https://vq-research.github.io/VQ-Transplant/assets/VQ_Transplant.pdf)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97-Checkpoints-ffd21e.svg)](https://huggingface.co/sunset-clouds/MMD-VAR)

**[Xianghong Fang](https://sunset-clouds.github.io/)<sup>1</sup> · Yuan Yuan<sup>2</sup> · Dehan Kong<sup>1</sup> · Tim G. J. Rudner<sup>1,3</sup>**

<sup>1</sup>University of Toronto &nbsp;&nbsp; <sup>2</sup>Boston College &nbsp;&nbsp; <sup>3</sup>Vijil

**Accepted at ICLR 2026**

</div>

> **TL;DR:** VQ-Transplant replaces the quantizer inside a frozen, pre-trained visual tokenizer and repairs the decoder–quantizer mismatch with lightweight decoder adaptation. On ImageNet-1K, it reduces training cost by **95%**, is **21.8× faster** than training VAR from scratch, and reaches **0.81 rFID** after 5 adaptation epochs.

<p align="center">
  <img src="figures/vq-transplant-pipeline.png" width="95%" alt="VQ-Transplant two-stage pipeline">
</p>

## Overview

Developing a new vector quantization method normally requires retraining an entire tokenizer—encoder, quantizer, and decoder—with expensive adversarial training. VQ-Transplant decouples quantizer research from full tokenizer training through two stages:

1. **VQ module substitution.** Freeze the pre-trained encoder and decoder, replace the native quantizer, and train only the new VQ module.
2. **Decoder adaptation.** Freeze the encoder and transplanted quantizer, then adapt the decoder to the new quantized latent space.

The framework supports both **multi-scale quantization** (`VAR/`) and **fixed-scale quantization** (`VQGAN/`), with Vanilla VQ, EMA VQ, Online VQ, Wasserstein VQ, and the proposed MMD VQ.

## Highlights

- **Plug-and-play VQ integration:** evaluate a new quantizer without end-to-end tokenizer retraining.
- **Resource efficient:** 2 × A100 GPUs and 22 total hours for the standard ImageNet-1K setting, compared with 16 × A100 GPUs and 60 hours for VAR training on OpenImages.
- **Strong reconstruction:** MMD-VAR with 8,192 codes reaches 0.81 rFID after 5 adaptation epochs and 0.74 rFID after 20 epochs.
- **Full codebook use:** MMD-VAR achieves 100% utilization with both 4,096 and 8,192 entries.
- **Broad evaluation:** ImageNet-1K, FFHQ, CelebA-HQ, and LSUN-Churches; PSNR, SSIM, LPIPS, rFID, and rIS are included in the repository.

## Model Zoo

The released checkpoints are MMD-VAR tokenizers trained with a 2-epoch VQ substitution stage followed by 20 epochs of decoder adaptation on ImageNet-1K.

<table width="100%">
  <thead>
    <tr>
      <th>Model</th>
      <th align="right">K</th>
      <th align="right">Tokens</th>
      <th align="right">Epochs</th>
      <th align="right">Util. ↑</th>
      <th align="right">PSNR ↑</th>
      <th align="right">SSIM ↑</th>
      <th align="right">LPIPS ↓</th>
      <th align="right">rFID ↓</th>
      <th align="right">rIS ↑</th>
      <th align="center">CKPT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>MMD-VAR</td><td align="right">4,096</td><td align="right">680</td><td align="right">20</td><td align="right">100%</td><td align="right">24.24</td><td align="right">63.3</td><td align="right">0.108</td><td align="right"><b>0.79</b></td><td align="right">199.1</td><td align="center"><a href="https://huggingface.co/sunset-clouds/MMD-VAR/resolve/main/MMD-VAR-4096-20.pth.tar?download=true">Download</a></td>
    </tr>
    <tr>
      <td>MMD-VAR</td><td align="right">8,192</td><td align="right">680</td><td align="right">20</td><td align="right">100%</td><td align="right">24.36</td><td align="right">63.7</td><td align="right"><b>0.103</b></td><td align="right"><b>0.74</b></td><td align="right"><b>201.0</b></td><td align="center"><a href="https://huggingface.co/sunset-clouds/MMD-VAR/resolve/main/MMD-VAR-8192-20.pth.tar?download=true">Download</a></td>
    </tr>
  </tbody>
</table>

Download both checkpoints with the Hugging Face CLI:

```bash
pip install -U huggingface_hub
hf download sunset-clouds/MMD-VAR --include "*.pth.tar" --local-dir checkpoints
```

## Main Results

### ImageNet-1K reconstruction

All VQ-Transplant results below use 5 decoder-adaptation epochs. The pre-trained VAR tokenizer is included as a reference.

<table width="100%">
  <thead>
    <tr>
      <th align="left">Method</th>
      <th align="center">VQ type</th>
      <th align="right">Codebook</th>
      <th align="right">Utilization ↑</th>
      <th align="right">PSNR ↑</th>
      <th align="right">SSIM ↑</th>
      <th align="right">LPIPS ↓</th>
      <th align="right">rFID ↓</th>
      <th align="right">rIS ↑</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>VAR tokenizer</td><td align="center">Multi-scale</td><td align="right">4,096</td><td align="right">100%</td><td align="right"><b>24.37</b></td><td align="right"><b>63.9</b></td><td align="right"><b>0.100</b></td><td align="right">0.92</td><td align="right">198.6</td>
    </tr>
    <tr>
      <td>MMD-VQ</td><td align="center">Fixed-scale</td><td align="right">65,536</td><td align="right">99.9%</td><td align="right">24.65</td><td align="right">65.0</td><td align="right">0.106</td><td align="right">0.86</td><td align="right">197.1</td>
    </tr>
    <tr>
      <td>MMD-VAR</td><td align="center">Multi-scale</td><td align="right">4,096</td><td align="right">100%</td><td align="right">24.16</td><td align="right">63.2</td><td align="right">0.108</td><td align="right">0.91</td><td align="right">199.2</td>
    </tr>
    <tr>
      <td><b>MMD-VAR</b></td><td align="center"><b>Multi-scale</b></td><td align="right"><b>8,192</b></td><td align="right"><b>100%</b></td><td align="right"><b>24.37</b></td><td align="right"><b>63.8</b></td><td align="right"><b>0.104</b></td><td align="right"><b>0.81</b></td><td align="right"><b>201.0</b></td>
    </tr>
  </tbody>
</table>

<p align="center">
  <img src="figures/transplant-refinement-imagenet.png" width="100%" alt="MMD-VAR ImageNet reconstructions before and after decoder adaptation">
  <br>
  <sub>MMD-VAR reconstruction before decoder adaptation (top) and after adaptation (bottom).</sub>
</p>

### Cross-dataset generalization

The fixed-scale implementation generalizes to datasets that are structurally different from ImageNet-1K and the tokenizer's OpenImages pre-training data.

| Dataset | Best model | Codebook | rFID after substitution ↓ | rFID after adaptation ↓ |
|:--|:--|--:|--:|--:|
| FFHQ | Wasserstein VQ | 32,768 | 2.27 | **1.21** |
| CelebA-HQ | MMD VQ | 16,384 | 2.96 | **2.60** |
| LSUN-Churches | Wasserstein VQ | 16,384 | 2.76 | **1.79** |

<table>
  <tr>
    <td align="center"><img src="figures/reconstruction-ffhq.png" alt="FFHQ reconstructions"><br><b>FFHQ</b></td>
  </tr>
  <tr>
    <td align="center"><img src="figures/reconstruction-celebahq.png" alt="CelebA-HQ reconstructions"><br><b>CelebA-HQ</b></td>
  </tr>
  <tr>
    <td align="center"><img src="figures/reconstruction-churches.png" alt="LSUN-Churches reconstructions"><br><b>LSUN-Churches</b></td>
  </tr>
</table>

Each panel shows original inputs (top), Wasserstein-VQ reconstructions (middle), and MMD-VQ reconstructions (bottom) at 256 × 256 resolution.

## Repository Structure

```text
VQ-Transplant/
├── assets/                 # Original inputs and reconstructed samples for four datasets
├── VAR/                    # Multi-scale VQ-Transplant implementation
│   ├── models/             # Multi-scale quantizers and tokenizer components
│   ├── scripts/            # Substitution, adaptation, and rFID launch scripts
│   ├── record/             # Training logs for both stages
│   └── results/            # PSNR/SSIM/LPIPS CSVs and rFID/rIS logs
├── VQGAN/                  # Fixed-scale VQ-Transplant implementation
│   ├── models/             # Fixed-scale quantizers and tokenizer components
│   ├── scripts/            # Substitution, adaptation, and rFID launch scripts
│   ├── record/             # Training logs for both stages
│   └── results/            # Metrics on ImageNet-1K, FFHQ, CelebA-HQ, and Churches
├── docs/                   # Project page
└── figures/                # README figures preserved independently of the paper source
```

## Setup

Clone the repository and create a Python environment with a CUDA-enabled PyTorch installation:

```bash
git clone https://github.com/VQ-Research/VQ-Transplant.git
cd VQ-Transplant

conda create -n vq-transplant python=3.10 -y
conda activate vq-transplant

# Install PyTorch for your CUDA version first: https://pytorch.org/get-started/locally/
pip install torchvision timm einops omegaconf safetensors scipy pandas pillow \
  opencv-python tqdm piq pyiqa ruamel.yaml clean-fid \
  pytorch-image-generation-metrics
```

### Data layout

`--dataset_dir` should contain the selected dataset in the layout expected by `torchvision.datasets.ImageFolder` (ImageNet-1K) or the corresponding loader in `VAR/data/` and `VQGAN/data/` (FFHQ, CelebA-HQ, and LSUN-Churches). All experiments use 256 × 256 images.

## Training

Ready-to-edit launch recipes for every quantizer are provided under each implementation's `scripts/` directory. The following example runs MMD-VAR on two GPUs.

### Stage I — VQ module substitution

```bash
cd VAR
torchrun --standalone --nproc_per_node=2 train_VAR_transplant.py \
  --VQ mmd_vq \
  --dataset_name ImageNet \
  --dataset_dir /path/to/data \
  --pretrained_tokenizer /path/to/vae_ch160v4096z32.pth \
  --global_batch_size 64 \
  --codebook_size 4096 \
  --codebook_dim 32 \
  --use_multiscale \
  --stage transplant \
  --alpha 1.0 --beta 1.0 --gamma 0.5
```

### Stage II — decoder adaptation

Pass the Stage-I checkpoint filename through `--checkpoint_name`:

```bash
torchrun --standalone --nproc_per_node=2 train_refinement.py \
  --VQ mmd_vq \
  --dataset_name ImageNet \
  --dataset_dir /path/to/data \
  --pretrained_tokenizer /path/to/vae_ch160v4096z32.pth \
  --checkpoint_dir /path/to/checkpoints \
  --checkpoint_name checkpoint-mmd-vq-stage1.pth.tar \
  --global_batch_size 64 \
  --codebook_size 4096 \
  --codebook_dim 32 \
  --use_multiscale \
  --stage refinement \
  --alpha 1.0 --beta 1.0 --gamma 1.0
```

For the fixed-scale experiments, use the analogous entry points `VQGAN/train_VQ_transplant.py` and `VQGAN/train_refinement.py`. See [`VAR/scripts`](VAR/scripts) and [`VQGAN/scripts`](VQGAN/scripts) for all configurations reported in the paper.

## Evaluation and Logs

- Reconstruction evaluation writes per-epoch **PSNR, SSIM, LPIPS, and reconstruction loss** to `results/transplant/` or `results/refinement/`.
- The scripts under `scripts/rFID/` compute **rFID and rIS** from saved reconstructions.
- Complete training output is preserved in `record/transplant/` and `record/refinement/`.
- Published metric outputs are available in [`VAR/results`](VAR/results) and [`VQGAN/results`](VQGAN/results).

The rFID scripts also contain the original cluster paths; update their sample and statistics paths before execution.

## Citation

If VQ-Transplant is useful for your research, please cite:

```bibtex
@inproceedings{fang2026vqtransplant,
  title     = {VQ-Transplant: Efficient VQ-Module Integration for Pre-trained Visual Tokenizers},
  author    = {Fang, Xianghong and Yuan, Yuan and Kong, Dehan and Rudner, Tim G. J.},
  booktitle = {International Conference on Learning Representations},
  year      = {2026}
}
```

## Acknowledgements

This repository builds on the pre-trained [VAR tokenizer](https://github.com/FoundationVision/VAR). We thank the authors of the quantization, tokenizer, perceptual-metric, and evaluation libraries that made this work possible.
