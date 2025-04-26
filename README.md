# Natural Scenes Reconstruction from fMRI using Deep Generative Models
Official repository for the paper "Decoding Visual Stimulations of Brain from fMRI using Deep Generative Models" by Berke Bayrak, Ahmet Rasim Emirdağı, and Demet Tangolar.

> This repository contains our course project for ECE6254: Statistical Machine Learning at Georgia Institute of Technology. Our work is built upon and extends the [Brain-Diffuser framework](https://arxiv.org/abs/2303.05334) by Furkan Ozcelik and Rufin VanRullen.

## Overview
We present significant improvements to the Brain-Diffuser framework through two key modifications:
1. Replacing deterministic DDIM sampling with stochastic DDPM sampling
2. Upgrading the generative backbone from Versatile Diffusion to Stable Diffusion

These modifications have led to substantial improvements in reconstruction quality while maintaining the core two-stage architecture of the original Brain-Diffuser work.

## Key Improvements

### 1. Enhanced Sampling Strategy
- Replaced deterministic DDIM sampling with stochastic DDPM sampling
- Achieved improved reconstruction fidelity with pixel correlation increasing from 0.344 to 0.562
- Better structural similarity with SSIM score improvement from 0.371 to 0.467

### 2. Upgraded Generative Model
- Replaced Versatile Diffusion with Stable Diffusion for the second stage reconstruction
- Further improved performance metrics:
  - Pixel correlation: 0.673
  - SSIM score: 0.701
- Enhanced image sharpness and structural coherence

## Results
Comparison of reconstructions using different approaches:
<p align="center"><img src="./figures/reconstruction_grid.png" width="600" ></p>

## Instructions 

### Requirements
* Create conda environment using environment.yml in the main directory:
```bash
conda env create -f environment.yml
```
Note: This is an extensive environment and may include redundant libraries. You may also create the environment by checking requirements yourself.

### Data Acquisition and Processing

1. Download NSD data from NSD AWS Server:
    ```bash
    cd data
    python download_nsddata.py
    ```
2. Download "COCO_73k_annots_curated.npy" file from [HuggingFace NSD](https://huggingface.co/datasets/pscotti/naturalscenesdataset/tree/main)
3. Prepare NSD data for the Reconstruction Task:
    ```bash
    cd data
    python prepare_nsddata.py -sub 1
    python prepare_nsddata.py -sub 2
    python prepare_nsddata.py -sub 5
    python prepare_nsddata.py -sub 7
    ```

### First Stage Reconstruction with VDVAE

1. Download pretrained VDVAE model files and put them in `vdvae/model/` folder:
```bash
wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets-2/imagenet64-iter-1600000-log.jsonl
wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets-2/imagenet64-iter-1600000-model.th
wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets-2/imagenet64-iter-1600000-model-ema.th
wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets-2/imagenet64-iter-1600000-opt.th
```
2. Extract VDVAE latent features for subject 'x': `python scripts/vdvae_extract_features.py -sub x`
3. Train regression models and save test predictions: `python scripts/vdvae_regression.py -sub x`
4. Reconstruct images from predicted features: `python scripts/vdvae_reconstruct_images.py -sub x`

### Second Stage Reconstruction with Stable Diffusion

1. Download the pretrained Stable Diffusion model and place it in the appropriate directory
2. Extract CLIP-Text features for subject 'x': `python scripts/cliptext_extract_features.py -sub x`
3. Train regression models for CLIP-Text features: `python scripts/cliptext_regression.py -sub x`
4. Reconstruct images using DDPM sampling: `python scripts/stablediffusion_reconstruct_images.py -sub x`

### Quantitative Evaluation
1. Save test images: `python scripts/save_test_images.py`
2. Extract evaluation features for test images: `python scripts/eval_extract_features.py -sub 0`
3. Extract evaluation features for reconstructed images: `python scripts/eval_extract_features.py -sub x`
4. Calculate metrics: `python scripts/evaluate_reconstruction.py -sub x`

## References
- Original Brain-Diffuser framework: [Brain-Diffuser](https://github.com/ozcelikfu/brain-diffuser)
- [Natural Scenes Dataset](https://naturalscenesdataset.org/)
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion)
- VDVAE implementation: [openai/vdvae](https://github.com/openai/vdvae)

## Citation
If you find this work useful, please cite our paper:
```bibtex
@article{bayrak2025decoding,
  title={Decoding Visual Stimulations of Brain from fMRI using Deep Generative Models},
  author={Bayrak, Berke and Emirdağı, Ahmet Rasim and Tangolar, Demet},
  year={2025}
}
```
