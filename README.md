# Manifold Preserving Guided Diffusion
The official PyTorch implementation of <a href="https://arxiv.org/abs//2311.16424">Manifold Preserving Guided Diffusion (MPGD)</a>. 
This repository has python implementation of MPGD, a training-free sampling method for both pre-trained pixel-space diffusion models and latent diffusion models in a variety of conditional generation applications.
Different guidance modalities we demonstrate are linear inverse problem solving (Gaussian deblurring and super resolution), FaceID guided generation CLIP guided generation and style guided generation with both pixel space diffusion models and latent diffusion models.
Our implementation is based on the <a href="https://github.com/DPS2022/diffusion-posterior-sampling">DPS</a> and  <a href="https://github.com/vvictoryuki/FreeDoM">FreeDoM</a> codebase, which are derived from the <a href="https://github.com/CompVis/stable-diffusion">Stable Diffusion</a> and <a href="https://github.com/ermongroup/ddim">DDIM</a> codebase.


## Installations
### Conda Installation
```
conda env create -f environment.yaml
conda activate mpgd
cd nonlinear/SD_style
pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip
pip install -e .
```
### Docker Installation
TODO


## Downloading Pretrained Models
TODO


## Running Inference
For linear inverse problem sovling, please refer to subdirectory `linear_inv/`. For pixel space FaceID guidance and CLIP guidance with pretrained CelebA-HQ model, please refer to subdirectory `nonlinear/Face-GD/`. For style guided Stable Diffusion, please refer to subdirectory `nonlinear/SD_style/`. Inside each subdirectory, there are detailed instructions on how to run each task.


## Citation
If you find our work interesting or helpful, please consider citing

```
@inproceedings{
    he2024manifold,
    title={Manifold Preserving Guided Diffusion},
    author={Yutong He and Naoki Murata and Chieh-Hsin Lai and Yuhta Takida and Toshimitsu Uesaka and Dongjun Kim and Wei-Hsiang Liao and Yuki Mitsufuji and J Zico Kolter and Ruslan Salakhutdinov and Stefano Ermon},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024},
    url={https://openreview.net/forum?id=o3BxOLoxm1}
}
```
