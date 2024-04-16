# Manifold Preserving Guided Diffusion
The official PyTorch implementation of <a href="https://arxiv.org/abs//2311.16424">Manifold Preserving Guided Diffusion (MPGD)</a>. 
This repository has python implementation of MPGD, a training-free sampling method for both pre-trained pixel-space diffusion models and latent diffusion models in a variety of conditional generation applications.
Different guidance modalities we demonstrate are linear inverse problem solving (Gaussian deblurring and super resolution), FaceID guided generation CLIP guided generation and style guided generation with both pixel space diffusion models and latent diffusion models.
Our implementation is based on the <a href="https://github.com/DPS2022/diffusion-posterior-sampling">DPS</a> and <a href="https://github.com/vvictoryuki/FreeDoM">FreeDoM</a> codebase, which are derived from the <a href="https://github.com/CompVis/stable-diffusion">Stable Diffusion</a> and <a href="https://github.com/ermongroup/ddim">DDIM</a> codebase.


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
To set up the environment for MPGD using Docker, follow these steps:

1. **Build the Docker Image**:
   From the directory containing the Dockerfile, execute the following command:
   ```bash
   docker build -t mpgd .
   ```

2. **Run the Docker Container**:
   To start the container with GPU support, use the run_docker_container.sh script:
   ```bash
   bash run_docker_container.sh
   ```

3. **Activate the Conda Environment**:
   Once inside the container, activate the Conda environment by running:
   ```bash
   conda activate mpgd
   ```


## Downloading Pretrained Models
1. **Linear Inverse Problem Solving**
   
   For linear inverse problem sovling, we use the pretrained pixel space FFHQ diffusion model provided by <a href="https://github.com/DPS2022/diffusion-posterior-sampling">DPS</a>. For manifold projection, we use the first stage models (VQGAN) of the latent diffusion provided by <a href="https://github.com/CompVis/latent-diffusion">Latent Diffusion</a>.
   You can download the pixel space FFHQ diffusion model <a href="https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh">here</a> and place it in `linear_inv/models/`.
   ```
   cd linear_inv
   mkdir models
   mv {DOWNLOAD_DIR}/ffqh_10m.pt ./models/
   ```
   For VQGAN models, you can download them <a href="https://github.com/CompVis/latent-diffusion?tab=readme-ov-file#pretrained-ldms">here</a> and place them in the respective folders in `nonlinear/SD_style/models/ldm/`. For example, for CelebA-HQ model, you can do
   ```
   cd nonlinear/SD_style/models/ldm/celeba256/
   mv {DOWNLOAD_DIR}/celeba.zip .
   unzip celeba.zip
   ```
2. **FaceID & CLIP Guidance Human Face Generation**

   For these two experiments, please follow the previous instruction to download the VQGAN models for manifold projection. For the pixel space human face diffusion model, we use the same checkpoints as <a href="https://github.com/vvictoryuki/FreeDoM">FreeDoM</a>, and you can download the both the CelebA-HQ diffusion model and the ArcFace model <a href="https://drive.google.com/drive/folders/1Szb-n-FGMb_c6dIuqYmdpcvNrc-8GmbD?usp=sharing">here</a> and place them in the directory below:
   ```
   cd nonlinear/Face-GD/
   mkdir exp/models
   mv {DOWNLOAD_DIR}/celeba_hq.ckpt exp/models/
   mv {DOWNLOAD_DIR}/model_ir_se50.pth exp/models/
   ```
   The CLIP model should be downloaded automatically when you run the scripts.

3. **Style Guided Stable Diffusion**

   Please download the pretrained Stable Diffusion 1.4 checkpoint <a href="https://huggingface.co/CompVis">here</a> and then place it in `nonlinear/SD_style/models/ldm/stable-diffusion-v1/`.

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
