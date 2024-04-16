# Using CUDA and cuDNN enabled Ubuntu base image
FROM nvidia/cuda:11.5.2-cudnn8-devel-ubuntu20.04

# Set environment variables to minimize layer usage and make them available globally
ENV CONDA_AUTO_UPDATE_CONDA=false \
    LC_CTYPE="C.UTF-8" \
    PATH=/opt/miniconda/bin:/opt/conda/envs/mpgd/bin:$PATH

# Pass the host timezone as a build argument
ARG TZ=UTC

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install system packages, Miniconda, and additional dependencies in a single RUN command
RUN apt update && apt install -y --no-install-recommends \
        nano wget curl git zip unzip \
        ca-certificates sudo bzip2 libx11-6 libopencv-dev screen \
    && apt clean \
    && rm -rf /var/lib/apt/lists/* \
    && curl -Lo /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh \
    && chmod +x /tmp/miniconda.sh \
    && /tmp/miniconda.sh -b -p /opt/miniconda \
    && rm /tmp/miniconda.sh \
    && echo "conda init bash" >> ~/.bashrc \
    && echo "conda activate mpgd" >> ~/.bashrc

# Copy the Conda environment file and create the environment
ARG CONDA_YAML="./environment.yaml"
COPY $CONDA_YAML /tmp/conda_packages.yaml
RUN conda env create -f /tmp/conda_packages.yaml \
    && conda clean -ya

# Copy your local directory to the container
COPY ./nonlinear /workspace/nonlinear

# Install custom Python packages and set up project
RUN . /opt/miniconda/etc/profile.d/conda.sh \
    && conda activate mpgd \
    && cd /workspace/nonlinear/SD_style \
    && pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers \
    && pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip \
    && pip install -e .