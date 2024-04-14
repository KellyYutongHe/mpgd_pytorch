FROM nvidia/cuda:11.5.2-cudnn8-devel-ubuntu20.04

# Install basic utilities
RUN apt-get clean && \
    apt-get -y update && \
    apt-get install -y --no-install-recommends \
    # add basic apt packages
	&& apt-get -y install nano wget curl git zip unzip \
	&& apt-get -y install ca-certificates sudo bzip2 libx11-6 \
    && apt-get clean \ 
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda and Python 3.8
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/opt/miniconda/bin:$PATH

RUN curl -Lo ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh
RUN chmod +x ~/miniconda.sh
RUN ~/miniconda.sh -b -p /opt/miniconda
RUN rm ~/miniconda.sh
# RUN conda update -n base -c defaults conda

# RUN conda install -y python==3.9

ARG CONDA_YAML="./environment.yaml"
COPY $CONDA_YAML /tmp/conda_packages.yaml

RUN conda env create -f /tmp/conda_packages.yaml
RUN conda clean -ya

RUN echo "conda init bash" >> ~/.bashrc
RUN echo "conda activate mpgd" >> ~/.bashrc
ENV PATH /opt/conda/envs/mpgd/bin:$PATH
# RUN conda update -n ddrm -c defaults conda && conda clean -ya

# RUN conda install -n base -y -c conda-forge ipykernel matplotlib seaborn
# RUN conda install -n base -y -c conda-forge torchmetrics


# so that soundfile module can read filename with not ascii code.
ENV LC_CTYPE "C.UTF-8"
