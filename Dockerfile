FROM nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    git python3-pip python-is-python3 vim \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /root
RUN git clone https://github.com/maxschebek/flow_diagrams.git
WORKDIR /root/flow_diagrams

ENV PIP_NO_CACHE_DIR=off
RUN pip install jaxlib==0.4.23+cuda11.cudnn86 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install -e .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

