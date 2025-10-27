# Use NVIDIA CUDA base image with CUDA 11.8
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set the working directory
WORKDIR /BioASQ2024

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV HUGGINGFACE_HUB_TOKEN=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
ENV CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,8

# Install system dependencies, Python, and pip
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    git wget curl vim \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set Python3 as the default
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN pip install --upgrade pip

# ✅ Install PyTorch with CUDA 11.8
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install additional dependencies
RUN pip install \
    transformers \
    datasets \
    accelerate \
    bitsandbytes \
    peft \
    huggingface_hub \
    scipy \
    numpy \
    deepspeed \
    fastapi \
    uvicorn

# Default command to run when the container starts
CMD ["bash"]