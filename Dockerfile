# =========================
# Base image: CUDA 11.8 runtime on Ubuntu 22.04
# =========================
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# -------------------------
# System dependencies
# -------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Make python3 the default "python"
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# -------------------------
# Install Python deps
#   - Torch prebuilt for CUDA 11.8
#   - Then the rest from requirements.txt
# -------------------------
# Copy only requirement files first for better layer caching
COPY requirements.txt /app/requirements.txt

# Install torch/torchvision/torchaudio matching CUDA 11.8 wheels
RUN pip install --upgrade pip && \
    pip install --extra-index-url https://download.pytorch.org/whl/cu118 \
        torch torchvision torchaudio && \
    pip install -r /app/requirements.txt

# -------------------------
# Copy application code
# -------------------------
COPY src /app/src
COPY configs /app/configs
COPY scripts /app/scripts

# Optional: create non-root user (uncomment if your environment requires it)
# RUN useradd -m runner && chown -R runner:runner /app
# USER runner

# -------------------------
# Runtime environment
# -------------------------
# Where HuggingFace will cache models (mounted or ephemeral).
# You can override at runtime: -e HF_HOME=/models_cache
ENV HF_HOME=/app/.cache/huggingface

# Do NOT bake tokens; pass at runtime:
#   -e HUGGINGFACE_HUB_TOKEN=...
# If you use WandB or others, pass them at runtime as well.

# Default command drops you in a shell (explicit entrypoints come from scripts)
CMD ["/bin/bash"]