#!/usr/bin/env bash
# Train inside the Docker image with sensible mounts and env.
# Usage:
#   ./scripts/docker_train.sh [-i IMAGE] [-c CONFIG] [-d DATA_DIR] [-r RESULTS_DIR] [-m CACHE_DIR]
# Example:
#   ./scripts/docker_train.sh -c configs/finetune_bioasq.yaml

set -euo pipefail

# ---- defaults (edit to taste) ----
IMAGE_DEFAULT="bioasq-llm-qa:cuda11.8"
CONFIG_DEFAULT="configs/finetune_bioasq.yaml"
DATA_DIR_DEFAULT="$(pwd)/data"
RESULTS_DIR_DEFAULT="$(pwd)/results"
CACHE_DIR_DEFAULT="$HOME/.cache/huggingface"      # shared HF cache
CONTAINER_NAME="bioasq-train-$(date +%s)"

# ---- parse flags ----
IMAGE="$IMAGE_DEFAULT"
CONFIG="$CONFIG_DEFAULT"
DATA_DIR="$DATA_DIR_DEFAULT"
RESULTS_DIR="$RESULTS_DIR_DEFAULT"
CACHE_DIR="$CACHE_DIR_DEFAULT"

usage() {
  echo "Usage: $0 [-i IMAGE] [-c CONFIG] [-d DATA_DIR] [-r RESULTS_DIR] [-m CACHE_DIR]"
  exit 1
}

while getopts ":i:c:d:r:m:h" opt; do
  case "$opt" in
    i) IMAGE="$OPTARG" ;;
    c) CONFIG="$OPTARG" ;;
    d) DATA_DIR="$OPTARG" ;;
    r) RESULTS_DIR="$OPTARG" ;;
    m) CACHE_DIR="$OPTARG" ;;
    h) usage ;;
    \?) echo "Invalid option -$OPTARG"; usage ;;
  esac
done

# ---- sanity checks ----
[[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG"; exit 2; }
mkdir -p "$RESULTS_DIR"
mkdir -p "$CACHE_DIR"

# Optional: Hugging Face token
HF_ENV=""
if [[ -n "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  HF_ENV="-e HUGGINGFACE_HUB_TOKEN=${HUGGINGFACE_HUB_TOKEN}"
fi

# ---- run ----
docker run --rm --gpus all --name "$CONTAINER_NAME" \
  -e HF_HOME=/models_cache \
  $HF_ENV \
  -v "$(pwd)":/app \
  -v "$DATA_DIR":/app/data \
  -v "$RESULTS_DIR":/app/results \
  -v "$CACHE_DIR":/models_cache \
  -w /app \
  "$IMAGE" \
  bash -lc "python -m bioasq_llm.training.finetune --config $CONFIG"