#!/usr/bin/env bash
# Inference inside the Docker image with mounts and env.
# Usage:
#   ./scripts/docker_infer.sh -i INPUT -o OUTPUT [-I IMAGE] [-C CONFIG] [-m CACHE_DIR]
# Example:
#   ./scripts/docker_infer.sh -i data/samples/sample_questions.jsonl -o results/generated/sample_answers.jsonl

set -euo pipefail

# ---- defaults ----
IMAGE_DEFAULT="bioasq-llm-qa:cuda11.8"
CONFIG_DEFAULT="configs/inference.yaml"
INPUT_DEFAULT="data/samples/sample_questions.jsonl"
OUTPUT_DEFAULT="results/generated/sample_answers.jsonl"
CACHE_DIR_DEFAULT="$HOME/.cache/huggingface"
CONTAINER_NAME="bioasq-infer-$(date +%s)"

# ---- parse flags ----
IMAGE="$IMAGE_DEFAULT"
CONFIG="$CONFIG_DEFAULT"
INPUT="$INPUT_DEFAULT"
OUTPUT="$OUTPUT_DEFAULT"
CACHE_DIR="$CACHE_DIR_DEFAULT"

usage() {
  echo "Usage: $0 -i INPUT -o OUTPUT [-I IMAGE] [-C CONFIG] [-m CACHE_DIR]"
  echo "Defaults: IMAGE=$IMAGE_DEFAULT CONFIG=$CONFIG_DEFAULT"
  exit 1
}

while getopts ":i:o:I:C:m:h" opt; do
  case "$opt" in
    i) INPUT="$OPTARG" ;;
    o) OUTPUT="$OPTARG" ;;
    I) IMAGE="$OPTARG" ;;
    C) CONFIG="$OPTARG" ;;
    m) CACHE_DIR="$OPTARG" ;;
    h) usage ;;
    \?) echo "Invalid option -$OPTARG"; usage ;;
  esac
done

[[ -f "$INPUT" ]] || { echo "Input not found: $INPUT"; exit 2; }
mkdir -p "$(dirname "$OUTPUT")"
mkdir -p "$CACHE_DIR"

HF_ENV=""
if [[ -n "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  HF_ENV="-e HUGGINGFACE_HUB_TOKEN=${HUGGINGFACE_HUB_TOKEN}"
fi

docker run --rm --gpus all --name "$CONTAINER_NAME" \
  -e HF_HOME=/models_cache \
  $HF_ENV \
  -v "$(pwd)":/app \
  -v "$CACHE_DIR":/models_cache \
  -w /app \
  "$IMAGE" \
  bash -lc "python -m bioasq_llm.inference.generate --config $CONFIG --input $INPUT --out $OUTPUT"