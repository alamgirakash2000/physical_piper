#!/bin/bash
# ============================================================
# GR00T N1.6 Fine-tuning Script for AgileX Piper Robot
# ============================================================
# 
# This script fine-tunes the GR00T N1.6-3B model on your collected
# pick-and-place demonstrations.
#
# Prerequisites:
# 1. Run setup first: ./setup_gr00t.sh
# 2. Ensure you have a GPU with at least 24GB VRAM (RTX 4090, A6000, etc.)
#
# Usage:
#   ./train_gr00t.sh
# ============================================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GROOT_DIR="${SCRIPT_DIR}/Isaac-GR00T"
DATASET_PATH="${SCRIPT_DIR}/../datasets/pick_the_white_cup_and_place_it_on_the_red_cup"
MODALITY_CONFIG="${SCRIPT_DIR}/piper_modality_config.py"
OUTPUT_DIR="${SCRIPT_DIR}/checkpoints/piper_finetune"
BASE_MODEL="nvidia/GR00T-N1.6-3B"

# Training hyperparameters
NUM_GPUS=1
MAX_STEPS=5000
SAVE_STEPS=${SAVE_STEPS:-500}
BATCH_SIZE=1  # Per-device batch size (minimum for 24GB GPU)
GRAD_ACCUM=8  # Gradient accumulation steps (effective batch = BATCH_SIZE * GRAD_ACCUM = 8)
# Keep dataloader single-process to avoid intermittent worker segfaults
# seen with forked workers + augmentation stacks on some systems.
DATALOADER_WORKERS=0
VIDEO_BACKEND=${VIDEO_BACKEND:-torchcodec}
OPTIMIZER=${OPTIMIZER:-adamw_bnb_8bit}
USE_FLASH_ATTN=${USE_FLASH_ATTN:-1}

echo "============================================================"
echo "GR00T N1.6 Fine-tuning for AgileX Piper Robot"
echo "============================================================"
echo ""
echo "Dataset:      ${DATASET_PATH}"
echo "Base Model:   ${BASE_MODEL}"
echo "Output:       ${OUTPUT_DIR}"
echo "Max Steps:    ${MAX_STEPS}"
echo "Batch Size:   ${BATCH_SIZE}"
echo "Video Backend:${VIDEO_BACKEND}"
echo "Optimizer:    ${OPTIMIZER}"
echo "Flash Attn:   ${USE_FLASH_ATTN}"
echo ""

# Check if GR00T directory exists
if [ ! -d "${GROOT_DIR}" ]; then
    echo "ERROR: Isaac-GR00T not found at ${GROOT_DIR}"
    echo "Please run: git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T Isaac-GR00T"
    exit 1
fi

# Check if dataset exists
if [ ! -d "${DATASET_PATH}" ]; then
    echo "ERROR: Dataset not found at ${DATASET_PATH}"
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

cd "${GROOT_DIR}"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate aeropiper

echo "Starting fine-tuning..."
echo ""

# Set GPU visibility and memory optimization
export CUDA_VISIBLE_DEVICES=0
export NUM_GPUS=${NUM_GPUS}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
# Run shard loading synchronously to avoid occasional decoder/thread crashes.
export GR00T_DISABLE_SHARD_PREFETCH=1
# Stabilize native kernels and expose more useful stack traces on hard crashes.
export FLASH_ATTENTION_DETERMINISTIC=1
export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=1
export TORCHDYNAMO_DISABLE=1
export GR00T_OPTIM="${OPTIMIZER}"
export GR00T_USE_FLASH_ATTENTION="${USE_FLASH_ATTN}"
# Force a fresh dynamic-module cache so patched local processor code is used.
export HF_MODULES_CACHE="/tmp/gr00t_hf_modules_$$"

# Run fine-tuning
python gr00t/experiment/launch_finetune.py \
    --base-model-path "${BASE_MODEL}" \
    --dataset-path "${DATASET_PATH}" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path "${MODALITY_CONFIG}" \
    --num-gpus ${NUM_GPUS} \
    --output-dir "${OUTPUT_DIR}" \
    --save-total-limit 3 \
    --save-steps ${SAVE_STEPS} \
    --max-steps ${MAX_STEPS} \
    --global-batch-size ${BATCH_SIZE} \
    --gradient-accumulation-steps ${GRAD_ACCUM} \
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader-num-workers ${DATALOADER_WORKERS} \
    --video-backend ${VIDEO_BACKEND}

echo ""
echo "============================================================"
echo "Fine-tuning complete!"
echo "Checkpoint saved to: ${OUTPUT_DIR}"
echo ""
echo "To evaluate, run:"
echo "  ./evaluate_gr00t.sh"
echo "============================================================"
