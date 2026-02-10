#!/bin/bash
# ============================================================
# GR00T N1.6 Environment Setup Script (Conda-based)
# ============================================================
# Sets up the gr00t fine-tuning environment inside the
# 'aeropiper' conda environment.
#
# Usage:
#   ./setup_gr00t.sh
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GROOT_DIR="${SCRIPT_DIR}/Isaac-GR00T"
CONDA_ENV_NAME="aeropiper"

echo "============================================================"
echo "Setting up GR00T N1.6 Environment (conda: ${CONDA_ENV_NAME})"
echo "============================================================"

# ---- 1. Clone / update the repo ----
if [ ! -d "${GROOT_DIR}" ]; then
    echo "Cloning Isaac-GR00T repository..."
    git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T.git "${GROOT_DIR}"
else
    echo "Isaac-GR00T already exists, updating..."
    cd "${GROOT_DIR}"
    git pull
    git submodule update --init --recursive
fi

cd "${GROOT_DIR}"

# ---- 2. Activate conda environment ----
# Source conda so we can use 'conda activate' in a script
eval "$(conda shell.bash hook)"

if ! conda env list | grep -q "^${CONDA_ENV_NAME} "; then
    echo "Creating conda environment '${CONDA_ENV_NAME}' with Python 3.10..."
    conda create -n "${CONDA_ENV_NAME}" python=3.10 -y
fi

echo ""
echo "Activating conda environment: ${CONDA_ENV_NAME}"
conda activate "${CONDA_ENV_NAME}"

echo "Python: $(which python) ($(python --version))"

# ---- 3. Install CUDA toolkit via conda ----
echo ""
echo "Installing CUDA toolkit 12.1 via conda..."
conda install -n "${CONDA_ENV_NAME}" cuda-toolkit=12.1 -c nvidia -y

# Re-activate to pick up new env vars
conda activate "${CONDA_ENV_NAME}"

echo "nvcc: $(nvcc --version 2>/dev/null | tail -1 || echo 'not found')"

# ---- 4. Install PyTorch with CUDA 12.6 support ----
echo ""
echo "Installing PyTorch 2.7.1 with CUDA 12.6..."
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126

# ---- 5. Install gr00t package (without flash-attn) ----
echo ""
echo "Installing gr00t and dependencies..."
pip install -e .

# ---- 6. Install flash-attn (needs torch + CUDA toolkit ready) ----
echo ""
echo "Building and installing flash-attn (this may take several minutes)..."
pip install flash-attn==2.7.4.post1 --no-build-isolation

# ---- 7. Install deepspeed ----
echo ""
echo "Installing deepspeed..."
pip install deepspeed==0.17.6

echo ""
echo "============================================================"
echo "Setup complete! Environment: ${CONDA_ENV_NAME}"
echo ""
echo "To use:"
echo "  conda activate ${CONDA_ENV_NAME}"
echo ""
echo "Verify by running:"
echo "  python -c \"from gr00t.data.dataset import LeRobotSingleDataset; print('GR00T installed successfully')\""
echo ""
echo "Then start training with:"
echo "  ./train_gr00t.sh"
echo "============================================================"
