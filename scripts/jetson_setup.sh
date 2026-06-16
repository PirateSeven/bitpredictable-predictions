#!/usr/bin/env bash
# Run this ON the Jetson Nano after SSH-ing in.
# Sets up Python 3.8, PyTorch 1.13, and the predictions pipeline.
set -euo pipefail

REPO_DIR="$HOME/bitpredictable-predictions"
VENV_DIR="$HOME/venv-bitpredictable"

echo "=== Jetson Nano setup for bitpredictable-predictions ==="

# ── 1. System packages ──────────────────────────────────────────────────────────
echo "[1/6] Installing system packages..."
sudo apt-get update -q
sudo apt-get install -y -q \
  software-properties-common git curl wget \
  python3.8 python3.8-venv python3.8-dev python3-pip \
  libopenblas-dev liblapack-dev gfortran

# ── 2. PyTorch for JetPack 4.6 ─────────────────────────────────────────────────
echo "[2/6] Downloading PyTorch 1.13 (NVIDIA Jetson build)..."
# Source: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
WHL_URL="https://developer.download.nvidia.com/compute/redist/jp/v461/pytorch/torch-1.13.0a0+410ce96a.nv22.12-cp38-cp38-linux_aarch64.whl"
WHL_FILE="$HOME/torch-jetson.whl"

if [ ! -f "$WHL_FILE" ]; then
  wget -q --show-progress "$WHL_URL" -O "$WHL_FILE" || {
    echo "ERROR: Could not download PyTorch wheel."
    echo "Please download manually from:"
    echo "  https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048"
    echo "and place it at: $WHL_FILE"
    exit 1
  }
fi

# ── 3. Virtual environment ──────────────────────────────────────────────────────
echo "[3/6] Creating Python 3.8 venv at $VENV_DIR..."
python3.8 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip --quiet

# ── 4. Install PyTorch + dependencies ──────────────────────────────────────────
echo "[4/6] Installing PyTorch from local wheel..."
pip install "$WHL_FILE"

echo "[5/6] Installing other dependencies..."
pip install \
  "numpy>=1.24,<2" \
  "pandas>=2.2" \
  "scikit-learn>=1.4" \
  "requests>=2.31" \
  "python-dotenv>=1.0" \
  "google-generativeai>=0.8,<1.0" \
  "ta>=0.11"

# ── 5. Clone repo ───────────────────────────────────────────────────────────────
echo "[5/6] Cloning repository..."
if [ -d "$REPO_DIR" ]; then
  echo "Repo already exists — pulling latest..."
  git -C "$REPO_DIR" pull
else
  git clone https://github.com/shichiji/bitpredictable-predictions.git "$REPO_DIR"
fi

# ── 6. Verify CUDA ──────────────────────────────────────────────────────────────
echo "[6/6] Verifying CUDA..."
python -c "
import torch
print('torch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA device:', torch.cuda.get_device_name(0))
"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Copy .env:  scp /path/to/.env jetson@192.168.0.112:$REPO_DIR/.env"
echo "  2. Run train:  source $VENV_DIR/bin/activate && cd $REPO_DIR && python -m pipeline.train"
