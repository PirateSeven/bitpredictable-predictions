#!/usr/bin/env bash
# Run this ON the Jetson Nano after SSH-ing in.
# Supports: JetPack 4.6.x (L4T R32.x), Python 3.6 (system default), PyTorch 1.10.0
set -euo pipefail

REPO_DIR="$HOME/bitpredictable-predictions"
VENV_DIR="$HOME/venv-bitpredictable"
# PyTorch 1.10.0 — official NVIDIA wheel for JetPack 4.6 + Python 3.6
WHL_URL="https://developer.download.nvidia.com/compute/redist/jp/v461/pytorch/torch-1.10.0-cp36-cp36m-linux_aarch64.whl"
WHL_FILE="$HOME/torch-1.10.0-cp36-cp36m-linux_aarch64.whl"

echo "=== Jetson Nano setup for bitpredictable-predictions ==="
echo "Python: $(python3 --version)"

# ── 1. System packages ──────────────────────────────────────────────────────────
echo "[1/6] Installing system packages..."
sudo apt-get update -q
sudo apt-get install -y -q \
  git curl wget \
  python3-venv python3-dev python3-pip \
  libopenblas-dev liblapack-dev gfortran

# ── 2. PyTorch wheel ────────────────────────────────────────────────────────────
echo "[2/6] Downloading PyTorch 1.10.0 (NVIDIA JetPack 4.6 / Python 3.6)..."
if [ ! -f "$WHL_FILE" ]; then
  wget --show-progress "$WHL_URL" -O "$WHL_FILE" || {
    echo ""
    echo "ERROR: Download failed. Please download manually from:"
    echo "  https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048"
    echo "Look for: PyTorch v1.10.0 > JetPack 4.6 > Python 3.6"
    echo "Place the .whl file at: $WHL_FILE"
    exit 1
  }
fi
if [ ! -s "$WHL_FILE" ]; then
  echo "ERROR: Downloaded file is empty. Try downloading manually."
  exit 1
fi

# ── 3. Virtual environment ──────────────────────────────────────────────────────
echo "[3/6] Creating Python 3.6 venv at $VENV_DIR..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip --quiet

# ── 4. Install PyTorch ──────────────────────────────────────────────────────────
echo "[4/6] Installing PyTorch from local wheel..."
pip install "$WHL_FILE"

# ── 5. Install other dependencies ───────────────────────────────────────────────
echo "[5/6] Installing pipeline dependencies (Python 3.6 compatible versions)..."
if [ -f "$REPO_DIR/requirements-jetson.txt" ]; then
  pip install -r "$REPO_DIR/requirements-jetson.txt"
else
  pip install \
    "numpy>=1.17.3,<1.20" \
    "pandas>=1.3,<2.0" \
    "ta>=0.11" \
    "scikit-learn>=0.24,<1.0" \
    "requests>=2.25" \
    "python-dotenv>=0.17"
fi

# ── 6. Clone repo ───────────────────────────────────────────────────────────────
echo "[6/6] Cloning repository..."
if [ -d "$REPO_DIR" ]; then
  echo "Repo already exists — pulling latest..."
  git -C "$REPO_DIR" pull
else
  git clone https://github.com/shichiji/bitpredictable-predictions.git "$REPO_DIR"
fi

# ── Verify ──────────────────────────────────────────────────────────────────────
echo ""
echo "=== Verifying installation ==="
python -c "
import torch, numpy, pandas, sklearn
print('torch   :', torch.__version__)
print('CUDA    :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU     :', torch.cuda.get_device_name(0))
print('numpy   :', numpy.__version__)
print('pandas  :', pandas.__version__)
print('sklearn :', sklearn.__version__)
"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Copy .env to Jetson:"
echo "       scp /Users/Seven/Desktop/repo/bitpredictable-predictions/.env dev@192.168.0.112:$REPO_DIR/.env"
echo "  2. Run training:"
echo "       source $VENV_DIR/bin/activate"
echo "       cd $REPO_DIR"
echo "       python -m pipeline.train"
