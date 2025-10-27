#!/bin/bash
# setup.sh - automate environment setup and training launch

set -e  # exit on any error

echo "=== Setting up ML environment ==="

# Step 1: Install requirements
echo "[1/3] Installing Python dependencies..."
pip install -r ML/requirements.txt

# Step 2: Download Kaggle dataset
echo "[2/3] Downloading AffectNet dataset from Kaggle..."

# Make sure Kaggle CLI is installed
if ! command -v kaggle &> /dev/null; then
    echo "Kaggle CLI not found. Installing..."
    pip install kaggle
fi

# Ensure Kaggle API key exists
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "⚠️ Kaggle API credentials not found!"
    echo "Please put your kaggle.json in ~/.kaggle/"
    echo "You can get it from: https://www.kaggle.com/settings"
    exit 1
fi

mkdir -p ML/datasets
kaggle datasets download -d fatihkgg/affectnet-yolo-format -p ML/datasets --unzip

# Step 3: Launch training
echo "[3/3] Starting training..."
python ML/train.py

echo "✅ All done!"
