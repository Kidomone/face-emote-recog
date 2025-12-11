#!/bin/bash
# setup.sh - create venv, install deps, download dataset, run training

set -e  # stop if any command fails

echo "=== Setting up ML environment ==="

# Step 0: Create and activate virtual environment
echo "[0/3] Creating virtual environment..."
sudo apt install python3 python3-pip

python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip safely
python -m pip install --upgrade pip

# Step 1: Install requirements
echo "[1/3] Installing Python dependencies..."
pip install -r ML/requirements.txt

# Step 2: Download Kaggle dataset
echo "[2/3] Downloading AffectNet dataset from Kaggle..."

mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Make sure kaggle CLI is available
if ! command -v kaggle &> /dev/null; then
    echo "Kaggle CLI not found, installing..."
    pip install kaggle
fi

# Check for API key
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "⚠️  Kaggle API credentials not found!"
    echo "Please place your kaggle.json in ~/.kaggle/"
    echo "You can get it from https://www.kaggle.com/settings"
    deactivate
    exit 1
fi

mkdir -p ML/datasets
kaggle datasets download -d fatihkgg/affectnet-yolo-format -p ML/datasets --unzip


echo "[Optional] Installing system libraries for OpenCV..."
if command -v apt-get &> /dev/null; then
    sudo apt-get update -y
    sudo apt-get install -y libgl1 libglib2.0-0
fi

# Step 3: Run training
echo "[3/3] Launching training..."

mv ML/datasets/YOLO_format ML/datasets/affectnet-yolo-format

python ML/train.py

echo "✅ All done! To reactivate environment later, run:"
echo "source .venv/bin/activate"
