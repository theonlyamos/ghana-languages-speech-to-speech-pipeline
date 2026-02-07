# Ghana Speech-to-Speech Pipeline - Environment Setup

This guide covers setting up the development environment for the Ghana S2S Pipeline.

## Recommended Python Version

**Python 3.10.x** or **Python 3.11.x** (3.10 preferred)

### Why these versions?
- `bitsandbytes`: Has compatibility issues with Python 3.12+
- `Coqui TTS`: Best tested on Python 3.9-3.11
- `PyTorch 2.x`: Full CUDA support for 3.10-3.11
- `transformers`: Optimal performance on 3.10+

**Avoid**: Python 3.12+ (bitsandbytes issues), Python 3.8 (EOL soon)

---

## Option 1: Conda/Mamba (Recommended)

Conda handles CUDA toolkit and complex ML dependencies better than pip.

### Install Miniconda or Mamba

```bash
# Option A: Miniconda (smaller)
# Download from: https://docs.conda.io/en/latest/miniconda.html

# Option B: Mamba (faster, drop-in conda replacement)
# Download from: https://github.com/conda-forge/miniforge#mambaforge
```

### Create Environment

```bash
# Create environment with Python 3.10
conda create -n ghana_s2s python=3.10 -y
conda activate ghana_s2s

# Install PyTorch with CUDA (adjust cuda version as needed)
# For CUDA 11.8:
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# For CUDA 12.1:
# conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install audio dependencies (conda handles system libs)
conda install -c conda-forge librosa soundfile ffmpeg -y

# Install remaining packages with pip
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

---

## Option 2: Pip + Venv

If you prefer pip, ensure you have CUDA toolkit installed separately.

### Prerequisites

1. **NVIDIA Driver**: 525+ for CUDA 12.x, 515+ for CUDA 11.8
2. **CUDA Toolkit**: Install from https://developer.nvidia.com/cuda-downloads
3. **cuDNN**: Install from https://developer.nvidia.com/cudnn

### Create Environment

```bash
# Create virtual environment
python3.10 -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip wheel setuptools
```

### Install PyTorch

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only (not recommended for training)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Install Dependencies

```bash
# Install all requirements
pip install -r requirements.txt

# Install espeak-ng for TTS (Linux)
sudo apt-get install espeak-ng libespeak-ng-dev

# For Windows: Download from https://github.com/espeak-ng/espeak-ng/releases
```

---

## Option 3: UV (Experimental - Fastest)

UV is a new ultra-fast Python package manager written in Rust.

### Install UV

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv
```

### Create Environment

```bash
# Create venv with specific Python version
uv venv --python 3.10

# Activate
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows

# Install PyTorch first (uv doesn't handle extra-index-url well yet)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install remaining with uv (much faster than pip)
uv pip install -r requirements.txt
```

---

## System Dependencies

### Linux (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install -y \
    ffmpeg \
    libsndfile1 \
    espeak-ng \
    libespeak-ng-dev \
    build-essential \
    python3-dev
```

### Windows

1. Install [FFmpeg](https://www.gyan.dev/ffmpeg/builds/)
2. Install [espeak-ng](https://github.com/espeak-ng/espeak-ng/releases)
3. Add both to PATH

### macOS

```bash
brew install ffmpeg espeak-ng
```

---

## Verify Full Installation

Run this script to verify everything is installed correctly:

```python
import sys
print(f"Python: {sys.version}")

# PyTorch
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Transformers
import transformers
print(f"Transformers: {transformers.__version__}")

# TTS
try:
    import TTS
    print(f"Coqui TTS: {TTS.__version__}")
except:
    print("Coqui TTS: Not installed")

# Audio
import librosa
import soundfile
print(f"Librosa: {librosa.__version__}")
print(f"Soundfile: {soundfile.__version__}")

print("\n✅ All core dependencies installed!")
```

---

## Troubleshooting

### "bitsandbytes" errors on Windows

```bash
# Use pre-built Windows wheel
pip install bitsandbytes-windows
```

### CUDA not detected

```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA version matches PyTorch
python -c "import torch; print(torch.version.cuda)"
```

### espeak-ng not found (TTS errors)

```bash
# Linux
sudo apt-get install espeak-ng

# Verify
espeak-ng --version
```

### Out of Memory during training

- Reduce `batch_size` in config.py
- Enable `use_8bit=True` for quantization
- Use `gradient_checkpointing=True`

---

## Recommended Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | RTX 3060 (12GB) | RTX 3090/4090 (24GB) |
| RAM | 16GB | 32GB+ |
| Storage | 100GB SSD | 500GB+ NVMe |
| CPU | 8 cores | 16+ cores |

---

## Quick Start Commands

```bash
# Conda (recommended)
conda create -n ghana_s2s python=3.10 -y
conda activate ghana_s2s
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install -r requirements.txt

# Launch notebook
jupyter notebook ghana_s2s_pipeline.ipynb
```
