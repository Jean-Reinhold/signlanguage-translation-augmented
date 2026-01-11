#!/bin/bash
# Setup script for Sign Language Translation Augmented project
# Usage: ./scripts/setup.sh [--all|--augmentation|--signformer|--notebooks]

set -e

echo "🚀 Setting up Sign Language Translation Augmented project..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for this session
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "📍 Using uv version: $(uv --version)"

# Navigate to project root
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)
echo "📂 Project root: $PROJECT_ROOT"

# --- External Disk Configuration ---
EXTERNAL_DISK="/mnt/disk3Tb"
EXTERNAL_VENV_DIR="$EXTERNAL_DISK/venvs/sign-augmented"
EXTERNAL_UV_CACHE="$EXTERNAL_DISK/uv-cache"

if [ -d "$EXTERNAL_DISK" ]; then
    echo "💾 External disk detected at $EXTERNAL_DISK"
    
    # 1. Ensure directories exist on external disk
    mkdir -p "$EXTERNAL_VENV_DIR"
    mkdir -p "$EXTERNAL_UV_CACHE"
    
    # 2. Set UV Cache Directory
    export UV_CACHE_DIR="$EXTERNAL_UV_CACHE"
    echo "📦 UV cache set to: $UV_CACHE_DIR"
    
    # 3. Handle .venv symlink
    if [ ! -L ".venv" ] && [ -d ".venv" ]; then
        echo "⚠️  Found existing local .venv directory. Moving to external disk..."
        rm -rf ".venv" # Removing it is cleaner if we're moving to symlink
    fi
    
    if [ ! -L ".venv" ]; then
        echo "🔗 Linking .venv to external disk: $EXTERNAL_VENV_DIR"
        ln -snf "$EXTERNAL_VENV_DIR" ".venv"
    fi
else
    echo "ℹ️  No external disk found at $EXTERNAL_DISK, using local storage."
fi
# -----------------------------------

# Create virtual environment with Python 3.11
echo "🐍 Creating virtual environment with Python 3.11..."
uv venv --python 3.11

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source .venv/bin/activate

# Install base dependencies
echo "📥 Installing base dependencies..."
uv pip install -e .

# Install optional dependencies based on argument
case "${1:-}" in
    --all)
        echo "📥 Installing ALL optional dependencies..."
        uv pip install -e ".[all]"
        ;;
    --augmentation)
        echo "📥 Installing augmentation dependencies..."
        uv pip install -e ".[augmentation]"
        ;;
    --signformer)
        echo "📥 Installing Signformer dependencies..."
        uv pip install -e ".[signformer]"
        ;;
    --notebooks)
        echo "📥 Installing notebook dependencies..."
        uv pip install -e ".[notebooks]"
        ;;
    --dev)
        echo "📥 Installing development dependencies..."
        uv pip install -e ".[dev]"
        ;;
    *)
        echo "ℹ️  Only base dependencies installed."
        echo "   Use --all, --augmentation, --signformer, --notebooks, or --dev for optional deps."
        ;;
esac

echo ""
echo "✅ Setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To verify installation:"
echo "  python -c 'import torch; print(f\"PyTorch: {torch.__version__}\")'"
echo "  python -c 'import lightning; print(f\"Lightning: {lightning.__version__}\")'"
