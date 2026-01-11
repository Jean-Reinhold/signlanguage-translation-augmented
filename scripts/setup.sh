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
