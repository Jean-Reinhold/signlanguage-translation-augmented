# =============================================================================
# SIGNFORMER TRAINING DOCKER IMAGE
# =============================================================================
#
# Based on TensorFlow GPU image for CUDA support
# Includes: PyTorch, TensorBoard, comprehensive logging
#
# BUILD:
#   docker build -f signformer/Dockerfile -t signformer:training .
#
# =============================================================================

FROM tensorflow/tensorflow:2.11.0-gpu

LABEL maintainer="SLT Research"
LABEL description="Signformer training with TensorBoard logging"
LABEL version="2.0"

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Setup working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY signformer/requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r /tmp/requirements.txt

# Install TensorBoard and additional monitoring tools
RUN python -m pip install --no-cache-dir \
    tensorboard>=2.11.0 \
    tensorboardX \
    tqdm \
    psutil \
    gputil

# Install Sophia optimizer (optional, for SophiaG)
RUN python -m pip install --no-cache-dir Sophia-Optimizer && \
    # Fix for SophiaG import error in some versions
    sed -i 's/from sophia.sophia import SophiaG, sophiag/from sophia.sophia import SophiaG/' \
    /usr/local/lib/python3.8/dist-packages/sophia/__init__.py 2>/dev/null || true

# Create directories for logs
RUN mkdir -p /app/logs /app/tensorboard

# Environment for better logging
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2

# Copy Signformer code
COPY signformer/ /app/Signformer/

WORKDIR /app/Signformer

# Default command keeps container running
CMD ["tail", "-f", "/dev/null"]
