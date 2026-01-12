# Dockerfile for SLT Dataset Augmentation
# Minimal image for text augmentation via back-translation (no GPU/PyTorch needed)

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install only the dependencies needed for augmentation (no torch)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        openai>=1.0.0 \
        nest-asyncio>=1.5.0 \
        python-dotenv>=1.0.0 \
        pandas>=2.0.0 \
        tqdm>=4.65.0 \
        "numpy>=1.24.0,<2.0.0"

# Copy the entire project
COPY . .

# Set Python path to include lib and src directories
ENV PYTHONPATH=/app:/app/lib:$PYTHONPATH
ENV PYTHONUNBUFFERED=1

# Default command (can be overridden)
CMD ["python", "-m", "src.augmentation.augment_dataset", "--help"]
