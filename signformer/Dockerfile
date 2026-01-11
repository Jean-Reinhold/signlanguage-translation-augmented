FROM tensorflow/tensorflow:2.11.0-gpu

# Setup working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt && \
    python -m pip install --no-cache-dir Sophia-Optimizer

# Default command keeps the container running (override in compose or at runtime)
CMD ["tail", "-f", "/dev/null"]