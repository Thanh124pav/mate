# Multi-stage build for MATE environment
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    ffmpeg \
    x264 \
    libx264-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace/mate

# Copy requirements first for better caching
COPY requirements.txt .
COPY new_requirements.txt .
COPY setup.py .
COPY pyproject.toml .
COPY MANIFEST.in .

# Copy source code
COPY mate/ ./mate/
COPY examples/ ./examples/
COPY scripts/ ./scripts/
COPY ray/ ./ray/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r new_requirements.txt || true

# Install MATE in editable mode
RUN pip install --no-cache-dir -e .

# Install additional RL dependencies
RUN pip install --no-cache-dir \
    "ray[rllib]>=1.12.0,<1.13.0" \
    tabulate \
    "pydantic<2.0" \
    "protobuf==3.20.3" \
    nashpy \
    tensorboard \
    setproctitle \
    wandb

# Set default command
CMD ["/bin/bash"]
