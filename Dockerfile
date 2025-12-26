# RunPod Serverless - Stable Video Diffusion (SVD-XT)
# Image-to-Video Generation - NO AUDIO
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# HuggingFace token - passed at BUILD TIME via RunPod
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/huggingface
ENV TRANSFORMERS_CACHE=/app/huggingface
ENV TORCH_HOME=/app/torch_cache

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model download script and run it
COPY download_model.py .
RUN python download_model.py && rm download_model.py

# Clear token from environment
ENV HF_TOKEN=""

# Verify dependencies
RUN python -c "import runpod; import torch; import diffusers; print('All OK')"

# Copy handler
COPY handler.py .

CMD ["python", "-u", "handler.py"]
