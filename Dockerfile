# RunPod Serverless - Stable Video Diffusion (SVD-XT)
# Image-to-Video Generation - NO AUDIO
# Based on: https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt

FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/huggingface
ENV TRANSFORMERS_CACHE=/app/huggingface
ENV DIFFUSERS_CACHE=/app/huggingface
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

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download SVD-XT model (fp16 variant) to avoid cold start delays
# Model size: ~10GB
RUN python -c "\
from diffusers import StableVideoDiffusionPipeline; \
import torch; \
print('Downloading SVD-XT model...'); \
pipe = StableVideoDiffusionPipeline.from_pretrained( \
    'stabilityai/stable-video-diffusion-img2vid-xt', \
    torch_dtype=torch.float16, \
    variant='fp16' \
); \
print('Model downloaded successfully!'); \
"

# Verify all dependencies are installed
RUN python -c "\
import runpod; \
import torch; \
import diffusers; \
import transformers; \
import accelerate; \
import PIL; \
import numpy; \
import imageio; \
import requests; \
print('All dependencies verified!'); \
print(f'PyTorch: {torch.__version__}'); \
print(f'Diffusers: {diffusers.__version__}'); \
print(f'Transformers: {transformers.__version__}'); \
print(f'CUDA available: {torch.cuda.is_available()}'); \
"

# Copy handler
COPY handler.py .

# RunPod serverless entry point
CMD ["python", "-u", "handler.py"]
