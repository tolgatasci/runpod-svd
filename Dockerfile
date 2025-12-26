# RunPod Serverless - Stable Video Diffusion (SVD-XT)
# Image-to-Video Generation - NO AUDIO
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# HuggingFace token - passed at BUILD TIME via RunPod, NOT stored in image
ARG HF_TOKEN

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

# Download SVD-XT model using build-time token
RUN python -c "\
import os; \
import torch; \
token = '${HF_TOKEN}'; \
if token and token != '\${HF_TOKEN}': \
    from huggingface_hub import login; \
    login(token=token); \
    print('Logged in to HuggingFace'); \
from diffusers import StableVideoDiffusionPipeline; \
print('Downloading SVD-XT model...'); \
pipe = StableVideoDiffusionPipeline.from_pretrained( \
    'stabilityai/stable-video-diffusion-img2vid-xt', \
    torch_dtype=torch.float16, \
    variant='fp16' \
); \
print('Model downloaded!'); \
"

# Clear token from environment (security)
ENV HF_TOKEN=""

# Verify
RUN python -c "import runpod; import torch; import diffusers; print('OK')"

# Copy handler
COPY handler.py .

CMD ["python", "-u", "handler.py"]
