# RunPod Serverless - Stable Video Diffusion (SVD-XT)
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/huggingface
ENV TRANSFORMERS_CACHE=/app/huggingface
ENV TORCH_HOME=/app/torch_cache

# System dependencies (ffmpeg, opencv deps, video codecs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Test all critical imports
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}')"
RUN python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
RUN python -c "import diffusers; print(f'Diffusers: {diffusers.__version__}')"
RUN python -c "from diffusers import StableVideoDiffusionPipeline; print('SVD Pipeline: OK')"
RUN python -c "from diffusers.utils import export_to_video; print('export_to_video: OK')"
RUN python -c "import scipy; print(f'SciPy: {scipy.__version__}')"
RUN python -c "import einops; print('Einops: OK')"

# Download model
COPY download_model.py .
RUN python -u download_model.py
RUN rm download_model.py

# Clear token from environment
ENV HF_TOKEN=""

# Copy handler
COPY handler.py .

CMD ["python", "-u", "handler.py"]
