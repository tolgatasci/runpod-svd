# RunPod Serverless - Stable Video Diffusion (SVD-XT)
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/huggingface
ENV TRANSFORMERS_CACHE=/app/huggingface
ENV TORCH_HOME=/app/torch_cache

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1-mesa-glx libglib2.0-0 git wget \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Test imports first
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}')"
RUN python -c "import diffusers; print(f'Diffusers: {diffusers.__version__}')"
RUN python -c "from diffusers import StableVideoDiffusionPipeline; print('SVD Pipeline OK')"

# Download model
COPY download_model.py .
RUN python -u download_model.py
RUN rm download_model.py

# Clear token
ENV HF_TOKEN=""

COPY handler.py .
CMD ["python", "-u", "handler.py"]
