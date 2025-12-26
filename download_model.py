"""Download SVD-XT model at build time"""
import os
import torch

# Get token from environment
token = os.environ.get("HF_TOKEN", "")

if token:
    from huggingface_hub import login
    login(token=token)
    print("Logged in to HuggingFace")
else:
    print("No HF_TOKEN, trying without auth")

from diffusers import StableVideoDiffusionPipeline

print("Downloading SVD-XT model (this takes a while)...")

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    torch_dtype=torch.float16,
    variant="fp16"
)

print("Model downloaded successfully!")
