"""Download SVD-XT model at build time"""
import os
import sys

print("=" * 50)
print("SVD-XT Model Download Script")
print("=" * 50)

# Get token from environment
token = os.environ.get("HF_TOKEN", "")
print(f"HF_TOKEN present: {bool(token)}")

try:
    if token:
        from huggingface_hub import login
        login(token=token)
        print("SUCCESS: Logged in to HuggingFace")
    else:
        print("WARNING: No HF_TOKEN, trying without auth")

    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    from diffusers import StableVideoDiffusionPipeline
    print("Diffusers imported successfully")

    print("\nDownloading SVD-XT model...")
    print("This will take several minutes...")

    # Try without variant first (more compatible)
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        torch_dtype=torch.float16
    )

    print("\n" + "=" * 50)
    print("SUCCESS: Model downloaded!")
    print("=" * 50)

except Exception as e:
    print("\n" + "=" * 50)
    print(f"ERROR: {type(e).__name__}")
    print(f"MESSAGE: {str(e)}")
    print("=" * 50)
    import traceback
    traceback.print_exc()
    sys.exit(1)
