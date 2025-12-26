"""
RunPod Serverless Handler for Stable Video Diffusion (SVD-XT)
Image-to-Video generation - NO AUDIO OUTPUT
Based on: https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt
"""

import runpod
import torch
import base64
import requests
import tempfile
import os
from io import BytesIO
from PIL import Image
import numpy as np

# Global model variable - loaded once on cold start
pipe = None


def load_model():
    """Load SVD-XT model on cold start"""
    global pipe

    if pipe is not None:
        return pipe

    from diffusers import StableVideoDiffusionPipeline
    from diffusers.utils import export_to_video

    print("Loading SVD-XT model...")
    print("Model: stabilityai/stable-video-diffusion-img2vid-xt")

    # Load SVD-XT (25 frames, better temporal consistency)
    # Note: No variant parameter - uses default model files
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        torch_dtype=torch.float16
    )

    # Move to GPU
    pipe = pipe.to("cuda")

    # Memory optimizations
    pipe.enable_model_cpu_offload()

    print("SVD-XT model loaded successfully!")
    return pipe


def download_image(url: str) -> Image.Image:
    """Download image from URL"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    response = requests.get(url, headers=headers, timeout=60)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))


def base64_to_image(base64_str: str) -> Image.Image:
    """Convert base64 string to PIL Image"""
    if "base64," in base64_str:
        base64_str = base64_str.split("base64,")[1]
    image_data = base64.b64decode(base64_str)
    return Image.open(BytesIO(image_data))


def video_to_base64(video_path: str) -> str:
    """Convert video file to base64 string"""
    with open(video_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def handler(job):
    """
    RunPod handler function for SVD-XT

    Input parameters (from official docs):
    - image_url: URL of the input image (optional if image_base64 provided)
    - image_base64: Base64 encoded input image (optional if image_url provided)
    - height: Output height (default: 576, must be divisible by 8)
    - width: Output width (default: 1024, must be divisible by 8)
    - num_frames: Number of frames (default: 25 for XT model)
    - num_inference_steps: Denoising steps (default: 25)
    - min_guidance_scale: Min CFG scale (default: 1.0)
    - max_guidance_scale: Max CFG scale (default: 3.0)
    - fps: Frames per second for output (default: 7)
    - motion_bucket_id: Motion intensity 1-255 (default: 127)
    - noise_aug_strength: Noise augmentation 0.0-1.0 (default: 0.02)
    - decode_chunk_size: VAE decode chunk size (default: 8, lower = less VRAM)
    - seed: Random seed for reproducibility (optional)

    Output:
    - video_base64: Base64 encoded MP4 video (NO AUDIO)
    - num_frames, fps, width, height, duration
    """

    job_input = job.get("input", {})

    # Validate input
    if not job_input.get("image_url") and not job_input.get("image_base64"):
        return {"error": "Either image_url or image_base64 is required"}

    try:
        # Load model
        pipe = load_model()
        from diffusers.utils import export_to_video

        # Load input image
        if job_input.get("image_url"):
            print(f"Downloading image from: {job_input['image_url']}")
            image = download_image(job_input["image_url"])
        else:
            print("Decoding base64 image...")
            image = base64_to_image(job_input["image_base64"])

        # Convert to RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Get parameters with defaults (from official docs)
        height = job_input.get("height", 576)
        width = job_input.get("width", 1024)
        num_frames = job_input.get("num_frames", 25)  # SVD-XT default
        num_inference_steps = job_input.get("num_inference_steps", 25)
        min_guidance_scale = job_input.get("min_guidance_scale", 1.0)
        max_guidance_scale = job_input.get("max_guidance_scale", 3.0)
        fps = job_input.get("fps", 7)
        motion_bucket_id = job_input.get("motion_bucket_id", 127)
        noise_aug_strength = job_input.get("noise_aug_strength", 0.02)
        decode_chunk_size = job_input.get("decode_chunk_size", 8)
        seed = job_input.get("seed")

        # Ensure dimensions are divisible by 8
        height = (height // 8) * 8
        width = (width // 8) * 8

        # Resize image to target dimensions
        image = image.resize((width, height), Image.LANCZOS)

        print(f"Generating video:")
        print(f"  Resolution: {width}x{height}")
        print(f"  Frames: {num_frames} @ {fps}fps")
        print(f"  Motion bucket: {motion_bucket_id}")
        print(f"  Inference steps: {num_inference_steps}")
        print(f"  Guidance scale: {min_guidance_scale} -> {max_guidance_scale}")

        # Set seed if provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(seed)
            print(f"  Seed: {seed}")

        # Generate video frames
        with torch.inference_mode():
            frames = pipe(
                image,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                min_guidance_scale=min_guidance_scale,
                max_guidance_scale=max_guidance_scale,
                fps=fps,
                motion_bucket_id=motion_bucket_id,
                noise_aug_strength=noise_aug_strength,
                decode_chunk_size=decode_chunk_size,
                generator=generator,
                output_type="pil"
            ).frames[0]

        print(f"Generated {len(frames)} frames")

        # Save to temporary video file (NO AUDIO)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            output_path = tmp.name

        # Export frames to video
        export_to_video(frames, output_path, fps=fps)
        print(f"Video saved: {output_path}")

        # Get file size
        file_size = os.path.getsize(output_path)
        print(f"File size: {file_size / 1024 / 1024:.2f} MB")

        # Convert to base64
        video_base64 = video_to_base64(output_path)

        # Cleanup temp file
        os.unlink(output_path)

        duration = len(frames) / fps

        return {
            "status": "success",
            "video_base64": video_base64,
            "num_frames": len(frames),
            "fps": fps,
            "width": width,
            "height": height,
            "duration": round(duration, 2),
            "file_size_mb": round(file_size / 1024 / 1024, 2)
        }

    except torch.cuda.OutOfMemoryError as e:
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        return {
            "status": "error",
            "error": "GPU out of memory. Try reducing resolution or decode_chunk_size.",
            "details": str(e)
        }

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error: {error_trace}")
        return {
            "status": "error",
            "error": str(e),
            "traceback": error_trace
        }


# RunPod serverless entry point
runpod.serverless.start({"handler": handler})
