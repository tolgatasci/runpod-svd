# SVD (Stable Video Diffusion) - RunPod Serverless

Image-to-Video generation using Stable Video Diffusion XT on RunPod Serverless.

## Features

- **SVD-XT Model**: 25 frames, high quality video generation
- **Flexible Input**: Support for image URL or base64
- **Customizable Motion**: Control motion intensity (1-255)
- **Portrait/Landscape**: Any resolution (divisible by 8)

## Deployment

### 1. Build Docker Image

```bash
# Build locally
docker build -t svd-runpod:latest .

# Or build on RunPod (recommended)
```

### 2. Push to Docker Hub

```bash
docker tag svd-runpod:latest YOUR_DOCKERHUB/svd-runpod:latest
docker push YOUR_DOCKERHUB/svd-runpod:latest
```

### 3. Create RunPod Serverless Endpoint

1. Go to [RunPod Serverless](https://www.runpod.io/console/serverless)
2. Click "New Endpoint"
3. Configure:
   - **Name**: `svd-img2vid`
   - **Container Image**: `YOUR_DOCKERHUB/svd-runpod:latest`
   - **GPU**: RTX 4090 or A100 (24GB+ VRAM recommended)
   - **Max Workers**: 1-3
   - **Idle Timeout**: 30s
   - **Execution Timeout**: 300s

4. Copy the **Endpoint ID**

### 4. Configure Environment

Add to your `.env`:

```bash
RUNPOD_API_KEY=your_runpod_api_key
SVD_ENDPOINT_ID=your_endpoint_id
```

## API Usage

### Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_url` | string | - | URL of input image |
| `image_base64` | string | - | Base64 encoded image |
| `num_frames` | int | 25 | Frames to generate (max 25) |
| `fps` | int | 7 | Output video FPS |
| `motion_bucket_id` | int | 127 | Motion intensity (1-255) |
| `noise_aug_strength` | float | 0.02 | Noise augmentation |
| `num_inference_steps` | int | 25 | Denoising steps |
| `width` | int | 1024 | Output width |
| `height` | int | 576 | Output height |
| `seed` | int | null | Random seed |

### Example Request

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "image_url": "https://example.com/image.jpg",
      "num_frames": 25,
      "fps": 8,
      "motion_bucket_id": 100,
      "width": 576,
      "height": 1024
    }
  }'
```

### Response

```json
{
  "status": "success",
  "video_base64": "AAAA...",
  "num_frames": 25,
  "fps": 8,
  "width": 576,
  "height": 1024,
  "duration": 3.125
}
```

## Python Wrapper Usage

```python
from svd_wrapper import generate_video_from_image

# From URL
success = generate_video_from_image(
    image_url="https://example.com/photo.jpg",
    output_path="output.mp4",
    motion_bucket_id=100,
    num_frames=25
)

# From local file
success = generate_video_from_image(
    image_path="presenter.png",
    output_path="animated.mp4",
    width=576,
    height=1024,
    motion_bucket_id=50  # Subtle motion for portraits
)

# Quick intro animation
from svd_wrapper import generate_intro_animation

success = generate_intro_animation(
    presenter_image="presenter.png",
    output_path="intro.mp4",
    motion_level="subtle"  # subtle, medium, or high
)
```

## CLI Usage

```bash
# Basic usage
python svd_wrapper.py -i photo.jpg -o output.mp4

# With options
python svd_wrapper.py \
  --image presenter.png \
  --output intro.mp4 \
  --frames 25 \
  --fps 8 \
  --motion 50 \
  --width 576 \
  --height 1024

# From URL
python svd_wrapper.py \
  -i "https://example.com/image.jpg" \
  -o result.mp4
```

## Motion Guide

| Level | motion_bucket_id | Use Case |
|-------|------------------|----------|
| Very Subtle | 20-40 | Portraits, minimal movement |
| Subtle | 40-70 | Talking heads, slight motion |
| Medium | 70-100 | General animations |
| Dynamic | 100-150 | Action scenes |
| High | 150-200+ | Dramatic movement |

## GPU Requirements

- **Minimum**: RTX 3090 / A6000 (24GB VRAM)
- **Recommended**: RTX 4090 / A100 (24-40GB VRAM)
- **Optimal**: A100 80GB (for batch processing)

## Troubleshooting

### Out of Memory

If you get OOM errors, try:
1. Reduce `num_frames` (14 instead of 25)
2. Reduce resolution
3. Enable VAE slicing in handler.py

### Slow Cold Starts

The model is ~10GB. First request takes 2-3 minutes. Subsequent requests are faster.

### Poor Quality

1. Increase `num_inference_steps` (30-50)
2. Use higher resolution source image
3. Adjust `noise_aug_strength` (0.01-0.05)
