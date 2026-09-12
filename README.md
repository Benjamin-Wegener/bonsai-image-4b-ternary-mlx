# Bonsai Image 4B (Ternary 1.58-Bit on Apple Silicon MLX)

A high-performance implementation and WebUI studio for **Bonsai Image 4B** (Ternary 1.58-Bit FLUX.2 Klein) running locally on Apple Silicon Metal via Apple MLX.

## Overview

Bonsai Image 4B uses a **1.58-bit ternary quantization scheme {-1, 0, 1}**, reducing the active diffusion transformer to just **1.21 GB** (down from 7.75 GB in FP16), while retaining over 95% of full-precision image fidelity and prompt comprehension.

### Features
- **Apple Silicon Native**: Powered by MLX Metal compute kernels.
- **Ultra-Fast 4-Step Sampling**: Employs FlowMatch-Euler discrete sampling with shift=3.0 (no CFG overhead required).
- **Interactive WebUI & API**: FastAPI server compatible with standard WebUI / Automatic1111 endpoints (`/sdapi/v1/txt2img`, `/sdapi/v1/img2img`).
- **Img2Img & Inpainting**: Built-in reference image conditioning and segment masking.
- **Low Memory Footprint**: Text encoder is dynamically evicted after prompt tokenization, keeping runtime RAM pressure under 3 GB during generation.

---

## Architecture & Model Weights

Model weights are not stored in this Git repository due to size. Download the weights from Hugging Face:

```bash
# Destination directory
mkdir -p models/bonsai-image-4B-ternary-mlx

# Download model from Hugging Face:
# https://huggingface.co/prism-ml/bonsai-image-ternary-4B-mlx-2bit
```

Required directory layout:
```text
models/bonsai-image-4B-ternary-mlx/
├── transformer-packed-mflux/
├── text_encoder-mlx-4bit/
├── vae/
├── tokenizer/
└── scheduler/
```

---

## Quickstart

### 1. Installation

Create a virtual environment and install the required dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Launching the Studio Server

Run the local server on port 12345:

```bash
export BONSAI_MODEL_DIR="$(pwd)/models/bonsai-image-4B-ternary-mlx"
python -m uvicorn server_12345:app --host 0.0.0.0 --port 12345
```

Open your browser at `http://127.0.0.1:12345/` to access the interactive web interface, or view the API documentation at `http://127.0.0.1:12345/docs`.

### 3. API Usage Example (Python Img2Img)

```python
import base64
import requests

with open("input.png", "rb") as f:
    b64_img = base64.b64encode(f.read()).decode("utf-8")

payload = {
    "prompt": "masterpiece, 8k resolution, photorealistic, intricate details",
    "init_image": f"data:image/png;base64,{b64_img}",
    "steps": 4,
    "cfg_scale": 1.0,
    "strength": 0.35,
    "width": 1024,
    "height": 1024,
    "seed": 42
}

response = requests.post("http://127.0.0.1:12345/sdapi/v1/img2img", json=payload)
result = response.json()

with open("output.png", "wb") as f:
    f.write(base64.b64decode(result["images"][0]))
```

---

## License
Apache 2.0
