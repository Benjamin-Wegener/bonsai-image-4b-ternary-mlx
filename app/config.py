"""Configuration and constants for FLUX.2 Klein 4B Ternary MLX server."""

from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("bonsai-server")

DEMO_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = DEMO_DIR / "models" / "bonsai-image-4B-ternary-mlx"
LORAS_DIR = Path("/Volumes/Untitled/genAI/loras")
WEB_DIR = DEMO_DIR / "web"
