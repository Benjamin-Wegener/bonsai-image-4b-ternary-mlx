"""FLUX Pipeline singleton manager."""

import mlx.core as mx
from app.config import MODEL_DIR, log

_pipeline = None

def get_pipeline():
    """Retrieve or initialize the FLUX.2 Klein 4B Ternary MLX pipeline."""
    global _pipeline
    if _pipeline is None:
        log.info(f"Loading FLUX.2 Klein 4B Ternary MLX pipeline from {MODEL_DIR}...")
        from backend.pipeline import FluxPipeline, PipelineConfig
        _pipeline = FluxPipeline(PipelineConfig(
            backend="bonsai-ternary-mlx",
            baked_model_path=str(MODEL_DIR),
            te_4bit=True,
            evict_text_encoder=True,
        ))
        log.info("FLUX Pipeline ready on Apple Silicon Metal.")
    return _pipeline
