"""Pydantic data schemas and request models."""

from typing import Optional, List
from pydantic import BaseModel, Field


class LoRAItem(BaseModel):
    path: str
    scale: float = 1.0


class SegmentRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image")
    num_regions: int = Field(default=6, description="Number of semantic regions")
    use_model: bool = Field(default=True, description="Whether to use FLUX.2 model for segmentation")
    steps: int = Field(default=4, description="Diffusion steps for model segmentation")


class T2IRequest(BaseModel):
    prompt: str = Field(..., description="Prompt text")
    negative_prompt: Optional[str] = Field(default="", description="Negative prompt")
    width: int = Field(default=512, description="Image width")
    height: int = Field(default=512, description="Image height")
    steps: int = Field(default=4, description="Sampling steps (default: 4)")
    cfg_scale: float = Field(default=1.0, description="Guidance scale (default: 1.0)")
    seed: Optional[int] = Field(default=42, description="Random seed (default: 42)")
    init_image: Optional[str] = Field(default=None, description="Base64 or path to reference/init image")
    mask_image: Optional[str] = Field(default=None, description="Base64 or path to mask image")
    mask_blur: Optional[int] = Field(default=8, description="Feathering blur radius for mask")
    invert_mask: Optional[bool] = Field(default=False, description="Invert mask")
    lora_path: Optional[str] = Field(default=None, description="Single LoRA path")
    lora_scale: Optional[float] = Field(default=1.0, description="Single LoRA scale")
    loras: Optional[List[LoRAItem]] = Field(default_factory=list, description="List of LoRAs")
    strength: Optional[float] = Field(default=0.35, description="Image strength")
    mode: Optional[str] = Field(default="img2img", description="Mode")
    batch_count: int = Field(default=1, description="Number of images")
