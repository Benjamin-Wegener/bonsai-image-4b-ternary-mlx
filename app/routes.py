"""FastAPI HTTP route definitions for generation, segmentation, progress, and WebUI."""

import base64
import gc
import io
import os
import secrets
import time
from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import HTMLResponse
from PIL import Image, ImageFilter
import numpy as np
import mlx.core as mx

from app.config import LORAS_DIR, WEB_DIR, log
from app.models import T2IRequest, SegmentRequest
from app.pipeline import get_pipeline
from app.lora import apply_loras_to_model
from app.progress import current_progress, LiveProgressCallback
from app.segmentation import generate_model_segmentation

router = APIRouter()
progress_cb = LiveProgressCallback(get_pipeline)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _free_memory():
    """Aggressively release MLX cache and Python objects."""
    gc.collect()
    try:
        mx.clear_cache()
    except Exception:
        pass


def _decode_b64_image(b64_str: str) -> bytes:
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    return base64.b64decode(b64_str)


def _save_temp_png(raw_bytes: bytes) -> str:
    path = f"/tmp/flux_{secrets.token_hex(8)}.png"
    with open(path, "wb") as f:
        f.write(raw_bytes)
    return path


def _crop_and_resize(img_pil: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Centre-crop then resize to target, keeping aspect ratio correct."""
    orig_w, orig_h = img_pil.size
    target_ratio = target_w / target_h
    orig_ratio = orig_w / orig_h

    if orig_ratio > target_ratio:
        crop_w = int(orig_h * target_ratio)
        left = (orig_w - crop_w) // 2
        cropped = img_pil.crop((left, 0, left + crop_w, orig_h))
    else:
        crop_h = int(orig_w / target_ratio)
        top = (orig_h - crop_h) // 2
        cropped = img_pil.crop((0, top, orig_w, top + crop_h))

    return cropped.resize((target_w, target_h), Image.Resampling.LANCZOS)


def _encode_mask(mask_b64: str, target_w: int, target_h: int,
                 lw: int, lh: int, blur_r: int, invert: bool) -> mx.array:
    """
    Decode, blur, and downsample the mask to latent resolution.
    Returns shape (1, lh*lw, 1) bfloat16, where:
      1.0 = generate new content (inpaint/replace here)
      0.0 = protect original reference image (keep original, untouched)
    """
    mask_bytes = _decode_b64_image(mask_b64)
    with Image.open(io.BytesIO(mask_bytes)) as mask_pil:
        mask_pil = mask_pil.convert("L").resize((target_w, target_h), Image.Resampling.BILINEAR)
        if blur_r > 0:
            mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=blur_r / 2.0))
        mask_small = mask_pil.resize((lw, lh), Image.Resampling.BILINEAR)
        mask_np = np.array(mask_small, dtype=np.float32) / 255.0

    # User draws/selects region to inpaint -> mask_np has 1.0 on inpaint area
    if invert:
        mask_np = 1.0 - mask_np

    return mx.array(mask_np.reshape(1, lh * lw, 1)).astype(mx.bfloat16)



# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/")
def get_root():
    index_file = WEB_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(), status_code=200)
    return HTMLResponse("<h1>FLUX.2 Klein Studio</h1>", status_code=200)


@router.get("/favicon.ico")
def get_favicon():
    return Response(status_code=204)


@router.get("/v1/loras")
def list_loras():
    results = []
    if LORAS_DIR.exists():
        for p in LORAS_DIR.glob("*.safetensors"):
            results.append({"name": p.stem, "path": str(p)})
    return results


@router.get("/v1/progress")
def get_progress():
    return current_progress


@router.post("/v1/segment_mask")
def segment_mask(req: SegmentRequest):
    try:
        img_bytes = _decode_b64_image(req.image)
        return generate_model_segmentation(img_bytes, num_regions=req.num_regions, steps=req.steps)
    except Exception as e:
        log.exception("Segmentation error")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/images/generations")
@router.post("/sdapi/v1/txt2img")
@router.post("/sdapi/v1/img2img")
def generate_images(req: T2IRequest):
    global current_progress
    p = get_pipeline()
    seed = req.seed if (req.seed is not None and req.seed >= 0) else secrets.randbelow(2**31)

    # LoRA setup
    loras_to_apply = []
    if req.loras:
        loras_to_apply = [(item.path, item.scale) for item in req.loras]
    elif req.lora_path:
        loras_to_apply = [(req.lora_path, req.lora_scale or 1.0)]
    apply_loras_to_model(p._model, loras_to_apply)

    p._model.callbacks.in_loop.clear()
    p._model.callbacks.before_loop.clear()
    p._model.callbacks.after_loop.clear()
    p._model.callbacks.register(progress_cb)

    current_progress.update({
        "active": True, "step": 0, "total_steps": req.steps, "pct": 0,
        "stage": "Encoding prompt...", "eta_s": 0.0, "elapsed_s": 0.0,
        "rate_str": "", "preview_b64": None
    })

    t0 = time.perf_counter()
    temp_img_path = None

    try:
        # ---------------------------------------------------------------
        # Image-conditioned path (img2img / inpaint)
        # ---------------------------------------------------------------
        if req.init_image:
            raw_bytes = _decode_b64_image(req.init_image)
            temp_img_path = _save_temp_png(raw_bytes)

            # Pre-process reference image to target resolution
            with Image.open(temp_img_path) as ref_pil:
                ref_pil = ref_pil.convert("RGB")
                processed = _crop_and_resize(ref_pil, req.width, req.height)
                processed.save(temp_img_path, format="PNG")

            from mflux.models.flux2.variants.edit.flux2_klein_edit_helpers import _Flux2KleinEditHelpers
            from mflux.models.flux2.flux2_initializer import Flux2Initializer
            from mflux.models.common.config.config import Config
            from mflux.utils.image_util import ImageUtil

            # Encode text (evicts text encoder from GPU after)
            current_progress["stage"] = "Encoding prompt..."
            pe, ti, _, _ = p._model._encode_prompt_pair(
                prompt=req.prompt, negative_prompt=" ",
                guidance=req.cfg_scale, max_sequence_length=512
            )
            mx.eval(pe, ti)
            _free_memory()

            # Prepare generation latents (pure noise seed)
            latents, latent_ids, lh, lw = _Flux2KleinEditHelpers.prepare_generation_latents(
                seed=seed, height=req.height, width=req.width
            )

            # Load transformer+VAE
            Flux2Initializer.load_transformer_and_vae(p._model)

            # ---------------------------------------------------------------
            # 1. Clean Original Latents (for perfect unmasked pixel reinjection)
            # ---------------------------------------------------------------
            clean_orig_latents, _ = _Flux2KleinEditHelpers.prepare_reference_image_conditioning(
                vae=p._model.vae,
                tiling_config=None,
                image_paths=[temp_img_path],
                height=req.height,
                width=req.width,
                batch_size=1,
            )
            clean_orig_latents = clean_orig_latents.astype(mx.bfloat16)
            try:
                mx.eval(clean_orig_latents)
            except Exception:
                mx.synchronize()

            # ---------------------------------------------------------------
            # 2. Pixel-Space Conditioned Image (Prevents VAE Out-of-Distribution Ghosting)
            # Fill the inpaint region with neutral gray in pixel space before VAE encode,
            # so the model knows the surrounding context without hallucinating the old object.
            # ---------------------------------------------------------------
            cond_img_path = temp_img_path
            mask_latent = None

            if req.mask_image:
                try:
                    # A. Latent mask for smooth reinjection and blend
                    mask_latent = _encode_mask(
                        req.mask_image,
                        req.width, req.height, lw, lh,
                        blur_r=req.mask_blur or 8,
                        invert=req.invert_mask or False
                    )

                    # B. Pixel-space mask (dilated hard mask) to cleanly replace the old object with neutral gray
                    mask_raw = _decode_b64_image(req.mask_image)
                    with Image.open(io.BytesIO(mask_raw)) as m_pil:
                        m_pil = m_pil.convert("L").resize((req.width, req.height), Image.Resampling.BILINEAR)
                        m_arr = np.array(m_pil, dtype=np.uint8)
                        
                        # Slight dilation (2-3px) to guarantee zero edge-convolution leakage from VAE
                        import cv2
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                        m_arr = cv2.dilate(m_arr, kernel, iterations=1)
                        
                        ref_np = np.array(processed).copy()
                        # Replace inpaint area with neutral 128 gray (in-distribution for natural VAE encoding)
                        ref_np[m_arr > 128] = [128, 128, 128]
                        
                        cond_pil = Image.fromarray(ref_np)
                        cond_img_path = f"/tmp/flux_cond_{secrets.token_hex(8)}.png"
                        cond_pil.save(cond_img_path, format="PNG")

                except Exception as ex:
                    log.warning(f"Pixel-space mask prep failed: {ex}")
                    cond_img_path = temp_img_path
                    mask_latent = None

            # 3. Edit conditioning latents (encoded from pixel-masked image)
            img_latents, img_latent_ids = _Flux2KleinEditHelpers.prepare_reference_image_conditioning(
                vae=p._model.vae,
                tiling_config=None,
                image_paths=[cond_img_path],
                height=req.height,
                width=req.width,
                batch_size=1,
            )
            img_latents = img_latents.astype(mx.bfloat16)
            try:
                mx.eval(img_latents, img_latent_ids)
            except Exception:
                mx.synchronize()

            # Cleanup temporary conditioning file if created
            if cond_img_path != temp_img_path and os.path.exists(cond_img_path):
                try:
                    os.remove(cond_img_path)
                except Exception:
                    pass

            # Deterministic noise for timestep-correct re-injection
            init_noise = mx.random.normal(latents.shape, key=mx.random.key(seed)).astype(mx.bfloat16)
            mx.eval(init_noise)

            config = Config(
                model_config=p._model.model_config,
                num_inference_steps=req.steps,
                height=req.height,
                width=req.width,
                guidance=req.cfg_scale,
                scheduler="flow_match_euler_discrete",
            )

            ctx = p._model.callbacks.start(seed=seed, prompt=req.prompt, config=config)
            ctx.before_loop(latents)

            hidden_states = mx.concatenate([latents, img_latents], axis=1)
            img_ids = mx.concatenate([latent_ids, img_latent_ids], axis=1)

            # ---------------------------------------------------------------
            # Denoising loop
            # ---------------------------------------------------------------
            for t in config.time_steps:
                ts = config.scheduler.timesteps[t]

                noise = p._model.transformer(
                    hidden_states=hidden_states,
                    encoder_hidden_states=pe,
                    timestep=ts,
                    img_ids=img_ids,
                    txt_ids=ti,
                    guidance=None,
                )
                noise_gen = noise[:, : latents.shape[1]]
                latents = config.scheduler.step(
                    noise=noise_gen, timestep=t,
                    latents=latents, sigmas=config.scheduler.sigmas
                )

                # --- Proper latent inpainting re-injection ---
                if mask_latent is not None and clean_orig_latents is not None:
                    next_idx = t + 1
                    sigma = float(config.scheduler.sigmas[next_idx]) \
                        if next_idx < len(config.scheduler.sigmas) else 0.0
                    noisy_orig = (1.0 - sigma) * clean_orig_latents + sigma * init_noise
                    latents = mask_latent * latents + (1.0 - mask_latent) * noisy_orig

                ctx.in_loop(t, latents)
                mx.eval(latents)   # materialise before next step; do NOT free other tensors here

                hidden_states = mx.concatenate([latents, img_latents], axis=1)

            ctx.after_loop(latents)

            # Final paste: replace unmasked pixels with the exact decoded originals
            if mask_latent is not None and clean_orig_latents is not None:
                latents = mask_latent * latents + (1.0 - mask_latent) * clean_orig_latents
                mx.eval(latents)

            current_progress["stage"] = "Decoding image..."
            packed = latents.reshape(
                latents.shape[0], lh, lw, latents.shape[-1]
            ).transpose(0, 3, 1, 2)
            decoded = p._model.vae.decode_packed_latents(packed, tiling_config=None)
            mx.eval(decoded)
            _free_memory()

            img_obj = ImageUtil.to_image(
                decoded_latents=decoded,
                config=config,
                seed=seed,
                prompt=req.prompt,
                negative_prompt=None,
                quantization=None,
                image_path=temp_img_path,
                generation_time=time.perf_counter() - t0,
            )
            out_io = io.BytesIO()
            img_obj.image.save(out_io, format="PNG")
            png_bytes = out_io.getvalue()

        # ---------------------------------------------------------------
        # Text-to-image path (no reference image)
        # ---------------------------------------------------------------
        else:
            png_bytes = p.generate_png(
                prompt=req.prompt,
                seed=seed,
                steps=req.steps,
                height=req.height,
                width=req.width,
                guidance=req.cfg_scale,
            )

        duration = time.perf_counter() - t0
        b64_str = base64.b64encode(png_bytes).decode("utf-8")

        current_progress.update({
            "active": False, "pct": 100, "stage": "Done!",
            "elapsed_s": round(duration, 2), "eta_s": 0.0
        })

        return {
            "created": int(time.time()),
            "data": [{"b64_json": b64_str, "seed": seed, "duration_s": round(duration, 2)}],
            "images": [b64_str],
            "parameters": req.model_dump()
        }

    except Exception as e:
        current_progress["active"] = False
        log.exception("Generation error")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        _free_memory()
        if temp_img_path and os.path.exists(temp_img_path):
            try:
                os.remove(temp_img_path)
            except Exception:
                pass
