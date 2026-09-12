"""Live diffusion progress and latent preview tracker."""

import base64
import io
import time
import mlx.core as mx
import numpy as np
from PIL import Image
from app.config import log

current_progress = {
    "active": False,
    "step": 0,
    "total_steps": 4,
    "pct": 0,
    "step_time_s": 0.0,
    "elapsed_s": 0.0,
    "eta_s": 0.0,
    "stage": "idle",
    "rate_str": "",
    "preview_b64": None
}


class LiveProgressCallback:
    """Callback hook executed during diffusion sampling to track speed, ETA, and preview."""
    def __init__(self, get_pipeline_fn):
        self.get_pipeline = get_pipeline_fn
        self.t_start = 0.0
        self.last_step_t = 0.0
        self.step_durations = []

    def call_before_loop(self, seed, prompt, latents, config, **kwargs):
        global current_progress
        self.t_start = time.perf_counter()
        self.last_step_t = self.t_start
        self.step_durations = []
        current_progress.update({
            "active": True,
            "step": 0,
            "total_steps": config.num_inference_steps,
            "pct": 0,
            "step_time_s": 0.0,
            "elapsed_s": 0.0,
            "eta_s": 0.0,
            "stage": "diffusion",
            "rate_str": "Starting steps...",
            "preview_b64": None
        })

    def call_in_loop(self, t, seed, prompt, latents, config, time_steps):
        global current_progress
        now = time.perf_counter()
        step_dur = now - self.last_step_t
        self.last_step_t = now
        
        step_num = t + 1
        total = config.num_inference_steps
        elapsed = now - self.t_start
        remaining_steps = total - step_num

        if t > 0:
            self.step_durations.append(step_dur)
            step_rate = sum(self.step_durations) / len(self.step_durations)
        else:
            step_rate = step_dur

        if time_steps and hasattr(time_steps, "format_dict"):
            rate = time_steps.format_dict.get("rate")
            if rate and rate > 0:
                step_rate = 1.0 / rate

        eta = remaining_steps * step_rate
        pct = int((step_num / total) * 90)
        rate_str = f"{step_rate:.2f}s/it" if step_rate >= 1.0 else f"{1.0/step_rate:.1f}it/s"

        preview_b64 = None
        try:
            p = self.get_pipeline()
            lh = config.height // 16
            lw = config.width // 16
            packed = latents.reshape(latents.shape[0], lh, lw, latents.shape[-1]).transpose(0, 3, 1, 2)
            dec = p._model.vae.decode_packed_latents(packed)
            arr = ((mx.clip(dec[0].transpose(1, 2, 0), -1.0, 1.0) + 1.0) * 127.5).astype(mx.uint8)
            img = Image.fromarray(np.array(arr))
            del dec, arr, packed
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=55)
            preview_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception as ex:
            log.warning(f"Preview decode error at step {t}: {ex}")
        finally:
            try:
                mx.clear_cache()
            except Exception:
                pass

        progress_dict = {
            "active": True,
            "step": step_num,
            "total_steps": total,
            "pct": pct,
            "step_time_s": round(step_dur, 2),
            "elapsed_s": round(elapsed, 1),
            "eta_s": round(eta, 1),
            "stage": f"Step {step_num}/{total}",
            "rate_str": rate_str
        }
        if preview_b64:
            progress_dict["preview_b64"] = preview_b64

        current_progress.update(progress_dict)

    def call_after_loop(self, seed, prompt, latents, config):
        global current_progress
        current_progress.update({
            "step": config.num_inference_steps,
            "pct": 95,
            "stage": "Decoding final image...",
            "eta_s": 0.0
        })
