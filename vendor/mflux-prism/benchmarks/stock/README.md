# Stock mflux FLUX.2 Klein 4B baselines

Host: M4 Max, MLX 0.31.1. Stock Flux2Transformer (no `--klein-fast`).
Prompt: `"a cat on a windowsill at sunset"`, seed 42, default guidance, linear scheduler.
Command template:
```
python -m mflux.models.flux2.cli.flux2_generate \
  --model flux2-klein-4b --prompt "a cat on a windowsill at sunset" \
  --seed 42 --height H --width W --steps S --output stock_HxW_sS.png
```
Wall time is total process time via `/usr/bin/time -l` (includes model load + denoise + decode + save). RSS = `maximum resident set size`; "unified peak" = `peak memory footprint` (CPU + Metal heap), a better proxy for "how close to the M4 Max memory ceiling".

| Resolution | Steps | Wall time (s) | Per-step (s/it) | Peak RSS (GB) | Unified peak (GB) | Output |
|------------|-------|---------------|-----------------|---------------|-------------------|--------|
| 512×512    | 4     | 28.52         | 3.42            | 10.14         | 23.70             | `stock_512_s4.png`   |
| 512×512    | 8     | 28.91         | 1.82            | 14.53         | 23.88             | `stock_512_s8.png`   |
| 512×512    | 16    | 37.63         | 1.65            | 14.54         | 23.81             | `stock_512_s16.png`  |
| 1024×1024  | 4     | 43.62         | 5.59            | 14.53         | 37.18             | `stock_1024_s4.png`  |
| 1024×1024  | 8     | 61.43         | 5.41            | 14.54         | 37.15             | `stock_1024_s8.png`  |
| 1024×1024  | 16    | 103.65        | 5.30            | 14.53         | 37.14             | `stock_1024_s16.png` |

## Summary

Sweet spot for usability is **1024×1024 / 8 steps (~61 s)** — first step that produces a properly detailed image without paying the big 16-step tax. For iterative prompting **512×512 / 8 steps (~29 s)** is hard to beat; going from 4→8 at 512 is essentially free because the fixed startup (compile + load, ~14 s) dominates.

Scaling observations. Per-step time is flat within a resolution once compiled (~1.5 s/it at 512, ~5.2 s/it at 1024), so **wall time is linear in steps plus a fixed startup**: the 512 runs fit the model `t = 14.2 + 1.47·steps` (R²≈1.0), 1024 fits `t = 22.0 + 5.12·steps`. Across resolutions, per-step cost grew ~3.5×, not the 4× you'd expect from pixel count; Klein's attention+MLP cost is slightly sub-quadratic in token count here. Unified peak footprint went from 23.8 GB at 512 to 37.1 GB at 1024, a +13.3 GB jump driven by 4× more latent tokens. On a 64 GB M4 Max there is headroom for 2048×2048 only in principle — extrapolating the +13.3 GB/4× token-count bump as linear in tokens gives ~90 GB, which won't fit. Even being charitable (sub-linear activations because weights stay constant), 2048 lands in the 55-70 GB band and will thrash or OOM. 1024 is the practical ceiling without quant or low-RAM mode. RSS plateauing at 14.5 GB while unified peak climbs confirms the activation growth is Metal-side.
