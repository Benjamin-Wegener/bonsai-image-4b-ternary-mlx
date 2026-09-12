"""Object-level image segmentation using OpenCV watershed + contour merging.

Replaces the previous FLUX.2-generation-based approach which only produced
unreliable colour clusters. OpenCV watershed follows actual object edges so
each region corresponds to a coherent visual object (person, background,
clothing, face, accessories, etc.).

`steps` (1-4) controls granularity:
  1 -> 3  regions  (coarse: subject + background + ground)
  2 -> 5  regions
  3 -> 8  regions
  4 -> 12 regions  (fine: face, hat, jacket, hands, bg, ground, props…)
"""

import base64
import io
import cv2
import numpy as np
from PIL import Image
from app.config import log


PALETTE = [
    (225, 29, 72),    # Rose Red
    (16, 185, 129),   # Emerald Green
    (59, 130, 246),   # Sky Blue
    (234, 179, 8),    # Amber Yellow
    (168, 85, 247),   # Purple
    (6, 182, 212),    # Cyan
    (249, 115, 22),   # Orange
    (236, 72, 153),   # Pink
    (132, 204, 22),   # Lime
    (99, 102, 241),   # Indigo
    (20, 184, 166),   # Teal
    (251, 191, 36),   # Yellow
]

STEPS_TO_REGIONS = {1: 3, 2: 5, 3: 8, 4: 12, 8: 8}


def _fill_watershed_borders(labels: np.ndarray) -> np.ndarray:
    """Replace -1 (watershed boundary) pixels by nearest valid label using cv2 inpainting."""
    result = labels.copy()
    border = (result == -1).astype(np.uint8) * 255
    # Dilate valid labels to bleed into borders
    kernel = np.ones((3, 3), np.uint8)
    for _ in range(8):
        valid_mask = (result > 0).astype(np.uint8)
        dilated = cv2.dilate((result * valid_mask).astype(np.uint16), kernel)
        result[result <= 0] = dilated[result <= 0].astype(np.int32)
        if not (result <= 0).any():
            break
    # Any remaining zeros -> assign 1
    result[result <= 0] = 1
    return result


def _watershed_segment(img_rgb: np.ndarray, k: int) -> np.ndarray:
    """
    Robust edge-aware watershed + Lab gradient segmentation.
    Produces coherent, clean object regions (subject, background, clothes, hair, etc.).
    """
    h, w = img_rgb.shape[:2]
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Edge gradient in Lab color space
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    blurred = cv2.bilateralFilter(lab, d=7, sigmaColor=50, sigmaSpace=50)

    gx = cv2.Sobel(blurred.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt((gx ** 2 + gy ** 2).sum(axis=-1))
    grad_u8 = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Adaptive multi-level thresholding for robust seeds
    markers = np.zeros((h, w), dtype=np.int32)
    
    # 1. Feature-space SLIC-like superpixels for initial seeds
    slic = cv2.ximgproc.createSuperpixelSLIC(img_bgr, algorithm=cv2.ximgproc.SLICO, region_size=max(20, int(min(h, w) / 10)), ruler=20.0) \
        if hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'createSuperpixelSLIC') else None

    if slic is not None:
        slic.iterate(10)
        labels = slic.getLabels()
        num_slic = slic.getNumberOfSuperpixels()
        # Find centroid of each superpixel as watershed seed
        next_lbl = 1
        for sp_id in range(num_slic):
            ys, xs = np.where(labels == sp_id)
            if len(xs) > 30:
                cy, cx = int(np.mean(ys)), int(np.mean(xs))
                markers[cy, cx] = next_lbl
                next_lbl += 1
    else:
        # Distance-transform based peaks + grid fallback
        _, thresh = cv2.threshold(grad_u8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
        dist_norm = cv2.normalize(dist, None, 0.0, 1.0, cv2.NORM_MINMAX)
        
        # Local peaks
        peak_mask = (dist_norm > 0.35).astype(np.uint8) * 255
        n_comp, comp_markers, stats, _ = cv2.connectedComponentsWithStats(peak_mask)
        next_lbl = 1
        min_area = max(30, int(h * w * 0.001))
        for lbl in range(1, n_comp):
            if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
                markers[comp_markers == lbl] = next_lbl
                next_lbl += 1

        if next_lbl < k:
            # Grid seed injection
            cols = int(np.ceil(np.sqrt(k * 1.5)))
            rows = int(np.ceil(k * 1.5 / cols))
            dx = w // (cols + 1)
            dy = h // (rows + 1)
            for r in range(1, rows + 1):
                for c in range(1, cols + 1):
                    cx, cy = c * dx, r * dy
                    if 0 <= cy < h and 0 <= cx < w and markers[cy, cx] == 0:
                        cv2.circle(markers, (cx, cy), 3, next_lbl, -1)
                        next_lbl += 1

    # Run watershed
    ws_markers = markers.copy()
    cv2.watershed(img_bgr, ws_markers)
    ws_markers = _fill_watershed_borders(ws_markers)

    # Merge similar adjacent regions by Lab color similarity until reaching k
    def merge_to_k(lmap, target_k):
        unique = sorted([l for l in np.unique(lmap) if l > 0])
        if len(unique) <= target_k:
            return lmap

        # Calculate mean Lab color for each region
        lab_f = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
        while len(unique) > target_k:
            means = {l: np.mean(lab_f[lmap == l], axis=0) for l in unique}
            sizes = {l: np.sum(lmap == l) for l in unique}
            
            # Find best adjacent neighbor pair to merge (color similarity + small region priority)
            best_pair = None
            min_dist = float('inf')
            
            for l in unique:
                mask = (lmap == l).astype(np.uint8)
                dil = cv2.dilate(mask, np.ones((5, 5), np.uint8))
                neighbors = set(np.unique(lmap[dil.astype(bool)])) - {l, 0}
                for n in neighbors:
                    color_dist = float(np.linalg.norm(means[l] - means[n]))
                    # Bias towards merging tiny regions first
                    weighted_dist = color_dist * (1.0 + (sizes[l] + sizes[n]) / (h * w * 0.5))
                    if weighted_dist < min_dist:
                        min_dist = weighted_dist
                        best_pair = (l, n)

            if not best_pair:
                break
            l_src, l_dst = best_pair
            lmap[lmap == l_src] = l_dst
            unique = sorted([l for l in np.unique(lmap) if l > 0])

        return lmap

    ws_markers = merge_to_k(ws_markers, k)

    # Re-index to 1..N contiguous
    out = np.zeros_like(ws_markers)
    for new_idx, old in enumerate(sorted(set(np.unique(ws_markers)) - {0}), start=1):
        out[ws_markers == old] = new_idx
    return out


def _kmeans_segment(img_rgb: np.ndarray, k: int) -> np.ndarray:
    """
    Spatially-aware k-means fallback using (R,G,B,x,y) features so regions
    stay spatially coherent and don't just group by colour globally.
    """
    h, w = img_rgb.shape[:2]
    y_g, x_g = np.mgrid[0:h, 0:w]
    # Normalise spatial coordinates to same scale as colour (0-255)
    x_n = (x_g / w * 255).astype(np.float32)
    y_n = (y_g / h * 255).astype(np.float32)
    rgb_f = img_rgb.astype(np.float32)
    features = np.stack([
        rgb_f[:, :, 0],
        rgb_f[:, :, 1],
        rgb_f[:, :, 2],
        x_n * 0.5,   # spatial weight
        y_n * 0.5,
    ], axis=-1).reshape(-1, 5)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
    _, labels, _ = cv2.kmeans(features, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    return labels.reshape(h, w).astype(np.int32) + 1


def generate_model_segmentation(img_bytes: bytes, num_regions: int = 8, steps: int = 4) -> dict:
    """
    Object-level image segmentation.  `steps` (1-4) controls granularity.
    Uses watershed for true object boundaries; spatially-aware k-means as fallback.
    """
    k = STEPS_TO_REGIONS.get(steps, num_regions)
    k = max(2, min(12, k))
    log.info(f"Segmenting image → target {k} object regions (steps={steps})")

    img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    orig_w, orig_h = img_pil.size

    MAX_DIM = 768
    scale = min(MAX_DIM / orig_w, MAX_DIM / orig_h, 1.0)
    work_w, work_h = int(orig_w * scale), int(orig_h * scale)
    work_pil = img_pil.resize((work_w, work_h), Image.Resampling.LANCZOS)
    img_rgb = np.array(work_pil, dtype=np.uint8)

    try:
        labels = _watershed_segment(img_rgb, k)
        log.info("Watershed segmentation succeeded")
    except Exception as ex:
        log.warning(f"Watershed failed ({ex}), using spatially-aware k-means")
        labels = _kmeans_segment(img_rgb, k)

    unique = sorted(int(l) for l in np.unique(labels) if l > 0)
    total_px = work_w * work_h

    seg_arr = np.zeros((work_h, work_w, 3), dtype=np.uint8)
    segments_info = []
    for new_idx, lbl in enumerate(unique):
        mask = (labels == lbl)
        count = int(np.sum(mask))
        if count == 0:
            continue
        pct = round(count / total_px * 100, 1)
        rgb = PALETTE[new_idx % len(PALETTE)]
        seg_arr[mask] = rgb
        segments_info.append({
            "id": new_idx,
            "color": f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}",
            "rgb": list(rgb),
            "percentage": pct,
            "label": f"Region {new_idx + 1}",
        })

    segments_info.sort(key=lambda s: s["percentage"], reverse=True)

    seg_pil = Image.fromarray(seg_arr).resize((orig_w, orig_h), Image.Resampling.NEAREST)
    buf = io.BytesIO()
    seg_pil.save(buf, format="PNG")

    return {
        "segmentation_b64": base64.b64encode(buf.getvalue()).decode(),
        "segments": segments_info,
    }
