"""Invariance tests for `quantize_affine_nbit` (min/max affine asymmetric).

Three canonical inputs are exercised at bits=1, group_size=128:

1. Pure ±s per group (zero-symmetric, two levels). Dequant recovers the input
   up to bf16 storage error on (scales, biases).
2. ±s + b per group (two levels, biased).
3. Continuous bf16 weights — sanity check that the per-group reconstruction
   error is bounded by the per-group range.

(scales, biases) are stored bf16 in the kernel ABI, so absolute tolerances
here are ~max(|w|) * 2^-7 ~ 4e-3 for weights of order 0.5.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mflux.models.flux2.model.flux2_transformer.klein_fast.blocks import (
    quantize_affine_nbit,
)

GROUP_SIZE = 128
BITS = 1


def _dequantize_min_max(packed: mx.array, scales: mx.array, biases: mx.array,
                         *, rows: int, cols: int, bits: int, group_size: int) -> np.ndarray:
    """Reconstruct fp32 weights from (packed, scales, biases) using MLX's
    affine dequant formula: w = q * scale + bias."""
    vals_per_u32 = 32 // bits
    max_q = (1 << bits) - 1
    packed_np = np.asarray(packed)
    scales_np = np.asarray(scales.astype(mx.float32))
    biases_np = np.asarray(biases.astype(mx.float32))

    n_u32 = cols // vals_per_u32
    shifts = (np.arange(vals_per_u32, dtype=np.uint32) * bits)
    q = ((packed_np[..., None] >> shifts) & max_q).reshape(rows, cols)
    n_groups = cols // group_size
    q_grouped = q.reshape(rows, n_groups, group_size).astype(np.float32)
    return (q_grouped * scales_np[:, :, None] + biases_np[:, :, None]).reshape(rows, cols)


@pytest.mark.fast
def test_pure_pm_s_group_recovers_exactly():
    rows, cols = 4, GROUP_SIZE * 3
    rng = np.random.default_rng(0)
    n_groups = cols // GROUP_SIZE
    s_per_group = rng.uniform(0.01, 0.5, size=(rows, n_groups)).astype(np.float32)
    signs = rng.choice([-1.0, 1.0], size=(rows, n_groups, GROUP_SIZE)).astype(np.float32)
    w = (signs * s_per_group[:, :, None]).reshape(rows, cols)

    packed, scales, biases = quantize_affine_nbit(
        mx.array(w), bits=BITS, group_size=GROUP_SIZE
    )
    recon = _dequantize_min_max(
        packed, scales, biases, rows=rows, cols=cols, bits=BITS, group_size=GROUP_SIZE
    )
    # bf16 mantissa is 7 bits → relative error ~2^-7 on the stored scale.
    np.testing.assert_allclose(recon, w, rtol=1e-2, atol=0)
    # Sign correctness is exact (q picks the right level).
    assert (np.sign(recon) == np.sign(w)).all()


@pytest.mark.fast
def test_pm_s_plus_b_group_recovers_exactly():
    rows, cols = 4, GROUP_SIZE * 3
    rng = np.random.default_rng(1)
    n_groups = cols // GROUP_SIZE
    s_per_group = rng.uniform(0.01, 0.3, size=(rows, n_groups)).astype(np.float32)
    b_per_group = rng.uniform(-0.2, 0.2, size=(rows, n_groups)).astype(np.float32)
    signs = rng.choice([-1.0, 1.0], size=(rows, n_groups, GROUP_SIZE)).astype(np.float32)
    w = (signs * s_per_group[:, :, None] + b_per_group[:, :, None]).reshape(rows, cols)

    packed, scales, biases = quantize_affine_nbit(
        mx.array(w), bits=BITS, group_size=GROUP_SIZE
    )
    recon = _dequantize_min_max(
        packed, scales, biases, rows=rows, cols=cols, bits=BITS, group_size=GROUP_SIZE
    )
    # bf16 storage on (scales, biases). Recon must be close to w in fp32. atol
    # scales with max(|w|) since bf16 rounding error is multiplicative.
    np.testing.assert_allclose(recon, w, rtol=0, atol=np.abs(w).max() * 2 ** -7)
    # The two levels per group must be picked correctly (no sign-of-(w-b) flips).
    grouped_w = w.reshape(rows, n_groups, GROUP_SIZE)
    grouped_recon = recon.reshape(rows, n_groups, GROUP_SIZE)
    centered_w = grouped_w - b_per_group[:, :, None]
    centered_recon = grouped_recon - b_per_group[:, :, None]
    assert (np.sign(centered_recon) == np.sign(centered_w)).all()


@pytest.mark.fast
def test_continuous_weights_recon_within_per_group_range():
    rows, cols = 4, GROUP_SIZE * 3
    rng = np.random.default_rng(2)
    w = rng.standard_normal(size=(rows, cols)).astype(np.float32) * 0.05

    packed, scales, biases = quantize_affine_nbit(
        mx.array(w), bits=BITS, group_size=GROUP_SIZE
    )
    recon = _dequantize_min_max(
        packed, scales, biases, rows=rows, cols=cols, bits=BITS, group_size=GROUP_SIZE
    )
    n_groups = cols // GROUP_SIZE
    grouped = w.reshape(rows, n_groups, GROUP_SIZE)
    per_group_range = grouped.max(axis=2) - grouped.min(axis=2)
    err = np.abs(recon - w).reshape(rows, n_groups, GROUP_SIZE)
    # Each value rounds to either group min or group max → element error
    # bounded by the per-group range plus a slack for bf16 storage.
    bf16_slack = np.abs(grouped).max() * 2 ** -7
    assert (err <= per_group_range[:, :, None] + bf16_slack).all()
    assert err.mean() < 0.5 * per_group_range.mean()


@pytest.mark.fast
def test_pm_s_plus_b_packed_qbits_pick_correct_level():
    """Sanity: for ±s+b inputs, q must be 0 where w = -s+b and 1 where w = +s+b."""
    rows, cols = 1, GROUP_SIZE
    s, b = 0.07, 0.04
    signs = np.array([1.0, -1.0] * (GROUP_SIZE // 2), dtype=np.float32).reshape(1, GROUP_SIZE)
    w = signs * s + b
    packed, scales, biases = quantize_affine_nbit(
        mx.array(w), bits=BITS, group_size=GROUP_SIZE
    )
    recon = _dequantize_min_max(
        packed, scales, biases, rows=rows, cols=cols, bits=BITS, group_size=GROUP_SIZE
    )
    # bf16 storage on (scales, biases): rtol ~1%
    np.testing.assert_allclose(recon, w, rtol=1e-2, atol=0)
    # Scales = 2s, biases = -s+b — bf16-rounded.
    np.testing.assert_allclose(np.asarray(scales.astype(mx.float32)), [[2 * s]], rtol=1e-2, atol=0)
    np.testing.assert_allclose(np.asarray(biases.astype(mx.float32)), [[-s + b]], rtol=1e-2, atol=0)
    # q picks the correct level: w[i] = +s+b → q=1, w[i] = -s+b → q=0.
    vals_per_u32 = 32 // BITS
    packed_np = np.asarray(packed)
    shifts = (np.arange(vals_per_u32, dtype=np.uint32) * BITS)
    q = ((packed_np[..., None] >> shifts) & 1).reshape(rows, cols)
    expected_q = (signs > 0).astype(np.uint32)
    assert (q == expected_q).all()
