"""Parity test: klein_fast.DoubleFlux2Block vs mflux Flux2TransformerBlock.

Both blocks run with matching bf16 weights, matching LayerNorm / RMSNorm eps,
and matching rotary embeddings. Outputs should agree within bf16 tolerance.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn
import numpy as np
import pytest

from mflux.models.flux2.model.flux2_transformer.pos_embed import Flux2PosEmbed
from mflux.models.flux2.model.flux2_transformer.klein_fast.blocks import (
    DoubleBlockWeights,
    DoubleFlux2Block,
    Flux2KleinBlockSpec,
)
from mflux.models.flux2.model.flux2_transformer.transformer_block import Flux2TransformerBlock


LAYER_NORM_EPS = 1e-6
# Canonical FLUX.2 Klein uses eps=1e-6 for both LayerNorm and RMSNorm (see
# black-forest-labs/FLUX.2-klein-4B transformer/config.json `eps` field, which
# diffusers' Flux2Attention/Flux2TransformerBlock apply to both norm types).
# mflux's Flux2Attention hardcodes RMSNorm(eps=1e-5); `_load_ref_block` overrides
# that to match the spec so parity is checked against canonical values.
RMS_NORM_EPS = 1e-6


def _random_bf16(shape: tuple[int, ...], seed: int, scale: float = 1.0) -> mx.array:
    rng = np.random.default_rng(seed)
    return mx.array((scale * rng.standard_normal(shape)).astype(np.float32)).astype(mx.bfloat16)


def _random_rms_norm_weight(head_dim: int, seed: int) -> mx.array:
    rng = np.random.default_rng(seed)
    return mx.array((1.0 + 0.05 * rng.standard_normal(head_dim)).astype(np.float32)).astype(mx.bfloat16)


def _xavier_bf16(out_features: int, in_features: int, seed: int) -> mx.array:
    """Weight with std = 1/sqrt(in_features); keeps matmul outputs in a well-scaled range.

    Real checkpoints are initialized near this scale; using it here lets us exercise
    the full block with realistic activation magnitudes.
    """
    std = 1.0 / math.sqrt(in_features)
    return _random_bf16((out_features, in_features), seed=seed, scale=std)


def _make_spec(dim: int, num_heads: int, head_dim: int, mlp_ratio: float) -> Flux2KleinBlockSpec:
    return Flux2KleinBlockSpec(
        dim=dim,
        num_heads=num_heads,
        head_dim=head_dim,
        mlp_ratio=mlp_ratio,
        layer_norm_eps=LAYER_NORM_EPS,
        rms_norm_eps=RMS_NORM_EPS,
        rope_theta=2000,
        axes_dims_rope=(head_dim // 4,) * 4,
    )


def _random_double_weights(spec: Flux2KleinBlockSpec, seed_base: int) -> DoubleBlockWeights:
    dim = spec.dim
    mlp_hidden_dim = spec.mlp_hidden_dim
    head_dim = spec.head_dim
    return DoubleBlockWeights(
        modulation_img=_xavier_bf16(spec.modulation_double_out_dim, dim, seed_base + 0),
        modulation_txt=_xavier_bf16(spec.modulation_double_out_dim, dim, seed_base + 1),
        to_q=_xavier_bf16(dim, dim, seed_base + 2),
        to_k=_xavier_bf16(dim, dim, seed_base + 3),
        to_v=_xavier_bf16(dim, dim, seed_base + 4),
        add_q_proj=_xavier_bf16(dim, dim, seed_base + 5),
        add_k_proj=_xavier_bf16(dim, dim, seed_base + 6),
        add_v_proj=_xavier_bf16(dim, dim, seed_base + 7),
        to_out=_xavier_bf16(dim, dim, seed_base + 8),
        to_add_out=_xavier_bf16(dim, dim, seed_base + 9),
        ff_linear_in=_xavier_bf16(2 * mlp_hidden_dim, dim, seed_base + 10),
        ff_linear_out=_xavier_bf16(dim, mlp_hidden_dim, seed_base + 11),
        ff_context_linear_in=_xavier_bf16(2 * mlp_hidden_dim, dim, seed_base + 12),
        ff_context_linear_out=_xavier_bf16(dim, mlp_hidden_dim, seed_base + 13),
        norm_q=_random_rms_norm_weight(head_dim, seed_base + 14),
        norm_k=_random_rms_norm_weight(head_dim, seed_base + 15),
        norm_added_q=_random_rms_norm_weight(head_dim, seed_base + 16),
        norm_added_k=_random_rms_norm_weight(head_dim, seed_base + 17),
    )


def _load_ref_block(ref: Flux2TransformerBlock, spec: Flux2KleinBlockSpec, w: DoubleBlockWeights) -> None:
    """Copy weights into the mflux reference block, overriding RMSNorm eps to match."""
    # Linear weights (all bias-less).
    ref.attn.to_q.weight = w.to_q
    ref.attn.to_k.weight = w.to_k
    ref.attn.to_v.weight = w.to_v
    ref.attn.add_q_proj.weight = w.add_q_proj
    ref.attn.add_k_proj.weight = w.add_k_proj
    ref.attn.add_v_proj.weight = w.add_v_proj
    ref.attn.to_out.weight = w.to_out
    ref.attn.to_add_out.weight = w.to_add_out
    ref.ff.linear_in.weight = w.ff_linear_in
    ref.ff.linear_out.weight = w.ff_linear_out
    ref.ff_context.linear_in.weight = w.ff_context_linear_in
    ref.ff_context.linear_out.weight = w.ff_context_linear_out

    # RMSNorm weights (per head_dim) and eps alignment.
    ref.attn.norm_q.weight = w.norm_q
    ref.attn.norm_k.weight = w.norm_k
    ref.attn.norm_added_q.weight = w.norm_added_q
    ref.attn.norm_added_k.weight = w.norm_added_k
    for rms in (ref.attn.norm_q, ref.attn.norm_k, ref.attn.norm_added_q, ref.attn.norm_added_k):
        rms.eps = spec.rms_norm_eps


def _build_ids(batch_size: int, txt_seq: int, img_h: int, img_w: int) -> mx.array:
    img_seq = img_h * img_w
    ids = np.zeros((batch_size, txt_seq + img_seq, 4), dtype=np.int32)
    # Text tokens: axis 3 holds position.
    ids[:, :txt_seq, 3] = np.arange(txt_seq, dtype=np.int32)
    # Image tokens: axis 1 = row, axis 2 = col.
    rows = np.repeat(np.arange(img_h, dtype=np.int32), img_w)
    cols = np.tile(np.arange(img_w, dtype=np.int32), img_h)
    ids[:, txt_seq:, 1] = rows
    ids[:, txt_seq:, 2] = cols
    return mx.array(ids)


def _build_modulation_params(
    linear_weight: mx.array,
    temb: mx.array,
    dim: int,
) -> tuple[tuple[mx.array, ...], ...]:
    """Modulation math used by both implementations: SiLU(temb) @ W^T -> reshape -> split."""
    temb_3d = temb if temb.ndim == 3 else temb[:, None, :]
    h = mx.matmul(mlx.nn.silu(temb_3d), linear_weight.transpose())
    mod = h.reshape((temb_3d.shape[0], temb_3d.shape[1], 2, 3, dim))
    shift_msa = mod[:, :, 0, 0, :]
    scale_msa = mod[:, :, 0, 1, :]
    gate_msa = mod[:, :, 0, 2, :]
    shift_mlp = mod[:, :, 1, 0, :]
    scale_mlp = mod[:, :, 1, 1, :]
    gate_mlp = mod[:, :, 1, 2, :]
    return (
        (shift_msa, scale_msa, gate_msa),
        (shift_mlp, scale_mlp, gate_mlp),
    )


def _fast_modulation_from_raw(
    raw: tuple[tuple[mx.array, ...], ...],
) -> tuple[tuple[mx.array, ...], ...]:
    """Convert (shift, scale, gate) tuples into the (1+scale, shift, gate) layer_norm form the fast block uses."""
    (shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp) = raw
    return (
        ((1.0 + scale_msa).reshape(-1), shift_msa.reshape(-1), gate_msa.reshape(-1)),
        ((1.0 + scale_mlp).reshape(-1), shift_mlp.reshape(-1), gate_mlp.reshape(-1)),
    )


@pytest.mark.fast
def test_double_block_parity_against_mflux_reference():
    # Small but realistic shapes.
    dim = 128
    num_heads = 4
    head_dim = 32
    mlp_ratio = 3.0
    batch_size = 1
    txt_seq = 5
    img_h, img_w = 4, 4
    img_seq = img_h * img_w

    spec = _make_spec(dim=dim, num_heads=num_heads, head_dim=head_dim, mlp_ratio=mlp_ratio)
    weights = _random_double_weights(spec, seed_base=1000)

    # Reference block (mflux).
    ref = Flux2TransformerBlock(
        dim=dim,
        num_attention_heads=num_heads,
        attention_head_dim=head_dim,
        mlp_ratio=mlp_ratio,
    )
    _load_ref_block(ref, spec, weights)

    # Fast block.
    fast_block = DoubleFlux2Block(
        spec=spec,
        weights=weights,
        precision="bf16",
        group_size=64,  # ignored in bf16 path
    )

    # Inputs: shared bf16 activations.
    hidden_states = _random_bf16((batch_size, img_seq, dim), seed=7)
    encoder_hidden_states = _random_bf16((batch_size, txt_seq, dim), seed=8)
    temb = _random_bf16((batch_size, dim), seed=9)

    # Modulation: compute raw (shift, scale, gate) from the same weights.
    raw_img_mod = _build_modulation_params(weights.modulation_img, temb, dim)
    raw_txt_mod = _build_modulation_params(weights.modulation_txt, temb, dim)

    # RoPE: mflux compact (S, D/2) form.
    ids = _build_ids(batch_size, txt_seq, img_h, img_w)
    pos_embed = Flux2PosEmbed(theta=spec.rope_theta, axes_dim=spec.axes_dims_rope)
    # Flux2PosEmbed accepts 2D ids, so collapse the batch axis (all batches share ids).
    cos, sin = pos_embed(ids[0])

    # Reference forward.
    ref_encoder_out, ref_hidden_out = ref(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        temb_mod_params_img=raw_img_mod,
        temb_mod_params_txt=raw_txt_mod,
        image_rotary_emb=(cos, sin),
    )

    # Fast forward (via forward_from_modulation to reuse the same modulation values).
    fast_img_mod = _fast_modulation_from_raw(raw_img_mod)
    fast_txt_mod = _fast_modulation_from_raw(raw_txt_mod)
    fast_encoder_out, fast_hidden_out = fast_block.forward_from_modulation(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        img_modulation=fast_img_mod,
        txt_modulation=fast_txt_mod,
        rotary_cos=cos,
        rotary_sin=sin,
    )

    mx.eval(ref_encoder_out, ref_hidden_out, fast_encoder_out, fast_hidden_out)

    atol = 1e-2
    rtol = 1e-2

    def _max_abs_diff(a: mx.array, b: mx.array) -> float:
        return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))

    hidden_diff = _max_abs_diff(ref_hidden_out, fast_hidden_out)
    encoder_diff = _max_abs_diff(ref_encoder_out, fast_encoder_out)

    assert math.isfinite(hidden_diff), f"hidden output contains non-finite values (max abs diff = {hidden_diff})"
    assert math.isfinite(encoder_diff), f"encoder output contains non-finite values (max abs diff = {encoder_diff})"

    # Relative check wrt reference magnitude to tolerate bf16 scaling.
    ref_hidden_scale = float(mx.max(mx.abs(ref_hidden_out.astype(mx.float32))))
    ref_encoder_scale = float(mx.max(mx.abs(ref_encoder_out.astype(mx.float32))))
    hidden_bound = atol + rtol * ref_hidden_scale
    encoder_bound = atol + rtol * ref_encoder_scale

    assert hidden_diff <= hidden_bound, (
        f"hidden max abs diff {hidden_diff:.4g} exceeds bound {hidden_bound:.4g} "
        f"(ref scale {ref_hidden_scale:.4g})"
    )
    assert encoder_diff <= encoder_bound, (
        f"encoder max abs diff {encoder_diff:.4g} exceeds bound {encoder_bound:.4g} "
        f"(ref scale {ref_encoder_scale:.4g})"
    )
