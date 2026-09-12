"""Parity test: Flux2KleinFastTransformer vs mflux Flux2Transformer.

Builds a tiny transformer in each implementation with matched random weights,
runs the same inputs through both, and verifies outputs agree within bf16
tolerance (atol=1e-2, rtol=1e-2 vs reference magnitude).

Small dims (dim=128, 4 heads, head_dim=32, 2 double + 2 single blocks) so
the test runs in seconds.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from mflux.models.flux2.model.flux2_transformer.klein_fast import (
    DoubleBlockWeights,
    Flux2KleinFastTransformer,
    MegakernelWeights,
    SingleBlockWeights,
)
from mflux.models.flux2.model.flux2_transformer.transformer import Flux2Transformer


LAYER_NORM_EPS = 1e-6
RMS_NORM_EPS = 1e-6


def _random_bf16(shape: tuple[int, ...], seed: int, scale: float = 1.0) -> mx.array:
    rng = np.random.default_rng(seed)
    return mx.array((scale * rng.standard_normal(shape)).astype(np.float32)).astype(mx.bfloat16)


def _random_rms_norm_weight(head_dim: int, seed: int) -> mx.array:
    rng = np.random.default_rng(seed)
    return mx.array((1.0 + 0.05 * rng.standard_normal(head_dim)).astype(np.float32)).astype(mx.bfloat16)


def _xavier_bf16(out_features: int, in_features: int, seed: int) -> mx.array:
    """Weight with std = 1/sqrt(in_features); keeps the megakernel forward well-conditioned."""
    std = 1.0 / math.sqrt(in_features)
    return _random_bf16((out_features, in_features), seed=seed, scale=std)


def _build_random_weights(
    *,
    dim: int,
    num_heads: int,
    head_dim: int,
    mlp_ratio: float,
    in_channels: int,
    context_dim: int,
    patch_out: int,
    num_double: int,
    num_single: int,
    seed: int,
) -> tuple[MegakernelWeights, mx.array, mx.array]:
    """Build MegakernelWeights + shared per-stream modulation weights.

    Returns (weights, shared_mod_img, shared_mod_txt, shared_mod_single).
    mflux keeps a single modulation linear per stream at the transformer level;
    the megakernel duplicates it per-block. We use the same underlying array
    references so outputs match exactly.
    """
    mlp_hidden_dim = int(dim * mlp_ratio)

    shared_mod_img = _xavier_bf16(6 * dim, dim, seed + 1)
    shared_mod_txt = _xavier_bf16(6 * dim, dim, seed + 2)
    shared_mod_single = _xavier_bf16(3 * dim, dim, seed + 3)

    def _double(idx: int) -> DoubleBlockWeights:
        base = seed + 1000 + idx * 100
        return DoubleBlockWeights(
            modulation_img=shared_mod_img,
            modulation_txt=shared_mod_txt,
            to_q=_xavier_bf16(dim, dim, base + 2),
            to_k=_xavier_bf16(dim, dim, base + 3),
            to_v=_xavier_bf16(dim, dim, base + 4),
            add_q_proj=_xavier_bf16(dim, dim, base + 5),
            add_k_proj=_xavier_bf16(dim, dim, base + 6),
            add_v_proj=_xavier_bf16(dim, dim, base + 7),
            to_out=_xavier_bf16(dim, dim, base + 8),
            to_add_out=_xavier_bf16(dim, dim, base + 9),
            ff_linear_in=_xavier_bf16(2 * mlp_hidden_dim, dim, base + 10),
            ff_linear_out=_xavier_bf16(dim, mlp_hidden_dim, base + 11),
            ff_context_linear_in=_xavier_bf16(2 * mlp_hidden_dim, dim, base + 12),
            ff_context_linear_out=_xavier_bf16(dim, mlp_hidden_dim, base + 13),
            norm_q=_random_rms_norm_weight(head_dim, base + 14),
            norm_k=_random_rms_norm_weight(head_dim, base + 15),
            norm_added_q=_random_rms_norm_weight(head_dim, base + 16),
            norm_added_k=_random_rms_norm_weight(head_dim, base + 17),
        )

    def _single(idx: int) -> SingleBlockWeights:
        base = seed + 5000 + idx * 100
        qkv_mlp_out = 3 * dim + 2 * mlp_hidden_dim
        out_proj_in = dim + mlp_hidden_dim
        return SingleBlockWeights(
            modulation=shared_mod_single,
            qkv_mlp_proj=_xavier_bf16(qkv_mlp_out, dim, base + 1),
            out_proj=_xavier_bf16(dim, out_proj_in, base + 2),
            norm_q=_random_rms_norm_weight(head_dim, base + 3),
            norm_k=_random_rms_norm_weight(head_dim, base + 4),
        )

    weights = MegakernelWeights(
        x_embedder=_xavier_bf16(dim, in_channels, seed + 10),
        context_embedder=_xavier_bf16(dim, context_dim, seed + 11),
        norm_out_linear=_xavier_bf16(2 * dim, dim, seed + 12),
        proj_out=_xavier_bf16(patch_out, dim, seed + 13),
        double_block_weights=[_double(i) for i in range(num_double)],
        single_block_weights=[_single(i) for i in range(num_single)],
    )
    return weights, shared_mod_img, shared_mod_txt, shared_mod_single


def _copy_weights_into_reference(
    ref: Flux2Transformer,
    weights: MegakernelWeights,
    shared_mod_img: mx.array,
    shared_mod_txt: mx.array,
    shared_mod_single: mx.array,
    rms_norm_eps: float,
) -> None:
    """Copy megakernel weights into the mflux reference transformer, aligning RMSNorm eps."""
    ref.x_embedder.weight = weights.x_embedder
    ref.context_embedder.weight = weights.context_embedder
    ref.norm_out.linear.weight = weights.norm_out_linear
    ref.proj_out.weight = weights.proj_out

    ref.double_stream_modulation_img.linear.weight = shared_mod_img
    ref.double_stream_modulation_txt.linear.weight = shared_mod_txt
    ref.single_stream_modulation.linear.weight = shared_mod_single

    for block, w in zip(ref.transformer_blocks, weights.double_block_weights):
        block.attn.to_q.weight = w.to_q
        block.attn.to_k.weight = w.to_k
        block.attn.to_v.weight = w.to_v
        block.attn.add_q_proj.weight = w.add_q_proj
        block.attn.add_k_proj.weight = w.add_k_proj
        block.attn.add_v_proj.weight = w.add_v_proj
        block.attn.to_out.weight = w.to_out
        block.attn.to_add_out.weight = w.to_add_out
        block.ff.linear_in.weight = w.ff_linear_in
        block.ff.linear_out.weight = w.ff_linear_out
        block.ff_context.linear_in.weight = w.ff_context_linear_in
        block.ff_context.linear_out.weight = w.ff_context_linear_out
        block.attn.norm_q.weight = w.norm_q
        block.attn.norm_k.weight = w.norm_k
        block.attn.norm_added_q.weight = w.norm_added_q
        block.attn.norm_added_k.weight = w.norm_added_k
        for rms in (block.attn.norm_q, block.attn.norm_k, block.attn.norm_added_q, block.attn.norm_added_k):
            rms.eps = rms_norm_eps

    for block, w in zip(ref.single_transformer_blocks, weights.single_block_weights):
        block.attn.to_qkv_mlp_proj.weight = w.qkv_mlp_proj
        block.attn.to_out.weight = w.out_proj
        block.attn.norm_q.weight = w.norm_q
        block.attn.norm_k.weight = w.norm_k
        for rms in (block.attn.norm_q, block.attn.norm_k):
            rms.eps = rms_norm_eps


def _copy_time_embed_weights(dst: Flux2Transformer | Flux2KleinFastTransformer, src: Flux2Transformer) -> None:
    """Mirror Flux2TimestepGuidanceEmbeddings weights from src onto dst."""
    dst.time_guidance_embed.linear_1.weight = src.time_guidance_embed.linear_1.weight
    dst.time_guidance_embed.linear_2.weight = src.time_guidance_embed.linear_2.weight
    if src.time_guidance_embed.guidance_linear_1 is not None:
        dst.time_guidance_embed.guidance_linear_1.weight = src.time_guidance_embed.guidance_linear_1.weight
        dst.time_guidance_embed.guidance_linear_2.weight = src.time_guidance_embed.guidance_linear_2.weight


def _build_ids(batch_size: int, txt_seq: int, img_h: int, img_w: int) -> tuple[mx.array, mx.array]:
    txt_ids = np.zeros((batch_size, txt_seq, 4), dtype=np.int32)
    txt_ids[:, :, 3] = np.arange(txt_seq, dtype=np.int32)
    img_seq = img_h * img_w
    img_ids = np.zeros((batch_size, img_seq, 4), dtype=np.int32)
    rows = np.repeat(np.arange(img_h, dtype=np.int32), img_w)
    cols = np.tile(np.arange(img_w, dtype=np.int32), img_h)
    img_ids[:, :, 1] = rows
    img_ids[:, :, 2] = cols
    return mx.array(img_ids), mx.array(txt_ids)


@pytest.mark.fast
def test_transformer_parity_against_mflux_reference():
    dim = 128
    num_heads = 4
    head_dim = 32
    mlp_ratio = 3.0
    num_double = 2
    num_single = 2
    in_channels = 16
    context_dim = 24
    timestep_guidance_channels = 32

    batch_size = 1
    txt_seq = 5
    img_h, img_w = 4, 4
    img_seq = img_h * img_w

    axes_dims_rope = (head_dim // 4,) * 4
    rope_theta = 2000
    patch_out = in_channels

    weights, shared_mod_img, shared_mod_txt, shared_mod_single = _build_random_weights(
        dim=dim,
        num_heads=num_heads,
        head_dim=head_dim,
        mlp_ratio=mlp_ratio,
        in_channels=in_channels,
        context_dim=context_dim,
        patch_out=patch_out,
        num_double=num_double,
        num_single=num_single,
        seed=1234,
    )

    ref = Flux2Transformer(
        patch_size=1,
        in_channels=in_channels,
        out_channels=in_channels,
        num_layers=num_double,
        num_single_layers=num_single,
        attention_head_dim=head_dim,
        num_attention_heads=num_heads,
        joint_attention_dim=context_dim,
        timestep_guidance_channels=timestep_guidance_channels,
        mlp_ratio=mlp_ratio,
        axes_dims_rope=axes_dims_rope,
        rope_theta=rope_theta,
        guidance_embeds=False,
    )
    _copy_weights_into_reference(
        ref,
        weights,
        shared_mod_img,
        shared_mod_txt,
        shared_mod_single,
        rms_norm_eps=RMS_NORM_EPS,
    )
    # Re-init timestep embedding weights with a Xavier-scaled random seed so the
    # downstream temb doesn't explode through the MLP.
    ref.time_guidance_embed.linear_1.weight = _xavier_bf16(
        dim, timestep_guidance_channels, seed=9001
    )
    ref.time_guidance_embed.linear_2.weight = _xavier_bf16(dim, dim, seed=9002)

    fast = Flux2KleinFastTransformer(
        weights=weights,
        precision="bf16",
        group_size=64,  # unused in bf16 path
        patch_size=1,
        in_channels=in_channels,
        out_channels=in_channels,
        num_layers=num_double,
        num_single_layers=num_single,
        attention_head_dim=head_dim,
        num_attention_heads=num_heads,
        joint_attention_dim=context_dim,
        timestep_guidance_channels=timestep_guidance_channels,
        mlp_ratio=mlp_ratio,
        axes_dims_rope=axes_dims_rope,
        rope_theta=rope_theta,
        guidance_embeds=False,
        layer_norm_eps=LAYER_NORM_EPS,
        rms_norm_eps=RMS_NORM_EPS,
    )
    _copy_time_embed_weights(fast, ref)

    rng = np.random.default_rng(42)
    hidden_states = mx.array(
        (0.1 * rng.standard_normal((batch_size, img_seq, in_channels))).astype(np.float32)
    ).astype(mx.bfloat16)
    encoder_hidden_states = mx.array(
        (0.1 * rng.standard_normal((batch_size, txt_seq, context_dim))).astype(np.float32)
    ).astype(mx.bfloat16)
    timestep = mx.array(np.array([0.5], dtype=np.float32)).astype(mx.bfloat16)

    img_ids, txt_ids = _build_ids(batch_size, txt_seq, img_h, img_w)

    ref_out = ref(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep,
        img_ids=img_ids,
        txt_ids=txt_ids,
        guidance=None,
    )
    fast_out = fast(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep,
        img_ids=img_ids,
        txt_ids=txt_ids,
        guidance=None,
    )
    mx.eval(ref_out, fast_out)

    assert ref_out.shape == fast_out.shape, f"shape mismatch: ref {ref_out.shape} vs fast {fast_out.shape}"

    atol = 1e-2
    rtol = 1e-2

    diff_f32 = mx.abs(ref_out.astype(mx.float32) - fast_out.astype(mx.float32))
    max_diff = float(mx.max(diff_f32))
    assert math.isfinite(max_diff), f"non-finite output max diff {max_diff}"
    ref_scale = float(mx.max(mx.abs(ref_out.astype(mx.float32))))
    bound = atol + rtol * ref_scale
    assert max_diff <= bound, (
        f"max abs diff {max_diff:.4g} exceeds bound {bound:.4g} (ref scale {ref_scale:.4g})"
    )
