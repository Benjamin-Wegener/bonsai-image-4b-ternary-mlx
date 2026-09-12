"""Smoke test for the klein_fast weight loader.

Two tests:

1. `test_loader_maps_fake_hf_dict_to_megakernel_weights` — fast unit test that
   builds a synthetic HF-style dict with the full Klein key scheme at small
   shapes, feeds it through the loader's pure-mapping entry point, and checks
   every MegakernelWeights field is populated with the expected shape/dtype.
   Also verifies the shared modulation arrays are referenced (not copied).

2. `test_load_real_klein_weights_and_compare_against_reference_transformer` —
   marked `@pytest.mark.slow` and `@pytest.mark.high_memory_requirement` so it
   is deselected by default (`addopts = '... -m "not high_memory_requirement"'`
   in the project's pyproject.toml). It runs end-to-end against the real HF
   Klein snapshot if already cached locally, otherwise skips with a message.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mflux.models.flux2.model.flux2_transformer.klein_fast import (
    Flux2KleinFastTransformer,
    Flux2KleinMegakernelSpec,
    MegakernelWeights,
    PackedWeight,
    double_block_weight_keys,
    find_packed_artifact_dir,
    load_klein_fast_packed_weights_from_disk,
    load_klein_fast_weights_from_hf,
    single_block_weight_keys,
)
from mflux.models.flux2.model.flux2_transformer.klein_fast.loader import (
    _GLOBAL_KEYS,
    _SHARED_MODULATION_KEYS,
    _build_megakernel_weights_from_packed,
    _build_megakernel_weights_from_raw,
)


# ---------------------------------------------------------------------------
# Unit test: fake HF dict -> MegakernelWeights mapping
# ---------------------------------------------------------------------------


def _make_fake_hf_dict(spec: Flux2KleinMegakernelSpec, in_channels: int, context_dim: int) -> dict[str, mx.array]:
    """Construct a dict keyed exactly like the HF FLUX.2 Klein checkpoint."""
    dim = spec.dim
    mlp_hidden = int(dim * spec.mlp_ratio)
    out_channels = in_channels

    rng = np.random.default_rng(0)

    def arr(shape: tuple[int, ...]) -> mx.array:
        return mx.array(rng.standard_normal(shape).astype(np.float32)).astype(mx.bfloat16)

    raw: dict[str, mx.array] = {
        _GLOBAL_KEYS["x_embedder"]: arr((dim, in_channels)),
        _GLOBAL_KEYS["context_embedder"]: arr((dim, context_dim)),
        _GLOBAL_KEYS["norm_out_linear"]: arr((2 * dim, dim)),
        _GLOBAL_KEYS["proj_out"]: arr((out_channels, dim)),
        _SHARED_MODULATION_KEYS["double_mod_img"]: arr((6 * dim, dim)),
        _SHARED_MODULATION_KEYS["double_mod_txt"]: arr((6 * dim, dim)),
        _SHARED_MODULATION_KEYS["single_mod"]: arr((3 * dim, dim)),
    }

    for i in range(spec.num_double_blocks):
        keys = double_block_weight_keys(i)
        shapes = {
            "to_q": (dim, dim),
            "to_k": (dim, dim),
            "to_v": (dim, dim),
            "add_q_proj": (dim, dim),
            "add_k_proj": (dim, dim),
            "add_v_proj": (dim, dim),
            "to_out": (dim, dim),
            "to_add_out": (dim, dim),
            "ff_linear_in": (2 * mlp_hidden, dim),
            "ff_linear_out": (dim, mlp_hidden),
            "ff_context_linear_in": (2 * mlp_hidden, dim),
            "ff_context_linear_out": (dim, mlp_hidden),
            "norm_q": (spec.head_dim,),
            "norm_k": (spec.head_dim,),
            "norm_added_q": (spec.head_dim,),
            "norm_added_k": (spec.head_dim,),
        }
        for field_name, shape in shapes.items():
            raw[keys[field_name]] = arr(shape)

    for i in range(spec.num_single_blocks):
        keys = single_block_weight_keys(i)
        qkv_mlp_out = 3 * dim + 2 * mlp_hidden
        out_proj_in = dim + mlp_hidden
        shapes = {
            "qkv_mlp_proj": (qkv_mlp_out, dim),
            "out_proj": (dim, out_proj_in),
            "norm_q": (spec.head_dim,),
            "norm_k": (spec.head_dim,),
        }
        for field_name, shape in shapes.items():
            raw[keys[field_name]] = arr(shape)

    return raw


@pytest.mark.fast
def test_loader_maps_fake_hf_dict_to_megakernel_weights():
    # Tiny spec exercising every code path without allocating Klein-sized tensors.
    spec = Flux2KleinMegakernelSpec(
        num_double_blocks=2,
        num_single_blocks=3,
        dim=64,
        num_heads=4,
        head_dim=16,
        mlp_ratio=3.0,
        in_channels=8,
        context_dim=12,
    )
    raw = _make_fake_hf_dict(spec, in_channels=spec.in_channels, context_dim=spec.context_dim)

    weights = _build_megakernel_weights_from_raw(raw, spec, dtype=mx.bfloat16)

    assert isinstance(weights, MegakernelWeights)
    assert weights.x_embedder.shape == (spec.dim, spec.in_channels)
    assert weights.context_embedder.shape == (spec.dim, spec.context_dim)
    assert weights.norm_out_linear.shape == (2 * spec.dim, spec.dim)
    assert weights.proj_out.shape == (spec.in_channels, spec.dim)
    for arr_ in (weights.x_embedder, weights.context_embedder, weights.norm_out_linear, weights.proj_out):
        assert arr_.dtype == mx.bfloat16

    assert len(weights.double_block_weights) == spec.num_double_blocks
    assert len(weights.single_block_weights) == spec.num_single_blocks

    # Shared modulation references — the megakernel stores the same underlying
    # array in every block. Compare identity via Python `is`.
    shared_img = weights.double_block_weights[0].modulation_img
    shared_txt = weights.double_block_weights[0].modulation_txt
    for block_w in weights.double_block_weights:
        assert block_w.modulation_img is shared_img
        assert block_w.modulation_txt is shared_txt

    shared_single = weights.single_block_weights[0].modulation
    for block_w in weights.single_block_weights:
        assert block_w.modulation is shared_single

    # Spot-check per-block linear shape.
    mlp_hidden = int(spec.dim * spec.mlp_ratio)
    db0 = weights.double_block_weights[0]
    assert db0.to_q.shape == (spec.dim, spec.dim)
    assert db0.ff_linear_in.shape == (2 * mlp_hidden, spec.dim)
    assert db0.ff_linear_out.shape == (spec.dim, mlp_hidden)
    assert db0.norm_q.shape == (spec.head_dim,)

    sb0 = weights.single_block_weights[0]
    assert sb0.qkv_mlp_proj.shape == (3 * spec.dim + 2 * mlp_hidden, spec.dim)
    assert sb0.out_proj.shape == (spec.dim, spec.dim + mlp_hidden)


@pytest.mark.fast
def test_loader_raises_on_missing_key():
    spec = Flux2KleinMegakernelSpec(
        num_double_blocks=1,
        num_single_blocks=1,
        dim=64,
        num_heads=4,
        head_dim=16,
        in_channels=8,
        context_dim=12,
    )
    raw = _make_fake_hf_dict(spec, in_channels=spec.in_channels, context_dim=spec.context_dim)
    del raw["x_embedder.weight"]
    with pytest.raises(KeyError, match="x_embedder.weight"):
        _build_megakernel_weights_from_raw(raw, spec, dtype=mx.bfloat16)


# ---------------------------------------------------------------------------
# Packed loader: artifact built externally
# ---------------------------------------------------------------------------


_BLOCK_LINEAR_FIELDS = frozenset({
    "to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj",
    "to_out", "to_add_out",
    "ff_linear_in", "ff_linear_out", "ff_context_linear_in", "ff_context_linear_out",
    "qkv_mlp_proj", "out_proj",
})


def _pack_fake_hf_dict(raw: dict[str, mx.array], spec: Flux2KleinMegakernelSpec, group_size: int) -> dict[str, mx.array]:
    """Pack every block linear in `raw`; pass through skip-pattern keys as bf16."""
    from mflux.models.flux2.model.flux2_transformer.klein_fast.blocks import quantize_affine_nbit

    packed: dict[str, mx.array] = dict(raw)

    block_weight_keys: set[str] = set()
    for i in range(spec.num_double_blocks):
        for field, hf_key in double_block_weight_keys(i).items():
            if field in _BLOCK_LINEAR_FIELDS:
                block_weight_keys.add(hf_key)
    for i in range(spec.num_single_blocks):
        for field, hf_key in single_block_weight_keys(i).items():
            if field in _BLOCK_LINEAR_FIELDS:
                block_weight_keys.add(hf_key)

    for hf_key in block_weight_keys:
        w = raw[hf_key]
        pw, scales, biases = quantize_affine_nbit(w, bits=1, group_size=group_size)
        prefix = hf_key[: -len(".weight")]
        packed[hf_key] = pw
        packed[f"{prefix}.scales"] = scales
        packed[f"{prefix}.biases"] = biases
    return packed


@pytest.mark.fast
def test_packed_loader_returns_packed_weights_for_block_linears():
    spec = Flux2KleinMegakernelSpec(
        num_double_blocks=2,
        num_single_blocks=3,
        dim=128,
        num_heads=4,
        head_dim=32,
        mlp_ratio=3.0,
        in_channels=8,
        context_dim=12,
    )
    group_size = 128
    raw = _make_fake_hf_dict(spec, in_channels=spec.in_channels, context_dim=spec.context_dim)
    packed = _pack_fake_hf_dict(raw, spec, group_size=group_size)

    weights = _build_megakernel_weights_from_packed(
        packed, spec, dtype=mx.bfloat16, bits=1, group_size=group_size
    )

    assert isinstance(weights, MegakernelWeights)
    assert weights.x_embedder.dtype == mx.bfloat16
    assert weights.proj_out.dtype == mx.bfloat16

    db0 = weights.double_block_weights[0]
    assert isinstance(db0.to_q, PackedWeight)
    assert db0.to_q.bits == 1
    assert db0.to_q.group_size == group_size
    assert db0.to_q.packed.dtype == mx.uint32
    assert db0.to_q.scales.dtype == mx.bfloat16
    assert db0.to_q.biases.dtype == mx.bfloat16
    assert db0.to_q.packed.shape == (spec.dim, spec.dim // 32)
    assert db0.to_q.scales.shape == (spec.dim, spec.dim // group_size)
    assert isinstance(db0.ff_linear_in, PackedWeight)
    # Norm weights stay as plain bf16 arrays.
    assert isinstance(db0.norm_q, mx.array)
    assert db0.norm_q.dtype == mx.bfloat16
    # Shared modulation arrays are also plain bf16.
    assert isinstance(db0.modulation_img, mx.array)

    sb0 = weights.single_block_weights[0]
    assert isinstance(sb0.qkv_mlp_proj, PackedWeight)
    assert isinstance(sb0.out_proj, PackedWeight)
    assert isinstance(sb0.modulation, mx.array)


@pytest.mark.fast
def test_packed_loader_raises_on_dangling_scales_without_biases():
    spec = Flux2KleinMegakernelSpec(
        num_double_blocks=1,
        num_single_blocks=1,
        dim=128,
        num_heads=4,
        head_dim=32,
        in_channels=8,
        context_dim=12,
    )
    group_size = 128
    raw = _make_fake_hf_dict(spec, in_channels=spec.in_channels, context_dim=spec.context_dim)
    packed = _pack_fake_hf_dict(raw, spec, group_size=group_size)

    victim = double_block_weight_keys(0)["to_q"]
    prefix = victim[: -len(".weight")]
    del packed[f"{prefix}.biases"]
    with pytest.raises(KeyError, match="biases"):
        _build_megakernel_weights_from_packed(
            packed, spec, dtype=mx.bfloat16, bits=1, group_size=group_size
        )


@pytest.mark.fast
def test_find_packed_artifact_dir(tmp_path):
    # No packed dir present yet.
    assert find_packed_artifact_dir(tmp_path) is None

    packed_dir = tmp_path / "transformer-packed-mflux"
    packed_dir.mkdir()
    # Dir exists but missing quantization_config.json → still None.
    assert find_packed_artifact_dir(tmp_path) is None

    (packed_dir / "quantization_config.json").write_text("{}")
    found = find_packed_artifact_dir(tmp_path)
    assert found == packed_dir


@pytest.mark.fast
def test_load_klein_fast_packed_weights_from_disk_end_to_end(tmp_path):
    """Write a synthetic packed artifact to disk, read it back, verify types."""
    import json as _json

    spec = Flux2KleinMegakernelSpec(
        num_double_blocks=1,
        num_single_blocks=1,
        dim=128,
        num_heads=4,
        head_dim=32,
        mlp_ratio=3.0,
        in_channels=8,
        context_dim=12,
    )
    group_size = 128
    raw = _make_fake_hf_dict(spec, in_channels=spec.in_channels, context_dim=spec.context_dim)
    packed = _pack_fake_hf_dict(raw, spec, group_size=group_size)

    packed_dir = tmp_path / "transformer-packed-mflux"
    packed_dir.mkdir()
    (packed_dir / "quantization_config.json").write_text(_json.dumps({"bits": 1, "group_size": group_size}))
    mx.save_safetensors(str(packed_dir / "diffusion_pytorch_model.safetensors"), packed)

    weights = load_klein_fast_packed_weights_from_disk(packed_dir, spec, dtype=mx.bfloat16)
    assert isinstance(weights.double_block_weights[0].to_q, PackedWeight)
    assert isinstance(weights.single_block_weights[0].qkv_mlp_proj, PackedWeight)


# ---------------------------------------------------------------------------
# Slow / high-memory: real Klein checkpoint load + forward parity
# ---------------------------------------------------------------------------


_HF_KLEIN_REPO_ID = "black-forest-labs/FLUX.2-klein-4B"


def _find_cached_klein_transformer_dir() -> Path | None:
    """Locate a cached FLUX.2 Klein transformer directory without downloading.

    Preference order:
      1. `$KLEIN_TRANSFORMER_DIR` (explicit override for CI / custom setups).
      2. Standard HF hub cache, searching `models--<org>--<repo>/snapshots/*/transformer`.
    """
    override = os.environ.get("KLEIN_TRANSFORMER_DIR")
    if override:
        p = Path(override).expanduser()
        return p if p.exists() else None

    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except ImportError:
        return None

    repo_cache = Path(HF_HUB_CACHE) / f"models--{_HF_KLEIN_REPO_ID.replace('/', '--')}" / "snapshots"
    if not repo_cache.exists():
        return None

    for snapshot in sorted(repo_cache.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        transformer_dir = snapshot / "transformer"
        if transformer_dir.exists() and any(
            f.suffix == ".safetensors" and not f.name.startswith("._") for f in transformer_dir.iterdir()
        ):
            return transformer_dir

    return None


@pytest.mark.slow
@pytest.mark.high_memory_requirement
def test_load_real_klein_weights_and_compare_against_reference_transformer():
    """End-to-end: load real Klein weights, run klein_fast + mflux reference, compare."""
    transformer_dir = _find_cached_klein_transformer_dir()
    if transformer_dir is None:
        pytest.skip(
            f"FLUX.2 Klein transformer weights not cached locally. Set "
            f"KLEIN_TRANSFORMER_DIR=/path/to/transformer or run `huggingface-cli "
            f"download {_HF_KLEIN_REPO_ID} --include 'transformer/*'` first."
        )

    spec = Flux2KleinMegakernelSpec(
        num_double_blocks=5,
        num_single_blocks=20,
        dim=3072,
        num_heads=24,
        head_dim=128,
        mlp_ratio=3.0,
        layer_norm_eps=1e-6,
        rms_norm_eps=1e-6,
        rope_theta=2000,
        axes_dims_rope=(32, 32, 32, 32),
        in_channels=128,
        context_dim=7680,
    )

    weights = load_klein_fast_weights_from_hf(transformer_dir, spec, dtype=mx.bfloat16)

    assert weights.x_embedder.shape == (spec.dim, spec.in_channels)
    assert weights.proj_out.shape == (spec.in_channels, spec.dim)
    assert len(weights.double_block_weights) == spec.num_double_blocks
    assert len(weights.single_block_weights) == spec.num_single_blocks

    fast = Flux2KleinFastTransformer(
        weights=weights,
        precision="bf16",
        group_size=64,
        patch_size=1,
        in_channels=spec.in_channels,
        out_channels=spec.in_channels,
        num_layers=spec.num_double_blocks,
        num_single_layers=spec.num_single_blocks,
        attention_head_dim=spec.head_dim,
        num_attention_heads=spec.num_heads,
        joint_attention_dim=spec.context_dim,
        timestep_guidance_channels=256,
        mlp_ratio=spec.mlp_ratio,
        axes_dims_rope=spec.axes_dims_rope,
        rope_theta=spec.rope_theta,
        guidance_embeds=False,
        layer_norm_eps=spec.layer_norm_eps,
        rms_norm_eps=spec.rms_norm_eps,
    )

    # Mirror time-guidance embedding weights from the HF checkpoint onto the
    # fast transformer (x_embedder / context_embedder / norm_out_linear /
    # proj_out are already wired through MegakernelWeights; time_guidance_embed
    # is owned by the wrapper and is not part of the megakernel).
    raw = {}
    for shard in transformer_dir.glob("*.safetensors"):
        if not shard.name.startswith("._"):
            raw.update(mx.load(str(shard)))
    fast.time_guidance_embed.linear_1.weight = raw[
        "time_guidance_embed.timestep_embedder.linear_1.weight"
    ].astype(mx.bfloat16)
    fast.time_guidance_embed.linear_2.weight = raw[
        "time_guidance_embed.timestep_embedder.linear_2.weight"
    ].astype(mx.bfloat16)

    # Tiny synthetic inputs — keep sequence lengths short so attention fits in
    # memory alongside the full Klein weight set.
    batch_size = 1
    txt_seq = 4
    img_h = img_w = 4
    img_seq = img_h * img_w

    rng = np.random.default_rng(7)
    hidden_states = mx.array(
        (0.1 * rng.standard_normal((batch_size, img_seq, spec.in_channels))).astype(np.float32)
    ).astype(mx.bfloat16)
    encoder_hidden_states = mx.array(
        (0.1 * rng.standard_normal((batch_size, txt_seq, spec.context_dim))).astype(np.float32)
    ).astype(mx.bfloat16)
    timestep = mx.array(np.array([0.5], dtype=np.float32)).astype(mx.bfloat16)

    txt_ids = np.zeros((batch_size, txt_seq, 4), dtype=np.int32)
    txt_ids[:, :, 3] = np.arange(txt_seq, dtype=np.int32)
    img_ids = np.zeros((batch_size, img_seq, 4), dtype=np.int32)
    img_ids[:, :, 1] = np.repeat(np.arange(img_h, dtype=np.int32), img_w)
    img_ids[:, :, 2] = np.tile(np.arange(img_w, dtype=np.int32), img_h)
    img_ids_mx = mx.array(img_ids)
    txt_ids_mx = mx.array(txt_ids)

    # Load the mflux reference transformer with the same checkpoint via the
    # standard initializer. We only need the transformer's output to compare.
    from mflux.models.common.weights.loading.weight_loader import WeightLoader
    from mflux.models.flux2.model.flux2_transformer.transformer import Flux2Transformer
    from mflux.models.flux2.weights.flux2_weight_definition import Flux2KleinWeightDefinition

    loaded = WeightLoader.load(
        weight_definition=Flux2KleinWeightDefinition,
        model_path=str(transformer_dir.parent),
    )
    ref = Flux2Transformer()
    ref.update(loaded.components["transformer"])
    # mflux's Flux2Attention hardcodes RMSNorm(eps=1e-5); klein_fast uses 1e-6
    # per the canonical Klein config. Align both for a clean parity comparison.
    for block in ref.transformer_blocks:
        for rms in (block.attn.norm_q, block.attn.norm_k, block.attn.norm_added_q, block.attn.norm_added_k):
            rms.eps = spec.rms_norm_eps
    for block in ref.single_transformer_blocks:
        for rms in (block.attn.norm_q, block.attn.norm_k):
            rms.eps = spec.rms_norm_eps

    ref_out = ref(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep,
        img_ids=img_ids_mx,
        txt_ids=txt_ids_mx,
        guidance=None,
    )
    fast_out = fast(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep,
        img_ids=img_ids_mx,
        txt_ids=txt_ids_mx,
        guidance=None,
    )
    mx.eval(ref_out, fast_out)

    assert ref_out.shape == fast_out.shape
    atol = 1e-2
    # bf16 non-associativity over 25 blocks with Klein's large intermediate magnitudes
    # (bf16's 7-bit mantissa accumulates ~2x more error than fp16's 10-bit over 25 blocks)
    rtol = 7e-2
    diff = mx.abs(ref_out.astype(mx.float32) - fast_out.astype(mx.float32))
    max_diff = float(mx.max(diff))
    ref_scale = float(mx.max(mx.abs(ref_out.astype(mx.float32))))
    bound = atol + rtol * ref_scale
    assert math.isfinite(max_diff)
    assert max_diff <= bound, (
        f"max abs diff {max_diff:.4g} exceeds bound {bound:.4g} (ref scale {ref_scale:.4g})"
    )
