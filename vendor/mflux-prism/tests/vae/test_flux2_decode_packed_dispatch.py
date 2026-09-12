"""Fast unit tests for Flux2VAE.decode_packed_latents tiling dispatch."""

from unittest.mock import patch

import mlx.core as mx
import pytest

from mflux.models.common.vae.tiling_config import TilingConfig
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE


@pytest.fixture(scope="module")
def vae():
    return Flux2VAE()


@pytest.fixture
def packed_latents():
    return mx.zeros((1, 128, 4, 4), dtype=mx.float32)


@pytest.mark.fast
def test_decode_packed_latents_no_tiling_dispatches_plain_decode(vae, packed_latents):
    decoded_marker = mx.ones((1, 3, 64, 64))
    with (
        patch("mflux.models.flux2.model.flux2_vae.vae.VAEUtil.decode", return_value=decoded_marker) as mock_decode,
    ):
        out = vae.decode_packed_latents(packed_latents)
    assert mock_decode.call_count == 1
    args, kwargs = mock_decode.call_args
    assert kwargs["tiling_config"] is None
    assert out is decoded_marker


@pytest.mark.fast
def test_decode_packed_latents_with_tiling_forwards_config(vae, packed_latents):
    decoded_marker = mx.ones((1, 3, 64, 64))
    tiling_config = TilingConfig(vae_decode_tiles_per_dim=8, vae_decode_overlap=8)
    with (
        patch("mflux.models.flux2.model.flux2_vae.vae.VAEUtil.decode", return_value=decoded_marker) as mock_decode,
    ):
        out = vae.decode_packed_latents(packed_latents, tiling_config=tiling_config)
    assert mock_decode.call_count == 1
    _, kwargs = mock_decode.call_args
    assert kwargs["tiling_config"] is tiling_config
    assert out is decoded_marker
