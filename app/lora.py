"""Multi-LoRA linear adapter wrapping and dynamic injection for FLUX.2."""

import os
import mlx.core as mx
import safetensors.torch
from app.config import log


class MultiLoRAWrappedLinear:
    """Dynamically wraps a base linear layer to add multi-LoRA branch outputs."""
    def __init__(self, base_linear, adapters: list[tuple[mx.array, mx.array, float]]):
        while isinstance(base_linear, MultiLoRAWrappedLinear):
            base_linear = base_linear.base_linear
        self.base_linear = base_linear
        self.adapters = [
            (A.astype(mx.bfloat16), B.astype(mx.bfloat16), float(s))
            for A, B, s in adapters
        ]
        self.out_features = base_linear.out_features

    def __call__(self, x: mx.array) -> mx.array:
        base_out = self.base_linear(x)
        orig_shape = x.shape
        flat_x = x.reshape((-1, orig_shape[-1])).astype(mx.bfloat16)
        
        lora_total = mx.zeros((*flat_x.shape[:-1], self.out_features), dtype=mx.bfloat16)
        for lora_A, lora_B, scale in self.adapters:
            lora_total = lora_total + ((flat_x @ lora_A.transpose()) @ lora_B.transpose()) * scale
            
        lora_out = lora_total.reshape((*orig_shape[:-1], self.out_features))
        return base_out + lora_out


def unwrap_linear(linear):
    """Unwraps any nested MultiLoRAWrappedLinear to the original base linear layer."""
    while isinstance(linear, MultiLoRAWrappedLinear):
        linear = linear.base_linear
    return linear


applied_loras_key = None

def apply_loras_to_model(model, loras: list[tuple[str, float]]):
    """Dynamically injects or updates LoRA adapters into transformer megakernel blocks."""
    global applied_loras_key
    valid_loras = [(p, s) for p, s in loras if p and os.path.exists(p) and s > 0]
    target_key = ";".join([f"{p}:{s:.4f}" for p, s in valid_loras]) if valid_loras else None

    if applied_loras_key == target_key:
        return

    from mflux.models.flux2.flux2_initializer import Flux2Initializer
    Flux2Initializer.load_transformer_and_vae(model)
    megakernel = model.transformer.megakernel

    for block in megakernel.double_blocks:
        block.img_qkv = unwrap_linear(block.img_qkv)
        block.txt_qkv = unwrap_linear(block.txt_qkv)
        block.to_out = unwrap_linear(block.to_out)
        block.to_add_out = unwrap_linear(block.to_add_out)
        block.ff_linear_in = unwrap_linear(block.ff_linear_in)
        block.ff_linear_out = unwrap_linear(block.ff_linear_out)
        block.ff_context_linear_in = unwrap_linear(block.ff_context_linear_in)
        block.ff_context_linear_out = unwrap_linear(block.ff_context_linear_out)

    for block in megakernel.single_blocks:
        block.qkv_mlp_proj = unwrap_linear(block.qkv_mlp_proj)
        block.out_proj = unwrap_linear(block.out_proj)

    if valid_loras:
        log.info(f"Applying {len(valid_loras)} Multi-LoRAs: {valid_loras}...")
        loaded_loras_data = []
        for path, scale in valid_loras:
            weights = {}
            with safetensors.safe_open(path, framework="pt") as f:
                for k in f.keys():
                    t = f.get_tensor(k)
                    weights[k] = mx.array(t.float().numpy()).astype(mx.bfloat16)
            loaded_loras_data.append((weights, scale))

        for i, block in enumerate(megakernel.double_blocks):
            pfx = f"diffusion_model.double_blocks.{i}"
            targets = [
                ("img_qkv", f"{pfx}.img_attn.qkv"),
                ("txt_qkv", f"{pfx}.txt_attn.qkv"),
                ("to_out", f"{pfx}.img_attn.proj"),
                ("to_add_out", f"{pfx}.txt_attn.proj"),
                ("ff_linear_in", f"{pfx}.img_mlp.0"),
                ("ff_linear_out", f"{pfx}.img_mlp.2"),
                ("ff_context_linear_in", f"{pfx}.txt_mlp.0"),
                ("ff_context_linear_out", f"{pfx}.txt_mlp.2"),
            ]

            for attr_name, weight_pfx in targets:
                adapters = []
                for w_dict, scale in loaded_loras_data:
                    k_A = f"{weight_pfx}.lora_A.weight"
                    k_B = f"{weight_pfx}.lora_B.weight"
                    if k_A in w_dict and k_B in w_dict:
                        adapters.append((w_dict[k_A], w_dict[k_B], scale))
                if adapters:
                    base_l = getattr(block, attr_name)
                    setattr(block, attr_name, MultiLoRAWrappedLinear(base_l, adapters))

        for i, block in enumerate(megakernel.single_blocks):
            pfx = f"diffusion_model.single_blocks.{i}"
            targets = [
                ("qkv_mlp_proj", f"{pfx}.linear1"),
                ("out_proj", f"{pfx}.linear2"),
            ]

            for attr_name, weight_pfx in targets:
                adapters = []
                for w_dict, scale in loaded_loras_data:
                    k_A = f"{weight_pfx}.lora_A.weight"
                    k_B = f"{weight_pfx}.lora_B.weight"
                    if k_A in w_dict and k_B in w_dict:
                        adapters.append((w_dict[k_A], w_dict[k_B], scale))
                if adapters:
                    base_l = getattr(block, attr_name)
                    setattr(block, attr_name, MultiLoRAWrappedLinear(base_l, adapters))

    def _forward_with_modulations(hidden_states, encoder_hidden_states, temb, rotary_cos, rotary_sin):
        precomputed = megakernel.prepare_all_modulations(temb)
        return megakernel.forward_from_modulations(hidden_states, encoder_hidden_states, precomputed, rotary_cos, rotary_sin)

    model.transformer._compiled_forward = mx.compile(_forward_with_modulations)
    applied_loras_key = target_key
