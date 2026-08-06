"""Scaled FP8 quantization matching ComfyUI's scaled_fp8 checkpoint convention.

Format reference: city96/ComfyUI-GGUF comfy/quant_ops.py QUANT_ALGOS["float8_e4m3fn"].
Per layer: <name> stored as torch.float8_e4m3fn, <name>.weight_scale as a
float32 scalar such that original ≈ stored.to(float32) * weight_scale.
"""

from __future__ import annotations

import torch

_FP8_MAX = 448.0  # float8_e4m3fn representable magnitude


def quantize_fp8_scaled(data: torch.Tensor, key: str) -> dict[str, torch.Tensor]:
    """Quantize one tensor to scaled float8_e4m3fn.

    Returns {key: fp8_tensor, f"{key}.weight_scale": float32 scalar}.
    """
    amax = data.abs().max()
    scale = (amax / _FP8_MAX) if amax > 0 else torch.tensor(1.0, dtype=torch.float32)
    scale = scale.to(torch.float32)
    scaled = (data.to(torch.float32) / scale).clamp(-_FP8_MAX, _FP8_MAX)
    return {
        key: scaled.to(torch.float8_e4m3fn),
        f"{key}.weight_scale": scale.reshape(1),
    }
