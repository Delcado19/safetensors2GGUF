"""Nvidia NVFP4 block-scaled quantization for safetensors output.

Format reference: city96/ComfyUI-GGUF comfy/quant_ops.py QUANT_ALGOS["nvfp4"]
(group_size=16, storage_t=uint8, params={weight_scale, weight_scale_2}) — the
same TensorRT-Model-Optimizer convention ComfyUI's native NVFP4 loader expects.
The E2M1 decode table is the same one gguf.quants.NVFP4 uses internally, kept
in sync by using the exact same kvalues tuple (avoids two independent, and
possibly diverging, 4-bit float codebooks in this repo).
"""

from __future__ import annotations

import torch

GROUP_SIZE = 16
# e2m1 values doubled — identical table to gguf.quants.NVFP4.kvalues
_KVALUES = torch.tensor(
    [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=torch.float32
)
_FP8_MAX = 448.0


def _nearest_e2m1_index(x: torch.Tensor) -> torch.Tensor:
    """Map float values (already divided by the block scale) to the nearest
    of the 16 E2M1 codebook entries; returns int64 indices 0..15."""
    diffs = (x.unsqueeze(-1) - _KVALUES.to(x.device)).abs()
    return diffs.argmin(dim=-1)


def quantize_nvfp4(data: torch.Tensor, key: str) -> dict[str, torch.Tensor]:
    """Quantize one 2D+ tensor to NVFP4 (uint8-packed, 16-elem block scale + global scale).

    Raises ValueError if the last dimension is not a multiple of 16.
    """
    if data.shape[-1] % GROUP_SIZE != 0:
        raise ValueError(
            f"NVFP4 requires last dim to be a multiple of {GROUP_SIZE}, got {data.shape[-1]}"
        )
    x = data.to(torch.float32)
    *lead, last = x.shape
    n_blocks_per_row = last // GROUP_SIZE
    blocks = x.reshape(*lead, n_blocks_per_row, GROUP_SIZE)

    global_amax = x.abs().max()
    scale_2 = (global_amax / (_FP8_MAX * 6.0)) if global_amax > 0 else torch.tensor(1.0)
    scale_2 = scale_2.to(torch.float32)

    block_amax = blocks.abs().amax(dim=-1, keepdim=True)
    block_scale = (block_amax / 6.0 / scale_2).clamp(min=1e-12)
    block_scale_fp8 = block_scale.to(torch.float8_e4m3fn)

    normalized = blocks / (block_scale_fp8.to(torch.float32) * scale_2)
    idx = _nearest_e2m1_index(normalized)  # (*lead, n_blocks, 16) -> index per elem

    idx = idx.reshape(*lead, n_blocks_per_row, GROUP_SIZE // 2, 2)
    packed = (idx[..., 0] | (idx[..., 1] << 4)).to(torch.uint8)
    packed = packed.reshape(*lead, last // 2)

    return {
        key: packed,
        f"{key}.weight_scale": block_scale_fp8.squeeze(-1),
        f"{key}.weight_scale_2": scale_2.reshape(1),
    }


def dequantize_nvfp4(tensors: dict[str, torch.Tensor], key: str) -> torch.Tensor:
    """Reverse quantize_nvfp4 — used by tests to verify round-trip accuracy."""
    packed = tensors[key]
    block_scale = tensors[f"{key}.weight_scale"].to(torch.float32)
    scale_2 = tensors[f"{key}.weight_scale_2"]

    lo = (packed & 0x0F).to(torch.int64)
    hi = ((packed >> 4) & 0x0F).to(torch.int64)
    *lead, half = packed.shape
    idx = torch.stack([lo, hi], dim=-1).reshape(*lead, half * 2)
    idx = idx.reshape(*lead, half * 2 // GROUP_SIZE, GROUP_SIZE)

    values = _KVALUES[idx]
    values = values * block_scale.unsqueeze(-1) * scale_2
    return values.reshape(*lead, half * 2)
