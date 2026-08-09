"""Tensor-wise INT8 quantization matching ComfyUI's int8_tensorwise checkpoint
convention, with an optional offline Hadamard rotation (ConvRot).

Format reference: Comfy-Org/ComfyUI comfy/quant_ops.py QUANT_ALGOS["int8_tensorwise"]
(comfy_kitchen.tensor.int8.TensorWiseINT8Layout) and comfy/ops.py
_load_quantized_weight_body. Unlike FP8/NVFP4 (QUANT_ALGOS["quantize_input"] left
at its True default), int8_tensorwise sets "quantize_input": False — ComfyUI never
dynamically quantizes *activations* for this format, only weights, so inference
always runs through the same shape-handling code path as an unquantized model.
That sidesteps a confirmed ComfyUI bug (Comfy-Org/ComfyUI#14595) where the
generic dynamic-activation-quantization path silently mishandles tensors on
newer/custom DiT architectures depending on whether they get reshaped to 2D or
3D before a Linear call — the likely root cause of the FP8/NVFP4 pose/identity
corruption this format was added to avoid (see docs/issues_analysis.md #15).

Plain (no rotation): single absmax scalar scale per whole tensor.
ConvRot: weight is rotated with a block-diagonal Hadamard matrix (each
`group_size`-wide slice of the input dimension gets its own orthogonal
rotation) before quantizing row-wise — this smears outlier magnitudes across
a group before rounding, the standard technique for keeping 8-bit (and lower)
quantization error low on activation-outlier-heavy transformer weights
(QuaRot/SpinQuant lineage). ComfyUI's kernel un-rotates on load, so the model
sees the original basis again; only the on-disk weight is rotated.
"""

from __future__ import annotations

import math

import torch

from safetensors_quant import layer_key

CONVROT_GROUP_SIZE = 256  # ComfyUI/comfy_kitchen default


def _build_hadamard(size: int, device, dtype: torch.dtype) -> torch.Tensor:
    """Normalized regular (power-of-4) Hadamard matrix — bit-for-bit the same
    construction as comfy_kitchen.tensor.int8_utils._build_hadamard, required
    since the on-disk rotation must be exactly invertible by ComfyUI's own
    un-rotation on load."""
    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(f"Regular Hadamard size must be a power of 4, got {size}")
    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=dtype, device=device,
    )
    h = h4
    current_size = 4
    while current_size < size:
        h = torch.kron(h, h4)
        current_size *= 4
    return h / (size ** 0.5)


def _rotate_weight(weight: torch.Tensor, h: torch.Tensor, group_size: int) -> torch.Tensor:
    """W_rot = W @ H_block^T, applied independently to each group_size-wide
    slice of the input (last) dimension."""
    out_f, in_f = weight.shape
    n_groups = in_f // group_size
    grouped = weight.reshape(out_f, n_groups, group_size)
    rotated = torch.matmul(grouped, h.T.to(dtype=weight.dtype, device=weight.device))
    return rotated.reshape(out_f, in_f)


def quantize_int8_tensorwise(data: torch.Tensor, key: str) -> dict[str, torch.Tensor]:
    """Plain tensor-wise INT8: single scalar scale for the whole tensor.

    Returns {key: int8_tensor, f"{layer_key(key)}.weight_scale": float32 scalar}.
    """
    amax = data.abs().max()
    scale = (amax.float() / 127.0).clamp(min=1e-30)
    q = (data.to(torch.float32) / scale).round().clamp(-128, 127).to(torch.int8)
    return {
        key: q,
        f"{layer_key(key)}.weight_scale": scale.reshape(1),
    }


def quantize_int8_convrot(
    data: torch.Tensor, key: str, group_size: int = CONVROT_GROUP_SIZE
) -> dict[str, torch.Tensor]:
    """ConvRot INT8: Hadamard-rotate the weight, then quantize row-wise
    (one scale per output row — required by comfy_kitchen's ConvRot kernel).

    Raises ValueError if the input (last) dimension isn't a multiple of
    group_size — caller must catch and fall back to plain tensorwise/F16.
    """
    out_f, in_f = data.shape
    if in_f % group_size != 0:
        raise ValueError(f"in_features {in_f} not divisible by convrot group_size {group_size}")

    h = _build_hadamard(group_size, device=data.device, dtype=data.dtype)
    rotated = _rotate_weight(data, h, group_size)

    abs_max = rotated.abs().amax(dim=-1, keepdim=True)
    scale = (abs_max.float() / 127.0).clamp(min=1e-30)
    q = (rotated.to(torch.float32) / scale).round().clamp(-128, 127).to(torch.int8)
    return {
        key: q,
        # [out_f, 1] — must stay 2D so ComfyUI's `q.float() * scale` broadcasts
        # per output row, not (wrongly) per the trailing in_features axis.
        f"{layer_key(key)}.weight_scale": scale,
    }
