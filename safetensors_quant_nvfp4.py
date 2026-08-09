"""Nvidia NVFP4 block-scaled quantization for safetensors output.

Format reference: city96/ComfyUI-GGUF comfy/quant_ops.py QUANT_ALGOS["nvfp4"]
(group_size=16, storage_t=uint8, params={weight_scale, weight_scale_2}) — the
same TensorRT-Model-Optimizer convention ComfyUI's native NVFP4 loader expects.
The E2M1 decode table below is the standard (undoubled) OCP magnitudes — NOT
gguf.quants.NVFP4.kvalues, whose values are doubled to compensate for that
module's own custom "ue4m3" block-scale encoding (a *0.5 factor baked into
its ue4m3_to_fp32 decode, see gguf/quants.py). ComfyUI's comfy_kitchen reads
block scales as plain torch.float8_e4m3fn (no such compensation), so pairing
gguf's doubled table with a plain-e4m3fn scale silently decoded every weight
at half its intended magnitude — confirmed by comparing against
comfy_kitchen.tensor.nvfp4.TensorCoreNVFP4Layout's own quantize/dequantize on
a live ComfyUI install (see docs/issues_analysis.md #11). Our own round-trip
tests couldn't catch this: quantizing and dequantizing with the same (wrong)
table cancels the error out, so it only surfaces against ComfyUI's decoder.
"""

from __future__ import annotations

import torch

from safetensors_quant import layer_key

GROUP_SIZE = 16
# Standard E2M1 magnitudes (OCP MX/NVFP4 spec): 0, 0.5, 1, 1.5, 2, 3, 4, 6 and
# their negatives — NOT doubled (see module docstring for why).
_KVALUES = torch.tensor(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32
)
_FP8_MAX = 448.0


def _nearest_e2m1_index(x: torch.Tensor) -> torch.Tensor:
    """Map float values (already divided by the block scale) to the nearest
    of the 16 E2M1 codebook entries; returns int64 indices 0..15."""
    diffs = (x.unsqueeze(-1) - _KVALUES.to(x.device)).abs()
    return diffs.argmin(dim=-1)


def _round_up(x: int, multiple: int) -> int:
    return ((x + multiple - 1) // multiple) * multiple


def _swizzle_index_map(m_padded: int, k_padded: int, device: torch.device) -> torch.Tensor:
    """Index map for comfy_kitchen/cuBLAS's 128x4 block-interleaved NVFP4
    scale-factor layout — mirrors vLLM's swizzle_blockscale()
    (vllm/model_executor/layers/quantization/utils/nvfp4_utils.py), the same
    layout TensorRT-Model-Optimizer NVFP4 checkpoints use.
    idx_map[i, j] is the flat index into a plain row-major (m_padded,
    k_padded) buffer of the value that belongs at swizzled position (i, j).
    Built once per shape and reused for both the forward swizzle and (in
    dequantize_nvfp4, for round-trip tests) its exact gather/scatter inverse.
    """
    idx = torch.arange(m_padded * k_padded, device=device).reshape(m_padded, k_padded)
    idx = idx.reshape(m_padded // 128, 4, 32, k_padded // 4, 4)
    idx = idx.permute(0, 3, 2, 1, 4).contiguous()
    return idx.reshape(m_padded, k_padded)


def _swizzle_block_scale(scale: torch.Tensor) -> torch.Tensor:
    """Pad a (rows, k_blocks) NVFP4 block-scale matrix to 128x4 alignment and
    rearrange it into the block-interleaved layout ComfyUI's comfy_kitchen
    CUDA kernel (scaled_mm_nvfp4) requires for its cuBLAS FP4 GEMM.

    The kernel asserts block_scale.size() == (roundup(N,128), roundup(K//16,4))
    — necessary but not sufficient: the values inside that shape are NOT
    plain row-major, they're 128x4-tile-interleaved. Zero-padding to the
    right shape without this permutation either crashes ("Invalid B scale
    shape" when N isn't already a multiple of 128 — reported for Lumina2/
    Z-Image's final_layer.linear.weight) or, when the shape happens to match,
    silently loads wrong scale values (garbage output, no crash).
    """
    # Fancy-indexing a float8_e4m3fn tensor isn't implemented on CPU
    # ("index_cpu not implemented for 'Float8_e4m3fn'") — gather/scatter via
    # a uint8 view instead, which is bit-identical for a pure permutation.
    m, k = scale.shape
    m_padded, k_padded = _round_up(m, 128), _round_up(k, 4)
    padded = torch.zeros(m_padded * k_padded, dtype=torch.uint8, device=scale.device)
    padded.view(m_padded, k_padded)[:m, :k] = scale.view(torch.uint8)
    idx_map = _swizzle_index_map(m_padded, k_padded, scale.device)
    swizzled = padded[idx_map.reshape(-1)].reshape(m_padded, k_padded)
    return swizzled.view(scale.dtype)


def _unswizzle_block_scale(swizzled: torch.Tensor, m: int, k: int) -> torch.Tensor:
    """Inverse of _swizzle_block_scale — used only by dequantize_nvfp4()'s
    test-only round-trip helper, via exact scatter-inverse of the same
    index map (not a hand-derived second permutation, to avoid an
    independently-wrong "inverse")."""
    m_padded, k_padded = swizzled.shape
    idx_map = _swizzle_index_map(m_padded, k_padded, swizzled.device)
    flat = torch.empty(m_padded * k_padded, dtype=torch.uint8, device=swizzled.device)
    flat[idx_map.reshape(-1)] = swizzled.reshape(-1).view(torch.uint8)
    return flat.reshape(m_padded, k_padded)[:m, :k].view(swizzled.dtype)


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

    # Guard against division by zero: if block_scale_fp8 underflows to 0.0 after fp8 cast,
    # explicitly set normalized to 0 instead of relying on NaN→argmin(0) coincidence.
    block_scale_fp32 = block_scale_fp8.to(torch.float32)
    normalized = torch.where(
        block_scale_fp32 == 0,
        torch.zeros_like(blocks),
        blocks / (block_scale_fp32 * scale_2)
    )
    idx = _nearest_e2m1_index(normalized)  # (*lead, n_blocks, 16) -> index per elem

    idx = idx.reshape(*lead, n_blocks_per_row, GROUP_SIZE // 2, 2)
    # hi_first=True convention (even-indexed element in the HIGH nibble) —
    # required because comfy_kitchen's real matmul kernel (scaled_mm_nvfp4,
    # used by ComfyUI's actual NVFP4 inference path) has no nibble-order
    # parameter at all; it's hardcoded to this convention. ck.dequantize_nvfp4
    # (a separate, standalone helper) does expose a hi_first argument, which
    # made the opposite (wrong) packing look correct in isolated round-trip
    # tests using that helper explicitly — but the real inference path never
    # goes through it with a non-default argument, so every weight was
    # nibble-swapped in actual ComfyUI generation despite passing that test.
    packed = (idx[..., 1] | (idx[..., 0] << 4)).to(torch.uint8)
    packed = packed.reshape(*lead, last // 2)

    weight_scale = block_scale_fp8.squeeze(-1)
    if len(lead) == 1:
        # The common case (nn.Linear weight, 2D (out_features, in_features)):
        # ComfyUI's cuBLAS NVFP4 kernel needs the 128x4-swizzled layout — see
        # _swizzle_block_scale. Higher-rank tensors (e.g. raw Conv2d kernels)
        # never reach that kernel path in ComfyUI (it reshapes them to 2D
        # itself before quantizing, a different on-disk convention this tool
        # doesn't replicate), so they're left unswizzled here.
        weight_scale = _swizzle_block_scale(weight_scale)

    prefix = layer_key(key)
    return {
        key: packed,
        f"{prefix}.weight_scale": weight_scale,
        f"{prefix}.weight_scale_2": scale_2.reshape(1),
    }


def dequantize_nvfp4(tensors: dict[str, torch.Tensor], key: str) -> torch.Tensor:
    """Reverse quantize_nvfp4 — used by tests to verify round-trip accuracy."""
    prefix = layer_key(key)
    packed = tensors[key]
    block_scale = tensors[f"{prefix}.weight_scale"]
    scale_2 = tensors[f"{prefix}.weight_scale_2"]

    lo = (packed & 0x0F).to(torch.int64)
    hi = ((packed >> 4) & 0x0F).to(torch.int64)
    *lead, half = packed.shape
    # hi_first=True: the even-indexed (first of each pair) element is in the
    # high nibble — must match quantize_nvfp4's packing above.
    idx = torch.stack([hi, lo], dim=-1).reshape(*lead, half * 2)
    idx = idx.reshape(*lead, half * 2 // GROUP_SIZE, GROUP_SIZE)

    if len(lead) == 1:
        # Undo _swizzle_block_scale's 128x4 interleave and drop the padding
        # rows/cols back down to the tensor's real (rows, n_blocks) shape.
        m, n_blocks = lead[0], idx.shape[-2]
        block_scale = _unswizzle_block_scale(block_scale, m, n_blocks)
    block_scale = block_scale.to(torch.float32)

    values = _KVALUES[idx]
    values = values * block_scale.unsqueeze(-1) * scale_2
    return values.reshape(*lead, half * 2)
