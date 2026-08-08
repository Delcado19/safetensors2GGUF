"""Quantized Safetensors → Safetensors output backend.

Companion to convert.py's GGUF writer: reuses the same architecture detection
and high-precision-tensor rule (keys_hiprec / 1D / <=1024 elems / >4D) but
writes a plain .safetensors file instead of a GGUF container. See
docs/superpowers/plans/2026-08-05-safetensors-output-and-text-encoder-support.md
for the ComfyUI-compatibility research this format registry is based on
(comfy/quant_ops.py QUANT_ALGOS in city96/ComfyUI-GGUF).
"""

from __future__ import annotations

import torch

from models.architectures import QUANTIZATION_THRESHOLD

# Ordered choices for the GUI dropdown: (display label, key)
SAFETENSORS_DTYPE_CHOICES: list[tuple[str, str]] = [
    ("F16       — Half precision",                                    "F16"),
    ("F16 mixed — Half precision, hiprec tensors stay F32",            "F16_MIXED"),
    ("FP8       — float8_e4m3fn, scaled (ComfyUI scaled-fp8 format)",  "FP8"),
    ("FP8 mixed — FP8 scaled, hiprec tensors stay F32",                "FP8_MIXED"),
    ("NVFP4     — Nvidia 4-bit blockscaled (16-elem blocks)",          "NVFP4"),
    ("NVFP4 mixed — NVFP4, hiprec tensors stay F32",                   "NVFP4_MIXED"),
]

def layer_key(key: str) -> str:
    """Return the module-prefix ComfyUI's quantized-op loader expects scale/
    comfy_quant sidecar tensors under.

    ComfyUI's per-layer loader (comfy/ops.py _load_quantized_weight_body)
    looks up scale tensors as ``{module_prefix}weight_scale`` — a SIBLING of
    ``{module_prefix}weight``, not nested under it. For a tensor named
    "foo.weight" that means the scale tensor must be named "foo.weight_scale",
    not "foo.weight.weight_scale". Confirmed against a real ComfyUI 0.29.2
    "unet unexpected" key dump that showed our old ``<key>.weight_scale``
    naming left every scale/comfy_quant tensor unconsumed — FP8 loaded raw
    unscaled bytes (visible as noise), NVFP4 crashed outright.
    """
    return key[: -len(".weight")] if key.endswith(".weight") else key


_MIXED_KEYS = {"F16_MIXED", "FP8_MIXED", "NVFP4_MIXED"}
_BASE_KEY = {
    "F16": "F16", "F16_MIXED": "F16",
    "FP8": "FP8", "FP8_MIXED": "FP8",
    "NVFP4": "NVFP4", "NVFP4_MIXED": "NVFP4",
}


def is_hiprec_st(key: str, data: torch.Tensor, model_arch, old_dtype: torch.dtype) -> bool:
    """Return True if ``key`` must stay high-precision (F32), mirroring
    convert._quant_type_for's rule so 'mixed' safetensors output matches the
    existing GGUF mixed-precision behaviour exactly."""
    if old_dtype not in (torch.float32, torch.bfloat16):
        return False
    n_dims = data.dim()
    if n_dims == 1:
        return True
    if data.numel() <= QUANTIZATION_THRESHOLD:
        return True
    if any(x in key for x in model_arch.keys_hiprec):
        return True
    return False


def quantize_tensor_st(
    data: torch.Tensor, key: str, model_arch, target_key: str
) -> dict[str, torch.Tensor]:
    """Quantize one tensor for safetensors output.

    Returns a dict of {tensor_name: tensor} — one entry for F16/FP8-unscaled,
    multiple entries (weight + scale tensors) for FP8-scaled and NVFP4.
    """
    old_dtype = data.dtype
    base = _BASE_KEY[target_key]
    mixed = target_key in _MIXED_KEYS

    if mixed and is_hiprec_st(key, data, model_arch, old_dtype):
        return {key: data.to(torch.float32)}

    if base == "F16":
        return {key: data.to(torch.float16)}

    # FP8/NVFP4 scale-tensor conventions only make sense for weight matrices:
    # ComfyUI's scaled-quant loader applies a layer's .weight_scale only to its
    # .weight tensor, so a quantized 1D tensor (bias, norm weight) would load
    # back unscaled and be wrong by up to ~448x. Unlike the `mixed and
    # is_hiprec_st(...)` check above, this must apply unconditionally — not
    # just in *_MIXED mode — because there is no accuracy or size benefit to
    # scale-quantizing tiny 1D tensors either way. Mirrors convert.py's
    # _quant_type_for, which always keeps 1D tensors at F32 regardless of
    # target_quant (review finding #2).
    if data.dim() == 1:
        return {key: data.to(torch.float32)}

    if base == "FP8":
        from safetensors_quant_fp8 import quantize_fp8_scaled
        return quantize_fp8_scaled(data, key)

    if base == "NVFP4":
        from safetensors_quant_nvfp4 import quantize_nvfp4
        try:
            return quantize_nvfp4(data, key)
        except ValueError:
            # Tensor's last dim isn't a multiple of 16 (e.g. 3x3 conv kernels
            # with last dim 3, or some DiT patch-embed layers) — NVFP4's
            # 16-element block packing can't apply. Fall back to a plain F16
            # write for this one tensor instead of crashing the whole
            # conversion after many tensors have already been processed
            # (review finding #1). No on_log hook exists at this call depth,
            # so this fallback is silent by design.
            return {key: data.to(torch.float16)}

    raise ValueError(f"Unknown target_key: {target_key!r}")
