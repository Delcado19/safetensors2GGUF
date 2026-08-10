"""Detect and reverse already-quantized weight tensors in a source checkpoint.

Mirrors Starnodes ComfyUI Model Converter's "Automatic Dequantization" feature:
instead of refusing to convert an already-quantized checkpoint outright (the
prior behavior — see docs/issues_analysis.md #12), reconstruct an approximate
float32 original for each recognized quantized layer and feed that back into
this tool's normal quantization pipeline. This is lossy (quantization is not
reversible) but strictly better than either refusing, or the corruption that
motivated the original guard (treating scale/comfy_quant sidecar tensors as
ordinary weights and quantizing them a second time).

Supports the two on-disk quantized formats this tool itself writes/reads:
int8_tensorwise (plain and ConvRot-rotated) and scaled float8_e4m3fn/e5m2, plus
NVFP4 via the existing dequantize_nvfp4(). Detection prefers the authoritative
per-layer ".comfy_quant" JSON sidecar (the convention real ComfyUI-native
quantized releases use, per comfy/utils.py's convert_old_quants()) and falls
back to a dtype + sibling-scale-tensor heuristic when that sidecar is absent
or unparseable.
"""

from __future__ import annotations

import json

import torch

from safetensors_quant import layer_key
from safetensors_quant_int8 import CONVROT_GROUP_SIZE, _build_hadamard
from safetensors_quant_nvfp4 import dequantize_nvfp4

_FLOAT8_DTYPES: tuple = tuple(
    d for d in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2", None),
    )
    if d is not None
)
_INT_QUANT_DTYPES: tuple = tuple(
    d for d in (getattr(torch, "int8", None), getattr(torch, "uint8", None))
    if d is not None
)


def _read_file_level_quant_metadata(state_dict, prefix: str) -> dict | None:
    """Read this layer's config from the source file's ``_quantization_metadata``
    header field — the convention this tool's *own* convert_to_safetensors()
    writes (file-level JSON with a "layers" dict), as opposed to real
    ComfyUI-native releases which write a per-layer ".comfy_quant" tensor
    sidecar instead (see _read_comfy_quant_sidecar). Without this, re-feeding
    this tool's own ConvRot output back in would only work by accident,
    because the per-row-scale heuristic below silently assumes
    CONVROT_GROUP_SIZE — reading the actual group size back out here removes
    that assumption for any file this tool wrote."""
    file_metadata = getattr(state_dict, "file_metadata", None)
    if file_metadata is None:
        return None
    raw = file_metadata().get("_quantization_metadata")
    if not raw:
        return None
    try:
        return json.loads(raw).get("layers", {}).get(prefix)
    except (json.JSONDecodeError, AttributeError):
        return None


def _read_comfy_quant_sidecar(state_dict, prefix: str) -> dict | None:
    """Decode a "{prefix}.comfy_quant" JSON sidecar tensor, if present and
    parseable, falling back to this tool's own file-level
    ``_quantization_metadata`` header field. Returns None (not an error) on
    any decode failure — callers fall back to the dtype/sibling-scale
    heuristic instead."""
    sidecar_key = f"{prefix}.comfy_quant"
    if sidecar_key in state_dict:
        try:
            raw = state_dict[sidecar_key]
            return json.loads(bytes(raw.tolist()).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            pass
    return _read_file_level_quant_metadata(state_dict, prefix)


def detect_quantized_weight(state_dict, key: str) -> str | None:
    """Return the detected on-disk quant format for a ".weight" tensor, or
    None if it isn't recognizably quantized. One of "int8_tensorwise",
    "float8_e4m3fn", "nvfp4"."""
    if not key.endswith(".weight"):
        return None
    prefix = layer_key(key)

    conf = _read_comfy_quant_sidecar(state_dict, prefix)
    if conf is not None:
        fmt = conf.get("format")
        if fmt in ("int8_tensorwise", "float8_e4m3fn", "nvfp4"):
            return fmt

    scale_key = f"{prefix}.weight_scale"
    if scale_key not in state_dict:
        return None

    dtype_of = getattr(state_dict, "dtype_of", None)
    dtype = dtype_of(key) if dtype_of is not None else state_dict[key].dtype

    if dtype in _INT_QUANT_DTYPES:
        if f"{prefix}.weight_scale_2" in state_dict:
            return "nvfp4"
        return "int8_tensorwise"
    if dtype in _FLOAT8_DTYPES:
        return "float8_e4m3fn"
    return None


def _convrot_group_size(state_dict, prefix: str) -> int:
    conf = _read_comfy_quant_sidecar(state_dict, prefix)
    if conf and conf.get("convrot"):
        size = conf.get("convrot_groupsize")
        if size:
            return int(size)
    return CONVROT_GROUP_SIZE


def dequantize_weight(state_dict, key: str, fmt: str, q: torch.Tensor) -> torch.Tensor:
    """Reconstruct an approximate float32 original for one already-quantized
    ".weight" tensor. ``q`` is the raw on-disk tensor for ``key`` (passed in
    rather than re-fetched via state_dict[key], so callers that already
    loaded it via streaming iteration don't materialize it twice)."""
    prefix = layer_key(key)

    if fmt == "nvfp4":
        return dequantize_nvfp4(
            {
                key: q,
                f"{prefix}.weight_scale": state_dict[f"{prefix}.weight_scale"],
                f"{prefix}.weight_scale_2": state_dict[f"{prefix}.weight_scale_2"],
            },
            key,
        )

    scale = state_dict[f"{prefix}.weight_scale"].to(torch.float32)
    value = q.to(torch.float32) * scale

    if fmt == "float8_e4m3fn":
        return value

    if fmt == "int8_tensorwise":
        if scale.numel() > 1:
            # ConvRot: per-row scale means the value is still in the rotated
            # basis — un-rotate it. H is orthogonal & normalized
            # (H @ H.T == I); the on-disk rotation was W_rot = W @ H.T, so
            # W = W_rot @ H (see safetensors_quant_int8._rotate_weight).
            group_size = _convrot_group_size(state_dict, prefix)
            out_f, in_f = value.shape
            if in_f % group_size != 0:
                raise ValueError(
                    f"'{key}': per-row weight_scale (ConvRot) but in_features "
                    f"{in_f} isn't divisible by group_size {group_size} — "
                    "inconsistent/unrecognized sidecar, cannot un-rotate."
                )
            h = _build_hadamard(group_size, device=value.device, dtype=value.dtype)
            grouped = value.reshape(out_f, in_f // group_size, group_size)
            value = torch.matmul(grouped, h).reshape(out_f, in_f)
        return value

    raise ValueError(f"Unsupported quantized format for dequantization: {fmt!r}")
