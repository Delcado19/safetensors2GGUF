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


def _find_scale_key(state_dict, prefix: str) -> str | None:
    """Return whichever per-layer scale sidecar key is present, checking
    both this tool's own ".weight_scale" convention and Comfy-Org's
    "fp8_scaled" repackaging convention ".scale_weight" (reversed word
    order). Found 2026-08-18 converting HiDream's
    llama_3.1_8b_instruct_fp8_scaled.safetensors: with only ".weight_scale"
    recognized, the checkpoint's actual ".weight" tensors (raw on-disk
    float8_e4m3fn bytes) were never detected as quantized at all, so they
    fed straight into the normal pipeline unscaled — numerically wrong
    output for every affected tensor, not just a missed optimization. The
    orphaned 0-dim ".scale_weight" sidecar tensors were then themselves
    treated as ordinary weights, crashing NVFP4's quantize_nvfp4() on
    `.shape[-1]` for a tensor with zero dimensions."""
    for suffix in (".weight_scale", ".scale_weight"):
        candidate = f"{prefix}{suffix}"
        if candidate in state_dict:
            return candidate
    return None


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

    scale_key = _find_scale_key(state_dict, prefix)
    if scale_key is None:
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


# int8/uint8 only — NOT float8: an isolated float8 weight (no .comfy_quant
# sidecar) is an existing, deliberately-supported case (coerced to float16,
# see test_float8_input_coerced_to_float16), not proof of a fully quantized
# checkpoint. int8/uint8 weights have no such legitimate unquantized use.
_ALREADY_QUANTIZED_DTYPES: tuple = tuple(
    d for d in (
        getattr(torch, "int8", None),
        getattr(torch, "uint8", None),
    )
    if d is not None
)

# Tensor-key suffixes for opaque, non-weight blobs that must survive a
# conversion completely byte-identical (dtype included) -- never routed
# through nan_to_num/quantize_tensor_st, both of which assume floating-point
# weight data. Currently just Comfy-Org's "spiece_model" (a raw SentencePiece
# tokenizer.model embedded as a 1D uint8 tensor, see _scan_quantized_layers()'s
# docstring) -- found 2026-08-19 after an F16 "conversion" silently upcast it
# to float16, which would have made SPieceTokenizer's protobuf parse fail on
# load in ComfyUI (garbage bytes, not a missing-key crash like the earlier,
# already-fixed unconditional-strip mistake).
_PASSTHROUGH_TENSOR_SUFFIXES = ("spiece_model",)


def _scan_quantized_layers(state_dict) -> tuple[dict[str, str], set[str]]:
    """Pre-scan for already-quantized ".weight" tensors so they can be
    automatically dequantized and cleanly re-quantized to a different target
    format, instead of refusing outright (mirrors Starnodes ComfyUI Model
    Converter's "Automatic Dequantization" feature; see dequantize.py and
    docs/issues_analysis.md #12 for why a hard refusal was the original
    behavior here).

    Shared by convert_safetensors.py (safetensors output) and convert.py
    (GGUF output) -- a pre-quantized source checkpoint (e.g. a community
    fp8_mixed Wan 2.2 checkpoint whose ".weight_scale" sidecars are 0-dim
    F32 tensors) needs the exact same dequantize-before-requantize handling
    regardless of output format. Originally lived only in
    convert_safetensors.py; convert.py's GGUF path had no equivalent, so it
    wrote the raw scale sidecars straight through as 0-dim GGUF tensors,
    which crashes llama-quantize with `GGML_ASSERT(n_dims >= 1 && n_dims <=
    GGML_MAX_DIMS)` -- found 2026-08-20 quantizing the remaining Wan 2.2
    diffusion-model formats.

    Returns (formats, skip): ``formats`` maps each detected quantized
    ".weight" key to its on-disk format string; ``skip`` is the set of
    scale/comfy_quant sidecar keys that must not be treated as ordinary
    weights during the main conversion loop below.

    Still raises if an int8/uint8 ".weight" is found with no recognizable
    scale sidecar — that case is genuinely unrecoverable (no scale to
    reconstruct the original magnitude from), not just unhandled.
    """
    # .scale_weight (reversed word order) is Comfy-Org's own "fp8_scaled"
    # repackaging convention, distinct from this tool's ".weight_scale" --
    # see dequantize.py's _find_scale_key() docstring for the bug this
    # closes (found 2026-08-18 converting a HiDream text encoder shipped in
    # that format).
    #
    # "{prefix}scaled_fp8" (bare, or prefixed like "diffusion_model.
    # scaled_fp8") is a separate Comfy-Org convention: a sentinel tensor with
    # 0 elements (dtype float8_e4m3fn) that comfy/utils.py's
    # convert_old_quants() checks for *presence*, not content, to detect its
    # own legacy "scaled_fp8" repackaging format. It carries no actual
    # weight data and is meaningless once dequantized/re-quantized by this
    # tool (our own output is never in that legacy format -- it either
    # carries its own `_quantization_metadata`, which makes ComfyUI skip
    # convert_old_quants()'s legacy branch entirely, or is plain F16 with no
    # scale sidecars left for that branch to act on either way). Left
    # unstripped, it silently rode through every quantized output
    # (confirmed harmless there only because ComfyUI's loader tolerates the
    # unmapped key) but crashes llama.cpp's convert_hf_to_gguf.py outright
    # with `ValueError: Can not map tensor 'scaled_fp8'` -- found 2026-08-18
    # attempting a GGUF conversion of a re-quantized HiDream text encoder
    # that started life as a Comfy-Org fp8_scaled checkpoint.
    #
    # NOTE: "{prefix}spiece_model" (Comfy-Org's self-contained SentencePiece-
    # tokenizer convention, comfy/text_encoders/wan.py's UMT5XXlTokenizer)
    # is deliberately NOT in this skip set, unlike scaled_fp8 above --
    # despite looking like the same class of sentinel tensor, it is load-
    # bearing: comfy/sd.py's load_text_encoder_state_dicts() actively reads
    # it out of the checkpoint into tokenizer_data["spiece_model"] to build
    # the CLIP object's tokenizer, for UMT5/T5/ACE/gemma/jina-family
    # encoders. It must reach the output file, byte-identical -- handled by
    # _PASSTHROUGH_TENSOR_SUFFIXES in the main conversion loop below instead
    # of this skip set, since "skip" here means "omit from the output
    # entirely" (right for scale/comfy_quant sidecars, wrong for a tensor
    # that has to survive). It only needs to be excluded from the *GGUF*
    # conversion path specifically (llama.cpp's converter reads the
    # tokenizer from an external vendored .model file instead, and can't
    # map a "spiece_model" byte-blob tensor to a weight) -- see
    # text_encoder_convert.py's convert_text_encoder() for that filtering,
    # done as a separate copy step so it never touches this function's
    # (user-facing) safetensors output.
    skip = {
        key for key in state_dict.keys()
        if key.endswith((
            ".weight_scale", ".weight_scale_2", ".comfy_quant", ".scale_weight",
            "scaled_fp8",
            # Per-layer *activation* quantization scale from a dynamic FP8
            # scheme (Comfy-Org's Qwen-Image-Edit-2511 fp8_scaled repackage
            # carries one alongside every .weight_scale) -- meaningless once
            # the weight is dequantized back to float, since it scaled a
            # runtime activation tensor, not this checkpoint's weights.
            # Missed by the original bug #18 fix (only .weight_scale-family
            # sidecars were skipped): found 2026-08-20 when a real batch
            # conversion's GGUF Q4_K_M step crashed llama-quantize.exe
            # (STATUS_STACK_BUFFER_OVERRUN, 0xC0000409) on the ~1600 orphaned
            # 0-dim .input_scale tensors carried straight through as GGUF
            # tensors -- same crash class the docstring above already
            # documents for un-skipped scale sidecars.
            ".input_scale",
        ))
    }
    formats: dict[str, str] = {}
    dtype_of = getattr(state_dict, "dtype_of", None)
    for key in state_dict.keys():
        if key in skip or not key.endswith(".weight"):
            continue
        fmt = detect_quantized_weight(state_dict, key)
        if fmt is not None:
            formats[key] = fmt
            continue
        dtype = dtype_of(key) if dtype_of is not None else state_dict[key].dtype
        if dtype in _ALREADY_QUANTIZED_DTYPES:
            raise ValueError(
                f"'{key}' is stored as {dtype} but has no recognizable "
                "weight_scale/comfy_quant sidecar to reconstruct it from — "
                "cannot safely dequantize. Convert from the original "
                "unquantized checkpoint instead."
            )
    return formats, skip


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

    scale = state_dict[_find_scale_key(state_dict, prefix)].to(torch.float32)
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
