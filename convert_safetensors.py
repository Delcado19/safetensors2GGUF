"""Safetensors → Safetensors quantized output.

Sibling to convert.py's GGUF writer: same architecture detection and tensor
loading, different output backend. Unlike GGUF output, no 5D side-car export,
no shape_fix rearrange, no 1D-pad-token unsqueeze — those exist only to
satisfy GGUF/llama-quantize constraints that plain safetensors doesn't have.
"""

from __future__ import annotations

import json
import os

import torch
from safetensors.torch import save_file

from convert import load_state_dict
from models.architectures import detect_arch
from safetensors_quant import layer_key, quantize_tensor_st

# Float8 dtypes — resolved once at import time; empty tuple on older PyTorch builds.
_FLOAT8_DTYPES: tuple = tuple(
    d for d in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2",   None),
    )
    if d is not None
)

_TARGET_TO_QUANT_FORMAT = {
    "FP8": "float8_e4m3fn", "FP8_MIXED": "float8_e4m3fn",
    "NVFP4": "nvfp4", "NVFP4_MIXED": "nvfp4",
}

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


def _check_not_already_quantized(state_dict):
    """Raise if state_dict is itself an already-quantized checkpoint.

    This tool assumes its source is a plain full-precision checkpoint. Some
    published checkpoints (e.g. ComfyUI-native int8_tensorwise+ConvRot
    releases) are already partially quantized, with their own per-layer
    ``.weight_scale``/``.comfy_quant`` sidecar tensors sitting right in the
    state dict. Nothing in this tool recognizes those as sidecars rather
    than ordinary weights — it would quantize the already-quantized integer
    codes a second time, AND quantize the pre-existing scale/comfy_quant
    tensors themselves as if they were regular weight matrices (producing
    corrupted, wrongly-named output; see docs/issues_analysis.md #12).
    Detecting and refusing beats silently producing a worse checkpoint.
    """
    quant_sidecar_keys = [k for k in state_dict.keys() if k.endswith(".comfy_quant")]
    if quant_sidecar_keys:
        n = len(quant_sidecar_keys)
        example = quant_sidecar_keys[0][: -len(".comfy_quant")]
        raise ValueError(
            f"Source checkpoint is already quantized ({n} layer(s) have a "
            f".comfy_quant sidecar, e.g. '{example}') — re-quantizing it "
            "would corrupt those layers a second time. Convert from the "
            "original unquantized checkpoint instead."
        )

    dtype_of = getattr(state_dict, "dtype_of", None)
    for key in state_dict.keys():
        if not key.endswith(".weight"):
            continue
        dtype = dtype_of(key) if dtype_of is not None else state_dict[key].dtype
        if dtype in _ALREADY_QUANTIZED_DTYPES:
            raise ValueError(
                f"Source checkpoint is already quantized ('{key}' is stored "
                f"as {dtype}, not a float type) — re-quantizing it would "
                "corrupt that layer. Convert from the original unquantized "
                "checkpoint instead."
            )


def convert_to_safetensors(
    path,
    dst_path=None,
    target_key="FP8",
    overwrite=False,
    on_progress=None,
    on_log=None,
    cancel_event=None,
    model_arch=None,
):
    """Convert a model checkpoint to a quantized .safetensors file.

    Args:
        path: Source model file path.
        dst_path: Output path; auto-generated as ``<src>-<target_key>.safetensors`` when None.
        target_key: One of safetensors_quant.SAFETENSORS_DTYPE_CHOICES' keys.
        overwrite: Skip existence check when True.
        on_progress: Optional callback(idx, total, key).
        on_log: Optional callback(msg); prints when None.
        cancel_event: Optional threading.Event; raises RuntimeError("cancelled") when set.
        model_arch: Pre-resolved architecture instance (e.g. a bare
            ``models.architectures.ModelTemplate()`` for text-encoder checkpoints,
            which aren't in ``arch_list`` and would otherwise fail ``detect_arch``).
            When None (default), auto-detected via ``detect_arch`` as before.

    Returns:
        (dst_path, model_arch)
    """
    def _log(msg):
        if on_log:
            on_log(msg)
        else:
            print(msg)

    state_dict = load_state_dict(path)
    _check_not_already_quantized(state_dict)
    if model_arch is None:
        model_arch = detect_arch(state_dict)
    _log(f"INFO:  Architecture: {model_arch.arch}")

    if dst_path is None:
        dst_path = f"{os.path.splitext(path)[0]}-{target_key}.safetensors"

    if os.path.isfile(dst_path) and not overwrite:
        raise OSError(f"Output exists and overwrite is disabled: {dst_path}")

    out_tensors: dict[str, torch.Tensor] = {}
    layer_formats: dict[str, dict] = {}
    quant_format = _TARGET_TO_QUANT_FORMAT.get(target_key)

    # Iterate state_dict.items() directly (not list(...)) — load_state_dict
    # returns a lazy _LazyStateDict for .safetensors sources that streams one
    # tensor at a time; materializing it into a list here would defeat that
    # and reintroduce the >RAM OOM crash fixed in commit 72a49dc (review
    # finding #3). len() is cheap — _LazyStateDict.__len__ reads the key map,
    # not tensor data.
    total = len(state_dict)
    for idx, (key, data) in enumerate(state_dict.items()):
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("cancelled")
        if on_progress:
            on_progress(idx + 1, total, key)
        if any(x in key for x in model_arch.keys_ignore):
            continue

        # Coerce dtypes that nan_to_num cannot handle:
        #   float8   → float16  (nan_to_num rejects float8)
        if _FLOAT8_DTYPES and data.dtype in _FLOAT8_DTYPES:
            data = data.to(torch.float16)

        data = torch.nan_to_num(data, nan=0.0, posinf=65504.0, neginf=-65504.0)
        quantized = quantize_tensor_st(data, key, model_arch, target_key)
        out_tensors.update(quantized)
        if quant_format and len(quantized) > 1:
            # ComfyUI's convert_old_quants() writes the comfy_quant sidecar as
            # "{layer}.comfy_quant" where `layer` is the module prefix (no
            # trailing "weight") — must match the scale-tensor naming in
            # safetensors_quant.layer_key(), or the sidecar never lines up
            # with the tensor ComfyUI's loader is inspecting.
            layer_formats[layer_key(key)] = {"format": quant_format}

    metadata = {"comfy.gguf_source_arch": model_arch.arch}
    if layer_formats:
        metadata["_quantization_metadata"] = json.dumps(
            {"format_version": "1.0", "layers": layer_formats}
        )

    _log(f"INFO:  Writing {len(out_tensors)} tensors → {dst_path}")
    save_file(out_tensors, dst_path, metadata=metadata)
    _log(f"INFO:  Done → {dst_path}")
    return dst_path, model_arch
