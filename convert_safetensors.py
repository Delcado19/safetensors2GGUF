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


def convert_to_safetensors(
    path,
    dst_path=None,
    target_key="FP8",
    overwrite=False,
    on_progress=None,
    on_log=None,
    cancel_event=None,
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

    Returns:
        (dst_path, model_arch)
    """
    def _log(msg):
        if on_log:
            on_log(msg)
        else:
            print(msg)

    state_dict = load_state_dict(path)
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
