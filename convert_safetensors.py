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
from safetensors_quant import quantize_tensor_st

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

    items = list(state_dict.items())
    total = len(items)
    for idx, (key, data) in enumerate(items):
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("cancelled")
        if on_progress:
            on_progress(idx + 1, total, key)
        if any(x in key for x in model_arch.keys_ignore):
            continue

        data = torch.nan_to_num(data, nan=0.0, posinf=65504.0, neginf=-65504.0)
        quantized = quantize_tensor_st(data, key, model_arch, target_key)
        out_tensors.update(quantized)
        if quant_format and len(quantized) > 1:
            layer_formats[key] = {"format": quant_format}

    metadata = {"comfy.gguf_source_arch": model_arch.arch}
    if layer_formats:
        metadata["_quantization_metadata"] = json.dumps(
            {"format_version": "1.0", "layers": layer_formats}
        )

    _log(f"INFO:  Writing {len(out_tensors)} tensors → {dst_path}")
    save_file(out_tensors, dst_path, metadata=metadata)
    _log(f"INFO:  Done → {dst_path}")
    return dst_path, model_arch
