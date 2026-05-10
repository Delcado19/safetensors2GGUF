"""Load a GGUF file back into a state dict of dequantized PyTorch tensors.

Used for verifying the converter output: shapes, dtypes and values can be
compared against the original safetensors model.

Supported quantization types: all types handled by gguf.quants.dequantize
(F32, F16, BF16, Q4_0, Q4_K, Q5_K, Q6_K, Q8_0, Q8_K, …).
"""

import logging
import warnings

import gguf
import torch
import numpy as np

logger = logging.getLogger(__name__)


def _read_str_field(reader, field_name):
    """Read a string metadata field, tolerating size-1 arrays (Issue #384)."""
    field = reader.get_field(field_name)
    if field is None:
        return None
    if len(field.types) != 1 or field.types[0] != gguf.GGUFValueType.STRING:
        raise TypeError(f"Expected STRING for '{field_name}', got {field.types}")
    return str(field.parts[field.data[-1]], encoding="utf-8")


def _read_scalar_field(reader, field_name, py_type):
    """Read a numeric/bool metadata field, tolerating size-1 arrays (Issue #384)."""
    field = reader.get_field(field_name)
    if field is None:
        return None
    return py_type(field.parts[field.data[-1]].flat[0])


def _read_orig_shape(reader, tensor_name):
    """Return the original tensor shape stored during shape_fix reshape, or None."""
    field = reader.get_field(f"comfy.gguf.orig_shape.{tensor_name}")
    if field is None:
        return None
    if (len(field.types) != 2
            or field.types[0] != gguf.GGUFValueType.ARRAY
            or field.types[1] != gguf.GGUFValueType.INT32):
        raise TypeError(
            f"Unexpected orig_shape type for '{tensor_name}': {field.types}"
        )
    return torch.Size(int(field.parts[i].flat[0]) for i in field.data)


def _dequantize(tensor):
    """Dequantize a GGUF tensor to a float32 torch.Tensor.

    Delegates to gguf.quants.dequantize for all quantized types.
    F32/F16/BF16 are re-typed directly without going through the quant path.
    """
    qtype = tensor.tensor_type
    data = tensor.data          # numpy memmap, shape = raw storage shape

    # Suppress the non-writable mmap warning — harmless for read-only verification
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
        if qtype == gguf.GGMLQuantizationType.F32:
            t = torch.from_numpy(data.view(np.float32))
        elif qtype == gguf.GGMLQuantizationType.F16:
            t = torch.from_numpy(data.view(np.float16)).float()
        elif qtype == gguf.GGMLQuantizationType.BF16:
            t = torch.from_numpy(data.view(np.uint16))
            t = t.view(torch.bfloat16).float()
        else:
            # K-quants and other types: delegate to gguf library
            arr = gguf.quants.dequantize(data, qtype)
            t = torch.from_numpy(arr).float()

    return t


def load_gguf(path, dequantize=True):
    """Load a GGUF file and return (state_dict, metadata).

    Args:
        path: Path to the .gguf file.
        dequantize: If True (default), return dequantized float32 tensors.
                    If False, return raw numpy memmaps with tensor_type info.

    Returns:
        state_dict: {tensor_name: torch.Tensor} (or raw tuple if not dequantize)
        metadata: dict with arch, file_type, quantization_version
    """
    reader = gguf.GGUFReader(path)

    metadata = {
        "arch":                 _read_str_field(reader, "general.architecture"),
        "file_type":            _read_scalar_field(reader, "general.file_type", int),
        "quantization_version": _read_scalar_field(reader, "general.quantization_version", int),
    }

    qtype_counts = {}
    state_dict = {}

    for tensor in reader.tensors:
        name = tensor.name
        qtype = tensor.tensor_type
        qtype_str = getattr(qtype, "name", repr(qtype))
        qtype_counts[qtype_str] = qtype_counts.get(qtype_str, 0) + 1

        if dequantize:
            t = _dequantize(tensor)

            # Restore original shape if the converter applied shape_fix reshape
            orig_shape = _read_orig_shape(reader, name)
            if orig_shape is not None:
                t = t.reshape(orig_shape)
            else:
                # GGUF stores dims in reversed order
                shape = torch.Size(reversed([int(d) for d in tensor.shape]))
                t = t.reshape(shape)

            state_dict[name] = t
        else:
            shape = _read_orig_shape(reader, name) or torch.Size(
                reversed([int(d) for d in tensor.shape])
            )
            state_dict[name] = (tensor.data, qtype, shape)

    logger.info("Loaded %d tensors: %s", len(state_dict),
                ", ".join(f"{k}={v}" for k, v in qtype_counts.items()))

    return state_dict, metadata
