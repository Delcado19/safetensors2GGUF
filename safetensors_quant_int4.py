"""Native ComfyUI ConvRot W4A4 storage, matching Kitchen 0.2.36's eager codec."""

import torch

from safetensors_quant import layer_key
from safetensors_quant_int8 import _build_hadamard, _rotate_weight


def eligible_int4(key, shape, model_arch):
    """Pack aligned Linear weights only; preserve detection and raw-access weights."""
    return (key.endswith('.weight') and len(shape) == 2 and
            shape[0] > 0 and shape[0] % 16 == 0 and shape[1] > 0 and shape[1] % 256 == 0 and
            not any(x in key for x in getattr(model_arch, 'keys_shape_critical', [])))


def quantize_int4_convrot(data, key):
    """Rotate, quantize symmetrically to [-7, 7], pack even columns in low nibbles."""
    h = _build_hadamard(256, data.device, data.dtype)
    rotated = _rotate_weight(data, h, 256)
    scales = rotated.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10) / 7
    # FP16 cannot represent Kitchen's 1e-10 floor; keep zero rows finite.
    scales = torch.where(scales == 0, torch.finfo(data.dtype).tiny, scales)
    codes = (rotated / scales).round().clamp(-7, 7).to(torch.int8)
    packed = ((codes[:, 0::2].to(torch.int32) & 15) |
              ((codes[:, 1::2].to(torch.int32) & 15) << 4)).to(torch.int8)
    return {key: packed, f'{layer_key(key)}.weight_scale': scales.reshape(-1).float()}


def dequantize_int4_convrot(data, scales, groupsize=256):
    """Restore signed nibbles and undo rotation for safe reconversion."""
    values = torch.stack((data.to(torch.int32) & 15, (data.to(torch.int32) >> 4) & 15), dim=-1)
    values = torch.where(values >= 8, values - 16, values).reshape(data.shape[0], -1).float()
    rotated = values * scales.float().reshape(-1, 1)
    h = _build_hadamard(groupsize, data.device, torch.float32)
    return _rotate_weight(rotated, h, groupsize)
