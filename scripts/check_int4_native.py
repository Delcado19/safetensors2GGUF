"""Check our INT4 codec against the installed Kitchen layout, without inference.

Run with a ComfyUI Python environment providing comfy-kitchen 0.2.36+.
This checks the storage contract, not model rendering or GPU compute support.
"""

import sys
from importlib.metadata import version
from pathlib import Path

import torch
from comfy_kitchen.tensor import TensorCoreConvRotW4A4Layout

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from safetensors_quant_int4 import dequantize_int4_convrot, quantize_int4_convrot  # noqa: E402


def main():
    """Fail if the installed native codec differs from the writer's pinned contract."""
    print('Kitchen', version('comfy-kitchen'))
    torch.manual_seed(12)
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        weight = torch.randn(32, 512).to(dtype)
        actual = quantize_int4_convrot(weight, 'block.weight')
        packed, params = TensorCoreConvRotW4A4Layout.quantize(weight)
        assert torch.equal(packed, actual['block.weight'])
        assert torch.equal(params.scale, actual['block.weight_scale'])
        native = TensorCoreConvRotW4A4Layout.dequantize(packed, params)
        restored = dequantize_int4_convrot(packed, params.scale).to(dtype)
        assert torch.equal(native, restored)
        print(dtype, 'native packing, scales and reconstruction identical')


if __name__ == '__main__':
    main()
