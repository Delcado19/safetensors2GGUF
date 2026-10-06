"""Native INT4 storage, streamed planning and reconversion regressions."""

import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from convert_safetensors import convert_to_safetensors
from models.architectures import ModelFlux
from safetensors_quant import estimate_safetensors_output_size, plan_tensor_output, quantize_tensor_st
from safetensors_quant_int4 import dequantize_int4_convrot


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
def test_int4_streamed_roundtrip(tmp_path, dtype):
    arch = ModelFlux()
    weights = torch.randn(16, 256, generator=torch.Generator().manual_seed(12)).to(dtype)
    state = {'block.weight': weights, 'block.bias': torch.zeros(16, dtype=dtype),
             'conv.weight': torch.randn(16, 16, 3, 3).to(dtype),
             'unaligned.weight': torch.randn(17, 256).to(dtype),
             'zero.weight': torch.zeros(16, 256, dtype=dtype)}
    source = tmp_path / 'source.safetensors'
    save_file(state, str(source))
    estimate = estimate_safetensors_output_size(str(source), 'INT4_CONVROT_MIXED', arch)
    target, _ = convert_to_safetensors(str(source), target_key='INT4_CONVROT_MIXED', model_arch=arch)
    with safe_open(target, framework='pt') as file:
        config = json.loads(file.metadata()['_quantization_metadata'])['layers']['block']
        assert config == {'format': 'convrot_w4a4', 'convrot_groupsize': 256, 'linear_dtype': 'int4'}
        for key, value in state.items():
            entries, _ = plan_tensor_output(key, value.shape, value.dtype, arch, 'INT4_CONVROT_MIXED')
            for name, planned_dtype, shape in entries:
                actual = file.get_tensor(name)
                assert tuple(actual.shape) == shape
                assert actual.dtype == {'I8': torch.int8, 'F32': torch.float32,
                                        'F16': torch.float16, 'BF16': torch.bfloat16}[planned_dtype]
        for key in ('block.bias', 'conv.weight', 'unaligned.weight'):
            assert torch.equal(file.get_tensor(key), state[key])
        assert estimate == sum(file.get_tensor(k).numel() * file.get_tensor(k).element_size() for k in file.keys())
        restored = dequantize_int4_convrot(file.get_tensor('block.weight'), file.get_tensor('block.weight_scale'))
        assert (restored - weights.float()).square().mean().sqrt() < 0.2
        assert torch.isfinite(file.get_tensor('zero.weight_scale')).all()
        assert not file.get_tensor('zero.weight').count_nonzero()
    reconverted, _ = convert_to_safetensors(target, target_key='F16', model_arch=arch)
    with safe_open(reconverted, framework='pt') as file:
        assert file.get_tensor('block.weight').shape == weights.shape
        assert torch.allclose(file.get_tensor('block.weight').float(), restored, atol=0.002)


def test_int4_shape_critical_passthrough():
    arch = ModelFlux()
    arch.keys_shape_critical = ['sensitive']
    weight = torch.randn(16, 256)
    result = quantize_tensor_st(weight, 'sensitive.weight', arch, 'INT4_CONVROT_MIXED')
    assert result == {'sensitive.weight': weight}
