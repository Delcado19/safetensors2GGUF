"""Frozen new model adapters and owned CPU quantization execute on real fixtures."""
import queue

import pytest
import torch
from safetensors.torch import save_file

from extended_backend import convert_extended, EXTENDED_ARCHES, _convrot8
from models.architectures import arch_list, detect_arch
from scripts.gguf_research_io import open_gguf


def source_file(path, arch):
    model = next(model for model in arch_list if model.arch == arch)
    tensors = {key: torch.ones(32, 256) for key in model.keys_detect[0]}
    tensors['probe.weight'] = torch.linspace(-1, 2, 32 * 256).reshape(32, 256)
    if arch in ('ltxv_upscaler', 'minimax_h3_vae'):
        tensors['probe.conv.weight'] = torch.ones(2, 3, 2, 4, 4)
    save_file(tensors, str(path))


@pytest.mark.parametrize('arch', sorted(EXTENDED_ARCHES))
@pytest.mark.parametrize('preset', ['F16', 'Q4_K_M', 'Q2_K', 'Q3_K_M', 'Q8_CR', 'Q4_CR'])
def test_frozen_extended_architecture_exports(tmp_path, arch, preset):
    from convert import load_state_dict
    source, output = tmp_path / 'source.safetensors', tmp_path / 'out.gguf'
    source_file(source, arch)
    original = source.read_bytes()
    assert detect_arch(load_state_dict(str(source))).arch == arch
    convert_extended(source, output, preset, arch=arch)
    with open_gguf(output) as reader:
        assert reader.get_field('general.architecture').contents() == arch
        assert reader.tensors
        if arch in ('ltxv_upscaler', 'minimax_h3_vae'):
            assert reader.get_field('comfy.gguf.orig_shape.probe.conv.weight').contents() == [2, 3, 2, 4, 4]
    assert source.read_bytes() == original


def test_krea_new_k_route_skips_external_native_binary(tmp_path):
    from conversion_service import _pipeline
    source, output = tmp_path / 'krea.safetensors', tmp_path / 'out.gguf'
    source_file(source, 'krea2')
    assert _pipeline(str(source), str(output), 'Q2_K', '', 0, False, False, queue.Queue()) == str(output)


def test_owned_convrot8_codec_contract():
    data = torch.linspace(-2, 3, 32 * 256).reshape(32, 256)
    packed, scale, config, shape = _convrot8(data)
    assert shape == (32, 256) and packed.dtype == torch.int8
    assert scale.shape == (32, 1) and torch.isfinite(scale).all()
    assert config['convrot_groupsize'] == 256 and config['weight_rotated']
