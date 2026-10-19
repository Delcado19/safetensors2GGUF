"""Frozen new model adapters and owned CPU quantization execute on real fixtures."""
import queue

import pytest
import torch
from safetensors.torch import save_file

from extended_backend import convert_extended, EXTENDED_ARCHES, FORMATS, _convrot8, _verify_payloads
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
@pytest.mark.parametrize('preset', FORMATS)
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


@pytest.mark.parametrize('damage', ['bytes', 'type', 'shape', 'missing', 'extra', 'architecture', 'text_only'])
def test_readback_rejects_corruption_and_contract_mismatch(tmp_path, damage):
    import hashlib
    import gguf
    import numpy as np

    output = tmp_path / 'fixture.gguf'
    data = np.ones((2, 32), dtype=np.float32)
    writer = gguf.GGUFWriter(str(output), 'ideogram')
    writer.add_tensor('probe.weight', data)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    plan = {'probe.weight': (gguf.GGMLQuantizationType.F32, data.shape, hashlib.sha256(data).hexdigest())}
    _verify_payloads(output, plan, 'ideogram')
    if damage == 'bytes':
        with open_gguf(output) as reader:
            offset = int(reader.tensors[0].data_offset)
        with output.open('r+b') as stream:
            stream.seek(offset)
            stream.write(b'\x01')
    elif damage == 'type':
        plan['probe.weight'] = (gguf.GGMLQuantizationType.F16, data.shape, plan['probe.weight'][2])
    elif damage == 'shape':
        plan['probe.weight'] = (gguf.GGMLQuantizationType.F32, (32, 2), plan['probe.weight'][2])
    elif damage == 'missing':
        plan.clear()
    elif damage == 'extra':
        plan['missing.weight'] = plan['probe.weight']
    with pytest.raises(ValueError, match='Extended output|Text-only VL'):
        _verify_payloads(output, plan, 'wrong' if damage == 'architecture' else 'ideogram',
                         text_only=damage == 'text_only')


def test_mismatched_architecture_preserves_existing_destination(tmp_path):
    source, output = tmp_path / 'source.safetensors', tmp_path / 'out.gguf'
    source_file(source, 'ideogram')
    output.write_bytes(b'previous output')
    with pytest.raises(RuntimeError, match='architecture mismatch'):
        convert_extended(source, output, 'F16', arch='minimax_h3', overwrite=True)
    assert output.read_bytes() == b'previous output'
    assert {p.name for p in tmp_path.iterdir()} == {'source.safetensors', 'out.gguf'}


def test_flux3_application_dispatch_and_patch_projection_protection(tmp_path):
    import gguf
    from conversion_service import _pipeline
    source, output = tmp_path / 'flux3.safetensors', tmp_path / 'flux3.gguf'
    source_file(source, 'flux3')
    from safetensors.torch import load_file
    # Detach mmap-backed tensors before replacing the fixture file on Windows.
    tensors = {key: value.clone() for key, value in load_file(source).items()}
    tensors['final_layer.image.linear.weight'] = torch.ones(32, 256, dtype=torch.bfloat16)
    tensors['emb_in.image.weight'] = torch.ones(32, 128, dtype=torch.bfloat16)
    save_file(tensors, source)
    events = queue.Queue()
    _pipeline(str(source), str(output), 'Q4_K_M', '', 0, False, False, events)
    with open_gguf(output) as reader:
        assert reader.get_field('general.architecture').contents() == 'flux3'
        types = {tensor.name: tensor.tensor_type for tensor in reader.tensors}
        assert types['final_layer.image.linear.weight'] == gguf.GGMLQuantizationType.F32
        assert types['emb_in.image.weight'] in {gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.BF16}
        assert types['probe.weight'] in {gguf.GGMLQuantizationType.Q4_K, gguf.GGMLQuantizationType.Q6_K}
    assert any(event[0] == 'progress_frac' and event[1] > 0 for event in list(events.queue))
