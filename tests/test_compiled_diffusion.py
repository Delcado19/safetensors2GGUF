"""All diffusion adapters must use the real bundled kernel and preserve special shapes."""
import queue
import threading

import gguf
import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from compiled_diffusion import PROTECTED, quantizable
from conversion_service import _pipeline
from models.architectures import arch_list
from scripts.gguf_research_io import open_gguf
from scripts.requantize_flux_q6 import FLUX_PRESETS
from scripts.compiled_kernel_package import _platform

pytestmark = pytest.mark.skipif(
    _platform() != {'system': 'win32', 'machine': 'x86_64', 'pointer_bits': 64},
    reason='Shipped Windows x86_64 package qualification')


def source_file(path, arch):
    model = next(model for model in arch_list if model.arch == arch)
    keys = next(keys for keys in model.keys_detect if not any(key in model.keys_banned for key in keys))
    data = {key: torch.ones(8, 256) for key in keys}
    data.update({'probe.linear.weight': torch.linspace(-2, 3, 8 * 256).reshape(8, 256),
                 'probe.to_v.weight': torch.ones(8, 256), 'probe.ffn_down.weight': torch.ones(8, 256),
                 'probe.odd.weight': torch.ones(8, 160), 'probe.norm.weight': torch.ones(256)})
    if arch in ('wan', 'hyvid'):
        data['probe.conv.weight'] = torch.ones(2, 4, 4, 4, 32, dtype=torch.float16)
        data['probe.modulation'] = torch.ones(1, 6, 256)
    if arch == 'wan':
        data['head.modulation'] = torch.ones(1, 6, 256)
    if arch == 'lumina2':
        data['x_pad_token'] = torch.ones(256)
        data['cap_pad_token'] = torch.ones(256)
    if arch == 'sd3':
        data['pos_embed'] = torch.ones(1, 8, 256)
    save_file(data, str(path))


@pytest.mark.parametrize('arch', PROTECTED)
@pytest.mark.parametrize('preset', FLUX_PRESETS)
def test_all_diffusion_adapters_use_bundled_math(tmp_path, arch, preset):
    source, output = tmp_path / 'source.safetensors', tmp_path / 'out.gguf'
    source_file(source, arch)
    original = source.read_bytes()
    _pipeline(str(source), str(output), preset, '', 0, False, False, queue.Queue(),
              gguf_backend='compiled_diffusion')
    with open_gguf(output) as reader:
        assert reader.get_field('general.architecture').contents() == arch
        tensors = {tensor.name: tensor for tensor in reader.tensors}
        assert tensors['probe.linear.weight'].tensor_type.name == preset[:4]
        assert tensors['probe.odd.weight'].tensor_type.name == (preset[:4] if arch in ('sd1', 'sdxl') else 'F16')
        if arch in ('wan', 'hyvid'):
            assert tensors['probe.conv.weight'].shape.tolist() == [32, 4, 4, 4, 2]
            assert tensors['probe.conv.weight'].tensor_type == gguf.GGMLQuantizationType.F32
            assert tensors['probe.modulation'].shape.tolist() == [256, 6, 1]
        if arch == 'lumina2':
            assert tensors['x_pad_token'].shape.tolist() == [256, 1]
        if arch == 'sd3':
            assert tensors['pos_embed'].shape.tolist() == [256, 8, 1]
    assert source.read_bytes() == original
    assert {path.name for path in tmp_path.iterdir()} == {'source.safetensors', 'out.gguf'}


@pytest.mark.parametrize('arch', PROTECTED)
def test_retained_preparation_matches_direct_payloads(tmp_path, arch):
    source = tmp_path / 'source.safetensors'
    source_file(source, arch)
    for keep in (False, True):
        _pipeline(str(source), str(tmp_path / f'{keep}.gguf'), 'Q4_K_M', '', 0, keep,
                  False, queue.Queue(), gguf_backend='compiled_diffusion')
    with open_gguf(tmp_path / 'False.gguf') as direct, open_gguf(tmp_path / 'True.gguf') as retained:
        a, b = ({tensor.name: (tensor.shape.tolist(), tensor.tensor_type, tensor.data.copy())
                 for tensor in reader.tensors} for reader in (direct, retained))
        assert a.keys() == b.keys()
        for key in a:
            assert a[key][:2] == b[key][:2]
            assert np.array_equal(a[key][2], b[key][2])


@pytest.mark.parametrize('arch', PROTECTED)
def test_architecture_protections(arch):
    for part in PROTECTED[arch]:
        assert not quantizable(arch, part + 'weight', [8, 256])


def test_cancel_preserves_existing_output(tmp_path):
    source, output = tmp_path / 'source.safetensors', tmp_path / 'out.gguf'
    source_file(source, 'wan')
    output.write_bytes(b'old model')
    cancel = threading.Event()
    cancel.set()
    from convert import ConversionCancelled
    with pytest.raises(ConversionCancelled):
        _pipeline(str(source), str(output), 'Q2_K', '', 0, False, True, queue.Queue(), cancel,
                  gguf_backend='compiled_diffusion')
    assert output.read_bytes() == b'old model'


@pytest.mark.parametrize('arch', PROTECTED)
def test_workbench_api_accepts_other_model_families(tmp_path, monkeypatch, arch):
    import web_api
    from tests.test_web_api import _client, _wait
    source, output = tmp_path / 'source.safetensors', tmp_path / 'out.gguf'
    source_file(source, arch)
    def no_native(*args, **kwargs):
        pytest.fail('Bundled diffusion must not discover or invoke an external quantizer')
    client = _client()
    monkeypatch.setattr(web_api, 'find_exe', no_native)
    params = {'source': str(source), 'destination': str(output), 'format': 'Q2_K',
              'gguf_backend': 'compiled_diffusion'}
    assert client.post('/api/inspect', json=params).status_code == 200
    response = client.post('/api/jobs', json=params)
    assert response.status_code == 202, response.text
    result = _wait(client, response.json()['id'])
    assert result['status'] == 'succeeded', result


def test_recorded_diffusion_parity_and_current_sources():
    import hashlib
    import json
    from pathlib import Path
    root = Path(__file__).parents[1]
    report = json.loads((root / 'data/compiled-diffusion-requalification-2026-10-19.json').read_text())
    assert report['all_equal'] and len(report['results']) == 66
    assert {row['architecture'] for row in report['results']} == set(PROTECTED) - {'qwen_image'}
    for row in report['results']:
        assert row['comparison']['all_equal']
    for name, digest in report['source_code_sha256'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
