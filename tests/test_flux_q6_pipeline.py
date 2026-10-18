"""Exercise opt-in Python FLUX Q6_K/Q5_K_M through real preparation and API jobs."""
import queue
import threading

import gguf
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from safetensors.torch import save_file

import conversion_service
import web_api
from convert import ConversionCancelled
from scripts.gguf_research_io import open_gguf
from scripts.reference_q6_k import quantize_q6_k
from scripts.reference_q5_k import quantize_q5_k
from tests.test_web_api import _wait


def _source(tmp_path):
    path = tmp_path / 'flux.safetensors'
    values = torch.linspace(-1, 2, 2048).reshape(8, 256)
    save_file({'double_blocks.0.img_attn.proj.weight': values,
               'img_in.weight': torch.ones(8, 256)}, str(path))
    return path, values


def _no_native(*args, **kwargs):
    pytest.fail('Explicit Python conversion must not invoke or locate a native quantizer')


@pytest.mark.parametrize('preset,backend', [('Q6_K', 'python_flux_q6'), ('Q6_K', 'python_flux'), ('Q5_K_M', 'python_flux')])
@pytest.mark.parametrize('keep', [False, True])
def test_real_pipeline_reuses_float_preparation_without_native(tmp_path, monkeypatch, keep, preset, backend):
    source, values = _source(tmp_path)
    destination = tmp_path / 'out.gguf'
    events = queue.Queue()
    monkeypatch.setattr(conversion_service, 'run_quantize', _no_native)
    result = conversion_service._pipeline(str(source), str(destination), preset, '', 0,
                                          keep, False, events, gguf_backend=backend)
    assert result == str(destination)
    with open_gguf(destination) as reader:
        assert reader.get_field('general.architecture').contents() == 'flux'
        tensors = {t.name: t for t in reader.tensors}
        tensor = tensors['double_blocks.0.img_attn.proj.weight']
        assert tensor.tensor_type == gguf.GGMLQuantizationType[preset.replace("_M", "")]
        expected = (quantize_q6_k if preset == "Q6_K" else quantize_q5_k)(
            values.numpy().astype(np.float16).astype(np.float32))
        assert tensor.data.tobytes() == expected.tobytes()
        assert tensors['img_in.weight'].tensor_type == gguf.GGMLQuantizationType.F32
    intermediates = list(tmp_path.glob('*-F16-tmp-*.gguf'))
    assert len(intermediates) == int(keep)
    progress = [e[1] for e in events.queue if e[0] == 'progress_frac']
    assert progress and progress == sorted(progress) and max(progress) <= 1


@pytest.mark.parametrize('preset', ['Q6_K', 'Q5_K_M'])
@pytest.mark.parametrize('failure', ['cancel', 'write'])
@pytest.mark.parametrize('keep', [False, True])
def test_failure_preserves_previous_output_and_respects_intermediate_choice(tmp_path, monkeypatch, failure, keep, preset):
    source, _ = _source(tmp_path)
    destination = tmp_path / 'out.gguf'
    destination.write_bytes(b'previous model')
    event = threading.Event()

    class Events(queue.Queue):
        def put(self, item):
            super().put(item)
            if failure == 'cancel' and item[0] == 'progress_frac' and item[2].startswith('[2/2]'):
                event.set()

    if failure == 'write':
        def fail(*args, **kwargs):
            raise OSError('disk full')
        monkeypatch.setattr('scripts.requantize_flux_q6.iter_q6_k' if preset == 'Q6_K'
                            else 'scripts.requantize_flux_q6.iter_q5_k', fail)
    monkeypatch.setattr(conversion_service, 'run_quantize', _no_native)
    with pytest.raises(ConversionCancelled if failure == 'cancel' else OSError):
        conversion_service._pipeline(str(source), str(destination), preset, '', 0,
                                     keep, True, Events(), event, gguf_backend='python_flux')
    assert destination.read_bytes() == b'previous model'
    assert len(list(tmp_path.glob('*-F16-tmp-*.gguf'))) == int(keep)
    assert not list(tmp_path.glob('.out.gguf.*'))


def test_native_remains_default(tmp_path, monkeypatch):
    source, _ = _source(tmp_path)
    destination = tmp_path / 'out.gguf'
    calls = []
    def native(src, dst, quant_type, **kwargs):
        calls.append(quant_type)
        destination.write_bytes(b'native result')
    monkeypatch.setattr(conversion_service, 'run_quantize', native)
    conversion_service._pipeline(str(source), str(destination), 'Q6_K', '', 0,
                                 False, False, queue.Queue())
    assert calls == ['Q6_K'] and destination.read_bytes() == b'native result'


def _client():
    client = TestClient(web_api.create_app(), base_url='http://localhost')
    client.headers['X-Workbench-Token'] = client.get('/api/config').json()['token']
    return client


@pytest.mark.parametrize('preset,backend', [('Q6_K', 'python_flux_q6'), ('Q6_K', 'python_flux'), ('Q5_K_M', 'python_flux')])
def test_python_api_inspection_and_real_job_skip_native_discovery(tmp_path, monkeypatch, preset, backend):
    source, _ = _source(tmp_path)
    client = _client()
    monkeypatch.setattr(web_api, 'find_exe', _no_native)
    monkeypatch.setattr(conversion_service, 'run_quantize', _no_native)
    params = {'source': str(source), 'destination': str(tmp_path / 'out.gguf'),
              'format': preset, 'gguf_backend': backend}
    assert client.post('/api/inspect', json=params).status_code == 200
    response = client.post('/api/jobs', json=params)
    assert response.status_code == 202, response.text
    result = _wait(client, response.json()['id'])
    assert result['status'] == 'succeeded', result
    assert any(f'experimental Python FLUX {preset}' in line for line in result['logs'])
    assert not list(tmp_path.glob('*-F16-tmp-*.gguf'))


@pytest.mark.parametrize('backend', ['python_flux_q6', 'python_flux'])
@pytest.mark.parametrize('override', [
    {'format': 'Q4_K_M'}, {'format': 'F16'}, {'container': 'safetensors'},
    {'model_kind': 'text_encoder'}, {'executable': 'quantizer.exe'}, {'threads': 2},
    {'precision_profile': 'conservative'},
])
def test_invalid_python_api_choices_are_rejected_before_work(tmp_path, monkeypatch, override, backend):
    source, _ = _source(tmp_path)
    client = _client()
    monkeypatch.setattr(conversion_service, 'convert_file', _no_native)
    params = {'source': str(source), 'format': 'Q6_K', 'gguf_backend': backend, **override}
    for endpoint in ('inspect', 'jobs'):
        response = client.post('/api/' + endpoint, json=params)
        assert response.status_code == 400, response.text
    assert not list(tmp_path.glob('*-F16-tmp-*.gguf'))


def test_wrong_architecture_and_unknown_backend_rejected(tmp_path, monkeypatch):
    source = tmp_path / 'sd3.safetensors'
    save_file({'joint_blocks.0.x_block.attn.qkv.weight': torch.ones(8, 256)}, str(source))
    client = _client()
    params = {'source': str(source), 'format': 'Q6_K', 'gguf_backend': 'python_flux_q6'}
    monkeypatch.setattr(conversion_service, 'convert_file', _no_native)
    for endpoint in ('inspect', 'jobs'):
        response = client.post('/api/' + endpoint, json=params)
        assert response.status_code == 400 and 'FLUX only' in response.text
    assert client.post('/api/jobs', json=params | {'gguf_backend': 'unknown'}).status_code == 422
    with pytest.raises(ValueError, match='FLUX only'):
        conversion_service._pipeline(str(source), str(tmp_path / 'out.gguf'), 'Q6_K', '', 0,
                                     False, False, queue.Queue(), gguf_backend='python_flux_q6')



def test_legacy_q6_identifier_does_not_silently_expand_to_q5(tmp_path):
    """Old clients retain their Q6-only contract; new clients use python_flux."""
    source, _ = _source(tmp_path)
    client = _client()
    for endpoint in ('inspect', 'jobs'):
        response = client.post('/api/' + endpoint, json={
            'source': str(source), 'format': 'Q5_K_M', 'gguf_backend': 'python_flux_q6'})
        assert response.status_code == 400


@pytest.mark.parametrize('preset', ['Q4_K_S', 'Q4_K_M', 'Q5_K_M', 'Q6_K'])
def test_bundled_kernel_api_matches_python_writer(tmp_path, monkeypatch, preset):
    """Real shipped DLL, F16 preparation, API dispatch and atomic writer agree."""
    from scripts.compiled_kernel_package import _platform
    if _platform() != {'system': 'win32', 'machine': 'x86_64', 'pointer_bits': 64}:
        pytest.skip('Shipped Windows package qualification')
    from scripts.requantize_flux_q6 import requantize
    source, _ = _source(tmp_path)
    client = _client()
    monkeypatch.setattr(web_api, 'find_exe', _no_native)
    monkeypatch.setattr(conversion_service, 'run_quantize', _no_native)
    output = tmp_path / 'compiled.gguf'
    params = {'source': str(source), 'destination': str(output), 'format': preset,
              'gguf_backend': 'compiled_flux', 'keep_intermediate': True}
    response = client.post('/api/jobs', json=params)
    assert response.status_code == 202, response.text
    result = _wait(client, response.json()['id'])
    assert result['status'] == 'succeeded', result
    intermediate, = tmp_path.glob('*-F16-tmp-*.gguf')
    expected = tmp_path / 'python.gguf'
    requantize(intermediate, expected, preset=preset)
    assert output.read_bytes() == expected.read_bytes()


def test_bundled_kernel_failure_precedes_preparation(tmp_path, monkeypatch):
    source, _ = _source(tmp_path)
    def fail(*args, **kwargs):
        raise ValueError('Kernel package checksum/path mismatch')
    monkeypatch.setattr('scripts.compiled_kernel_package._platform',
                        lambda: {'system': 'win32', 'machine': 'x86_64', 'pointer_bits': 64})
    monkeypatch.setattr('scripts.compiled_kernel_package.load_package', fail)
    monkeypatch.setattr(conversion_service, 'convert_file', _no_native)
    destination = tmp_path / 'out.gguf'
    destination.write_bytes(b'previous model')
    with pytest.raises(ValueError, match='checksum'):
        conversion_service._pipeline(str(source), str(destination), 'Q4_K_M', '', 0,
                                     False, True, queue.Queue(), gguf_backend='compiled_flux')
    assert destination.read_bytes() == b'previous model'
    assert not list(tmp_path.glob('*-F16-tmp-*.gguf'))


@pytest.mark.parametrize('preset', ['Q2_K', 'Q3_K_M'])
def test_low_bit_api_matches_recorded_independent_native_payloads(tmp_path, monkeypatch, preset):
    import hashlib
    import json
    from pathlib import Path
    from scripts.compiled_kernel_package import _platform
    if _platform() != {'system': 'win32', 'machine': 'x86_64', 'pointer_bits': 64}:
        pytest.skip('Shipped Windows x86_64 package qualification')
    source, _ = _source(tmp_path)
    client = _client()
    monkeypatch.setattr(web_api, 'find_exe', _no_native)
    output = tmp_path / 'out.gguf'
    params = {'source': str(source), 'destination': str(output), 'format': preset,
              'gguf_backend': 'compiled_flux'}
    assert client.post('/api/inspect', json=params).status_code == 200
    response = client.post('/api/jobs', json=params)
    assert response.status_code == 202, response.text
    result = _wait(client, response.json()['id'])
    assert result['status'] == 'succeeded', result
    report = json.loads((Path(__file__).parents[1] /
                         'data/compiled-kernel-levels-validation.json').read_text())
    expected = report['api_fixture_native'][preset]
    with open_gguf(output) as reader:
        actual = [{'name': t.name, 'type': t.tensor_type.name,
                   'payload_sha256': hashlib.sha256(t.data).hexdigest()} for t in reader.tensors]
    assert actual == expected
    assert not list(tmp_path.glob('*-F16-tmp-*.gguf'))


@pytest.mark.parametrize('preset', ['Q2_K', 'Q3_K_M'])
@pytest.mark.parametrize('outcome', ['cancel', 'failure', 'bad_bytes'])
def test_low_bit_pipeline_keeps_old_output_on_kernel_failure(tmp_path, monkeypatch, preset, outcome):
    from scripts.compiled_kernel_package import _platform
    if _platform() != {'system': 'win32', 'machine': 'x86_64', 'pointer_bits': 64}:
        pytest.skip('Shipped Windows x86_64 package qualification')
    source, _ = _source(tmp_path)
    destination = tmp_path / 'out.gguf'
    destination.write_bytes(b'previous model')
    event = threading.Event()
    from scripts.compiled_kernel_package import load_package, ROOT
    real_kernel, manifest = load_package(ROOT / 'third_party/cpu-kernels/win32-x86_64')
    def kernel(*args):
        if outcome == 'cancel':
            event.set()
            return real_kernel(*args)
        if outcome == 'failure':
            raise RuntimeError('Kernel failure')
        return np.zeros(1, dtype=np.uint8)
    monkeypatch.setattr('scripts.compiled_kernel_package.load_package', lambda *_: (kernel, manifest))
    expected = {'cancel': ConversionCancelled, 'failure': RuntimeError, 'bad_bytes': ValueError}[outcome]
    with pytest.raises(expected):
        conversion_service._pipeline(str(source), str(destination), preset, '', 0,
                                     False, True, queue.Queue(), event, gguf_backend='compiled_flux')
    assert destination.read_bytes() == b'previous model'
    assert not list(tmp_path.glob('*-F16-tmp-*.gguf'))
    assert not list(tmp_path.glob('.out.gguf.*'))


@pytest.mark.parametrize('filename', ['kernel.json', 'converter-kernel.dll', 'LICENSE'])
def test_pipeline_cannot_overwrite_its_bundled_package(tmp_path, monkeypatch, filename):
    from scripts.compiled_kernel_package import ROOT
    source, _ = _source(tmp_path)
    destination = ROOT / 'third_party/cpu-kernels/win32-x86_64' / filename
    before = destination.read_bytes()
    monkeypatch.setattr('scripts.compiled_kernel_package._platform',
                        lambda: {'system': 'win32', 'machine': 'x86_64', 'pointer_bits': 64})
    monkeypatch.setattr('scripts.compiled_kernel_package.load_package', _no_native)
    monkeypatch.setattr(conversion_service, 'convert_file', _no_native)
    with pytest.raises(ValueError, match='different files'):
        conversion_service._pipeline(str(source), str(destination), 'Q2_K', '', 0,
                                     False, True, queue.Queue(), gguf_backend='compiled_flux')
    assert destination.read_bytes() == before
