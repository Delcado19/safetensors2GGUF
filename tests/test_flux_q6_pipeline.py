"""Exercise opt-in Python FLUX Q6_K through real preparation and API jobs."""
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
from tests.test_web_api import _wait


def _source(tmp_path):
    path = tmp_path / 'flux.safetensors'
    values = torch.linspace(-1, 2, 2048).reshape(8, 256)
    save_file({'double_blocks.0.img_attn.proj.weight': values,
               'img_in.weight': torch.ones(8, 256)}, str(path))
    return path, values


def _no_native(*args, **kwargs):
    pytest.fail('Explicit Python conversion must not invoke or locate a native quantizer')


@pytest.mark.parametrize('keep', [False, True])
def test_real_pipeline_reuses_float_preparation_without_native(tmp_path, monkeypatch, keep):
    source, values = _source(tmp_path)
    destination = tmp_path / 'out.gguf'
    events = queue.Queue()
    monkeypatch.setattr(conversion_service, 'run_quantize', _no_native)
    result = conversion_service._pipeline(str(source), str(destination), 'Q6_K', '', 0,
                                          keep, False, events, gguf_backend='python_flux_q6')
    assert result == str(destination)
    with open_gguf(destination) as reader:
        assert reader.get_field('general.architecture').contents() == 'flux'
        tensors = {t.name: t for t in reader.tensors}
        tensor = tensors['double_blocks.0.img_attn.proj.weight']
        assert tensor.tensor_type == gguf.GGMLQuantizationType.Q6_K
        expected = quantize_q6_k(values.numpy().astype(np.float16).astype(np.float32))
        assert tensor.data.tobytes() == expected.tobytes()
        assert tensors['img_in.weight'].tensor_type == gguf.GGMLQuantizationType.F32
    intermediates = list(tmp_path.glob('*-F16-tmp-*.gguf'))
    assert len(intermediates) == int(keep)
    progress = [e[1] for e in events.queue if e[0] == 'progress_frac']
    assert progress and progress == sorted(progress) and max(progress) <= 1


@pytest.mark.parametrize('failure', ['cancel', 'write'])
@pytest.mark.parametrize('keep', [False, True])
def test_failure_preserves_previous_output_and_respects_intermediate_choice(tmp_path, monkeypatch, failure, keep):
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
        monkeypatch.setattr('scripts.requantize_flux_q6.iter_q6_k', fail)
    monkeypatch.setattr(conversion_service, 'run_quantize', _no_native)
    with pytest.raises(ConversionCancelled if failure == 'cancel' else OSError):
        conversion_service._pipeline(str(source), str(destination), 'Q6_K', '', 0,
                                     keep, True, Events(), event, gguf_backend='python_flux_q6')
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


def test_python_api_inspection_and_real_job_skip_native_discovery(tmp_path, monkeypatch):
    source, _ = _source(tmp_path)
    client = _client()
    monkeypatch.setattr(web_api, 'find_exe', _no_native)
    monkeypatch.setattr(conversion_service, 'run_quantize', _no_native)
    params = {'source': str(source), 'destination': str(tmp_path / 'out.gguf'),
              'format': 'Q6_K', 'gguf_backend': 'python_flux_q6'}
    assert client.post('/api/inspect', json=params).status_code == 200
    response = client.post('/api/jobs', json=params)
    assert response.status_code == 202, response.text
    result = _wait(client, response.json()['id'])
    assert result['status'] == 'succeeded', result
    assert any('experimental Python FLUX Q6_K' in line for line in result['logs'])
    assert not list(tmp_path.glob('*-F16-tmp-*.gguf'))


@pytest.mark.parametrize('override', [
    {'format': 'Q4_K_M'}, {'format': 'F16'}, {'container': 'safetensors'},
    {'model_kind': 'text_encoder'}, {'executable': 'quantizer.exe'}, {'threads': 2},
    {'precision_profile': 'conservative'},
])
def test_invalid_python_api_choices_are_rejected_before_work(tmp_path, monkeypatch, override):
    source, _ = _source(tmp_path)
    client = _client()
    monkeypatch.setattr(conversion_service, 'convert_file', _no_native)
    params = {'source': str(source), 'format': 'Q6_K', 'gguf_backend': 'python_flux_q6', **override}
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
