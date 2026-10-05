"""Local API security, real conversions, and cooperative job lifecycle checks."""
import time
import threading

import pytest
import torch
from fastapi.testclient import TestClient
from safetensors.torch import save_file

import web_api


def _client():
    client = TestClient(web_api.create_app(), base_url='http://localhost')
    response = client.get('/api/config')
    assert response.headers['cache-control'] == 'no-store'
    client.headers['X-Workbench-Token'] = response.json()['token']
    return client


def _source(tmp_path):
    source = tmp_path / 'qwen.safetensors'
    save_file({'transformer_blocks.0.img_mod.1.weight': torch.randn(128, 256),
               'time_text_embed.timestep_embedder.linear_2.weight': torch.ones(2, 2),
               'transformer_blocks.0.attn.norm_added_q.weight': torch.ones(2),
               'transformer_blocks.0.img_mlp.net.0.proj.weight': torch.randn(128, 256),
               '__index_timestep_zero__': torch.empty(0)}, str(source))
    return source


def _wait(client, job_id):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        state = client.get(f'/api/jobs/{job_id}').json()
        if state['status'] in {'succeeded', 'failed', 'cancelled'}:
            return state
        time.sleep(.02)
    pytest.fail('Conversion worker did not finish')


def test_session_origin_and_file_browser(tmp_path):
    """Other websites cannot read paths or issue local conversion commands."""
    client = _client()
    _source(tmp_path)
    response = client.get('/api/files', params={'path': str(tmp_path)})
    assert response.status_code == 200
    assert response.json()['entries'][0]['name'] == 'qwen.safetensors'
    assert client.get('/api/files', headers={'X-Workbench-Token': ''}).status_code == 403
    assert client.get('/api/files', headers={'Origin': 'https://evil.example'}).status_code == 403
    assert client.get('/api/files', headers={'Host': 'evil.example'}).status_code == 400
    assert client.get('/api/files', params={'path': str(tmp_path / 'missing')}).status_code == 400


@pytest.mark.parametrize(('container', 'format_key'), [('safetensors', 'FP8_MIXED'), ('gguf', 'F16')])
def test_real_conversion_and_inspection(tmp_path, container, format_key):
    """Exercise actual writers, estimates and overwrite protection through HTTP."""
    client = _client()
    source = _source(tmp_path)
    output = tmp_path / f'converted.{container}'
    params = {'source': str(source), 'destination': str(output),
              'container': container, 'format': format_key}
    inspection = client.post('/api/inspect', json=params)
    assert inspection.status_code == 200, inspection.text
    assert inspection.json()['architecture'] == 'qwen_image'
    assert inspection.json()['estimated_bytes'] > 0
    started = client.post('/api/jobs', json=params)
    assert started.status_code == 202, started.text
    state = _wait(client, started.json()['id'])
    assert state['status'] == 'succeeded', state
    assert state['progress'] == 1 and output.stat().st_size > 0
    assert source.exists()
    assert client.post('/api/jobs', json=params).status_code == 400
    assert client.post('/api/jobs', json={**params, 'destination': str(source)}).status_code == 400
    assert client.post('/api/jobs', json={**params, 'format': 'INVENTED'}).status_code == 400
    assert client.post('/api/jobs', json={**params, 'threads': -1}).status_code == 422


def test_single_worker_cancellation_and_failure(tmp_path, monkeypatch):
    """Concurrent jobs are rejected and cancellation reaches the writer's event."""
    client = _client()
    source = _source(tmp_path)
    entered = threading.Event()

    def blocked(*args, cancel_event, **kwargs):
        entered.set()
        assert cancel_event.wait(5)
        raise RuntimeError('cancelled')

    monkeypatch.setattr(web_api, 'convert_to_safetensors', blocked)
    params = {'source': str(source), 'destination': str(tmp_path / 'output.safetensors'),
              'container': 'safetensors', 'format': 'FP8'}
    job_id = client.post('/api/jobs', json=params).json()['id']
    assert entered.wait(5)
    assert client.post('/api/jobs', json=params).status_code == 409
    assert client.post(f'/api/jobs/{job_id}/cancel', json={}).status_code == 200
    assert _wait(client, job_id)['status'] == 'cancelled'
    assert not (tmp_path / 'output.safetensors').exists()
    assert client.get('/api/jobs/missing').status_code == 404

    def fail(*args, **kwargs):
        raise ValueError('deliberate failure')

    monkeypatch.setattr(web_api, 'convert_to_safetensors', fail)
    failed = client.post('/api/jobs', json=params).json()['id']
    assert _wait(client, failed)['error'] == 'deliberate failure'
