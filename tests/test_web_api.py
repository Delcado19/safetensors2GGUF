"""Local API security, real conversions, and cooperative job lifecycle checks."""
import time
import threading

import pytest
import torch
from fastapi.testclient import TestClient
from safetensors.torch import save_file
from safetensors import safe_open

import web_api
import hf_download
from types import SimpleNamespace
import shutil
import model_support
from pathlib import Path
import gguf
import numpy as np
from safetensors.torch import load_file


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


def test_krea_inspection_dispatch_and_format_guards(tmp_path):
    """Expose Krea capabilities explicitly and execute its pinned worker via the API."""
    source = tmp_path / 'krea.safetensors'
    save_file({'first.weight': torch.ones(64, 64),
               'txtfusion.projector.weight': torch.ones(64, 64),
               'blocks.0.attn.wq.weight': torch.randn(64, 64)}, str(source))
    client = _client()
    params = {'source': str(source), 'destination': str(tmp_path / 'krea.gguf'),
              'container': 'gguf', 'format': 'Q4_K_M'}
    inspection = client.post('/api/inspect', json=params)
    assert inspection.status_code == 200
    data = inspection.json()
    assert data['architecture'] == 'krea2' and data['backend'] == 'molbal'
    assert data['estimated_bytes'] is None and data['support'] == 'unknown'
    assert 'molbal GGUF loader' in data['support_reason']
    assert data['formats'] == [list(choice) for choice in web_api.KREA_CHOICES]
    assert client.post('/api/jobs', json=params).status_code == 400
    params['format'] = 'Q4_0'
    assert client.post('/api/jobs', json=params | {'threads': 2}).status_code == 400
    response = client.post('/api/jobs', json=params)
    assert response.status_code == 202
    job = _wait(client, response.json()['id'])
    assert job['status'] == 'succeeded', job['error']
    assert Path(job['output']).is_file()
    other = _source(tmp_path)
    assert client.post('/api/jobs', json=params | {'source': str(other), 'destination': str(tmp_path / 'other.gguf')}).status_code == 400


def test_support_matrix_uses_shared_classifications_and_reasons():
    """The workbench exposes the existing support registry, including negative evidence."""
    client = _client()
    response = client.get('/api/support')
    assert response.status_code == 200
    data = response.json()
    assert data == model_support.build_workbench_support_tables()
    assert len({row['id'] for row in data['diffusion']['rows']}) == len(data['diffusion']['rows'])
    assert len({row['id'] for row in data['text_encoder']['rows']}) == len(data['text_encoder']['rows'])
    assert [key for key in data['diffusion']['formats'] if key != 'INT4_CONVROT_MIXED'] == [key for _, key in model_support.TABLE_FORMATS]
    assert data['text_encoder']['formats'] == [key for _, key in model_support.TEXT_ENCODER_TABLE_FORMATS]
    clip = next(row for row in data['text_encoder']['rows'] if row['family'] == 'clip-l')
    assert clip['GGUF'] == 'bad' and clip['GGUF__reason']
    assert clip['F16'] == 'verified'
    qwen = next(row for row in data['diffusion']['rows'] if row['arch'] == 'qwen_image')
    assert qwen['GGUF'] == 'verified' and qwen['GGUF__scope'] == 'Q4_K_M smoke test'
    assert '3962' in qwen['GGUF__reason'] and 'quality parity' in qwen['GGUF__reason']
    krea = next(row for row in data['diffusion']['rows'] if row['arch'] == 'krea2')
    assert krea['GGUF'] == 'caution'
    assert krea['GGUF__scope'] == 'Q4_0 / molbal loader'
    assert krea['GGUF__format'] == 'Q4_0'
    assert 'GGUF__pending' not in krea
    assert 'Four successful renders' in krea['GGUF__reason']
    assert 'not an unquantized-original' in krea['GGUF__reason']
    assert qwen['NVFP4_MIXED'] == 'caution' and 'whiskers' in qwen['NVFP4_MIXED__reason']
    assert qwen['INT8_MIXED'] == 'caution' and 'small visible detail' in qwen['INT8_MIXED__reason']
    lumina = next(row for row in data['diffusion']['rows'] if row['id'] == 'lumina2_base')
    assert lumina['NVFP4_MIXED'] == 'bad'
    assert 'Default/Base policy' in lumina['NVFP4_MIXED__reason']
    turbo = next(row for row in data['diffusion']['rows'] if row['id'] == 'lumina2_turbo')
    assert turbo['NVFP4_MIXED'] == 'caution' and turbo['precision_profile'] == 'z_image_turbo'
    assert turbo['FP8_MIXED'] == 'unknown' and turbo['INT8_MIXED'] == 'unknown'
    assert 'Two prompts' in turbo['NVFP4_MIXED__reason']
    full = next(row for row in data['text_encoder']['rows'] if row['id'] == 'mistral-small-3.2-24b_full')
    pruned = next(row for row in data['text_encoder']['rows'] if row['id'] == 'mistral-small-3.2-24b_flux2_pruned')
    assert full['FP8_MIXED'] == 'unknown' and pruned['FP8_MIXED'] == 'verified'
    assert full['GGUF__label'] == 'Integration pending'
    assert pruned['GGUF__label'] == 'Configuration pending'
    assert '30 layers' in pruned['GGUF__reason'] and '40-layer' in full['GGUF__scope']
    # Display projection cannot remove existing production/export guards.
    assert model_support.text_encoder_support_level('mistral-small-3.2-24b', 'GGUF') == 'bad'
    assert not any('Family' in row['display_name'] for row in data['diffusion']['rows'])
    qwen_encoder = next(row for row in data['text_encoder']['rows'] if row['family'] == 'qwen3-8b')
    assert qwen_encoder['GGUF'] == 'caution' and 'Q5_K_M' in qwen_encoder['GGUF__reason']
    prototype = 'INT4_CONVROT_MIXED'
    assert prototype in data['diffusion']['formats']
    assert prototype not in data['text_encoder']['formats']
    for row in data['diffusion']['rows']:
        assert row[prototype] == ('caution' if row['arch'] == 'sdxl' else 'unknown')
        assert row[prototype + '__selectable'] == 'false'
        assert row[prototype + '__reason']
    sdxl = next(row for row in data['diffusion']['rows'] if row['arch'] == 'sdxl')
    assert 'nine successful' in sdxl[prototype + '__reason']
    assert 'ears, face' in sdxl[prototype + '__reason']
    assert prototype not in str(client.get('/api/config').json()['formats'])
    assert client.get('/api/support', headers={'X-Workbench-Token': ''}).status_code == 403
    assert client.get('/api/support', headers={'Origin': 'https://evil.example'}).status_code == 403


def test_hf_metadata_and_real_merge_through_api(tmp_path, monkeypatch):
    """The protected API selects one pinned variant and runs the real streaming writer."""
    source_a, source_b = tmp_path / 'a.safetensors', tmp_path / 'b.safetensors'
    save_file({'a.weight': torch.ones(2, 2)}, str(source_a))
    save_file({'b.weight': torch.zeros(3)}, str(source_b))
    revision = 'b' * 40
    files = {'variant/a.safetensors': source_a, 'variant/b.safetensors': source_b,
             'other/model.safetensors': source_a}
    class Api:
        def __init__(self, **kwargs):
            pass
        def model_info(self, repo_id, **kwargs):
            return SimpleNamespace(sha=revision, siblings=[SimpleNamespace(rfilename=name, size=path.stat().st_size) for name, path in files.items()])
        def list_repo_files(self, repo_id, **kwargs):
            assert kwargs['revision'] == revision
            return list(files)
    transfers = []
    def transfer(repo_id, filename, local_dir, **kwargs):
        assert kwargs['revision'] == revision
        transfers.append(filename)
        output = Path(local_dir) / filename
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(files[filename], output)
        return str(output)
    monkeypatch.setattr(hf_download, 'HfApi', Api)
    monkeypatch.setattr(hf_download, 'hf_hub_download', transfer)
    client = _client()
    destination = tmp_path / 'downloads'
    params = {'source': 'org/model', 'destination': str(destination)}
    response = client.post('/api/hf/inspect', json=params)
    assert response.status_code == 200, response.text
    assert len(response.json()['groups']) == 2
    assert transfers == [] and not destination.exists()
    assert client.post('/api/hf/inspect', json=params, headers={'Origin': 'https://evil.example'}).status_code == 403
    assert client.post('/api/hf/download', json={**params, 'token': 'not-accepted'}).status_code == 422
    assert client.post('/api/hf/download', json={**params, 'source': '../model'}).status_code == 400
    assert client.post('/api/hf/download', json={**params, 'subfolder': '../variant'}).status_code == 400
    assert client.post('/api/hf/download', json={**params, 'destination': str(source_a)}).status_code == 400
    assert client.post('/api/hf/download', json={'source': 'org/model'}).status_code == 400
    response = client.post('/api/hf/download', json=params)
    state = _wait(client, response.json()['id'])
    assert state['status'] == 'failed' and 'subfolder' in state['error']
    params |= {'revision': revision, 'subfolder': 'variant'}
    response = client.post('/api/hf/download', json=params)
    assert response.status_code == 202
    state = _wait(client, response.json()['id'])
    assert state['status'] == 'succeeded', state
    assert set(load_file(state['output'])) == {'a.weight', 'b.weight'}
    assert transfers == ['variant/a.safetensors', 'variant/b.safetensors']
    assert not (destination / '.hf_download_org_model').exists()
    original = Path(state['output']).read_bytes()
    response = client.post('/api/hf/download', json=params)
    assert _wait(client, response.json()['id'])['status'] == 'failed'
    assert Path(state['output']).read_bytes() == original


def test_hf_job_cancellation_and_secret_safe_errors(tmp_path, monkeypatch):
    """Cancellation reaches the downloader; transport failures never return credentials."""
    entered = threading.Event()
    def blocked(*args, cancel_event, **kwargs):
        entered.set()
        assert cancel_event.wait(5)
        raise RuntimeError('cancelled')
    monkeypatch.setattr(web_api, 'download_repo_as_single_safetensors', blocked)
    client = _client()
    params = {'source': 'org/model', 'destination': str(tmp_path), 'revision': 'a' * 40}
    response = client.post('/api/hf/download', json=params)
    assert response.status_code == 202 and entered.wait(5)
    assert response.json()['indeterminate']
    assert client.post('/api/tools', json={'operation': 'analyze', 'source': str(_source(tmp_path))}).status_code == 409
    client.post(f"/api/jobs/{response.json()['id']}/cancel", json={})
    assert _wait(client, response.json()['id'])['status'] == 'cancelled'
    monkeypatch.setenv('HF_TOKEN', 'fake-secret')
    def fail(*args, **kwargs):
        raise RuntimeError('failure fake-secret')
    monkeypatch.setattr(web_api, 'download_repo_as_single_safetensors', fail)
    response = client.post('/api/hf/download', json=params)
    state = _wait(client, response.json()['id'])
    assert state['status'] == 'failed' and 'fake-secret' not in str(state)
    class TransportFailure(Exception):
        response = SimpleNamespace(status_code=403)
    monkeypatch.setattr(web_api, 'inspect_repo', lambda *args: (_ for _ in ()).throw(TransportFailure('fake-secret signed-url')))
    response = client.post('/api/hf/inspect', json=params)
    assert response.status_code == 400
    assert 'access denied' in response.text and 'fake-secret' not in response.text


def test_component_tools_roundtrip_and_guards(tmp_path):
    """Export real components, compare a local reference, and reject bad inputs."""
    client = _client()
    source = tmp_path / 'embedded.safetensors'
    tensor = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    save_file({'first_stage_model.decoder.weight': tensor,
               'conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight': tensor.clone()}, str(source))
    root = tmp_path / 'models'
    params = {'operation': 'components', 'source': str(source), 'destination': str(root),
              'components': ['vae', 'clip_l']}
    response = client.post('/api/tools', json=params)
    assert response.status_code == 202, response.text
    state = _wait(client, response.json()['id'])
    assert state['status'] == 'succeeded', state
    assert len(state['outputs']) == 2
    assert torch.equal(load_file(root / 'vae' / 'embedded-vae.safetensors')['decoder.weight'], tensor)
    assert torch.equal(load_file(root / 'clip' / 'embedded-clip_l.safetensors')['text_model.embeddings.token_embedding.weight'], tensor)
    assert client.post('/api/tools', json={**params, 'components': []}).status_code == 400
    assert client.post('/api/tools', json={**params, 'components': ['invalid']}).status_code == 422
    assert client.post('/api/tools', json=params, headers={'X-Workbench-Token': ''}).status_code == 403
    assert client.post('/api/tools', json={**params, 'destination': str(source)}).status_code == 400
    response = client.post('/api/tools', json=params)
    assert _wait(client, response.json()['id'])['status'] == 'failed'
    save_file({'decoder.weight': tensor}, str(root / 'vae' / 'sdxlVAE.safetensors'))
    response = client.post('/api/tools', json={**params, 'operation': 'analyze'})
    state = _wait(client, response.json()['id'])
    assert state['status'] == 'succeeded', state
    assert state['result'][0]['status'] == 'matches local standard'
    assert state['result'][1]['status'] == 'no local reference'


@pytest.mark.parametrize('operation', ['pad_tokens', 'restore_5d'])
def test_real_repair_tools(tmp_path, operation):
    """Repair real GGUF shapes/sidecars without changing source bytes."""
    client = _client()
    source = tmp_path / 'source.gguf'
    writer = gguf.GGUFWriter(path=None, arch='lumina2')
    writer.add_tensor('x_pad_token', np.arange(4, dtype=np.float32))
    writer.add_tensor('ordinary.weight', np.ones((2, 2), dtype=np.float32))
    writer.write_header_to_file(path=str(source))
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()
    original = source.read_bytes()
    sidecar = tmp_path / 'source.gguf.5d.safetensors'
    save_file({'restored.weight': torch.ones(1, 2, 3, 4, 5)}, str(sidecar))
    params = {'operation': operation, 'source': str(source)}
    response = client.post('/api/tools', json=params)
    assert response.status_code == 202, response.text
    state = _wait(client, response.json()['id'])
    assert state['status'] == 'succeeded', state
    reader = gguf.GGUFReader(state['output'])
    tensors = {item.name: item for item in reader.tensors}
    if operation == 'pad_tokens':
        assert tensors['x_pad_token'].data.shape == (1, 4)
        np.testing.assert_array_equal(tensors['x_pad_token'].data, np.arange(4).reshape(1, 4))
    else:
        assert len(tensors['restored.weight'].shape) == 5
        assert tensors['restored.weight'].tensor_type == gguf.GGMLQuantizationType.F32
    np.testing.assert_array_equal(tensors['ordinary.weight'].data, np.ones((2, 2)))
    assert source.read_bytes() == original
    assert client.post('/api/tools', json=params).status_code == 400
    assert client.post('/api/tools', json={**params, 'destination': str(source), 'overwrite': True}).status_code == 400
    assert client.post('/api/tools', json={**params, 'destination': str(sidecar)}).status_code == 400
    if operation == 'restore_5d':
        assert client.post('/api/tools', json={**params, 'destination': str(tmp_path / 'new.gguf'), 'sidecar': str(tmp_path / 'missing.safetensors')}).status_code == 400


def test_diffusion_extraction_uses_existing_pipeline(tmp_path):
    """Checkpoint diffusion extraction writes into the loader's expected folder."""
    client = _client()
    source = _source(tmp_path)
    weights = {'model.diffusion_model.' + key: value.clone() for key, value in load_file(source).items()}
    weights['first_stage_model.decoder.weight'] = torch.ones(2, 2)
    save_file(weights, str(source))
    response = client.post('/api/tools', json={'operation': 'diffusion', 'source': str(source),
        'destination': str(tmp_path / 'models'), 'container': 'safetensors', 'format': 'F16'})
    assert response.status_code == 202, response.text
    state = _wait(client, response.json()['id'])
    assert state['status'] == 'succeeded', state
    assert 'diffusion_models' in state['output']
    exported = load_file(state['output'])
    assert exported
    assert not any(key.startswith(('model.diffusion_model.', 'first_stage_model.')) for key in exported)


def test_tools_share_concurrency_and_cancel(tmp_path, monkeypatch):
    """Tool workers participate in the same lock/cancellation contract as conversion."""
    client = _client()
    source = _source(tmp_path)
    entered = threading.Event()
    def blocked(*args, cancel_event, **kwargs):
        entered.set()
        assert cancel_event.wait(5)
        raise web_api.ConversionCancelled()
    monkeypatch.setattr(web_api, 'analyze_components', blocked)
    response = client.post('/api/tools', json={'operation': 'analyze', 'source': str(source)})
    assert response.status_code == 202
    assert entered.wait(5)
    assert response.json()['indeterminate']
    assert client.post('/api/jobs', json={'source': str(source), 'container': 'safetensors', 'format': 'F16'}).status_code == 409
    client.post(f"/api/jobs/{response.json()['id']}/cancel", json={})
    assert _wait(client, response.json()['id'])['status'] == 'cancelled'


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
    locations = client.get('/api/locations')
    assert locations.status_code == 200
    assert any(item['label'] == 'Home' for item in locations.json())
    assert any(item['group'] == 'Drives' for item in locations.json())
    assert all(Path(item['path']).is_absolute() for item in locations.json())
    assert client.get('/api/locations', headers={'X-Workbench-Token': ''}).status_code == 403
    assert client.get('/api/locations', headers={'Origin': 'https://evil.example'}).status_code == 403


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


@pytest.mark.parametrize('format_key', ['FP8_MIXED', 'F16_ST'])
def test_text_encoder_real_writer_preserves_prefix_and_embedding(tmp_path, format_key):
    """Native CLIPLoader needs original module paths and unpacked embedding tables."""
    source = tmp_path / 'encoder.safetensors'
    embedding = torch.randn(32, 128, dtype=torch.bfloat16)
    save_file({'model.embed_tokens.weight': embedding,
               'model.layers.0.self_attn.q_proj.weight': torch.randn(128, 128, dtype=torch.bfloat16),
               'model.layers.0.input_layernorm.weight': torch.ones(128, dtype=torch.bfloat16)}, str(source))
    client = _client()
    params = {'model_kind': 'text_encoder', 'source': str(source),
              'destination': str(tmp_path), 'container': 'safetensors', 'format': format_key}
    inspection = client.post('/api/inspect', json=params)
    assert inspection.status_code == 200, inspection.text
    assert inspection.json()['estimated_bytes'] > 0
    assert inspection.json()['architecture'] == 'Unknown family'
    started = client.post('/api/jobs', json=params)
    assert started.status_code == 202, started.text
    state = _wait(client, started.json()['id'])
    assert state['status'] == 'succeeded', state
    assert state['model_kind'] == 'text_encoder' and not state['indeterminate']
    with safe_open(state['output'], framework='pt') as result:
        assert 'model.layers.0.self_attn.q_proj.weight' in result.keys()
        preserved = result.get_tensor('model.embed_tokens.weight')
        assert preserved.shape == embedding.shape
        assert torch.equal(preserved.float(), embedding.float())
        if format_key == 'F16_ST':
            assert all(result.get_tensor(key).dtype == torch.float16 for key in result.keys())
    assert client.post('/api/jobs', json=params).status_code == 400
    assert client.post('/api/jobs', json={**params, 'destination': str(source)}).status_code == 400
    assert client.post('/api/jobs', json={**params, 'container': 'gguf'}).status_code == 400
    assert client.post('/api/jobs', json={**params, 'executable': 'diffusion.exe'}).status_code == 400


def test_text_encoder_gguf_dispatch_and_offline_guards(tmp_path, monkeypatch):
    """GGUF uses the text backend; reject incompatible families before external work."""
    client = _client()
    source = _source(tmp_path)
    params = {'model_kind': 'text_encoder', 'source': str(source),
              'destination': str(tmp_path / 'encoder-{ftype}'), 'container': 'gguf', 'format': 'F16'}
    assert client.post('/api/jobs', json=params).status_code == 400  # Unknown family without override.
    monkeypatch.setattr(web_api, 'detect_text_encoder_family', lambda state: 'clip-l')
    for route in ('inspect', 'jobs'):
        rejected = client.post(f'/api/{route}', json=params)
        assert rejected.status_code == 400 and 'safetensors' in rejected.json()['detail']
    # Detectable overrides must agree with the checkpoint, not change its identity.
    assert client.post('/api/jobs', json={**params, 'base_repo_id': 'Qwen/Qwen3-8B'}).status_code == 400
    monkeypatch.setattr(web_api, 'detect_text_encoder_family', lambda state: 'qwen3-8b')
    inspected = client.post('/api/inspect', json=params)
    assert inspected.status_code == 200
    assert inspected.json()['estimated_bytes'] is None
    entered = threading.Event()

    def blocked(weights, repo, destination, format_key, *, on_log, cancel_event, overwrite):
        assert weights == str(source) and repo == '' and format_key == 'F16'
        assert destination.endswith('encoder-F16.gguf') and not overwrite
        on_log('Text encoder conversion started')
        entered.set()
        assert cancel_event.wait(5)
        raise RuntimeError('cancelled')

    monkeypatch.setattr(web_api, 'convert_text_encoder_any', blocked)
    started = client.post('/api/jobs', json=params)
    assert started.status_code == 202, started.text
    job_id = started.json()['id']
    assert entered.wait(5)
    assert client.get(f'/api/jobs/{job_id}').json()['indeterminate']
    assert client.post('/api/jobs', json=params).status_code == 409
    client.post(f'/api/jobs/{job_id}/cancel', json={})
    assert _wait(client, job_id)['status'] == 'cancelled'
    assert not (tmp_path / 'encoder-F16.gguf').exists()


def test_text_encoder_invalid_checkpoint_returns_input_error(tmp_path):
    """Malformed headers do not escape preflight as an internal server failure."""
    source = tmp_path / 'broken.safetensors'
    source.write_bytes(b'not a checkpoint')
    params = {'model_kind': 'text_encoder', 'source': str(source),
              'container': 'safetensors', 'format': 'F16_ST'}
    client = _client()
    assert client.post('/api/jobs', json=params).status_code == 400
    assert client.post('/api/inspect', json=params).status_code == 400
