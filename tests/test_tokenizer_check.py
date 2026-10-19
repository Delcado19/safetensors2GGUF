"""Tokenizer repair preserves weights, rejects malformed inputs and remains opt-in."""
import base64
import json
import threading
import shutil

import pytest
import torch
from safetensors.torch import load_file, save_file

import tokenizer_check as check
from convert import ConversionCancelled
from tests.test_web_api import _client, _wait


@pytest.fixture
def model(tmp_path, monkeypatch):
    monkeypatch.setattr(check, '_context', lambda _: 258)
    source = tmp_path / 'encoder.safetensors'
    save_file({'weight': torch.arange(32, dtype=torch.bfloat16)}, source, metadata={'source': 'fixture'})
    obj = {'config': {'default_num_special_tokens': 2, 'default_vocab_size': 258},
           'special_tokens': [{'rank': 0, 'token_str': '<unk>'}, {'rank': 1, 'token_str': '<s>'}],
           'vocab': [{'rank': value, 'token_bytes': base64.b64encode(bytes([value])).decode()}
                     for value in range(256)]}
    tokenizer = tmp_path / 'tekken.json'
    tokenizer.write_text(json.dumps(obj))
    return source, tokenizer


def test_missing_repair_and_corrupt_replacement_preserve_weights(model, tmp_path):
    source, tokenizer = model
    assert check.check_tokenizer(source)['status'] == 'missing'
    original = source.read_bytes()
    output = tmp_path / 'fixed.safetensors'
    check.repair_tokenizer(source, output, tokenizer)
    assert check.check_tokenizer(output)['status'] == 'valid'
    assert source.read_bytes() == original
    assert torch.equal(load_file(output)['weight'], load_file(source)['weight'])
    corrupted = tmp_path / 'corrupt.safetensors'
    save_file({'weight': load_file(source)['weight'], 'tekken_model': torch.tensor([0], dtype=torch.uint8)}, corrupted)
    assert check.check_tokenizer(corrupted)['status'] == 'invalid'
    repaired = tmp_path / 'repaired.safetensors'
    check.repair_tokenizer(corrupted, repaired, tokenizer)
    assert check.check_tokenizer(repaired)['status'] == 'valid'


@pytest.mark.parametrize('fault', ['vocab_size', 'duplicate', 'invalid_bytes', 'missing_byte'])
def test_invalid_tokenizer_leaves_existing_files(model, tmp_path, fault):
    source, tokenizer = model
    obj = json.loads(tokenizer.read_text())
    if fault == 'vocab_size':
        obj['config']['default_vocab_size'] = 259
    elif fault == 'duplicate':
        obj['vocab'][0]['rank'] = 1
    elif fault == 'invalid_bytes':
        obj['vocab'][0]['token_bytes'] = '!'
    else:
        obj['vocab'][0]['token_bytes'] = base64.b64encode(b'ab').decode()
    tokenizer.write_text(json.dumps(obj))
    output = tmp_path / 'out.safetensors'
    with pytest.raises(ValueError):
        check.repair_tokenizer(source, output, tokenizer)
    assert not output.exists()
    assert list(tmp_path.glob('.*.safetensors')) == []


def test_repair_guards_alias_existing_and_cancellation(model, tmp_path):
    source, tokenizer = model
    original = source.read_bytes()
    with pytest.raises(ValueError):
        check.repair_tokenizer(source, source, tokenizer)
    output = tmp_path / 'out.safetensors'
    output.write_bytes(b'existing')
    with pytest.raises(OSError):
        check.repair_tokenizer(source, output, tokenizer)
    assert output.read_bytes() == b'existing'
    event = threading.Event()
    event.set()
    with pytest.raises(ConversionCancelled):
        check.repair_tokenizer(source, tmp_path / 'cancelled.safetensors', tokenizer, cancel_event=event)
    assert not (tmp_path / 'cancelled.safetensors').exists()
    assert source.read_bytes() == original


def test_huggingface_bytelevel_conversion_covers_every_byte(tmp_path):
    # Use the shipped real tokenizer to check non-ASCII ByteLevel mappings too.
    from pathlib import Path
    path = Path(__file__).parents[1] / 'text_encoder_configs/ernie-image-pe/tokenizer.json'
    obj = check._validate(check._tokenizer_bytes(path, 131072), 131072)
    assert len(obj['special_tokens']) == 1000
    assert len(obj['vocab']) == 130072
    assert next(row['token_str'] for row in obj['special_tokens'] if row['rank'] == 34) == '[THINK]'


def test_api_offers_repair_and_reports_new_copy(model):
    source, tokenizer = model
    client = _client()
    response = client.post('/api/inspect', json={'source': str(source), 'model_kind': 'text_encoder',
                          'container': 'safetensors', 'format': 'F16_ST'})
    assert response.status_code == 200, response.text
    assert response.json()['tokenizer']['status'] == 'missing'
    request = {'operation': 'tokenizer_repair', 'source': str(source), 'tokenizer': str(tokenizer)}
    assert client.post('/api/tools', json=request | {'overwrite': True}).status_code == 400
    assert client.post('/api/tools', json=request | {'tokenizer': ''}).status_code == 400
    response = client.post('/api/tools', json=request)
    assert response.status_code == 202, response.text
    result = _wait(client, response.json()['id'])
    assert result['status'] == 'succeeded', result
    assert result['tokenizer']['status'] == 'valid'
    assert result['output'].endswith('-tokenizer-fixed.safetensors')


def test_unknown_family_has_no_speculative_fix(tmp_path):
    source = tmp_path / 'other.safetensors'
    save_file({'weight': torch.ones(2)}, source)
    assert check.check_tokenizer(source) == {'status': 'not_checked', 'repairable': False}


def test_download_success_reports_missing_tokenizer_without_repairing(model, tmp_path, monkeypatch):
    import web_api
    source, _ = model
    output = tmp_path / 'download.safetensors'
    def download(*args, **kwargs):
        shutil.copyfile(source, output)
        return str(output)
    monkeypatch.setattr(web_api, 'download_repo_as_single_safetensors', download)
    client = _client()
    response = client.post('/api/hf/download', json={'source': 'org/encoder',
                           'destination': str(tmp_path), 'revision': 'a' * 40})
    assert response.status_code == 202, response.text
    result = _wait(client, response.json()['id'])
    assert result['status'] == 'succeeded'
    assert result['tokenizer']['status'] == 'missing'
    assert output.read_bytes() == source.read_bytes()
    assert not list(tmp_path.glob('*-tokenizer-fixed.safetensors'))
