"""Tokenizer repair preserves weights, rejects malformed inputs and remains opt-in."""
import base64
import json
import threading
import shutil
from pathlib import Path

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


def test_sentencepiece_repair_preserves_weights(tmp_path, monkeypatch):
    import io
    import sentencepiece
    writer = io.BytesIO()
    sentencepiece.SentencePieceTrainer.train(
        sentence_iterator=iter(['a small tokenizer fixture', 'another sentence for the tokenizer']),
        model_writer=writer, vocab_size=32, hard_vocab_limit=False, minloglevel=2)
    tokenizer = tmp_path / 'spiece.model'
    tokenizer.write_bytes(writer.getvalue())
    monkeypatch.setattr(check, '_blob_context', lambda _: ('spiece_model', 64))
    source = tmp_path / 'umt5.safetensors'
    save_file({'shared.weight': torch.ones(64, 4)}, source)
    original = source.read_bytes()
    assert check.check_tokenizer(source)['status'] == 'missing'
    output = tmp_path / 'fixed.safetensors'
    check.repair_tokenizer(source, output, tokenizer)
    assert check.check_tokenizer(output)['status'] == 'valid'
    assert source.read_bytes() == original
    assert torch.equal(load_file(source)['shared.weight'], load_file(output)['shared.weight'])
    monkeypatch.setattr(check, '_blob_context', lambda _: ('spiece_model', 2))
    assert check.check_tokenizer(output)['status'] == 'invalid'
    with pytest.raises(ValueError, match='vocabulary'):
        check.repair_tokenizer(source, tmp_path / 'bad.safetensors', tokenizer)


@pytest.fixture
def qwen_resources(tmp_path, monkeypatch):
    # Build native ComfyUI resources from the complete owned vocabulary, without network access.
    configs = Path(__file__).parents[1] / 'text_encoder_configs/qwen3-4b'
    obj = json.loads((configs / 'tokenizer.json').read_text(encoding='utf8'))
    reference = tmp_path / 'healthy'
    reference.mkdir()
    (reference / 'vocab.json').write_text(json.dumps(obj['model']['vocab']), encoding='utf8')
    merges = obj['model']['merges']
    (reference / 'merges.txt').write_text('#version: 0.2\n' + '\n'.join(
        ' '.join(row) if isinstance(row, list) else row for row in merges), encoding='utf8')
    shutil.copyfile(configs / 'tokenizer_config.json', reference / 'tokenizer_config.json')
    target = tmp_path / 'runtime/qwen25_tokenizer'
    source = tmp_path / 'qwen.safetensors'
    save_file({'weight': torch.ones(2)}, source)
    monkeypatch.setattr(check, '_blob_context', lambda _: None)
    monkeypatch.setattr(check, '_runtime_context', lambda _: (target, 151936))
    return source, reference, target


def test_runtime_check_repair_and_backup(qwen_resources):
    source, reference, target = qwen_resources
    original = source.read_bytes()
    assert check.check_tokenizer(source)['status'] == 'missing'
    check.repair_runtime_tokenizer(source, reference)
    assert check.check_tokenizer(source)['status'] == 'valid'
    (target / 'vocab.json').write_text('{}')
    assert check.check_tokenizer(source)['status'] == 'invalid'
    check.repair_runtime_tokenizer(source, reference)
    assert check.check_tokenizer(source)['status'] == 'valid'
    backups = list(target.parent.glob('qwen25_tokenizer.backup-*'))
    assert len(backups) == 1 and (backups[0] / 'vocab.json').read_text() == '{}'
    assert source.read_bytes() == original
    with pytest.raises(ValueError, match='separate'):
        check.repair_runtime_tokenizer(source, target)


def test_runtime_repair_cancellation_and_failed_publish_roll_back(qwen_resources, monkeypatch):
    source, reference, target = qwen_resources
    check.repair_runtime_tokenizer(source, reference)
    before = {path.name: path.read_bytes() for path in target.iterdir()}
    event = threading.Event()
    event.set()
    with pytest.raises(ConversionCancelled):
        check.repair_runtime_tokenizer(source, reference, cancel_event=event)
    rename = Path.rename
    def fail_publish(path, destination):
        if path.name == 'resources':
            raise OSError('publication failed')
        return rename(path, destination)
    monkeypatch.setattr(Path, 'rename', fail_publish)
    with pytest.raises(OSError, match='publication'):
        check.repair_runtime_tokenizer(source, reference)
    assert {path.name: path.read_bytes() for path in target.iterdir()} == before
    assert not list(target.parent.glob('qwen25_tokenizer.backup-*'))


def test_runtime_invalid_reference_does_not_change_assets(qwen_resources):
    source, reference, target = qwen_resources
    (reference / 'tokenizer_config.json').write_text('{broken')
    with pytest.raises(ValueError):
        check.repair_runtime_tokenizer(source, reference)
    assert not target.exists()


def test_api_check_existing_and_runtime_repair(qwen_resources):
    source, reference, target = qwen_resources
    client = _client()
    request = {'operation': 'tokenizer_check', 'source': str(source)}
    assert client.post('/api/tools', json=request | {'overwrite': True}).status_code == 400
    response = client.post('/api/tools', json=request)
    assert response.status_code == 202
    result = _wait(client, response.json()['id'])
    assert result['status'] == 'succeeded' and result['tokenizer']['status'] == 'missing'
    response = client.post('/api/tools', json={'operation': 'tokenizer_repair',
                           'source': str(source), 'tokenizer': str(reference)})
    assert response.status_code == 202, response.text
    result = _wait(client, response.json()['id'])
    assert result['status'] == 'succeeded' and result['tokenizer']['status'] == 'valid'
    assert target.is_dir() and not list(source.parent.glob('*-tokenizer-fixed.safetensors'))
