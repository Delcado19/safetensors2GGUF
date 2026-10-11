"""Credential precedence, persistence and write-only API boundaries."""

import pytest
from fastapi.testclient import TestClient

import civitai_credentials as credentials
import web_api


@pytest.fixture
def store(monkeypatch):
    """Provide an isolated fake vault without changing real stored credentials."""
    for name in ('CIVITAI_API_KEY', 'CIVITAI_TOKEN', 'CODEX_CIVITAI_API_KEY'):
        monkeypatch.delenv(name, raising=False)
    vault = ['vault-secret']

    def fake_vault(value=None):
        if value is not None:
            vault[0] = value
        return vault[0]

    monkeypatch.setattr(credentials, '_vault_key', fake_vault)
    return credentials.CivitaiCredentials(), vault


def test_precedence_clear_and_explicit_persistence(store, monkeypatch):
    """Session input wins; clearing restores fallback and persistence is explicit."""
    manager, vault = store
    assert manager.resolve() == ('vault-secret', 'windows_vault')
    monkeypatch.setenv('CIVITAI_API_KEY', 'env-secret')
    assert manager.resolve() == ('env-secret', 'environment')
    assert manager.set('manual-secret')['source'] == 'session'
    assert vault[0] == 'vault-secret'
    assert 'manual-secret' not in str(manager.status())
    manager.set('')
    assert manager.resolve() == ('env-secret', 'environment')
    manager.set('saved-secret', persist=True)
    assert vault[0] == 'saved-secret'
    assert manager.resolve() == ('saved-secret', 'session')


def test_invalid_keys_and_vault_failure_preserve_session(store, monkeypatch):
    """Invalid input or failed persistence leaves the usable session key intact."""
    manager, _ = store
    manager.set('previous')
    for key in ('bad key', 'bad\nkey', 'x' * 1025):
        with pytest.raises(ValueError):
            manager.set(key)
    def failed(value=None):
        raise OSError('write failed')
    monkeypatch.setattr(credentials, '_vault_key', failed)
    with pytest.raises(OSError):
        manager.set('replacement', persist=True)
    assert manager.resolve() == ('previous', 'session')


def test_api_never_echoes_keys_and_requires_session(store):
    """Protect credential operations and prevent secret reflection in API errors."""
    client = TestClient(web_api.create_app(), base_url='http://localhost')
    path = '/api/civitai/credentials'
    assert client.get(path).status_code == 403
    assert client.post(path, json={'key': 'secret'}).status_code == 403
    client.headers['X-Workbench-Token'] = client.get('/api/config').json()['token']
    result = client.post(path, json={'key': 'manual-secret'})
    assert result.status_code == 200
    assert result.json()['source'] == 'session'
    assert result.headers['cache-control'] == 'no-store'
    assert 'manual-secret' not in client.get(path).text
    for body in ({'key': ['manual-secret']}, {'key': 'manual-secret', 'unknown': 1},
                 {'key': 'manual-secret', 'persist': 'invalid'}):
        result = client.post(path, json=body)
        assert result.status_code == 400
        assert 'manual-secret' not in result.text
    assert client.post(path, json={'key': ''}).json()['source'] == 'windows_vault'
    assert client.post(path, json={'key': 'secret'}, headers={'Origin': 'https://example.com'}).status_code == 403
