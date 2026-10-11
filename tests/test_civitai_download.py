"""Civitai resume, fallback, cancellation and transactional output contracts."""

import hashlib
import io
import json
import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import civitai_download as download
from convert import ConversionCancelled
import web_api


DATA = b'0123456789abcdef'
DIGEST = hashlib.sha256(DATA).hexdigest()


class Response(io.BytesIO):
    """In-memory HTTP body with the headers required by range validation."""
    def __init__(self, data=DATA, status=200, headers=None):
        super().__init__(data)
        self.status = status
        self.headers = headers or {'Content-Length': str(len(data))}


@pytest.fixture
def model(monkeypatch):
    """Pin tiny metadata and chunk sizes to exercise transfer failures cheaply."""
    file = {'id': 22, 'name': 'model.safetensors', 'bytes': len(DATA), 'sha256': DIGEST}
    monkeypatch.setattr(download, 'inspect_version', lambda *args: {'version': 11, 'files': [file]})
    monkeypatch.setattr(download, 'CHUNK', 3)
    monkeypatch.setattr('time.sleep', lambda seconds: None)
    return file


def partial_path(folder):
    """Locate the staging file bound to the synthetic model's identity and hash."""
    return folder / f'.civitai-11-22-{DIGEST}.part'


@pytest.mark.parametrize('source,expected', [
    ('11', (11, None)),
    ('https://civitai.red/api/download/models/11?fileId=22', (11, 22)),
    ('https://civitai.com/models/9/model?modelVersionId=11', (11, None)),
])
def test_source_identity(source, expected):
    """Supported links retain an exact version and optional file selection."""
    assert download.parse_source(source) == expected


@pytest.mark.parametrize('source', ['https://evil.test/api/download/models/11',
    'https://civitai.red/api/download/models/11?token=secret',
    'https://civitai.red/models/11', 'https://civitai.red/api/download/models/11?fileId=22&fileId=23',
    'https://user@civitai.red/api/download/models/11', '../11'])
def test_source_trust_boundary(source):
    """Reject arbitrary hosts, ambiguous identities and credentials in links."""
    with pytest.raises(download.CivitaiError):
        download.parse_source(source)


def test_inspect_accepts_current_diffusion_model_label(monkeypatch):
    """Current Krea API labels remain usable without admitting unsafe files."""
    metadata = {'files': [
        {'id': 3160899, 'name': 'solstice_v2_3160899.safetensors',
         'type': 'Diffusion Model', 'hashes': {'SHA256': DIGEST}, 'sizeKB': 1.5},
        {'id': 1, 'name': '../escape.safetensors', 'type': 'Model', 'hashes': {'SHA256': DIGEST}},
        {'id': 2, 'name': 'training.safetensors', 'type': 'Training Data', 'hashes': {'SHA256': DIGEST}},
    ]}
    monkeypatch.setattr(download, '_open', lambda *args, **kwargs: Response(json.dumps(metadata).encode()))
    result = download.inspect_version('https://civitai.red/api/download/models/3276318?fileId=3160899')
    assert result['selected_file'] == 3160899
    assert result['files'] == [{'id': 3160899, 'name': 'solstice_v2_3160899.safetensors',
                               'sha256': DIGEST, 'bytes': 1536}]


def test_interrupted_transfer_refreshes_link_and_never_appends_200(tmp_path, monkeypatch, model):
    """Retry across domains while preserving bytes when a server ignores Range."""
    calls = []
    class Broken(Response):
        def read(self, count=-1):
            if self.tell() >= 3:
                raise ConnectionError('signed-url secret')
            return super().read(count)
    def open_response(url, key, offset, cancel_event):
        calls.append((url, offset))
        if len(calls) == 1:
            return Broken()
        if len(calls) == 2:
            return Response()  # .com ignores Range: do not append duplicate bytes.
        return Response(DATA[3:], 206, {'Content-Range': f'bytes 3-15/{len(DATA)}'})
    monkeypatch.setattr(download, '_open', open_response)
    logs = []
    target = download.download_model('11', 22, tmp_path, 'secret', on_log=logs.append)
    assert (tmp_path / 'model.safetensors').read_bytes() == DATA
    assert target == str(tmp_path / 'model.safetensors')
    assert [offset for _, offset in calls] == [0, 3, 3]
    assert ['civitai.red' in url for url, _ in calls] == [True, False, True]
    assert not partial_path(tmp_path).exists()
    assert 'secret' not in str(logs)


def test_resume_after_restart_and_wrong_range_preserves_bytes(tmp_path, monkeypatch, model):
    """Persistent partials survive invalid responses and resume on a later run."""
    partial = partial_path(tmp_path)
    partial.write_bytes(DATA[:5])
    offsets = []
    def wrong(url, key, offset, event):
        offsets.append(offset)
        return Response(DATA, 206, {'Content-Range': 'bytes 0-15/16'})
    monkeypatch.setattr(download, '_open', wrong)
    with pytest.raises(download.CivitaiError, match='four attempts'):
        download.download_model('11', 22, tmp_path, 'secret')
    assert offsets == [5] * 4
    assert partial.read_bytes() == DATA[:5]
    monkeypatch.setattr(download, '_open', lambda *args: Response(DATA[5:], 206, {'Content-Range': 'bytes 5-15/16'}))
    download.download_model('11', 22, tmp_path, 'secret')
    assert (tmp_path / 'model.safetensors').read_bytes() == DATA


def test_cancel_preserves_partial_then_resume(tmp_path, monkeypatch, model):
    """Cancellation retains transferred bytes without publishing an incomplete model."""
    event = threading.Event()
    monkeypatch.setattr(download, '_open', lambda *args: Response())
    with pytest.raises(ConversionCancelled):
        download.download_model('11', 22, tmp_path, 'secret', cancel_event=event,
                                on_progress=lambda *args: event.set())
    assert partial_path(tmp_path).read_bytes() == DATA[:3]
    assert not (tmp_path / 'model.safetensors').exists()


def test_checksum_failure_preserves_existing_output(tmp_path, monkeypatch, model):
    """Failed integrity checks retain both the old output and diagnostic partial."""
    target = tmp_path / 'model.safetensors'
    target.write_bytes(b'original')
    monkeypatch.setattr(download, '_open', lambda *args: Response(b'bad-data'))
    with pytest.raises(download.CivitaiError, match='SHA256 mismatch'):
        download.download_model('11', 22, tmp_path, 'secret', overwrite=True)
    assert target.read_bytes() == b'original'
    assert partial_path(tmp_path).read_bytes() == b'bad-data'


def test_completed_partial_verifies_without_network(tmp_path, monkeypatch, model):
    """A completed transfer interrupted before publication requires no redownload."""
    partial_path(tmp_path).write_bytes(DATA)
    monkeypatch.setattr(download, '_open', lambda *args: pytest.fail('Completed partial must not redownload'))
    download.download_model('11', 22, tmp_path, 'secret')
    assert (tmp_path / 'model.safetensors').read_bytes() == DATA


def test_auth_denial_is_not_retried(tmp_path, monkeypatch, model):
    """Account access errors stop immediately rather than wasting retry attempts."""
    calls = []
    def denied(*args):
        calls.append(args)
        raise download.CivitaiError('Civitai access denied.')
    monkeypatch.setattr(download, '_open', denied)
    with pytest.raises(download.CivitaiError, match='access denied'):
        download.download_model('11', 22, tmp_path, 'secret')
    assert len(calls) == 1


@pytest.mark.parametrize('host', ['b2.civitai.com', *download.STORAGE_HOSTS])
def test_redirect_does_not_forward_authorization(monkeypatch, host):
    """Trusted storage receives the requested byte offset but no API credential."""
    requests = []
    import urllib.error
    def open_response(request, timeout):
        requests.append(request)
        if len(requests) == 1:
            raise urllib.error.HTTPError(request.full_url, 302, 'redirect',
                {'Location': f'https://{host}/file?signature=private'}, None)
        return Response()
    monkeypatch.setattr(download.urllib.request, 'build_opener', lambda *args: SimpleNamespace(open=open_response))
    download._open('https://civitai.red/api/download/models/11', 'secret', 5).close()
    assert requests[0].get_header('Authorization') == 'Bearer secret'
    assert requests[1].get_header('Authorization') is None
    assert requests[1].get_header('Range') == 'bytes=5-'


@pytest.mark.parametrize('host', ['evil.test', 'another-account.r2.cloudflarestorage.com'])
def test_untrusted_storage_redirect_is_refused(monkeypatch, host):
    """Reject foreign storage accounts before making a redirected request."""
    import urllib.error
    calls = []
    def redirect(request, timeout):
        calls.append(request)
        raise urllib.error.HTTPError(request.full_url, 302, 'redirect',
            {'Location': f'https://{host}/file'}, None)
    monkeypatch.setattr(download.urllib.request, 'build_opener', lambda *args: SimpleNamespace(open=redirect))
    with pytest.raises(download.CivitaiError, match='unsupported download host'):
        download._open('https://civitai.red/api/download/models/11', 'secret', 5)
    assert len(calls) == 1


def test_api_dispatch_and_secret_safe_failure(tmp_path, monkeypatch):
    """Canonical job identities and errors never expose the resolved API key."""
    client = TestClient(web_api.create_app(), base_url='http://localhost')
    path = '/api/civitai/download'
    assert client.post(path, json={'source': '11'}).status_code == 403
    client.headers['X-Workbench-Token'] = client.get('/api/config').json()['token']
    client.app.state.civitai_credentials.set('api-key-test-value')
    params = {'source': 'https://civitai.red/api/download/models/11?fileId=22', 'destination': str(tmp_path)}
    monkeypatch.setattr(web_api, 'download_model', lambda *args, **kwargs: str(tmp_path / 'model.safetensors'))
    # Run synchronously so assertions inspect a deterministic completed job.
    class InlineThread:
        def __init__(self, target, daemon):
            self.target = target

        def start(self):
            self.target()
    monkeypatch.setattr(web_api, 'threading', SimpleNamespace(
        Thread=InlineThread, Event=threading.Event, Lock=threading.Lock))
    result = client.post(path, json=params)
    assert result.status_code == 202
    assert result.json()['status'] == 'succeeded'
    assert result.json()['source'] == '11'
    assert result.json()['operation'] == 'civitai_download'
    assert 'api-key-test-value' not in result.text
    assert client.post(path, json=params | {'file_id': 23}).status_code == 400
    def failure(*args, **kwargs):
        raise OSError('api-key-test-value signed-url')
    monkeypatch.setattr(web_api, 'download_model', failure)
    result = client.post(path, json=params)
    assert result.json()['status'] == 'failed'
    assert 'signed-url' not in result.text and 'api-key-test-value' not in result.text
