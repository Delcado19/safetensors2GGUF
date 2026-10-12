"""Pinned Civitai file downloads with persistent partials and verified publication."""

import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import shutil
import urllib.error
import urllib.parse
import urllib.request

from convert import ConversionCancelled

HOSTS = ('civitai.red', 'civitai.com')
# Observed Solstice download redirect; do not trust arbitrary R2 account hosts.
STORAGE_HOSTS = {'civitai-delivery-worker-prod.5ac0637cfd0766c97916cefa3764fbdf.r2.cloudflarestorage.com'}
CHUNK = 1024 * 1024


class CivitaiError(ValueError):
    """Safe user-facing errors containing no transport URLs or credentials."""


class _Retry(Exception):
    pass


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args):
        return None


def parse_source(source):
    """Accept a version ID or trusted model/download link, never arbitrary URLs."""
    source = source.strip()
    if re.fullmatch(r'[1-9]\d{0,11}', source):
        return int(source), None
    parsed = urllib.parse.urlsplit(source)
    if (parsed.scheme != 'https' or parsed.netloc not in HOSTS or parsed.fragment):
        raise CivitaiError('Enter a Civitai version ID or HTTPS model/download link.')
    query = urllib.parse.parse_qs(parsed.query)
    if set(query) - {'fileId', 'modelVersionId'} or any(len(values) != 1 for values in query.values()):
        raise CivitaiError('Use a Civitai link without tokens or additional parameters.')
    match = re.fullmatch(r'/api/download/models/([1-9]\d{0,11})/?', parsed.path)
    version = match[1] if match else query.get('modelVersionId', [''])[0]
    if not match and not re.fullmatch(r'/models/[1-9]\d{0,11}(?:/[^/]*)?/?', parsed.path):
        raise CivitaiError('Use a model link with modelVersionId or a direct download link.')
    file_id = query.get('fileId', [None])[0]
    if not re.fullmatch(r'[1-9]\d{0,11}', version) or (file_id and not re.fullmatch(r'[1-9]\d{0,11}', file_id)):
        raise CivitaiError('Choose a specific model version in the Civitai link.')
    return int(version), int(file_id) if file_id else None


def _cancel(event):
    if event is not None and event.is_set():
        raise ConversionCancelled('cancelled')


def _open(url, key, offset=None, cancel_event=None):
    """Resolve fresh signed URLs without forwarding the key to storage hosts."""
    opener = urllib.request.build_opener(_NoRedirect)
    for _ in range(8):
        _cancel(cancel_event)
        parsed = urllib.parse.urlsplit(url)
        # Only Civitai's HTTPS storage domains may receive redirected requests.
        if (parsed.scheme != 'https' or parsed.username or parsed.password or
                not parsed.hostname or not (parsed.hostname in HOSTS or parsed.hostname in STORAGE_HOSTS or
                                            parsed.hostname.endswith('.civitai.com'))):
            raise CivitaiError('Civitai returned an unsupported download host.')
        headers = {'User-Agent': 'safetensors2GGUF/0.1', 'Accept-Encoding': 'identity'}
        if key and parsed.hostname in HOSTS:
            headers['Authorization'] = 'Bearer ' + key
        if offset is not None:
            headers['Range'] = f'bytes={offset}-'
        try:
            response = opener.open(urllib.request.Request(url, headers=headers), timeout=20)
        except urllib.error.HTTPError as exc:
            if exc.code in (301, 302, 303, 307, 308):
                location = exc.headers.get('Location')
                exc.close()
                if not location:
                    raise CivitaiError('Civitai returned an invalid redirect.') from None
                url = urllib.parse.urljoin(url, location)
                continue
            code = exc.code
            exc.close()
            if code in (408, 429) or 500 <= code < 600:
                raise _Retry() from None
            if code in (401, 403):
                if parsed.hostname not in HOSTS:
                    # Signed storage links can expire: resolve a fresh link on retry.
                    raise _Retry() from None
                raise CivitaiError('Civitai access denied. Set a valid API key and check model access.') from None
            if code == 404:
                raise CivitaiError('Civitai version or file was not found.') from None
            raise CivitaiError(f'Civitai request failed (HTTP {code}).') from None
        return response
    raise CivitaiError('Too many Civitai redirects.')


def inspect_version(source, key='', cancel_event=None):
    """Read version metadata and expose only safe, hash-pinned model file choices."""
    version, requested_file = parse_source(source)
    for host in HOSTS:
        try:
            with _open(f'https://{host}/api/v1/model-versions/{version}', key, cancel_event=cancel_event) as response:
                data = response.read(2 * 1024 * 1024 + 1)
            if len(data) > 2 * 1024 * 1024:
                raise CivitaiError('Civitai metadata is too large.')
            metadata = json.loads(data)
            break
        except (_Retry, urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException):
            continue
        except CivitaiError:
            raise
        except (ValueError, UnicodeError):
            raise CivitaiError('Could not read Civitai model metadata.') from None
    else:
        raise CivitaiError('Could not contact Civitai. Retry later.')
    files = []
    for item in metadata.get('files', []):
        name, digest = item.get('name', ''), item.get('hashes', {}).get('SHA256', '').lower()
        # Windows-safe names and hashes are trust boundaries, not uploader promises.
        # Civitai labels diffusion weights as Model, Diffusion Model, or UNet.
        if (item.get('type') not in {'Model', 'Diffusion Model', 'UNet'} or Path(name).suffix.lower() not in {'.safetensors', '.gguf'} or
                not name or len(name) > 200 or name.rstrip(' .') != name or
                any(ord(char) < 32 or char in '<>:"/\\|?*' for char in name) or
                name.split('.')[0].upper() in {'CON', 'PRN', 'AUX', 'NUL', *{f'COM{i}' for i in range(1, 10)}, *{f'LPT{i}' for i in range(1, 10)}} or
                not re.fullmatch(r'[0-9a-f]{64}', digest)):
            continue
        file_id = item.get('id')
        if not isinstance(file_id, int) or file_id <= 0:
            continue
        files.append({'id': file_id, 'name': name, 'sha256': digest,
                      'bytes': int(item['sizeKB'] * 1024) if isinstance(item.get('sizeKB'), (int, float)) else None})
    if not files or (requested_file and not any(f['id'] == requested_file for f in files)):
        raise CivitaiError('No matching safetensors/GGUF model file with SHA256 was found.')
    return {'version': version, 'selected_file': requested_file, 'files': files}


def download_model(source, file_id, destination, key, overwrite=False,
                   on_log=None, on_progress=None, cancel_event=None):
    """Resume the same hash-pinned file, retry transient faults, verify before publish.

    Partial filenames encode version, file and SHA256, preventing cross-version
    concatenation. A server returning 200 to a resume request is never appended.
    Cancellation, network failure and checksum failure retain the partial file.
    """
    if not key:
        raise CivitaiError('Set a Civitai API key before downloading.')
    metadata = inspect_version(source, key, cancel_event)
    file = next((f for f in metadata['files'] if f['id'] == file_id), None)
    if file is None:
        raise CivitaiError('Selected file is not part of this model version.')
    folder = Path(destination)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / file['name']
    partial = folder / f'.civitai-{metadata["version"]}-{file_id}-{file["sha256"]}.part'
    if target.is_symlink() or partial.is_symlink():
        raise CivitaiError('Download paths must not be symbolic links.')
    if target.exists() and not overwrite:
        raise CivitaiError('Output exists. Enable replacement or choose another folder.')
    def log(message):
        if on_log:
            on_log(message)
    # A previous run may have finished transferring but stopped during verification.
    total = file['bytes'] if partial.exists() and partial.stat().st_size == file['bytes'] else None
    failures = 0
    while True:
        _cancel(cancel_event)
        offset = partial.stat().st_size if partial.exists() else 0
        if total is not None and offset == total:
            break
        if total is not None and offset > total:
            raise CivitaiError('Partial file is larger than the model. Choose another folder.')
        # Alternate domains and resolve the signed URL again on every attempt.
        host = HOSTS[failures % len(HOSTS)]
        url = f'https://{host}/api/download/models/{metadata["version"]}?fileId={file_id}'
        try:
            with _open(url, key, offset, cancel_event) as response:
                if response.status == 206:
                    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
                    if not match or int(match[1]) != offset or int(match[2]) != int(match[3]) - 1:
                        raise _Retry()
                    size = int(match[3])
                elif response.status == 200 and offset == 0:
                    size = int(response.headers.get('Content-Length', '0'))
                else:
                    # Live .com testing returned 200 for Range: preserve bytes and try .red.
                    raise _Retry()
                if size <= 0 or (total is not None and size != total):
                    raise CivitaiError('Civitai file size changed; partial preserved.')
                total = size
                if shutil.disk_usage(folder).free < total - offset + CHUNK:
                    raise CivitaiError('Insufficient free space for this model.')
                log(f'{"Resuming" if offset else "Downloading"} {file["name"]} at byte {offset}.')
                with partial.open('ab' if partial.exists() else 'xb') as output:
                    while offset < total:
                        _cancel(cancel_event)
                        data = response.read(min(CHUNK, total - offset))
                        if not data:
                            raise _Retry()
                        output.write(data)
                        offset += len(data)
                        if on_progress:
                            on_progress(offset / total, f'Downloading {file["name"]}')
                    if response.read(1):
                        raise CivitaiError('Unexpected extra download data; partial preserved.')
                    output.flush()
                    os.fsync(output.fileno())
        except (_Retry, urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException):
            failures += 1
            if failures >= 4:
                raise CivitaiError('Civitai transfer failed after four attempts. Partial preserved; start again to resume.') from None
            log('Transfer interrupted or resume unsupported. Retrying with a fresh link and alternate domain.')
            delay = 2 ** (failures - 1)
            if cancel_event is not None:
                if cancel_event.wait(delay):
                    _cancel(cancel_event)
            else:
                import time
                time.sleep(delay)
    log('Verifying SHA256 before publishing.')
    digest = hashlib.sha256()
    with partial.open('rb') as source_file:
        while data := source_file.read(CHUNK):
            _cancel(cancel_event)
            digest.update(data)
            if on_progress:
                on_progress(1.0, 'Verifying SHA256')
    if digest.hexdigest() != file['sha256']:
        raise CivitaiError('SHA256 mismatch. Partial preserved; choose another folder for a clean download.')
    _cancel(cancel_event)
    # Hard-link publication refuses a concurrently created destination, as atomic_output does.
    if overwrite:
        os.replace(partial, target)
    else:
        os.link(partial, target)
        partial.unlink()
    return str(target)
