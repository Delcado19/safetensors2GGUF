"""Pinned local text-conversion tools; normal conversion never fetches or compiles."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
import tempfile
from scripts.compiled_kernel_package import _platform

ROOT = Path(__file__).resolve().parent
REFERENCES = ROOT / 'third_party/converter_references'


def validate_text_destination(destination):
    """Prevent model output from replacing shipped tools or cached converter sources."""
    target = Path(destination).resolve()
    for directory in (ROOT / 'third_party', ROOT / 'build/pinned-text-converter'):
        if target == directory.resolve() or target.is_relative_to(directory.resolve()):
            raise ValueError('Model output must not overwrite managed converter tools or sources')


def _digest(path):
    with Path(path).open('rb') as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
        return digest.hexdigest()


def text_source(cache):
    """Materialize and verify the shipped source snapshot, including sibling modules.

    Existing extracted files are checked against the trusted archive on every use.
    Publish a fresh cache atomically; corrupt caches fail rather than self-repair.
    """
    entry = next(row for row in json.loads((REFERENCES / 'sources.json').read_text())['sources']
                 if row['id'] == 'llama-text-installed')
    archive = REFERENCES / entry['archive']
    if _digest(archive) != entry['sha256']:
        raise ValueError('Pinned text converter archive checksum mismatch')
    cache = Path(cache)
    destination = cache / entry['commit']
    cache.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as source, tempfile.TemporaryDirectory(dir=cache) as temporary:
        stage = Path(temporary) / 'source'
        stage.mkdir()
        for member in source:
            parts = PurePosixPath(member.name).parts
            if not parts or parts[0] != entry['archive_root'] or '..' in parts:
                raise ValueError('Unsafe pinned converter archive member')
            if not member.isfile():
                continue
            relative = Path(*parts[1:])
            expected = source.extractfile(member).read()
            if destination.exists():
                target = destination / relative
                if (not target.resolve().is_relative_to(destination.resolve()) or target.is_symlink()
                        or not target.is_file() or _digest(target) != hashlib.sha256(expected).hexdigest()):
                    raise ValueError('Pinned text converter cache checksum/path mismatch: ' + str(relative))
            else:
                target = stage / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(expected)
        if not destination.exists():
            try:
                stage.rename(destination)
            except FileExistsError:
                # Another conversion published first; validate it before executing.
                return text_source(cache)
    return destination


def text_quantizer():
    """Return the verified unpatched CPU executable; never discover an image binary."""
    host = _platform()
    if host != {'system': 'win32', 'machine': 'x86_64', 'pointer_bits': 64}:
        raise RuntimeError('No bundled text quantizer for this platform; install a supported distribution')
    directory = ROOT / 'third_party/text-tools/win32-x86_64'
    manifest = json.loads((directory / 'tools.json').read_text())
    if manifest.get('schema') != 1 or manifest.get('platform') != host:
        raise ValueError('Bundled text quantizer platform/schema mismatch')
    entry = next(row for row in json.loads((REFERENCES / 'sources.json').read_text())['sources']
                 if row['id'] == 'llama-text-installed')
    if manifest['source_commit'] != entry['commit'] or manifest['source_archive_sha256'] != entry['sha256']:
        raise ValueError('Bundled text quantizer source mismatch')
    if set(manifest['files']) != {'llama-quantize.exe', 'LICENSE'}:
        raise ValueError('Invalid bundled text quantizer file list')
    for name, digest in manifest['files'].items():
        path = directory / name
        if path.resolve().parent != directory.resolve() or _digest(path) != digest:
            raise ValueError('Bundled text quantizer checksum/path mismatch: ' + name)
    if _digest(directory / 'LICENSE') != _digest(REFERENCES / 'llama-text-installed-LICENSE'):
        raise ValueError('Bundled text quantizer license mismatch')
    return directory / 'llama-quantize.exe'
