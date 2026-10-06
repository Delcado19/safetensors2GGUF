"""Verify the project-owned molbal snapshot offline without importing its code."""
import ast
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath


def verify_snapshot(folder: Path | None = None) -> dict:
    """Check archive integrity, safe paths, completeness, license and adapter entrypoints."""
    folder = folder or Path(__file__).resolve().parents[1] / 'third_party' / 'molbal'
    manifest = json.loads((folder / 'source.json').read_text(encoding='utf-8'))
    archive = folder / manifest['archive']
    payload = archive.read_bytes()
    if len(payload) != manifest['size_bytes'] or hashlib.sha256(payload).hexdigest() != manifest['sha256']:
        raise ValueError('Molbal source archive size/checksum mismatch')
    root = manifest['archive_root']
    with tarfile.open(archive) as source:
        members = source.getmembers()
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or '..' in path.parts or path.parts[0] != root or not (member.isfile() or member.isdir()):
                raise ValueError(f'Unexpected snapshot member: {member.name}')
        if sum(member.isfile() for member in members) != manifest['file_count']:
            raise ValueError('Incomplete molbal source snapshot')
        if source.extractfile(root + '/LICENSE').read() != (folder / 'LICENSE').read_bytes():
            raise ValueError('Snapshot license does not match the preserved license')
        for filename in ('tools/convert.py', 'loader.py', 'ops.py', 'lora.py', 'dequant.py'):
            code = source.extractfile(root + '/' + filename).read().decode('utf-8')
            module = ast.parse(code, filename=filename)
            if filename == 'tools/convert.py':
                names = {node.name for node in module.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
                if not {'convert_file', 'convert_safetensors_streamed', 'parse_args', 'ModelKrea2'} <= names:
                    raise ValueError('Snapshot lacks the planned Krea conversion entrypoints')
    return {'commit': manifest['commit'], 'sha256': manifest['sha256'], 'files': manifest['file_count'], 'verified': True}


if __name__ == '__main__':
    print(json.dumps(verify_snapshot()))
