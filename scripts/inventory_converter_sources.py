"""Verify frozen analysis sources and inventory Python model rules without imports."""
import argparse
import ast
import hashlib
import json
import posixpath
import tarfile
from pathlib import Path, PurePosixPath


REFERENCES = Path(__file__).resolve().parents[1] / 'third_party' / 'converter_references'


def verify_references(folder=REFERENCES):
    """Validate all archive hashes, file counts, paths and preserved root licenses."""
    manifest = json.loads((folder / 'sources.json').read_text(encoding='utf-8'))
    for entry in manifest['sources']:
        archive = folder / entry['archive']
        digest = hashlib.sha256()
        with archive.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        if archive.stat().st_size != entry['size_bytes'] or digest.hexdigest() != entry['sha256']:
            raise ValueError(f"Source checksum mismatch: {entry['id']}")
        root = entry['archive_root']
        with tarfile.open(archive) as source:
            members = source.getmembers()
            names = set()
            for member in members:
                path = PurePosixPath(member.name)
                if (path.is_absolute() or '..' in path.parts or not path.parts
                        or path.parts[0] != root or member.name in names):
                    raise ValueError(f'Unsafe/duplicate source path: {member.name}')
                names.add(member.name)
                if member.issym():
                    # b3962 includes relative Swift-package header links within its tree.
                    target = posixpath.normpath(posixpath.join(str(path.parent), member.linkname))
                    if PurePosixPath(member.linkname).is_absolute() or not target.startswith(root + '/'):
                        raise ValueError(f'Unsafe source link: {member.name}')
                elif not (member.isfile() or member.isdir()):
                    raise ValueError(f'Unsupported source member: {member.name}')
            if sum(member.isfile() for member in members) != entry['file_count']:
                raise ValueError(f"Source file count mismatch: {entry['id']}")
            licenses = [name for name in entry['license_files'] if '/' not in name]
            if not licenses:
                raise ValueError(f"Missing root license: {entry['id']}")
            for name in licenses:
                preserved = folder / (entry['id'] + '-LICENSE' + Path(name).suffix)
                if source.extractfile(root + '/' + name).read() != preserved.read_bytes():
                    raise ValueError(f"License mismatch: {entry['id']}")
    return manifest


def model_rules(code):
    """Extract explicit Python model declarations; inheritance remains unresolved."""
    models = []
    for node in ast.parse(code).body:
        if not isinstance(node, ast.ClassDef):
            continue
        fields = {}
        for item in node.body:
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name) and (target.id.startswith('keys_') or target.id in {
                            'arch', 'shape_fix', 'preserve_nd_shapes'}):
                        fields[target.id] = ast.unparse(item.value)
        registrations = [ast.unparse(dec) for dec in node.decorator_list
                         if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute)
                         and dec.func.attr == 'register']
        if fields or registrations:
            models.append({'class': node.name, 'line': node.lineno,
                           'bases': [ast.unparse(base) for base in node.bases],
                           'declared_rules': fields, 'registrations': registrations,
                           'methods': [item.name for item in node.body if isinstance(item, ast.FunctionDef)]})
    return models


def build_inventory(folder=REFERENCES):
    """Build a source-only comparison inventory after verifying every snapshot."""
    manifest = verify_references(folder)
    result = {'status': 'Static declarations, not runtime support or numerical parity', 'sources': []}
    for entry in manifest['sources']:
        record = {'id': entry['id'], 'commit': entry['commit'], 'python_models': {}}
        with tarfile.open(folder / entry['archive']) as source:
            # Recent llama.cpp moved model adapters out of the CLI into conversion/.
            filenames = ['tools/convert.py', 'convert_hf_to_gguf.py']
            prefix = entry['archive_root'] + '/'
            filenames.extend(member.name[len(prefix):] for member in source.getmembers()
                             if member.isfile() and member.name.startswith(prefix + 'conversion/')
                             and member.name.endswith('.py'))
            for filename in filenames:
                try:
                    code = source.extractfile(entry['archive_root'] + '/' + filename).read().decode('utf-8')
                except KeyError:
                    continue
                rules = model_rules(code)
                if rules:
                    record['python_models'][filename] = rules
                for node in ast.parse(code).body:
                    if isinstance(node, ast.Assign) and any(
                            isinstance(target, ast.Name) and target.id == 'QUANT_TYPE_MAP'
                            for target in node.targets):
                        record['quantization_keys'] = [ast.literal_eval(key) for key in node.value.keys]
        result['sources'].append(record)
    local = Path(__file__).resolve().parents[1]
    result['project_models'] = model_rules((local / 'models' / 'architectures.py').read_text(encoding='utf-8'))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Write JSON inventory; otherwise print it')
    args = parser.parse_args()
    payload = json.dumps(build_inventory(), indent=2) + '\n'
    if args.output:
        args.output.write_text(payload, encoding='utf-8')
    else:
        print(payload)
