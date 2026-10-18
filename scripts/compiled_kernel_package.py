"""Explicit offline build and verification of a research-only CPU kernel package.

Packages identify integrity and host compatibility, not publisher authenticity.
Only load a package you trust. No downloads, automatic builds or backend fallback.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys

from scripts.check_compiled_kernel import _load, _encode, _guard_checks
from scripts.check_q6_pilot import pilot_vectors
from scripts.compare_native_reference_vectors import extract_source, run_command
from scripts.inventory_converter_sources import REFERENCES, verify_references


ROOT = Path(__file__).resolve().parents[1]
FORMATS = {2: 'Q2_K', 3: 'Q3_K', 4: 'Q4_K', 5: 'Q5_K', 6: 'Q6_K'}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _platform():
    machine = platform.machine().lower()
    return {'system': sys.platform, 'machine': {'amd64': 'x86_64', 'arm64': 'aarch64'}.get(machine, machine),
            'pointer_bits': ctypes.sizeof(ctypes.c_void_p) * 8}


def _library_name():
    return ('converter-kernel.dll' if sys.platform == 'win32' else
            'libconverter-kernel.dylib' if sys.platform == 'darwin' else 'libconverter-kernel.so')


def _source_identity():
    sources = json.loads((REFERENCES / 'sources.json').read_text(encoding='utf8'))['sources']
    entry = next(row for row in sources if row['id'] == 'llama-b3962')
    return {key: entry[key] for key in ('id', 'commit', 'sha256')}


def _self_test(encode):
    if not __debug__:
        raise ValueError('Kernel guard self-tests require Python without -O')
    _guard_checks(encode)
    count = 0
    vectors = list(pilot_vectors())
    for bits in FORMATS:
        report_path = ROOT / ('data/q2-q3-kernel-goldens.json' if bits < 4 else
                              f'data/q{bits}-pilot-comparison.json')
        report = json.loads(report_path.read_text(encoding='utf8'))
        goldens = {row['vector']: row for row in report['results']
                   if bits >= 4 or row['bits'] == bits}
        for name, data in vectors:
            row = goldens[name]
            if hashlib.sha256(data.tobytes()).hexdigest() != row['input_sha256']:
                raise ValueError(f'Kernel golden input changed: {name}')
            payload = _encode(encode, data, bits)
            if hashlib.sha256(payload).hexdigest() != row['payload_sha256']:
                raise ValueError(f'Compiled kernel differs from native golden: {bits}, {name}')
            count += 1
    return {'golden_comparisons': count, 'guard_checks': 10, 'formats': list(FORMATS.values())}


def verify_package(directory):
    """Validate host, source identity and every packaged file before native loading.

    Reject traversal and file links escaping the selected package. The manifest
    is not a signature; selecting the directory explicitly grants native-code trust.
    """
    directory = Path(directory).resolve(strict=True)
    manifest = json.loads((directory / 'kernel.json').read_text(encoding='utf8'))
    if manifest.get('schema') != 1 or manifest.get('platform') != _platform():
        raise ValueError('Kernel package version or platform mismatch')
    if manifest.get('source') != _source_identity():
        raise ValueError('Kernel package source does not match the pinned archive')
    library = _library_name()
    files = manifest.get('files')
    if manifest.get('library') != library or not isinstance(files, dict) or set(files) != {
            library, 'LICENSE', 'kernel.c', 'CMakeLists.txt'}:
        raise ValueError('Invalid kernel package file list')
    for name, digest in files.items():
        path = directory / name
        if path.resolve(strict=True).parent != directory or _sha(path) != digest:
            raise ValueError(f'Kernel package checksum/path mismatch: {name}')
    for name in ('kernel.c', 'CMakeLists.txt'):
        if files[name] != _sha(ROOT / 'tools/converter-kernel' / name):
            raise ValueError(f'Kernel package adapter version mismatch: {name}')
    if files['LICENSE'] != _sha(REFERENCES / 'llama-b3962-LICENSE'):
        raise ValueError('Kernel package license mismatch')
    return manifest


def load_package(directory):
    """Load an explicit trusted package, replay 725 goldens, return callback/manifest."""
    manifest = verify_package(directory)
    library, encode = _load(Path(directory) / manifest['library'], manifest['files'][manifest['library']])
    checked = _self_test(encode)
    if checked != manifest.get('self_test'):
        raise ValueError('Kernel package self-test record mismatch')

    def kernel(data, bits):
        # Capture the library so its handle outlives every writer callback.
        return _encode(library.converter_encode, data, bits)

    return kernel, manifest


def build_package(directory, work, cmake):
    """Build verified archived sources offline; publish kernel.json only after checks.

    Both destinations must be fresh and disjoint. Build logs survive failures;
    a failed package copy can leave an incomplete folder without a manifest.
    No existing package is overwritten or repaired implicitly.
    """
    directory, work = Path(directory).resolve(), Path(work).resolve()
    if directory.exists() or work.exists():
        raise FileExistsError('Choose fresh package and work directories')
    if directory.is_relative_to(work) or work.is_relative_to(directory):
        raise ValueError('Package and work directories must be disjoint')
    if not cmake or not shutil.which(str(cmake)):
        raise ValueError('CMake is required; use --cmake with its executable path')
    source = next(row for row in verify_references()['sources'] if row['id'] == 'llama-b3962')
    work.mkdir(parents=True, exist_ok=False)
    extract_source(source, work / 'source')
    env = os.environ.copy()
    env['GIT_CEILING_DIRECTORIES'] = str(work)
    commands = [[str(cmake), '-S', str(ROOT / 'tools/converter-kernel'), '-B', str(work / 'build'),
                 '-DREFERENCE_SOURCE=' + str(work / 'source'), '-DCMAKE_BUILD_TYPE=Release'],
                [str(cmake), '--build', str(work / 'build'), '--config', 'Release',
                 '--target', 'converter-kernel', '-j', '2']]
    for index, command in enumerate(commands):
        result = run_command(command, text=True, env=env)
        log = work / f'build-{index}.log'
        log.write_text(result.stdout + result.stderr, encoding='utf8')
        if result.returncode:
            raise RuntimeError(f'Kernel build failed; see {log}')
    candidates = list((work / 'build').rglob(_library_name()))
    if len(candidates) != 1:
        raise ValueError('Build must produce exactly one kernel library')
    library = candidates[0]
    handle, encode = _load(library, _sha(library))
    checked = _self_test(encode)
    manifest = {'schema': 1, 'platform': _platform(), 'source': _source_identity(),
                'library': library.name, 'self_test': checked, 'commands': commands,
                'compiler': next((work / 'build/CMakeFiles').glob('*/CMakeCCompiler.cmake')).read_text(encoding='utf8'),
                'cache': (work / 'build/CMakeCache.txt').read_text(encoding='utf8')}
    directory.mkdir(parents=True, exist_ok=False)
    for source_path, name in ((library, library.name), (REFERENCES / 'llama-b3962-LICENSE', 'LICENSE'),
                              (ROOT / 'tools/converter-kernel/kernel.c', 'kernel.c'),
                              (ROOT / 'tools/converter-kernel/CMakeLists.txt', 'CMakeLists.txt')):
        shutil.copyfile(source_path, directory / name)
    manifest['files'] = {path.name: _sha(path) for path in directory.iterdir()}
    # A partially copied package has no manifest and cannot be selected successfully.
    with (directory / 'kernel.json').open('x', encoding='utf8') as stream:
        json.dump(manifest, stream, indent=2)
        stream.write('\n')
    verify_package(directory)
    del handle
    return manifest


def main():
    """Build an explicit package or verify/replay a previously trusted local package."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path, help='Fresh package directory, or existing with --verify')
    parser.add_argument('--work', type=Path, help='Fresh directory for preserved sources and build logs')
    parser.add_argument('--cmake', default=shutil.which('cmake'))
    parser.add_argument('--verify', action='store_true', help='Verify trusted package including native self-tests')
    args = parser.parse_args()
    if args.verify:
        _, manifest = load_package(args.output)
    else:
        if not args.work:
            parser.error('--work is required for a build')
        manifest = build_package(args.output, args.work, args.cmake)
    print(json.dumps({'package': str(args.output.resolve()), 'platform': manifest['platform'],
                      'self_test': manifest['self_test'], 'library_sha256': manifest['files'][manifest['library']]},
                     indent=2))


if __name__ == '__main__':
    main()
