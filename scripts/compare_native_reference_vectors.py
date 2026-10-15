"""Build frozen CPU GGML probes and compare block math without model dispatch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile

import gguf
import numpy as np

from scripts.converter_reference_vectors import load_molbal_k, measure_encoding, reference_vectors
from scripts.inventory_converter_sources import REFERENCES, verify_references


ROOT = Path(__file__).resolve().parents[1]
NATIVE_SOURCES = ('llama-b3962', 'llama-text-installed', 'llama-current')
FORMATS = ('Q8_0', 'Q4_0', 'Q4_1', 'Q5_0', 'Q5_1', 'Q2_K', 'Q3_K', 'Q4_K', 'Q5_K', 'Q6_K')


def run_command(command, **kwargs):
    """Capture subprocesses without flashing console windows on Windows."""
    return subprocess.run(command, capture_output=True, timeout=600,
                          creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), **kwargs)


def extract_source(entry, destination, folder=REFERENCES):
    """Materialize verified regular source files into a fresh build-only folder."""
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(folder / entry['archive']) as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            relative = Path(*member.name.split('/')[1:])
            target = destination / relative
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError('Source path escapes build directory')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.extractfile(member).read())


def build_probe(entry, work, cmake, folder=REFERENCES):
    """Build a static CPU probe and retain commands, compiler metadata and hashes."""
    source, build = work / 'source', work / 'build'
    extract_source(entry, source, folder)
    env = os.environ.copy()
    # Archived CMake must not identify this parent repository as its own revision.
    env['GIT_CEILING_DIRECTORIES'] = str(work.resolve())
    commands = [[cmake, '-S', str(ROOT / 'tools' / 'converter-reference'), '-B', str(build),
                 f'-DREFERENCE_SOURCE={source}', '-DCMAKE_BUILD_TYPE=Release'],
                [cmake, '--build', str(build), '--config', 'Release', '--target', 'converter-reference', '-j', '2']]
    for index, command in enumerate(commands):
        result = run_command(command, text=True, env=env)
        log = work / f'build-{index}.log'
        log.write_text(result.stdout + result.stderr, encoding='utf-8')
        if result.returncode:
            raise RuntimeError(f'Native build failed; see {log}')
    candidates = list(build.rglob('converter-reference.exe' if os.name == 'nt' else 'converter-reference'))
    executable = next(path for path in candidates if path.is_file())
    compiler_file = next(build.glob('CMakeFiles/*/CMakeCCompiler.cmake'))
    compiler_code = compiler_file.read_text(encoding='utf-8')
    compiler = {}
    for key in ('CMAKE_C_COMPILER', 'CMAKE_C_COMPILER_ID', 'CMAKE_C_COMPILER_VERSION'):
        match = re.search(r'set\(' + key + r' "([^"]+)"\)', compiler_code)
        compiler[key] = match.group(1) if match else 'unknown'
    cache = (build / 'CMakeCache.txt').read_text(encoding='utf-8')
    flags = [line for line in cache.splitlines() if line.startswith(('GGML_', 'BUILD_SHARED_LIBS:',
             'CMAKE_C_FLAGS', 'CMAKE_CXX_FLAGS', 'CMAKE_GENERATOR:', 'CMAKE_BUILD_TYPE:')) and '=' in line]
    return executable, {'id': entry['id'], 'commit': entry['commit'], 'archive_sha256': entry['sha256'],
                        'executable_sha256': hashlib.sha256(executable.read_bytes()).hexdigest(),
                        'compiler': compiler, 'flags': flags, 'commands': commands,
                        'executable': str(executable),
                        'probe_sha256': hashlib.sha256((ROOT / 'tools/converter-reference/probe.c').read_bytes()).hexdigest(),
                        'cmake_project_sha256': hashlib.sha256(
                            (ROOT / 'tools/converter-reference/CMakeLists.txt').read_bytes()).hexdigest()}


def native_encode(executable, data, qtype):
    """Enforce the binary protocol and cross-check native versus common decoding."""
    if data.ndim != 2 or data.dtype != np.float32 or not np.isfinite(data).all():
        raise ValueError('Native probe requires finite 2D float32 input')
    block, size = gguf.GGML_QUANT_SIZES[qtype]
    result = run_command([str(executable), qtype.name, *map(str, data.shape)], input=data.tobytes())
    if data.shape[-1] % block:
        if result.returncode == 2 and b'Incompatible row width' in result.stderr:
            raise ValueError('Native encoder rejected incompatible rows')
        raise RuntimeError('Native encoder did not explicitly reject incompatible rows')
    if result.returncode:
        raise RuntimeError(f'Native encoder failed: {result.stderr.decode(errors="replace")}')
    payload_size = data.size // block * size
    if len(result.stdout) != payload_size + data.nbytes:
        raise ValueError('Invalid native output payload length')
    payload = np.frombuffer(result.stdout[:payload_size], dtype=np.uint8).reshape(
        *data.shape[:-1], data.shape[-1] // block * size)
    native = np.frombuffer(result.stdout[payload_size:], dtype='<f4').reshape(data.shape)
    common = gguf.dequantize(payload, qtype)
    if not np.isfinite(native).all() or not np.allclose(native, common, atol=1e-6, rtol=1e-6):
        raise ValueError('Native and gguf-py decoders disagree')
    return payload, float(np.max(np.abs(native - common)))


def check_protocol(executable):
    """Exercise native argument, length and non-finite guards on the real binary."""
    good = np.zeros((1, 256), dtype=np.float32).tobytes()
    nan = np.full((1, 256), np.nan, dtype=np.float32).tobytes()
    cases = [('unknown_format', ['invalid', '1', '256'], good, 2),
             ('negative_dimension', ['Q4_K', '-1', '256'], good, 2),
             ('excessive_dimensions', ['Q4_K', '1048576', '256'], b'', 2),
             ('incompatible_row', ['Q4_K', '256', '2'], good * 2, 2),
             ('short_payload', ['Q4_K', '1', '256'], good[:-1], 3),
             ('extra_payload', ['Q4_K', '1', '256'], good + b'x', 3),
             ('nonfinite_input', ['Q4_K', '1', '256'], nan, 3)]
    results = []
    for name, arguments, payload, expected in cases:
        result = run_command([str(executable), *arguments], input=payload)
        if result.returncode != expected or result.stdout:
            raise ValueError(f'Native protocol guard failed: {name}')
        results.append({'case': name, 'exit_code': result.returncode, 'status': 'rejected'})
    return results


def compare_native(work, cmake, folder=REFERENCES):
    """Build three pinned kernels and compare deterministic bytes and errors."""
    if sys.byteorder != 'little':
        raise ValueError('Initial native probes require a little-endian host')
    manifest = verify_references(folder)
    molbal, provenance = load_molbal_k(folder)
    report = {'status': 'Pure reference kernels; no model/preset/runtime parity claim',
              'platform': platform.platform(), 'python': platform.python_version(),
              'numpy': np.__version__, 'gguf_decoder_sha256': hashlib.sha256(
                  Path(gguf.quants.__file__).read_bytes()).hexdigest(),
              'molbal': provenance, 'builds': [], 'results': []}
    reference_bytes = {}
    for sid in NATIVE_SOURCES:
        print(f'Building {sid}', flush=True)
        entry = next(item for item in manifest['sources'] if item['id'] == sid)
        executable, build = build_probe(entry, work / sid, cmake, folder)
        build['protocol_checks'] = check_protocol(executable)
        report['builds'].append(build)
        for name in FORMATS:
            qtype = gguf.GGMLQuantizationType[name]
            block, _ = gguf.GGML_QUANT_SIZES[qtype]
            for label, data in reference_vectors().items():
                row = {'source': sid, 'type': name, 'vector': label, 'shape': list(data.shape),
                       'input_sha256': hashlib.sha256(data.tobytes()).hexdigest()}
                if data.shape[-1] % block:
                    try:
                        native_encode(executable, data, qtype)
                    except ValueError as error:
                        if str(error) != 'Native encoder rejected incompatible rows':
                            raise
                    else:
                        raise ValueError('Native probe accepted incompatible rows')
                    row['status'] = 'rejected_incompatible_row'
                else:
                    payload, decoder_difference = native_encode(executable, data, qtype)
                    row.update(measure_encoding(lambda values, quant: native_encode(executable, values, quant)[0],
                                                data, qtype))
                    key = (name, label)
                    baseline = reference_bytes.setdefault(key, payload.tobytes())
                    row['equals_b3962_bytes'] = payload.tobytes() == baseline
                    row['native_decoder_max_difference'] = decoder_difference
                    if name in ('Q4_K', 'Q5_K', 'Q6_K'):
                        other = molbal(data, qtype)
                        row['equals_molbal_bytes'] = np.array_equal(payload, other)
                        row['molbal_rmse'] = measure_encoding(molbal, data, qtype)['rmse']
                report['results'].append(row)
        print(f'Compared {sid}', flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cmake', default=shutil.which('cmake'), help='CMake executable; compiler must be installed')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.cmake:
        parser.error('CMake was not found; supply --cmake PATH')
    build_root = ROOT / 'build' / 'converter-reference'
    build_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='run-', dir=build_root))
    args.output.write_text(json.dumps(compare_native(work, args.cmake), indent=2, allow_nan=False) + '\n', encoding='utf-8')
