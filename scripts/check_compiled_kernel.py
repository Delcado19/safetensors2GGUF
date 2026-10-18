"""Compare an explicitly trusted research C ABI with NumPy and frozen native bytes.

Loads only a caller-selected library with a matching SHA256; never builds,
downloads, or selects a production backend. Run the build documented in
docs/compiled-kernel-pilot.md first. Timings exclude file I/O and source loading.
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np

from conversion_io import atomic_output, validate_output
from scripts.check_q6_pilot import installed_weight_vectors, pilot_vectors, verified_native_builds
from scripts.compare_native_reference_vectors import native_encode
from scripts.gguf_research_io import open_gguf
from scripts.reference_q4_k import iter_q4_k
from scripts.reference_q5_k import iter_q5_k
from scripts.reference_q6_k import iter_q6_k
import gguf


ROOT = Path(__file__).resolve().parents[1]
ITERATORS = {4: iter_q4_k, 5: iter_q5_k, 6: iter_q6_k}


def _load(path, expected):
    path = Path(path).resolve(strict=True)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('Research library checksum mismatch')
    library = ctypes.CDLL(str(path))
    encode = library.converter_encode
    encode.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t,
                       ctypes.c_void_p, ctypes.c_size_t]
    encode.restype = ctypes.c_int
    return library, encode


def _encode(encode, data, bits):
    data = np.asarray(data)
    if (bits not in (2, 3, 4, 5, 6) or data.dtype != np.float32 or not data.size or data.size > 1048576
            or data.ndim == 0 or data.shape[-1] % 256 or not np.isfinite(data).all()):
        raise ValueError('Expected bounded finite float32 rows divisible by 256')
    data = np.ascontiguousarray(data)
    result = np.empty((data.size // 256, {2: 84, 3: 110, 4: 144, 5: 176, 6: 210}[bits]), dtype=np.uint8)
    status = encode(bits, data.ctypes.data, data.size, result.ctypes.data, result.size)
    if status:
        raise ValueError(f'Research kernel rejected input: {status}')
    return result


def _guard_checks(encode):
    data = np.zeros((1, 256), dtype=np.float32)
    output = np.full(144, 0xA5, dtype=np.uint8)
    for bits, pointer, count, target, capacity in (
        (1, data.ctypes.data, 256, output.ctypes.data, 144),
        (4, None, 256, output.ctypes.data, 144),
        (4, data.ctypes.data, 256, None, 144),
        (4, data.ctypes.data, 0, output.ctypes.data, 144),
        (4, data.ctypes.data, 255, output.ctypes.data, 144),
        (4, data.ctypes.data, 1048832, output.ctypes.data, 144),
        (4, data.ctypes.data, 256, output.ctypes.data, 143),
    ):
        assert encode(bits, pointer, count, target, capacity) == 1
        assert np.all(output == 0xA5)
    for bad in (np.nan, np.inf, -np.inf):
        data[0, -1] = bad
        assert encode(4, data.ctypes.data, 256, output.ctypes.data, 144) == 2
        assert np.all(output == 0xA5)


def main():
    """Retain source hashes, guard/parity checks and alternating slice timings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--build-record', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--models-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    for source in (args.library, args.source, args.build_record):
        validate_output(source, args.output)
    library, encode = _load(args.library, args.sha256)
    _guard_checks(encode)
    builds = verified_native_builds(json.loads((ROOT / 'data/converter-native-comparison.json').read_text()))
    build_record = json.loads(args.build_record.read_text())
    assert build_record['library_sha256'] == args.sha256
    assert build_record['commit'] == builds[0]['commit']
    assert build_record['archive_sha256'] == builds[0]['archive_sha256']
    audit = json.loads((ROOT / 'data/installed-model-policy-headers.json').read_text())
    sampled, _ = installed_weight_vectors(args.models_root, audit)
    vectors = list(pilot_vectors()) + sampled
    rows = []
    for bits in ITERATORS:
        golden = json.loads((ROOT / f'data/q{bits}-pilot-comparison.json').read_text())
        expected = {r['vector']: r for r in golden['results']}
        for name, data in vectors:
            row = expected[name]
            assert hashlib.sha256(data.tobytes()).hexdigest() == row['input_sha256'], name
            actual = _encode(encode, data, bits)
            assert hashlib.sha256(actual).hexdigest() == row['payload_sha256'], (bits, name)
        rows.append({'format': f'Q{bits}_K', 'golden_inputs': len(vectors), 'all_equal': True})
    before = args.source.stat()
    with open_gguf(args.source) as reader:
        tensor = next(t for t in reader.tensors if t.tensor_type == gguf.GGMLQuantizationType.F16
                      and len(t.shape) == 2 and t.shape[0] % 256 == 0 and t.n_elements >= 1048576)
        data = np.array(tensor.data.reshape(-1)[:1048576], dtype=np.float32).reshape(-1, 256)
        name = tensor.name
    benchmarks = []
    for bits, iterator in ITERATORS.items():
        times = {'numpy': [], 'compiled': [], 'native_direct': []}
        hashes = {}
        direct_output = np.empty((data.size // 256, {2: 84, 3: 110, 4: 144, 5: 176, 6: 210}[bits]), dtype=np.uint8)
        # Initialize outside measurement; alternate order to reduce warm-cache bias.
        _encode(encode, data[:256], bits)
        for repeat in range(7):
            order = ('numpy', 'compiled', 'native_direct')
            for backend in (order if repeat % 2 == 0 else order[::-1]):
                start = time.perf_counter()
                digest = hashlib.sha256()
                if backend == 'numpy':
                    for chunk in iterator(data):
                        digest.update(chunk)
                elif backend == 'compiled':
                    for offset in range(0, len(data), 256):
                        digest.update(_encode(encode, data[offset:offset + 256], bits))
                else:
                    # Same native math, one call/preallocated output, no Python scan.
                    assert encode(bits, data.ctypes.data, data.size,
                                  direct_output.ctypes.data, direct_output.size) == 0
                    digest.update(direct_output)
                times[backend].append(time.perf_counter() - start)
                hashes[backend] = digest.hexdigest()
        native, _ = native_encode(builds[0]['executable'], data, gguf.GGMLQuantizationType[f'Q{bits}_K'])
        assert hashes['numpy'] == hashes['compiled'] == hashes['native_direct'] == hashlib.sha256(native).hexdigest()
        benchmarks.append({'format': f'Q{bits}_K', 'seconds': times,
                           'median_seconds': {key: statistics.median(value) for key, value in times.items()},
                           'payload_sha256': hashes['compiled'], 'native_equal': True})
    after = args.source.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    report = {'scope': 'Research C ABI only; no application integration or full-model speed claim',
              'python': sys.version, 'numpy': np.__version__, 'platform': sys.platform,
              'library': str(args.library.resolve()), 'library_sha256': args.sha256,
              'build': build_record,
              'source_code_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                  for name in ('tools/converter-kernel/kernel.c', 'tools/converter-kernel/CMakeLists.txt',
                               'scripts/check_compiled_kernel.py', 'scripts/reference_q4_k.py',
                               'scripts/reference_q5_k.py', 'scripts/reference_q6_k.py')},
              'native_reference': builds[0], 'guard_checks': 10, 'parity': rows,
              'source': str(args.source.resolve()), 'source_stat_stable': True,
              'tensor': name, 'values': data.size, 'chunk_blocks': 256,
              'input_sha256': hashlib.sha256(data).hexdigest(), 'benchmarks': benchmarks}
    with atomic_output(args.output) as temporary:
        Path(temporary).write_text(json.dumps(report, indent=2) + '\n', encoding='utf8')
    print(json.dumps(benchmarks, indent=2))
    del library  # Keep the CDLL alive until all calls have completed.


if __name__ == '__main__':
    main()
