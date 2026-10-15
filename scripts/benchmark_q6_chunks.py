"""Measure Q6_K chunk sizes on a prepared FLUX slice in isolated CPU processes.

Kernel-only timings exclude source loading and output file I/O. Peak RSS includes
imports/input allocation and is not a full-model memory guarantee.
"""
import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import gguf
import numpy as np

from conversion_io import atomic_output, validate_output
from scripts.gguf_research_io import open_gguf
from scripts.reference_q6_k import iter_q6_k


def _peak_rss():
    if sys.platform != 'win32':
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == 'darwin' else value * 1024)
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ('peak_working', 'working', 'peak_paged',
             'paged', 'peak_nonpaged', 'nonpaged', 'pagefile', 'peak_pagefile')]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return counters.peak_working


def _worker(source, blocks, values, repeats):
    with open_gguf(source) as reader:
        tensor = next(t for t in reader.tensors if t.tensor_type == gguf.GGMLQuantizationType.F16
                      and len(t.shape) == 2 and t.shape[0] % 256 == 0 and t.n_elements >= values)
        data = np.array(tensor.data.reshape(-1)[:values], dtype=np.float32, copy=True)
        name = tensor.name
    baseline = _peak_rss()
    durations, hashes = [], []
    for _ in range(repeats):
        digest = hashlib.sha256()
        start = time.perf_counter()
        for chunk in iter_q6_k(data, chunk_blocks=blocks):
            digest.update(chunk)
        durations.append(time.perf_counter() - start)
        hashes.append(digest.hexdigest())
    assert len(set(hashes)) == 1
    return {'chunk_blocks': blocks, 'values': values, 'tensor': name,
            'input_float32_sha256': hashlib.sha256(data).hexdigest(), 'payload_sha256': hashes[0],
            'seconds': durations, 'median_seconds': statistics.median(durations),
            'baseline_peak_rss_bytes': baseline, 'process_peak_rss_bytes': _peak_rss()}


def main():
    """Write fresh benchmark evidence; each chunk candidate gets an isolated worker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output')
    parser.add_argument('--reference', help='Optional native Q6_K file for matching tensor-prefix validation')
    parser.add_argument('--values', type=int, default=1048576)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--worker', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.values <= 0 or args.values % 256 or args.repeats <= 0:
        parser.error('values must be positive/divisible by 256 and repeats positive')
    if args.worker is not None:
        print(json.dumps(_worker(args.source, args.worker, args.values, args.repeats)))
        return
    if not args.output:
        parser.error('--output is required')
    validate_output(args.source, args.output)
    before = Path(args.source).stat()
    rows = []
    for blocks in (16, 64, 256, 1024, 4096):
        process = subprocess.run([sys.executable, '-m', 'scripts.benchmark_q6_chunks',
                                  '--source', args.source, '--values', str(args.values),
                                  '--repeats', str(args.repeats), '--worker', str(blocks)],
                                 check=True, capture_output=True, text=True,
                                 creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        rows.append(json.loads(process.stdout))
    after = Path(args.source).stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    assert len({(r['input_float32_sha256'], r['payload_sha256'], r['tensor']) for r in rows}) == 1
    report = {'scope': 'Isolated processes; sequential repeats of one float32 slice; kernel plus payload hashing only',
              'source': str(Path(args.source).resolve()), 'source_bytes': before.st_size,
              'source_mtime_ns': before.st_mtime_ns, 'python': sys.version,
              'numpy': np.__version__, 'platform': sys.platform, 'all_payloads_equal': True,
              'benchmark_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'results': rows}
    if args.reference:
        validate_output(args.reference, args.output)
        reference_before = Path(args.reference).stat()
        with open_gguf(args.reference) as reader:
            tensor = next(t for t in reader.tensors if t.name == rows[0]['tensor'])
            assert tensor.tensor_type == gguf.GGMLQuantizationType.Q6_K
            assert tensor.n_elements >= args.values
            prefix = tensor.data.view(np.uint8).reshape(-1)[:args.values // 256 * 210]
            native_hash = hashlib.sha256(prefix).hexdigest()
        reference_after = Path(args.reference).stat()
        assert (reference_before.st_size, reference_before.st_mtime_ns) == (
            reference_after.st_size, reference_after.st_mtime_ns)
        assert native_hash == rows[0]['payload_sha256']
        report['native_prefix'] = {'path': str(Path(args.reference).resolve()),
                                   'bytes': args.values // 256 * 210, 'sha256': native_hash,
                                   'equal': True}
    with atomic_output(args.output) as temporary:
        Path(temporary).write_text(json.dumps(report, indent=2) + '\n', encoding='utf8', newline='\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
