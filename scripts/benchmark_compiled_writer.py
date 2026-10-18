"""Measure bounded compiled FLUX writer phases on a small real-weight slice."""
import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import queue
import statistics
from time import perf_counter

from safetensors import safe_open
from safetensors.torch import save_file

from conversion_io import validate_output
from conversion_service import _pipeline
from convert import convert_file
from quantize import PYTHON_PRECISIONS
from scripts import requantize_flux_q6 as writer
from scripts.compiled_kernel_package import load_package
from scripts.check_flux_q6_path import verified_flux_reference
from scripts.compare_native_reference_vectors import run_command
from scripts.compare_gguf_outputs import compare_files


def main():
    """Compare serial/parallel files and verified native bytes without inference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--kernel-package', required=True, type=Path)
    parser.add_argument('--references', required=True, type=Path)
    parser.add_argument('--work', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    for path in (args.source, args.references):
        validate_output(path, args.output)
    if args.output.exists() or args.work.exists():
        parser.error('Choose fresh work and report paths')
    logging.getLogger('gguf').setLevel(logging.ERROR)
    reference = verified_flux_reference(json.loads(args.references.read_text(encoding='utf8')))
    started = perf_counter()
    kernel, manifest = load_package(args.kernel_package)
    verification_seconds = perf_counter() - started
    before = args.source.stat()
    key = 'double_blocks.0.img_attn.proj.weight'
    started = perf_counter()
    with safe_open(args.source, framework='pt', device='cpu') as source:
        sample = source.get_slice(key)[:1024, :].clone()
        protected = source.get_slice('img_in.weight')[:8, :256].clone()
    slice_read_seconds = perf_counter() - started
    args.work.mkdir(parents=True, exist_ok=False)
    subset = args.work / 'slice.safetensors'
    save_file({key: sample, 'img_in.weight': protected}, str(subset))
    prepared = args.work / 'prepared.gguf'
    started = perf_counter()
    convert_file(str(subset), str(prepared), interact=False, target_quant=PYTHON_PRECISIONS['F16'][0],
                 apply_unsqueeze=False, on_log=lambda *_: None)
    preparation_seconds = perf_counter() - started
    workers = writer._COMPILED_WORKERS
    report = {'scope': 'Warm real-weight slice writer benchmark; excludes imports and full-model I/O. '
                       'Serial and parallel timing includes encode/write/readback/file hash. No inference.',
              'source': str(args.source), 'source_tensor': key, 'slice_shape': list(sample.shape),
              'slice_read_seconds': slice_read_seconds, 'kernel_verification_seconds': verification_seconds,
              'float_preparation_seconds': preparation_seconds, 'cpu_count': os.cpu_count(),
              'parallel_workers': workers, 'kernel_sha256': manifest['files'][manifest['library']],
              'native_reference': reference, 'results': []}
    try:
        for preset in writer.FLUX_PRESETS:
            trials = {'serial': [], 'parallel': []}
            paths = {mode: args.work / f'{preset}-{mode}.gguf' for mode in trials}
            # Discard one warmup per mode, then alternate three measured repetitions.
            for repeat in range(4):
                order = ('serial', 'parallel') if repeat % 2 == 0 else ('parallel', 'serial')
                for mode in order:
                    writer._COMPILED_WORKERS = 1 if mode == 'serial' else workers
                    started = perf_counter()
                    result = writer.requantize(prepared, paths[mode], preset=preset,
                                               compiled_kernel=kernel, overwrite=repeat > 0)
                    elapsed = perf_counter() - started
                    if repeat:
                        trials[mode].append({'total_seconds': elapsed, **result['timings']})
            assert paths['serial'].read_bytes() == paths['parallel'].read_bytes()
            native = args.work / f'{preset}-native.gguf'
            command = run_command([reference['executable'], str(prepared), str(native), preset, '1'], text=True)
            native.with_suffix('.log').write_text(command.stdout + command.stderr, encoding='utf8')
            if command.returncode:
                raise RuntimeError(f'Native fixture failed: {native}')
            comparison = compare_files(paths['parallel'], native)
            assert comparison['all_equal']
            row = {'preset': preset, 'trials': trials,
                   'median_seconds': {mode: statistics.median(t['total_seconds'] for t in values)
                                      for mode, values in trials.items()},
                   'serial_parallel_file_bytes_equal': True, 'native_comparison': comparison}
            report['results'].append(row)
            print(preset, row['median_seconds'], flush=True)
    finally:
        writer._COMPILED_WORKERS = workers
    events = queue.Queue()
    _pipeline(str(subset), str(args.work / 'application-Q2_K.gguf'), 'Q2_K', '', 0,
              False, False, events, gguf_backend='compiled_flux')
    report['application_phase_logs'] = [e[1] for e in events.queue if e[0] == 'log' and ' s' in e[1]]
    assert not list(args.work.glob('*-F16-tmp-*.gguf'))
    assert (before.st_size, before.st_mtime_ns) == (args.source.stat().st_size, args.source.stat().st_mtime_ns)
    report['source_stat_unchanged'] = True
    report['source_code_sha256'] = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in (
        'conversion_service.py', 'scripts/requantize_flux_q6.py', 'scripts/benchmark_compiled_writer.py')}
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf8', newline='\n')


if __name__ == '__main__':
    main()
