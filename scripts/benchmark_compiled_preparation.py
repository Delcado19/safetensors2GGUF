"""Compare retained-F16 and direct-final-spool application paths on real slices."""
import argparse
import hashlib
import json
import logging
from pathlib import Path
import queue
import statistics
from time import perf_counter

from safetensors import safe_open
from safetensors.torch import save_file

from conversion_service import _pipeline
from scripts.check_flux_q6_path import verified_flux_reference
from scripts.compare_gguf_outputs import compare_files
from scripts.compare_native_reference_vectors import run_command
from scripts.gguf_research_io import open_gguf
from scripts.requantize_flux_q6 import FLUX_PRESETS


def main():
    """Retain byte/native parity, application times and logical payload I/O counts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--references', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.work.exists():
        parser.error('Choose fresh output and work paths')
    reference = verified_flux_reference(json.loads(args.references.read_text(encoding='utf8')))
    before = args.source.stat()
    key = 'double_blocks.0.img_attn.proj.weight'
    with safe_open(args.source, framework='pt', device='cpu') as source:
        tensors = {key: source.get_slice(key)[:1024, :].clone(),
                   'img_in.weight': source.get_slice('img_in.weight')[:8, :256].clone()}
    args.work.mkdir(parents=True, exist_ok=False)
    sample = args.work / 'slice.safetensors'
    save_file(tensors, str(sample))
    logging.getLogger('gguf').setLevel(logging.ERROR)
    report = {'scope': 'Warm application slice timings and logical payload I/O; not full-model '
                       'or physical disk throughput. No inference or large conversion.',
              'source': str(args.source), 'source_bytes': before.st_size, 'source_mtime_ns': before.st_mtime_ns,
              'slice_shape': list(tensors[key].shape), 'subset_sha256': hashlib.sha256(sample.read_bytes()).hexdigest(),
              'native_reference': reference, 'results': []}
    for preset in FLUX_PRESETS:
        folder = args.work / preset
        folder.mkdir()
        paths = {mode: folder / f'{mode}.gguf' for mode in ('retained_float', 'direct')}
        trials = {mode: [] for mode in paths}
        logs = {}
        intermediate = None
        for repeat in range(4):
            modes = ('retained_float', 'direct') if repeat % 2 == 0 else ('direct', 'retained_float')
            for mode in modes:
                events = queue.Queue()
                old = set(folder.glob('*-F16-tmp-*.gguf'))
                started = perf_counter()
                _pipeline(str(sample), str(paths[mode]), preset, '', 0, mode == 'retained_float',
                          repeat > 0, events, gguf_backend='compiled_flux')
                elapsed = perf_counter() - started
                if repeat:
                    trials[mode].append(elapsed)
                new = set(folder.glob('*-F16-tmp-*.gguf')) - old
                assert len(new) == (1 if mode == 'retained_float' else 0)
                if new:
                    intermediate = next(iter(new))
                logs[mode] = [e[1] for e in events.queue if e[0] == 'log']
        assert paths['retained_float'].read_bytes() == paths['direct'].read_bytes()
        native = folder / 'native.gguf'
        command = run_command([reference['executable'], str(intermediate), str(native), preset, '1'], text=True)
        native.with_suffix('.log').write_text(command.stdout + command.stderr, encoding='utf8')
        if command.returncode:
            raise RuntimeError(f'Native slice conversion failed: {native}')
        comparison = compare_files(paths['direct'], native)
        assert comparison['all_equal']
        with open_gguf(intermediate) as reader:
            prepared_bytes = sum(t.n_bytes for t in reader.tensors)
        with open_gguf(paths['direct']) as reader:
            final_bytes = sum(t.n_bytes for t in reader.tensors)
        report['results'].append({'preset': preset, 'trials_seconds': trials,
                                 'median_seconds': {mode: statistics.median(values) for mode, values in trials.items()},
                                 'retained_direct_file_bytes_equal': True, 'native_comparison': comparison,
                                 'prepared_payload_bytes': prepared_bytes, 'final_payload_bytes': final_bytes,
                                 'logical_payload_writes': {'retained_float': 2 * prepared_bytes + final_bytes,
                                                            'direct': 2 * final_bytes},
                                 'phase_logs': logs})
        print(preset, report['results'][-1]['median_seconds'], flush=True)
    assert (before.st_size, before.st_mtime_ns) == (args.source.stat().st_size, args.source.stat().st_mtime_ns)
    report['source_stat_unchanged'] = True
    report['source_code_sha256'] = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in (
        'convert.py', 'conversion_service.py', 'compiled_flux_writer.py', 'scripts/benchmark_compiled_preparation.py')}
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf8', newline='\n')


if __name__ == '__main__':
    main()
