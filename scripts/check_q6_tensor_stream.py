"""Research-only full-tensor Q6_K stream check; output is raw blocks, not GGUF."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import gguf
import numpy as np
from safetensors import safe_open
import torch

from conversion_io import atomic_output, validate_output
from convert import ConversionCancelled
from scripts.check_q6_pilot import verified_native_builds
from scripts.compare_native_reference_vectors import ROOT, native_encode
from scripts.reference_q6_k import quantize_q6_k


def stream_tensor(source, tensor_name, destination, references, *, cancel_event=None, on_progress=None):
    """Atomically write one plain floating-point matrix as raw .q6k blocks.

    Read at most 65,536 values or one complete row, whichever is larger.
    Compare every chunk with all three verified native references before writing.
    Cancellation, changed input or a late bad value discards the temporary file;
    existing outputs and source aliases are rejected. Returns metrics/provenance,
    never weight values. No model eligibility, mixed policy or loader is implied.
    """
    source, destination = Path(source), Path(destination)
    validate_output(source, destination)
    if destination.suffix != '.q6k':
        raise ValueError('Raw tensor output must use .q6k, not a model file extension')
    builds = verified_native_builds(references)
    before = source.stat()
    input_hash, output_hash = hashlib.sha256(), hashlib.sha256()
    started = time.perf_counter()
    encode_seconds = squared_error = max_error = 0.0
    chunks = written = 0

    def check_cancelled():
        if cancel_event is not None and cancel_event.is_set():
            raise ConversionCancelled('Q6_K tensor check cancelled')

    check_cancelled()
    with safe_open(str(source), framework='pt', device='cpu') as model:
        if any(name.endswith('.comfy_quant') for name in model.keys()):
            raise ValueError('Quantized sources require a separate reconstruction policy')
        tensor = model.get_slice(tensor_name)
        shape, dtype = tensor.get_shape(), tensor.get_dtype()
        if dtype not in ('F16', 'BF16', 'F32') or len(shape) != 2 or min(shape) <= 0 or shape[1] % 256:
            raise ValueError('Require a nonempty plain F16/BF16/F32 matrix with rows divisible by 256')
        rows_per_chunk = max(1, 65536 // shape[1])
        with atomic_output(destination) as temporary:
            with open(temporary, 'wb') as output:
                for start in range(0, shape[0], rows_per_chunk):
                    check_cancelled()
                    data = tensor[start:start + rows_per_chunk, :].to(torch.float32).numpy()
                    tick = time.perf_counter()
                    packed = quantize_q6_k(data)
                    encode_seconds += time.perf_counter() - tick
                    for build in builds:
                        check_cancelled()
                        expected, _ = native_encode(build['executable'], data, gguf.GGMLQuantizationType.Q6_K)
                        if not np.array_equal(packed, expected):
                            raise ValueError(f'Native Q6_K mismatch at row {start}: {build["id"]}')
                    decoded = gguf.dequantize(packed, gguf.GGMLQuantizationType.Q6_K)
                    error = decoded.astype(np.float64) - data
                    squared_error += float(np.sum(error * error))
                    max_error = max(max_error, float(np.max(np.abs(error))))
                    input_hash.update(data.tobytes())
                    payload = packed.tobytes()
                    output.write(payload)
                    output_hash.update(payload)
                    written += len(payload)
                    chunks += 1
                    if on_progress is not None:
                        on_progress(min(start + rows_per_chunk, shape[0]), shape[0])
            check_cancelled()
            after = source.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError('Source changed during tensor streaming')
            expected_bytes = shape[0] * shape[1] // 256 * 210
            if written != expected_bytes or Path(temporary).stat().st_size != expected_bytes:
                raise ValueError('Incomplete Q6_K tensor output')
            disk_hash = hashlib.sha256()
            with open(temporary, 'rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    check_cancelled()
                    disk_hash.update(block)
            if disk_hash.digest() != output_hash.digest():
                raise ValueError('Q6_K output readback checksum mismatch')
            check_cancelled()
    return {'status': 'Full raw tensor check; not a complete GGUF model or loader test',
            'source': str(source), 'tensor': tensor_name, 'shape': shape, 'source_dtype': dtype,
            'source_bytes': before.st_size, 'source_mtime_ns': before.st_mtime_ns,
            'output': str(destination), 'output_bytes': written, 'chunks': chunks,
            'max_source_chunk_values': min(rows_per_chunk, shape[0]) * shape[1],
            'float32_input_sha256': input_hash.hexdigest(), 'payload_sha256': output_hash.hexdigest(),
            'rmse': (squared_error / (shape[0] * shape[1])) ** 0.5, 'max_abs_error': max_error,
            'encode_seconds': encode_seconds, 'total_seconds': time.perf_counter() - started,
            'float32_encode_mib_per_second': shape[0] * shape[1] * 4 / 1024**2 / encode_seconds,
            'numpy': np.__version__, 'torch': torch.__version__,
            'pilot_source_sha256': hashlib.sha256((ROOT / 'scripts/reference_q6_k.py').read_bytes()).hexdigest(),
            'checker_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'native_comparisons': chunks * len(builds), 'all_equal': True,
            'references': [{key: build[key] for key in ('id', 'commit', 'archive_sha256', 'executable_sha256')}
                           for build in builds]}


def main():
    """Write a raw tensor under the explicit destination and a separate JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--tensor', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--references', type=Path, default=ROOT / 'data/converter-native-comparison.json')
    args = parser.parse_args()
    validate_output(args.source, args.report)
    validate_output(args.output, args.report)
    result = stream_tensor(args.source, args.tensor, args.output, json.loads(args.references.read_text()))
    with atomic_output(args.report) as temporary:
        Path(temporary).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
