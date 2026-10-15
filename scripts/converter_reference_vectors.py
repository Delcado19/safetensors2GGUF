"""Run small CPU encoder probes; no weights, native build or inference required."""
import argparse
import ast
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
import tarfile

import gguf
import numpy as np

from scripts.inventory_converter_sources import REFERENCES, verify_references


def reference_vectors():
    """Return deterministic float32 probes, including incompatible row widths."""
    ramp = np.linspace(-2, 2, 512, dtype=np.float32).reshape(2, 256)
    outlier = np.zeros((2, 256), dtype=np.float32)
    outlier[0, 0], outlier[1, -1] = 100, -100
    return {'zero': np.zeros_like(ramp), 'constant_positive': np.full_like(ramp, 0.75),
            'constant_negative': np.full_like(ramp, -0.75), 'ramp': ramp,
            'outliers': outlier,
            'positive_ramp': np.linspace(1, 2, 512, dtype=np.float32).reshape(2, 256),
            'negative_ramp': np.linspace(-2, -1, 512, dtype=np.float32).reshape(2, 256),
            'rounding_probe': np.tile(np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float32), (2, 64)),
            'one_block': ramp[:1].copy(),
            'standard_block_only': ramp[:, :32].copy(),
            'incompatible_rows': ramp.reshape(256, 2),
            'unaligned': ramp[:, :255].copy()}


def load_molbal_k(folder=REFERENCES):
    """Load only the five audited math functions from the verified pinned source.

    This deliberately executes function definitions, unlike the static inventory.
    It never imports Molbal's CLI, model classes, loaders or GPU dependencies.
    """
    manifest = verify_references(folder)
    entry = next(item for item in manifest['sources'] if item['id'] == 'molbal-current')
    with tarfile.open(folder / entry['archive']) as source:
        code = source.extractfile(entry['archive_root'] + '/tools/convert.py').read().decode('utf-8')
    names = {'_pack_k_scales', '_pack_fp16_pair', '_quantize_k_affine', '_quantize_k_q6', 'quantize_k'}
    nodes = [node for node in ast.parse(code).body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != names:
        raise ValueError('Pinned Molbal K encoder functions are incomplete')
    namespace = {'np': np, 'gguf': gguf, 'K_QUANT_BLOCK_SIZE': 256}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<verified molbal K math>', 'exec'), namespace)
    return namespace['quantize_k'], {'commit': entry['commit'], 'archive_sha256': entry['sha256'],
                                    'file': 'tools/convert.py',
                                    'functions': {node.name: [node.lineno, node.end_lineno] for node in nodes}}


def measure_encoding(encoder, data, qtype):
    """Check byte layout/determinism and measure reconstruction with gguf-py.

    Encoding errors propagate. The caller distinguishes explicitly incompatible
    shapes from failures on valid inputs; errors must not become a passing row.
    """
    encoded = encoder(data.copy(), qtype)
    block, size = gguf.GGML_QUANT_SIZES[qtype]
    expected_shape = (*data.shape[:-1], data.shape[-1] // block * size)
    if data.shape[-1] % block or encoded.dtype != np.uint8 or encoded.shape != expected_shape:
        raise ValueError('Encoder returned an invalid GGUF byte layout')
    if not np.array_equal(encoded, encoder(data.copy(), qtype)):
        raise ValueError('Encoder returned non-deterministic bytes')
    reconstructed = gguf.dequantize(encoded, qtype)
    if reconstructed.shape != data.shape or not np.isfinite(reconstructed).all():
        raise ValueError('Decoder returned an invalid or non-finite reconstruction')
    error = reconstructed.astype(np.float64) - data.astype(np.float64)
    return {'status': 'encoded', 'bytes': encoded.nbytes,
            'payload_sha256': hashlib.sha256(encoded.tobytes()).hexdigest(),
            'reconstructed_min': float(reconstructed.min()),
            'reconstructed_max': float(reconstructed.max()),
            'rmse': float(np.sqrt(np.mean(error ** 2))),
            'max_absolute_error': float(np.max(np.abs(error)))}


def run_vectors(folder=REFERENCES):
    """Probe Molbal K encoders and installed standard encoders using one decoder."""
    if sys.byteorder != 'little':
        raise ValueError('These initial reference probes require a little-endian host')
    k_encoder, provenance = load_molbal_k(folder)
    decoder_file = Path(gguf.quants.__file__)
    report = {'status': 'Encoder probes only; no native parity or model-quality claim',
              'molbal': provenance,
              'runtime': {'numpy': np.__version__, 'gguf': version('gguf'), 'byteorder': sys.byteorder,
                          'decoder_sha256': hashlib.sha256(decoder_file.read_bytes()).hexdigest()},
              'results': []}
    for name in ('Q8_0', 'Q4_0', 'Q4_1', 'Q5_0', 'Q5_1', 'Q4_K', 'Q5_K', 'Q6_K'):
        qtype = gguf.GGMLQuantizationType[name]
        encoder = k_encoder if name.endswith('_K') else gguf.quantize
        block, _ = gguf.GGML_QUANT_SIZES[qtype]
        for label, data in reference_vectors().items():
            row = {'encoder': 'molbal' if name.endswith('_K') else 'installed-gguf-py',
                   'type': name, 'vector': label, 'shape': list(data.shape),
                   'input_sha256': hashlib.sha256(data.tobytes()).hexdigest()}
            if data.shape[-1] % block:
                try:
                    encoder(data.copy(), qtype)
                except (ValueError, gguf.quants.QuantError) as error:
                    row.update(status='rejected_incompatible_row', error=type(error).__name__)
                else:
                    raise ValueError(f'{name} accepted incompatible rows: {data.shape}')
            else:
                row.update(measure_encoding(encoder, data, qtype))
            report['results'].append(row)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(run_vectors(), indent=2, allow_nan=False) + '\n', encoding='utf-8')
