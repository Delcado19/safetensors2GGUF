"""Compare shared NumPy Q5_K/Q6_K block pilots against three pinned native encoders."""
import argparse
import hashlib
import json
from pathlib import Path

import gguf
import numpy as np

from conversion_io import atomic_output, validate_output

from scripts.compare_native_reference_vectors import ROOT, NATIVE_SOURCES, native_encode
from scripts.converter_reference_vectors import reference_vectors
from scripts.inventory_converter_sources import verify_references
from scripts.reference_q6_k import quantize_q6_k
from scripts.reference_q5_k import quantize_q5_k


def pilot_vectors():
    """Yield fixed edge cases and reproducible random tensors across eight decades."""
    for label, data in reference_vectors().items():
        if data.shape[-1] % 256 == 0:
            yield label, data
    rng = np.random.default_rng(20261015)
    yield 'fp16_scale_underflow', np.full((1, 256), 1e-6, dtype=np.float32)
    yield 'below_group_epsilon', np.full((1, 256), 1e-16, dtype=np.float32)
    yield 'epsilon_boundary', np.full((1, 256), 1e-15, dtype=np.float32)
    for sign in (1, -1):
        yield f'absolute_max_tie_{sign}', np.tile(np.array([sign, -sign, 0.5, 0], dtype=np.float32), (1, 64))
    mixed = rng.standard_normal((2, 256), dtype=np.float32)
    mixed[0] = 0
    mixed[1, :128] = 0
    yield 'zero_blocks_and_groups', mixed
    yield 'noncontiguous', rng.standard_normal((2, 512), dtype=np.float32)[:, ::2]
    yield 'three_dimensions', rng.standard_normal((2, 2, 256), dtype=np.float32)
    for index in range(128):
        data = rng.standard_normal((2, 256 if index % 2 else 512), dtype=np.float32)
        scale = np.float32(10 ** rng.uniform(-4, 4))
        if index % 3 == 0:
            data = np.abs(data)
        elif index % 3 == 1:
            data = -np.abs(data)
        yield f'random_{index:03}', data * scale


def installed_weight_vectors(root, audit):
    """Read three tiny 2D slices per plain-FP family, preserving installed files."""
    from safetensors import safe_open
    import torch
    from convert import _build_key_map
    from scripts.inventory_installed_model_headers import inspect_header

    vectors, sources, seen = [], {}, set()
    for model in audit['models']:
        family = model.get('architecture') or model.get('family')
        if (not family or family in seen or model.get('has_raw_byte_tensors')
                or model.get('comfy_quant_tensor_count') or not model.get('dtype_tensor_counts')
                or not set(model['dtype_tensor_counts']) <= {'F16', 'BF16', 'F32'}):
            continue
        path = root / model['path']
        before = path.stat()
        current = inspect_header(path, model['category'])
        if current['canonical_header_sha256'] != model['canonical_header_sha256']:
            raise ValueError(f'Model header changed; rerun the installed header audit: {path}')
        count = 0
        with safe_open(str(path), framework='pt', device='cpu') as source:
            # Merged checkpoints contain CLIP/VAE as well; sample their diffusion keys.
            key_map = _build_key_map(source.keys(), strip_prefixes=model['category'] != 'text_encoders')
            for logical in sorted(key_map):
                name = key_map[logical]
                tensor = source.get_slice(name)
                shape = tensor.get_shape()
                if not name.endswith('.weight') or len(shape) != 2 or shape[0] < 2 or shape[1] % 256:
                    continue
                data = tensor[:2, :256].to(torch.float32).numpy().copy()
                label = f'installed:{model["path"]}:{name}'
                vectors.append((label, data))
                sources[label] = {'model': model['path'], 'family': family, 'tensor': name,
                                  'stored_shape': shape, 'slice': [[0, 2], [0, 256]],
                                  'source_dtype': tensor.get_dtype(), 'source_bytes': before.st_size,
                                  'canonical_header_sha256': current['canonical_header_sha256']}
                count += 1
                if count == 3:
                    break
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f'Model changed during sampling: {path}')
        if count:
            seen.add(family)
    return vectors, sources


def verified_native_builds(reference_report):
    """Require all three pinned source archives and unchanged native executables."""
    builds = reference_report['builds']
    expected = {item['id']: item for item in verify_references()['sources'] if item['id'] in NATIVE_SOURCES}
    if len(builds) != 3 or {build['id'] for build in builds} != set(NATIVE_SOURCES):
        raise ValueError('All three pinned native references are required')
    for build in builds:
        source = expected[build['id']]
        if build['commit'] != source['commit'] or build['archive_sha256'] != source['sha256']:
            raise ValueError('Native reference source provenance mismatch')
        path = Path(build['executable'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != build['executable_sha256']:
            raise ValueError('Native reference executable checksum mismatch')
    return builds


def check_pilot(reference_report, vectors=None, *, format_key="Q6_K"):
    """Compare Q5_K/Q6_K exact bytes and retain reconstruction/input evidence."""
    encoders = {'Q6_K': quantize_q6_k, 'Q5_K': quantize_q5_k}
    if format_key not in encoders:
        raise ValueError("Pilot supports Q5_K or Q6_K only")
    encoder = encoders[format_key]
    builds = verified_native_builds(reference_report)
    report = {'status': f'Research {format_key} pilot; no model/preset dispatch or performance claim',
              'format': format_key,
              'numpy': np.__version__, 'random_seed': 20261015,
              'pilot_source_sha256': hashlib.sha256((ROOT / f'scripts/reference_{format_key.lower()}.py').read_bytes()).hexdigest(),
              'references': [{'id': build['id'], 'commit': build['commit'],
                              'archive_sha256': build['archive_sha256'],
                              'executable_sha256': build['executable_sha256']} for build in builds],
              'results': []}
    qtype = gguf.GGMLQuantizationType[format_key]
    for label, data in pilot_vectors() if vectors is None else vectors:
        actual = encoder(data)
        decoded = gguf.dequantize(actual, qtype)
        error = decoded.astype(np.float64) - data.astype(np.float64)
        row = {'vector': label, 'shape': list(data.shape),
               'input_sha256': hashlib.sha256(data.tobytes()).hexdigest(),
               'payload_sha256': hashlib.sha256(actual.tobytes()).hexdigest(),
               'chunk_checks': [{'chunk_blocks': limit,
                                 'equal_bytes': np.array_equal(actual, encoder(data, chunk_blocks=limit))}
                                for limit in (1, 3)],
               'rmse': float(np.sqrt(np.mean(error ** 2))), 'references': []}
        for build in builds:
            expected, difference = native_encode(build['executable'], data.reshape(-1, data.shape[-1]), qtype)
            expected = expected.reshape(actual.shape)
            row['references'].append({'id': build['id'], 'equal_bytes': np.array_equal(actual, expected),
                                      'different_bytes': int(np.count_nonzero(actual != expected)),
                                      'native_decoder_max_difference': difference})
        report['results'].append(row)
    if not report['results']:
        raise ValueError(f'At least one {format_key} vector is required')
    report['all_equal'] = all(item['equal_bytes'] for row in report['results']
                              for item in row['references'] + row['chunk_checks'])
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--format', choices=('Q6_K', 'Q5_K'), default='Q6_K')
    parser.add_argument('--references', type=Path, default=ROOT / 'data/converter-native-comparison.json')
    parser.add_argument('--models-root', type=Path, help='Also sample audited plain floating-point models read-only')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    validate_output(args.references, args.output)
    vectors, sources = list(pilot_vectors()), {}
    if args.models_root:
        audit = json.loads((ROOT / 'data/installed-model-policy-headers.json').read_text(encoding='utf-8'))
        sampled, sources = installed_weight_vectors(args.models_root, audit)
        vectors.extend(sampled)
    report = check_pilot(json.loads(args.references.read_text(encoding='utf-8')), vectors, format_key=args.format)
    for row in report['results']:
        if row['vector'] in sources:
            row['model_sample'] = sources[row['vector']]
    with atomic_output(args.output) as temporary:
        Path(temporary).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n',
                                   encoding='utf-8', newline='\n')
    if not report['all_equal']:
        raise SystemExit(f'{args.format} pilot differs from native reference; inspect the saved report')
