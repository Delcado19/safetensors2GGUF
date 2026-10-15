"""Read installed safetensors headers for policy coverage; never load weights."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct
from types import SimpleNamespace

from convert import _build_key_map, _read_safetensors_header
from models.architectures import detect_arch
from text_encoder_convert import detect_text_encoder_family


def inspect_header(path, category):
    """Summarize shape/dtype evidence with existing architecture/family detection."""
    with path.open('rb') as stream:
        prefix = stream.read(8)
    if len(prefix) != 8:
        raise ValueError('Truncated safetensors header prefix')
    length, = struct.unpack('<Q', prefix)
    if length > 64 * 1024 * 1024 or length > path.stat().st_size - 8:
        raise ValueError('Invalid or oversized safetensors header')
    header = _read_safetensors_header(str(path))
    tensors = {name: value for name, value in header.items() if name != '__metadata__'}
    dtypes = Counter(item['dtype'] for item in tensors.values())
    matrices = {name: item for name, item in tensors.items()
                if len(item['shape']) == 2 and name.endswith('.weight') and math.prod(item['shape']) > 1024}
    aligned = {name: item for name, item in matrices.items() if item['shape'][-1] % 256 == 0}
    record = {'bytes': path.stat().st_size, 'header_bytes': length, 'tensor_count': len(tensors),
              'canonical_header_sha256': hashlib.sha256(json.dumps(header, sort_keys=True).encode()).hexdigest(),
              'dtype_tensor_counts': dict(dtypes), 'stored_matrix_count_over_1024': len(matrices),
              'stored_k_row_aligned_matrix_count': len(aligned),
              'stored_k_row_unaligned_examples': list(name for name in matrices if name not in aligned)[:12],
              'has_raw_byte_tensors': any(dtype in dtypes for dtype in ('I8', 'U8')),
              'comfy_quant_tensor_count': sum(name.endswith('.comfy_quant') for name in tensors)}
    try:
        if category == 'text_encoders':
            state = {name: SimpleNamespace(shape=item['shape']) for name, item in tensors.items()}
            record['family'] = detect_text_encoder_family(state)
        else:
            logical = _build_key_map(tensors)
            model = detect_arch(logical)
            record['architecture'] = model.arch
            record['protected_pattern_count'] = len(model.keys_hiprec)
    except (AssertionError, ValueError, RuntimeError) as error:
        record['detection_error'] = str(error)
    return record


def inventory_headers(root):
    """Inspect only checkpoint/diffusion/text safetensors, excluding LoRAs and VAEs."""
    records = []
    for category in ('diffusion_models', 'text_encoders', 'checkpoints'):
        for path in sorted((root / category).rglob('*.safetensors')):
            record = {'path': path.relative_to(root).as_posix(), 'category': category}
            try:
                record.update(inspect_header(path, category))
            except (OSError, ValueError, KeyError, TypeError, struct.error) as error:
                record['inspection_error'] = str(error)
            records.append(record)
    return {'status': 'Header-only stored shapes; not dequantized eligibility, conversion or render validation', 'models': records}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.models_root.is_dir():
        parser.error('Model root must be an existing directory')
    args.output.write_text(json.dumps(inventory_headers(args.models_root), indent=2) + '\n', encoding='utf-8')
