"""Verify official SDXL files and print reproducible normalized stock fingerprints.

Read-only: uv run python scripts/check_stock_components.py I:/
No network, downloads or model-file writes. Verify file SHA256 before hashing.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from component_extract import STOCK_COMPONENTS, _component_hash, _tensor_hash, _vae_mapping  # noqa: E402

SOURCES = {
    'vae': ('diffusion_pytorch_model.safetensors', 'vae/diffusion_pytorch_model.safetensors',
            '1598f3d24932bcfe6634e8b618ea1e30ab1d57f5aad13a6d2de446d2199f2341'),
    'clip_l': ('clip-l.safetensors', 'text_encoder/model.safetensors',
               '5c3d6454dd2d23414b56aa1b5858a72487a656937847b6fea8d0606d7a42cdbc'),
    'clip_g': ('clip-g.safetensors', 'text_encoder_2/model.safetensors',
               '3a6032f63d37ae02bbc74ccd6a27440578cd71701f96532229d0154f55a8d3ff'),
}


def main():
    """Emit provenance and FP32/original plus FP16/BF16 cast hashes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--verify', action='store_true', help='Fail if the shipped registry differs')
    args = parser.parse_args()
    manifest = {}
    for component, (filename, repo_path, expected) in SOURCES.items():
        path = args.directory / filename
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f'{component}: file SHA256 does not match the official source')
        records = {name: {} for name in ('FP32', 'FP16', 'BF16')}
        with safe_open(str(path), framework='pt', device='cpu') as source:
            mapping = _vae_mapping(source.keys()) if component == 'vae' else {
                key: (key, lambda t: t) for key in source.keys()}
            for key, (source_key, transform) in mapping.items():
                tensor = transform(source.get_tensor(source_key))
                if tensor.dtype != torch.float32:
                    raise ValueError(f'{component}: expected official FP32 tensors')
                for name, dtype in (('FP32', torch.float32), ('FP16', torch.float16), ('BF16', torch.bfloat16)):
                    records[name][key] = _tensor_hash(tensor.to(dtype))
        manifest[component] = {
            'source_url': f'https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/{repo_path}',
            'source_file_sha256': expected,
            'tensor_count': len(mapping),
            'hash_scheme': 'normalized-tensors-v1',
            'variants': [{'label': f'SDXL 1.0 {component.replace("_", "-").upper()} {name}' + (' cast' if name != 'FP32' else ''),
                          'component_hash': _component_hash(values)} for name, values in records.items()],
        }
    if args.verify:
        if manifest != STOCK_COMPONENTS:
            raise ValueError('The shipped stock registry does not match the verified files')
        print('Verified official file SHA256 and all nine stock/cast fingerprints.')
    else:
        print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
