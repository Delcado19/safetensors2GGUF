"""Load a FLUX.3 GGUF with pinned Molbal and installed ComfyUI on CPU, without inference.

Run with the installed ComfyUI Python. All paths must be supplied explicitly;
the backend is the extracted, verified third_party/molbal-flux3 source snapshot.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import types


def main():
    """Check complete logical tensor shapes and native parameters, then record hashes."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('comfy', 'backend', 'source', 'gguf', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a fresh report path')
    sys.argv = [sys.argv[0], '--cpu']
    sys.path.insert(0, str(args.comfy.resolve()))
    package = types.ModuleType('flux3_validation_backend')
    package.__path__ = [str(args.backend.resolve())]
    sys.modules[package.__name__] = package
    loader = importlib.import_module(package.__name__ + '.loader')
    ops = importlib.import_module(package.__name__ + '.ops')
    import comfy.sd
    import torch
    from safetensors import safe_open

    state, extra = loader.gguf_sd_loader(str(args.gguf))
    assert extra['arch_str'] == 'flux3', 'Expected FLUX.3 GGUF'
    with safe_open(args.source, framework='pt') as checkpoint:
        shapes = {key: list(checkpoint.get_slice(key).get_shape()) for key in checkpoint.keys()}
    loaded = len(state)
    assert set(state) == set(shapes), 'Loader tensor names differ from source'
    assert all(list(state[key].shape) == shape for key, shape in shapes.items()), 'Loader shape mismatch'
    model = comfy.sd.load_diffusion_model_state_dict(
        state, model_options={'custom_operations': ops.GGMLOps(), 'dtype': torch.bfloat16},
        metadata=extra.get('metadata', {}))
    assert model is not None, 'Native model detection failed'
    # Molbal state_dict emits packed meta estimates; compare actual parameters.
    native = dict(model.model.diffusion_model.named_parameters())
    # ComfyUI's inherited Flux processor migrates RMSNorm scale to weight.
    shapes = {key[:-6] + '.weight' if key.endswith('_norm.scale') else key: shape
              for key, shape in shapes.items()}
    missing, unexpected = sorted(set(shapes) - set(native)), sorted(set(native) - set(shapes))
    mismatch = [key for key in shapes if key in native and list(native[key].shape) != shapes[key]]
    assert not missing and not unexpected and not mismatch, (missing, unexpected, mismatch)

    def digest(path):
        result = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                result.update(chunk)
        return result.hexdigest()

    report = {
        'scope': 'Complete Molbal FLUX.3 GGUF loading and native ComfyUI CPU construction; no forward pass or render.',
        'source': str(args.source), 'source_bytes': args.source.stat().st_size,
        'source_sha256': digest(args.source), 'output': str(args.gguf),
        'output_bytes': args.gguf.stat().st_size, 'output_sha256': digest(args.gguf),
        'architecture': extra['arch_str'], 'source_tensors': len(shapes), 'loaded_tensors': loaded,
        'native_state_tensors': len(native), 'missing': missing, 'unexpected': unexpected,
        'shape_mismatches': mismatch, 'native_class': type(model.model).__name__,
        'device': 'cpu', 'inference_tested': False,
    }
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf8')
    print(json.dumps(report, indent=2))
    # Dispose before ComfyUI's interpreter-shutdown callbacks disappear.
    del model


if __name__ == '__main__':
    main()
