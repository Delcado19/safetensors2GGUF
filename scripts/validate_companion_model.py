"""Validate a complete local original-key export and real CPU ComfyUI construction.

Use the installed ComfyUI Python. No node installation, model download or rendering
is performed; additional model directories exist only in this validation process.
"""
import argparse
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
from time import perf_counter


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _check_warnings(family, warnings):
    # Native SDXL CLIP-L has no learned projection; no other missing weights
    # (especially diffusion weights) may silently qualify a constructed model.
    allowed = {"clip missing: ['text_projection.weight']"} if family == 'clip-l' else set()
    invalid = [line for line in warnings if line not in allowed and
               any(marker in line for marker in ('clip missing:', 'unet missing:', 'unet unexpected:'))]
    if invalid:
        raise ValueError('Native model has missing or unexpected weights: ' + repr(invalid))


def main():
    """Preserve sources and record independent export, tensor and construction evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comfy-root', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--family', required=True)
    parser.add_argument('--clip-type', default='stable_diffusion')
    parser.add_argument('--prefix', default='')
    parser.add_argument('--preset', default='Q4_K_M')
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.work.exists() or args.output.exists():
        parser.error('Choose fresh work and report paths')
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root), str(args.comfy_root)]
    os.environ['HF_HUB_OFFLINE'] = '1'
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
    if args.family == 'ernie_image':
        # Keep both complete 8B source/model copies within this CPU host's RAM.
        comfy.cli_args.args.bf16_unet = True
    import comfy.sd
    import folder_paths
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file, load_file
    from original_text_gguf import export_original_text
    from comfyui_s2gguf import S2GGUFTextEncoderLoader, S2GGUFDiffusionLoader
    from comfyui_s2gguf.codec import load_original_text

    torch.set_num_threads(min(8, os.cpu_count() or 1))
    args.work.mkdir(parents=True)
    before = args.source.stat()
    warnings = []

    class Capture(logging.Handler):
        def emit(self, record):
            warnings.append(record.getMessage())

    logging.getLogger().addHandler(Capture(level=logging.WARNING))
    source = args.source
    if args.prefix:
        with safe_open(source, framework='pt', device='cpu') as checkpoint:
            extracted = {key[len(args.prefix):]: checkpoint.get_tensor(key) for key in checkpoint.keys()
                         if key.startswith(args.prefix)}
        if not extracted:
            raise ValueError('No encoder tensors matched the requested prefix')
        # SDXL stores OpenCLIP-bigG in its original resblock namespace.
        if 'transformer.resblocks.0.ln_1.weight' in extracted:
            extracted = comfy.utils.clip_text_transformers_convert(extracted, '', '')
        source = args.work / 'encoder.safetensors'
        save_file(extracted, str(source))
        del extracted
    output = args.work / (args.family + '-' + args.preset + '.gguf')
    started = perf_counter()
    print('Exporting', source, flush=True)
    export_original_text(source, output, args.family, args.preset, on_log=print)
    export_seconds = perf_counter() - started
    print('Checking complete tensor set', flush=True)
    restored = load_original_text(output)
    with safe_open(source, framework='pt', device='cpu') as checkpoint:
        assert set(restored) == set(checkpoint.keys())
        for key, value in restored.items():
            assert list(value.shape) == checkpoint.get_slice(key).get_shape(), key
            assert torch.isfinite(value).all(), key
            if not value.is_floating_point():
                assert torch.equal(value, checkpoint.get_tensor(key)), key
    vision = [key for key in restored if key.startswith(('model.visual.', 'visual.'))]
    deepstack = [key for key in vision if 'deepstack' in key]
    count = len(restored)
    del restored
    gc.collect()

    def signature(model):
        state = model.model.state_dict() if args.family == 'ernie_image' else model.cond_stage_model.state_dict()
        return {key: list(value.shape) for key, value in state.items()}

    print('Constructing native source baseline on CPU', flush=True)
    state = load_file(str(source))
    if args.family == 'ernie_image':
        baseline = comfy.sd.load_diffusion_model_state_dict(state, model_options={'load_device': torch.device('cpu')})
    else:
        baseline = comfy.sd.load_text_encoder_state_dicts(
            state_dicts=[state], clip_type=comfy.sd.CLIPType[args.clip_type.upper()],
            model_options={'load_device': torch.device('cpu'), 'offload_device': torch.device('cpu')})
    if baseline is None:
        raise ValueError('Native source construction failed')
    expected = signature(baseline)
    baseline_warnings = warnings[:]
    warnings.clear()
    del baseline, state
    gc.collect()
    category = 'diffusion_models' if args.family == 'ernie_image' else 'text_encoders'
    folder_paths.add_model_folder_path(category, str(args.work.resolve()))
    print('Constructing companion model on CPU', flush=True)
    if args.family == 'ernie_image':
        model = S2GGUFDiffusionLoader().load(output.name)[0]
    else:
        model = S2GGUFTextEncoderLoader().load(output.name, args.clip_type, device='cpu')[0]
    actual = signature(model)
    assert actual == expected, 'Native/companion constructed parameter shapes differ'
    _check_warnings(args.family, warnings)
    _check_warnings(args.family, baseline_warnings)
    after = args.source.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    report = {
        'scope': 'Complete local checkpoint export/readback and real CPU model construction; no inference/rendering.',
        'source': str(args.source), 'prefix': args.prefix, 'family': args.family,
        'preset': args.preset, 'clip_type': args.clip_type,
        'source_bytes': before.st_size, 'source_mtime_ns': before.st_mtime_ns, 'source_stat_unchanged': True,
        'source_sha256_after_construction': _digest(args.source), 'output_sha256': _digest(output),
        'output': str(output), 'output_bytes': output.stat().st_size, 'export_seconds': export_seconds,
        'source_tensor_count': count, 'vision_tensor_count': len(vision), 'deepstack_tensor_count': len(deepstack),
        'native_parameter_shapes_equal': True, 'native_parameter_count': len(actual),
        'baseline_warnings': baseline_warnings, 'companion_warnings': warnings,
        'diffusion_parameter_dtypes': (sorted({str(value.dtype) for value in model.model.parameters()})
                                      if args.family == 'ernie_image' else []),
        'comfy_root': str(args.comfy_root),
        'source_code_sha256': {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                               for name in ('original_text_gguf.py', 'comfyui_s2gguf/codec.py', 'comfyui_s2gguf/__init__.py',
                                            'scripts/validate_companion_model.py')},
        'comfy_sd_sha256': hashlib.sha256((args.comfy_root / 'comfy/sd.py').read_bytes()).hexdigest(),
    }
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf8')
    print('PASS', args.output, flush=True)


if __name__ == '__main__':
    main()
