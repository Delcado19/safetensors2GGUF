"""Isolated adapters over the complete frozen Molbal snapshot, without tool downloads."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

import gguf
import numpy as np
import torch

from conversion_io import atomic_output, iter_process_output, validate_output
from convert import ConversionCancelled
from scripts.gguf_research_io import open_gguf

ROOT = Path(__file__).resolve().parent
EXTENDED_ARCHES = {'ideogram', 'ltxv_upscaler', 'minimax_h3', 'minimax_h3_vae',
                   'minimax_music3', 'qwen_image21'}
FORMATS = ('F16', 'BF16', 'Q8_0', 'Q5_1', 'Q5_0', 'Q4_1', 'Q4_0',
           'Q2_K', 'Q3_K_M', 'Q4_K_S', 'Q4_K_M', 'Q5_K_S', 'Q5_K_M', 'Q6_K',
           'Q8_CR', 'Q4_CR', 'Q4_CR_W4A4')
CHOICES = [(key, key) for key in FORMATS]
NOTE = 'Requires the matching molbal/ComfyUI architecture loader; export checks do not establish inference quality.'
_PROGRESS = 'S2G_EXT_PROGRESS '


def validate_request(source, format_key, arch=None):
    """Validate format contracts before any extraction or model payload work."""
    if Path(source).suffix.lower() != '.safetensors' or format_key not in FORMATS:
        raise ValueError('Extended GGUF requires safetensors and an explicitly supported format')
    if arch == 'qwen3vl_text_only' and format_key != 'Q8_CR':
        raise ValueError('The Krea text-only Qwen3-VL adapter requires explicit Q8_CR')


def convert_extended(source, destination, format_key, *, arch=None, overwrite=False,
                     cancel_event=None, on_log=None, on_progress=None):
    """Run the pinned backend in isolation and publish only readable complete output."""
    validate_request(source, format_key, arch)
    from bundled_tools import validate_text_destination
    validate_text_destination(destination)
    validate_output(source, destination, overwrite)
    references = ROOT / 'third_party/converter_references'
    entry = next(row for row in json.loads((references / 'sources.json').read_text())['sources']
                 if row['id'] == 'molbal-current')
    archive = references / entry['archive']
    with archive.open('rb') as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != entry['sha256']:
        raise ValueError('Extended converter source archive checksum mismatch')
    before = Path(source).stat()
    if on_log:
        on_log(f"INFO: Frozen molbal {entry['commit']}; CPU only. {NOTE}")
    with tempfile.TemporaryDirectory(prefix='extended-', dir=Path(destination).resolve().parent) as scratch:
        with tarfile.open(archive) as packed:
            for member in packed:
                if not member.isfile():
                    continue
                target = Path(scratch).joinpath(*Path(member.name).parts)
                if not target.resolve().is_relative_to(Path(scratch).resolve()):
                    raise ValueError('Unsafe extended converter source member')
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(packed.extractfile(member).read())
        backend = Path(scratch) / entry['archive_root'] / 'tools/convert.py'
        with atomic_output(destination, overwrite) as temporary:
            command = [sys.executable, '-u', str(Path(__file__).resolve()), str(backend),
                       str(Path(source).resolve()), temporary, format_key, arch or '']
            proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding='utf8', errors='replace', cwd=scratch,
                                    env=os.environ | {'PYTHONIOENCODING': 'utf8',
                                                     'TMPDIR': scratch, 'TEMP': scratch, 'TMP': scratch},
                                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            tail = []
            for line in iter_process_output(proc, cancel_event, ConversionCancelled):
                line = line.rstrip()
                if line.startswith(_PROGRESS):
                    phase, index, total = json.loads(line[len(_PROGRESS):])
                    if on_progress:
                        on_progress(index, total, phase)
                else:
                    tail = (tail + [line])[-15:]
                    if on_log:
                        on_log(line)
            if proc.returncode:
                raise RuntimeError('Extended conversion failed: ' + '\n'.join(tail))
            with open_gguf(temporary) as reader:
                if not reader.tensors:
                    raise ValueError('Extended converter produced an empty GGUF')
            after = Path(source).stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError('Source changed during extended conversion')
            if cancel_event is not None and cancel_event.is_set():
                raise ConversionCancelled()
    return str(destination)


def _convrot8(weight, convrot_groupsize=256, device=None):
    """Match the pinned F32 per-row codec using the project's owned Kitchen-compatible math."""
    from safetensors_quant_int8 import _build_hadamard, _rotate_weight
    weight = weight.to(device=device or 'cpu', dtype=torch.float32)
    shape = tuple(weight.shape)
    group = next((size for size in (convrot_groupsize, 64, 16, 4) if weight.shape[1] % size == 0), None)
    if group:
        weight = _rotate_weight(weight, _build_hadamard(group, weight.device, weight.dtype), group)
    scale = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-9) / 127
    data = (weight / scale).round().clamp(-128, 127).to(torch.int8)
    conf = dict(format='int8_tensorwise', convrot=group is not None,
                weight_rotated=group is not None, per_row=True)
    if group:
        conf['convrot_groupsize'] = group
    return data, scale, conf, shape


def _worker(backend, source, destination, format_key, expected_arch):
    spec = importlib.util.spec_from_file_location('molbal_extended', backend)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    upstream.quantize_int8_convrot = _convrot8
    # Replace the known clipping NumPy K encoders, retaining frozen architecture policy.
    if format_key.startswith(('Q2_', 'Q3_', 'Q4_K', 'Q5_K', 'Q6_K')):
        from scripts.compiled_kernel_package import load_package, _platform
        host = _platform()
        kernel, _ = load_package(ROOT / 'third_party/cpu-kernels' / (host['system'] + '-' + host['machine']))
        def encode(data, qtype):
            shape = data.shape
            values = np.asarray(data, dtype=np.float32).reshape(-1)
            size = gguf.GGML_QUANT_SIZES[qtype][1]
            output = np.empty(values.size // 256 * size, dtype=np.uint8)
            for start in range(0, values.size, 65536):
                block = kernel(values[start:start + 65536].reshape(-1, 256), int(qtype.name[1]))
                offset = start // 256 * size
                output[offset:offset + block.size] = block.reshape(-1)
            return output.reshape(*shape[:-1], shape[-1] // 256 * size)
        upstream.quantize_k = encode
        if format_key in ('Q2_K', 'Q3_K_M'):
            qtype = gguf.GGMLQuantizationType[format_key[:4]]
            upstream.QUANT_TYPE_MAP[format_key] = (qtype, getattr(gguf.LlamaFileType, 'MOSTLY_' + format_key))
            upstream.K_QUANT_TYPES |= {qtype, gguf.GGMLQuantizationType.Q3_K}
            upstream.K_QUANT_SELECTIONS |= {format_key}
            original_plan = upstream.plan_k_quantization
            def low_bit_plan(state, model, selection):
                plan = original_plan(state, model, 'Q4_K')
                # Project low-bit extension: frozen protections stay intact; sensitive
                # V/QKV/down roles receive one additional bit, in sorted source order.
                for name, target in plan.items():
                    if target == gguf.GGMLQuantizationType.Q4_K:
                        category = upstream._k_quant_category(name)
                        promoted = category in ('attention_v', 'attention_qkv', 'ffn_down')
                        plan[name] = gguf.GGMLQuantizationType[f'Q{int(qtype.name[1]) + int(promoted)}_K']
                return plan
            upstream.plan_k_quantization = low_bit_plan
    layout, _ = upstream._streamed_safetensors_layout(source)
    actual = upstream.detect_arch(layout).arch
    wanted = 'qwen3vl' if expected_arch == 'qwen3vl_text_only' else expected_arch
    if wanted and actual != wanted:
        raise ValueError(f'Extended architecture mismatch: expected {wanted}, got {actual}')
    def progress(phase, index, total):
        print(_PROGRESS + json.dumps([phase, index, total]), flush=True)
    upstream.convert_file(source, dst_path=destination, interact=False, overwrite=True,
                          quant_type_name=format_key, streamed=True, quantization_device='cpu',
                          progress_callback=progress)
    with open_gguf(destination) as reader:
        if reader.get_field('general.architecture').contents() != actual:
            raise ValueError('Extended output architecture mismatch')
        if wanted == 'qwen3vl':
            marker = reader.get_field('comfy.gguf.qwen3vl.text_only')
            if marker is None or not marker.contents():
                raise ValueError('Text-only VL output must be explicitly marked')


if __name__ == '__main__':
    _worker(*sys.argv[1:])
