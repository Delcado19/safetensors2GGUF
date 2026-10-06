"""Offline pinned molbal adapter for Krea diffusion GGUF conversion.

Upstream runs in a disposable subprocess: its imports and direct writer must not
affect the application or publish a partial destination. No ComfyUI install is
changed. Synthetic export validation is not render compatibility evidence.
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

import gguf

from conversion_io import atomic_output, iter_process_output, validate_output
from convert import ConversionCancelled
from scripts.verify_vendor_sources import verify_snapshot

VENDOR = Path(__file__).resolve().parent / 'third_party' / 'molbal'
KREA_FORMATS = ('F16', 'BF16', 'Q8_0', 'Q5_1', 'Q5_0', 'Q4_1', 'Q4_0')
KREA_CHOICES = [(key, key) for key in KREA_FORMATS]
LOADER_NOTE = ('Requires a Krea-capable ComfyUI core and molbal GGUF loader. '
               'Q4_0 has a two-prompt render test on one checkpoint with visible '
               'drift versus its INT8 source. Other checkpoints/precisions, LoRA '
               'and forced offload validation remain pending.')
_PROGRESS_PREFIX = 'KREA_PROGRESS '


def _check_output(path, expected=None):
    # Retain the instance even when __init__ raises: GGUFReader has no close API
    # and an exception traceback otherwise keeps a corrupt file mapped on Windows.
    reader = gguf.GGUFReader.__new__(gguf.GGUFReader)
    try:
        reader.__init__(str(path))
        field = reader.get_field('general.architecture')
        if not field or field.contents() != 'krea2' or not reader.tensors:
            raise ValueError('Molbal output is not a nonempty Krea GGUF.')
        if len({tensor.name for tensor in reader.tensors}) != len(reader.tensors):
            raise ValueError('Molbal output contains duplicate tensor names.')
        actual = {tensor.name: tuple(int(dim) for dim in reversed(tensor.shape)) for tensor in reader.tensors}
        if expected is not None and actual != expected:
            raise ValueError('Krea GGUF tensor names/shapes do not match the normalized source.')
    finally:
        if hasattr(reader, 'data'):
            reader.data._mmap.close()


def validate_krea_request(source, format_key):
    """Reject unsupported source containers/types before invoking upstream."""
    if format_key not in KREA_FORMATS:
        raise ValueError('Krea GGUF supports only ' + ', '.join(KREA_FORMATS) +
                         '; choose explicitly. K-quants and ConvRot are not enabled.')
    if Path(source).suffix != '.safetensors':
        raise ValueError('The Krea adapter requires a .safetensors source.')


def convert_krea(source, destination, format_key, *, overwrite=False,
                 on_log=None, on_progress=None, cancel_event=None):
    """Stream a Krea source through the pinned converter and publish atomically."""
    validate_krea_request(source, format_key)
    validate_output(source, destination, overwrite)
    if cancel_event is not None and cancel_event.is_set():
        raise ConversionCancelled()
    provenance = verify_snapshot(VENDOR)
    manifest = json.loads((VENDOR / 'source.json').read_text(encoding='utf-8'))
    if on_log:
        on_log(f"INFO: molbal backend {provenance['commit']} / {format_key}. {LOADER_NOTE}")
    # Fresh verified extraction avoids trusting an editable persistent code cache.
    with tempfile.TemporaryDirectory(prefix='krea-' + provenance['commit'][:8] + '-') as scratch:
        with tarfile.open(VENDOR / manifest['archive']) as archive:
            # verify_snapshot allows only rooted regular files/directories.
            for member in archive.getmembers():
                target = Path(scratch).joinpath(*Path(member.name).parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archive.extractfile(member).read())
        backend = Path(scratch) / manifest['archive_root'] / 'tools' / 'convert.py'
        with atomic_output(destination, overwrite) as temporary:
            command = [sys.executable, '-u', str(Path(__file__).resolve()),
                       str(backend), str(Path(source).resolve()), temporary, format_key]
            proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding='utf-8', errors='replace',
                                    cwd=scratch, env=os.environ | {'PYTHONIOENCODING': 'utf-8',
                                                                 'TMPDIR': scratch, 'TEMP': scratch, 'TMP': scratch},
                                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            tail = []
            for line in iter_process_output(proc, cancel_event, ConversionCancelled):
                line = line.rstrip()
                if line.startswith(_PROGRESS_PREFIX):
                    phase, index, total = json.loads(line[len(_PROGRESS_PREFIX):])
                    if on_progress:
                        on_progress(index, total, phase)
                else:
                    tail = (tail + [line])[-12:]
                    if on_log:
                        on_log(line)
            if proc.returncode:
                raise RuntimeError(f'Molbal Krea conversion failed ({proc.returncode}): ' + '\n'.join(tail))
            _check_output(temporary)
            if cancel_event is not None and cancel_event.is_set():
                raise ConversionCancelled()
    return str(destination)


def _worker(backend, source, destination, format_key):
    """Run unmodified upstream with explicit arguments and machine-readable progress."""
    validate_krea_request(source, format_key)
    spec = importlib.util.spec_from_file_location('molbal_convert', backend)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    layout, _ = upstream._streamed_safetensors_layout(source)
    if upstream.detect_arch(layout).arch != 'krea2':
        raise ValueError('The molbal adapter accepts only Krea diffusion models.')
    # The streamed upstream path does not restore native INT8/ConvRot/NVFP4
    # sources. Reject rather than treating packed values as ordinary weights.
    allowed = {'torch.float32', 'torch.float16', 'torch.bfloat16',
               'torch.float8_e4m3fn', 'torch.float8_e5m2'}
    for key, tensor in layout.items():
        if key.endswith('.comfy_quant'):
            raise ValueError('Native quantized sources are not enabled; use unquantized or scalar-scaled FP8 Krea weights.')
        if str(tensor.dtype) not in allowed or tensor.numel() == 0:
            raise ValueError(f'Unsupported Krea source tensor: {key} ({tensor.dtype}, {tuple(tensor.shape)})')
    def progress(phase, index, total):
        print(_PROGRESS_PREFIX + json.dumps([phase, index, total]), flush=True)
    upstream.convert_file(source, dst_path=destination, interact=False,
                          overwrite=True, quant_type_name=format_key, streamed=True,
                          quantization_device='cpu', progress_callback=progress)
    expected = {key: tuple(tensor.shape) for key, tensor in layout.items() if tensor.ndim > 0}
    _check_output(destination, expected)


if __name__ == '__main__':
    _worker(*sys.argv[1:])
