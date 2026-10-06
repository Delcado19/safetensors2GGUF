"""Real pinned exports and atomic failure/cancellation checks without user models."""
import queue
import subprocess
import threading

import gguf
import numpy as np
import pytest
import torch
from safetensors.torch import save_file

import krea_backend
from conversion_service import _pipeline
from convert import ConversionCancelled


def _source(tmp_path):
    source = tmp_path / 'krea.safetensors'
    save_file({'first.weight': torch.ones(64, 64),
               'txtfusion.projector.weight': torch.ones(64, 64),
               'blocks.0.attn.wq.weight': torch.randn(128, 64),
               'blocks.0.txtmlp.up.weight': torch.randn(128, 64),
               'blocks.0.prenorm.scale': torch.ones(64)}, str(source))
    return source


@pytest.mark.parametrize('format_key', krea_backend.KREA_FORMATS)
def test_pinned_krea_exports_actual_types_shapes_and_protection(tmp_path, format_key):
    """Exercise every pinned format and the anchored precision protection rules."""
    source = _source(tmp_path)
    destination = tmp_path / f'{format_key}.gguf'
    events = queue.Queue()
    assert _pipeline(str(source), str(destination), format_key, '', 0, False, False, events) == str(destination)
    reader = gguf.GGUFReader(str(destination))
    try:
        tensors = {tensor.name: tensor for tensor in reader.tensors}
        assert len(tensors) == 5 and reader.get_field('general.architecture').contents() == 'krea2'
        # Anchored protection must not retain every per-block txtmlp tensor as F32.
        assert tensors['first.weight'].tensor_type == gguf.GGMLQuantizationType.F32
        assert tensors['blocks.0.prenorm.scale'].tensor_type == gguf.GGMLQuantizationType.F32
        for key in ('blocks.0.attn.wq.weight', 'blocks.0.txtmlp.up.weight'):
            tensor = tensors[key]
            assert tensor.tensor_type == gguf.GGMLQuantizationType[format_key]
            assert tuple(tensor.shape) == (64, 128)
            restored = gguf.dequantize(tensor.data, tensor.tensor_type)
            assert np.isfinite(restored).all()
        assert any(event[0] == 'progress_frac' for event in list(events.queue))
        assert any('molbal backend' in str(event) for event in list(events.queue))
    finally:
        reader.data._mmap.close()
    assert not list(tmp_path.glob('.*.gguf'))


def test_krea_input_guards_preserve_originals(tmp_path):
    """Wrong formats, source aliases and existing outputs must fail before writing."""
    source = _source(tmp_path)
    original = source.read_bytes()
    destination = tmp_path / 'existing.gguf'
    destination.write_bytes(b'original')
    for target, format_key, overwrite, message in (
        (source, 'Q4_0', True, 'different files'),
        (destination, 'Q4_0', False, 'overwrite is disabled'),
        (destination, 'Q4_K_M', True, 'choose explicitly'),
    ):
        with pytest.raises((ValueError, OSError), match=message):
            krea_backend.convert_krea(source, target, format_key, overwrite=overwrite)
    assert source.read_bytes() == original and destination.read_bytes() == b'original'


@pytest.mark.parametrize('script', ["import sys; sys.exit(3)", "pass",
                                  "import pathlib,sys; pathlib.Path(sys.argv[1]).write_bytes(b'corrupt')"])
def test_krea_failed_or_corrupt_worker_cannot_publish(tmp_path, monkeypatch, script):
    """A failed or zero-exit invalid child cannot replace the user's destination."""
    source = _source(tmp_path)
    destination = tmp_path / 'existing.gguf'
    destination.write_bytes(b'original')
    real_popen = subprocess.Popen
    def child(command, **kwargs):
        return real_popen([command[0], '-c', script, command[-2]], **kwargs)
    monkeypatch.setattr(krea_backend.subprocess, 'Popen', child)
    with pytest.raises(Exception):
        krea_backend.convert_krea(source, destination, 'Q4_0', overwrite=True)
    assert destination.read_bytes() == b'original'
    assert not list(tmp_path.glob('.*.gguf'))


def test_krea_silent_worker_cancellation_is_reaped(tmp_path, monkeypatch):
    """Cancel a silent child and verify process reaping and temporary cleanup."""
    source = _source(tmp_path)
    destination = tmp_path / 'existing.gguf'
    destination.write_bytes(b'original')
    event = threading.Event()
    real_popen = subprocess.Popen
    children = []
    def child(command, **kwargs):
        process = real_popen([command[0], '-c', 'import time; time.sleep(60)'], **kwargs)
        children.append(process)
        event.set()
        return process
    monkeypatch.setattr(krea_backend.subprocess, 'Popen', child)
    with pytest.raises(ConversionCancelled):
        krea_backend.convert_krea(source, destination, 'Q4_0', overwrite=True, cancel_event=event)
    assert children[0].poll() is not None
    assert destination.read_bytes() == b'original'
    assert not list(tmp_path.glob('.*.gguf'))


def test_krea_native_packed_source_is_rejected(tmp_path):
    """Reject already packed weights that upstream streaming cannot reconstruct."""
    source = _source(tmp_path)
    save_file({'first.weight': torch.ones(64, 64),
               'txtfusion.projector.weight': torch.ones(64, 64),
               'blocks.0.attn.wq.weight': torch.ones(64, 64, dtype=torch.int8)}, str(source))
    destination = tmp_path / 'output.gguf'
    with pytest.raises(RuntimeError, match='Unsupported Krea source tensor'):
        krea_backend.convert_krea(source, destination, 'Q4_0')
    assert not destination.exists()


def test_krea_scaled_fp8_source_restores_weight_magnitude(tmp_path):
    """Verify scalar FP8 scales affect weight values and are omitted from GGUF."""
    source = tmp_path / 'fp8.safetensors'
    save_file({'first.weight': torch.ones(64, 64),
               'txtfusion.projector.weight': torch.ones(64, 64),
               'blocks.0.attn.wq.weight': torch.full((64, 64), .5).to(torch.float8_e4m3fn),
               'blocks.0.attn.wq.weight_scale': torch.tensor(2.)}, str(source))
    destination = tmp_path / 'q8.gguf'
    krea_backend.convert_krea(source, destination, 'Q8_0')
    reader = gguf.GGUFReader(str(destination))
    try:
        tensor = next(tensor for tensor in reader.tensors if tensor.name == 'blocks.0.attn.wq.weight')
        assert tensor.tensor_type == gguf.GGMLQuantizationType.Q8_0
        assert np.allclose(gguf.dequantize(tensor.data, tensor.tensor_type), 1., atol=.02)
        assert not any(tensor.name.endswith('_scale') for tensor in reader.tensors)
    finally:
        reader.data._mmap.close()
