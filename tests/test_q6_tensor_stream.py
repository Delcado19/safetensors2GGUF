"""Streaming storage safety and full-matrix comparison without installed models."""
import hashlib
import json
from pathlib import Path
import threading

import numpy as np
import pytest
from safetensors.torch import save_file
import torch

from convert import ConversionCancelled
from scripts import check_q6_tensor_stream as streaming
from scripts.reference_q6_k import quantize_q6_k


@pytest.fixture
def oracle(monkeypatch):
    """Stub native execution only; native byte parity has separate golden tests."""
    monkeypatch.setattr(streaming, 'verified_native_builds', lambda report: [
        dict(id=name, executable=name, commit='test', archive_sha256='test', executable_sha256='test')
        for name in ('old', 'installed', 'new')])
    monkeypatch.setattr(streaming, 'native_encode', lambda executable, data, qtype: (quantize_q6_k(data), 0.0))


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
def test_full_tensor_stream_matches_array_bytes_and_preserves_source(tmp_path, oracle, dtype):
    """Source reconstruction is explicit and row chunks cover the final short batch."""
    source, output = tmp_path / 'source.safetensors', tmp_path / 'tensor.q6k'
    data = torch.linspace(-1, 2, 129 * 768).reshape(129, 768).to(dtype)
    save_file({'weight': data}, str(source))
    before = source.read_bytes()
    expected = quantize_q6_k(data.float().numpy()).tobytes()
    progress = []
    result = streaming.stream_tensor(source, 'weight', output, {}, on_progress=lambda *args: progress.append(args))
    assert output.read_bytes() == expected
    assert result['payload_sha256'] == hashlib.sha256(expected).hexdigest()
    assert result['chunks'] == 2 and result['native_comparisons'] == 6
    assert result['max_source_chunk_values'] <= 65536
    assert result['all_equal'] and result['rmse'] < 0.04
    assert progress == [(85, 129), (129, 129)]
    assert source.read_bytes() == before
    assert len(list(tmp_path.iterdir())) == 2


@pytest.mark.parametrize('failure', ['cancel', 'nan', 'mismatch', 'source_change', 'io_error', 'readback'])
def test_late_failures_never_publish_partial_tensor(tmp_path, oracle, monkeypatch, failure):
    """Every failure after at least one written chunk removes the temporary file."""
    source, output = tmp_path / 'source.safetensors', tmp_path / 'tensor.q6k'
    data = torch.ones(257, 256)
    if failure == 'nan':
        data[-1, -1] = float('nan')
    save_file({'weight': data}, str(source))
    event = threading.Event()

    def progress(done, total):
        if failure == 'cancel':
            event.set()
        elif failure == 'source_change':
            import os
            stat = source.stat()
            os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000))
        elif failure == 'mismatch':
            monkeypatch.setattr(streaming, 'native_encode', lambda *args: (np.zeros((1, 210), dtype=np.uint8), 0.0))
        elif failure == 'io_error':
            def fail(*args, **kwargs):
                raise OSError('injected I/O error')
            monkeypatch.setattr(streaming, 'quantize_q6_k', fail)
        elif failure == 'readback':
            temporary = next(tmp_path.glob('.tensor.q6k.*.q6k'))
            with temporary.open('r+b') as damaged:
                damaged.write(b'\xff')

    with pytest.raises((ValueError, ConversionCancelled, OSError)):
        streaming.stream_tensor(source, 'weight', output, {}, cancel_event=event, on_progress=progress)
    assert list(tmp_path.iterdir()) == [source]


def test_existing_destination_and_source_alias_are_rejected(tmp_path, oracle):
    """The research writer never overwrites any existing output or original."""
    source, output = tmp_path / 'source.safetensors', tmp_path / 'tensor.q6k'
    save_file({'weight': torch.ones(2, 256)}, str(source))
    output.write_bytes(b'original')
    with pytest.raises(OSError, match='exists'):
        streaming.stream_tensor(source, 'weight', output, {})
    with pytest.raises(ValueError, match='different files'):
        streaming.stream_tensor(source, 'weight', source, {})
    assert output.read_bytes() == b'original'


@pytest.mark.parametrize('tensor', [torch.ones(2, 256, dtype=torch.int8), torch.ones(2, 255), torch.ones(256)])
def test_nonplain_or_unaligned_tensors_do_not_create_output(tmp_path, oracle, tensor):
    """Do not treat stored integer data or incompatible shapes as float weights."""
    source, output = tmp_path / 'source.safetensors', tmp_path / 'tensor.q6k'
    save_file({'weight': tensor}, str(source))
    with pytest.raises(ValueError, match='plain'):
        streaming.stream_tensor(source, 'weight', output, {})
    assert list(tmp_path.iterdir()) == [source]


def test_quantized_marker_rejects_source_before_export(tmp_path, oracle):
    """A floating-point storage dtype alone cannot establish plain source semantics."""
    source = tmp_path / 'source.safetensors'
    save_file({'weight': torch.ones(2, 256), 'weight.comfy_quant': torch.ones(1)}, str(source))
    with pytest.raises(ValueError, match='reconstruction policy'):
        streaming.stream_tensor(source, 'weight', tmp_path / 'tensor.q6k', {})
    assert list(tmp_path.iterdir()) == [source]


def test_recorded_full_tensor_evidence_has_complete_chunk_coverage():
    """Retained live evidence must describe whole matrices, not sampled rows."""
    reports = [json.loads(path.read_text()) for path in
               sorted((Path(__file__).parents[1] / 'data').glob('q6-stream-*.json'))]
    assert len(reports) == 3
    assert sum(row['native_comparisons'] for row in reports) == 912
    assert sum(np.prod(row['shape']) for row in reports) == 19_808_256
    assert {row['source_dtype'] for row in reports} == {'F16', 'BF16'}
    for row in reports:
        rows, columns = row['shape']
        batch = max(1, 65536 // columns)
        assert row['chunks'] == (rows + batch - 1) // batch
        assert row['native_comparisons'] == row['chunks'] * 3
        assert row['output_bytes'] == rows * columns // 256 * 210
        assert row['all_equal'] and len(row['references']) == 3
