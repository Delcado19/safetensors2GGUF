"""Native golden payloads and edge/stream contracts for research Q5_K math."""
import hashlib
import json
from pathlib import Path

import gguf
import numpy as np
import pytest

from scripts.check_q6_pilot import check_pilot, pilot_vectors
from scripts.reference_q5_k import iter_q5_k, quantize_q5_k


@pytest.mark.parametrize('chunk_blocks', [1, 3, 256])
def test_synthetic_vectors_match_native_golden_payloads(chunk_blocks):
    """Replay measured reference hashes; unit tests do not need native binaries."""
    root = Path(__file__).parents[1]
    report = json.loads((root / 'data/q5-pilot-comparison.json').read_text())
    rows = {r['vector']: r for r in report['results']}
    for name, data in pilot_vectors():
        actual = quantize_q5_k(data, chunk_blocks=chunk_blocks)
        assert actual.dtype == np.uint8
        assert actual.shape == (*data.shape[:-1], data.shape[-1] // 256 * 176)
        assert hashlib.sha256(actual).hexdigest() == rows[name]['payload_sha256'], name
        assert all(r['equal_bytes'] for r in rows[name]['references'])
    assert len(list(pilot_vectors())) == 145 and report['all_equal']
    assert report['pilot_source_sha256'] == hashlib.sha256((root / 'scripts/reference_q5_k.py').read_bytes()).hexdigest()
    samples = [r for r in report['results'] if 'model_sample' in r]
    assert len(samples) == 21 and len({r['model_sample']['family'] for r in samples}) == 7
    assert all(r['shape'] == [2, 256] and all(ref['equal_bytes'] for ref in r['references']) for r in samples)


def test_positive_ramp_and_signed_zero_headers_are_correct():
    """Avoid the frozen Molbal clipping bug and preserve native positive-zero minima."""
    data = np.linspace(1, 2, 512, dtype=np.float32).reshape(2, 256)
    decoded = gguf.dequantize(quantize_q5_k(data), gguf.GGMLQuantizationType.Q5_K)
    assert np.sqrt(np.mean((decoded - data) ** 2)) < 0.02
    for value in (0, 0.75, 1e-6):
        encoded = quantize_q5_k(np.full((1, 256), value, dtype=np.float32))
        assert not encoded[0, 2:4].any()


def test_strided_stream_is_bounded_and_does_not_mutate_input(monkeypatch):
    """Copy each logical window only and return independently owned packed chunks."""
    from scripts import reference_q5_k as codec
    original = np.linspace(-2, 2, 5 * 3 * 512, dtype=np.float32).reshape(5, 3, 512)
    before = original.tobytes()
    original.flags.writeable = False
    data = original.transpose(1, 0, 2)[::-1, :, ::-1]
    expected = quantize_q5_k(np.ascontiguousarray(data))
    encode = codec._encode
    sizes = []
    def bounded(chunk):
        sizes.append(chunk.size)
        assert chunk.size <= 7 * 256
        return encode(chunk)
    monkeypatch.setattr(codec, '_encode', bounded)
    chunks = list(iter_q5_k(data, chunk_blocks=7))
    assert sizes == [7 * 256] * 4 + [2 * 256]
    assert not np.shares_memory(chunks[0], chunks[1])
    assert np.array_equal(np.concatenate(chunks).reshape(expected.shape), expected)
    assert original.tobytes() == before


@pytest.mark.parametrize('limit', [0, -1, 1.5, '2', True, np.bool_(False)])
def test_invalid_chunk_limits_fail(limit):
    with pytest.raises(ValueError, match='positive integer'):
        list(iter_q5_k(np.zeros(256, dtype=np.float32), chunk_blocks=limit))


@pytest.mark.parametrize('data', [np.zeros(256, dtype=np.float16), np.zeros(256, dtype=np.float64),
                                np.zeros(256, dtype=np.int32), np.array(0, dtype=np.float32),
                                np.zeros((0, 256), dtype=np.float32), np.zeros((1, 0), dtype=np.float32),
                                np.zeros((256, 2), dtype=np.float32)])
def test_invalid_dtype_and_shapes_fail(data):
    with pytest.raises(ValueError, match='nonempty float32 rows'):
        quantize_q5_k(data)


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf, np.finfo(np.float32).max])
def test_invalid_or_overflowing_values_fail(value):
    with pytest.raises(ValueError):
        quantize_q5_k(np.full(256, value, dtype=np.float32))


def test_early_close_and_late_failure_restore_numpy_state():
    """Cancellation does not encode later chunks or leave a global error mode active."""
    data = np.zeros((3, 256), dtype=np.float32)
    data[-1, -1] = np.nan
    errors = np.geterr().copy()
    stream = iter_q5_k(data, chunk_blocks=1)
    assert not next(stream).any()
    assert np.geterr() == errors
    stream.close()
    stream = iter_q5_k(data, chunk_blocks=1)
    next(stream)
    next(stream)
    with pytest.raises(ValueError, match='finite input'):
        next(stream)
    assert np.geterr() == errors


def test_unsupported_pilot_format_is_explicitly_rejected():
    with pytest.raises(ValueError, match='Q5_K or Q6_K'):
        check_pilot({}, format_key='IQ2_XXS')


@pytest.mark.parametrize('alias', [False, True])
def test_checker_rejects_existing_report_before_native_work(tmp_path, alias):
    """New format support must not overwrite reports, model files or its reference input."""
    import subprocess
    import sys
    output = tmp_path / 'retained.json'
    output.write_bytes(b'original')
    reference = output if alias else tmp_path / 'missing-reference.json'
    result = subprocess.run([sys.executable, '-m', 'scripts.check_q6_pilot', '--format', 'Q5_K',
                             '--references', str(reference), '--output', str(output)],
                            capture_output=True, text=True,
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    assert result.returncode != 0
    assert ('different files' if alias else 'overwrite is disabled') in result.stderr
    assert output.read_bytes() == b'original'
