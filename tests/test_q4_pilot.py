"""Native golden parity and streaming contracts for shared Q4_K mathematics."""
import hashlib
import json
from pathlib import Path

import gguf
import numpy as np
import pytest

from scripts.check_q6_pilot import pilot_vectors
from scripts.reference_q4_k import iter_q4_k, quantize_q4_k


@pytest.mark.parametrize('chunk_blocks', [1, 3, 256])
def test_q4_native_goldens_and_shared_source_provenance(chunk_blocks):
    """Replay all synthetic hashes while retaining seven-family native evidence."""
    root = Path(__file__).parents[1]
    r = json.loads((root / 'data/q4-pilot-comparison.json').read_text())
    rows = {row['vector']: row for row in r['results']}
    assert r['all_equal'] and len(rows) == 166
    for name, data in pilot_vectors():
        encoded = quantize_q4_k(data, chunk_blocks=chunk_blocks)
        assert encoded.shape == (*data.shape[:-1], data.shape[-1] // 256 * 144)
        assert hashlib.sha256(encoded).hexdigest() == rows[name]['payload_sha256'], name
    assert sum(len(row['references']) for row in rows.values()) == 498
    assert all(ref['equal_bytes'] for row in rows.values() for ref in row['references'])
    assert all(check['equal_bytes'] for row in rows.values() for check in row['chunk_checks'])
    samples = [row for row in rows.values() if 'model_sample' in row]
    assert len(samples) == 21 and len({row['model_sample']['family'] for row in samples}) == 7
    for name, digest in r['implementation_source_sha256'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest


def test_q4_positive_ramp_and_strided_incremental_validation():
    """Avoid saturation, preserve logical order and stop before a later invalid chunk."""
    x = np.linspace(1, 2, 1024, dtype=np.float32).reshape(2, 512)[:, ::-1]
    before = x.tobytes()
    decoded = gguf.dequantize(quantize_q4_k(x), gguf.GGMLQuantizationType.Q4_K)
    assert np.sqrt(np.mean((decoded - x) ** 2)) < .04
    assert x.tobytes() == before
    chunks = list(iter_q4_k(x, chunk_blocks=1))
    assert len(chunks) == 4 and all(chunk.shape == (1, 144) for chunk in chunks)
    assert not np.shares_memory(chunks[0], chunks[1])
    assert np.concatenate(chunks).tobytes() == quantize_q4_k(x).tobytes()
    bad = np.zeros((2, 256), dtype=np.float32)
    bad[-1, -1] = np.nan
    errors = np.geterr().copy()
    stream = iter_q4_k(bad, chunk_blocks=1)
    assert not next(stream).any()
    stream.close()
    stream = iter_q4_k(bad, chunk_blocks=1)
    next(stream)
    with pytest.raises(ValueError, match='finite input'):
        next(stream)
    assert np.geterr() == errors


@pytest.mark.parametrize('data', [np.zeros(256, dtype=np.float16), np.zeros(255, dtype=np.float32),
                                 np.zeros((0, 256), dtype=np.float32), np.array(0, dtype=np.float32)])
def test_q4_invalid_inputs_rejected(data):
    with pytest.raises(ValueError, match='nonempty float32'):
        quantize_q4_k(data)


@pytest.mark.parametrize('limit', [0, -1, 1.5, True])
def test_q4_invalid_chunk_sizes(limit):
    with pytest.raises(ValueError, match='positive integer'):
        list(iter_q4_k(np.zeros(256, dtype=np.float32), chunk_blocks=limit))
