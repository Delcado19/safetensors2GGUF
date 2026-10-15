"""Native golden bytes and input/precision contracts for the shared Q6_K pilot."""
import hashlib
import json
from pathlib import Path

import gguf
import numpy as np
import pytest

from scripts.check_q6_pilot import check_pilot, pilot_vectors
from scripts.reference_q6_k import quantize_q6_k


def test_deterministic_pilot_vectors_match_recorded_native_bytes():
    """Recompute all 145 synthetic vectors; native golden results were measured live."""
    report = json.loads((Path(__file__).parents[1] / 'data/q6-pilot-comparison.json').read_text())
    rows = {row['vector']: row for row in report['results']}
    count = 0
    for label, data in pilot_vectors():
        result = quantize_q6_k(data)
        row = rows[label]
        assert hashlib.sha256(data.tobytes()).hexdigest() == row['input_sha256'], label
        assert hashlib.sha256(result.tobytes()).hexdigest() == row['payload_sha256'], label
        assert all(item['equal_bytes'] and item['different_bytes'] == 0 for item in row['references']), label
        count += 1
    assert count == 145
    assert report['all_equal']


def test_readonly_strided_input_is_preserved_and_nd_layout_decodes():
    """Packing preserves input bytes and leading dimensions, including strided views."""
    original = np.linspace(-1, 1, 2048, dtype=np.float32).reshape(2, 2, 512)
    before = original.tobytes()
    original.flags.writeable = False
    data = original[:, :, ::2]
    encoded = quantize_q6_k(data)
    assert encoded.shape == (2, 2, 210)
    assert encoded.dtype == np.uint8
    decoded = gguf.dequantize(encoded, gguf.GGMLQuantizationType.Q6_K)
    assert decoded.shape == data.shape
    assert np.isfinite(decoded).all()
    assert np.max(np.abs(decoded - data)) < 0.04
    assert original.tobytes() == before


@pytest.mark.parametrize('data', [np.zeros((1, 256), dtype=np.float16),
                               np.zeros((1, 256), dtype=np.float64),
                               np.zeros((1, 256), dtype=np.int32),
                               np.array(0, dtype=np.float32),
                               np.zeros((0, 256), dtype=np.float32),
                               np.zeros((1, 0), dtype=np.float32),
                               np.zeros((1, 255), dtype=np.float32),
                               np.zeros((256, 2), dtype=np.float32)])
def test_invalid_dtype_or_row_shape_is_rejected(data):
    """Reject rather than cast source precision or flatten incompatible rows."""
    with pytest.raises(ValueError, match='nonempty float32 rows'):
        quantize_q6_k(data)


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_nonfinite_values_are_rejected(value):
    """Do not emit apparently valid bytes for invalid source values."""
    with pytest.raises(ValueError, match='finite input'):
        quantize_q6_k(np.full((1, 256), value, dtype=np.float32))


def test_extreme_finite_values_fail_without_silent_overflow():
    """The bounded numerical prototype rejects overflowing scale-search arithmetic."""
    with pytest.raises(ValueError, match='numerical range'):
        quantize_q6_k(np.full((1, 256), np.finfo(np.float32).max, dtype=np.float32))


def test_real_model_evidence_records_slices_not_complete_model_claims():
    """Retained evidence spans seven families and both source floating-point types."""
    report = json.loads((Path(__file__).parents[1] / 'data/q6-pilot-comparison.json').read_text())
    samples = [row for row in report['results'] if 'model_sample' in row]
    assert len(samples) == 21
    assert len({row['model_sample']['family'] for row in samples}) == 7
    assert {row['model_sample']['source_dtype'] for row in samples} == {'F16', 'BF16'}
    for row in samples:
        assert row['shape'] == [2, 256]
        assert row['model_sample']['slice'] == [[0, 2], [0, 256]]
        assert len(row['model_sample']['canonical_header_sha256']) == 64
        assert all(item['equal_bytes'] for item in row['references'])


def test_incomplete_or_changed_native_references_are_rejected():
    """An empty comparison or a different source revision cannot establish parity."""
    with pytest.raises(ValueError, match='All three pinned'):
        check_pilot({'builds': []}, [])
    report = json.loads((Path(__file__).parents[1] / 'data/converter-native-comparison.json').read_text())
    report['builds'][0]['commit'] = 'unexpected-revision'
    with pytest.raises(ValueError, match='source provenance mismatch'):
        check_pilot(report, [])


def test_byte_mismatch_and_empty_probe_set_do_not_pass(tmp_path, monkeypatch):
    """A plausible byte layout still fails parity if one packed byte changes."""
    from scripts import check_q6_pilot as pilot

    report = json.loads((Path(__file__).parents[1] / 'data/converter-native-comparison.json').read_text())
    for build in report['builds']:
        path = tmp_path / build['id']
        path.write_bytes(b'not executed')
        build.update(executable=str(path), executable_sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    def different(executable, data, qtype):
        result = quantize_q6_k(data)
        result[0, 0] ^= 1
        return result, 0.0

    monkeypatch.setattr(pilot, 'native_encode', different)
    result = check_pilot(report, [('zero', np.zeros((1, 256), dtype=np.float32))])
    assert not result['all_equal']
    assert all(item['different_bytes'] == 1 for item in result['results'][0]['references'])
    with pytest.raises(ValueError, match='At least one'):
        check_pilot(report, [])
