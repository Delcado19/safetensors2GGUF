"""Portable contract and recorded-evidence checks; no compiler or DLL required."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.check_compiled_kernel import _encode, _load


def test_compiled_pilot_records_native_parity_and_source_provenance():
    root = Path(__file__).parents[1]
    report = json.loads((root / 'data/compiled-kernel-pilot.json').read_text())
    assert report['guard_checks'] == 10 and report['source_stat_stable']
    assert report['build']['library_sha256'] == report['library_sha256']
    assert report['build']['commit'] == report['native_reference']['commit']
    assert report['build']['archive_sha256'] == report['native_reference']['archive_sha256']
    assert len(report['parity']) == 3
    assert all(row['all_equal'] and row['golden_inputs'] == 166 for row in report['parity'])
    # Historical reports retain their original hashes; current adapters are requalified below.
    followup = json.loads((root / 'data/compiled-kernel-levels-validation.json').read_text())
    for name, digest in report['source_code_sha256'].items():
        expected = followup['source_code_sha256'].get(name, digest)
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
    for row in report['benchmarks']:
        assert row['native_equal']
        for backend in ('numpy', 'compiled', 'native_direct'):
            assert len(row['seconds'][backend]) == 7
            assert all(value > 0 for value in row['seconds'][backend])


def test_python_boundary_preserves_strided_input_and_propagates_rejection(tmp_path):
    calls = []

    def reject(*args):
        calls.append(args)
        return 3

    data = np.arange(512, dtype=np.float32).reshape(2, 256)[:, ::-1]
    before = data.tobytes()
    with pytest.raises(ValueError, match='kernel rejected input: 3'):
        _encode(reject, data, 4)
    assert data.tobytes() == before and len(calls) == 1
    for bad in (np.array(0, dtype=np.float32), np.zeros(255, dtype=np.float32),
                data.astype(np.float16), np.full((1, 256), np.nan, dtype=np.float32),
                np.zeros((4097, 256), dtype=np.float32)):
        with pytest.raises(ValueError, match='bounded finite'):
            _encode(reject, bad, 4)
    assert len(calls) == 1
    library = tmp_path / 'untrusted.dll'
    library.write_bytes(b'not a library')
    with pytest.raises(ValueError, match='checksum mismatch'):
        _load(library, '0' * 64)
