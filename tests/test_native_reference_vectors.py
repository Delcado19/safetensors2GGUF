"""Native reference protocol guards without requiring a compiler in pytest."""
import json
import hashlib
import io
from pathlib import Path
from types import SimpleNamespace
import tarfile

import gguf
import numpy as np
import pytest

from scripts import compare_native_reference_vectors as native


def test_source_extraction_rejects_escape_and_preserves_existing_tree(tmp_path):
    """Even build-only extraction cannot escape its fresh target or overwrite one."""
    archive = tmp_path / 'unsafe.tar.gz'
    with tarfile.open(archive, 'w:gz') as source:
        member = tarfile.TarInfo('root/../escape')
        member.size = 1
        source.addfile(member, io.BytesIO(b'x'))
    entry = {'archive': archive.name}
    destination = tmp_path / 'source'
    with pytest.raises(ValueError, match='escapes'):
        native.extract_source(entry, destination, tmp_path)
    assert not (tmp_path / 'escape').exists()
    sentinel = destination / 'keep.txt'
    sentinel.write_text('keep')
    with pytest.raises(FileExistsError):
        native.extract_source(entry, destination, tmp_path)
    assert sentinel.read_text() == 'keep'


def test_native_payload_and_independent_decoder_agree(monkeypatch):
    """Packed bytes and native float output must both satisfy the protocol."""
    data = np.linspace(-2, 2, 256, dtype=np.float32).reshape(1, 256)
    qtype = gguf.GGMLQuantizationType.Q4_0
    packed = gguf.quantize(data, qtype)
    decoded = gguf.dequantize(packed, qtype)
    response = SimpleNamespace(returncode=0, stderr=b'', stdout=packed.tobytes() + decoded.tobytes())
    monkeypatch.setattr(native, 'run_command', lambda *args, **kwargs: response)
    result, difference = native.native_encode(Path('probe'), data, qtype)
    assert np.array_equal(result, packed)
    assert difference == 0
    response.stdout = response.stdout[:-1]
    with pytest.raises(ValueError, match='payload length'):
        native.native_encode(Path('probe'), data, qtype)
    response.stdout = packed.tobytes() + (decoded + 1).tobytes()
    with pytest.raises(ValueError, match='decoders disagree'):
        native.native_encode(Path('probe'), data, qtype)


def test_native_shape_rejection_must_be_explicit(monkeypatch):
    """A crash or unrelated native error cannot count as correct shape rejection."""
    response = SimpleNamespace(returncode=2, stderr=b'Incompatible row width', stdout=b'')
    monkeypatch.setattr(native, 'run_command', lambda *args, **kwargs: response)
    qtype = gguf.GGMLQuantizationType.Q4_K
    with pytest.raises(ValueError, match='rejected incompatible rows'):
        native.native_encode(Path('probe'), np.zeros((256, 2), dtype=np.float32), qtype)
    response.returncode = 3
    with pytest.raises(RuntimeError, match='explicitly reject'):
        native.native_encode(Path('probe'), np.zeros((256, 2), dtype=np.float32), qtype)
    with pytest.raises(RuntimeError, match='encoder failed'):
        native.native_encode(Path('probe'), np.zeros((1, 256), dtype=np.float32), qtype)


@pytest.mark.parametrize('data', [np.zeros(256, dtype=np.float32), np.zeros((1, 256), dtype=np.float64),
                               np.full((1, 256), np.nan, dtype=np.float32)])
def test_invalid_native_input_never_starts_process(monkeypatch, data):
    """Only finite two-dimensional float32 tensors reach the C harness."""
    def unexpected(*args, **kwargs):
        pytest.fail('Invalid input must not start a process')

    monkeypatch.setattr(native, 'run_command', unexpected)
    with pytest.raises(ValueError, match='finite 2D float32'):
        native.native_encode(Path('probe'), data, gguf.GGMLQuantizationType.Q4_K)


def test_saved_native_comparison_covers_all_sources_and_formats():
    """Retained evidence covers every input and records independent decoder checks."""
    report = json.loads((Path(__file__).parents[1] / 'data/converter-native-comparison.json').read_text())
    assert {item['id'] for item in report['builds']} == set(native.NATIVE_SOURCES)
    assert len(report['results']) == len(native.NATIVE_SOURCES) * len(native.FORMATS) * len(native.reference_vectors())
    assert {row['type'] for row in report['results']} == set(native.FORMATS)
    for build in report['builds']:
        assert len(build['executable_sha256']) == 64
        assert len(build['archive_sha256']) == 64
        assert build['compiler']['CMAKE_C_COMPILER_VERSION'] != 'unknown'
        probe = Path(__file__).parents[1] / 'tools/converter-reference/probe.c'
        assert build['probe_sha256'] == hashlib.sha256(probe.read_bytes()).hexdigest()
        assert len(build['protocol_checks']) == 7
    for row in report['results']:
        if row['status'] == 'encoded':
            assert row['native_decoder_max_difference'] < 1e-4
            assert row['bytes'] > 0
            assert row['equals_b3962_bytes']
