"""GGUF layout, padding and atomic publication for the research Q6_K path."""
import hashlib
import json
from pathlib import Path
import threading

import gguf
import numpy as np
import pytest

from convert import ConversionCancelled
from scripts import check_q6_gguf_container as container
from scripts.reference_q6_k import quantize_q6_k


@pytest.fixture
def inputs(tmp_path):
    """Use odd block counts so alignment padding is required between tensors."""
    reports, arrays = [], []
    for index, shape in enumerate(((1, 256), (3, 512), (5, 768))):
        data = np.linspace(-1, 2, np.prod(shape), dtype=np.float32).reshape(shape)
        payload = quantize_q6_k(data).tobytes()
        raw, report = tmp_path / f'{index}.q6k', tmp_path / f'{index}.json'
        raw.write_bytes(payload)
        report.write_text(json.dumps({'shape': list(shape), 'all_equal': True,
                                     'source_dtype': 'F32', 'tensor': f'tensor.{index}.weight',
                                     'output': raw.name, 'output_bytes': len(payload),
                                     'payload_sha256': hashlib.sha256(payload).hexdigest()}))
        reports.append(report)
        arrays.append(data)
    return reports, arrays


def test_container_preserves_shapes_bytes_decode_and_padding(tmp_path, inputs):
    """Quantized byte dimensions must not leak into logical GGUF tensor shapes."""
    reports, arrays = inputs
    destination = tmp_path / 'research.gguf'
    before = {path: path.read_bytes() for path in tmp_path.iterdir()}
    result = container.package_tensors(reports, destination)
    reader = gguf.GGUFReader(str(destination))
    try:
        assert reader.get_field('general.architecture').contents() == 's2gguf_q6_research'
        assert reader.get_field('s2gguf.research_only').contents()
        assert len(reader.tensors) == 3
        for index, (tensor, source) in enumerate(zip(reader.tensors, arrays)):
            assert tensor.tensor_type == gguf.GGMLQuantizationType.Q6_K
            assert tuple(reversed(tensor.shape)) == source.shape
            restored = gguf.dequantize(tensor.data, tensor.tensor_type)
            assert restored.shape == source.shape
            assert np.max(np.abs(restored - source)) < 0.06
            assert tensor.data_offset % reader.alignment == 0
            if index:
                previous = reader.tensors[index - 1]
                gap = reader.data[previous.data_offset + previous.n_bytes:tensor.data_offset]
                assert len(gap) < reader.alignment and not gap.any()
    finally:
        reader.data._mmap.close()
    assert result['all_equal'] and result['tensor_count'] == 3
    assert result['file_sha256'] == hashlib.sha256(destination.read_bytes()).hexdigest()
    assert all(path.read_bytes() == content for path, content in before.items())
    assert not list(tmp_path.glob('.research.gguf.*'))


@pytest.mark.parametrize('change', ['truncate', 'extend', 'corrupt', 'failed_report', 'bad_shape', 'duplicate'])
def test_invalid_inputs_do_not_publish(tmp_path, inputs, change):
    reports, _ = inputs
    raw = tmp_path / '0.q6k'
    payload = raw.read_bytes()
    if change == 'truncate':
        raw.write_bytes(payload[:-1])
    elif change == 'extend':
        raw.write_bytes(payload + b'x')
    elif change == 'corrupt':
        raw.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])
    else:
        report = json.loads(reports[0].read_text())
        if change == 'failed_report':
            report['all_equal'] = False
        elif change == 'bad_shape':
            report['shape'] = [256, 1]
        else:
            reports.append(reports[0])
        reports[0].write_text(json.dumps(report))
    with pytest.raises(ValueError):
        container.package_tensors(reports, tmp_path / 'research.gguf')
    assert not (tmp_path / 'research.gguf').exists()
    assert not list(tmp_path.glob('.research.gguf.*'))


@pytest.mark.parametrize('failure', ['changed_raw', 'cancel', 'bad_header'])
def test_late_failure_removes_output_and_closes_windows_mapping(tmp_path, inputs, monkeypatch, failure):
    reports, _ = inputs
    event = threading.Event()
    original = gguf.GGUFWriter.close

    def close(writer):
        original(writer)
        if failure == 'bad_header':
            with writer.path.open('r+b') as stream:
                stream.write(b'FAIL')

    monkeypatch.setattr(gguf.GGUFWriter, 'close', close)
    original_padding = gguf.GGUFWriter.write_padding

    def padding(writer, stream, size):
        original_padding(writer, stream, size)
        if failure == 'cancel':
            event.set()
        elif failure == 'changed_raw':
            (tmp_path / '0.q6k').write_bytes(b'changed')

    monkeypatch.setattr(gguf.GGUFWriter, 'write_padding', padding)
    with pytest.raises((ValueError, ConversionCancelled)):
        container.package_tensors(reports, tmp_path / 'research.gguf', cancel_event=event)
    assert not (tmp_path / 'research.gguf').exists()
    assert not list(tmp_path.glob('.research.gguf.*'))


def test_empty_input_and_existing_output_are_rejected(tmp_path, inputs):
    with pytest.raises(ValueError, match='At least one'):
        container.package_tensors([], tmp_path / 'research.gguf')
    destination = tmp_path / 'existing.gguf'
    destination.write_bytes(b'original')
    with pytest.raises(OSError, match='exists'):
        container.package_tensors(inputs[0], destination)
    assert destination.read_bytes() == b'original'


def test_retained_container_evidence_matches_stream_payloads():
    """Container proof links exact raw payload hashes and original logical shapes."""
    data = Path(__file__).parents[1] / 'data'
    result = json.loads((data / 'q6-gguf-container.json').read_text())
    sources = [data / f'q6-stream-{name}.json' for name in ('flux', 'qwen', 'sdxl')]
    assert result['all_equal'] and result['tensor_count'] == 3
    assert result['architecture'] == 's2gguf_q6_research'
    assert result['source_report_hash_normalization'] == 'CRLF-to-LF'
    assert result['source_report_sha256'] == [
        hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest() for path in sources]
    for tensor, source in zip(result['tensors'], sources):
        report = json.loads(source.read_text())
        assert tensor['shape'] == report['shape'] and tensor['name'] == report['tensor']
        assert tensor['payload_sha256'] == report['payload_sha256']
        assert tensor['bytes'] == report['output_bytes'] and tensor['data_offset'] % 32 == 0
        assert tensor['type'] == 'Q6_K'


def test_report_newline_translation_preserves_embedded_provenance(tmp_path, inputs):
    """The same report in a Windows or Linux checkout must produce identical GGUF."""
    reports, _ = inputs
    for path in reports:
        path.write_bytes(path.read_bytes().replace(b'\r\n', b'\n'))
    lf = container.package_tensors(reports, tmp_path / 'lf.gguf')
    for path in reports:
        path.write_bytes(path.read_bytes().replace(b'\n', b'\r\n'))
    crlf = container.package_tensors(reports, tmp_path / 'crlf.gguf')
    assert lf['source_report_sha256'] == crlf['source_report_sha256']
    assert lf['file_sha256'] == crlf['file_sha256']
