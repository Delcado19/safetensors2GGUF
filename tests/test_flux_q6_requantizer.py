"""Guard the research FLUX quantizer-stage policy and mixed-precision writing."""
import threading
import json
from pathlib import Path

import gguf
import numpy as np
import pytest

from convert import ConversionCancelled
from scripts.requantize_flux_q6 import requantize


def _source(path, *, dtype=np.float16, architecture='flux', alignment=32, extra=None):
    arrays = {'double_blocks.0.img_attn.proj.weight': np.linspace(-1, 2, 2048).reshape(8, 256).astype(dtype),
              'img_in.weight': np.ones((8, 256), dtype=dtype),
              'norm.weight': np.ones(256, dtype=dtype),
              'spatial.weight': np.ones((2, 3, 256), dtype=dtype),
              'odd.weight': np.ones((8, 160), dtype=dtype)}
    arrays.update(extra or {})
    writer = gguf.GGUFWriter(str(path), architecture)
    try:
        writer.add_custom_alignment(alignment)
        writer.add_string('test.string', 'preserved')
        writer.add_array('test.strings', ['first', 'second'])
        writer.add_bool('test.bool', True)
        writer.add_uint32('test.count', 12)
        writer.add_array('comfy.gguf.orig_shape.spatial.weight', [2, 3, 256])
        for name, data in arrays.items():
            writer.add_tensor(name, data)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        writer.close()
    return arrays


@pytest.mark.parametrize('dtype', [np.float16, np.float32])
@pytest.mark.parametrize('alignment', [32, 64])
def test_requantizer_preserves_metadata_protection_and_shapes(tmp_path, dtype, alignment):
    source, destination = tmp_path / 'source.gguf', tmp_path / 'q6.gguf'
    arrays = _source(source, dtype=dtype, alignment=alignment)
    before = source.read_bytes()
    result = requantize(source, destination)
    reader = gguf.GGUFReader(str(destination))
    try:
        assert reader.get_field('general.architecture').contents() == 'flux'
        assert reader.get_field('test.string').contents() == 'preserved'
        assert reader.get_field('test.strings').contents() == ['first', 'second']
        assert reader.get_field('test.bool').contents()
        assert reader.get_field('test.count').contents() == 12
        assert reader.get_field('comfy.gguf.orig_shape.spatial.weight').contents() == [2, 3, 256]
        assert reader.get_field('general.file_type').contents() == int(gguf.LlamaFileType.MOSTLY_Q6_K)
        assert reader.alignment == alignment
        for tensor in reader.tensors:
            assert tensor.data_offset % alignment == 0
            assert tensor.shape.tolist()[::-1] == list(arrays[tensor.name].shape)
            if tensor.name == 'double_blocks.0.img_attn.proj.weight':
                assert tensor.tensor_type == gguf.GGMLQuantizationType.Q6_K
                assert np.max(np.abs(gguf.dequantize(tensor.data, tensor.tensor_type) - arrays[tensor.name])) < 0.06
            elif tensor.name == 'odd.weight':
                assert tensor.tensor_type == gguf.GGMLQuantizationType.F16
            else:
                assert tensor.data.tobytes() == arrays[tensor.name].tobytes()
    finally:
        reader.data._mmap.close()
    assert source.read_bytes() == before
    assert len(result['tensors']) == len(arrays)


@pytest.mark.parametrize('extra', [
    {'bad.weight': np.ones((1, 256), dtype=np.float16)},
    {'bad.weight': np.full((8, 256), np.nan, dtype=np.float16)},
    {'model.diffusion_model.bad.weight': np.ones((8, 256), dtype=np.float16)},
])
def test_unsupported_or_invalid_source_never_publishes(tmp_path, extra):
    source, destination = tmp_path / 'source.gguf', tmp_path / 'q6.gguf'
    _source(source, extra=extra)
    with pytest.raises(ValueError):
        requantize(source, destination)
    assert not destination.exists() and not list(tmp_path.glob('.q6.gguf.*'))


def test_cancellation_and_existing_outputs_preserve_files(tmp_path):
    source, destination = tmp_path / 'source.gguf', tmp_path / 'q6.gguf'
    _source(source)
    event = threading.Event()
    with pytest.raises(ConversionCancelled):
        requantize(source, destination, cancel_event=event, on_progress=lambda *_: event.set())
    assert not destination.exists() and not list(tmp_path.glob('.q6.gguf.*'))
    destination.write_bytes(b'original')
    with pytest.raises(OSError):
        requantize(source, destination)
    assert destination.read_bytes() == b'original'
    with pytest.raises(ValueError, match='different files'):
        requantize(source, source)


def test_other_architectures_are_not_silently_routed_to_flux_policy(tmp_path):
    source = tmp_path / 'source.gguf'
    _source(source, architecture='sdxl')
    with pytest.raises(ValueError, match='FLUX only'):
        requantize(source, tmp_path / 'q6.gguf')


def test_quantized_input_is_rejected_instead_of_requantized_silently(tmp_path):
    source, intermediate = tmp_path / 'source.gguf', tmp_path / 'already-q6.gguf'
    _source(source)
    requantize(source, intermediate)
    with pytest.raises(ValueError, match='F16/F32'):
        requantize(intermediate, tmp_path / 'again.gguf')
    assert not (tmp_path / 'again.gguf').exists()


def test_live_file_parity_evidence_covers_both_source_types_and_fallback():
    """CI checks retained native outcomes, without requiring the external binary."""
    report = json.loads((Path(__file__).parents[1] / 'data/flux-q6-path-comparison.json').read_text())
    assert report['all_equal'] and len(report['results']) == 8
    assert sum(len(row['tensors']) for row in report['results']) == 196
    for row in report['results']:
        assert row['metadata_equal'] and row['tensor_counts_equal']
        assert all(tensor['equal'] for tensor in row['tensors'])
        if 'unaligned-True' in row['scenario']:
            assert next(tensor['type'] for tensor in row['tensors'] if tensor['name'] == 'odd.weight') == 'F16'


def test_fixture_and_comparison_provenance_guards(tmp_path):
    from scripts.check_flux_q6_path import compare_path
    from scripts.compare_converter_policies import write_fixture

    with pytest.raises(ValueError, match='F16 or F32'):
        write_fixture(tmp_path / 'bad.gguf', 'flux', {'x.weight': (8, 256)}, dtype=np.int32)
    report = json.loads((Path(__file__).parents[1] / 'data/converter-policy-comparison.json').read_text())
    report['builds'][0]['commit'] = 'unexpected-source'
    with pytest.raises(ValueError, match='provenance mismatch'):
        compare_path(tmp_path / 'artifacts', report)
    assert not (tmp_path / 'artifacts').exists()
