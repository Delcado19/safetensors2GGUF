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


@pytest.mark.parametrize('outcome', ['success', 'cancel', 'invalid'])
def test_explicit_overwrite_is_transactional(tmp_path, outcome):
    """An existing model survives cancellation and invalid values after partial writes."""
    source, destination = tmp_path / 'source.gguf', tmp_path / 'q6.gguf'
    extra = {'late.weight': np.full((8, 256), np.nan, dtype=np.float16)} if outcome == 'invalid' else None
    _source(source, extra=extra)
    original_source = source.read_bytes()
    destination.write_bytes(b'existing model')
    event = threading.Event()
    callback = (lambda *_: event.set()) if outcome == 'cancel' else None
    if outcome == 'success':
        result = requantize(source, destination, overwrite=True)
        import hashlib
        assert hashlib.sha256(destination.read_bytes()).hexdigest() == result['file_sha256']
        assert destination.read_bytes() != b'existing model'
    else:
        with pytest.raises(ConversionCancelled if outcome == 'cancel' else ValueError):
            requantize(source, destination, overwrite=True, cancel_event=event, on_progress=callback)
        assert destination.read_bytes() == b'existing model'
    assert source.read_bytes() == original_source
    assert not list(tmp_path.glob('.q6.gguf.*'))
    with pytest.raises(ValueError, match='different files'):
        requantize(source, source, overwrite=True)


@pytest.mark.parametrize('name', [
    'x.attn_v.weight', 'x.to_v.weight', 'x.v.weight', 'x.attn.w1v.weight',
    'x.attn.w2v.weight', 'x_attn.v_proj.weight', 'x.attn_qkv.weight',
    'x.attn.qkv.weight', 'x.attention.qkv.weight', 'x.ffn_down.weight',
    'experts.0.w2.weight', 'x.ffn.2.weight', 'x.ff.net.2.weight',
    'x.mlp.layer2.weight', 'x.adaln_modulation_mlp.2.weight', 'x.feed_forward.w2.weight'])
def test_q5_m_promotes_city96_roles(name):
    from scripts.requantize_flux_q6 import flux_q6_type
    assert flux_q6_type(name, (8, 256), gguf.GGMLQuantizationType.F32,
                        preset='Q5_K_M') == gguf.GGMLQuantizationType.Q6_K


@pytest.mark.parametrize('outcome', ['success', 'cancel', 'invalid'])
def test_q5_m_mixed_writer_is_transactional(tmp_path, outcome):
    from scripts.reference_q5_k import quantize_q5_k
    from scripts.gguf_research_io import open_gguf
    source, destination = tmp_path / 'source.gguf', tmp_path / 'q5.gguf'
    extra = {'late.weight': np.full((8, 256), np.nan, dtype=np.float16)} if outcome == 'invalid' else None
    arrays = _source(source, extra=extra)
    destination.write_bytes(b'previous model')
    event = threading.Event()
    if outcome != 'success':
        with pytest.raises(ConversionCancelled if outcome == 'cancel' else ValueError):
            requantize(source, destination, preset='Q5_K_M', overwrite=True,
                       cancel_event=event, on_progress=lambda *_: event.set() if outcome == 'cancel' else None)
        assert destination.read_bytes() == b'previous model'
    else:
        requantize(source, destination, preset='Q5_K_M', overwrite=True)
        with open_gguf(destination) as reader:
            assert reader.get_field('general.file_type').contents() == int(gguf.LlamaFileType.MOSTLY_Q5_K_M)
            tensor = reader.tensors[0]
            assert tensor.tensor_type == gguf.GGMLQuantizationType.Q5_K
            assert tensor.data.tobytes() == quantize_q5_k(arrays[tensor.name].astype(np.float32)).tobytes()
    assert not list(tmp_path.glob('.q5.gguf.*'))


def test_q5_file_parity_evidence():
    report = json.loads((Path(__file__).parents[1] / 'data/flux-q5-path-comparison.json').read_text())
    assert report['preset'] == 'Q5_K_M' and report['all_equal']
    assert len(report['results']) == 8
    assert sum(len(row['tensors']) for row in report['results']) == 196
    assert all(t['equal'] for row in report['results'] for t in row['tensors'])
    assert {t['type'] for row in report['results'] for t in row['tensors']} == {'Q5_K', 'Q6_K', 'F16', 'F32'}


def test_unknown_preset_rejected_before_output(tmp_path):
    with pytest.raises(ValueError, match='presets'):
        requantize(tmp_path / 'missing', tmp_path / 'output', preset='Q3_K_L')
    assert not (tmp_path / 'output').exists()



@pytest.mark.parametrize('preset,filename', [('Q4_K_S', 'q4s'), ('Q4_K_M', 'q4m')])
def test_q4_file_parity_and_ordered_fallback_evidence(preset, filename):
    """Retain native byte parity and ensure fallback consumes a Q4_S V slot."""
    report = json.loads((Path(__file__).parents[1] / f'data/flux-{filename}-path-comparison.json').read_text())
    assert report['preset'] == preset and report['all_equal']
    assert len(report['results']) == 8
    assert sum(len(row['tensors']) for row in report['results']) == 200
    for row in report['results']:
        assert row['metadata_equal'] and row['alignment_equal'] and row['tensor_counts_equal']
        assert all(t['equal'] for t in row['tensors'])
        attention = [t for t in row['tensors'] if t['name'].startswith('blocks.') and '.to_v.weight' in t['name']]
        assert len(attention) == 8
        odd = 'unaligned-True' in row['scenario']
        if odd:
            assert row['tensors'][0]['name'] == 'prefix.to_v.weight'
            assert row['tensors'][0]['type'] == 'F16'
        if preset == 'Q4_K_S':
            count = 3 if odd else 4
            assert [t['type'] for t in attention] == ['Q5_K'] * count + ['Q4_K'] * (8 - count)
            assert attention[0]['name'] == ('blocks.7.attn.to_v.weight' if 'reverse-True' in row['scenario']
                                           else 'blocks.0.attn.to_v.weight')
        else:
            assert all(t['type'] == 'Q6_K' for t in attention)


@pytest.mark.parametrize('preset', ['Q4_K_S', 'Q4_K_M'])
@pytest.mark.parametrize('outcome', ['success', 'cancel', 'invalid'])
def test_q4_writer_preserves_old_output_on_failure(tmp_path, preset, outcome):
    from scripts.gguf_research_io import open_gguf
    from scripts.reference_q4_k import quantize_q4_k
    source, destination = tmp_path / 'source.gguf', tmp_path / 'q4.gguf'
    extra = {'late.weight': np.full((8, 256), np.nan, dtype=np.float16)} if outcome == 'invalid' else None
    arrays = _source(source, extra=extra)
    original = source.read_bytes()
    destination.write_bytes(b'previous model')
    event = threading.Event()
    if outcome == 'success':
        requantize(source, destination, preset=preset, overwrite=True)
        with open_gguf(destination) as r:
            assert r.get_field('general.file_type').contents() == int(getattr(gguf.LlamaFileType, 'MOSTLY_' + preset))
            assert r.tensors[0].tensor_type == gguf.GGMLQuantizationType.Q4_K
            assert r.tensors[0].data.tobytes() == quantize_q4_k(arrays[r.tensors[0].name].astype(np.float32)).tobytes()
    else:
        with pytest.raises(ConversionCancelled if outcome == 'cancel' else ValueError):
            requantize(source, destination, preset=preset, overwrite=True, cancel_event=event,
                       on_progress=lambda *_: event.set() if outcome == 'cancel' else None)
        assert destination.read_bytes() == b'previous model'
    assert source.read_bytes() == original
    assert not list(tmp_path.glob('.q4.gguf.*'))


@pytest.mark.parametrize('preset', ['Q6_K', 'Q5_K_M', 'Q4_K_M', 'Q4_K_S'])
@pytest.mark.parametrize('outcome', ['success', 'cancel', 'failure', 'bad_bytes'])
def test_explicit_compiled_callback_keeps_writer_guards(tmp_path, preset, outcome):
    """Block-math injection preserves file bytes, bounds and atomic failure handling."""
    from scripts.check_compiled_kernel import ITERATORS
    source, destination = tmp_path / 'source.gguf', tmp_path / 'output.gguf'
    _source(source, extra={'large.weight': np.ones((300, 256), dtype=np.float16),
                          'block.attn.qkv.weight': np.ones((8, 256), dtype=np.float16),
                          'block.to_v.weight': np.ones((8, 256), dtype=np.float16)})
    baseline = tmp_path / 'numpy.gguf'
    requantize(source, baseline, preset=preset)
    destination.write_bytes(b'old model')
    before = source.read_bytes()
    event = threading.Event()
    calls = []

    def kernel(data, bits):
        assert data.dtype == np.float32 and data.ndim == 2
        assert data.size <= 65536 and data.flags.c_contiguous
        calls.append(bits)
        if outcome == 'failure':
            raise RuntimeError('Kernel failed')
        if outcome == 'bad_bytes':
            return np.zeros(1, dtype=np.float32)
        if outcome == 'cancel':
            event.set()
        return np.concatenate(list(ITERATORS[bits](data)))

    if outcome == 'success':
        requantize(source, destination, preset=preset, overwrite=True, compiled_kernel=kernel)
        assert destination.read_bytes() == baseline.read_bytes()
        assert len(calls) >= 4
    else:
        expected = {'cancel': ConversionCancelled, 'failure': RuntimeError, 'bad_bytes': ValueError}[outcome]
        with pytest.raises(expected):
            requantize(source, destination, preset=preset, overwrite=True,
                       cancel_event=event, compiled_kernel=kernel)
        assert destination.read_bytes() == b'old model'
        assert calls
    assert source.read_bytes() == before
    assert not list(tmp_path.glob('.output.gguf.*'))


@pytest.mark.parametrize('arguments', [
    ['--kernel-library', 'untrusted.dll'], ['--kernel-sha256', '0' * 64],
])
def test_compiled_cli_requires_explicit_library_and_hash_together(tmp_path, monkeypatch, arguments):
    """Incomplete native-code selection must fail before creating an output."""
    from scripts.requantize_flux_q6 import main
    destination = tmp_path / 'output.gguf'
    monkeypatch.setattr('sys.argv', ['requantize', '--source', str(tmp_path / 'missing.gguf'),
                                   '--output', str(destination), *arguments])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert not destination.exists()


@pytest.mark.parametrize('preset', ['Q2_K', 'Q3_K_M'])
def test_low_bit_presets_require_explicit_kernel_before_io(tmp_path, preset):
    with pytest.raises(ValueError, match='explicitly selected compiled kernel'):
        requantize(tmp_path / 'missing', tmp_path / 'out.gguf', preset=preset)
    assert not (tmp_path / 'out.gguf').exists()


def test_q3_ordered_v_promotions_count_fallback_but_skip_protected():
    from scripts.requantize_flux_q6 import flux_q6_type
    q = gguf.GGMLQuantizationType
    for index, expected in enumerate((q.Q5_K, q.Q5_K, q.Q4_K, q.Q4_K)):
        assert flux_q6_type('block.to_v.weight', [8, 256], q.F32,
                            preset='Q3_K_M', attention_v_index=index) == expected
    assert flux_q6_type('block.to_v.weight', [8, 160], q.F32, preset='Q3_K_M') == q.F16
    assert flux_q6_type('img_in.to_v.weight', [8, 256], q.F32, preset='Q3_K_M') == q.F32
    assert flux_q6_type('block.attn.qkv.weight', [8, 256], q.F16, preset='Q3_K_M') == q.Q4_K
    assert flux_q6_type('block.ffn_down.weight', [8, 256], q.F16, preset='Q3_K_M') == q.Q4_K
    assert flux_q6_type('block.to_v.weight', [8, 256], q.F16, preset='Q2_K') == q.Q3_K
    assert flux_q6_type('block.ffn_down.weight', [8, 256], q.F16, preset='Q2_K') == q.Q2_K


@pytest.mark.parametrize('preset', ['Q2_K', 'Q3_K_M', 'Q4_K_S', 'Q4_K_M', 'Q5_K_M', 'Q6_K'])
def test_parallel_compiled_writer_is_bounded_ordered_and_native_equal(tmp_path, monkeypatch, preset):
    from scripts import requantize_flux_q6 as module
    from scripts.compiled_kernel_package import load_package, _platform, ROOT
    import time
    if _platform() != {'system': 'win32', 'machine': 'x86_64', 'pointer_bits': 64}:
        pytest.skip('Shipped Windows x86_64 package qualification')
    kernel, _ = load_package(ROOT / 'third_party/cpu-kernels/win32-x86_64')
    source = tmp_path / 'input.gguf'
    _source(source, extra={'large.weight': np.linspace(-2, 3, 2305 * 256, dtype=np.float16).reshape(2305, 256)})
    monkeypatch.setattr(module, '_COMPILED_WORKERS', 1)
    serial = tmp_path / 'serial.gguf'
    requantize(source, serial, preset=preset, compiled_kernel=kernel)
    monkeypatch.setattr(module, '_COMPILED_WORKERS', 3)
    lock = threading.Lock()
    active = peak = 0
    def tracked(data, bits):
        nonlocal active, peak
        assert data.size <= 65536
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            time.sleep(0.005)
            return kernel(data, bits)
        finally:
            with lock:
                active -= 1
    parallel = tmp_path / 'parallel.gguf'
    result = requantize(source, parallel, preset=preset, compiled_kernel=tracked)
    assert serial.read_bytes() == parallel.read_bytes()
    assert 1 < peak <= 3 and active == 0
    assert result['compiled_workers'] == 3
    assert set(result['timings']) == {'encode_write_seconds', 'readback_seconds', 'file_hash_seconds'}
    assert all(t >= 0 for t in result['timings'].values())


@pytest.mark.parametrize('outcome', ['cancel', 'failure', 'bad_bytes'])
def test_parallel_worker_failure_keeps_atomic_output_and_stops_workers(tmp_path, monkeypatch, outcome):
    from scripts import requantize_flux_q6 as module
    monkeypatch.setattr(module, '_COMPILED_WORKERS', 3)
    source, destination = tmp_path / 'input.gguf', tmp_path / 'output.gguf'
    _source(source, extra={'large.weight': np.ones((2305, 256), dtype=np.float16)})
    destination.write_bytes(b'previous output')
    event = threading.Event()
    active = 0
    lock = threading.Lock()
    def kernel(data, bits):
        nonlocal active
        with lock:
            active += 1
        try:
            size = data.size // 256 * gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType[f'Q{bits}_K']][1]
            if data.size == 65536:
                if outcome == 'failure':
                    raise RuntimeError('worker failed')
                if outcome == 'cancel':
                    event.set()
                if outcome == 'bad_bytes':
                    size = 1
            return np.zeros(size, dtype=np.uint8)
        finally:
            with lock:
                active -= 1
    expected = {'cancel': ConversionCancelled, 'failure': RuntimeError, 'bad_bytes': ValueError}[outcome]
    with pytest.raises(expected):
        requantize(source, destination, preset='Q2_K', compiled_kernel=kernel,
                   cancel_event=event, overwrite=True)
    assert destination.read_bytes() == b'previous output' and active == 0
    assert not list(tmp_path.glob('.output.gguf.*'))
