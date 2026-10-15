"""Bounded GGUF comparison must reject plausible but different output files."""
import threading
import json
from pathlib import Path
from collections import Counter

import gguf
import numpy as np
import pytest

from convert import ConversionCancelled
from scripts.compare_gguf_outputs import compare_files


def _write(path, *, dtype=np.float32, value=1, count=2, metadata_type='int', alignment=32):
    writer = gguf.GGUFWriter(str(path), 'flux')
    try:
        if alignment is not None:
            writer.add_custom_alignment(alignment)
        if metadata_type == 'int':
            writer.add_uint32('test.value', 1)
        else:
            writer.add_float32('test.value', 1.0)
        for index in range(count):
            writer.add_tensor(f'{index}.weight', np.full((1025, 256), value, dtype=dtype))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        writer.close()


def test_identical_files_compare_across_multiple_windows(tmp_path):
    left, right = tmp_path / 'left.gguf', tmp_path / 'right.gguf'
    _write(left)
    _write(right)
    result = compare_files(left, right)
    assert result['all_equal'] and len(result['tensors']) == 2
    assert json.loads(json.dumps(result))['alignment_equal'] is True
    assert all(row['equal'] and row['left_sha256'] == row['right_sha256'] for row in result['tensors'])
    left.unlink()
    right.unlink()  # No live Windows mmap remains after comparison.


def test_explicit_default_alignment_does_not_require_whole_file_identity(tmp_path):
    left, right = tmp_path / 'left.gguf', tmp_path / 'right.gguf'
    _write(left)
    _write(right, alignment=None)
    result = compare_files(left, right)
    assert result['all_equal'] and result['alignment_equal']
    assert result['left_sha256'] != result['right_sha256']


@pytest.mark.parametrize('change', ['late_byte', 'dtype', 'metadata', 'alignment', 'count', 'empty'])
def test_semantic_mismatches_do_not_pass(tmp_path, change):
    left, right = tmp_path / 'left.gguf', tmp_path / 'right.gguf'
    _write(left)
    options = {'dtype': {'dtype': np.float16}, 'metadata': {'metadata_type': 'float'},
               'alignment': {'alignment': 64}, 'count': {'count': 1}, 'empty': {'count': 0}}
    _write(right, **options.get(change, {}))
    if change == 'late_byte':
        reader = gguf.GGUFReader(str(right))
        offset = reader.tensors[0].data_offset + 1024 * 1024 + 5
        reader.data._mmap.close()
        with right.open('r+b') as stream:
            stream.seek(offset)
            stream.write(b'\xff')
    assert not compare_files(left, right)['all_equal']


def test_cancel_and_bad_second_header_release_both_mappings(tmp_path):
    left, right = tmp_path / 'left.gguf', tmp_path / 'right.gguf'
    _write(left)
    _write(right)
    event = threading.Event()
    event.set()
    with pytest.raises(ConversionCancelled):
        compare_files(left, right, cancel_event=event)
    right.write_bytes(b'bad header')
    with pytest.raises(ValueError):
        compare_files(left, right)
    left.unlink()
    right.unlink()


def test_recorded_full_model_comparison_retains_complete_numeric_evidence():
    """Check retained live results; CI does not materialize the 9B checkpoint."""
    report = json.loads((Path(__file__).parents[1] / 'data/flux-q6-full-model-comparison.json').read_text())
    assert report['all_equal'] and report['metadata_equal'] and report['alignment_equal']
    assert report['tensor_counts_equal'] and len(report['tensors']) == 201
    assert Counter(row['type'] for row in report['tensors']) == {'Q6_K': 115, 'F32': 86}
    assert all(row['equal'] and row['left_sha256'] == row['right_sha256'] for row in report['tensors'])
    assert report['input_provenance']['source_header']['dtype_tensor_counts'] == {'BF16': 201}
    assert report['runs']['native']['exit_code'] == 0
    assert report['runs']['python']['seconds'] > 0



def test_recorded_q5_full_model_comparison_retains_complete_numeric_evidence():
    """Retain full Q5 mixed-policy parity without distributing checkpoint weights."""
    report = json.loads((Path(__file__).parents[1] / 'data/flux-q5-full-model-comparison.json').read_text())
    q6 = json.loads((Path(__file__).parents[1] / 'data/flux-q6-full-model-comparison.json').read_text())
    assert report['all_equal'] and report['metadata_equal'] and report['alignment_equal']
    assert report['tensor_counts_equal'] and len(report['tensors']) == 201
    assert report['preset'] == 'Q5_K_M'
    assert Counter(row['type'] for row in report['tensors']) == {'Q5_K': 99, 'Q6_K': 16, 'F32': 86}
    assert all(row['equal'] and row['left_sha256'] == row['right_sha256'] for row in report['tensors'])
    assert report['source_sha256'] == q6['input_provenance']['prepared_sha256']
    assert report['source_bytes'] == q6['input_provenance']['prepared_bytes']
    assert report['native'] == q6['runs']['native']['native']
    assert report['native_seconds'] > 0 and report['python_seconds'] > 0



def test_q5_user_render_evidence_has_identical_pixels_and_seed_caveat():
    """Distinguish Python/native runtime parity from a different-seed original."""
    r = json.loads((Path(__file__).parents[1] / 'data/flux-q5-user-render-comparison.json').read_text())
    original, python, native = r['images']
    assert r['python_native_identical_filtered_scanlines']
    assert python['ihdr_hex'] == native['ihdr_hex']
    assert python['filtered_scanlines_sha256'] == native['filtered_scanlines_sha256']
    assert all(image['all_png_chunk_crcs_valid'] for image in r['images'])
    a, b = python['metadata']['prompt'], native['metadata']['prompt']
    assert {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)} == {'115'}
    assert a['115']['class_type'] == b['115']['class_type'] == 'UnetLoaderGGUF'
    assert a['96']['inputs']['noise_seed'] == b['96']['inputs']['noise_seed'] == r['q5_seed']
    assert original['metadata']['prompt']['96']['inputs']['noise_seed'] == r['original_seed']
    assert r['original_seed'] != r['q5_seed']



def test_q5_same_seed_original_changes_only_loader_and_model_connection():
    """Keep the corrected same-seed comparison separate from the first original."""
    r = json.loads((Path(__file__).parents[1] / 'data/flux-q5-user-render-comparison.json').read_text())
    original, python = r['same_seed_original'], r['images'][1]
    a, b = original['metadata']['prompt'], python['metadata']['prompt']
    assert original['all_png_chunk_crcs_valid']
    assert original['ihdr_hex'] == python['ihdr_hex']
    assert a['96']['inputs']['noise_seed'] == b['96']['inputs']['noise_seed'] == r['q5_seed']
    assert {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)} == {'108', '113', '115'}
    assert {k: v for k, v in a['113']['inputs'].items() if k != 'model'} == {
        k: v for k, v in b['113']['inputs'].items() if k != 'model'}


def test_compiled_full_flux_report_retains_native_bytes_and_live_failure_checks():
    """Keep complete file-level evidence separate from the earlier kernel timings."""
    root = Path(__file__).parents[1]
    report = json.loads((root / 'data/flux-compiled-q4m-full-model-comparison.json').read_text())
    previous = json.loads((root / 'data/flux-q4m-full-model-comparison.json').read_text())
    pilot = json.loads((root / 'data/compiled-kernel-pilot.json').read_text())
    assert report['all_equal'] and report['metadata_equal'] and report['alignment_equal']
    assert report['tensor_counts_equal'] and len(report['tensors']) == 201
    assert Counter(row['type'] for row in report['tensors']) == {'Q4_K': 99, 'Q5_K': 16, 'F32': 86}
    assert all(row['equal'] and row['left_sha256'] == row['right_sha256'] for row in report['tensors'])
    assert report['left_sha256'] == previous['left_sha256']
    assert report['right_sha256'] == report['reference_sha256_before'] == previous['right_sha256']
    assert report['source_sha256'] == previous['source_sha256'] and report['source_stat_stable']
    assert report['library_sha256'] == pilot['library_sha256'] == report['kernel_build']['library_sha256']
    assert report['native_provenance']['executable_sha256'] == previous['native']['executable_sha256']
    assert report['output_bytes'] == 5560654144
    assert report['cli_check']['equal']
    assert report['cli_check']['library_sha256'] == report['library_sha256']
    assert len(report['live_fixtures']) == 32 and all(row['equal'] for row in report['live_fixtures'])
    assert {row['scenario'] for row in report['live_failure_checks']} == {'cancel', 'late_nan', 'failure'}
    assert all(row['old_output_preserved'] and row['source_preserved'] and row['temporary_cleaned']
               and row['native_calls'] > 0 for row in report['live_failure_checks'])
    assert all(value > 0 for value in report['seconds'].values())



def test_recorded_q4m_full_model_comparison_retains_complete_numeric_evidence():
    """Retain complete Q4 mixed-policy proof without shipping model weights."""
    root = Path(__file__).parents[1]
    r = json.loads((root / 'data/flux-q4m-full-model-comparison.json').read_text())
    q5 = json.loads((root / 'data/flux-q5-full-model-comparison.json').read_text())
    assert r['all_equal'] and r['metadata_equal'] and r['alignment_equal']
    assert r['tensor_counts_equal'] and len(r['tensors']) == 201
    assert r['preset'] == 'Q4_K_M'
    assert Counter(row['type'] for row in r['tensors']) == {'Q4_K': 99, 'Q5_K': 16, 'F32': 86}
    assert all(row['equal'] and row['left_sha256'] == row['right_sha256'] for row in r['tensors'])
    assert r['source_sha256'] == q5['source_sha256'] and r['source_bytes'] == q5['source_bytes']
    assert r['native'] == q5['native'] and r['native_exit_code'] == 0
    assert r['source_stat_stable']
    assert r['native_seconds'] > 0 and r['python_seconds'] > 0
    assert set(r['source_code_sha256']) == {
        'scripts/reference_q4_k.py', 'scripts/reference_q5_k.py',
        'scripts/reference_q6_k.py', 'scripts/requantize_flux_q6.py'}



def test_q4m_user_renders_have_equal_pixels_and_matching_original_settings():
    """Retain scoped rendering/LoRA parity without equating original image fidelity."""
    r = json.loads((Path(__file__).parents[1] / 'data/flux-q4m-user-render-comparison.json').read_text())
    original, python, native = r['images']
    assert r['python_native_identical_filtered_scanlines']
    assert python['ihdr_hex'] == native['ihdr_hex'] == original['ihdr_hex']
    assert python['filtered_scanlines_sha256'] == native['filtered_scanlines_sha256']
    assert all(image['all_png_chunk_crcs_valid'] for image in r['images'])
    a, b, c = [image['metadata']['prompt'] for image in r['images']]
    assert a['96'] == b['96'] == c['96']
    assert a['96']['inputs']['noise_seed'] == r['seed'] == 572241167616859
    assert {k for k in b.keys() | c.keys() if b.get(k) != c.get(k)} == {'116'}
    assert {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)} == {'108', '113', '116'}
    assert {k: v for k, v in a['113']['inputs'].items() if k != 'model'} == {
        k: v for k, v in b['113']['inputs'].items() if k != 'model'}



def test_q4m_no_lora_triplet_preserves_generation_settings_and_native_pixels():
    """Disabled entries and a downstream save-mode difference must stay explicit."""
    r = json.loads((Path(__file__).parents[1] / 'data/flux-q4m-user-render-no-lora.json').read_text())
    original, python, native = r['images']
    a, b, c = [image['metadata']['prompt'] for image in r['images']]
    assert r['python_native_identical_filtered_scanlines']
    assert original['ihdr_hex'] == python['ihdr_hex'] == native['ihdr_hex']
    assert python['filtered_scanlines_sha256'] == native['filtered_scanlines_sha256']
    assert all(image['all_png_chunk_crcs_valid'] for image in r['images'])
    assert a['96'] == b['96'] == c['96']
    assert a['96']['inputs']['noise_seed'] == r['seed']
    for prompt in (a, b, c):
        assert not prompt['113']['inputs']['lora_1']['on']
        assert not prompt['113']['inputs']['lora_2']['on']
    assert {k for k in b.keys() | c.keys() if b.get(k) != c.get(k)} == {'116'}
    assert {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)} == {'108', '112', '113', '116'}
    assert {k: v for k, v in a['112']['inputs'].items() if k != 'save_mode'} == {
        k: v for k, v in b['112']['inputs'].items() if k != 'save_mode'}
    assert {k: v for k, v in a['113']['inputs'].items() if k != 'model'} == {
        k: v for k, v in b['113']['inputs'].items() if k != 'model'}
