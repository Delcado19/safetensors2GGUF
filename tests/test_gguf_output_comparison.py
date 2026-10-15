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
