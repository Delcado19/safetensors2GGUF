"""Check benchmark slice selection, native-compatible bytes and retained evidence."""
import hashlib
import json
from pathlib import Path

import gguf
import numpy as np

from scripts.benchmark_q6_chunks import _worker
from scripts.reference_q6_k import quantize_q6_k


def test_slice_benchmark_preserves_bytes_across_chunk_sizes(tmp_path):
    """Measure a small real GGUF slice without loading more than the requested prefix."""
    source = tmp_path / 'source.gguf'
    values = np.linspace(-2, 1, 4096, dtype=np.float16).reshape(16, 256)
    writer = gguf.GGUFWriter(str(source), 'flux')
    try:
        writer.add_tensor('block.weight', values)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        writer.close()
    expected = hashlib.sha256(quantize_q6_k(values[:8].astype(np.float32))).hexdigest()
    for size in (1, 16, 256):
        result = _worker(source, size, 2048, 2)
        assert result['values'] == 2048 and result['tensor'] == 'block.weight'
        assert result['payload_sha256'] == expected
        assert len(result['seconds']) == 2 and min(result['seconds']) > 0
        assert result['process_peak_rss_bytes'] >= result['baseline_peak_rss_bytes'] > 0


def test_retained_chunk_evidence_matches_native_prefix():
    """Do not enforce timing thresholds: recorded results describe one measured host."""
    root = Path(__file__).parents[1]
    report = json.loads((root / 'data/q6-chunk-benchmark-native.json').read_text())
    assert report['all_payloads_equal'] and report['native_prefix']['equal']
    assert report['benchmark_source_sha256'] == hashlib.sha256(
        (root / 'scripts/benchmark_q6_chunks.py').read_bytes()).hexdigest()
    assert {r['chunk_blocks'] for r in report['results']} == {16, 64, 256, 1024, 4096}
    assert all(r['payload_sha256'] == report['native_prefix']['sha256'] for r in report['results'])
    assert all(r['values'] == 1048576 and len(r['seconds']) == 3 for r in report['results'])
