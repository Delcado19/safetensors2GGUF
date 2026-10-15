"""Policy research uses real rules, stable order and header-only model inspection."""
import json
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest

from scripts.compare_converter_policies import apply_city96_patch, fixture_tensors, load_molbal_planner, plan_molbal
from scripts.compare_native_reference_vectors import run_command
from scripts.inventory_converter_sources import verify_references
from scripts.inventory_installed_model_headers import inspect_header, inventory_headers


@pytest.fixture(scope='module')
def planner():
    """Use the frozen planner rather than a reimplementation of its rules."""
    return load_molbal_planner(verify_references())


def test_molbal_planning_is_order_independent_and_preserves_scope(planner):
    """Native order sensitivity can be compared against an actual stable planner."""
    normal = plan_molbal(planner, 'flux', fixture_tensors('flux'), 'Q4_K_S')
    reverse = plan_molbal(planner, 'flux', fixture_tensors('flux', reverse=True), 'Q4_K_S')
    assert normal == reverse
    assert normal['types']['blocks.7.attn.to_v.weight'] == 'Q5_K'
    assert normal['types']['small.weight'] == 'F32'
    assert plan_molbal(planner, 'flux', fixture_tensors('flux'), 'Q2_K')['status'] == 'unsupported_preset'


def test_molbal_alignment_and_anchored_protection_are_distinct(planner):
    """SD shape-fix and Krea anchoring cannot become generic fallback/substrings."""
    odd = fixture_tensors('flux', incompatible=True)
    assert plan_molbal(planner, 'flux', odd, 'Q4_K_M')['status'] == 'rejected'
    assert plan_molbal(planner, 'sdxl', odd, 'Q4_K_M')['types']['odd.weight'] == 'Q4_K'
    plan = plan_molbal(planner, 'krea2', {'tmlp.0.weight': (8, 256),
                                      'blocks.0.txtmlp.0.weight': (8, 256)}, 'Q4_K_M')
    assert plan['types']['tmlp.0.weight'] == 'F32'
    assert plan['types']['blocks.0.txtmlp.0.weight'] == 'Q4_K'


def test_header_inventory_does_not_require_weight_payload(tmp_path):
    """A schematic huge embedding is inspected without allocating its tensor data."""
    folder = tmp_path / 'text_encoders'
    folder.mkdir()
    header = {'model.embed_tokens.weight': {'dtype': 'BF16', 'shape': [151936, 2560], 'data_offsets': [0, 0]}}
    for layer in range(36):
        header[f'model.layers.{layer}.self_attn.q_proj.weight'] = {
            'dtype': 'BF16', 'shape': [2560, 2560], 'data_offsets': [0, 0]}
    payload = json.dumps(header).encode()
    path = folder / 'schematic-header.safetensors'
    path.write_bytes(struct.pack('<Q', len(payload)) + payload)
    result = inspect_header(path, 'text_encoders')
    assert result['family'] == 'qwen3-4b'
    assert result['tensor_count'] == 37
    assert result['stored_k_row_aligned_matrix_count'] == 37
    assert result['bytes'] == 8 + len(payload)
    assert len(inventory_headers(tmp_path)['models']) == 1


@pytest.mark.parametrize('payload', [b'x', struct.pack('<Q', 64 * 1024 * 1024 + 1), struct.pack('<Q', 100)])
def test_invalid_header_size_is_rejected(tmp_path, payload):
    """Do not read arbitrary lengths from truncated or malformed header prefixes."""
    path = tmp_path / 'bad.safetensors'
    path.write_bytes(payload)
    with pytest.raises(ValueError, match='header'):
        inspect_header(path, 'diffusion_models')


def test_patch_materializes_inside_nested_repository(tmp_path):
    """Git discovery must not silently skip an inner standalone source tree."""
    parent = tmp_path / 'parent'
    parent.mkdir()
    result = run_command(['git', 'init', '--quiet', str(parent)], text=True)
    assert result.returncode == 0, result.stderr
    source = parent / 'nested/source'
    (source / 'src').mkdir(parents=True)
    target = source / 'src/llama.cpp'
    target.write_text('before\n', encoding='utf-8')
    patch = source.parent / 'change.patch'
    patch.write_text('--- a/src/llama.cpp\n+++ b/src/llama.cpp\n@@ -1 +1 @@\n-before\n'
                     '+static ggml_type img_tensor_get_type\n', encoding='utf-8')
    apply_city96_patch(source, patch)
    assert 'img_tensor_get_type' in target.read_text(encoding='utf-8')


def test_skipped_patch_is_not_treated_as_applied(tmp_path, monkeypatch):
    """Successful exit codes alone are insufficient evidence of patched sources."""
    from scripts import compare_converter_policies as policies

    (tmp_path / 'src').mkdir()
    (tmp_path / 'src/llama.cpp').write_text('unpatched', encoding='utf-8')
    monkeypatch.setattr(policies, 'run_command', lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stderr=''))
    with pytest.raises(RuntimeError, match='did not materialize'):
        apply_city96_patch(tmp_path, tmp_path / 'patch')


def test_saved_real_quantizer_policy_decisions():
    """Recorded native choices preserve order sensitivity, fallbacks and T5 drift."""
    report = json.loads((Path(__file__).parents[1] / 'data/converter-policy-comparison.json').read_text())
    assert len(report['results']) == 84
    assert all(row['status'] == 'converted' for row in report['results'])
    rows = {(row['scenario'], row['source'], row['preset']): row for row in report['results']}
    first = rows['flux', 'llama-b3962', 'Q4_K_S']
    reverse = rows['flux-reverse', 'llama-b3962', 'Q4_K_S']
    assert first['types']['blocks.0.attn.to_v.weight'] == 'Q5_K'
    assert reverse['types']['blocks.0.attn.to_v.weight'] == 'Q4_K'
    assert first['molbal'] == reverse['molbal']
    assert first['types']['img_in.weight'] == 'F16'
    assert first['types']['spatial.weight'] == 'F16'
    odd = rows['flux-unaligned', 'llama-b3962', 'Q4_K_M']
    assert odd['types']['odd.weight'] == 'F16'
    assert odd['molbal']['status'] == 'rejected'
    old = rows['t5', 'llama-b3962', 'Q4_K_M']['types']
    installed = rows['t5', 'llama-text-installed', 'Q4_K_M']['types']
    current = rows['t5', 'llama-current', 'Q4_K_M']['types']
    assert installed == current
    assert sum(old[name] != dtype for name, dtype in current.items()) == 18
    assert len(report['builds'][0]['patched_files_sha256']) == 3
