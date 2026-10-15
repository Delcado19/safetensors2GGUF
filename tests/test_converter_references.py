"""Static policy resolution and small numerical reference probes."""
import json
from pathlib import Path

import gguf
import numpy as np
import pytest

from scripts.converter_reference_rules import build_rules_report, effective_rules
from scripts.converter_reference_vectors import load_molbal_k, measure_encoding, reference_vectors, run_vectors


def test_inherited_rules_replace_lists_and_use_python_method_order():
    """Resolve imported aliases and a C3 diamond without executing source code."""
    modules = {'models/base.py': '''raise RuntimeError("must not execute")
class Root:
    keys_hiprec = ["root"]
    def transform(self): pass
class Left(Root):
    pass
class Right(Root):
    keys_hiprec = ["right"]
    def transform(self): pass
''', 'models/child.py': '''from .base import Left as First, Right
class Child(First, Right):
    keys_ignore = []
'''}
    child = effective_rules(modules, {'Child'})[0]
    assert child['mro'] == ['models/child.py:Child', 'models/base.py:Left',
                            'models/base.py:Right', 'models/base.py:Root']
    assert child['effective_fields']['keys_hiprec']['expression'] == "['right']"
    assert child['effective_methods']['transform']['class'] == 'Right'
    assert not child['unresolved_bases']


def test_unknown_bases_and_dynamic_class_writes_are_explicit():
    """An unsupported base or conditional assignment cannot appear fully resolved."""
    rule = effective_rules({'model.py': '''class Model(External):
    if CONDITION:
        keys_hiprec = ["conditional"]
'''})[0]
    assert rule['unresolved_bases'] == ['model.py:External']
    assert rule['dynamic_class_statements']
    assert not rule['effective_fields']


@pytest.mark.parametrize('code', [
    'class A(B): pass\nclass B(A): pass',
    'class A: pass\nclass B: pass\nclass X(A,B): pass\nclass Y(B,A): pass\nclass Z(X,Y): pass',
])
def test_invalid_inheritance_is_rejected(code):
    """Cycles and contradictory precedence cannot silently change tensor rules."""
    with pytest.raises(ValueError, match='inheritance'):
        effective_rules({'model.py': code})


def test_frozen_rules_resolve_wan_and_cross_module_text_adapters():
    """Real snapshots retain inherited transforms and exact policy locations."""
    report = build_rules_report()
    sources = {item['id']: item for item in report['sources']}
    for name in ('city96-patch', 'molbal-current'):
        wan = next(item for item in sources[name]['models'] if item['class'] == 'ModelWan')
        assert wan['effective_methods']['handle_nd_tensor']['class'] == 'ModelHyVid'
        assert wan['effective_fields']['keys_ignore']['class'] == 'ModelTemplate'
        assert not wan['unresolved_bases']
        assert sources[name]['cpp_policy_sections']['image_type_selection']['sha256']
    for name in ('llama-text-installed', 'llama-current'):
        mistral = next(item for item in sources[name]['models'] if item['class'] == 'MistralModel')
        assert 'conversion/llama.py:LlamaModel' in mistral['mro']
        mistral3 = next(item for item in sources[name]['models'] if item['class'] == 'Mistral3Model')
        assert any('Ministral3ForCausalLM' in item for item in mistral3['registrations'])
        nested = next(item for item in sources[name]['models']
                      if item['class'] == 'Mistral3Model.Ministral3Model')
        assert 'conversion/llama.py:LlamaModel' in nested['mro']
        assert not sources[name]['missing_requested_classes']
        assert not any(item['unresolved_bases'] for item in sources[name]['models'])


def test_math_probes_distinguish_layout_from_quality_and_row_alignment():
    """Known affine clipping stays visible even though GGUF byte layouts decode."""
    report = run_vectors()
    rows = {(row['type'], row['vector']): row for row in report['results']}
    for name, size in [('Q4_K', 144), ('Q5_K', 176), ('Q6_K', 210)]:
        assert rows[name, 'zero']['rmse'] == 0
        assert rows[name, 'one_block']['bytes'] == size
        # Total elements align to 256 here, but rows do not: total-only checks are unsafe.
        assert rows[name, 'incompatible_rows']['status'] == 'rejected_incompatible_row'
        assert rows[name, 'standard_block_only']['status'] == 'rejected_incompatible_row'
    for name in ('Q4_K', 'Q5_K'):
        # b77b9d8 _quantize_k_affine clips positive-only nonconstant groups.
        assert rows[name, 'positive_ramp']['rmse'] > 1
    assert rows['Q6_K', 'positive_ramp']['max_absolute_error'] < 0.04
    assert rows['Q8_0', 'standard_block_only']['status'] == 'encoded'
    saved = json.loads((Path(__file__).parents[1] / 'data/converter-reference-vectors.json').read_text())
    assert [(row['type'], row['vector'], row['input_sha256']) for row in saved['results']] == [
        (row['type'], row['vector'], row['input_sha256']) for row in report['results']]


def test_invalid_encoder_layout_is_not_accepted():
    """The harness rejects a malformed payload before calculating any error metric."""
    with pytest.raises(ValueError, match='byte layout'):
        measure_encoding(lambda data, qtype: np.zeros((1, 1), dtype=np.uint8),
                         reference_vectors()['one_block'], gguf.GGMLQuantizationType.Q4_K)


@pytest.mark.parametrize('name, offset', [('Q4_K', 16), ('Q5_K', 48)])
def test_positive_affine_codes_saturate_independently_of_decoder(name, offset):
    """The packed codes themselves are all maximum values on the positive ramp."""
    encoder, _ = load_molbal_k()
    payload = encoder(reference_vectors()['positive_ramp'][:1], gguf.GGMLQuantizationType[name])
    assert np.all(payload[0, offset:] == 0xFF)
    if name == 'Q5_K':
        assert np.all(payload[0, 16:48] == 0xFF)  # Every fifth-bit plane is also set.


def test_non_deterministic_encoder_is_not_accepted():
    """Changing bytes on the second call must fail the determinism contract."""
    calls = []

    def unstable(data, qtype):
        result = gguf.quantize(data, qtype)
        if calls:
            result[0, 0] ^= 1
        calls.append(True)
        return result

    with pytest.raises(ValueError, match='non-deterministic'):
        measure_encoding(unstable, reference_vectors()['one_block'], gguf.GGMLQuantizationType.Q8_0)
