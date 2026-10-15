"""Offline reference integrity and source-only model inventory regression checks."""
import json
import shutil

import pytest

from scripts.inventory_converter_sources import REFERENCES, build_inventory, model_rules, verify_references


def test_reference_archives_and_split_llama_adapters_are_inventoried():
    """Preserved sources verify and modern adapters are found beyond the CLI wrapper."""
    inventory = build_inventory()
    sources = {entry['id']: entry for entry in inventory['sources']}
    assert len(sources) == 6
    assert 'Q4_K_M' in sources['molbal-current']['quantization_keys']
    assert 'Q2_K' not in sources['molbal-current']['quantization_keys']
    for source in ('llama-text-installed', 'llama-current'):
        assert any(name.startswith('conversion/') for name in sources[source]['python_models'])
    manifest = verify_references()
    assert manifest['submodules'][0]['source'] == 'kompute-b3962'


def test_corrupt_reference_is_rejected_before_inspection(tmp_path):
    """A modified source cannot be accepted as its pinned reference revision."""
    manifest = json.loads((REFERENCES / 'sources.json').read_text())
    entry = next(item for item in manifest['sources'] if item['id'] == 'city96-patch')
    manifest['sources'] = [entry]
    (tmp_path / 'sources.json').write_text(json.dumps(manifest))
    shutil.copyfile(REFERENCES / entry['archive'], tmp_path / entry['archive'])
    with (tmp_path / entry['archive']).open('ab') as output:
        output.write(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        verify_references(tmp_path)


def test_declarations_are_parsed_without_executing_source():
    """Rules retain expressions and inheritance rather than pretending to resolve them."""
    code = '''raise RuntimeError("must not execute")
@ModelBase.register("ExampleEncoder")
class Example(ModelTemplate):
    arch = "example"
    keys_hiprec = ["first."]
    keys_detect = [(f"{PREFIX}weight",)]
    def modify_tensors(self): pass
'''
    rules = model_rules(code)
    assert rules[0]['bases'] == ['ModelTemplate']
    assert rules[0]['declared_rules']['keys_hiprec'] == "['first.']"
    assert rules[0]['registrations'] == ["ModelBase.register('ExampleEncoder')"]
    assert rules[0]['methods'] == ['modify_tensors']
