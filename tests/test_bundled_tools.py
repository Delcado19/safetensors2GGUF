"""Pinned tools must work offline and reject modified sources or binaries."""
import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest

import bundled_tools as tools


@pytest.fixture
def reference(tmp_path, monkeypatch):
    folder = tmp_path / 'references'
    folder.mkdir()
    archive = folder / 'text.tar.gz'
    with tarfile.open(archive, 'w:gz') as output:
        for name in ('convert_hf_to_gguf.py', 'conversion/model.py'):
            data = b'# pinned source\n'
            member = tarfile.TarInfo('pinned/' + name)
            member.size = len(data)
            output.addfile(member, io.BytesIO(data))
    entry = {'id': 'llama-text-installed', 'commit': 'pinned-commit', 'archive': archive.name,
             'archive_root': 'pinned', 'sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}
    (folder / 'sources.json').write_text(json.dumps({'sources': [entry]}))
    monkeypatch.setattr(tools, 'REFERENCES', folder)
    return folder


def test_pinned_source_extracts_and_reuses_offline(tmp_path, reference, monkeypatch):
    import subprocess
    def no_process(*args, **kwargs):
        pytest.fail('Pinned tool preparation must not launch external tools')
    monkeypatch.setattr(subprocess, 'Popen', no_process)
    first = tools.text_source(tmp_path / 'cache')
    assert (first / 'conversion/model.py').is_file()
    assert tools.text_source(tmp_path / 'cache') == first


@pytest.mark.parametrize('damage', ['archive', 'cache', 'missing'])
def test_source_damage_fails_explicitly(tmp_path, reference, damage):
    directory = tools.text_source(tmp_path / 'cache')
    if damage == 'archive':
        (reference / 'text.tar.gz').write_bytes(b'wrong archive')
    elif damage == 'cache':
        (directory / 'conversion/model.py').write_bytes(b'changed code')
    else:
        (directory / 'conversion/model.py').unlink()
    with pytest.raises(ValueError, match='checksum'):
        tools.text_source(tmp_path / 'cache')


def test_normal_text_entrypoints_never_clone_or_build(tmp_path, monkeypatch):
    import text_encoder_convert as conversion
    monkeypatch.delenv(conversion.LLAMA_CPP_HOME_ENV, raising=False)
    monkeypatch.setattr(tools, 'text_source', lambda cache: tmp_path)
    monkeypatch.setattr(tools, 'text_quantizer', lambda: tmp_path / 'text.exe')
    assert conversion.ensure_llama_cpp() == tmp_path
    assert conversion.ensure_plain_llama_quantize() == tmp_path / 'text.exe'


def test_real_shipped_text_binary_is_pinned():
    if tools._platform() != {'system': 'win32', 'machine': 'x86_64', 'pointer_bits': 64}:
        pytest.skip('Windows binary package')
    path = tools.text_quantizer()
    assert path.name == 'llama-quantize.exe'
    assert path.parent == Path(tools.ROOT) / 'third_party/text-tools/win32-x86_64'


@pytest.mark.parametrize('relative', ['third_party/text-tools/win32-x86_64/llama-quantize.exe',
                                     'third_party/converter_references/sources.json',
                                     'build/pinned-text-converter/version/convert_hf_to_gguf.py'])
def test_text_outputs_cannot_replace_managed_tools(relative):
    with pytest.raises(ValueError, match='overwrite managed'):
        tools.validate_text_destination(tools.ROOT / relative)


def test_recorded_static_text_quantizer_matches_previous_tool():
    report = json.loads((tools.ROOT / 'data/bundled-text-quantizer-validation.json').read_text())
    assert report['all_equal'] and len(report['results']) == 18
    assert set(report['imports']) == {'KERNEL32.dll', 'ADVAPI32.dll'}
    assert all(row['comparison']['all_equal'] for row in report['results'])
    assert tools._digest(tools.ROOT / 'third_party/text-tools/win32-x86_64/llama-quantize.exe') == report['binary_sha256']


def test_real_export_smoke_records_current_sources():
    report = json.loads((tools.ROOT / 'data/bundled-text-export-smoke.json').read_text())
    assert report['architecture'] == 'qwen3' and report['source_unchanged']
    assert report['tensor_count'] == 14
    assert report['tensor_types'] == {'Q4_K': 6, 'Q6_K': 3, 'F32': 5}
    for name, digest in report['source_code_sha256'].items():
        assert tools._digest(tools.ROOT / name) == digest
