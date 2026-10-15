"""Reject incompatible or altered native packages before loading executable code."""
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from scripts import compiled_kernel_package as package


@pytest.fixture
def kernel_package(tmp_path):
    directory = tmp_path / 'package'
    directory.mkdir()
    library = package._library_name()
    (directory / library).write_bytes(b'not executable; metadata tests must never load it')
    for name in ('kernel.c', 'CMakeLists.txt'):
        shutil.copyfile(package.ROOT / 'tools/converter-kernel' / name, directory / name)
    shutil.copyfile(package.REFERENCES / 'llama-b3962-LICENSE', directory / 'LICENSE')
    manifest = {'schema': 1, 'platform': package._platform(), 'source': package._source_identity(),
                'library': library, 'files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                             for p in directory.iterdir()}}
    (directory / 'kernel.json').write_text(json.dumps(manifest), encoding='utf8')
    assert package.verify_package(directory) == manifest
    return directory, manifest


@pytest.mark.parametrize('damage', ['platform', 'source', 'schema', 'traversal', 'library', 'adapter', 'license'])
def test_invalid_package_never_loads_native_code(kernel_package, monkeypatch, damage):
    directory, manifest = kernel_package
    if damage == 'platform':
        manifest['platform']['pointer_bits'] = 0
    elif damage == 'source':
        manifest['source']['commit'] = 'unrecognized'
    elif damage == 'schema':
        manifest['schema'] = 99
    elif damage == 'traversal':
        manifest['files']['../outside.dll'] = '0' * 64
    elif damage == 'library':
        (directory / manifest['library']).write_bytes(b'changed')
    else:
        name = 'kernel.c' if damage == 'adapter' else 'LICENSE'
        (directory / name).write_bytes(b'changed')
        manifest['files'][name] = hashlib.sha256(b'changed').hexdigest()
    (directory / 'kernel.json').write_text(json.dumps(manifest), encoding='utf8')

    def forbidden(*_):
        pytest.fail('An incompatible package must not load native code')

    monkeypatch.setattr(package, '_load', forbidden)
    with pytest.raises(ValueError):
        package.load_package(directory)


def test_build_rejects_existing_or_overlapping_destinations_without_work(tmp_path, monkeypatch):
    def forbidden(*_):
        pytest.fail('Invalid destinations must fail before source/build work')

    monkeypatch.setattr(package, 'verify_references', forbidden)
    existing = tmp_path / 'existing'
    existing.mkdir()
    marker = existing / 'preserve'
    marker.write_bytes(b'original')
    with pytest.raises(FileExistsError):
        package.build_package(existing, tmp_path / 'work', 'cmake')
    assert marker.read_bytes() == b'original'
    assert not (tmp_path / 'work').exists()
    with pytest.raises(ValueError, match='disjoint'):
        package.build_package(tmp_path / 'outer', tmp_path / 'outer/inner', 'cmake')
    with pytest.raises(ValueError, match='CMake is required'):
        package.build_package(tmp_path / 'out', tmp_path / 'work', None)
    assert not (tmp_path / 'out').exists() and not (tmp_path / 'work').exists()


@pytest.mark.parametrize('extra', [
    ['--kernel-library', 'raw.dll'], ['--kernel-sha256', '0' * 64],
])
def test_package_cli_rejects_conflicting_raw_selection(tmp_path, monkeypatch, extra):
    from scripts.requantize_flux_q6 import main
    output = tmp_path / 'output.gguf'
    monkeypatch.setattr('sys.argv', ['requantize', '--source', str(tmp_path / 'source.gguf'),
                                   '--output', str(output), '--kernel-package', 'trusted', *extra])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2 and not output.exists()


def test_recorded_windows_package_build_and_real_cli_evidence():
    """Keep actual build/self-test evidence without requiring a compiler in unit CI."""
    root = Path(__file__).parents[1]
    report = json.loads((root / 'data/compiled-kernel-package-validation.json').read_text())
    manifest = report['manifest']
    assert manifest['platform'] == {'system': 'win32', 'machine': 'x86_64', 'pointer_bits': 64}
    assert manifest['source'] == package._source_identity()
    assert manifest['self_test'] == {'golden_comparisons': 435, 'guard_checks': 10,
                                     'formats': ['Q4_K', 'Q5_K', 'Q6_K']}
    assert report['corrupt_library_rejected_before_load']
    assert len(report['cli']) == 4 and all(row['equal_file_bytes'] for row in report['cli'])
    assert {row['preset'] for row in report['cli']} == {'Q6_K', 'Q5_K_M', 'Q4_K_S', 'Q4_K_M'}
    for name, digest in report['source_code_sha256'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize('alias', ['manifest', 'library', 'hardlink'])
def test_writer_rejects_package_output_alias_before_loading(kernel_package, tmp_path, monkeypatch, alias):
    """Even explicit overwrite must preserve a selected package and hardlink aliases."""
    from scripts.requantize_flux_q6 import main
    directory, manifest = kernel_package
    output = directory / ('kernel.json' if alias == 'manifest' else manifest['library'])
    if alias == 'hardlink':
        output = tmp_path / 'outside.gguf'
        output.hardlink_to(directory / manifest['library'])
    before = output.read_bytes()

    def forbidden(*_):
        pytest.fail('An alias must be rejected before loading the native package')

    monkeypatch.setattr(package, 'load_package', forbidden)
    monkeypatch.setattr('sys.argv', ['requantize', '--source', str(tmp_path / 'missing.gguf'),
                                   '--output', str(output), '--overwrite', '--kernel-package', str(directory)])
    with pytest.raises(ValueError, match='different files'):
        main()
    assert output.read_bytes() == before
