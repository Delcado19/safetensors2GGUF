"""Reject incompatible or altered native packages before loading executable code."""
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from scripts import compiled_kernel_package as package


def _qualified_source_hashes(root):
    # Keep historical measurements immutable; each follow-up qualifies changed sources.
    hashes = {}
    for name in ('compiled-kernel-levels-validation', 'compiled-kernel-performance-validation',
                 'compiled-kernel-direct-preparation', 'compiled-diffusion-validation',
                 'compiled-diffusion-requalification-2026-10-19',
                 'tokenizer-check-validation-2026-10-19',
                 'compiled-diffusion-progress-requalification-2026-10-19'):
        hashes.update(json.loads((root / f'data/{name}.json').read_text())['source_code_sha256'])
    return hashes


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
    # Historical reports retain their original hashes; current adapters are requalified below.
    current = _qualified_source_hashes(root)
    for name, digest in report['source_code_sha256'].items():
        expected = current.get(name, digest)
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected


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



def test_current_kernel_levels_record_independent_native_file_parity():
    root = Path(__file__).parents[1]
    report = json.loads((root / 'data/compiled-kernel-levels-validation.json').read_text())
    assert report['all_equal']
    assert report['manifest']['self_test'] == {
        'golden_comparisons': 725, 'guard_checks': 10,
        'formats': ['Q2_K', 'Q3_K', 'Q4_K', 'Q5_K', 'Q6_K']}
    assert {r['preset'] for r in report['paths']} == {
        'Q2_K', 'Q3_K_M', 'Q4_K_S', 'Q4_K_M', 'Q5_K_M', 'Q6_K'}
    for row in report['paths']:
        assert row['all_equal'] and len(row['results']) == 8
        assert all(t['equal'] for case in row['results'] for t in case['tensors'])
        assert all(case['metadata_equal'] and case['alignment_equal'] for case in row['results'])
    assert set(report['api_fixture_native']) == {'Q2_K', 'Q3_K_M'}
    # Preserve the original single-worker evidence; current writer qualification follows it.
    current = _qualified_source_hashes(root)
    for name, digest in report['source_code_sha256'].items():
        # UI hashes are historical provenance, not inputs to native writer parity.
        if name.startswith('frontend/'):
            continue
        expected = current.get(name, digest)
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
    for term in ('vcruntime', 'msvcp', 'ucrtbase', 'api-ms-win-crt'):
        assert term not in report['dll_dependencies'].lower()


@pytest.mark.parametrize('bits', [2, 3])
def test_shipped_low_bit_kernel_rejects_invalid_buffers_without_writing(bits):
    import numpy as np
    if package._platform() != {'system': 'win32', 'machine': 'x86_64', 'pointer_bits': 64}:
        pytest.skip('Shipped Windows x86_64 package qualification')
    directory = package.ROOT / 'third_party/cpu-kernels/win32-x86_64'
    manifest = package.verify_package(directory)
    library, encode = package._load(directory / manifest['library'], manifest['files'][manifest['library']])
    values = np.ones(256, dtype=np.float32)
    output = np.full({2: 84, 3: 110}[bits], 0xA5, dtype=np.uint8)
    assert encode(bits, values.ctypes.data, 255, output.ctypes.data, output.size) == 1
    assert encode(bits, values.ctypes.data, 256, output.ctypes.data, output.size - 1) == 1
    assert np.all(output == 0xA5)
    values[0] = np.nan
    assert encode(bits, values.ctypes.data, 256, output.ctypes.data, output.size) == 2
    assert np.all(output == 0xA5)
    assert library is not None



def test_parallel_writer_evidence_retains_native_equality_and_current_sources():
    root = Path(__file__).parents[1]
    report = json.loads((root / 'data/compiled-kernel-performance-validation.json').read_text())
    assert report['source_stat_unchanged']
    assert {r['preset'] for r in report['results']} == {
        'Q2_K', 'Q3_K_M', 'Q4_K_S', 'Q4_K_M', 'Q5_K_M', 'Q6_K'}
    for row in report['results']:
        assert row['serial_parallel_file_bytes_equal'] and row['native_comparison']['all_equal']
        assert len(row['trials']['serial']) == len(row['trials']['parallel']) == 3
    assert any('F16 preparation/write:' in line for line in report['application_phase_logs'])
    assert any('readback_seconds:' in line for line in report['application_phase_logs'])
    current = _qualified_source_hashes(root)
    for name, digest in report['source_code_sha256'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == current.get(name, digest)



def test_direct_spool_evidence_preserves_bytes_and_reduces_logical_writes():
    root = Path(__file__).parents[1]
    report = json.loads((root / 'data/compiled-kernel-direct-preparation.json').read_text())
    assert report['source_stat_unchanged'] and len(report['results']) == 6
    for row in report['results']:
        assert row['retained_direct_file_bytes_equal'] and row['native_comparison']['all_equal']
        assert row['logical_payload_writes']['direct'] < row['logical_payload_writes']['retained_float']
        assert len(row['trials_seconds']['direct']) == len(row['trials_seconds']['retained_float']) == 3
        assert any('no float GGUF intermediate' in line for line in row['phase_logs']['direct'])
    current = _qualified_source_hashes(root)
    for name, digest in report['source_code_sha256'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == current.get(name, digest)
