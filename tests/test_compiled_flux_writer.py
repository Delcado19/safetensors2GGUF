"""Direct compiled spooling must preserve legacy preparation and atomic safety."""
import queue
import threading

import pytest
import torch
from safetensors.torch import save_file

from conversion_service import _pipeline
from convert import ConversionCancelled
from scripts.compiled_kernel_package import ROOT, _platform, load_package
from scripts import requantize_flux_q6 as quant


def _source(path, dtype):
    tensors = {
        'double_blocks.0.img_attn.proj.weight': torch.linspace(-2, 3, 257 * 256).reshape(257, 256),
        'img_in.weight': torch.ones(8, 256),
        'norm.weight': torch.ones(256),
        'odd.weight': torch.ones(8, 160),
        'spatial.weight': torch.ones(2, 3, 256),
        'a.to_v.weight': torch.ones(8, 160),
        **{f'b.{index}.to_v.weight': torch.ones(8, 256) for index in range(6)},
        'block.attn.qkv.weight': torch.ones(8, 256),
        'block.ffn_down.weight': torch.ones(8, 256),
    }
    if dtype == 'scaled_fp8':
        tensors = {key: value.to(torch.float8_e4m3fn) for key, value in tensors.items()}
        tensors['double_blocks.0.img_attn.proj.weight_scale'] = torch.tensor(0.5)
    else:
        tensors = {key: value.to(getattr(torch, dtype)) for key, value in tensors.items()}
    save_file(tensors, str(path))


@pytest.fixture(autouse=True)
def windows_package():
    if _platform() != {'system': 'win32', 'machine': 'x86_64', 'pointer_bits': 64}:
        pytest.skip('Shipped Windows x86_64 package qualification')


@pytest.mark.parametrize('dtype', ['float32', 'float16', 'bfloat16', 'float8_e4m3fn', 'scaled_fp8'])
@pytest.mark.parametrize('preset', quant.FLUX_PRESETS)
def test_direct_matches_retained_float_intermediate_bytes(tmp_path, dtype, preset):
    source = tmp_path / 'flux.safetensors'
    _source(source, dtype)
    legacy, direct = tmp_path / 'legacy.gguf', tmp_path / 'direct.gguf'
    original = source.read_bytes()
    _pipeline(str(source), str(legacy), preset, '', 0, True, False, queue.Queue(), gguf_backend='compiled_flux')
    event = queue.Queue()
    _pipeline(str(source), str(direct), preset, '', 0, False, False, event, gguf_backend='compiled_flux')
    assert legacy.read_bytes() == direct.read_bytes()
    assert source.read_bytes() == original
    assert len(list(tmp_path.glob('*-F16-tmp-*.gguf'))) == 1
    assert any('no float GGUF intermediate' in e[1] for e in event.queue if e[0] == 'log')


@pytest.mark.parametrize('failure', ['worker', 'cancel', 'write', 'readback', 'source_change'])
def test_direct_failure_preserves_previous_output_and_cleans_spool(tmp_path, monkeypatch, failure):
    import compiled_flux_writer as direct
    import gguf
    source, output = tmp_path / 'flux.safetensors', tmp_path / 'out.gguf'
    _source(source, 'float32')
    output.write_bytes(b'previous output')
    event = threading.Event()
    kernel, manifest = load_package(ROOT / 'third_party/cpu-kernels/win32-x86_64')
    def callback(data, bits):
        if failure == 'worker':
            raise RuntimeError('worker failure')
        if failure == 'cancel':
            event.set()
        if failure == 'source_change':
            import os
            stat = source.stat()
            os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000000))
        return kernel(data, bits)
    monkeypatch.setattr('scripts.compiled_kernel_package.load_package', lambda *_: (callback, manifest))
    if failure == 'write':
        def write_fail(*args, **kwargs):
            raise OSError('disk full')
        monkeypatch.setattr(gguf.GGUFWriter, 'write_tensors_to_file', write_fail)
    if failure == 'readback':
        def bad_reader(*args):
            raise ValueError('readback failure')
        monkeypatch.setattr(direct, 'open_gguf', bad_reader)
    expected = {'worker': RuntimeError, 'cancel': ConversionCancelled, 'write': OSError,
                'readback': ValueError, 'source_change': ValueError}[failure]
    with pytest.raises(expected):
        _pipeline(str(source), str(output), 'Q2_K', '', 0, False, True, queue.Queue(),
                  event, gguf_backend='compiled_flux')
    assert output.read_bytes() == b'previous output'
    assert {p.name for p in tmp_path.iterdir()} == {source.name, output.name}



def test_complete_direct_q2_run_reproduces_previous_model_bytes():
    import hashlib
    import json
    from pathlib import Path
    root = Path(__file__).parents[1]
    direct = json.loads((root / 'data/flux2-q2-direct-conversion.json').read_text())
    previous = json.loads((root / 'data/flux2-q2-bundled-conversion.json').read_text())
    assert direct['status'] == 'succeeded' and direct['matches_previous_q2_output']
    assert direct['source_sha256'] == previous['source_sha256']
    assert direct['output_sha256'] == previous['output_sha256']
    assert direct['output_bytes'] == previous['output_bytes'] == 3356285248
    assert direct['tensor_types'] == {'F32': 86, 'Q2_K': 115}
    assert direct['tensor_count'] == 201 and direct['source_stat_unchanged']
    assert any('no float GGUF intermediate' in line for line in direct['phase_log_lines'])
    for name, digest in direct['source_code_sha256'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
