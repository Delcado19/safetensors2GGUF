"""Complete VL/CLIP/audio tensor preservation and companion node contracts."""
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from comfyui_s2gguf import S2GGUFTextEncoderLoader, NODE_CLASS_MAPPINGS
from comfyui_s2gguf.codec import load_original_text
from original_text_gguf import export_original_text
from scripts.gguf_research_io import open_gguf


def fixture_source(path):
    data = {'model.language_model.embed_tokens.weight': torch.ones(8, 256),
            'model.language_model.layers.0.self_attn.q_proj.weight': torch.linspace(-1, 1, 8192).reshape(32, 256),
            'model.language_model.layers.0.self_attn.v_proj.weight': torch.ones(32, 256),
            'model.visual.blocks.0.attn.qkv.weight': torch.linspace(-1, 1, 8192).reshape(32, 256),
            'model.visual.patch_embed.proj.weight': torch.ones(2, 3, 2, 4, 4),
            'model.visual.deepstack_merger_list.0.linear_fc2.weight': torch.ones(32, 256),
            'tokenizer_json': torch.arange(32, dtype=torch.uint8),
            'indices': torch.tensor(7, dtype=torch.int64),
            'empty_buffer': torch.empty(0),
            'empty_matrix': torch.empty(2, 0),
            'empty_rows': torch.empty(0, 256),
            'scalar': torch.tensor(0.5)}
    save_file(data, str(path))
    return data


@pytest.mark.parametrize('preset', ['F32', 'F16', 'BF16', 'Q8_0', 'Q2_K', 'Q3_K_M', 'Q4_K_S', 'Q4_K_M', 'Q5_K_M', 'Q6_K'])
def test_complete_vl_export_keeps_vision_and_original_shapes(tmp_path, preset):
    source, output = tmp_path / 'vl.safetensors', tmp_path / 'vl.gguf'
    expected = fixture_source(source)
    original = source.read_bytes()
    export_original_text(source, output, 'qwen3-vl', preset)
    actual = load_original_text(output)
    assert actual.keys() == expected.keys()
    for name, tensor in actual.items():
        assert tensor.shape == expected[name].shape and torch.isfinite(tensor).all()
    assert torch.equal(actual['tokenizer_json'], expected['tokenizer_json'])
    assert torch.equal(actual['indices'], expected['indices'])
    for name in ('empty_buffer', 'empty_matrix', 'empty_rows'):
        assert torch.equal(actual[name], expected[name])
    assert torch.equal(actual['model.visual.patch_embed.proj.weight'], expected['model.visual.patch_embed.proj.weight'])
    if preset == 'F32':
        for name in actual:
            assert torch.equal(actual[name], expected[name])
    with open_gguf(output) as reader:
        assert reader.get_field('comfy.s2g.vision_tensor_count').contents() == 3
        assert reader.get_field('comfy.s2g.loader').contents() == 'S2GGUFTextEncoderLoader'
    assert source.read_bytes() == original


@pytest.mark.parametrize('family', ['clip-l', 'clip-bigg', 'qwen2.5-vl-7b', 'minimax-music3', 'ernie_image'])
def test_other_original_layout_families_roundtrip(tmp_path, family):
    source, output = tmp_path / 'source.safetensors', tmp_path / 'out.gguf'
    expected = fixture_source(source)
    messages = []
    export_original_text(source, output, family, 'Q4_K_M', on_log=messages.append)
    assert set(load_original_text(output)) == set(expected)
    assert ('Original Diffusion Model' if family == 'ernie_image' else 'Original Text / Vision Encoder') in messages[-1]


def test_progress_reports_encoding_writing_and_readback_without_changing_bytes(tmp_path):
    source = tmp_path / 'source.safetensors'
    fixture_source(source)
    baseline, observed = tmp_path / 'baseline.gguf', tmp_path / 'observed.gguf'
    export_original_text(source, baseline, 'ernie_image', 'Q4_K_M')
    events = []
    export_original_text(source, observed, 'ernie_image', 'Q4_K_M',
                         on_progress=lambda fraction, phase: events.append((fraction, phase)))
    assert baseline.read_bytes() == observed.read_bytes()
    assert events[0][0] == 0 and events[-1][0] == 1
    assert [fraction for fraction, _ in events] == sorted(fraction for fraction, _ in events)
    assert any('Encoding tensor' in phase for _, phase in events)
    assert any('Writing GGUF' in phase for _, phase in events)
    assert any('readback' in phase for _, phase in events)


def test_diffusion_pipeline_forwards_original_export_progress(tmp_path, monkeypatch):
    import queue
    import conversion_service
    source, output = tmp_path / 'source.safetensors', tmp_path / 'out.gguf'
    fixture_source(source)
    monkeypatch.setattr(conversion_service, 'detect_arch', lambda _: SimpleNamespace(arch='ernie_image'))
    events = queue.Queue()
    conversion_service._pipeline(str(source), str(output), 'Q4_K_M', '', 0, False, False, events)
    recorded = list(events.queue)
    assert any(event[0] == 'progress_frac' and 'Encoding tensor' in event[2] for event in recorded)
    assert any(event[0] == 'progress_frac' and event[1] == 1 for event in recorded)


def test_cancellation_and_corruption_preserve_previous_output(tmp_path):
    from convert import ConversionCancelled
    source, output = tmp_path / 'vl.safetensors', tmp_path / 'out.gguf'
    fixture_source(source)
    output.write_bytes(b'previous model')
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(ConversionCancelled):
        export_original_text(source, output, 'qwen3-vl', 'F16', overwrite=True, cancel_event=cancel)
    assert output.read_bytes() == b'previous model'
    assert not list(tmp_path.glob('.out.*'))


def test_companion_node_passes_complete_state_to_native_comfyui(tmp_path, monkeypatch):
    source, output = tmp_path / 'vl.safetensors', tmp_path / 'out.gguf'
    expected = fixture_source(source)
    export_original_text(source, output, 'qwen3-vl', 'F16')
    captured = {}
    def load(**kwargs):
        captured.update(kwargs)
        return 'clip-result'
    fake_sd = SimpleNamespace(CLIPType={'KREA2': 'krea2'}, load_text_encoder_state_dicts=load)
    monkeypatch.setitem(sys.modules, 'comfy', SimpleNamespace(sd=fake_sd))
    monkeypatch.setitem(sys.modules, 'comfy.sd', fake_sd)
    monkeypatch.setitem(sys.modules, 'folder_paths', SimpleNamespace(
        folder_names_and_paths={},
        get_full_path_or_raise=lambda category, name: str(output), get_folder_paths=lambda name: []))
    assert S2GGUFTextEncoderLoader().load('out.gguf', 'krea2') == ('clip-result',)
    assert set(captured['state_dicts'][0]) == set(expected)
    assert captured['clip_type'] == 'krea2'
    assert set(NODE_CLASS_MAPPINGS) == {'S2GGUFTextEncoderLoader', 'S2GGUFDiffusionLoader'}


def test_companion_selector_finds_gguf_without_changing_native_extensions(tmp_path, monkeypatch):
    from comfyui_s2gguf import S2GGUFDiffusionLoader
    native_extensions = {'.safetensors'}
    registry = {'diffusion_models': ([str(tmp_path)], native_extensions)}
    def names(category):
        paths, extensions = registry[category]
        return [path.name for folder in paths for path in Path(folder).iterdir()
                if path.suffix in extensions]
    (tmp_path / 'ernie.gguf').touch()
    (tmp_path / 'ernie.safetensors').touch()
    monkeypatch.setitem(sys.modules, 'folder_paths', SimpleNamespace(
        folder_names_and_paths=registry, get_folder_paths=lambda category: registry[category][0],
        get_filename_list=names))
    assert S2GGUFDiffusionLoader.INPUT_TYPES()['required']['model'][0] == ['ernie.gguf']
    assert registry['diffusion_models'][1] == {'.safetensors'}


def test_vl_detected_before_plain_qwen_signature(tmp_path):
    from text_encoder_convert import detect_text_encoder_family
    from convert import load_state_dict
    source = tmp_path / 'vl.safetensors'
    fixture_source(source)
    assert detect_text_encoder_family(load_state_dict(str(source), strip_prefixes=False)) == 'qwen3-vl'


def test_flat_qwen3vl_language_namespace_keeps_full_vl_routing():
    from text_encoder_convert import detect_text_encoder_family
    # The installed INT8 Qwen3-VL-8B repack uses the plain-Qwen3 text signature.
    state = {'model.embed_tokens.weight': torch.empty(151936, 4096, device='meta'),
             'model.visual.merger.linear_fc2.weight': torch.empty(4096, 4608, device='meta')}
    state.update({f'model.layers.{index}.self_attn.q_proj.weight': torch.empty(4096, 4096, device='meta')
                  for index in range(36)})
    assert detect_text_encoder_family(state) == 'qwen3-vl'
    del state['model.visual.merger.linear_fc2.weight']
    assert detect_text_encoder_family(state) == 'qwen3-8b'


def test_vision_only_qwen_source_is_not_a_complete_vl_encoder():
    from text_encoder_convert import detect_text_encoder_family
    with pytest.raises(ValueError, match='without language embeddings'):
        detect_text_encoder_family({'model.visual.merger.linear_fc2.weight': torch.empty(2, 2)})


def test_public_flat_vl_export_retains_vision_without_plain_text_converter(tmp_path, monkeypatch):
    from text_encoder_convert import convert_text_encoder_kquant
    source, output = tmp_path / 'flat-vl.safetensors', tmp_path / 'flat-vl.gguf'
    state = fixture_source(source)
    state = {key.replace('model.language_model.', 'model.'): value for key, value in state.items()}
    save_file(state, str(source))
    monkeypatch.setattr('text_encoder_convert.ensure_plain_llama_quantize',
                        lambda **kwargs: pytest.fail('Full VL must not use the plain text quantizer'))
    convert_text_encoder_kquant(str(source), '', str(output))
    restored = load_original_text(output)
    assert set(restored) == set(state)
    assert 'model.visual.deepstack_merger_list.0.linear_fc2.weight' in restored
