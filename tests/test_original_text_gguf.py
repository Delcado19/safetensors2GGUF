"""Complete VL/CLIP/audio tensor preservation and companion node contracts."""
import sys
import threading
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
    export_original_text(source, output, family, 'Q4_K_M')
    assert set(load_original_text(output)) == set(expected)


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
        get_full_path_or_raise=lambda category, name: str(output), get_folder_paths=lambda name: []))
    assert S2GGUFTextEncoderLoader().load('out.gguf', 'krea2') == ('clip-result',)
    assert set(captured['state_dicts'][0]) == set(expected)
    assert captured['clip_type'] == 'krea2'
    assert set(NODE_CLASS_MAPPINGS) == {'S2GGUFTextEncoderLoader', 'S2GGUFDiffusionLoader'}


def test_vl_detected_before_plain_qwen_signature(tmp_path):
    from text_encoder_convert import detect_text_encoder_family
    from convert import load_state_dict
    source = tmp_path / 'vl.safetensors'
    fixture_source(source)
    assert detect_text_encoder_family(load_state_dict(str(source), strip_prefixes=False)) == 'qwen3-vl'
