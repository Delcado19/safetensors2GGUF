"""Tests for SDXL component analysis and extraction."""

from __future__ import annotations

import torch
import pytest
import threading
from safetensors.torch import load_file, save_file

from component_extract import analyze_components, extract_components, format_component_analysis
import component_extract
from convert import ConversionCancelled


def _write_minimal_sdxl_checkpoint(path):
    tensors = {
        "first_stage_model.decoder.conv_in.weight": torch.ones(1, 1),
        "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight": torch.ones(2, 2),
        "conditioner.embedders.0.transformer.text_model.embeddings.position_ids": torch.arange(2),
        "conditioner.embedders.1.model.token_embedding.weight": torch.arange(4, dtype=torch.float32).reshape(2, 2),
        "conditioner.embedders.1.model.positional_embedding": torch.ones(2, 2),
        "conditioner.embedders.1.model.ln_final.weight": torch.ones(2),
        "conditioner.embedders.1.model.ln_final.bias": torch.zeros(2),
        "conditioner.embedders.1.model.text_projection": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_weight": torch.arange(
            12,
            dtype=torch.float32,
        ).reshape(6, 2),
        "conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_bias": torch.arange(
            6,
            dtype=torch.float32,
        ),
        "conditioner.embedders.1.model.transformer.resblocks.0.attn.out_proj.weight": torch.ones(2, 2),
        "conditioner.embedders.1.model.transformer.resblocks.0.attn.out_proj.bias": torch.zeros(2),
        "conditioner.embedders.1.model.transformer.resblocks.0.ln_1.weight": torch.ones(2),
        "conditioner.embedders.1.model.transformer.resblocks.0.ln_1.bias": torch.zeros(2),
        "conditioner.embedders.1.model.transformer.resblocks.0.ln_2.weight": torch.ones(2),
        "conditioner.embedders.1.model.transformer.resblocks.0.ln_2.bias": torch.zeros(2),
        "conditioner.embedders.1.model.transformer.resblocks.0.mlp.c_fc.weight": torch.ones(4, 2),
        "conditioner.embedders.1.model.transformer.resblocks.0.mlp.c_fc.bias": torch.zeros(4),
        "conditioner.embedders.1.model.transformer.resblocks.0.mlp.c_proj.weight": torch.ones(2, 4),
        "conditioner.embedders.1.model.transformer.resblocks.0.mlp.c_proj.bias": torch.zeros(2),
    }
    save_file(tensors, path)


def test_extraction_preflight_and_atomic_cancellation(tmp_path, monkeypatch):
    """Later conflicts prevent any export; cancellation never replaces an output."""
    checkpoint = tmp_path / 'model.safetensors'
    _write_minimal_sdxl_checkpoint(checkpoint)
    clip = tmp_path / 'clip'
    clip.mkdir()
    existing = clip / 'model-clip_g.safetensors'
    existing.write_bytes(b'keep')
    with pytest.raises(OSError, match='overwrite'):
        extract_components(checkpoint)
    assert not (clip / 'model-clip_l.safetensors').exists()
    cancel = threading.Event()
    original_save = component_extract.save_file
    def cancelling_save(tensors, path):
        original_save(tensors, path)
        cancel.set()
    monkeypatch.setattr(component_extract, 'save_file', cancelling_save)
    with pytest.raises(ConversionCancelled):
        extract_components(checkpoint, overwrite=True, cancel_event=cancel)
    assert existing.read_bytes() == b'keep'
    assert not (clip / 'model-clip_l.safetensors').exists()
    assert not list(clip.glob('.*.safetensors'))
    monkeypatch.setattr(component_extract, 'save_file', original_save)
    cancel.clear()
    completed = []
    def cancel_after_publication(path):
        completed.append(path)
        cancel.set()
    with pytest.raises(ConversionCancelled):
        extract_components(checkpoint, extract_vae=True, overwrite=True,
                           cancel_event=cancel, on_output=cancel_after_publication)
    assert len(completed) == 1
    assert (tmp_path / 'vae' / 'model-vae.safetensors').is_file()
    assert existing.read_bytes() == b'keep'


def test_analyze_components_marks_exact_vae_and_different_clip_l(tmp_path):
    checkpoint = tmp_path / "models" / "checkpoints" / "model.safetensors"
    checkpoint.parent.mkdir(parents=True)
    _write_minimal_sdxl_checkpoint(checkpoint)

    vae_dir = tmp_path / "models" / "vae"
    clip_dir = tmp_path / "models" / "clip"
    vae_dir.mkdir()
    clip_dir.mkdir()
    save_file({"decoder.conv_in.weight": torch.ones(1, 1)}, vae_dir / "sdxlVAE.safetensors")
    save_file(
        {"text_model.embeddings.token_embedding.weight": torch.zeros(2, 2)},
        clip_dir / "clip_l.safetensors",
    )

    results = {item.name: item for item in analyze_components(checkpoint)}

    assert results["vae"].is_exact_standard
    assert results["clip_l"].status == "differs from local reference"
    assert "matches local reference" in format_component_analysis([results["vae"]])


def test_extract_components_writes_selected_files_and_maps_clip_g(tmp_path):
    checkpoint = tmp_path / "models" / "checkpoints" / "model.safetensors"
    checkpoint.parent.mkdir(parents=True)
    _write_minimal_sdxl_checkpoint(checkpoint)

    written = extract_components(
        checkpoint,
        extract_vae=False,
        extract_clip_l=True,
        extract_clip_g=True,
    )

    assert {item.name for item in written} == {"clip_l", "clip_g"}

    clip_l = load_file(tmp_path / "models" / "clip" / "model-clip_l.safetensors")
    assert set(clip_l) == {"text_model.embeddings.token_embedding.weight"}

    clip_g = load_file(tmp_path / "models" / "clip" / "model-clip_g.safetensors")
    assert torch.equal(
        clip_g["text_model.encoder.layers.0.self_attn.q_proj.weight"],
        torch.tensor([[0.0, 1.0], [2.0, 3.0]]),
    )
    assert torch.equal(
        clip_g["text_model.encoder.layers.0.self_attn.k_proj.bias"],
        torch.tensor([2.0, 3.0]),
    )
    assert torch.equal(
        clip_g["text_projection.weight"],
        torch.tensor([[1.0, 3.0], [2.0, 4.0]]),
    )


def test_component_hash_identity_and_safe_reuse(tmp_path):
    """Packaging changes match; dtype/extra tensors never qualify for reuse."""
    checkpoint = tmp_path / 'model.safetensors'
    _write_minimal_sdxl_checkpoint(checkpoint)
    reference = tmp_path / 'vae' / 'sdxlVAE.safetensors'
    reference.parent.mkdir()
    save_file({'decoder.conv_in.weight': torch.ones(1, 1)}, reference,
              metadata={'irrelevant': 'different file packaging'})
    analysis = analyze_components(checkpoint, components=('vae',))[0]
    assert analysis.component_hash == analysis.reference_hash
    original = reference.read_bytes()
    result = extract_components(checkpoint, extract_vae=True, extract_clip_l=False,
                                extract_clip_g=False, reuse_identical=True)
    assert result[0].reused and result[0].path == str(reference)
    assert not (reference.parent / 'model-vae.safetensors').exists()
    assert reference.read_bytes() == original
    save_file({'decoder.conv_in.weight': torch.ones(1, 1).half()}, reference)
    assert not analyze_components(checkpoint, components=('vae',))[0].is_exact_standard
    result = extract_components(checkpoint, extract_vae=True, extract_clip_l=False,
                                extract_clip_g=False, reuse_identical=True)
    assert not result[0].reused
    save_file({'decoder.conv_in.weight': torch.ones(1, 1), 'extra': torch.ones(1)}, reference)
    assert not analyze_components(checkpoint, components=('vae',))[0].is_exact_standard


def test_unreadable_reference_does_not_block_export(tmp_path):
    checkpoint = tmp_path / 'model.safetensors'
    _write_minimal_sdxl_checkpoint(checkpoint)
    reference = tmp_path / 'vae' / 'sdxlVAE.safetensors'
    reference.parent.mkdir()
    reference.write_bytes(b'not a safetensors file')
    observed = []
    exported = extract_components(checkpoint, extract_vae=True, extract_clip_l=False,
                                  extract_clip_g=False, reuse_identical=True, on_analysis=observed.extend)
    assert not exported[0].reused
    assert observed[0].status == 'unknown: local reference unreadable'
    assert observed[0].reference_hash is None


def test_stock_recognition_without_local_reference_and_modified_rejection(tmp_path, monkeypatch):
    """Recognition alone never skips a needed export; changed values lose identity."""
    source = tmp_path / 'embedded.safetensors'
    weight = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    save_file({'first_stage_model.decoder.weight': weight}, source)
    from component_extract import _component_hash, _tensor_hash
    fingerprint = _component_hash({'decoder.weight': _tensor_hash(weight)})
    monkeypatch.setitem(component_extract.STOCK_COMPONENTS, 'vae', {
        'variants': [{'component_hash': fingerprint, 'label': 'verified fixture FP32'}]})
    item = analyze_components(source, components=('vae',))[0]
    assert item.stock_match == 'verified fixture FP32' and not item.has_reference
    assert item.status == 'recognized stock: verified fixture FP32'
    exported = extract_components(source, extract_vae=True, extract_clip_l=False,
                                  extract_clip_g=False, reuse_identical=True)
    assert not exported[0].reused
    assert load_file(exported[0].path)['decoder.weight'].equal(weight)
    save_file({'first_stage_model.decoder.weight': weight + 1}, source)
    assert analyze_components(source, components=('vae',))[0].stock_match is None


def test_diffusers_vae_reference_matches_embedded_ldm_layout(tmp_path):
    source = tmp_path / 'model.safetensors'
    attention = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    save_file({'first_stage_model.decoder.mid.attn_1.q.weight': attention[:, :, None, None],
               'first_stage_model.decoder.up.3.block.0.nin_shortcut.weight': attention.clone()}, source)
    reference = tmp_path / 'vae' / 'sdxlVAE.safetensors'
    reference.parent.mkdir()
    save_file({'decoder.mid_block.attentions.0.to_q.weight': attention,
               'decoder.up_blocks.0.resnets.0.conv_shortcut.weight': attention.clone()}, reference)
    item = analyze_components(source, components=('vae',))[0]
    assert item.is_exact_standard and item.component_hash == item.reference_hash
    reused = extract_components(source, extract_vae=True, extract_clip_l=False,
                                extract_clip_g=False, reuse_identical=True)
    assert reused[0].reused
