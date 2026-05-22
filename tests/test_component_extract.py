"""Tests for SDXL component analysis and extraction."""

from __future__ import annotations

import torch
from safetensors.torch import load_file, save_file

from component_extract import analyze_components, extract_components, format_component_analysis


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
    assert results["clip_l"].status == "differs from local standard"
    assert "matches local standard" in format_component_analysis([results["vae"]])


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
