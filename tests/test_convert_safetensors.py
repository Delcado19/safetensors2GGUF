"""Tests for convert_safetensors.py — safetensors-to-safetensors quantized output."""

from __future__ import annotations

import torch
from safetensors.torch import load_file, save_file

from convert_safetensors import convert_to_safetensors


def _write_minimal_flux(tmp_path):
    src = tmp_path / "model.safetensors"
    sd = {
        "double_blocks.0.img_attn.proj.weight": torch.randn(64, 64, dtype=torch.float32),
        "double_blocks.0.img_attn.proj.bias": torch.randn(64, dtype=torch.float32),
    }
    save_file(sd, str(src))
    return src


class TestConvertToSafetensors:
    def test_writes_output_file(self, tmp_path):
        src = _write_minimal_flux(tmp_path)
        dst, arch = convert_to_safetensors(str(src), target_key="F16", overwrite=True)
        assert dst.endswith(".safetensors")
        import os
        assert os.path.isfile(dst)
        assert arch is not None and arch.arch == "flux"

    def test_output_tensor_dtype_matches_target(self, tmp_path):
        src = _write_minimal_flux(tmp_path)
        dst, _ = convert_to_safetensors(str(src), target_key="F16", overwrite=True)
        out = load_file(dst)
        assert out["double_blocks.0.img_attn.proj.weight"].dtype == torch.float16

    def test_fp8_output_includes_scale_tensors(self, tmp_path):
        src = _write_minimal_flux(tmp_path)
        dst, _ = convert_to_safetensors(str(src), target_key="FP8", overwrite=True)
        out = load_file(dst)
        assert "double_blocks.0.img_attn.proj.weight_scale" in out

    def test_fp8_non_mixed_bias_has_no_scale_sibling(self, tmp_path):
        # Regression for review finding #2: non-mixed FP8 must not scale-
        # quantize 1D bias tensors — no consumer reads a bias-scale, so the
        # bias would load unscaled and be wrong by up to ~448x.
        src = _write_minimal_flux(tmp_path)
        dst, _ = convert_to_safetensors(str(src), target_key="FP8", overwrite=True)
        out = load_file(dst)
        assert "double_blocks.0.img_attn.proj.bias.weight_scale" not in out
        assert out["double_blocks.0.img_attn.proj.bias"].dtype == torch.float32

    def test_quantization_metadata_written(self, tmp_path):
        src = _write_minimal_flux(tmp_path)
        dst, _ = convert_to_safetensors(str(src), target_key="FP8", overwrite=True)
        from safetensors import safe_open
        with safe_open(dst, framework="pt") as f:
            meta = f.metadata()
        assert meta is not None and "_quantization_metadata" in meta

    def test_quantization_metadata_layer_key_has_no_weight_suffix(self, tmp_path):
        # Regression: ComfyUI's convert_old_quants() writes the comfy_quant
        # sidecar as "{layer}.comfy_quant" where `layer` is the module prefix
        # WITHOUT a trailing "weight" component. A "layers" dict keyed by the
        # full tensor name (".../proj.weight") produces a sidecar tensor name
        # ComfyUI's loader never looks up, silently discarding weight_scale
        # and leaving the model to load the raw quantized bytes unscaled.
        import json

        src = _write_minimal_flux(tmp_path)
        dst, _ = convert_to_safetensors(str(src), target_key="FP8", overwrite=True)
        from safetensors import safe_open
        with safe_open(dst, framework="pt") as f:
            meta = f.metadata()
        layers = json.loads(meta["_quantization_metadata"])["layers"]
        assert "double_blocks.0.img_attn.proj" in layers
        assert "double_blocks.0.img_attn.proj.weight" not in layers

    def test_refuses_overwrite_without_flag(self, tmp_path):
        src = _write_minimal_flux(tmp_path)
        dst_path = str(tmp_path / "out.safetensors")
        convert_to_safetensors(str(src), dst_path=dst_path, target_key="F16", overwrite=True)
        try:
            convert_to_safetensors(str(src), dst_path=dst_path, target_key="F16", overwrite=False)
            assert False, "expected OSError"
        except OSError:
            pass

    def test_float8_input_coerced_to_float16(self, tmp_path):
        """Regression: float8 inputs must be coerced to float16 before nan_to_num.

        Verify that re-quantizing a checkpoint that's already in float8 format
        (e.g. pre-quantized releases) does not raise NotImplementedError from nan_to_num.
        """
        src = tmp_path / "model.safetensors"
        sd = {
            "double_blocks.0.img_attn.proj.weight": torch.randn(64, 64, dtype=torch.float32).to(
                torch.float8_e4m3fn
            ) if hasattr(torch, "float8_e4m3fn") else torch.randn(64, 64, dtype=torch.float32),
            "double_blocks.0.img_attn.proj.bias": torch.randn(64, dtype=torch.float32),
        }
        save_file(sd, str(src))
        # Should not raise NotImplementedError
        dst, _ = convert_to_safetensors(str(src), target_key="F16", overwrite=True)
        out = load_file(dst)
        assert out["double_blocks.0.img_attn.proj.weight"].dtype == torch.float16
