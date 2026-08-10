"""Tests for convert_safetensors.py — safetensors-to-safetensors quantized output."""

from __future__ import annotations

import pytest
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

    def test_on_log_emits_a_line_per_tensor(self, tmp_path):
        """GUI regression: the Convert -> Safetensors tab's log box only showed
        the arch/writing/done lines, never which tensor was being processed
        (unlike the GGUF tab's handle_tensors, which logs per-tensor).
        on_log must fire once per non-ignored tensor by default."""
        src = _write_minimal_flux(tmp_path)
        logged: list[str] = []
        convert_to_safetensors(str(src), target_key="F16", overwrite=True, on_log=logged.append)
        tensor_lines = [
            line for line in logged
            if "double_blocks.0.img_attn.proj.weight" in line and "img_attn.proj.weight_scale" not in line
        ]
        assert len(tensor_lines) == 1

    def test_log_tensor_every_throttles_but_keeps_first_and_last(self, tmp_path):
        src = tmp_path / "model.safetensors"
        sd = {
            f"double_blocks.{i}.img_attn.proj.weight": torch.randn(64, 64, dtype=torch.float32)
            for i in range(5)
        }
        save_file(sd, str(src))
        logged: list[str] = []
        convert_to_safetensors(
            str(src), target_key="F16", overwrite=True,
            on_log=logged.append, log_tensor_every=3,
        )
        tensor_lines = [line for line in logged if "double_blocks." in line]
        # 5 tensors, every-3rd + first + last -> idx 0, 2, 4
        assert len(tensor_lines) == 3
        assert "double_blocks.0." in tensor_lines[0]
        assert "double_blocks.4." in tensor_lines[-1]


class TestAlreadyQuantizedGuard:
    """docs/issues_analysis.md #12: refuse to re-quantize a checkpoint that's
    already quantized (e.g. ComfyUI-native int8_tensorwise+ConvRot releases),
    rather than silently corrupting its pre-existing weight_scale/comfy_quant
    sidecar tensors by treating them as ordinary weights."""

    def test_raises_on_comfy_quant_sidecar_present(self, tmp_path):
        src = tmp_path / "model.safetensors"
        sd = {
            "double_blocks.0.img_attn.proj.weight": torch.randint(
                -128, 127, (64, 64), dtype=torch.int8
            ),
            "double_blocks.0.img_attn.proj.weight_scale": torch.randn(64, 1, dtype=torch.float32),
            "double_blocks.0.img_attn.proj.comfy_quant": torch.zeros(8, dtype=torch.uint8),
        }
        save_file(sd, str(src))
        with pytest.raises(ValueError, match="already quantized"):
            convert_to_safetensors(str(src), target_key="FP8", overwrite=True)

    def test_raises_on_int8_weight_without_sidecar(self, tmp_path):
        src = tmp_path / "model.safetensors"
        sd = {
            "double_blocks.0.img_attn.proj.weight": torch.randint(
                -128, 127, (64, 64), dtype=torch.int8
            ),
            "double_blocks.0.img_attn.proj.bias": torch.randn(64, dtype=torch.float32),
        }
        save_file(sd, str(src))
        with pytest.raises(ValueError, match="already quantized"):
            convert_to_safetensors(str(src), target_key="FP8", overwrite=True)

    def test_does_not_raise_on_plain_unquantized_checkpoint(self, tmp_path):
        src = _write_minimal_flux(tmp_path)
        # Should not raise
        convert_to_safetensors(str(src), target_key="FP8", overwrite=True)
