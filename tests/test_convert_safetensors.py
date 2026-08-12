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

    def test_no_on_log_callback_survives_non_utf8_console(self, tmp_path, monkeypatch):
        # Regression: the print() fallback used when on_log is None must not
        # crash on a Windows console stuck on a legacy codepage (cp1252) that
        # can't encode the non-ASCII characters in the log text (e.g. "->" as
        # U+2192) — this previously aborted the conversion mid-run, before
        # save_file() ever wrote the output.
        import sys

        class Cp1252Stdout:
            encoding = "cp1252"

            def write(self, s):
                s.encode("cp1252")  # raises UnicodeEncodeError on non-Latin-1 text

            def flush(self):
                pass

        monkeypatch.setattr(sys, "stdout", Cp1252Stdout())
        src = _write_minimal_flux(tmp_path)
        dst, _ = convert_to_safetensors(str(src), target_key="FP8", overwrite=True)
        import os
        assert os.path.isfile(dst)

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
    """docs/issues_analysis.md #12/dequantize.py: an already-quantized source
    checkpoint (e.g. ComfyUI-native int8_tensorwise+ConvRot releases) is
    automatically dequantized and cleanly re-quantized, instead of refusing
    outright — except when there's no recognizable scale to reconstruct the
    original magnitude from, which stays a hard refusal (genuinely
    unrecoverable, not just unhandled)."""

    def test_dequantizes_plain_int8_checkpoint_instead_of_refusing(self, tmp_path):
        src = tmp_path / "model.safetensors"
        original = torch.randn(64, 64, dtype=torch.float32)
        amax = original.abs().max()
        scale = amax / 127.0
        q = (original / scale).round().clamp(-128, 127).to(torch.int8)
        sd = {
            "double_blocks.0.img_attn.proj.weight": q,
            "double_blocks.0.img_attn.proj.weight_scale": scale.reshape(1),
            # Garbage sidecar (not valid JSON) — must fall back to the
            # dtype/scale heuristic instead of raising or crashing.
            "double_blocks.0.img_attn.proj.comfy_quant": torch.zeros(8, dtype=torch.uint8),
        }
        save_file(sd, str(src))
        # Should not raise; should reconstruct values close to `original`.
        dst, _ = convert_to_safetensors(str(src), target_key="F16", overwrite=True)
        out = load_file(dst)
        restored = out["double_blocks.0.img_attn.proj.weight"].to(torch.float32)
        assert torch.allclose(restored, original, atol=scale.item() * 1.5)

    def test_raises_on_int8_weight_without_recognizable_scale(self, tmp_path):
        src = tmp_path / "model.safetensors"
        sd = {
            "double_blocks.0.img_attn.proj.weight": torch.randint(
                -128, 127, (64, 64), dtype=torch.int8
            ),
            "double_blocks.0.img_attn.proj.bias": torch.randn(64, dtype=torch.float32),
        }
        save_file(sd, str(src))
        with pytest.raises(ValueError, match="no recognizable"):
            convert_to_safetensors(str(src), target_key="FP8", overwrite=True)

    def test_round_trips_this_tools_own_int8_mixed_convrot_output(self, tmp_path):
        """The file-level _quantization_metadata this tool's own
        convert_to_safetensors() writes (not a per-layer .comfy_quant sidecar
        tensor, unlike real ComfyUI-native releases) must still be read back
        so re-feeding this tool's own ConvRot output uses the actual
        convrot_groupsize instead of silently assuming CONVROT_GROUP_SIZE."""
        src = tmp_path / "model.safetensors"
        original = torch.randn(64, 256, dtype=torch.float32)  # 256: triggers ConvRot
        save_file(
            {
                "double_blocks.0.img_attn.proj.weight": original,
                "double_blocks.0.img_attn.proj.bias": torch.randn(64, dtype=torch.float32),
            },
            str(src),
        )
        int8_path, _ = convert_to_safetensors(str(src), target_key="INT8_MIXED", overwrite=True)
        # weight_scale must be per-row (ConvRot actually took effect) or this
        # test isn't exercising the code path it claims to.
        assert load_file(int8_path)["double_blocks.0.img_attn.proj.weight_scale"].numel() > 1

        roundtrip_path, _ = convert_to_safetensors(int8_path, target_key="F16", overwrite=True)
        restored = load_file(roundtrip_path)["double_blocks.0.img_attn.proj.weight"].to(torch.float32)
        assert torch.allclose(restored, original, atol=0.05)

    def test_does_not_raise_on_plain_unquantized_checkpoint(self, tmp_path):
        src = _write_minimal_flux(tmp_path)
        # Should not raise
        convert_to_safetensors(str(src), target_key="FP8", overwrite=True)


class TestFp8FullPrecisionFlag:
    def test_fp8_layer_config_defaults_full_precision_matrix_mult_true(self, tmp_path):
        import json
        from safetensors import safe_open

        src = _write_minimal_flux(tmp_path)
        dst, _ = convert_to_safetensors(str(src), target_key="FP8", overwrite=True)
        with safe_open(dst, framework="pt", device="cpu") as f:
            meta = json.loads(f.metadata()["_quantization_metadata"])
        layer_conf = next(iter(meta["layers"].values()))
        assert layer_conf["format"] == "float8_e4m3fn"
        assert layer_conf["full_precision_matrix_mult"] is True

    def test_fp8_full_precision_flag_can_be_disabled(self, tmp_path):
        import json
        from safetensors import safe_open

        src = _write_minimal_flux(tmp_path)
        dst, _ = convert_to_safetensors(
            str(src), target_key="FP8", overwrite=True, full_precision_fp8=False,
        )
        with safe_open(dst, framework="pt", device="cpu") as f:
            meta = json.loads(f.metadata()["_quantization_metadata"])
        layer_conf = next(iter(meta["layers"].values()))
        assert "full_precision_matrix_mult" not in layer_conf

    def test_full_precision_flag_absent_for_non_fp8_formats(self, tmp_path):
        import json
        from safetensors import safe_open

        src = _write_minimal_flux(tmp_path)
        dst, _ = convert_to_safetensors(str(src), target_key="INT8", overwrite=True)
        with safe_open(dst, framework="pt", device="cpu") as f:
            meta = json.loads(f.metadata()["_quantization_metadata"])
        layer_conf = next(iter(meta["layers"].values()))
        assert "full_precision_matrix_mult" not in layer_conf
