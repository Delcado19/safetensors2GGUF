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


class TestStreamingPrimitives:
    def test_iter_output_keys_skips_quant_sidecars_and_ignored_keys(self):
        import torch
        from convert_safetensors import _iter_output_keys
        from models.architectures import ModelTemplate

        state_dict = {
            "a.weight": torch.randn(4, 4),
            "a.weight_scale": torch.randn(1),  # in quant_skip_keys
            "spiece_model": torch.randint(0, 255, (10,), dtype=torch.uint8),
            "ignored.weight": torch.randn(4, 4),
        }
        arch = ModelTemplate()
        arch.keys_ignore = ["ignored"]
        result = list(_iter_output_keys(state_dict, arch, quant_skip_keys={"a.weight_scale"}))
        assert result == [("a.weight", False), ("spiece_model", True)]

    def test_tensor_bytes_roundtrips_bfloat16(self):
        import torch
        from convert_safetensors import _tensor_bytes
        t = torch.tensor([1.5, -2.25], dtype=torch.bfloat16)
        raw = _tensor_bytes(t)
        assert len(raw) == 4  # 2 elements * 2 bytes
        rebuilt = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16)
        assert torch.equal(rebuilt, t)

    def test_tensor_bytes_roundtrips_0dim_scalar(self):
        import torch
        from convert_safetensors import _tensor_bytes
        t = torch.tensor(3.5, dtype=torch.float32)
        raw = _tensor_bytes(t)
        assert len(raw) == 4
        rebuilt = torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(())
        assert torch.equal(rebuilt, t)

    def test_build_header_computes_sequential_offsets(self):
        from convert_safetensors import _build_header
        entries = [
            ("a.weight", "F16", (4, 4)),       # 4*4*2 = 32 bytes
            ("a.weight_scale", "F32", (1,)),   # 4 bytes
        ]
        header, total = _build_header(entries, {})
        assert header["a.weight"]["data_offsets"] == [0, 32]
        assert header["a.weight_scale"]["data_offsets"] == [32, 36]
        assert total == 36

    def test_build_header_includes_metadata(self):
        from convert_safetensors import _build_header
        header, _ = _build_header([("a.weight", "F16", (1,))], {"foo": "bar"})
        assert header["__metadata__"] == {"foo": "bar"}

    def test_write_header_roundtrips_via_safe_open(self, tmp_path):
        from convert_safetensors import _build_header, _write_header
        header, total = _build_header([("a.weight", "F16", (2,))], {"foo": "bar"})
        path = tmp_path / "hdr_test.safetensors"
        with open(path, "wb") as fh:
            _write_header(fh, header)
            fh.write(b"\x00" * total)  # dummy data section matching declared size
        from safetensors import safe_open
        with safe_open(str(path), framework="pt") as f:
            assert f.metadata() == {"foo": "bar"}
            assert "a.weight" in f.keys()


class TestConvertToSafetensors:
    def test_writes_output_file(self, tmp_path):
        src = _write_minimal_flux(tmp_path)
        dst, arch = convert_to_safetensors(str(src), target_key="F16", overwrite=True)
        assert dst.endswith(".safetensors")
        import os
        assert os.path.isfile(dst)
        assert arch is not None and arch.arch == "flux"

    def test_fp8_default_filename_uses_external_naming(self, tmp_path):
        # The auto-generated filename (dst_path=None) spells out FP8's
        # external "fp8_e4m3fn_scaled" naming (Civitai/Comfy-Org convention),
        # not the internal "FP8" target_key -- see safetensors_quant.py's
        # filename_suffix_for()/_FILENAME_SUFFIX.
        src = _write_minimal_flux(tmp_path)
        dst, _ = convert_to_safetensors(str(src), target_key="FP8", overwrite=True)
        assert dst.endswith("-fp8_e4m3fn_scaled.safetensors")

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

    def test_warns_when_output_is_larger_than_input(self, tmp_path, monkeypatch):
        """A tiny/oddly-shaped model can gain more from per-layer scale
        tensors than it loses from quantizing, so "quantized" doesn't always
        imply "smaller". User-reported: silently shipping a file that is
        both larger AND lower precision than the input defeats the point of
        quantizing at all — surface it instead of staying quiet."""
        import os

        src = _write_minimal_flux(tmp_path)
        real_getsize = os.path.getsize

        def fake_getsize(p):
            # Only the destination is forced "larger"; the source read must
            # still reflect its true size or the comparison is meaningless.
            return real_getsize(p) + 1_000_000 if str(p) != str(src) else real_getsize(p)

        monkeypatch.setattr("convert_safetensors.os.path.getsize", fake_getsize)
        logged: list[str] = []
        convert_to_safetensors(str(src), target_key="FP8", overwrite=True, on_log=logged.append)
        assert any("WARNING" in line and "larger" in line for line in logged)

    def test_no_warning_when_output_shrinks(self, tmp_path):
        src = _write_minimal_flux(tmp_path)
        logged: list[str] = []
        convert_to_safetensors(str(src), target_key="FP8", overwrite=True, on_log=logged.append)
        assert not any("larger" in line for line in logged)


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

    def test_dequantizes_comfy_org_scale_weight_checkpoint(self, tmp_path):
        # Regression, found 2026-08-18 converting HiDream's
        # llama_3.1_8b_instruct_fp8_scaled.safetensors: Comfy-Org's own
        # "fp8_scaled" repackaging convention uses ".scale_weight" (reversed
        # word order), which this tool didn't recognize at all. The actual
        # ".weight" tensor (raw float8_e4m3fn bytes) was fed straight into
        # the pipeline unscaled -- numerically wrong for every affected
        # layer, not just a missed optimization -- and the orphaned 0-dim
        # ".scale_weight" sidecar got treated as an ordinary weight,
        # crashing NVFP4's quantize_nvfp4() on `.shape[-1]` for a shape
        # with zero dimensions (IndexError: tuple index out of range).
        src = tmp_path / "model.safetensors"
        original = torch.randn(64, 64, dtype=torch.float32)
        amax = original.abs().max()
        scale = (amax / 448.0).clamp(min=1e-12)
        q = (original / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        sd = {
            "double_blocks.0.img_attn.proj.weight": q,
            "double_blocks.0.img_attn.proj.scale_weight": scale,  # 0-dim, Comfy-Org naming
        }
        save_file(sd, str(src))

        # Must not crash on the 0-dim sidecar, and must actually dequantize
        # (not just pass the guard) -- restored value close to `original`.
        dst, _ = convert_to_safetensors(str(src), target_key="F16", overwrite=True)
        out = load_file(dst)
        assert "double_blocks.0.img_attn.proj.scale_weight" not in out
        restored = out["double_blocks.0.img_attn.proj.weight"].to(torch.float32)
        assert torch.allclose(restored, original, atol=scale.item(), rtol=0.15)

        # The original NVFP4 crash reproduction.
        convert_to_safetensors(str(src), target_key="NVFP4", overwrite=True)

    def test_strips_scaled_fp8_sentinel_tensor(self, tmp_path):
        # Regression, found 2026-08-18 attempting to GGUF-convert a
        # re-quantized HiDream text encoder: Comfy-Org's fp8_scaled
        # checkpoints carry a top-level "scaled_fp8" sentinel (0 elements,
        # float8_e4m3fn dtype) that comfy/utils.py's convert_old_quants()
        # checks for presence, not content. It carries no real weight data
        # but rode straight through this tool's conversion pipeline into
        # every output format -- harmless for ComfyUI's safetensors loader
        # (which tolerates the unmapped key) but crashed llama.cpp's
        # convert_hf_to_gguf.py outright with "Can not map tensor
        # 'scaled_fp8'" when that output was fed into GGUF conversion.
        src = tmp_path / "model.safetensors"
        sd = {
            "double_blocks.0.img_attn.proj.weight": torch.randn(64, 64, dtype=torch.float32),
            "scaled_fp8": torch.zeros(0, dtype=torch.float32),
        }
        save_file(sd, str(src))

        dst, _ = convert_to_safetensors(str(src), target_key="F16", overwrite=True)
        out = load_file(dst)
        assert "scaled_fp8" not in out
        assert "double_blocks.0.img_attn.proj.weight" in out

    def test_preserves_spiece_model_sentinel_tensor_byte_identical(self, tmp_path):
        # Regression, found 2026-08-19: comfy/text_encoders/wan.py's
        # UMT5XXlTokenizer embeds the raw SentencePiece tokenizer.model
        # bytes as a top-level 1D uint8 "spiece_model" tensor so ComfyUI can
        # load the checkpoint standalone -- comfy/sd.py's
        # load_text_encoder_state_dicts() actively reads it back out to
        # build the tokenizer. Two separate mistakes fixed here in the same
        # session: (1) an earlier attempt stripped the key entirely
        # (treating it like the harmless scaled_fp8 sentinel above), which
        # broke every safetensors output of a self-contained-tokenizer
        # checkpoint in ComfyUI with "ValueError: invalid tokenizer" --
        # SPieceTokenizer has no other fallback; (2) the fix for that then
        # left the key present but still routed through the normal
        # nan_to_num/quantize_tensor_st pipeline, which silently upcast the
        # uint8 byte blob to float16 -- same on-disk key, but corrupted
        # bytes SentencePiece's protobuf parser can't read either. Must
        # survive completely untouched: same dtype, same bytes.
        src = tmp_path / "model.safetensors"
        blob = torch.randint(0, 255, (4548313,), dtype=torch.uint8)
        sd = {
            "encoder.block.0.layer.0.SelfAttention.q.weight": torch.randn(64, 64, dtype=torch.float32),
            "spiece_model": blob,
        }
        save_file(sd, str(src))

        from text_encoder_convert import _TEXT_ENCODER_MODEL_ARCH

        dst, _ = convert_to_safetensors(
            str(src), target_key="F16", overwrite=True, model_arch=_TEXT_ENCODER_MODEL_ARCH,
        )
        out = load_file(dst)
        assert "spiece_model" in out
        assert out["spiece_model"].dtype == torch.uint8
        assert torch.equal(out["spiece_model"], blob)
        assert "encoder.block.0.layer.0.SelfAttention.q.weight" in out

    def test_preserves_tekken_model_sentinel_tensor_byte_identical(self, tmp_path):
        # Regression, found 2026-08-23: comfy/text_encoders/flux.py's
        # load_mistral_tokenizer() (FLUX.2 dev's mistral_3_small_flux2 text
        # encoder) embeds the raw Tekken tokenizer JSON as a top-level 1D
        # uint8 "tekken_model" tensor and calls json.loads() on it directly
        # -- same sentinel pattern as "spiece_model" above, just not yet
        # covered by _PASSTHROUGH_TENSOR_SUFFIXES. Without this fix, a real
        # FP8_MIXED build upcast the JSON bytes to float32 like any other
        # weight, and ComfyUI failed loading it with
        # json.decoder.JSONDecodeError: "Expecting value: line 1 column 1
        # (char 0)" on the resulting garbage.
        # strip_prefixes=False, matching the real call path
        # (convert_text_encoder_to_safetensors() always passes it, precisely
        # because "model." is this family's genuine module path, not a
        # diffusion-UNet wrapper) -- without it, convert.py's prefix-strip
        # heuristic drops "tekken_model" entirely (it doesn't contain the
        # "model." substring, unlike every real weight key here), a
        # different bug this test isn't about.
        src = tmp_path / "model.safetensors"
        blob = torch.randint(0, 255, (19399895,), dtype=torch.uint8)
        sd = {
            "model.layers.0.self_attn.q_proj.weight": torch.randn(64, 64, dtype=torch.float32),
            "tekken_model": blob,
        }
        save_file(sd, str(src))

        from text_encoder_convert import _TEXT_ENCODER_MODEL_ARCH

        dst, _ = convert_to_safetensors(
            str(src), target_key="F16", overwrite=True, model_arch=_TEXT_ENCODER_MODEL_ARCH,
            strip_prefixes=False,
        )
        out = load_file(dst)
        assert "tekken_model" in out
        assert out["tekken_model"].dtype == torch.uint8
        assert torch.equal(out["tekken_model"], blob)
        assert "model.layers.0.self_attn.q_proj.weight" in out


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


class TestNvfp4FullPrecisionFlag:
    # comfy/ops.py's _load_quantized_module() reads "full_precision_matrix_mult"
    # from .comfy_quant generically (not FP8-specific) -- MixedPrecisionOps.
    # Linear.forward()'s `not self._full_precision_mm` gate applies to every
    # quant_format identically. Mirrors TestFp8FullPrecisionFlag above.
    def test_nvfp4_layer_config_defaults_full_precision_matrix_mult_true(self, tmp_path):
        import json
        from safetensors import safe_open

        src = _write_minimal_flux(tmp_path)
        dst, _ = convert_to_safetensors(str(src), target_key="NVFP4", overwrite=True)
        with safe_open(dst, framework="pt", device="cpu") as f:
            meta = json.loads(f.metadata()["_quantization_metadata"])
        layer_conf = next(iter(meta["layers"].values()))
        assert layer_conf["format"] == "nvfp4"
        assert layer_conf["full_precision_matrix_mult"] is True

    def test_nvfp4_full_precision_flag_can_be_disabled(self, tmp_path):
        import json
        from safetensors import safe_open

        src = _write_minimal_flux(tmp_path)
        dst, _ = convert_to_safetensors(
            str(src), target_key="NVFP4", overwrite=True, full_precision_nvfp4=False,
        )
        with safe_open(dst, framework="pt", device="cpu") as f:
            meta = json.loads(f.metadata()["_quantization_metadata"])
        layer_conf = next(iter(meta["layers"].values()))
        assert "full_precision_matrix_mult" not in layer_conf


class TestStreamingWriterCorrectness:
    def test_multi_tensor_multi_format_smoke(self, tmp_path):
        # Exercises header/offset math across a mix of tensor shapes in one
        # file -- 1D (bias), 2D non-block-aligned, 2D block-aligned
        # (triggers ConvRot/NVFP4's real packing math, not just fallbacks),
        # and >=3D (conv fallback) -- for every safetensors target format.
        import torch
        from safetensors.torch import save_file, load_file
        from convert_safetensors import convert_to_safetensors

        src = tmp_path / "model.safetensors"
        sd = {
            "double_blocks.0.img_attn.proj.weight": torch.randn(64, 512, dtype=torch.float32),
            "double_blocks.0.img_attn.proj.bias": torch.randn(64, dtype=torch.float32),
            "double_blocks.0.img_attn.qkv.weight": torch.randn(96, 100, dtype=torch.float32),
            "conv_stem.weight": torch.randn(32, 4, 3, 3, dtype=torch.float32),
        }
        save_file(sd, str(src))

        for target_key in ("F16", "FP8", "FP8_MIXED", "INT8", "INT8_MIXED", "NVFP4", "NVFP4_MIXED"):
            dst, _ = convert_to_safetensors(
                str(src), target_key=target_key, overwrite=True
            )
            out = load_file(dst)
            assert "double_blocks.0.img_attn.proj.weight" in out
            assert "double_blocks.0.img_attn.proj.bias" in out
            assert "double_blocks.0.img_attn.qkv.weight" in out
            assert "conv_stem.weight" in out
            # Conv weight is always F16 regardless of target format (>=3D
            # guard, quantize_tensor_st/plan_tensor_output's shared rule).
            assert out["conv_stem.weight"].dtype == torch.float16

    def test_plan_pass2_mismatch_raises(self, tmp_path, monkeypatch):
        # Forces plan_tensor_output to disagree with the real quantize_
        # tensor_st for one call, and confirms the streaming writer's
        # safety assert catches it instead of silently writing misaligned
        # bytes.
        import torch
        from safetensors.torch import save_file
        import convert_safetensors as cs

        src = tmp_path / "model.safetensors"
        save_file(
            {"double_blocks.0.img_attn.proj.weight": torch.randn(64, 64, dtype=torch.float32)},
            str(src),
        )

        real_plan = cs.plan_tensor_output

        def _wrong_plan(*args, **kwargs):
            entries, conf = real_plan(*args, **kwargs)
            # Corrupt the first entry's declared shape so it disagrees with
            # what quantize_tensor_st will actually produce.
            name, dtype, shape = entries[0]
            entries[0] = (name, dtype, tuple(d + 1 for d in shape))
            return entries, conf

        monkeypatch.setattr(cs, "plan_tensor_output", _wrong_plan)
        with pytest.raises(AssertionError, match="plan mismatch"):
            cs.convert_to_safetensors(str(src), target_key="F16", overwrite=True)
