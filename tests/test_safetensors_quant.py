"""Tests for safetensors_quant.py — dtype registry and per-tensor quantization."""

from __future__ import annotations

import torch

from safetensors_quant import (
    SAFETENSORS_DTYPE_CHOICES,
    format_recommendation,
    is_hiprec_st,
    quantize_tensor_st,
)
from models.architectures import (
    ModelAura,
    ModelFlux,
    ModelHyVid,
    ModelLTXV,
    ModelLumina2,
    ModelQwenImage,
    ModelSD3,
    ModelSDXL,
    ModelWan,
    CosmosPredict2,
)


class TestRegistry:
    def test_choices_is_list_of_tuples(self):
        assert isinstance(SAFETENSORS_DTYPE_CHOICES, list)
        for label, key in SAFETENSORS_DTYPE_CHOICES:
            assert isinstance(label, str) and label
            assert isinstance(key, str) and key

    def test_expected_keys_present(self):
        # FP8 is now re-offered (re-enabled with full_precision_matrix_mult=true
        # default in convert_safetensors.py). NVFP4 is still not offered in the
        # GUI dropdown (docs/issues_analysis.md #15) — quantize_tensor_st still
        # supports it internally (TestQuantizeTensorNvfp4 exercises that dispatch
        # directly), it's just not user-selectable.
        keys = {k for _, k in SAFETENSORS_DTYPE_CHOICES}
        assert keys == {"F16", "F16_MIXED", "FP8", "FP8_MIXED", "INT8", "INT8_MIXED"}

    def test_fp8_is_offered_again(self):
        keys = {key for _, key in SAFETENSORS_DTYPE_CHOICES}
        assert "FP8" in keys
        assert "FP8_MIXED" in keys

    def test_no_duplicate_keys(self):
        keys = [k for _, k in SAFETENSORS_DTYPE_CHOICES]
        assert len(keys) == len(set(keys))


class TestHiprec:
    def test_1d_tensor_is_hiprec(self):
        data = torch.zeros(64, dtype=torch.float32)
        assert is_hiprec_st("some.weight", data, ModelFlux(), torch.float32)

    def test_small_2d_tensor_is_hiprec(self):
        data = torch.zeros(4, 4, dtype=torch.float32)
        assert is_hiprec_st("some.weight", data, ModelFlux(), torch.float32)

    def test_large_2d_tensor_is_not_hiprec(self):
        data = torch.zeros(64, 64, dtype=torch.float32)
        assert not is_hiprec_st("some.weight", data, ModelFlux(), torch.float32)

    def test_keys_hiprec_key_is_hiprec(self):
        data = torch.zeros(64, 64, dtype=torch.bfloat16)
        assert is_hiprec_st("x_pad_token", data, ModelLumina2(), torch.bfloat16)


class TestQuantizeTensorF16:
    def test_f16_plain_casts_dtype(self):
        data = torch.randn(64, 64, dtype=torch.float32)
        out = quantize_tensor_st(data, "block.weight", ModelFlux(), "F16")
        assert set(out.keys()) == {"block.weight"}
        assert out["block.weight"].dtype == torch.float16

    def test_f16_mixed_keeps_hiprec_tensor_f32(self):
        data = torch.randn(4, 4, dtype=torch.float32)
        out = quantize_tensor_st(data, "block.bias", ModelFlux(), "F16_MIXED")
        assert out["block.bias"].dtype == torch.float32

    def test_mixed_hiprec_tensor_keeps_original_dtype_not_forced_f32(self):
        # Regression: hiprec protection must preserve the SOURCE dtype, not
        # force-upcast to F32. A bfloat16 source doubled in size for zero
        # precision benefit (ComfyUI casts every loaded weight to its own
        # compute_dtype regardless of on-disk dtype) — invisible while
        # keys_hiprec covered only a few small tensors per model, but once it
        # grew to cover a large fraction of a model this made *_MIXED output
        # larger than the unquantized bf16 source (docs/issues_analysis.md #15).
        data = torch.randn(4, 4, dtype=torch.bfloat16)
        out = quantize_tensor_st(data, "block.bias", ModelFlux(), "F16_MIXED")
        assert out["block.bias"].dtype == torch.bfloat16

    def test_f16_mixed_casts_large_tensor_f16(self):
        data = torch.randn(64, 64, dtype=torch.float32)
        out = quantize_tensor_st(data, "block.weight", ModelFlux(), "F16_MIXED")
        assert out["block.weight"].dtype == torch.float16


class TestQuantizeTensorFp8:
    def test_fp8_returns_weight_and_scale(self):
        data = torch.randn(64, 64, dtype=torch.float32)
        out = quantize_tensor_st(data, "block.weight", ModelFlux(), "FP8")
        assert "block.weight" in out
        assert "block.weight_scale" in out
        assert out["block.weight"].dtype == torch.float8_e4m3fn

    def test_fp8_mixed_keeps_hiprec_tensor_f32_unscaled(self):
        data = torch.randn(4, 4, dtype=torch.float32)
        out = quantize_tensor_st(data, "block.bias", ModelFlux(), "FP8_MIXED")
        assert set(out.keys()) == {"block.bias"}
        assert out["block.bias"].dtype == torch.float32


class TestQuantizeTensorNvfp4:
    def test_nvfp4_returns_packed_and_two_scales(self):
        data = torch.randn(32, 32, dtype=torch.float32)
        out = quantize_tensor_st(data, "block.weight", ModelFlux(), "NVFP4")
        assert set(out.keys()) == {
            "block.weight", "block.weight_scale", "block.weight_scale_2",
        }

    def test_nvfp4_mixed_keeps_hiprec_tensor_unpacked(self):
        data = torch.randn(4, 4, dtype=torch.float32)
        out = quantize_tensor_st(data, "block.bias", ModelFlux(), "NVFP4_MIXED")
        assert set(out.keys()) == {"block.bias"}
        assert out["block.bias"].dtype == torch.float32

    def test_nvfp4_non_multiple_of_16_last_dim_falls_back_to_f16(self):
        # Regression for review finding #1: a real conv weight (e.g. 3x3 conv,
        # last dim 3) must not crash the whole conversion — quantize_nvfp4
        # raises ValueError internally, and the dispatcher must catch it and
        # fall back to a plain F16 write for that one tensor.
        data = torch.randn(8, 3, dtype=torch.float32)
        out = quantize_tensor_st(data, "conv.weight", ModelFlux(), "NVFP4")
        assert set(out.keys()) == {"conv.weight"}
        assert out["conv.weight"].dtype == torch.float16
        assert out["conv.weight"].shape == data.shape

    def test_nvfp4_shape_critical_tensor_falls_back_to_f16_unconditionally(self):
        # Regression: ComfyUI's model_detection.py infers Lumina2's cap_feat_dim
        # from cap_embedder.1.weight.shape[1] BEFORE dequantizing. NVFP4 halves
        # the on-disk last dim, corrupting that inference and crashing model
        # load (docs/issues_analysis.md #9). Must be skipped even in non-mixed
        # base NVFP4 mode — this is a shape-safety constraint, not a precision
        # one gated by *_MIXED.
        data = torch.randn(32, 32, dtype=torch.float32)
        out = quantize_tensor_st(data, "cap_embedder.1.weight", ModelLumina2(), "NVFP4")
        assert set(out.keys()) == {"cap_embedder.1.weight"}
        assert out["cap_embedder.1.weight"].dtype == torch.float16
        assert out["cap_embedder.1.weight"].shape == data.shape

    def test_nvfp4_pad_token_falls_back_to_f16_unconditionally(self):
        # Regression: x_pad_token/cap_pad_token are nn.Parameters NextDiT's
        # __init__ hardcodes to a fixed [1, dim] shape from the detected
        # architecture. keys_hiprec only protects them in *_MIXED mode; in
        # plain non-mixed NVFP4 they were packed like any other 2D tensor,
        # halving the last dim and making load_state_dict's strict shape
        # check fail outright (docs/issues_analysis.md #14).
        data = torch.randn(1, 3840, dtype=torch.float32)
        out = quantize_tensor_st(data, "x_pad_token", ModelLumina2(), "NVFP4")
        assert set(out.keys()) == {"x_pad_token"}
        assert out["x_pad_token"].dtype == torch.float16
        assert out["x_pad_token"].shape == data.shape

    def test_nvfp4_non_shape_critical_tensor_still_packs(self):
        # Sanity: keys_shape_critical must not blanket-disable NVFP4 for an
        # architecture — only the specific flagged tensor(s).
        data = torch.randn(32, 32, dtype=torch.float32)
        out = quantize_tensor_st(data, "layers.0.attention.qkv.weight", ModelLumina2(), "NVFP4")
        assert out["layers.0.attention.qkv.weight"].dtype == torch.uint8

    def test_nvfp4_shape_critical_protection_per_architecture(self):
        # Verifies keys_shape_critical for every architecture audited against a
        # live ComfyUI model_detection.py (see docs/issues_analysis.md #9): each
        # listed tensor must fall back to F16 under non-mixed NVFP4, since
        # ComfyUI reads its raw on-disk shape to infer hyperparameters before
        # any dequantization happens.
        cases = [
            (ModelFlux(), "img_in.weight"),
            (ModelSD3(), "context_embedder.weight"),
            (ModelAura(), "cond_seq_linear.weight"),
            (CosmosPredict2(), "x_embedder.proj.1.weight"),
            (ModelQwenImage(), "img_in.weight"),
            (ModelHyVid(), "txt_in.input_embedder.weight"),
            (ModelWan(), "head.modulation"),
            (ModelLTXV(), "transformer_blocks.0.attn2.to_k.weight"),
        ]
        for arch, key in cases:
            data = torch.randn(32, 32, dtype=torch.float32)
            out = quantize_tensor_st(data, key, arch, "NVFP4")
            assert set(out.keys()) == {key}, f"{arch.arch}/{key}"
            assert out[key].dtype == torch.float16, f"{arch.arch}/{key}"
            assert out[key].shape == data.shape, f"{arch.arch}/{key}"


class TestQuantizeTensor1dNeverScaled:
    """Regression for review finding #2: 1D tensors (biases, norm weights)
    must never get a .weight_scale sibling under FP8/NVFP4, even in
    non-mixed mode — no consumer reads a bias-scale, so a scaled 1D tensor
    loads back unscaled and wrong by up to ~448x."""

    def test_fp8_non_mixed_1d_tensor_stays_plain_f32(self):
        data = torch.randn(64, dtype=torch.float32)
        out = quantize_tensor_st(data, "block.bias", ModelFlux(), "FP8")
        assert set(out.keys()) == {"block.bias"}
        assert out["block.bias"].dtype == torch.float32

    def test_nvfp4_non_mixed_1d_tensor_stays_plain_f32(self):
        data = torch.randn(64, dtype=torch.float32)
        out = quantize_tensor_st(data, "block.bias", ModelFlux(), "NVFP4")
        assert set(out.keys()) == {"block.bias"}
        assert out["block.bias"].dtype == torch.float32

    def test_fp8_non_mixed_1d_tensor_protected_even_with_f16_source(self):
        # is_hiprec_st gates its threshold/keys_hiprec logic on
        # old_dtype in (float32, bfloat16) — for a float16-sourced checkpoint
        # it always returns False. Confirm the new unconditional 1D check
        # does not rely on that gate (it must protect 1D tensors regardless
        # of source dtype).
        data = torch.randn(64, dtype=torch.float16)
        out = quantize_tensor_st(data, "block.bias", ModelFlux(), "FP8")
        assert set(out.keys()) == {"block.bias"}
        assert "block.bias.weight_scale" not in out


class TestQuantizeTensorInt8:
    def test_int8_convrot_applied_when_divisible(self):
        # 512 % 256 == 0 -> ConvRot path, per-row (out_features, 1) scale.
        data = torch.randn(8, 512, dtype=torch.float32)
        out = quantize_tensor_st(data, "block.weight", ModelFlux(), "INT8")
        assert set(out.keys()) == {"block.weight", "block.weight_scale"}
        assert out["block.weight"].dtype == torch.int8
        assert out["block.weight"].shape == (8, 512)
        assert out["block.weight_scale"].shape == (8, 1)

    def test_int8_falls_back_to_plain_tensorwise_when_not_divisible(self):
        # 100 is not a multiple of CONVROT_GROUP_SIZE (256) -> plain path,
        # scalar scale.
        data = torch.randn(8, 100, dtype=torch.float32)
        out = quantize_tensor_st(data, "block.weight", ModelFlux(), "INT8")
        assert out["block.weight"].dtype == torch.int8
        assert out["block.weight_scale"].numel() == 1

    def test_int8_non_2d_tensor_uses_plain_tensorwise(self):
        # ConvRot's block-Hadamard only applies to 2D [out, in] weights.
        data = torch.randn(4, 4, 4, dtype=torch.float32)
        out = quantize_tensor_st(data, "conv.weight", ModelFlux(), "INT8")
        assert out["conv.weight"].dtype == torch.int8
        assert out["conv.weight_scale"].numel() == 1

    def test_int8_mixed_keeps_hiprec_tensor_f32_unscaled(self):
        data = torch.randn(4, 4, dtype=torch.float32)
        out = quantize_tensor_st(data, "block.bias", ModelFlux(), "INT8_MIXED")
        assert set(out.keys()) == {"block.bias"}
        assert out["block.bias"].dtype == torch.float32

    def test_int8_1d_tensor_stays_plain_f32(self):
        data = torch.randn(64, dtype=torch.float32)
        out = quantize_tensor_st(data, "block.bias", ModelFlux(), "INT8")
        assert set(out.keys()) == {"block.bias"}
        assert out["block.bias"].dtype == torch.float32

    def test_int8_shape_critical_tensor_falls_back_to_f16_unconditionally(self):
        data = torch.randn(32, 32, dtype=torch.float32)
        out = quantize_tensor_st(data, "cap_embedder.1.weight", ModelLumina2(), "INT8")
        assert out["cap_embedder.1.weight"].dtype == torch.float16
        assert out["cap_embedder.1.weight"].shape == data.shape

    def test_int8_convrot_round_trips_close(self):
        # End-to-end sanity: dequantize (unrotate + rescale) recovers the
        # original weight within INT8-rotation-induced error.
        from safetensors_quant_int8 import CONVROT_GROUP_SIZE, _build_hadamard, _rotate_weight

        torch.manual_seed(0)
        data = torch.randn(16, 512, dtype=torch.float32)
        out = quantize_tensor_st(data, "block.weight", ModelFlux(), "INT8")
        q, scale = out["block.weight"], out["block.weight_scale"]

        h = _build_hadamard(CONVROT_GROUP_SIZE, device=q.device, dtype=torch.float32)
        recon = _rotate_weight(q.float() * scale, h, CONVROT_GROUP_SIZE)
        assert torch.allclose(recon, data, atol=0.05)


class TestFormatRecommendation:
    def test_f16_always_ok(self):
        level, msg = format_recommendation(ModelLumina2(), "F16")
        assert level == "ok"
        assert msg

    def test_plain_int8_warns_on_sensitive_architecture(self):
        level, msg = format_recommendation(ModelLumina2(), "INT8")
        assert level == "warn"
        assert "lumina2" in msg

    def test_int8_mixed_ok_on_sensitive_architecture(self):
        level, msg = format_recommendation(ModelLumina2(), "INT8_MIXED")
        assert level == "ok"
        assert "lumina2" in msg

    def test_render_verified_architecture_has_no_caveat(self):
        _, msg = format_recommendation(ModelLumina2(), "INT8_MIXED")
        assert "not yet confirmed" not in msg

    def test_unverified_sensitive_architecture_discloses_caveat(self):
        _, msg = format_recommendation(ModelFlux(), "INT8_MIXED")
        assert "not yet confirmed" in msg

    def test_plain_int8_ok_on_architecture_without_hiprec(self):
        assert ModelSDXL().keys_hiprec == []
        level, msg = format_recommendation(ModelSDXL(), "INT8")
        assert level == "ok"
        assert "sdxl" in msg

    def test_plain_fp8_warns_on_sensitive_architecture(self):
        # full_precision_matrix_mult only fixes ComfyUI's compute path, not
        # precision already lost when a keys_hiprec tensor is written as FP8
        # on disk -- plain FP8 must warn like plain INT8 does.
        level, msg = format_recommendation(ModelLumina2(), "FP8")
        assert level == "warn"
        assert "lumina2" in msg

    def test_fp8_mixed_ok_on_sensitive_architecture(self):
        level, msg = format_recommendation(ModelFlux(), "FP8_MIXED")
        assert level == "ok"
        assert msg

    def test_fp8_ok_on_architecture_without_hiprec(self):
        assert ModelSDXL().keys_hiprec == []
        level, msg = format_recommendation(ModelSDXL(), "FP8")
        assert level == "ok"
        assert "full_precision_matrix_mult" in msg
        level, msg = format_recommendation(ModelSDXL(), "FP8_MIXED")
        assert level == "ok"
        assert msg

    def test_fp8_mixed_on_lumina2_discloses_caveat_despite_int8_being_verified(self):
        # lumina2 is in _RENDER_VERIFIED_ARCHES for INT8_MIXED, but no FP8
        # output from this tool has ever been render-tested on any
        # architecture (model_support.support_level() keeps FP8/FP8_MIXED at
        # SUPPORT_CAUTION everywhere) -- FP8_MIXED must still disclose the
        # caveat here, not silently inherit INT8's verified status.
        _, msg = format_recommendation(ModelLumina2(), "FP8_MIXED")
        assert "not yet confirmed" in msg

    def test_plain_fp8_warn_does_not_overclaim_fp8_specific_testing(self):
        # The stronger "has shown visible ... corruption in testing" claim is
        # backed by an actual plain-INT8 render test on lumina2, not FP8 --
        # the FP8 warning must not imply FP8 itself was tested that way.
        _, msg = format_recommendation(ModelLumina2(), "FP8")
        assert "plain FP8 has shown visible pose/identity corruption in testing" not in msg

    def test_render_verified_architecture_warn_claims_shown_in_testing(self):
        # lumina2 was actually rendered end-to-end -- the strong claim is
        # accurate here and should say so plainly.
        _, msg = format_recommendation(ModelLumina2(), "INT8")
        assert "has shown visible pose/identity corruption in testing" in msg

    def test_unverified_architecture_warn_does_not_overclaim(self):
        # flux's keys_hiprec was never render-tested by this project (only
        # cross-referenced against a community blacklist) -- the warning
        # must not assert corruption "shown in testing" for an architecture
        # nobody has actually rendered with this tool's output.
        _, msg = format_recommendation(ModelFlux(), "INT8")
        assert "has shown visible pose/identity corruption in testing" not in msg
        assert "lumina2" in msg  # still cites the actual evidence source
        assert "hasn't been render-tested on this specific architecture" in msg
