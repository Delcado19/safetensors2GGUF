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
    ModelSD1,
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

    def test_fp8_shape_critical_key_stays_f16(self):
        # Real bug found 2026-08-14 render-testing CLIP-L/bigG text encoders:
        # unlike NVFP4/INT8 below, the FP8 branch never checked
        # keys_shape_critical at all. FP8 doesn't corrupt on-disk shape (so
        # this was invisible for shape-detection-only cases), but some keys
        # are read via a bare `.weight` attribute access that bypasses
        # dequant entirely (comfy/clip_model.py's position_embedding) --
        # those need value protection under FP8 too, not just NVFP4/INT8.
        data = torch.randn(64, 64, dtype=torch.float32)
        out = quantize_tensor_st(data, "img_in.weight", ModelFlux(), "FP8")
        assert set(out.keys()) == {"img_in.weight"}
        assert out["img_in.weight"].dtype == torch.float16

    def test_fp8_conv_weight_stays_f16(self):
        # Real bug found 2026-08-14 render-testing SDXL: FP8 had no dim check
        # at all, so a Conv2d kernel like input_blocks.0.0.weight
        # ([320,4,3,3]) quantized to F8_E4M3 without incident here -- but
        # ComfyUI's MixedPrecisionOps has no quantized Conv2d loader (only
        # Linear/MoEExperts/Embedding), so it silently misinterpreted the raw
        # FP8 bytes as floats and the render collapsed to a black image.
        data = torch.randn(4, 4, 3, 3, dtype=torch.float32)
        out = quantize_tensor_st(data, "conv.weight", ModelFlux(), "FP8")
        assert set(out.keys()) == {"conv.weight"}
        assert out["conv.weight"].dtype == torch.float16


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
            # txt_in.weight/vector_in.in_layer.weight: added 2026-08-13 after a
            # live ComfyUI crash converting FLUX.2 Klein 9B to plain NVFP4 --
            # model_detection.py reads txt_in.weight.shape[1] for context_in_dim,
            # halved on-disk by NVFP4 packing, RuntimeError in comfy_kitchen's
            # dequantize path (mat1/mat2 shape mismatch). vector_in shares the
            # same raw-shape-read pattern for vec_in_dim.
            (ModelFlux(), "txt_in.weight"),
            (ModelFlux(), "vector_in.in_layer.weight"),
            (ModelSD3(), "context_embedder.weight"),
            # y_embedder.mlp.0.weight: added 2026-08-16 after a live ComfyUI
            # crash rendering SD3-medium NVFP4 ("mat1 and mat2 shapes cannot
            # be multiplied") -- model_detection.py reads its raw .shape[1]
            # for adm_in_channels, halved on-disk by NVFP4 packing.
            (ModelSD3(), "y_embedder.mlp.0.weight"),
            (ModelAura(), "cond_seq_linear.weight"),
            (CosmosPredict2(), "x_embedder.proj.1.weight"),
            (ModelQwenImage(), "img_in.weight"),
            (ModelHyVid(), "txt_in.input_embedder.weight"),
            (ModelWan(), "head.modulation"),
            (ModelLTXV(), "transformer_blocks.0.attn2.to_k.weight"),
            # ModelSDXL/ModelSD1: added 2026-08-13 closing a documented audit
            # gap (docs/issues_analysis.md #9) -- ComfyUI's model_detection.py
            # dynamically scans for the first attn2.to_k.weight it finds rather
            # than a fixed key name, so the substring must match every block's
            # tensor, not just one literal key.
            (ModelSDXL(), "input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight"),
            (ModelSD1(), "input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight"),
            # label_emb.0.0.weight: added 2026-08-14 after a live ComfyUI crash
            # loading a real NVFP4 SDXL diffusion model ("'NoneType' object has
            # no attribute 'quant_config'") -- model_detection.py reads its raw
            # .shape[1] for adm_in_channels, halved on-disk by NVFP4 packing.
            (ModelSDXL(), "label_emb.0.0.weight"),
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

    def test_int8_non_2d_but_not_conv_tensor_uses_plain_tensorwise(self):
        # ConvRot's block-Hadamard only applies to 2D [out, in] weights; a
        # non-2D, non-conv (2D-ish but odd-shaped) tensor still isn't caught
        # by the >=3D conv guard below, so it falls through to tensorwise.
        data = torch.randn(8, 100, dtype=torch.float32)
        out = quantize_tensor_st(data, "block.weight", ModelFlux(), "INT8")
        assert out["block.weight"].dtype == torch.int8

    def test_int8_conv_weight_stays_f16(self):
        # ComfyUI's MixedPrecisionOps has no quantized Conv2d loader (see
        # quantize_tensor_st's ndim >= 3 guard) -- a real Conv2d kernel
        # ([out, in, kh, kw]) must never get a weight_scale sidecar.
        data = torch.randn(4, 4, 3, 3, dtype=torch.float32)
        out = quantize_tensor_st(data, "conv.weight", ModelFlux(), "INT8")
        assert set(out.keys()) == {"conv.weight"}
        assert out["conv.weight"].dtype == torch.float16

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
        # flux itself is now render-verified (see safetensors_quant.py's
        # _RENDER_VERIFIED_MIXED docstring) -- qwen_image is still untested.
        _, msg = format_recommendation(ModelQwenImage(), "INT8_MIXED")
        assert "not yet confirmed" in msg


class TestEstimateSafetensorsOutputSize:
    """Cross-checks estimate_safetensors_output_size() against the REAL
    per-tensor output of quantize_tensor_st() -- the estimator duplicates
    quantize_tensor_st's branching logic from a header only (no tensor data),
    so this is the "one runnable check" that catches the two implementations
    drifting apart."""

    @staticmethod
    def _actual_bytes(state_dict: dict, model_arch, target_key: str) -> int:
        total = 0
        for key, data in state_dict.items():
            for tensor in quantize_tensor_st(data, key, model_arch, target_key).values():
                total += tensor.numel() * tensor.element_size()
        return total

    @staticmethod
    def _build_and_estimate(tmp_path, state_dict: dict, model_arch, target_key: str) -> int:
        from safetensors.torch import save_file
        from safetensors_quant import estimate_safetensors_output_size

        path = tmp_path / "model.safetensors"
        save_file(state_dict, str(path))
        return estimate_safetensors_output_size(str(path), target_key, model_arch)

    def _state_dict(self):
        torch.manual_seed(0)
        return {
            "bias.weight": torch.randn(64, dtype=torch.float32),               # 1D
            "small.weight": torch.randn(8, 8, dtype=torch.float32),            # small 2D (<=1024 elems)
            "attn.q_proj.weight": torch.randn(64, 256, dtype=torch.bfloat16),  # keys_hiprec match (ModelLumina2)
            "large.weight": torch.randn(64, 256, dtype=torch.float32),         # large, block/group-aligned
            "odd.weight": torch.randn(64, 130, dtype=torch.float32),           # large, NOT 16/256-aligned
        }

    def test_f16_matches_exactly(self, tmp_path):
        sd = self._state_dict()
        actual = self._actual_bytes(sd, ModelLumina2(), "F16")
        estimated = self._build_and_estimate(tmp_path, sd, ModelLumina2(), "F16")
        assert estimated == actual

    def test_f16_mixed_matches_exactly(self, tmp_path):
        sd = self._state_dict()
        actual = self._actual_bytes(sd, ModelLumina2(), "F16_MIXED")
        estimated = self._build_and_estimate(tmp_path, sd, ModelLumina2(), "F16_MIXED")
        assert estimated == actual

    def test_fp8_matches_exactly(self, tmp_path):
        sd = self._state_dict()
        actual = self._actual_bytes(sd, ModelLumina2(), "FP8")
        estimated = self._build_and_estimate(tmp_path, sd, ModelLumina2(), "FP8")
        assert estimated == actual

    def test_fp8_mixed_matches_exactly(self, tmp_path):
        sd = self._state_dict()
        actual = self._actual_bytes(sd, ModelLumina2(), "FP8_MIXED")
        estimated = self._build_and_estimate(tmp_path, sd, ModelLumina2(), "FP8_MIXED")
        assert estimated == actual

    def test_int8_matches_exactly(self, tmp_path):
        sd = self._state_dict()
        actual = self._actual_bytes(sd, ModelLumina2(), "INT8")
        estimated = self._build_and_estimate(tmp_path, sd, ModelLumina2(), "INT8")
        assert estimated == actual

    def test_int8_mixed_matches_exactly(self, tmp_path):
        sd = self._state_dict()
        actual = self._actual_bytes(sd, ModelLumina2(), "INT8_MIXED")
        estimated = self._build_and_estimate(tmp_path, sd, ModelLumina2(), "INT8_MIXED")
        assert estimated == actual

    def test_nvfp4_matches_exactly(self, tmp_path):
        sd = self._state_dict()
        actual = self._actual_bytes(sd, ModelLumina2(), "NVFP4")
        estimated = self._build_and_estimate(tmp_path, sd, ModelLumina2(), "NVFP4")
        assert estimated == actual

    def test_nvfp4_mixed_matches_exactly(self, tmp_path):
        sd = self._state_dict()
        actual = self._actual_bytes(sd, ModelLumina2(), "NVFP4_MIXED")
        estimated = self._build_and_estimate(tmp_path, sd, ModelLumina2(), "NVFP4_MIXED")
        assert estimated == actual

    def test_unreadable_file_returns_none(self, tmp_path):
        from safetensors_quant import estimate_safetensors_output_size
        bad = tmp_path / "not_a_safetensors_file.safetensors"
        bad.write_bytes(b"garbage")
        assert estimate_safetensors_output_size(str(bad), "FP8", ModelLumina2()) is None

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

    def test_fp8_mixed_on_lumina2_is_now_render_verified(self):
        # A second same-seed/prompt user comparison (2026-08-11) confirmed
        # FP8_MIXED on lumina2 end-to-end: composition/identity/outfit
        # preserved, only a single secondary prop deviated (judged tolerable
        # quantization variance). No caveat now -- same bar INT8_MIXED
        # already met.
        _, msg = format_recommendation(ModelLumina2(), "FP8_MIXED")
        assert "not yet confirmed" not in msg

    def test_fp8_mixed_on_unverified_architecture_discloses_caveat(self):
        # flux itself is now render-verified (see safetensors_quant.py's
        # _RENDER_VERIFIED_MIXED docstring) -- qwen_image is still untested.
        _, msg = format_recommendation(ModelQwenImage(), "FP8_MIXED")
        assert "not yet confirmed" in msg

    def test_plain_fp8_on_lumina2_claims_shown_in_testing(self):
        # Plain FP8 on lumina2 is now directly render-confirmed corrupted
        # (same-seed/prompt user comparison, 2026-08-11: outfit/background
        # drift) -- the strong claim is accurate here and should say so.
        _, msg = format_recommendation(ModelLumina2(), "FP8")
        assert "plain FP8 has shown visible pose/identity corruption in testing" in msg

    def test_plain_fp8_on_unconfirmed_architecture_does_not_overclaim(self):
        # qwen_image has neither INT8 nor FP8 directly render-tested -- the
        # FP8 warning must not imply FP8 itself was tested there, the way it
        # would be wrong to claim for lumina2 before the confirmation above.
        # (flux WAS render-tested and cleared plain FP8/INT8 -- see
        # safetensors_quant.py's _RENDER_VERIFIED_MIXED docstring -- so it's
        # no longer a valid "unconfirmed" example for this test.)
        _, msg = format_recommendation(ModelQwenImage(), "FP8")
        assert "plain FP8 has shown visible pose/identity corruption in testing" not in msg

    def test_render_verified_architecture_warn_claims_shown_in_testing(self):
        # lumina2 was actually rendered end-to-end -- the strong claim is
        # accurate here and should say so plainly.
        _, msg = format_recommendation(ModelLumina2(), "INT8")
        assert "has shown visible pose/identity corruption in testing" in msg

    def test_unverified_architecture_warn_does_not_overclaim(self):
        # qwen_image's keys_hiprec was never render-tested by this project
        # (only cross-referenced against a community blacklist) -- the
        # warning must not assert corruption "shown in testing" for an
        # architecture nobody has actually rendered with this tool's output.
        # (flux WAS render-tested and cleared plain INT8 -- see
        # test_plain_int8_render_verified_on_flux_recommends_ok below --
        # so it's no longer a valid "unconfirmed" example for this test.)
        _, msg = format_recommendation(ModelQwenImage(), "INT8")
        assert "has shown visible pose/identity corruption in testing" not in msg
        assert "lumina2" in msg  # still cites the actual evidence source
        assert "hasn't been render-tested on this specific architecture" in msg

    def test_plain_int8_render_verified_on_flux_recommends_ok(self):
        # flux was directly render-tested with plain INT8 (FLUX.2 Klein 9B,
        # 2026-08-12, zero visible deviation across 3 same-seed comparisons)
        # -- recommending against it with "hasn't been render-tested" would
        # now be false, not just unconfirmed. See safetensors_quant.py's
        # _RENDER_VERIFIED_MIXED docstring for the full writeup.
        level, msg = format_recommendation(ModelFlux(), "INT8")
        assert level == "ok"
        assert "render-tested on `flux`" in msg
        assert "hasn't been render-tested" not in msg
