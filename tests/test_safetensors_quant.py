"""Tests for safetensors_quant.py — dtype registry and per-tensor quantization."""

from __future__ import annotations

import torch

from safetensors_quant import (
    SAFETENSORS_DTYPE_CHOICES,
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
        keys = {k for _, k in SAFETENSORS_DTYPE_CHOICES}
        assert keys == {"F16", "F16_MIXED", "FP8", "FP8_MIXED", "NVFP4", "NVFP4_MIXED"}

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
