"""Tests for safetensors_quant.py — dtype registry and per-tensor quantization."""

from __future__ import annotations

import torch

from safetensors_quant import (
    SAFETENSORS_DTYPE_CHOICES,
    is_hiprec_st,
    quantize_tensor_st,
)
from models.architectures import ModelFlux, ModelLumina2


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
