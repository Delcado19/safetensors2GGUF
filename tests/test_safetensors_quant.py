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
