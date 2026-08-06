"""Tests for safetensors_quant_fp8.py — scaled FP8 quantization."""

from __future__ import annotations

import torch

from safetensors_quant_fp8 import quantize_fp8_scaled


class TestQuantizeFp8Scaled:
    def test_returns_weight_and_scale(self):
        data = torch.randn(64, 64, dtype=torch.float32) * 10
        out = quantize_fp8_scaled(data, "block.weight")
        assert set(out.keys()) == {"block.weight", "block.weight.weight_scale"}

    def test_weight_dtype_is_fp8_e4m3fn(self):
        data = torch.randn(64, 64, dtype=torch.float32)
        out = quantize_fp8_scaled(data, "block.weight")
        assert out["block.weight"].dtype == torch.float8_e4m3fn

    def test_scale_dtype_is_float32_scalar(self):
        data = torch.randn(64, 64, dtype=torch.float32)
        out = quantize_fp8_scaled(data, "block.weight")
        scale = out["block.weight.weight_scale"]
        assert scale.dtype == torch.float32
        assert scale.numel() == 1

    def test_dequant_reconstructs_within_fp8_tolerance(self):
        torch.manual_seed(0)
        data = torch.randn(128, 128, dtype=torch.float32) * 5
        out = quantize_fp8_scaled(data, "block.weight")
        recon = out["block.weight"].to(torch.float32) * out["block.weight.weight_scale"]
        # FP8 e4m3fn has ~2 decimal digits of mantissa precision
        assert torch.allclose(recon, data, atol=data.abs().max().item() * 0.1)

    def test_zero_tensor_does_not_divide_by_zero(self):
        data = torch.zeros(16, 16, dtype=torch.float32)
        out = quantize_fp8_scaled(data, "block.weight")
        assert torch.isfinite(out["block.weight"].to(torch.float32)).all()
        assert torch.isfinite(out["block.weight.weight_scale"]).all()
