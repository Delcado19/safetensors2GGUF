"""Tests for safetensors_quant_nvfp4.py — Nvidia NVFP4 block-scaled quantization."""

from __future__ import annotations

import torch

from safetensors_quant_nvfp4 import quantize_nvfp4


class TestQuantizeNvfp4:
    def test_returns_three_tensors(self):
        data = torch.randn(32, 32, dtype=torch.float32)
        out = quantize_nvfp4(data, "block.weight")
        assert set(out.keys()) == {
            "block.weight", "block.weight.weight_scale", "block.weight.weight_scale_2",
        }

    def test_weight_is_packed_uint8_half_last_dim(self):
        data = torch.randn(4, 32, dtype=torch.float32)
        out = quantize_nvfp4(data, "block.weight")
        w = out["block.weight"]
        assert w.dtype == torch.uint8
        assert w.shape == (4, 16)  # 32 elems / 2 per byte

    def test_scale_is_fp8_e4m3fn_per_16_block(self):
        data = torch.randn(4, 32, dtype=torch.float32)
        out = quantize_nvfp4(data, "block.weight")
        scale = out["block.weight.weight_scale"]
        assert scale.dtype == torch.float8_e4m3fn
        assert scale.shape == (4, 2)  # 32 elems / 16-block

    def test_scale_2_is_global_float32_scalar(self):
        data = torch.randn(4, 32, dtype=torch.float32)
        out = quantize_nvfp4(data, "block.weight")
        s2 = out["block.weight.weight_scale_2"]
        assert s2.dtype == torch.float32
        assert s2.numel() == 1

    def test_raises_on_non_multiple_of_16_last_dim(self):
        data = torch.randn(4, 17, dtype=torch.float32)
        try:
            quantize_nvfp4(data, "block.weight")
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_dequant_reconstructs_within_nvfp4_tolerance(self):
        torch.manual_seed(0)
        data = torch.randn(8, 64, dtype=torch.float32) * 3
        out = quantize_nvfp4(data, "block.weight")
        from safetensors_quant_nvfp4 import dequantize_nvfp4  # test-only helper
        recon = dequantize_nvfp4(out, "block.weight")
        assert recon.shape == data.shape
        # 4-bit float has coarse steps; allow generous relative tolerance
        assert torch.allclose(recon, data, atol=data.abs().max().item() * 0.35)
