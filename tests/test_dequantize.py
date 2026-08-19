"""Tests for dequantize.py — reconstructing already-quantized layers so they
can be cleanly re-quantized instead of refusing outright."""

from __future__ import annotations

import json

import torch

from dequantize import detect_quantized_weight, dequantize_weight
from safetensors_quant_fp8 import quantize_fp8_scaled
from safetensors_quant_int8 import quantize_int8_convrot, quantize_int8_tensorwise


class TestDetectQuantizedWeight:
    def test_returns_none_for_unquantized_float_weight(self):
        sd = {"layer.weight": torch.randn(8, 8, dtype=torch.float32)}
        assert detect_quantized_weight(sd, "layer.weight") is None

    def test_returns_none_for_non_weight_key(self):
        sd = {"layer.bias": torch.randint(-128, 127, (8,), dtype=torch.int8)}
        assert detect_quantized_weight(sd, "layer.bias") is None

    def test_detects_plain_int8_via_scale_heuristic(self):
        data = torch.randn(8, 8, dtype=torch.float32)
        sd = quantize_int8_tensorwise(data, "layer.weight")
        assert detect_quantized_weight(sd, "layer.weight") == "int8_tensorwise"

    def test_detects_fp8_via_scale_heuristic(self):
        data = torch.randn(8, 8, dtype=torch.float32)
        sd = quantize_fp8_scaled(data, "layer.weight")
        assert detect_quantized_weight(sd, "layer.weight") == "float8_e4m3fn"

    def test_comfy_quant_sidecar_takes_priority_over_heuristic(self):
        data = torch.randn(8, 256, dtype=torch.float32)
        sd = quantize_int8_convrot(data, "layer.weight")
        conf = json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256})
        sd["layer.comfy_quant"] = torch.frombuffer(bytearray(conf.encode("utf-8")), dtype=torch.uint8)
        assert detect_quantized_weight(sd, "layer.weight") == "int8_tensorwise"

    def test_malformed_comfy_quant_sidecar_falls_back_to_heuristic(self):
        data = torch.randn(8, 8, dtype=torch.float32)
        sd = quantize_int8_tensorwise(data, "layer.weight")
        sd["layer.comfy_quant"] = torch.zeros(8, dtype=torch.uint8)  # not valid JSON
        assert detect_quantized_weight(sd, "layer.weight") == "int8_tensorwise"

    def test_detects_comfy_org_scale_weight_naming(self):
        # Regression: Comfy-Org's own "fp8_scaled" repackaging convention
        # uses ".scale_weight" (reversed word order), not this tool's
        # ".weight_scale". Found 2026-08-18 converting HiDream's
        # llama_3.1_8b_instruct_fp8_scaled.safetensors -- with only
        # ".weight_scale" recognized, the actual raw float8_e4m3fn ".weight"
        # tensors were never detected as quantized at all (fed unscaled into
        # the normal pipeline, numerically wrong output), and the orphaned
        # 0-dim ".scale_weight" sidecar crashed NVFP4's quantize_nvfp4() on
        # `.shape[-1]` for a shape with zero dimensions.
        sd = {
            "layer.weight": torch.zeros(8, 8, dtype=torch.float8_e4m3fn),
            "layer.scale_weight": torch.tensor(2.0, dtype=torch.float32),
        }
        assert detect_quantized_weight(sd, "layer.weight") == "float8_e4m3fn"


class TestDequantizeWeight:
    def test_plain_int8_round_trips_close(self):
        original = torch.randn(16, 16, dtype=torch.float32)
        sd = quantize_int8_tensorwise(original, "layer.weight")
        restored = dequantize_weight(sd, "layer.weight", "int8_tensorwise", sd["layer.weight"])
        scale = sd["layer.weight_scale"].item()
        assert torch.allclose(restored, original, atol=scale * 1.5)

    def test_convrot_int8_round_trips_close(self):
        original = torch.randn(8, 256, dtype=torch.float32)
        sd = quantize_int8_convrot(original, "layer.weight")
        restored = dequantize_weight(sd, "layer.weight", "int8_tensorwise", sd["layer.weight"])
        max_scale = sd["layer.weight_scale"].max().item()
        assert torch.allclose(restored, original, atol=max_scale * 3)

    def test_fp8_round_trips_close(self):
        original = torch.randn(16, 16, dtype=torch.float32)
        sd = quantize_fp8_scaled(original, "layer.weight")
        restored = dequantize_weight(sd, "layer.weight", "float8_e4m3fn", sd["layer.weight"])
        scale = sd["layer.weight_scale"].item()
        # float8_e4m3fn has a 3-bit mantissa: error is proportional to
        # magnitude (relative), not a fixed absolute step like INT8.
        assert torch.allclose(restored, original, atol=scale, rtol=0.15)

    def test_scale_weight_naming_round_trips_close(self):
        # Companion to test_detects_comfy_org_scale_weight_naming: the
        # actual value reconstruction must also work through the alias, not
        # just detection.
        original = torch.randn(16, 16, dtype=torch.float32)
        sd = quantize_fp8_scaled(original, "layer.weight")
        sd["layer.scale_weight"] = sd.pop("layer.weight_scale")
        restored = dequantize_weight(sd, "layer.weight", "float8_e4m3fn", sd["layer.weight"])
        scale = sd["layer.scale_weight"].item()
        assert torch.allclose(restored, original, atol=scale, rtol=0.15)

    def test_convrot_with_incompatible_group_size_raises(self):
        # weight_scale shaped as per-row (ConvRot-looking) but in_features
        # isn't divisible by any sane group_size — must raise clearly, not
        # crash with an opaque reshape error.
        sd = {
            "layer.weight": torch.randint(-128, 127, (4, 10), dtype=torch.int8),
            "layer.weight_scale": torch.randn(4, 1, dtype=torch.float32),
        }
        import pytest
        with pytest.raises(ValueError, match="divisible"):
            dequantize_weight(sd, "layer.weight", "int8_tensorwise", sd["layer.weight"])
