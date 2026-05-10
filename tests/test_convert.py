"""Tests for conversion logic and architecture detection."""

import os
import tempfile

import gguf
import numpy as np
import pytest
import torch

from models.architectures import (
    QUANTIZATION_THRESHOLD,
    REARRANGE_THRESHOLD,
    CosmosPredict2,
    ModelAura,
    ModelFlux,
    ModelHiDream,
    ModelHyVid,
    ModelLTXV,
    ModelLumina2,
    ModelSD1,
    ModelSD3,
    ModelSDXL,
    ModelWan,
    detect_arch,
    is_model_arch,
)
from convert import _quant_type_for, handle_tensors, strip_prefix


# ---------------------------------------------------------------------------
# Architecture detection
# ---------------------------------------------------------------------------

class TestDetectArch:
    def test_flux_diffusers(self):
        sd = {"transformer_blocks.0.attn.norm_added_k.weight": torch.zeros(1)}
        # diffusers format is in keys_banned → should raise
        with pytest.raises(AssertionError, match="not allowed"):
            detect_arch(sd)

    def test_flux_non_diffusers(self):
        sd = {"double_blocks.0.img_attn.proj.weight": torch.zeros(1)}
        assert detect_arch(sd).arch == "flux"

    def test_sd1(self):
        sd = {"down_blocks.0.downsamplers.0.conv.weight": torch.zeros(1)}
        assert detect_arch(sd).arch == "sd1"

    def test_sdxl(self):
        sd = {
            "down_blocks.0.downsamplers.0.conv.weight": torch.zeros(1),
            "add_embedding.linear_1.weight": torch.zeros(1),
        }
        assert detect_arch(sd).arch == "sdxl"

    def test_lumina2(self):
        sd = {
            "cap_embedder.1.weight": torch.zeros(1),
            "context_refiner.0.attention.qkv.weight": torch.zeros(1),
        }
        assert detect_arch(sd).arch == "lumina2"

    def test_unknown_raises(self):
        with pytest.raises(AssertionError, match="Unknown model architecture"):
            detect_arch({"some.random.key": torch.zeros(1)})


# ---------------------------------------------------------------------------
# shape_fix: only SD1 and SDXL should have it enabled
# ---------------------------------------------------------------------------

class TestShapeFix:
    @pytest.mark.parametrize("cls", [ModelSD1, ModelSDXL])
    def test_shape_fix_enabled(self, cls):
        assert cls().shape_fix is True

    @pytest.mark.parametrize("cls", [
        ModelFlux, ModelSD3, ModelAura, ModelHiDream,
        CosmosPredict2, ModelLTXV, ModelHyVid, ModelWan, ModelLumina2,
    ])
    def test_shape_fix_disabled(self, cls):
        assert cls().shape_fix is False


# ---------------------------------------------------------------------------
# Quantization type selection
# ---------------------------------------------------------------------------

class TestQuantType:
    def test_1d_tensor_always_f32(self):
        data = torch.zeros(256)
        qtype = _quant_type_for(data, "some.weight", ModelFlux(), torch.float32)
        assert qtype == gguf.GGMLQuantizationType.F32

    def test_small_tensor_f32(self):
        # n_params = 32*32 = 1024 ≤ QUANTIZATION_THRESHOLD
        data = torch.zeros(32, 32)
        qtype = _quant_type_for(data, "key", ModelFlux(), torch.float32)
        assert qtype == gguf.GGMLQuantizationType.F32

    def test_large_tensor_f16(self):
        data = torch.zeros(64, 64)  # 4096 params > threshold
        qtype = _quant_type_for(data, "key", ModelFlux(), torch.float16)
        assert qtype == gguf.GGMLQuantizationType.F16

    def test_large_tensor_bf16_source(self):
        data = torch.zeros(64, 64)
        qtype = _quant_type_for(data, "key", ModelFlux(), torch.bfloat16)
        assert qtype == gguf.GGMLQuantizationType.BF16

    def test_hiprec_key_f32(self):
        data = torch.zeros(64, 64)
        qtype = _quant_type_for(data, "x_pad_token", ModelLumina2(), torch.bfloat16)
        assert qtype == gguf.GGMLQuantizationType.F32


# ---------------------------------------------------------------------------
# nan_to_num clamping
# ---------------------------------------------------------------------------

class TestNanToNum:
    def _run_handle(self, tensor_dict, model_arch, tmp_path):
        """Helper: write tensors via handle_tensors, return dst path."""
        dst = str(tmp_path / "out.gguf")
        writer = gguf.GGUFWriter(path=None, arch=model_arch.arch)
        writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
        writer.add_file_type(gguf.LlamaFileType.MOSTLY_F16)
        handle_tensors(writer, tensor_dict, model_arch)
        writer.write_header_to_file(path=dst)
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=False)
        writer.close()
        return dst

    def test_inf_values_do_not_crash(self, tmp_path):
        """Tensors with inf values must be clamped, not cause a crash."""
        sd = {"double_blocks.0.img_attn.proj.weight": torch.full((64, 64), float("inf"))}
        dst = self._run_handle(sd, ModelFlux(), tmp_path)
        assert os.path.isfile(dst)

    def test_nan_values_do_not_crash(self, tmp_path):
        sd = {"double_blocks.0.img_attn.proj.weight": torch.full((64, 64), float("nan"))}
        dst = self._run_handle(sd, ModelFlux(), tmp_path)
        assert os.path.isfile(dst)


# ---------------------------------------------------------------------------
# shape_fix round-trip: reshape metadata must survive write → read
# ---------------------------------------------------------------------------

class TestShapeFixRoundtrip:
    def _make_sd1_tensor(self):
        """Return an SD1-keyed tensor that triggers shape_fix reshaping.

        Conditions (from handle_tensors):
          - shape_fix=True (SD1)
          - ndim > 1
          - n_params >= REARRANGE_THRESHOLD (512)
          - n_params divisible by 256
          - last dim NOT divisible by 256
        Shape (4, 320): n_params=1280 ✓, last=320 (320/256=1.25) ✓
        """
        return torch.randn(4, 320)

    def test_orig_shape_metadata_written(self, tmp_path):
        key = "down_blocks.0.downsamplers.0.conv.weight"
        sd = {key: self._make_sd1_tensor()}
        dst = str(tmp_path / "sd1.gguf")

        writer = gguf.GGUFWriter(path=None, arch="sd1")
        writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
        writer.add_file_type(gguf.LlamaFileType.MOSTLY_F16)
        handle_tensors(writer, sd, ModelSD1())
        writer.write_header_to_file(path=dst)
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=False)
        writer.close()

        reader = gguf.GGUFReader(dst)
        field = reader.get_field(f"comfy.gguf.orig_shape.{key}")
        assert field is not None, "orig_shape metadata must be present after reshape"

        restored = tuple(int(field.parts[i][0]) for i in field.data)
        assert restored == (4, 320), f"Expected (4, 320), got {restored}"

    def test_flux_no_orig_shape_metadata(self, tmp_path):
        """Flux must never write orig_shape metadata (shape_fix=False)."""
        key = "double_blocks.0.img_attn.proj.weight"
        sd = {key: torch.randn(4, 320)}
        dst = str(tmp_path / "flux.gguf")

        writer = gguf.GGUFWriter(path=None, arch="flux")
        writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
        writer.add_file_type(gguf.LlamaFileType.MOSTLY_F16)
        handle_tensors(writer, sd, ModelFlux())
        writer.write_header_to_file(path=dst)
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=False)
        writer.close()

        reader = gguf.GGUFReader(dst)
        field = reader.get_field(f"comfy.gguf.orig_shape.{key}")
        assert field is None, "Flux must not write orig_shape metadata"


# ---------------------------------------------------------------------------
# strip_prefix
# ---------------------------------------------------------------------------

class TestStripPrefix:
    def test_strips_diffusion_model_prefix(self):
        sd = {"model.diffusion_model.blocks.0.weight": torch.zeros(1)}
        result = strip_prefix(sd)
        assert "blocks.0.weight" in result
        assert "model.diffusion_model.blocks.0.weight" not in result

    def test_no_prefix_unchanged(self):
        sd = {"blocks.0.weight": torch.zeros(1), "blocks.1.weight": torch.zeros(1)}
        result = strip_prefix(sd)
        assert set(result.keys()) == {"blocks.0.weight", "blocks.1.weight"}
