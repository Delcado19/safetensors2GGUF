"""Tests for conversion logic and architecture detection."""

import os

import gguf
import numpy as np
import pytest
import torch

from models.architectures import (
    CosmosPredict2,
    ModelAura,
    ModelFlux,
    ModelHiDream,
    ModelHyVid,
    ModelLTXV,
    ModelLumina2,
    ModelQwenImage,
    ModelSD1,
    ModelSD3,
    ModelSDXL,
    ModelWan,
    detect_arch,
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

    def test_qwen_image(self):
        # Qwen-Image / Qwen-Image-Edit: must match before Flux/SD3 would
        # falsely trigger their banned-key guard on the shared norm_added_k
        # and add_q_proj tensors.
        sd = {
            "time_text_embed.timestep_embedder.linear_2.weight": torch.zeros(1),
            "transformer_blocks.0.attn.norm_added_q.weight": torch.zeros(1),
            "transformer_blocks.0.img_mlp.net.0.proj.weight": torch.zeros(1),
            # Tensors that Flux/SD3 would mark as banned — must not trip
            # because Qwen-Image is detected first.
            "transformer_blocks.0.attn.norm_added_k.weight": torch.zeros(1),
            "transformer_blocks.0.attn.add_q_proj.weight": torch.zeros(1),
        }
        assert detect_arch(sd).arch == "qwen_image"

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
        ModelQwenImage,
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

    def test_log_tensor_every_throttles_tensor_lines(self):
        class _FakeWriter:
            def add_array(self, *a, **kw): pass
            def add_tensor(self, *a, **kw): pass

        sd = {
            f"double_blocks.{idx}.img_attn.proj.weight": torch.zeros(64, 64)
            for idx in range(5)
        }
        logs: list[str] = []

        handle_tensors(
            _FakeWriter(),
            sd,
            ModelFlux(),
            on_log=logs.append,
            log_tensor_every=2,
        )

        tensor_logs = [line for line in logs if "torch.float32" in line]
        assert len(tensor_logs) == 4  # first, every 2nd, last

    def test_lumina2_pad_tokens_unsqueeze_can_be_disabled_for_kquant_intermediate(self):
        class _FakeWriter:
            def __init__(self):
                self.shapes = {}

            def add_array(self, *a, **kw): pass

            def add_tensor(self, key, data, raw_dtype=None):
                self.shapes[key] = tuple(data.shape)

        sd = {"x_pad_token": torch.zeros(3840, dtype=torch.float32)}

        writer = _FakeWriter()
        handle_tensors(writer, sd, ModelLumina2(), on_log=lambda *a: None)
        assert writer.shapes["x_pad_token"] == (1, 3840)

        writer_no_unsqueeze = _FakeWriter()
        handle_tensors(
            writer_no_unsqueeze,
            sd,
            ModelLumina2(),
            on_log=lambda *a: None,
            apply_unsqueeze=False,
        )
        assert writer_no_unsqueeze.shapes["x_pad_token"] == (3840,)


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

    def test_strip_prefixes_false_keeps_model_prefix_intact(self):
        # A standalone text-encoder state dict's "model." is its own genuine
        # HF module path (e.g. Qwen3's "model.layers.0...."), not a
        # diffusion-checkpoint wrapper -- callers that know the source is a
        # text encoder must be able to opt out of the default stripping, or
        # ComfyUI's own text-encoder architecture detection (comfy/sd.py's
        # detect_te_model, which looks for literal "model.layers.0...."
        # keys) breaks on the output file.
        sd = {"model.layers.0.self_attn.q_proj.weight": torch.zeros(1)}
        result = strip_prefix(sd, strip_prefixes=False)
        assert "model.layers.0.self_attn.q_proj.weight" in result
        assert "layers.0.self_attn.q_proj.weight" not in result


# ---------------------------------------------------------------------------
# 5D tensor handling: multiple 5D tensors must not crash (Issue #291)
# ---------------------------------------------------------------------------

class TestFiveDTensorHandling:
    def test_multiple_5d_tensors_append(self, tmp_path, monkeypatch):
        """Two 5D tensors for the same arch must both land in the fix file."""
        monkeypatch.chdir(tmp_path)
        arch = ModelWan()

        data1 = np.zeros((2, 3, 4, 5, 6), dtype=np.float32)
        data2 = np.zeros((1, 2, 3, 4, 5), dtype=np.float32)

        arch.handle_nd_tensor("rope.freqs", data1)
        arch.handle_nd_tensor("rope.freqs2", data2)   # must not raise

        from safetensors.torch import load_file
        fix = load_file("fix_5d_tensors_wan.safetensors")
        assert "rope.freqs" in fix
        assert "rope.freqs2" in fix


# ---------------------------------------------------------------------------
# Float8 dtype handling — regression for Lumina2 (Issue: nan_to_num unsupported)
# ---------------------------------------------------------------------------

class TestFloat8Handling:
    @pytest.mark.skipif(
        not hasattr(torch, "float8_e4m3fn"),
        reason="float8_e4m3fn not available in this PyTorch build",
    )
    def test_float8_e4m3fn_does_not_raise(self, tmp_path):
        """handle_tensors must convert float8 to float16 before nan_to_num."""
        class _FakeWriter:
            def add_array(self, *a, **kw): pass
            def add_tensor(self, *a, **kw): pass

        sd = {"img_mod.weight": torch.zeros(256, 256, dtype=torch.float8_e4m3fn)}
        handle_tensors(_FakeWriter(), sd, ModelLumina2())  # must not raise

    @pytest.mark.skipif(
        not hasattr(torch, "float8_e5m2"),
        reason="float8_e5m2 not available in this PyTorch build",
    )
    def test_float8_e5m2_does_not_raise(self, tmp_path):
        class _FakeWriter:
            def add_array(self, *a, **kw): pass
            def add_tensor(self, *a, **kw): pass

        sd = {"img_mod.weight": torch.zeros(256, 256, dtype=torch.float8_e5m2)}
        handle_tensors(_FakeWriter(), sd, ModelLumina2())  # must not raise
