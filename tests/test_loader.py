"""Tests for loader.py — including the critical shape_fix round-trip."""

import gguf
import numpy as np
import torch

from convert import handle_tensors
from loader import load_gguf
from models.architectures import ModelFlux, ModelSD1, ModelSDXL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_gguf(path, arch_obj, state_dict):
    """Convert state_dict to GGUF via handle_tensors and write to path."""
    writer = gguf.GGUFWriter(path=None, arch=arch_obj.arch)
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_F16)
    handle_tensors(writer, state_dict, arch_obj)
    writer.write_header_to_file(path=path)
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()


# ---------------------------------------------------------------------------
# Shape-fix round-trip — THE critical test (closes the matmul-bug loop)
# ---------------------------------------------------------------------------

class TestShapeFixRoundtrip:
    """Verify that convert → write GGUF → load → reshape gives back the original shape."""

    def test_sd1_reshaped_tensor_restores(self, tmp_path):
        """SD1 tensor (4, 320): written as (5, 256) with orig_shape, loaded back as (4, 320)."""
        key = "down_blocks.0.downsamplers.0.conv.weight"
        original = torch.randn(4, 320)
        dst = str(tmp_path / "sd1.gguf")

        _write_gguf(dst, ModelSD1(), {key: original})

        sd, meta = load_gguf(dst)

        assert sd[key].shape == (4, 320), (
            f"Expected (4, 320), got {sd[key].shape} — orig_shape not restored"
        )
        assert meta["arch"] == "sd1"

    def test_sdxl_reshaped_tensor_restores(self, tmp_path):
        key = "down_blocks.0.downsamplers.0.conv.weight"
        original = torch.randn(4, 320)
        dst = str(tmp_path / "sdxl.gguf")
        # Need SDXL detection keys alongside the conv weight
        sd_in = {
            key: original,
            "add_embedding.linear_1.weight": torch.randn(64, 64),
        }
        _write_gguf(dst, ModelSDXL(), sd_in)

        sd, meta = load_gguf(dst)
        assert sd[key].shape == (4, 320)
        assert meta["arch"] == "sdxl"

    def test_flux_no_reshape(self, tmp_path):
        """Flux tensors must NOT be reshaped (shape_fix=False)."""
        key = "double_blocks.0.img_attn.proj.weight"
        original = torch.randn(4, 320)
        dst = str(tmp_path / "flux.gguf")

        _write_gguf(dst, ModelFlux(), {key: original})

        sd, _ = load_gguf(dst)
        assert sd[key].shape == (4, 320)

    def test_values_preserved_within_f16_tolerance(self, tmp_path):
        """Round-trip through F16 GGUF must preserve values within ±1e-3."""
        key = "double_blocks.0.img_attn.proj.weight"
        original = torch.randn(64, 64).to(torch.float16)
        dst = str(tmp_path / "flux.gguf")

        _write_gguf(dst, ModelFlux(), {key: original})
        sd, _ = load_gguf(dst)

        loaded = sd[key].float()
        original_f = original.float()
        assert torch.allclose(original_f, loaded, atol=1e-3), (
            f"Max deviation: {(original_f - loaded).abs().max().item():.6f}"
        )

    def test_f32_1d_tensor_exact(self, tmp_path):
        """1D tensors are stored as F32 and must round-trip exactly."""
        key = "some.norm.weight"
        original = torch.randn(256)
        dst = str(tmp_path / "flux.gguf")

        _write_gguf(dst, ModelFlux(), {key: original})
        sd, _ = load_gguf(dst)

        assert torch.allclose(original, sd[key], atol=0.0)


# ---------------------------------------------------------------------------
# Metadata reading
# ---------------------------------------------------------------------------

class TestMetadata:
    def _make_gguf(self, tmp_path, arch_obj):
        dst = str(tmp_path / "model.gguf")
        _write_gguf(dst, arch_obj, {
            "double_blocks.0.img_attn.proj.weight": torch.randn(64, 64)
        })
        return dst

    def test_arch_in_metadata(self, tmp_path):
        path = self._make_gguf(tmp_path, ModelFlux())
        _, meta = load_gguf(path)
        assert meta["arch"] == "flux"

    def test_quantization_version_present(self, tmp_path):
        path = self._make_gguf(tmp_path, ModelFlux())
        _, meta = load_gguf(path)
        assert meta["quantization_version"] is not None

    def test_flat0_scalar_read(self, tmp_path):
        """Metadata reads must not crash on size-1 1D arrays (Issue #384)."""
        path = self._make_gguf(tmp_path, ModelFlux())
        # load_gguf uses _read_scalar_field internally; if Issue #384 is present
        # this call would raise "only 0-dimensional arrays can be converted to Python scalars"
        _, meta = load_gguf(path)
        assert isinstance(meta["file_type"], int)


# ---------------------------------------------------------------------------
# Raw (non-dequantized) mode
# ---------------------------------------------------------------------------

class TestRawMode:
    def test_raw_mode_returns_tuples(self, tmp_path):
        key = "double_blocks.0.img_attn.proj.weight"
        dst = str(tmp_path / "flux.gguf")
        _write_gguf(dst, ModelFlux(), {key: torch.randn(64, 64)})

        sd, _ = load_gguf(dst, dequantize=False)
        data, qtype, shape = sd[key]

        assert isinstance(data, np.ndarray)
        assert isinstance(shape, torch.Size)
        assert shape == torch.Size([64, 64])
