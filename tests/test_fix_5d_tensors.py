"""Tests for fix_5d_tensors.py."""

import os
import tempfile

import gguf
import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from fix_5d_tensors import fix_5d_tensors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_minimal_gguf(path, arch="wan", tensors=None):
    """Write a minimal GGUF with the given tensors (all F32)."""
    writer = gguf.GGUFWriter(path=None, arch=arch)
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_F16)
    for name, data in (tensors or {}).items():
        arr = data.numpy().astype(np.float32)
        q = gguf.quants.quantize(arr, gguf.GGMLQuantizationType.F32)
        writer.add_tensor(name, q, raw_dtype=gguf.GGMLQuantizationType.F32)
    writer.write_header_to_file(path=path)
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()


def _make_fix_file(path, tensors):
    """Write a side-car safetensors fix file."""
    save_file({k: v.clone() for k, v in tensors.items()}, path)


# ---------------------------------------------------------------------------
# Basic round-trip
# ---------------------------------------------------------------------------

class TestFix5dRoundtrip:
    def test_5d_tensor_inserted(self, tmp_path):
        """A 5D tensor from the fix file must appear in the output GGUF."""
        src = str(tmp_path / "src.gguf")
        dst = str(tmp_path / "dst.gguf")
        fix = str(tmp_path / "fix.safetensors")

        _make_minimal_gguf(src, tensors={"blocks.0.attn.bias": torch.zeros(4, 4)})
        _make_fix_file(fix, {"rope.freqs": torch.zeros(2, 3, 4, 5, 6)})

        inserted = fix_5d_tensors(src, dst, fix_path=fix, overwrite=True)

        assert "rope.freqs" in inserted
        reader = gguf.GGUFReader(dst)
        names = {t.name for t in reader.tensors}
        assert "rope.freqs" in names
        assert "blocks.0.attn.bias" in names

    def test_5d_tensor_stored_as_f32(self, tmp_path):
        """Inserted 5D tensors must always be F32."""
        src = str(tmp_path / "src.gguf")
        dst = str(tmp_path / "dst.gguf")
        fix = str(tmp_path / "fix.safetensors")

        _make_minimal_gguf(src, tensors={"w": torch.zeros(4, 4)})
        _make_fix_file(fix, {"rope.freqs": torch.zeros(2, 3, 4, 5, 6)})

        fix_5d_tensors(src, dst, fix_path=fix, overwrite=True)

        reader = gguf.GGUFReader(dst)
        t = next(t for t in reader.tensors if t.name == "rope.freqs")
        assert t.tensor_type == gguf.GGMLQuantizationType.F32

    def test_multiple_5d_tensors(self, tmp_path):
        """All 5D tensors in the fix file must be inserted."""
        src = str(tmp_path / "src.gguf")
        dst = str(tmp_path / "dst.gguf")
        fix = str(tmp_path / "fix.safetensors")

        _make_minimal_gguf(src, tensors={"w": torch.zeros(4, 4)})
        _make_fix_file(fix, {
            "rope.freqs":  torch.zeros(2, 3, 4, 5, 6),
            "rope.freqs2": torch.zeros(1, 2, 3, 4, 5),
        })

        inserted = fix_5d_tensors(src, dst, fix_path=fix, overwrite=True)

        assert len(inserted) == 2
        reader = gguf.GGUFReader(dst)
        names = {t.name for t in reader.tensors}
        assert "rope.freqs" in names
        assert "rope.freqs2" in names

    def test_placement_after_bias(self, tmp_path):
        """5D weight is inserted directly after its .bias counterpart."""
        src = str(tmp_path / "src.gguf")
        dst = str(tmp_path / "dst.gguf")
        fix = str(tmp_path / "fix.safetensors")

        _make_minimal_gguf(src, tensors={
            "blocks.0.other": torch.zeros(4, 4),
            "blocks.0.attn.bias": torch.zeros(4),
        })
        _make_fix_file(fix, {"blocks.0.attn.weight": torch.zeros(2, 3, 4, 5, 6)})

        fix_5d_tensors(src, dst, fix_path=fix, overwrite=True)

        reader = gguf.GGUFReader(dst)
        names = [t.name for t in reader.tensors]
        bias_idx = names.index("blocks.0.attn.bias")
        weight_idx = names.index("blocks.0.attn.weight")
        assert weight_idx == bias_idx + 1, "5D weight must follow its bias"

    def test_overwrite_guard(self, tmp_path):
        """Without --overwrite, raising OSError if dst exists."""
        src = str(tmp_path / "src.gguf")
        dst = str(tmp_path / "dst.gguf")
        fix = str(tmp_path / "fix.safetensors")

        _make_minimal_gguf(src, tensors={"w": torch.zeros(4, 4)})
        _make_fix_file(fix, {"rope.freqs": torch.zeros(2, 3, 4, 5, 6)})
        open(dst, "w").close()  # create empty dst

        with pytest.raises(OSError, match="overwrite"):
            fix_5d_tensors(src, dst, fix_path=fix, overwrite=False)

    def test_missing_fix_file_raises(self, tmp_path):
        src = str(tmp_path / "src.gguf")
        _make_minimal_gguf(src, tensors={"w": torch.zeros(4, 4)})

        with pytest.raises(FileNotFoundError, match="fix file not found"):
            fix_5d_tensors(src, str(tmp_path / "dst.gguf"),
                           fix_path=str(tmp_path / "nonexistent.safetensors"))

    def test_arch_preserved(self, tmp_path):
        """Output GGUF must have the same architecture as the input."""
        src = str(tmp_path / "src.gguf")
        dst = str(tmp_path / "dst.gguf")
        fix = str(tmp_path / "fix.safetensors")

        _make_minimal_gguf(src, arch="hyvid", tensors={"w": torch.zeros(4, 4)})
        _make_fix_file(fix, {"rope.freqs": torch.zeros(2, 3, 4, 5, 6)})

        fix_5d_tensors(src, dst, fix_path=fix, overwrite=True)

        reader = gguf.GGUFReader(dst)
        field = reader.get_field("general.architecture")
        arch = str(field.parts[field.data[-1]], "utf-8")
        assert arch == "hyvid"
