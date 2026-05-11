"""Tests for quantize.py — type registry and llama-quantize subprocess wrapper."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import json
import struct

from quantize import (
    ALL_QUANT_CHOICES,
    LLAMA_QUANT_KEYS,
    PYTHON_PRECISIONS,
    SIZE_RATIOS,
    _QUANT_BYTES_PER_ELEM,
    _QUANTIZATION_THRESHOLD,
    estimate_output_size,
    find_exe,
    run_quantize,
)


# ---------------------------------------------------------------------------
# Type registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_all_quant_choices_is_list_of_tuples(self):
        assert isinstance(ALL_QUANT_CHOICES, list)
        for label, key in ALL_QUANT_CHOICES:
            assert isinstance(label, str) and label
            assert isinstance(key, str) and key

    def test_no_duplicate_keys(self):
        keys = [k for _, k in ALL_QUANT_CHOICES]
        assert len(keys) == len(set(keys)), "Duplicate quantization keys in ALL_QUANT_CHOICES"

    def test_python_precisions_keys_present_in_choices(self):
        choice_keys = {k for _, k in ALL_QUANT_CHOICES}
        for key in PYTHON_PRECISIONS:
            assert key in choice_keys, f"PYTHON_PRECISIONS key '{key}' missing from ALL_QUANT_CHOICES"

    def test_llama_keys_present_in_choices(self):
        choice_keys = {k for _, k in ALL_QUANT_CHOICES}
        for key in LLAMA_QUANT_KEYS:
            assert key in choice_keys, f"LLAMA_QUANT_KEYS entry '{key}' missing from ALL_QUANT_CHOICES"

    def test_python_and_llama_sets_are_disjoint(self):
        overlap = set(PYTHON_PRECISIONS) & LLAMA_QUANT_KEYS
        assert not overlap, f"Keys appear in both sets: {overlap}"

    def test_lq_label_for_llama_types(self):
        label_map = {k: lbl for lbl, k in ALL_QUANT_CHOICES}
        for key in LLAMA_QUANT_KEYS:
            if key in label_map:
                assert "[lq]" in label_map[key], (
                    f"llama-quantize type '{key}' is missing '[lq]' marker in its label"
                )

    def test_python_precisions_have_gguf_types(self):
        import gguf
        for key, (qtype, ftype) in PYTHON_PRECISIONS.items():
            assert isinstance(qtype, gguf.GGMLQuantizationType), key
            assert isinstance(ftype, gguf.LlamaFileType), key

    def test_required_python_types_present(self):
        for key in ("F16", "BF16", "F32", "Q8_0"):
            assert key in PYTHON_PRECISIONS, f"Expected '{key}' in PYTHON_PRECISIONS"

    def test_required_kquant_types_present(self):
        for key in ("Q4_K_M", "Q4_K_S", "Q5_K_M", "Q6_K", "Q3_K_M", "Q2_K"):
            assert key in LLAMA_QUANT_KEYS, f"Expected '{key}' in LLAMA_QUANT_KEYS"

    def test_size_ratios_keys_match_choices(self):
        choice_keys = {k for _, k in ALL_QUANT_CHOICES}
        for key in SIZE_RATIOS:
            assert key in choice_keys, f"SIZE_RATIOS key '{key}' missing from ALL_QUANT_CHOICES"

    def test_size_ratios_are_positive_floats(self):
        for key, ratio in SIZE_RATIOS.items():
            assert isinstance(ratio, float) and ratio > 0, f"Bad ratio for '{key}': {ratio}"

    def test_f16_ratio_is_baseline(self):
        assert SIZE_RATIOS["F16"] == 1.0

    def test_f32_ratio_is_double_f16(self):
        assert SIZE_RATIOS["F32"] == 2.0

    def test_kquant_ratios_smaller_than_f16(self):
        for key in LLAMA_QUANT_KEYS:
            if key in SIZE_RATIOS:
                assert SIZE_RATIOS[key] < 1.0, f"Expected ratio < 1.0 for '{key}'"


# ---------------------------------------------------------------------------
# find_exe
# ---------------------------------------------------------------------------

class TestFindExe:
    def test_returns_none_when_missing(self, tmp_path):
        with patch("quantize.DEFAULT_EXE", tmp_path / "nonexistent.exe"):
            assert find_exe() is None

    def test_returns_path_when_present(self, tmp_path):
        fake_exe = tmp_path / "llama-quantize.exe"
        fake_exe.touch()
        with patch("quantize.DEFAULT_EXE", fake_exe):
            result = find_exe()
            assert result == fake_exe


# ---------------------------------------------------------------------------
# run_quantize — subprocess interaction
# ---------------------------------------------------------------------------

def _mock_proc(lines: list[str], returncode: int = 0) -> MagicMock:
    proc = MagicMock()
    proc.stdout = iter(lines)
    proc.returncode = returncode
    proc.wait.return_value = None
    return proc


class TestRunQuantize:
    def test_raises_file_not_found_when_exe_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            run_quantize("src.gguf", "dst.gguf", "Q4_K_M", exe=tmp_path / "nope.exe")

    def test_raises_file_not_found_when_exe_none_and_default_missing(self):
        with patch("quantize.DEFAULT_EXE", Path("/nonexistent/llama-quantize.exe")):
            with pytest.raises(FileNotFoundError):
                run_quantize("src.gguf", "dst.gguf", "Q4_K_M")

    def test_calls_subprocess_with_correct_args(self, tmp_path):
        fake_exe = tmp_path / "llama-quantize.exe"
        fake_exe.touch()
        proc = _mock_proc(["[  1/  5] layer\n"])

        with patch("quantize.subprocess.Popen", return_value=proc) as mock_popen:
            run_quantize("src.gguf", "dst.gguf", "Q4_K_M", exe=fake_exe)

        cmd = mock_popen.call_args[0][0]
        assert str(fake_exe) in cmd
        assert "src.gguf" in cmd
        assert "dst.gguf" in cmd
        assert "Q4_K_M" in cmd

    def test_appends_nthreads_when_given(self, tmp_path):
        fake_exe = tmp_path / "llama-quantize.exe"
        fake_exe.touch()
        proc = _mock_proc([])

        with patch("quantize.subprocess.Popen", return_value=proc) as mock_popen:
            run_quantize("src.gguf", "dst.gguf", "Q6_K", exe=fake_exe, nthreads=8)

        cmd = mock_popen.call_args[0][0]
        assert "8" in cmd

    def test_progress_callback_fired_on_matching_lines(self, tmp_path):
        fake_exe = tmp_path / "llama-quantize.exe"
        fake_exe.touch()
        lines = [
            "[  1/ 10] blk.0.attn.weight\n",
            "some info line\n",
            "[  5/ 10] blk.4.ffn.weight\n",
            "[ 10/ 10] output.weight\n",
        ]
        proc = _mock_proc(lines)
        calls: list[tuple[int, int]] = []

        with patch("quantize.subprocess.Popen", return_value=proc):
            run_quantize(
                "src.gguf", "dst.gguf", "Q4_K_M",
                exe=fake_exe,
                on_progress=lambda idx, total, desc: calls.append((idx, total)),
            )

        assert calls == [(1, 10), (5, 10), (10, 10)]

    def test_log_callback_receives_all_non_empty_lines(self, tmp_path):
        fake_exe = tmp_path / "llama-quantize.exe"
        fake_exe.touch()
        lines = ["line one\n", "\n", "line two\n"]
        proc = _mock_proc(lines)
        logged: list[str] = []

        with patch("quantize.subprocess.Popen", return_value=proc):
            run_quantize(
                "src.gguf", "dst.gguf", "Q4_K_M",
                exe=fake_exe,
                on_log=logged.append,
            )

        # First log entry is the "$ cmd" info line; remaining are from stdout
        stdout_lines = [l for l in logged if not l.startswith("INFO:")]
        assert "line one" in stdout_lines
        assert "line two" in stdout_lines
        assert "" not in stdout_lines  # empty lines are dropped

    def test_raises_runtime_error_on_nonzero_exit(self, tmp_path):
        fake_exe = tmp_path / "llama-quantize.exe"
        fake_exe.touch()
        proc = _mock_proc([], returncode=1)

        with patch("quantize.subprocess.Popen", return_value=proc):
            with pytest.raises(RuntimeError, match="code 1"):
                run_quantize("src.gguf", "dst.gguf", "Q4_K_M", exe=fake_exe)

    def test_uses_bufsize_1_for_line_buffering(self, tmp_path):
        fake_exe = tmp_path / "llama-quantize.exe"
        fake_exe.touch()
        proc = _mock_proc([])

        with patch("quantize.subprocess.Popen", return_value=proc) as mock_popen:
            run_quantize("src.gguf", "dst.gguf", "Q4_K_M", exe=fake_exe)

        kwargs = mock_popen.call_args[1]
        assert kwargs.get("bufsize") == 1, "bufsize=1 required to avoid buffering delay"


# ---------------------------------------------------------------------------
# estimate_output_size / _estimate_from_safetensors
# ---------------------------------------------------------------------------

def _make_safetensors(tmp_path, tensors: dict) -> str:
    """Write a minimal safetensors file with the given tensor metadata.

    tensors: { name: {"dtype": "F16", "shape": [...]} }
    """
    header = {}
    offset = 0
    for name, meta in tensors.items():
        import math
        n = math.prod(meta["shape"]) if meta["shape"] else 1
        dtype_bytes = {"F32": 4, "F16": 2, "BF16": 2, "I32": 4}.get(meta["dtype"], 2)
        nbytes = n * dtype_bytes
        header[name] = {
            "dtype": meta["dtype"],
            "shape": meta["shape"],
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    header_json = json.dumps(header).encode("utf-8")
    path = str(tmp_path / "model.safetensors")
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(header_json)))
        fh.write(header_json)
        fh.write(b"\x00" * offset)  # dummy tensor data
    return path


class TestEstimateOutputSize:
    def test_returns_none_for_missing_file(self):
        assert estimate_output_size("/nonexistent/model.safetensors", "Q4_K_M") is None

    def test_returns_none_for_empty_src(self):
        assert estimate_output_size("", "Q4_K_M") is None

    def test_1d_tensor_counted_as_f32(self, tmp_path):
        # BF16 source, 1D tensor → must be stored as F32 (4 bytes/elem)
        path = _make_safetensors(tmp_path, {
            "norm.weight": {"dtype": "BF16", "shape": [1024]},
        })
        est = estimate_output_size(path, "Q4_K_M")
        assert est == 1024 * 4  # F32

    def test_small_tensor_counted_as_f32(self, tmp_path):
        path = _make_safetensors(tmp_path, {
            "tiny.weight": {"dtype": "BF16", "shape": [32, 32]},  # 1024 elements = threshold
        })
        est = estimate_output_size(path, "Q4_K_M")
        assert est == 32 * 32 * 4  # F32 (≤ threshold)

    def test_f16_1d_not_forced_to_f32(self, tmp_path):
        # F16 source + 1D → NOT forced to F32 (only BF16/F32 sources trigger that rule)
        path = _make_safetensors(tmp_path, {
            "norm.weight": {"dtype": "F16", "shape": [1024]},
        })
        est = estimate_output_size(path, "Q4_K_M")
        bpe = _QUANT_BYTES_PER_ELEM["Q4_K_M"]
        assert est == int(1024 * bpe)

    def test_large_2d_bf16_tensor_uses_quant_bpe(self, tmp_path):
        path = _make_safetensors(tmp_path, {
            "attn.weight": {"dtype": "BF16", "shape": [2048, 2048]},  # 4M elements
        })
        n = 2048 * 2048
        est = estimate_output_size(path, "Q4_K_M")
        bpe = _QUANT_BYTES_PER_ELEM["Q4_K_M"]
        assert est == int(n * bpe)

    def test_mixed_model_separates_f32_and_quantizable(self, tmp_path):
        path = _make_safetensors(tmp_path, {
            "norm.weight":  {"dtype": "BF16", "shape": [512]},        # 1D → F32
            "attn.weight":  {"dtype": "BF16", "shape": [1024, 1024]}, # quantizable
        })
        n_f32 = 512
        n_quant = 1024 * 1024
        bpe = _QUANT_BYTES_PER_ELEM["Q3_K_M"]
        expected = n_f32 * 4 + int(n_quant * bpe)
        assert estimate_output_size(path, "Q3_K_M") == expected

    def test_f32_quant_key_makes_everything_f32(self, tmp_path):
        path = _make_safetensors(tmp_path, {
            "attn.weight": {"dtype": "BF16", "shape": [512, 512]},
        })
        est = estimate_output_size(path, "F32")
        assert est == 512 * 512 * 4  # all F32

    def test_non_safetensors_uses_ratio_fallback(self, tmp_path):
        # Write a dummy file of known size
        p = tmp_path / "model.ckpt"
        p.write_bytes(b"\x00" * 1_000_000)
        from quantize import SIZE_RATIOS
        ratio = SIZE_RATIOS["Q4_K_M"]
        est = estimate_output_size(str(p), "Q4_K_M")
        assert est == int(1_000_000 * ratio)

    def test_quant_bpe_all_keys_covered(self):
        for key in ALL_QUANT_CHOICES:
            _, k = key
            assert k in _QUANT_BYTES_PER_ELEM, f"Missing bpe entry for {k}"
