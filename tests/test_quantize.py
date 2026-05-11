"""Tests for quantize.py — type registry and llama-quantize subprocess wrapper."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from quantize import (
    ALL_QUANT_CHOICES,
    LLAMA_QUANT_KEYS,
    PYTHON_PRECISIONS,
    SIZE_RATIOS,
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
