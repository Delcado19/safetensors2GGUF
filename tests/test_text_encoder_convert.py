"""Tests for text_encoder_convert.py — auto-cloning llama.cpp and running convert_hf_to_gguf.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from text_encoder_convert import (
    TEXT_ENCODER_OUTTYPES,
    ensure_llama_cpp,
    find_convert_script,
)


class TestOuttypes:
    def test_is_list_of_tuples(self):
        assert isinstance(TEXT_ENCODER_OUTTYPES, list)
        for label, value in TEXT_ENCODER_OUTTYPES:
            assert isinstance(label, str) and label
            assert isinstance(value, str) and value


class TestEnsureLlamaCpp:
    def test_returns_existing_dir_without_cloning(self, tmp_path):
        (tmp_path / "convert_hf_to_gguf.py").write_text("# stub")
        with patch("text_encoder_convert._llama_cpp_dir", return_value=tmp_path), \
             patch("text_encoder_convert.subprocess.run") as mock_run:
            found = ensure_llama_cpp()
            assert found == tmp_path
            mock_run.assert_not_called()

    def test_clones_when_missing(self, tmp_path):
        clone_dir = tmp_path / "llama.cpp"

        def _fake_clone(cmd, **kwargs):
            clone_dir.mkdir(parents=True)
            (clone_dir / "convert_hf_to_gguf.py").write_text("# stub")
            return subprocess.CompletedProcess(cmd, 0)

        with patch("text_encoder_convert._llama_cpp_dir", return_value=clone_dir), \
             patch("text_encoder_convert.subprocess.run", side_effect=_fake_clone) as mock_run:
            found = ensure_llama_cpp()
            assert found == clone_dir
            assert (clone_dir / "convert_hf_to_gguf.py").is_file()
            cmd = mock_run.call_args[0][0]
            assert cmd[:2] == ["git", "clone"]
            assert "--depth" in cmd
            assert str(clone_dir) in cmd
            assert "https://github.com/ggml-org/llama.cpp" in cmd

    def test_raises_runtime_error_when_git_not_found(self, tmp_path):
        with patch("text_encoder_convert._llama_cpp_dir", return_value=tmp_path / "llama.cpp"), \
             patch("text_encoder_convert.subprocess.run", side_effect=FileNotFoundError()):
            try:
                ensure_llama_cpp()
                assert False, "expected RuntimeError"
            except RuntimeError as exc:
                assert "git" in str(exc).lower()

    def test_raises_runtime_error_when_clone_fails(self, tmp_path):
        err = subprocess.CalledProcessError(1, ["git", "clone"], stderr="fatal: network error")
        with patch("text_encoder_convert._llama_cpp_dir", return_value=tmp_path / "llama.cpp"), \
             patch("text_encoder_convert.subprocess.run", side_effect=err):
            try:
                ensure_llama_cpp()
                assert False, "expected RuntimeError"
            except RuntimeError as exc:
                assert "network error" in str(exc)


class TestFindConvertScript:
    def test_returns_script_path_under_llama_cpp_dir(self, tmp_path):
        with patch("text_encoder_convert.ensure_llama_cpp", return_value=tmp_path):
            assert find_convert_script() == tmp_path / "convert_hf_to_gguf.py"


class TestFetchBaseConfigFiles:
    def test_downloads_config_and_available_tokenizer_files(self, tmp_path):
        from text_encoder_convert import fetch_base_config_files

        calls = []

        def _fake_download(repo_id, filename, local_dir):
            calls.append(filename)
            if filename not in ("config.json", "tokenizer.json"):
                raise Exception("not found")  # simulate missing optional file
            out = Path(local_dir) / filename
            out.write_text("{}")
            return str(out)

        with patch("text_encoder_convert.hf_hub_download", side_effect=_fake_download):
            downloaded = fetch_base_config_files("Qwen/Qwen3-8B", tmp_path)

        assert "config.json" in downloaded
        assert "tokenizer.json" in downloaded
        assert (tmp_path / "config.json").is_file()

    def test_raises_if_config_json_missing(self, tmp_path):
        from text_encoder_convert import fetch_base_config_files

        def _always_fail(repo_id, filename, local_dir):
            raise Exception("404")

        with patch("text_encoder_convert.hf_hub_download", side_effect=_always_fail):
            try:
                fetch_base_config_files("bad/repo", tmp_path)
                assert False, "expected RuntimeError"
            except RuntimeError:
                pass


class TestConvertTextEncoder:
    def test_propagates_error_when_llama_cpp_unavailable(self, tmp_path):
        from text_encoder_convert import convert_text_encoder

        weights = tmp_path / "model.safetensors"
        weights.write_bytes(b"stub")
        with patch("text_encoder_convert.find_convert_script", side_effect=RuntimeError("git not found")):
            try:
                convert_text_encoder(str(weights), "Qwen/Qwen3-8B")
                assert False, "expected RuntimeError"
            except RuntimeError:
                pass

    def test_runs_subprocess_with_expected_args(self, tmp_path):
        from text_encoder_convert import convert_text_encoder

        weights = tmp_path / "model.safetensors"
        weights.write_bytes(b"stub")
        script = tmp_path / "convert_hf_to_gguf.py"
        script.write_text("# stub")

        with patch("text_encoder_convert.find_convert_script", return_value=script), \
             patch("text_encoder_convert.fetch_base_config_files", return_value=["config.json"]), \
             patch("text_encoder_convert.subprocess.Popen") as mock_popen:
            mock_proc = mock_popen.return_value
            mock_proc.stdout = iter(["INFO: done\n"])
            mock_proc.wait.return_value = 0
            mock_proc.returncode = 0

            out = convert_text_encoder(
                str(weights), "Qwen/Qwen3-8B", dst_path=str(tmp_path / "out.gguf"),
                outtype="f16",
            )

        assert out == str(tmp_path / "out.gguf")
        called_cmd = mock_popen.call_args[0][0]
        assert sys.executable == called_cmd[0]
        assert str(script) == called_cmd[1]
        assert "--outtype" in called_cmd
        assert "f16" in called_cmd
