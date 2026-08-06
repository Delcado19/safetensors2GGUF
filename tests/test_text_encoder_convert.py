"""Tests for text_encoder_convert.py — locating convert_hf_to_gguf.py and embedded Python."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from text_encoder_convert import (
    TEXT_ENCODER_OUTTYPES,
    find_convert_script,
    find_embedded_python,
)


class TestOuttypes:
    def test_is_list_of_tuples(self):
        assert isinstance(TEXT_ENCODER_OUTTYPES, list)
        for label, value in TEXT_ENCODER_OUTTYPES:
            assert isinstance(label, str) and label
            assert isinstance(value, str) and value


class TestFindConvertScript:
    def test_returns_none_when_not_found(self, tmp_path, monkeypatch):
        monkeypatch.setenv("COMFYUI_EASY_INSTALL_HOME", str(tmp_path / "nonexistent"))
        with patch("text_encoder_convert._easy_install_roots", return_value=[tmp_path]):
            assert find_convert_script() is None

    def test_finds_script_under_easy_install_root(self, tmp_path):
        script_dir = tmp_path / "python_embeded" / "Lib" / "site-packages" / "llama_cpp" / "bin"
        script_dir.mkdir(parents=True)
        script = script_dir / "convert_hf_to_gguf.py"
        script.write_text("# stub")
        with patch("text_encoder_convert._easy_install_roots", return_value=[tmp_path]):
            found = find_convert_script()
            assert found == script


class TestFindEmbeddedPython:
    def test_finds_python_exe_under_easy_install_root(self, tmp_path):
        py_dir = tmp_path / "python_embeded"
        py_dir.mkdir(parents=True)
        py_exe = py_dir / "python.exe"
        py_exe.write_text("stub")
        with patch("text_encoder_convert._easy_install_roots", return_value=[tmp_path]):
            found = find_embedded_python()
            assert found == py_exe


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
