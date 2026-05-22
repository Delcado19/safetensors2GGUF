"""Tests for GUI helper functions that do not require launching Gradio."""

from __future__ import annotations

from unittest.mock import patch

import gradio as gr

import gui


class TestLlamaQuantizePicker:
    def test_browse_llama_quantize_uses_selected_file(self, tmp_path):
        selected = tmp_path / "llama-quantize.exe"
        selected.touch()

        with patch("gui._browse", return_value=str(selected)):
            path, info = gui.browse_llama_quantize("")

        assert path == str(selected)
        assert str(selected) in info
        assert "Using llama-quantize" in info

    def test_browse_llama_quantize_keeps_current_path_on_cancel(self, tmp_path):
        current = tmp_path / "existing" / "llama-quantize.exe"
        current.parent.mkdir()
        current.touch()

        with patch("gui._browse", return_value=""):
            path, info = gui.browse_llama_quantize(str(current))

        assert path == str(current)
        assert str(current) in info

    def test_detect_llama_quantize_path_returns_path_and_guidance(self, tmp_path):
        exe = tmp_path / "llama-quantize.exe"
        exe.touch()

        with patch("gui.find_exe", return_value=exe):
            path, info = gui.detect_llama_quantize_path()

        assert path == str(exe)
        assert "Using llama-quantize" in info
        assert str(exe) in info

    def test_llama_quantize_info_reports_missing_path(self):
        info = gui.llama_quantize_info("")

        assert "No patched llama-quantize binary selected." in info
        assert "Generic upstream llama.cpp release binaries are not selected automatically." in info or (
            "ComfyUI Easy-Install bundled llama-quantize.exe" in info
        )

    def test_llama_quantize_info_mentions_city96_on_non_windows(self, tmp_path):
        exe = tmp_path / "llama-quantize"
        exe.touch()

        with patch("gui.os.name", "posix"):
            info = gui.llama_quantize_info("")

        assert "city96/ComfyUI-GGUF + lcpp.patch" in info
        assert "Generic upstream llama.cpp release binaries are not selected automatically." in info


def _markdown_values(app: gr.Blocks) -> list[str]:
    """Collect the rendered text of every gr.Markdown component in the app."""
    return [
        block.value
        for block in app.blocks.values()
        if isinstance(block, gr.Markdown) and isinstance(getattr(block, "value", None), str)
    ]


class TestLayoutParity:
    """Each tab should start with a description Markdown so Gradio computes
    a consistent tab width (Gradio 6 sizes tabs by their first child)."""

    def test_build_app_runs_without_error(self):
        app = gui.build_app()
        assert isinstance(app, gr.Blocks)

    def test_convert_tab_has_description_header(self):
        app = gui.build_app()
        markdown_values = _markdown_values(app)
        assert any(
            "Safetensors / CKPT" in v and "GGUF" in v
            for v in markdown_values
        ), "Convert tab is missing its description Markdown — tab width parity depends on it."

    def test_all_tabs_have_a_description_header(self):
        app = gui.build_app()
        markdown_values = _markdown_values(app)
        # Convert tab + Fix Pad Tokens + Fix 5D Tensors + Extract Components descriptions
        assert any("Safetensors / CKPT" in v for v in markdown_values)
        assert any("x_pad_token" in v for v in markdown_values)
        assert any("Re-insert 5D tensors" in v for v in markdown_values)
        assert any("Analyze an **SDXL** checkpoint" in v for v in markdown_values)
