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


class TestRunTeConvertCancel:
    """Regression for review finding #4: cancelling the Text Encoder tab must
    render 'Cancelled', not 'Error: cancelled', matching run_st_convert's
    existing special-case handling of RuntimeError("cancelled")."""

    def test_cancelled_run_reports_cancelled_not_error(self, tmp_path):
        src = tmp_path / "model.safetensors"
        src.touch()

        with patch("gui.convert_text_encoder_any", side_effect=RuntimeError("cancelled")):
            *_, (log, status) = gui.run_te_convert(str(src), "org/repo", "", "F16")

        assert status == "Cancelled"
        assert "Error" not in status

    def test_other_runtime_error_still_reports_error(self, tmp_path):
        src = tmp_path / "model.safetensors"
        src.touch()

        with patch("gui.convert_text_encoder_any", side_effect=RuntimeError("boom")):
            *_, (log, status) = gui.run_te_convert(str(src), "org/repo", "", "F16")

        assert status == "Error"
        assert "boom" in log


class TestDynamicDropdownAnnotation:
    def test_annotate_safetensors_choices_marks_caution_entries(self, tmp_path):
        import torch
        from safetensors.torch import save_file

        src = tmp_path / "model.safetensors"
        save_file(
            {
                "double_blocks.0.img_attn.proj.weight": torch.randn(8, 8),
                "img_in.weight": torch.randn(8, 8),
            },
            str(src),
        )
        update = gui.annotate_safetensors_choices(str(src))
        labels_by_key = {key: label for label, key in update["choices"]}
        assert labels_by_key["INT8"].startswith("⚠")
        assert labels_by_key["INT8_MIXED"].startswith("⚠")
        assert not labels_by_key["F16"].startswith("⚠")
        # FP8/FP8_MIXED are CAUTION unconditionally (never render-verified by
        # this tool), independent of this architecture's keys_hiprec.
        assert labels_by_key["FP8"].startswith("⚠")
        assert labels_by_key["FP8_MIXED"].startswith("⚠")

    def test_annotate_safetensors_choices_marks_bad_entries_with_cross(self, tmp_path):
        import torch
        from safetensors.torch import save_file

        # lumina2's own keys_detect signature (models/architectures.py).
        src = tmp_path / "lumina2.safetensors"
        save_file(
            {
                "cap_embedder.1.weight": torch.randn(8, 8),
                "context_refiner.0.attention.qkv.weight": torch.randn(8, 8),
            },
            str(src),
        )
        update = gui.annotate_safetensors_choices(str(src))
        labels_by_key = {key: label for label, key in update["choices"]}
        # Plain INT8 is a confirmed-bad combination on lumina2
        # (docs/issues_analysis.md #15) -- ✗, not the merely-unverified ⚠.
        assert labels_by_key["INT8"].startswith("✗")
        # INT8_MIXED is lumina2's one actually render-verified combination.
        assert not labels_by_key["INT8_MIXED"].startswith(("⚠", "✗"))

    def test_annotate_safetensors_choices_no_source_returns_unmodified(self):
        update = gui.annotate_safetensors_choices("")
        from safetensors_quant import SAFETENSORS_DTYPE_CHOICES
        assert update["choices"] == [tuple(c) for c in SAFETENSORS_DTYPE_CHOICES]

    def test_annotate_gguf_choices_no_source_returns_unmodified(self):
        update = gui.annotate_gguf_choices("")
        from quantize import ALL_QUANT_CHOICES
        assert update["choices"] == [tuple(c) for c in ALL_QUANT_CHOICES]


def test_convert_safetensors_tab_present():
    """The renamed 'Convert -> GGUF' tab and new 'Convert -> Safetensors' tab
    (Task 5) must both exist, and the new tab's format dropdown must offer
    the SAFETENSORS_DTYPE_CHOICES keys (Task 1)."""
    app = gui.build_app()
    label_texts = [
        getattr(block, "label", None)
        for block in app.blocks.values()
    ]
    assert "Output format" in label_texts

    dropdown = next(
        block for block in app.blocks.values()
        if isinstance(block, gr.Dropdown) and getattr(block, "label", None) == "Output format"
    )
    from safetensors_quant import SAFETENSORS_DTYPE_CHOICES
    assert dropdown.choices == [tuple(c) for c in SAFETENSORS_DTYPE_CHOICES]


def test_text_encoder_tab_present():
    """The 'Convert Text Encoder' tab (Task 10) must exist."""
    from gui import build_app
    app = build_app()
    label_texts = [getattr(b, "label", None) for b in app.blocks.values()]
    assert any("Base model HF repo ID" == t for t in label_texts)


def test_model_support_tab_present():
    app = gui.build_app()
    label_texts = [
        getattr(b, "label", None)
        for b in app.blocks.values()
        if isinstance(b, gr.Dataframe)
    ]
    assert "Model Support" in label_texts


def test_support_table_rows_cover_every_architecture():
    from models.architectures import arch_list
    rows = gui._support_table_rows_for_dataframe()
    assert len(rows) == len(arch_list)


def test_support_table_cell_html_uses_the_right_symbol():
    from model_support import SUPPORT_CAUTION, SUPPORT_VERIFIED
    assert "✓" in gui._support_table_cell_html(SUPPORT_VERIFIED)
    assert "⚠" in gui._support_table_cell_html(SUPPORT_CAUTION)


def test_apply_support_table_selection_gguf_column():
    from unittest.mock import MagicMock
    evt = MagicMock()
    evt.index = (0, 1)  # row 0, first format column after Model = "GGUF"
    tabs_update, quant_update, st_format_update = gui.apply_support_table_selection(evt)
    assert tabs_update.get("selected") == 0
    assert quant_update.get("value") == "Q4_K_M"


def test_apply_support_table_selection_safetensors_column():
    from unittest.mock import MagicMock
    from model_support import TABLE_FORMATS

    # Find the column index for "INT8_MIXED" (Model column is index 0, so
    # +1 for TABLE_FORMATS' own 0-based position).
    col = 1 + next(i for i, (_, key) in enumerate(TABLE_FORMATS) if key == "INT8_MIXED")
    evt = MagicMock()
    evt.index = (0, col)
    tabs_update, quant_update, st_format_update = gui.apply_support_table_selection(evt)
    assert tabs_update.get("selected") == 1
    assert st_format_update.get("value") == "INT8_MIXED"


def test_apply_support_table_selection_model_column_is_a_noop():
    from unittest.mock import MagicMock
    evt = MagicMock()
    evt.index = (0, 0)  # the Model name column itself
    tabs_update, quant_update, st_format_update = gui.apply_support_table_selection(evt)
    assert "selected" not in tabs_update


def test_apply_text_encoder_support_table_selection_gguf_column():
    from unittest.mock import MagicMock
    evt = MagicMock()
    evt.index = (0, 1)  # row 0, first format column after the label = "GGUF"
    tabs_update, format_update = gui.apply_text_encoder_support_table_selection(evt)
    assert tabs_update.get("selected") == 2
    assert format_update.get("value") == "Q4_K_M"


def test_apply_text_encoder_support_table_selection_nvfp4_column():
    from unittest.mock import MagicMock
    from model_support import TEXT_ENCODER_TABLE_FORMATS

    # Unlike the diffusion-model table, NVFP4 IS a real TEXT_ENCODER_FORMAT_CHOICES
    # entry, so it must select normally rather than no-op.
    col = 1 + next(i for i, (_, key) in enumerate(TEXT_ENCODER_TABLE_FORMATS) if key == "NVFP4")
    evt = MagicMock()
    evt.index = (0, col)
    tabs_update, format_update = gui.apply_text_encoder_support_table_selection(evt)
    assert tabs_update.get("selected") == 2
    assert format_update.get("value") == "NVFP4"


def test_apply_text_encoder_support_table_selection_label_column_is_a_noop():
    from unittest.mock import MagicMock
    evt = MagicMock()
    evt.index = (0, 0)
    tabs_update, format_update = gui.apply_text_encoder_support_table_selection(evt)
    assert "selected" not in tabs_update
