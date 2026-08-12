"""Web UI for safetensors2GGUF — Gradio frontend for GGUF convert, safetensors convert, text-encoder convert, quantize, and fix_5d."""

from __future__ import annotations

import base64
import os
import queue
import threading
from pathlib import Path
from typing import Generator

# Disable Gradio telemetry/analytics before the library is imported so that no
# outbound HTTP requests are made at startup (HuggingFace telemetry endpoint,
# pkg-version check, Google Fonts CDN are all suppressed).
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr

from component_extract import (
    analyze_components,
    default_output_root,
    extract_components,
    format_component_analysis,
)
from convert import ConversionCancelled, convert_file, load_state_dict
from convert_safetensors import convert_to_safetensors
from fix_5d_tensors import fix_5d_tensors as _fix_5d
from fix_pad_tokens import fix_pad_tokens as _fix_pad
from models.architectures import detect_arch
from model_support import (
    SUPPORT_BAD,
    SUPPORT_CAUTION,
    SUPPORT_SYMBOL,
    TABLE_FORMATS,
    TEXT_ENCODER_TABLE_FORMATS,
    build_support_table,
    build_text_encoder_support_table,
    support_level,
)
from quantize import (
    ALL_QUANT_CHOICES,
    DEFAULT_EXE,
    EASY_INSTALL_ROOT_ENV,
    LLAMA_QUANT_KEYS,
    PYTHON_PRECISIONS,
    SIZE_RATIOS,
    estimate_output_size,
    find_exe,
    run_quantize,
)
from safetensors_quant import SAFETENSORS_DTYPE_CHOICES, estimate_safetensors_output_size, format_recommendation
from text_encoder_convert import (
    TEXT_ENCODER_FORMAT_CHOICES,
    TEXT_ENCODER_SAFETENSORS_FORMATS,
    _TEXT_ENCODER_MODEL_ARCH,
    convert_text_encoder_any,
)

# ──────────────────────────────────────────────────────────────────────────────
# Cancel support
# ──────────────────────────────────────────────────────────────────────────────

_active_cancel: threading.Event | None = None
# Separate cancel slot for the Convert -> Safetensors tab so a running GGUF
# conversion and a running safetensors conversion don't share (and stomp on)
# the same cancel_event.
_active_cancel_st: threading.Event | None = None
# Separate cancel slot for the Convert Text Encoder tab, distinct from
# both _active_cancel and _active_cancel_st, so all three conversion tabs can
# run (and be cancelled) independently without stomping on each other.
_active_cancel_te: threading.Event | None = None
GUI_TENSOR_LOG_EVERY = 25


def request_cancel() -> str:
    """Signal the active conversion to stop; called by the Cancel button."""
    if _active_cancel is not None:
        _active_cancel.set()
        return "Cancelling…"
    return "No active conversion"


def request_cancel_st() -> str:
    """Signal the active Convert -> Safetensors conversion to stop."""
    if _active_cancel_st is not None:
        _active_cancel_st.set()
        return "Cancelling…"
    return "No active conversion"


def request_cancel_te() -> str:
    """Signal the active Convert Text Encoder conversion to stop."""
    if _active_cancel_te is not None:
        _active_cancel_te.set()
        return "Cancelling…"
    return "No active conversion"


# ──────────────────────────────────────────────────────────────────────────────
# Native file picker (tkinter — no file upload, no size limit)
# ──────────────────────────────────────────────────────────────────────────────

def _browse(filetypes: list[tuple[str, str]]) -> str:
    """Open a native Windows file-open dialog; return the selected path or ''."""
    result: list[str] = [""]
    done = threading.Event()

    def _run() -> None:
        try:
            import tkinter
            import tkinter.filedialog
            root = tkinter.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            result[0] = tkinter.filedialog.askopenfilename(filetypes=filetypes) or ""
            root.destroy()
        except Exception:
            pass
        finally:
            done.set()

    threading.Thread(target=_run, daemon=True).start()
    done.wait(timeout=120)
    return result[0]


_MODEL_TYPES = [
    ("Model files", "*.safetensors *.ckpt *.pt *.bin *.pth"),
    ("All files", "*.*"),
]
_GGUF_TYPES = [
    ("GGUF files", "*.gguf"),
    ("All files", "*.*"),
]
_LLAMA_QUANTIZE_TYPES = [
    ("llama-quantize", "llama-quantize.exe llama-quantize"),
    ("All files", "*.*"),
]


def browse_model() -> str:
    """Open a native file dialog filtered to model formats; return the selected path."""
    return _browse(_MODEL_TYPES)


def browse_gguf() -> str:
    """Open a native file dialog filtered to GGUF files; return the selected path."""
    return _browse(_GGUF_TYPES)


def browse_models_root() -> str:
    """Open a native directory dialog for the ComfyUI models root."""
    return _pick_dir()


def llama_quantize_info(path: str | Path | None) -> str:
    """Return setup guidance for the selected or auto-detected llama-quantize."""
    path_str = str(path or "").strip()
    lines: list[str] = []
    if path_str and Path(path_str).is_file():
        lines.append(f"Using llama-quantize:\n{path_str}")
    else:
        lines.append("No patched llama-quantize binary selected.")

    if os.name == "nt":
        lines.extend([
            "",
            "Windows: the preferred binary is the ComfyUI Easy-Install bundled llama-quantize.exe.",
            f"Auto-detected default: {DEFAULT_EXE}",
            f"Optional Easy-Install root override: {EASY_INSTALL_ROOT_ENV}",
            "If it is not found, use Browse and select llama-quantize.exe.",
        ])
    else:
        lines.extend([
            "",
            "macOS/Linux: build llama-quantize from city96/ComfyUI-GGUF + lcpp.patch, then select the built binary with Browse.",
            "Generic upstream llama.cpp release binaries are not selected automatically.",
        ])
    return "\n".join(lines)


def detect_llama_quantize_path() -> tuple[str, str]:
    """Re-run conservative llama-quantize detection for the GUI."""
    exe = find_exe()
    path = str(exe or "")
    return path, llama_quantize_info(path)


def browse_llama_quantize(current: str) -> tuple[str, str]:
    """Select llama-quantize via file picker; keep current value on cancel."""
    selected = _browse(_LLAMA_QUANTIZE_TYPES)
    path = selected or (current or "")
    return path, llama_quantize_info(path)


def _pick_dir() -> str:
    """Open a native directory picker; return the selected path or ''."""
    result: list[str] = [""]
    done = threading.Event()

    def _run() -> None:
        try:
            import tkinter
            import tkinter.filedialog
            root = tkinter.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            result[0] = tkinter.filedialog.askdirectory(title="Select output directory") or ""
            root.destroy()
        except Exception:
            pass
        finally:
            done.set()

    threading.Thread(target=_run, daemon=True).start()
    done.wait(timeout=120)
    return result[0]


def browse_and_set_dst(src: str, quant_key: str) -> str:
    """Open directory picker and compose a full output file path.

    Combines the chosen directory with an auto-generated filename derived from
    the source model path and selected quantization key.  If the user has not
    yet chosen a source file the raw directory path is returned instead.
    """
    dir_path = _pick_dir()
    if not dir_path:
        return ""  # user cancelled — leave dst_path unchanged
    src = (src or "").strip()
    if src:
        stem = _strip_model_suffix(src)
        # Use {ftype} so the actual quant key is inserted at conversion time,
        # not at browse time — prevents stale names when the user changes quant after browsing.
        return str(Path(dir_path) / f"{stem.name}-{{ftype}}.gguf")
    # No source selected yet — just show the directory with a trailing separator
    return dir_path + "\\"


def browse_and_set_dst_st(src: str) -> str:
    """Open directory picker and compose a .safetensors output path template.

    Mirrors browse_and_set_dst but for the Convert -> Safetensors tab: that
    tab's browse button previously opened a model-file *open* dialog
    (_MODEL_TYPES via askopenfilename), which is wrong for choosing an output
    location. Uses the same {ftype} placeholder trick as the GGUF tab so the
    filename reflects whatever output format is selected at convert time, not
    the format that happened to be selected when Browse was clicked.
    """
    dir_path = _pick_dir()
    if not dir_path:
        return ""  # user cancelled — leave dst_path unchanged
    src = (src or "").strip()
    if src:
        stem = _strip_model_suffix(src)
        return str(Path(dir_path) / f"{stem.name}-{{ftype}}.safetensors")
    return dir_path + "\\"


# ──────────────────────────────────────────────────────────────────────────────
# Size estimate
# ──────────────────────────────────────────────────────────────────────────────

def _fmt_size(bytes_: float) -> str:
    gb = bytes_ / 1e9
    if gb >= 10:
        return f"{gb:.0f} GB"
    if gb >= 1:
        return f"{gb:.1f} GB"
    return f"{bytes_ / 1e6:.0f} MB"


def update_size_estimate(src: str, quant_key: str) -> str:
    """Return a Markdown string with the estimated output size.

    For .safetensors sources the safetensors header is analysed (no tensor
    data loaded) so that F32-forced tensors and quantizable tensors are
    counted separately — gives accurate results for diffusion models where
    a large fraction of parameters are 1D/small (norms, biases, embeddings).
    For other formats a source_size × SIZE_RATIOS fallback is used.
    """
    src = (src or "").strip()
    est = estimate_output_size(src, quant_key)
    if est is None:
        return ""
    src_bytes = os.path.getsize(src) if os.path.isfile(src) else 0
    line = f"Estimated output: **{_fmt_size(est)}**"
    if src_bytes > 0 and est < src_bytes:
        pct = (1 - est / src_bytes) * 100
        line += f" &nbsp;·&nbsp; {pct:.0f}% smaller than source ({_fmt_size(src_bytes)})"
    elif src_bytes > 0 and est > src_bytes:
        line += f" &nbsp;·&nbsp; source: {_fmt_size(src_bytes)}"
    return line


def _safetensors_size_estimate_line(model_arch, src: str, target_key: str) -> str:
    """Return a Markdown string with the estimated safetensors-quant output
    size, or "" if the architecture couldn't be detected or the source can't
    be read as a safetensors header — same fail-quiet convention as
    _format_recommendation_update(). Needs model_arch (not just src) because
    the *_MIXED formats' size depends on keys_hiprec, unlike GGUF's
    update_size_estimate() above."""
    if model_arch is None:
        return ""
    src = (src or "").strip()
    if not src or not os.path.isfile(src):
        return ""
    est = estimate_safetensors_output_size(src, target_key, model_arch)
    if est is None:
        return ""
    src_bytes = os.path.getsize(src)
    line = f"Estimated output: **{_fmt_size(est)}**"
    if src_bytes > 0 and est < src_bytes:
        pct = (1 - est / src_bytes) * 100
        line += f" &nbsp;·&nbsp; {pct:.0f}% smaller than source ({_fmt_size(src_bytes)})"
    elif src_bytes > 0 and est > src_bytes:
        line += f" &nbsp;·&nbsp; source: {_fmt_size(src_bytes)}"
    return line


def update_safetensors_size_estimate(src: str, target_key: str) -> str:
    """Safetensors-tab equivalent of update_size_estimate() — wired to
    st_format_dropdown's own .change() (target_key changed, src unchanged);
    st_src_path's .change() uses the combined
    update_format_recommendation_choices_and_size() below instead."""
    return _safetensors_size_estimate_line(_detected_arch_or_none(src), src, target_key)


def update_text_encoder_size_estimate(src: str, format_key: str) -> str:
    """Text-encoder-tab equivalent of update_size_estimate()/
    update_safetensors_size_estimate().

    FP8/FP8_MIXED/NVFP4/NVFP4_MIXED go through estimate_safetensors_output_size()
    with this tool's fixed text-encoder ModelTemplate (_TEXT_ENCODER_MODEL_ARCH)
    — there's no per-checkpoint architecture detection for text encoders the
    way there is for diffusion models, just one fixed keys_shape_critical set.

    GGUF-family keys (F32/F16/BF16/Q8_0/K-quants) deliberately use the plain
    SIZE_RATIOS ratio, NOT estimate_output_size()'s per-tensor header analysis:
    that analysis's F32-forcing heuristic was tuned for THIS project's own
    convert.py GGUF writer (used for diffusion models), not llama.cpp's own
    convert_hf_to_gguf.py + llama-quantize (used for text encoders) — applying
    it here would look precise while silently assuming the wrong converter's
    behavior. SIZE_RATIOS's own reference data (Llama-3-8B via llama-quantize)
    is actually the better-matched source for an LLM-style text encoder.
    """
    src = (src or "").strip()
    if not src or not os.path.isfile(src):
        return ""
    if format_key in TEXT_ENCODER_SAFETENSORS_FORMATS:
        return _safetensors_size_estimate_line(_TEXT_ENCODER_MODEL_ARCH, src, format_key)
    ratio = SIZE_RATIOS.get(format_key)
    if ratio is None:
        return ""
    src_bytes = os.path.getsize(src)
    est = int(src_bytes * ratio)
    line = f"Estimated output: **{_fmt_size(est)}**"
    if est < src_bytes:
        pct = (1 - ratio) * 100
        line += f" &nbsp;·&nbsp; {pct:.0f}% smaller than source ({_fmt_size(src_bytes)})"
    elif est > src_bytes:
        line += f" &nbsp;·&nbsp; source: {_fmt_size(src_bytes)}"
    return line


def _detected_arch_or_none(src: str):
    """Best-effort architecture detection shared by the format-hint and
    dropdown-annotation helpers below. Reads only the safetensors header (via
    the same lazy load_state_dict() the conversion itself uses) — no tensor
    data loaded. Returns None on any failure (missing file, unrecognized
    architecture) so callers can fall back to unmodified/empty output."""
    src = (src or "").strip()
    if not src or not os.path.isfile(src):
        return None
    try:
        return detect_arch(load_state_dict(src))
    except Exception:
        return None


def _format_recommendation_update(model_arch, target_key: str):
    """Build the format-recommendation Markdown gr.update() from an
    already-detected architecture (or None)."""
    if model_arch is None:
        return gr.update(value="", elem_classes=["fmt-hint"])
    try:
        level, message = format_recommendation(model_arch, target_key)
    except Exception:
        return gr.update(value="", elem_classes=["fmt-hint"])
    if not message:
        return gr.update(value="", elem_classes=["fmt-hint"])
    icon = "✓" if level == "ok" else "⚠"
    return gr.update(value=f"{icon} {message}", elem_classes=["fmt-hint", f"fmt-hint-{level}"])


def update_format_recommendation(src: str, target_key: str):
    """Return a gr.update() for the format-recommendation Markdown, advising
    whether ``target_key`` suits the detected architecture of ``src`` — the
    Convert -> Safetensors equivalent of the GGUF tab's static
    "recommended ★" label, but per-architecture (see
    safetensors_quant.format_recommendation). Wired to st_format_dropdown's
    own .change() (target_key changed, src unchanged) — st_src_path's
    .change() uses the combined update_format_recommendation_and_choices()
    below instead, to avoid detecting the architecture twice on every
    source-path edit."""
    return _format_recommendation_update(_detected_arch_or_none(src), target_key)


def _annotate_choices_for_arch(model_arch, choices, column_key: str | None = None):
    """Return a gr.update(choices=...) prefixing a ⚠ (SUPPORT_CAUTION,
    unverified) or ✗ (SUPPORT_BAD, actually confirmed wrong) to any matching
    choice for ``model_arch`` (or unmodified choices if None).

    ``column_key``, when given, is the fixed TABLE_FORMATS format key every
    choice maps to (the GGUF quant dropdown: every K-quant level collapses
    onto the table's single "GGUF" column). When omitted, each choice's own
    key is used directly (the safetensors dtype dropdown, one key per
    TABLE_FORMATS column)."""
    if model_arch is None:
        return gr.update(choices=[tuple(c) for c in choices])

    sensitive = bool(model_arch.keys_hiprec)
    out = []
    for label, key in choices:
        fmt_key = column_key if column_key is not None else key
        level = support_level(model_arch.arch, sensitive, fmt_key)
        if level == SUPPORT_BAD:
            label = f"✗ {label}"
        elif level == SUPPORT_CAUTION:
            label = f"⚠ {label}"
        out.append((label, key))
    return gr.update(choices=out)


def annotate_safetensors_choices(src: str):
    """Dropdown-annotation for the Convert -> Safetensors format dropdown."""
    return _annotate_choices_for_arch(_detected_arch_or_none(src), SAFETENSORS_DTYPE_CHOICES)


def annotate_gguf_choices(src: str):
    """Dropdown-annotation for the Convert -> GGUF quant dropdown. Every
    ALL_QUANT_CHOICES key maps to the table's single "GGUF" column, and
    model_support.support_level() always returns SUPPORT_VERIFIED for it
    unconditionally (no per-architecture GGUF branch exists) — so this is
    guaranteed to never mark anything today. Skips architecture detection
    entirely rather than paying a full checkpoint load (torch.load() for
    .ckpt/.pt/.bin/.pth sources) for a result that can't change; revisit if
    support_level() ever grows an architecture-specific GGUF case."""
    return gr.update(choices=[tuple(c) for c in ALL_QUANT_CHOICES])


def update_format_recommendation_choices_and_size(src: str, target_key: str):
    """Combined st_src_path.change() handler: detects the architecture once
    and drives the format-recommendation hint, the format-dropdown ⚠
    annotations, and the size estimate — instead of separate handlers each
    independently re-detecting (each a full torch.load() for non-safetensors
    sources)."""
    model_arch = _detected_arch_or_none(src)
    return (
        _format_recommendation_update(model_arch, target_key),
        _annotate_choices_for_arch(model_arch, SAFETENSORS_DTYPE_CHOICES),
        _safetensors_size_estimate_line(model_arch, src, target_key),
    )


_SUPPORT_CELL_COLOR = {
    "verified": "var(--s2g-support-good)",
    "caution": "var(--s2g-support-caution)",
    "bad": "var(--s2g-support-bad)",
    "unknown": "var(--s2g-muted)",
}


def _support_table_cell_html(level: str) -> str:
    """Return the colored-symbol HTML for one Model Support table cell."""
    symbol = SUPPORT_SYMBOL[level]
    color = _SUPPORT_CELL_COLOR[level]
    return f'<span style="color:{color};font-weight:700;font-size:1.1em;">{symbol}</span>'


def _support_table_rows_for_dataframe() -> list[list[str]]:
    """Build the gr.Dataframe row data: [display_name_html, *format_cells]."""
    rows = []
    for row in build_support_table():
        display_html = (
            f'<span>{row["display_name"].split(" (")[0]}</span> '
            f'<span style="color:var(--s2g-muted);font-size:0.85em;">'
            f'({row["arch"]})</span>'
        )
        cells = [display_html]
        for _, format_key in TABLE_FORMATS:
            cells.append(_support_table_cell_html(row[format_key]))
        rows.append(cells)
    return rows


def _text_encoder_support_rows_for_dataframe() -> list[list[str]]:
    """One gr.Dataframe row per vendored text-encoder family (see
    model_support.build_text_encoder_support_table()), mirroring
    _support_table_rows_for_dataframe() below for the diffusion-model table.

    Unlike diffusion-model architectures, these families aren't structurally
    related to each other -- Qwen3/Mistral (decoder-only LLMs), T5 (encoder-
    decoder), and CLIP (a contrastive vision-language model) are unrelated
    architectures that happen to each be usable as a diffusion model's text
    encoder. The conversion code itself stays generic across all of them
    (text_encoder_convert.py has no per-family branching); only the *support
    evidence* per (family, format) pair differs, which is what this table
    now tracks row-by-row instead of collapsing into one generic row.
    """
    rows = []
    for row in build_text_encoder_support_table():
        display_html = (
            f'<span>{row["display_name"].split(" (")[0]}</span> '
            f'<span style="color:var(--s2g-muted);font-size:0.85em;">'
            f'({row["family"]})</span>'
        )
        cells = [display_html]
        for _, format_key in TEXT_ENCODER_TABLE_FORMATS:
            cells.append(_support_table_cell_html(row[format_key]))
        rows.append(cells)
    return rows


def apply_text_encoder_support_table_selection(evt: gr.SelectData):
    """Handle a click on the text-encoder support table: switch to the
    Convert Text Encoder tab and pre-select that format. Unlike the main
    Model Support table, every TEXT_ENCODER_TABLE_FORMATS key is a real
    TEXT_ENCODER_FORMAT_CHOICES entry (NVFP4 isn't excluded here the way it
    is for diffusion models), so no no-op guard is needed."""
    _row, col = evt.index
    if col == 0:
        return gr.update(), gr.update()
    format_key = TEXT_ENCODER_TABLE_FORMATS[col - 1][1]
    value = "Q4_K_M" if format_key == "GGUF" else format_key
    return gr.update(selected=2), gr.update(value=value)


_SAFETENSORS_DTYPE_KEYS = {key for _, key in SAFETENSORS_DTYPE_CHOICES}


def apply_support_table_selection(evt: gr.SelectData):
    """Handle a click on the Model Support table: switch to the matching
    Convert tab and pre-select that format, or no-op for the Model column.

    The GGUF column collapses every K-quant level into one cell (see
    model_support.TABLE_FORMATS's comment) — clicking it can't pick a
    specific quant level, so it defaults to Q4_K_M, this tool's own
    "recommended ★" choice, same as GGUF's own dropdown default.

    NVFP4/NVFP4_MIXED appear in the table (TABLE_FORMATS) but are
    deliberately absent from SAFETENSORS_DTYPE_CHOICES — no verified safe
    mode exists for them yet (safetensors_quant.py, docs/issues_analysis.md
    #15). Clicking those cells must stay a no-op like the Model column, or
    it would set the Safetensors dropdown to a value outside its own
    choices and let a NVFP4 conversion run through the GUI despite that.
    """
    row, col = evt.index
    if col == 0:
        return gr.update(), gr.update(), gr.update()

    format_key = TABLE_FORMATS[col - 1][1]
    if format_key == "GGUF":
        return gr.update(selected=0), gr.update(value="Q4_K_M"), gr.update()
    if format_key not in _SAFETENSORS_DTYPE_KEYS:
        return gr.update(), gr.update(), gr.update()
    return gr.update(selected=1), gr.update(), gr.update(value=format_key)


_SUPPORT_TABLE_LEGEND_HTML = """
<div style="font-size: var(--type-small); color: var(--s2g-muted); margin-top: 8px;">
  <strong style="color:var(--s2g-support-good);">✓ Verified</strong> — actually converted, loaded, and rendered correctly in ComfyUI with this tool's own output.
  &nbsp;·&nbsp;
  <strong style="color:var(--s2g-support-caution);">⚠ Caution</strong> — technically supported by this tool, but not render-tested for this architecture: no evidence either way.
  &nbsp;·&nbsp;
  <strong style="color:var(--s2g-support-bad);">✗ Known issue</strong> — actually render-tested and confirmed to produce wrong/broken output.
  &nbsp;·&nbsp;
  <strong style="color:var(--s2g-muted);">? Unknown</strong> — this combination has never been attempted.
</div>
"""


# ──────────────────────────────────────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────────────────────────────────────

CSS = """
/* ── Design tokens ─────────────────────────────────────────────────────────
   Named palette (not Gradio's stock hue presets) + a real type scale.
   Light values live on :root; dark values re-declare the same names under
   Gradio's own dark-mode selector (":root.dark, :root .dark" — Gradio toggles
   a class, not prefers-color-scheme, so that's the selector that must match
   to stay in sync with the rest of the UI, incl. a manual ?__theme override).
*/
:root {
    color-scheme: light dark;

    --s2g-bg: #f7f8fa;
    --s2g-surface: #ffffff;
    --s2g-line: #d9e0e7;

    --s2g-text: #111827;
    --s2g-muted: #5f6b7a;

    --s2g-accent: #0891b2;
    --s2g-accent-2: #0f766e;
    --s2g-accent-soft: rgba(8, 145, 178, 0.12);
    --s2g-danger: #dc2626;
    --s2g-warn: #b7791f;
    --s2g-warn-soft: rgba(183, 121, 31, 0.12);

    /* Support-table cell colors ONLY (verified/caution/bad symbols + legend) --
       deliberately literal red/yellow/green rather than the teal --s2g-accent
       used for general UI branding elsewhere, per explicit user request for
       an unambiguous traffic-light scheme on these two tables specifically. */
    --s2g-support-good: #16a34a;
    --s2g-support-caution: #ca8a04;
    --s2g-support-bad: #dc2626;

    --s2g-shadow: 0 16px 42px -28px rgba(15, 23, 42, 0.45);

    --font-ui: "Segoe UI Variable", "Aptos", "Segoe UI", system-ui, sans-serif;
    --font-mono: "Cascadia Mono", "JetBrains Mono", Consolas, ui-monospace, monospace;

    --type-display: 1.625rem;  /* app title */
    --type-body: 0.925rem;     /* intro / description text */
    --type-small: 0.8125rem;   /* subtitle, size-info, status */

    /* Every Gradio block container (tabs, rows, columns, HTML, Group…) now
       inherits the page background — removes the "floating panel" look
       everywhere except .card, which gets its own explicit surface below. */
    --block-background-fill: var(--background-fill-primary);
}
:root.dark, :root .dark {
    --s2g-bg: #0b0f14;
    --s2g-surface: #111820;
    --s2g-line: #26323f;

    --s2g-text: #edf3f8;
    --s2g-muted: #a7b3bf;

    --s2g-accent: #22d3ee;
    --s2g-accent-2: #2dd4bf;
    --s2g-accent-soft: rgba(34, 211, 238, 0.14);
    --s2g-danger: #f87171;
    --s2g-warn: #f5b84b;
    --s2g-warn-soft: rgba(245, 184, 75, 0.14);

    --s2g-support-good: #4ade80;
    --s2g-support-caution: #fbbf24;
    --s2g-support-bad: #f87171;

    --s2g-shadow: 0 20px 52px -30px rgba(0, 0, 0, 0.75);
}
:root {
    --background-fill-primary: var(--s2g-bg);
    --border-color-primary: var(--s2g-line);
    --body-text-color: var(--s2g-text);
    --body-text-color-subdued: var(--s2g-muted);
}

/* ── Scroll prevention ──────────────────────────────────────────────────── */
html, body { overflow-anchor: none !important; scroll-behavior: auto !important; }

/* ── Page width / base type ─────────────────────────────────────────────── */
.gradio-container {
    /* 1100px fit 6 tabs at this font stack's intended metrics; the 7th
       ("Model Support") pushed Gradio's tab-nav into its auto-collapse "..."
       overflow menu, which hid "Extract Components"/"Model Support" behind
       it entirely on a real Windows/Brave render (font-fallback rendering
       runs noticeably wider than in-CI Chromium) -- Gradio measures actual
       rendered tab widths via JS, not a CSS wrap, so the only fix is more
       room. Widened well past the minimum needed at default metrics to
       leave headroom for wider font-fallback renders too. */
    max-width: 1600px !important; margin: 0 auto !important; padding-top: 8px !important;
    font-family: var(--font-ui);
}

/* ── App header / banner ─────────────────────────────────────────────────
   A technical "compiler console" surface, not a marketing gradient: subtle
   blueprint grid + a hex-cell motif (matching the ⬡ tab icons), CSS-only
   (no images/SVG files), tinted with the accent tokens above so it tracks
   light/dark automatically. */
#app-header {
    position: relative;
    overflow: hidden;
    isolation: isolate;
    margin: 0 0 14px;
    padding: 22px 26px 20px;
    border: 1px solid var(--s2g-line);
    border-radius: 14px;
    background:
        radial-gradient(circle at 85% 15%, var(--s2g-accent-soft), transparent 32%),
        linear-gradient(135deg, var(--s2g-accent-soft), transparent 45%),
        var(--s2g-surface);
    box-shadow: var(--s2g-shadow);
    line-height: 1.4;
}
/* Blueprint grid, faded out toward the right where the hex motif sits */
#app-header::before {
    content: "";
    position: absolute; inset: 0; z-index: -2;
    opacity: 0.55;
    background-image:
        linear-gradient(to right, var(--s2g-line) 1px, transparent 1px),
        linear-gradient(to bottom, var(--s2g-line) 1px, transparent 1px);
    background-size: 28px 28px;
    mask-image: linear-gradient(90deg, black, transparent 75%);
    -webkit-mask-image: linear-gradient(90deg, black, transparent 75%);
}
/* Header mark: files -> quantized block icon (assets/header_logo.png),
   embedded as a data: URI so the app never fetches an external asset —
   consistent with the no-outbound-requests-at-startup policy above. Sized to
   fit inside the header's own content height (clips at overflow:hidden). */
#app-header::after {
    content: "";
    position: absolute; right: 24px; top: 50%; transform: translateY(-50%); z-index: -1;
    width: 62px; height: 62px;
    background: url("__HEADER_LOGO_DATA_URI__") no-repeat center / contain;
}
#app-title {
    max-width: 760px; margin: 0 0 6px 0;
    color: var(--s2g-text);
    font-size: var(--type-display); font-weight: 700; line-height: 1.15;
    letter-spacing: -0.018em;
}
#app-sub {
    max-width: 780px; margin: 0;
    color: var(--s2g-muted);
    font-size: var(--type-small); line-height: 1.55;
}
#app-sub strong { color: var(--s2g-text); font-weight: 650; }

/* ── Tab intro text: set off from the settings card below it ─────────────── */
.intro {
    background: var(--s2g-accent-soft);
    border-left: 3px solid var(--s2g-accent);
    border-radius: 8px;
    padding: 10px 14px !important;
    margin-bottom: 12px !important;
}
.intro p { margin: 0 !important; font-size: var(--type-body); color: var(--s2g-muted); }

/* ── Per-architecture format recommendation badge (Convert -> Safetensors) ─ */
.fmt-hint { min-height: 0 !important; }
.fmt-hint p {
    margin: 6px 0 0 !important; font-size: var(--type-small); font-weight: 600;
    padding: 6px 10px; border-radius: 6px; display: inline-block;
}
.fmt-hint-ok p   { background: var(--s2g-accent-soft); color: var(--s2g-accent); }
.fmt-hint-warn p { background: var(--s2g-warn-soft);   color: var(--s2g-warn); }

/* ── Strip outer wrappers: tab-container, any block ancestor of .card ────── */
.tabitem, .tab-content, .tabs > .tabitem,
[role="tabpanel"], .tabs > div > div,
.block:has(> .form.card), .block:has(> .card) {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
    padding: 0 !important;
}

/* ── Input card ──────────────────────────────────────────────────────────── */
.card {
    border: 1.5px solid var(--s2g-line) !important;
    border-radius: 14px !important;
    padding: 18px 18px 14px !important;
    background: transparent !important;
    box-shadow: none !important;
    margin-bottom: 10px !important;
    transition: border-color 150ms ease;
}
.card:focus-within { border-color: var(--s2g-accent) !important; }
.card > .form, .card .form {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    border-radius: 0 !important;
}

/* ── Browse button column ───────────────────────────────────────────────── */
.browse-col {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
    align-self: flex-start !important;
    display: flex !important;
    justify-content: flex-start !important;
}
.browse-col .block {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
}
.browse-col button {
    width: 100% !important;
}
.path-input {
    min-width: 0 !important;
}

/* ── Path input boxes: rounded ─────────────────────────────────────────── */
.path-input { margin-top: 0 !important; }
.path-input .block { border-radius: 8px !important; }
.path-input textarea { border-radius: 6px !important; }

/* ── Size estimate ──────────────────────────────────────────────────────── */
#size-info { font-size: var(--type-small); color: var(--s2g-muted); padding-top: 8px; }

/* ── Status textbox: Gradio default styling, no custom border colour ─────── */
#conv-status, #pad-status, #fix-status, #extract-status, #st-status {
    margin-top: 6px !important;
    position: sticky !important;
    bottom: 0 !important;
    z-index: 200 !important;
}
#conv-status textarea, #pad-status textarea, #fix-status textarea, #extract-status textarea, #st-status textarea {
    font-family: var(--font-mono) !important;
    font-size: var(--type-small) !important;
    font-weight: 600 !important;
    resize: none !important;
}

/* ── Action buttons ─────────────────────────────────────────────────────── */
#convert-btn, #fix-pad-btn, #fix-5d-btn, #extract-btn, #st-convert-btn {
    min-height: 44px !important; font-size: 1em !important; font-weight: 600 !important;
    transition: transform 120ms cubic-bezier(0.23, 1, 0.32, 1) !important;
}
#convert-btn:active, #fix-pad-btn:active, #fix-5d-btn:active, #extract-btn:active, #st-convert-btn:active {
    transform: scale(0.98);
}
#cancel-btn, #st-cancel-btn { min-height: 44px !important; }
#cancel-btn button, #cancel-btn, #st-cancel-btn button, #st-cancel-btn {
    background: var(--s2g-danger) !important; border-color: var(--s2g-danger) !important; color: #fff !important;
    border-radius: 8px !important;
}
#cancel-btn button:hover, #cancel-btn:hover, #st-cancel-btn button:hover, #st-cancel-btn:hover { filter: brightness(0.9); }

/* ── Log areas ──────────────────────────────────────────────────────────── */
#conv-log textarea, #pad-log textarea, #fix-log textarea, #extract-log textarea, #st-log textarea {
    font-family: var(--font-mono); font-size: 0.8em; line-height: 1.45;
}
"""


def _load_header_logo_data_uri() -> str:
    """Base64-embed assets/header_logo.png as a data: URI — keeps the header
    mark self-contained (no external request, no Gradio static-file route to
    configure) and gui.py itself small (the encoded bytes live in the binary
    asset file, not as a literal string in source). Falls back to no image
    (background stays transparent) if the asset is ever missing."""
    path = Path(__file__).parent / "assets" / "header_logo.png"
    try:
        return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        return ""


CSS = CSS.replace("__HEADER_LOGO_DATA_URI__", _load_header_logo_data_uri())

_THEME = gr.themes.Default(
    primary_hue="cyan",
    secondary_hue="teal",
    neutral_hue="zinc",
    # Segoe UI Variable / Aptos are Windows-native (no CDN fetch needed, matches
    # the app's telemetry-off / no-outbound-requests-at-startup policy above);
    # both fall back to Inter-equivalent system stacks where unavailable.
    font=["Segoe UI Variable", "Aptos", "Segoe UI", "system-ui", "sans-serif"],
    font_mono=["Cascadia Mono", "JetBrains Mono", "Consolas", "ui-monospace", "monospace"],
)

_HEADER_HTML = """
<div id="app-header">
  <div id="app-title">⬡ safetensors → GGUF</div>
  <div id="app-sub">Convert model checkpoints to GGUF &nbsp;·&nbsp;
    <strong>llama.cpp</strong> / <strong>ComfyUI-GGUF</strong> &nbsp;·&nbsp;
    Types marked <strong>[lq]</strong> require the llama-quantize binary.
  </div>
</div>
"""

# ──────────────────────────────────────────────────────────────────────────────
# Streaming infrastructure
# ──────────────────────────────────────────────────────────────────────────────

def _run_job(fn, *args, **kwargs) -> tuple[queue.Queue, threading.Event, dict]:
    """Run fn in a daemon thread; fn must accept on_progress and on_log kwargs."""
    q: queue.Queue = queue.Queue()
    done = threading.Event()
    result: dict = {}

    def _on_progress(idx, total, key):
        q.put(("progress", idx, total, key))

    def _on_log(msg):
        q.put(("log", msg))

    def worker():
        try:
            out = fn(*args, on_progress=_on_progress, on_log=_on_log, **kwargs)
            result["out"] = out[0] if isinstance(out, tuple) else out
        except Exception as exc:
            result["error"] = str(exc)
        finally:
            q.put(("done",))
            done.set()

    threading.Thread(target=worker, daemon=True).start()
    return q, done, result


def _stream(
    q: queue.Queue,
    done: threading.Event,
    result: dict,
) -> Generator[tuple[str, str], None, None]:
    """Drain q; yield (log_text, status_text).

    Handles:
      ("progress", idx, total, key)  — absolute counters
      ("progress_frac", frac, desc)  — pre-computed fraction [0..1]
      ("log", msg)
      ("done",)

    Progress percentage is embedded in the status text; no gr.Progress element
    is used to avoid Gradio injecting a UI element that triggers scrollIntoView.
    """
    log_lines: list[str] = []
    status = "Running…"

    while True:
        try:
            msg = q.get(timeout=0.05)
        except queue.Empty:
            yield "\n".join(log_lines), status
            if done.is_set() and q.empty():
                break
            continue

        kind = msg[0]
        if kind == "done":
            break
        elif kind == "progress":
            _, idx, total, key = msg
            short = key if len(key) <= 48 else f"…{key[-46:]}"
            pct = f"  {int(idx / total * 100)}%" if total else ""
            status = f"{idx} / {total}{pct}   {short}"
        elif kind == "progress_frac":
            _, frac, desc = msg
            pct = int(max(0.0, min(1.0, frac)) * 100)
            status = f"{desc}  ({pct}%)"
        elif kind == "log":
            log_lines.append(msg[1])

        yield "\n".join(log_lines), status

    done.wait()

    if result.get("cancelled"):
        log_lines.append("\n⚠️  Cancelled — RAM released")
        yield "\n".join(log_lines), "Cancelled"
    elif "error" in result:
        log_lines.append(f"\n❌  {result['error']}")
        yield "\n".join(log_lines), "Error"
    else:
        out_path = result.get("out", "")
        log_lines.append(f"\n✅  Done → {out_path}")
        yield "\n".join(log_lines), "Done ✓"


# ──────────────────────────────────────────────────────────────────────────────
# Path helpers
# ──────────────────────────────────────────────────────────────────────────────

def _strip_model_suffix(p: str | Path) -> Path:
    stem = Path(p)
    while stem.suffix in {".safetensors", ".ckpt", ".pt", ".bin", ".pth", ".gguf"}:
        stem = stem.with_suffix("")
    return stem


def _auto_dst(src: str) -> str:
    if not src or not src.strip():
        return ""
    return str(_strip_model_suffix(src.strip())) + "-{ftype}.gguf"


def _resolve_dst(src: str, dst: str | None, quant_key: str) -> str | None:
    """Resolve the user-supplied output path to a concrete file path.

    Rules (in order):
      - Empty / None  → return None  (convert_file auto-generates next to source)
      - Ends with / or backslash, or is an existing directory
                      → append  <model_stem>-<quant_key>.gguf  to that directory
      - Contains {ftype}
                      → replace {ftype} with quant_key
      - Otherwise     → use as-is (caller supplied a full file path)
    """
    if not dst:
        return None
    dst = dst.strip()
    if not dst:
        return None

    if dst.endswith(("/", "\\")) or Path(dst).is_dir():
        stem = _strip_model_suffix(src)
        return str(Path(dst) / f"{stem.name}-{quant_key}.gguf")

    if "{ftype}" in dst:
        return dst.replace("{ftype}", quant_key)

    return dst


def _resolve_dst_st(src: str, dst: str | None, target_key: str) -> str | None:
    """Resolve the user-supplied Safetensors output path; mirrors _resolve_dst."""
    if not dst:
        return None
    dst = dst.strip()
    if not dst:
        return None

    if dst.endswith(("/", "\\")) or Path(dst).is_dir():
        stem = _strip_model_suffix(src)
        return str(Path(dst) / f"{stem.name}-{target_key}.safetensors")

    if "{ftype}" in dst:
        return dst.replace("{ftype}", target_key)

    return dst


# ──────────────────────────────────────────────────────────────────────────────
# Conversion pipeline
# ──────────────────────────────────────────────────────────────────────────────

def _pipeline(
    src: str,
    dst: str,
    quant_key: str,
    exe_path: str,
    nthreads: int | float | None,
    keep_intermediate: bool,
    overwrite: bool,
    q: queue.Queue,
    cancel_event: threading.Event | None = None,
) -> str:
    """Full conversion pipeline; writes progress/log events to q.

    Single-step (Python types): safetensors → GGUF
    Two-step (K-quants):        safetensors → F16 GGUF → quantized GGUF
    Optional third step:        auto-run fix_5d when a side-car file exists
    """
    def _log(msg: str) -> None:
        q.put(("log", msg))

    def _frac(frac: float, desc: str) -> None:
        q.put(("progress_frac", max(0.0, min(1.0, frac)), desc))

    src = src.strip()
    dst_raw = dst.strip() if dst else ""
    exe = exe_path.strip() or None
    nthreads = int(nthreads or 0) or None

    is_kquant = quant_key in LLAMA_QUANT_KEYS
    # Total-step count for k-quants is 2 normally, 3 if a 5D side-car exists.
    # We only know needs_fix after step 1, so step 1 shows "1/2+" as a hint.
    step1_scale = 0.45 if is_kquant else 1.0
    step1_label = "[1/2+]" if is_kquant else "[1/1]"

    # ── Step 1: Python conversion ────────────────────────────────────────────
    _log(f"INFO:  {step1_label} Converting to {'F16 GGUF…' if is_kquant else quant_key + ' GGUF…'}")

    if is_kquant:
        target_qt, _ = PYTHON_PRECISIONS["F16"]
        # Resolve the final destination first so the intermediate lands next to it
        final_dst = _resolve_dst(src, dst_raw, quant_key)
        if final_dst:
            intermediate = str(Path(final_dst).parent / (_strip_model_suffix(src).name + "-F16-tmp.gguf"))
        else:
            intermediate = str(_strip_model_suffix(src)) + "-F16-tmp.gguf"
        step1_dst: str | None = intermediate
    else:
        target_qt, _ = PYTHON_PRECISIONS[quant_key]
        intermediate = None
        step1_dst = _resolve_dst(src, dst_raw, quant_key)

    def _prog1(idx, total, key):
        _frac((idx / total * step1_scale) if total else 0.0, f"{step1_label} Tensor {idx}/{total}")

    _, arch = convert_file(
        src, step1_dst,
        interact=False, overwrite=True,
        on_progress=_prog1, on_log=_log,
        target_quant=target_qt, cancel_event=cancel_event,
        log_tensor_every=GUI_TENSOR_LOG_EVERY,
        apply_unsqueeze=not is_kquant,
    )

    if is_kquant and getattr(arch, "shape_fix", False):
        _log(
            "WARNING: SD1/SDXL model uses tensor reshaping (shape_fix). "
            "llama-quantize may not handle this correctly — verify the output."
        )

    if not is_kquant:
        fix_path = Path(f"fix_5d_tensors_{arch.arch}.safetensors")
        if fix_path.is_file():
            _log(
                f"INFO:  5D tensor side-car found: {fix_path}. "
                "Use the 'Fix 5D Tensors' tab to insert them after llama-quantize."
            )
        # step1_dst is already resolved; fall back to source dir if None (auto-generated)
        return step1_dst or str(_strip_model_suffix(src)) + f"-{quant_key}.gguf"

    # ── Step 2: llama-quantize ───────────────────────────────────────────────
    fix_path = Path(f"fix_5d_tensors_{arch.arch}.safetensors")
    needs_fix = fix_path.is_file()
    # llama-quantize collapses [1, D] pad token shapes back to [D]; re-apply fix afterwards
    needs_pad_fix = bool(getattr(arch, 'keys_unsqueeze', None))
    total_steps = 2 + (1 if needs_fix else 0) + (1 if needs_pad_fix else 0)
    step2_end = 0.85 if needs_fix or needs_pad_fix else 0.95

    _log(f"INFO:  [2/{total_steps}] Quantizing to {quant_key} via llama-quantize…")

    # final_dst was already resolved above for the K-quant branch
    if not final_dst:
        final_dst = str(_strip_model_suffix(src)) + f"-{quant_key}.gguf"

    def _prog2(idx, total, key):
        _frac(0.45 + (idx / total * (step2_end - 0.45)) if total else 0.45, f"[2/{total_steps}] {idx}/{total}")

    run_quantize(
        intermediate, final_dst, quant_key,
        exe=exe, on_progress=_prog2, on_log=_log,
        nthreads=nthreads, cancel_event=cancel_event,
    )

    if not keep_intermediate and intermediate:
        try:
            Path(intermediate).unlink()
            _log(f"INFO:  Removed intermediate: {intermediate}")
        except OSError:
            pass

    result_path = final_dst
    current_step = 3

    # ── Step 3 (optional): fix 5D tensors ───────────────────────────────────
    if needs_fix:
        step3_end = 0.92 if needs_pad_fix else 1.0
        _log(f"INFO:  [{current_step}/{total_steps}] Auto-fixing 5D tensors from {fix_path}…")
        fixed_dst = result_path.replace(".gguf", "-fixed.gguf")

        def _prog3(idx, total, key):
            _frac(step2_end + (idx / total * (step3_end - step2_end)) if total else step2_end,
                  f"[{current_step}/{total_steps}] Tensor {idx}/{total}")

        _fix_5d(result_path, fixed_dst, fix_path=str(fix_path), overwrite=True, on_progress=_prog3, on_log=_log)
        _log(f"INFO:  5D tensors inserted → {fixed_dst}")
        result_path = fixed_dst
        current_step += 1

    # ── Step N (optional): re-fix pad token shapes collapsed by llama-quantize
    if needs_pad_fix:
        _log(f"INFO:  [{current_step}/{total_steps}] Re-fixing pad token shapes ([1, D] collapsed by llama-quantize)…")
        padfix_tmp = result_path + ".padfix.tmp"
        _fix_pad(result_path, padfix_tmp, overwrite=True, on_log=_log)
        Path(padfix_tmp).replace(Path(result_path))

    return result_path


def run_convert(
    src: str,
    dst: str,
    quant_key: str,
    exe_path: str,
    nthreads: int | float | None,
    keep_intermediate: bool,
    overwrite: bool,
) -> Generator[tuple[str, str], None, None]:
    """Run the full conversion pipeline and stream (log_text, status_text) updates.

    Wires a cancel_event to _active_cancel so the Cancel button can interrupt
    the pipeline.  Calls gc.collect() after completion or cancellation to
    release GPU/CPU memory immediately.
    """
    global _active_cancel

    if not src or not src.strip():
        yield "❌  No source file selected.", "Error — no input"
        return

    cancel_event = threading.Event()
    _active_cancel = cancel_event

    q: queue.Queue = queue.Queue()
    done = threading.Event()
    result: dict = {}

    def worker():
        import gc
        try:
            result["out"] = _pipeline(
                src, dst, quant_key, exe_path, nthreads,
                keep_intermediate, overwrite, q, cancel_event,
            )
        except ConversionCancelled:
            result["cancelled"] = True
        except Exception as exc:
            result["error"] = str(exc)
        finally:
            gc.collect()
            q.put(("done",))
            done.set()

    threading.Thread(target=worker, daemon=True).start()
    try:
        yield from _stream(q, done, result)
    finally:
        _active_cancel = None


def run_st_convert(
    src: str,
    dst: str,
    fmt: str,
    overwrite: bool,
) -> Generator[tuple[str, str], None, None]:
    """Run convert_to_safetensors in a background thread and stream (log_text, status_text).

    Mirrors run_convert's worker-thread + cancel_event pattern (Task 5 review
    fix: the tab originally ran conversion synchronously in the click handler,
    which blocked the UI with no live log/progress and no way to cancel).
    Uses a dedicated _active_cancel_st slot rather than _active_cancel so this
    tab's Cancel button never interferes with a concurrent GGUF conversion.
    """
    global _active_cancel_st

    if not src or not src.strip():
        yield "❌  No source file selected.", "Error — no input"
        return

    cancel_event = threading.Event()
    _active_cancel_st = cancel_event

    q: queue.Queue = queue.Queue()
    done = threading.Event()
    result: dict = {}

    def worker() -> None:
        try:
            out_path, _ = convert_to_safetensors(
                src.strip(),
                dst_path=_resolve_dst_st(src.strip(), dst, fmt),
                target_key=fmt,
                overwrite=overwrite,
                on_progress=lambda idx, total, key: q.put(("progress", idx, total, key)),
                on_log=lambda msg: q.put(("log", msg)),
                cancel_event=cancel_event,
                log_tensor_every=GUI_TENSOR_LOG_EVERY,
            )
            result["out"] = out_path
        except RuntimeError as exc:
            # convert_to_safetensors raises plain RuntimeError("cancelled") on
            # cancel_event (no dedicated exception type, unlike convert.py's
            # ConversionCancelled) — distinguish it so _stream shows "Cancelled"
            # instead of treating a user-requested stop as an error.
            if str(exc) == "cancelled":
                result["cancelled"] = True
            else:
                result["error"] = str(exc)
        except Exception as exc:
            result["error"] = str(exc)
        finally:
            q.put(("done",))
            done.set()

    threading.Thread(target=worker, daemon=True).start()
    try:
        yield from _stream(q, done, result)
    finally:
        _active_cancel_st = None


def run_te_convert(
    src: str,
    repo_id: str,
    dst: str,
    format_key: str,
) -> Generator[tuple[str, str], None, None]:
    """Run convert_text_encoder_any in a background thread and stream (log_text, status_text).

    Mirrors run_st_convert's worker-thread + cancel_event + _stream pattern
    (see Task 5 review fix) instead of the plan brief's literal synchronous
    example, so this tab gets live log streaming and a working Cancel button
    like every other conversion tab in this file. Uses a dedicated
    _active_cancel_te slot, separate from _active_cancel and _active_cancel_st,
    so this tab's Cancel button never interferes with a concurrent GGUF or
    safetensors conversion. The base repo ID is optional for GGUF formats:
    convert_text_encoder()/convert_text_encoder_kquant() auto-detect the base
    model family from the weights' own tensor shapes (detect_text_encoder_family)
    when it's left blank, falling back to a manual repo ID only for base models
    outside this tool's vendored candidate families. FP8/NVFP4 safetensors
    formats need no HuggingFace download at all, so an empty repo ID is
    always fine for those. A RuntimeError from the worker (auto-detection
    failed AND no repo ID given) surfaces through the normal error handler
    below rather than a separate pre-flight check.
    """
    global _active_cancel_te

    if not src or not src.strip():
        yield "❌  No source file selected.", "Error — no input"
        return

    cancel_event = threading.Event()
    _active_cancel_te = cancel_event

    q: queue.Queue = queue.Queue()
    done = threading.Event()
    result: dict = {}

    def worker() -> None:
        try:
            out_path = convert_text_encoder_any(
                src.strip(),
                repo_id.strip() if repo_id else "",
                dst_path=(dst.strip() or None) if dst else None,
                format_key=format_key,
                on_log=lambda msg: q.put(("log", msg)),
                cancel_event=cancel_event,
            )
            result["out"] = out_path
        except RuntimeError as exc:
            # convert_text_encoder_any raises plain RuntimeError("cancelled") on
            # cancel_event, same convention as convert_to_safetensors — see
            # run_st_convert's matching handler above (review finding #4:
            # without this, cancelling this tab showed "Error: cancelled"
            # instead of "Cancelled").
            if str(exc) == "cancelled":
                result["cancelled"] = True
            else:
                result["error"] = str(exc)
        except Exception as exc:
            result["error"] = str(exc)
        finally:
            q.put(("done",))
            done.set()

    threading.Thread(target=worker, daemon=True).start()
    try:
        yield from _stream(q, done, result)
    finally:
        _active_cancel_te = None


def run_fix_pad_tokens(
    src: str,
    dst: str,
    overwrite: bool,
) -> Generator[tuple[str, str], None, None]:
    """Run fix_pad_tokens in a background thread and stream (log_text, status_text) updates."""
    if not src or not src.strip():
        yield "❌  No source GGUF selected.", "Error — no input"
        return
    src = src.strip()
    dst = dst.strip() or src.replace(".gguf", "-fixed.gguf")
    q, done, result = _run_job(_fix_pad, src, dst, overwrite=overwrite)
    yield from _stream(q, done, result)


def run_fix(
    src: str,
    dst: str,
    fix: str,
    overwrite: bool,
) -> Generator[tuple[str, str], None, None]:
    """Run fix_5d_tensors in a background thread and stream (log_text, status_text) updates."""
    if not src or not src.strip():
        yield "❌  No source GGUF selected.", "Error — no input"
        return

    src = src.strip()
    dst = dst.strip() or src.replace(".gguf", "-fixed.gguf")
    q, done, result = _run_job(_fix_5d, src, dst, fix.strip() or None, overwrite)
    yield from _stream(q, done, result)


def analyze_sdxl_components(src: str, models_root: str) -> tuple[str, str, str]:
    """Analyze embedded SDXL VAE/CLIP components for GUI display."""
    if not src or not src.strip():
        return "❌  No checkpoint selected.", "Error — no input", ""

    src = src.strip()
    if not Path(src).is_file():
        return f"❌  Source file not found: {src}", "Error", ""

    root = Path(models_root.strip()) if models_root and models_root.strip() else default_output_root(src)
    try:
        analysis = analyze_components(src, root)
    except Exception as exc:
        return f"❌  {exc}", "Error", str(root)

    return format_component_analysis(analysis), "Analysis complete", str(root)


def run_extract_components(
    src: str,
    models_root: str,
    extract_vae: bool,
    extract_clip_l: bool,
    extract_clip_g: bool,
    overwrite: bool,
) -> Generator[tuple[str, str], None, None]:
    """Extract selected SDXL components and stream (log_text, status_text)."""
    if not src or not src.strip():
        yield "❌  No checkpoint selected.", "Error — no input"
        return
    if not any((extract_vae, extract_clip_l, extract_clip_g)):
        yield "❌  No components selected.", "Error — no selection"
        return

    src = src.strip()
    root = Path(models_root.strip()) if models_root and models_root.strip() else default_output_root(src)
    q: queue.Queue = queue.Queue()
    done = threading.Event()
    result: dict = {}

    def worker() -> None:
        try:
            def _on_log(msg: str) -> None:
                q.put(("log", msg))

            written = extract_components(
                src,
                root,
                extract_vae=extract_vae,
                extract_clip_l=extract_clip_l,
                extract_clip_g=extract_clip_g,
                overwrite=overwrite,
                on_log=_on_log,
            )
            result["out"] = ", ".join(item.path for item in written) or "nothing written"
        except Exception as exc:
            result["error"] = str(exc)
        finally:
            q.put(("done",))
            done.set()

    threading.Thread(target=worker, daemon=True).start()
    yield from _stream(q, done, result)


# ──────────────────────────────────────────────────────────────────────────────
# Layout
# ──────────────────────────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    """Construct and return the Gradio Blocks application."""
    default_exe = str(find_exe() or "")
    default_exe_info = llama_quantize_info(default_exe)

    with gr.Blocks(title="safetensors → GGUF") as app:
        gr.HTML(_HEADER_HTML)

        with gr.Tabs() as main_tabs:

            # ── Convert → GGUF ─────────────────────────────────────────────
            with gr.Tab("⬡ Convert → GGUF", id=0):
                gr.Markdown(
                    "Convert a **Safetensors / CKPT** model checkpoint to **GGUF**.  "
                    "Python-native precisions write directly; K-quants run a 2-step "
                    "pipeline via the bundled `llama-quantize` binary.  5D-tensor and "
                    "pad-token fixes are chained automatically when needed.",
                    elem_classes=["intro"],
                )
                with gr.Column(elem_classes=["card"]):
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=5, elem_classes=["path-input"]):
                            src_path = gr.Textbox(
                                label="Source model",
                                placeholder="model.safetensors / .ckpt / .pt / .bin / .pth",
                                lines=1, max_lines=1,
                            )
                        with gr.Column(scale=0, min_width=124, elem_classes=["browse-col"]):
                            browse_conv_btn = gr.Button("Browse", size="sm")
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=5, elem_classes=["path-input"]):
                            dst_path = gr.Textbox(
                                label="Output path",
                                placeholder="Auto-generated next to source",
                                lines=1, max_lines=1,
                            )
                        with gr.Column(scale=0, min_width=124, elem_classes=["browse-col"]):
                            browse_dst_btn = gr.Button("Browse", size="sm")
                    quant_dropdown = gr.Dropdown(
                        choices=ALL_QUANT_CHOICES,
                        value="Q4_K_M",
                        label="Quantization",
                    )
                size_info = gr.Markdown("", elem_id="size-info")

                with gr.Accordion("Advanced", open=False):
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=5, elem_classes=["path-input"]):
                            exe_path = gr.Textbox(
                                label="llama-quantize path",
                                value=default_exe,
                                placeholder="Use Browse to select llama-quantize",
                                lines=1,
                                max_lines=1,
                                interactive=False,
                                info="Required for [lq] types. Auto-detected from Easy-Install, LLAMA_QUANTIZE_PATH, or PATH.",
                            )
                        with gr.Column(scale=0, min_width=124, elem_classes=["browse-col"]):
                            browse_llama_btn = gr.Button("Browse", size="sm")
                    detect_llama_btn = gr.Button("Detect llama-quantize", size="sm")
                    llama_update_info = gr.Textbox(
                        label="llama-quantize setup",
                        value=default_exe_info,
                        lines=6,
                        max_lines=7,
                        interactive=False,
                    )
                    nthreads = gr.Number(
                        label="llama-quantize threads",
                        value=0,
                        precision=0,
                        minimum=0,
                        info="0 = let llama-quantize choose automatically.",
                    )
                    with gr.Row():
                        keep_intermediate = gr.Checkbox(label="Keep F16 intermediate", value=False)
                        overwrite_conv = gr.Checkbox(label="Overwrite existing output", value=False)

                with gr.Row():
                    convert_btn = gr.Button("▶  Convert", variant="primary", scale=5, elem_id="convert-btn")
                    cancel_btn  = gr.Button("✕",          variant="stop",    scale=1, elem_id="cancel-btn")

                conv_status = gr.Textbox(
                    value="Ready", show_label=False, interactive=False,
                    lines=1, max_lines=1, elem_id="conv-status",
                )
                conv_log = gr.Textbox(
                    label="Log", lines=10, max_lines=10,
                    interactive=False, autoscroll=False, elem_id="conv-log",
                )

            # ── Convert → Safetensors ──────────────────────────────────────
            with gr.Tab("⬢ Convert → Safetensors", id=1):
                gr.Markdown(
                    "Convert a **Safetensors / CKPT** model checkpoint to a quantized "
                    "**Safetensors** file — no GGUF, no llama-quantize. INT8 uses "
                    "ComfyUI's `int8_tensorwise` convention (per-layer `weight_scale`), "
                    "rotating each weight with a block-Hadamard transform (ConvRot) "
                    "before quantizing where the input dimension allows it — loads "
                    "natively in ComfyUI without the GGUF loader node. FP8 defaults to "
                    "`full_precision_matrix_mult=true`, which makes ComfyUI skip its "
                    "dynamic activation-quantization compute path entirely, matching "
                    "the safety profile of \"scaled_fp8\" checkpoints already circulating "
                    "on Civitai/HuggingFace. NVFP4/MXFP8 are not offered here: ComfyUI "
                    "dynamically quantizes *activations* too for those formats, which "
                    "produced visibly wrong output on Lumina2/Z-Image in our testing — "
                    "INT8 only quantizes weights, avoiding that entirely "
                    "(docs/issues_analysis.md #15, #16).",
                    elem_classes=["intro"],
                )
                with gr.Column(elem_classes=["card"]):
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=5, elem_classes=["path-input"]):
                            st_src_path = gr.Textbox(
                                label="Source model",
                                placeholder="model.safetensors / .ckpt / .pt / .bin / .pth",
                                lines=1, max_lines=1,
                            )
                        with gr.Column(scale=0, min_width=124, elem_classes=["browse-col"]):
                            browse_st_src_btn = gr.Button("Browse", size="sm")
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=5, elem_classes=["path-input"]):
                            st_dst_path = gr.Textbox(
                                label="Output path",
                                placeholder="Auto-generated next to source",
                                lines=1, max_lines=1,
                            )
                        with gr.Column(scale=0, min_width=124, elem_classes=["browse-col"]):
                            browse_st_dst_btn = gr.Button("Browse", size="sm")
                    st_format_dropdown = gr.Dropdown(
                        choices=SAFETENSORS_DTYPE_CHOICES,
                        # Default to the mixed variant, not plain "INT8": mixed
                        # keeps hiprec tensors at F32 for extra safety margin,
                        # and is the recommended default (review finding #2).
                        value="INT8_MIXED",
                        label="Output format",
                    )
                    overwrite_st = gr.Checkbox(label="Overwrite existing output", value=False)
                st_format_info = gr.Markdown("", elem_id="st-format-info", elem_classes=["fmt-hint"])
                st_size_info = gr.Markdown("", elem_id="size-info")

                with gr.Row():
                    st_convert_btn = gr.Button("▶  Convert", variant="primary", scale=5, elem_id="st-convert-btn")
                    st_cancel_btn  = gr.Button("✕",          variant="stop",    scale=1, elem_id="st-cancel-btn")

                st_status = gr.Textbox(
                    value="Ready", show_label=False, interactive=False,
                    lines=1, max_lines=1, elem_id="st-status",
                )
                st_log = gr.Textbox(
                    label="Log", lines=10, max_lines=10,
                    interactive=False, autoscroll=False, elem_id="st-log",
                )

                def _browse_st_src():
                    return browse_model()

                browse_st_src_btn.click(_browse_st_src, outputs=st_src_path)
                browse_st_dst_btn.click(
                    browse_and_set_dst_st,
                    inputs=[st_src_path],
                    outputs=st_dst_path,
                )
                st_src_path.change(
                    update_format_recommendation_choices_and_size,
                    inputs=[st_src_path, st_format_dropdown],
                    outputs=[st_format_info, st_format_dropdown, st_size_info],
                )
                st_format_dropdown.change(
                    update_format_recommendation,
                    inputs=[st_src_path, st_format_dropdown],
                    outputs=st_format_info,
                )
                st_format_dropdown.change(
                    update_safetensors_size_estimate,
                    inputs=[st_src_path, st_format_dropdown],
                    outputs=st_size_info,
                )
                st_convert_event = st_convert_btn.click(
                    fn=run_st_convert,
                    inputs=[st_src_path, st_dst_path, st_format_dropdown, overwrite_st],
                    outputs=[st_log, st_status],
                    show_progress="hidden",
                )
                st_cancel_btn.click(fn=request_cancel_st, outputs=[st_status], cancels=[st_convert_event])

            # ── Convert Text Encoder ─────────────────────────────────────────
            with gr.Tab("✎ Convert Text Encoder", id=2):
                gr.Markdown(
                    "Convert a **bare single-file text-encoder checkpoint** (Qwen3, "
                    "Mistral, T5/UMT5, …) to GGUF or quantized safetensors.\n\n"
                    "**GGUF formats** (F32/F16/BF16/Q8_0/K-quants) need the base "
                    "model's `config.json`/tokenizer files — downloaded automatically "
                    "from the HuggingFace repo ID you provide (the **original base "
                    "model's** repo, not the fine-tune's, which usually doesn't have "
                    "one). Runs `convert_hf_to_gguf.py` from an auto-cloned llama.cpp "
                    "checkout (first run downloads it, needs `git` + internet). "
                    "K-quants (Q6_K…Q2_K) additionally build a **plain** llama-quantize "
                    "from that checkout via `cmake` on first use — needs `cmake` + a "
                    "C++ compiler. This is deliberately a different binary than the "
                    "City96-patched one used for diffusion models (see "
                    "docs/building-llama-quantize.md) — that patch isn't safe for LLM/text GGUFs.\n\n"
                    "**FP8/NVFP4 formats** write a quantized `.safetensors` file "
                    "instead — no base repo ID, no HuggingFace download, no llama.cpp "
                    "involved at all.",
                    elem_classes=["intro"],
                )
                with gr.Column(elem_classes=["card"]):
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=5, elem_classes=["path-input"]):
                            te_src_path = gr.Textbox(
                                label="Text-encoder weights file",
                                placeholder="qwen3_8b_abliterated.safetensors",
                                lines=1, max_lines=1,
                            )
                        with gr.Column(scale=0, min_width=124, elem_classes=["browse-col"]):
                            browse_te_src_btn = gr.Button("Browse", size="sm")
                    te_base_repo = gr.Textbox(
                        label="Base model HF repo ID (optional — auto-detected if blank)",
                        placeholder="Auto-detected from the weights; override e.g. Qwen/Qwen3-8B",
                        lines=1, max_lines=1,
                        info="The ORIGINAL base model's repo (config.json/tokenizer source), not the fine-tune's. Left blank, the tool fingerprints the weights' own tensor shapes against its vendored candidate families (Qwen3-4B/8B, Mistral-Small-3.2-24B, CLIP-L/bigG, T5-XXL, Qwen2.5-VL-7B, ERNIE-Image PE) — only needed manually for other base models. Ignored for FP8/NVFP4 formats.",
                    )
                    te_dst_path = gr.Textbox(
                        label="Output path",
                        placeholder="Auto-generated next to source",
                        lines=1, max_lines=1,
                    )
                    te_format = gr.Dropdown(
                        choices=TEXT_ENCODER_FORMAT_CHOICES, value="F16", label="Format",
                    )
                te_size_info = gr.Markdown("", elem_id="size-info")

                with gr.Row():
                    te_convert_btn = gr.Button("▶  Convert", variant="primary", scale=5, elem_id="te-convert-btn")
                    te_cancel_btn  = gr.Button("✕",          variant="stop",    scale=1, elem_id="te-cancel-btn")

                te_status = gr.Textbox(
                    value="Ready", show_label=False, interactive=False,
                    lines=1, max_lines=1, elem_id="te-status",
                )
                te_log = gr.Textbox(
                    label="Log", lines=10, max_lines=10,
                    interactive=False, autoscroll=False, elem_id="te-log",
                )

                def _browse_te_src():
                    return browse_model()

                browse_te_src_btn.click(_browse_te_src, outputs=te_src_path)
                te_src_path.change(
                    update_text_encoder_size_estimate,
                    inputs=[te_src_path, te_format],
                    outputs=te_size_info,
                )
                te_format.change(
                    update_text_encoder_size_estimate,
                    inputs=[te_src_path, te_format],
                    outputs=te_size_info,
                )
                te_convert_event = te_convert_btn.click(
                    fn=run_te_convert,
                    inputs=[te_src_path, te_base_repo, te_dst_path, te_format],
                    outputs=[te_log, te_status],
                    show_progress="hidden",
                )
                te_cancel_btn.click(fn=request_cancel_te, outputs=[te_status], cancels=[te_convert_event])

            # ── Fix Pad Tokens ─────────────────────────────────────────────
            with gr.Tab("⚒ Fix Pad Tokens", id=3):
                gr.Markdown(
                    "Correct `x_pad_token` / `cap_pad_token` shape `[D]` → `[1, D]` in an "
                    "existing **Lumina 2** GGUF.  Required when ComfyUI raises "
                    "*size mismatch for x_pad_token*.  New conversions are not affected.",
                    elem_classes=["intro"],
                )
                with gr.Column(elem_classes=["card"]):
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=5, elem_classes=["path-input"]):
                            fix_pad_src = gr.Textbox(
                                label="Source GGUF",
                                placeholder="model.gguf",
                                lines=1, max_lines=1,
                            )
                        with gr.Column(scale=0, min_width=124, elem_classes=["browse-col"]):
                            browse_fix_pad_btn = gr.Button("Browse", size="sm")
                    fix_pad_dst = gr.Textbox(
                        label="Output path",
                        placeholder="Auto-generated (source-fixed.gguf)",
                        lines=1, max_lines=1,
                    )
                    overwrite_fix_pad = gr.Checkbox(label="Overwrite existing output", value=False)

                fix_pad_btn = gr.Button("▶  Fix Pad Tokens", variant="primary", elem_id="fix-pad-btn")

                fix_pad_status = gr.Textbox(
                    value="Ready", show_label=False, interactive=False,
                    lines=1, max_lines=1, elem_id="pad-status",
                )
                pad_log = gr.Textbox(
                    label="Log", lines=8, max_lines=8,
                    interactive=False, autoscroll=False, elem_id="pad-log",
                )

            # ── Fix 5D Tensors ─────────────────────────────────────────────
            with gr.Tab("⚙ Fix 5D Tensors", id=4):
                gr.Markdown(
                    "Re-insert 5D tensors into a quantized GGUF.  "
                    "**Required for HunyuanVideo / Wan** when using llama-quantize outside "
                    "the Convert tab — the Convert tab chains this step automatically.",
                    elem_classes=["intro"],
                )
                with gr.Column(elem_classes=["card"]):
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=5, elem_classes=["path-input"]):
                            fix_src = gr.Textbox(
                                label="Source GGUF (quantized)",
                                placeholder="model-Q8_0.gguf",
                                lines=1, max_lines=1,
                            )
                        with gr.Column(scale=0, min_width=124, elem_classes=["browse-col"]):
                            browse_fix_btn = gr.Button("Browse", size="sm")
                    fix_dst = gr.Textbox(
                        label="Output path",
                        placeholder="Auto-generated (source-fixed.gguf)",
                        lines=1, max_lines=1,
                    )
                    fix_file = gr.Textbox(
                        label="Side-car file",
                        placeholder="fix_5d_tensors_<arch>.safetensors — auto-detected when empty",
                        lines=1, max_lines=1,
                    )
                    overwrite_fix = gr.Checkbox(label="Overwrite existing output", value=False)

                fix_btn = gr.Button("▶  Fix 5D Tensors", variant="primary", elem_id="fix-5d-btn")

                fix_status = gr.Textbox(
                    value="Ready", show_label=False, interactive=False,
                    lines=1, max_lines=1, elem_id="fix-status",
                )
                fix_log = gr.Textbox(
                    label="Log", lines=8, max_lines=8,
                    interactive=False, autoscroll=False, elem_id="fix-log",
                )

            # ── Extract Components ─────────────────────────────────────────
            with gr.Tab("▤ Extract Components", id=5):
                gr.Markdown(
                    "Analyze an **SDXL** checkpoint for embedded VAE, CLIP-L, and CLIP-G "
                    "components, compare them with local standard files when present, "
                    "then export selected components to the ComfyUI models folder.",
                    elem_classes=["intro"],
                )
                with gr.Column(elem_classes=["card"]):
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=5, elem_classes=["path-input"]):
                            extract_src = gr.Textbox(
                                label="Source checkpoint",
                                placeholder="SDXL .safetensors checkpoint",
                                lines=1, max_lines=1,
                            )
                        with gr.Column(scale=0, min_width=124, elem_classes=["browse-col"]):
                            browse_extract_src_btn = gr.Button("Browse", size="sm")
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=5, elem_classes=["path-input"]):
                            extract_root = gr.Textbox(
                                label="ComfyUI models root",
                                placeholder="Auto-detected from source path",
                                lines=1, max_lines=1,
                            )
                        with gr.Column(scale=0, min_width=124, elem_classes=["browse-col"]):
                            browse_extract_root_btn = gr.Button("Browse", size="sm")
                    with gr.Row():
                        extract_vae = gr.Checkbox(label="VAE", value=False)
                        extract_clip_l = gr.Checkbox(label="CLIP-L", value=True)
                        extract_clip_g = gr.Checkbox(label="CLIP-G", value=True)
                    overwrite_extract = gr.Checkbox(label="Overwrite existing output", value=False)

                with gr.Row():
                    analyze_components_btn = gr.Button("Analyze", variant="secondary", scale=1)
                    extract_components_btn = gr.Button(
                        "▶  Extract Selected",
                        variant="primary",
                        scale=2,
                        elem_id="extract-btn",
                    )

                extract_status = gr.Textbox(
                    value="Ready", show_label=False, interactive=False,
                    lines=1, max_lines=1, elem_id="extract-status",
                )
                extract_log = gr.Textbox(
                    label="Analysis / Log", lines=10, max_lines=12,
                    interactive=False, autoscroll=False, elem_id="extract-log",
                )

            # ── Model Support ──────────────────────────────────────────────
            with gr.Tab("⊞ Model Support", id=6):
                gr.Markdown(
                    "Which quantization formats this tool supports for each "
                    "detectable architecture. Click a cell to jump to the "
                    "matching Convert tab with that format pre-selected.",
                    elem_classes=["intro"],
                )
                support_table = gr.Dataframe(
                    label="Model Support",
                    headers=["Model", *[label for label, _ in TABLE_FORMATS]],
                    datatype="html",
                    value=_support_table_rows_for_dataframe(),
                    interactive=False,
                    wrap=True,
                )
                gr.HTML(_SUPPORT_TABLE_LEGEND_HTML)
                gr.Markdown(
                    "**Text encoders** (Convert Text Encoder tab), one row per "
                    "vendored base-model family this tool auto-detects "
                    "(`detect_text_encoder_family()`). Click a cell to jump to "
                    "the Convert Text Encoder tab with that format pre-selected.",
                    elem_classes=["intro"],
                )
                te_support_table = gr.Dataframe(
                    label="Text Encoder Support",
                    headers=["Family", *[label for label, _ in TEXT_ENCODER_TABLE_FORMATS]],
                    datatype="html",
                    value=_text_encoder_support_rows_for_dataframe(),
                    interactive=False,
                    wrap=True,
                )
                gr.HTML(_SUPPORT_TABLE_LEGEND_HTML)

        # ── Events ─────────────────────────────────────────────────────────
        support_table.select(
            apply_support_table_selection,
            outputs=[main_tabs, quant_dropdown, st_format_dropdown],
        )
        te_support_table.select(
            apply_text_encoder_support_table_selection,
            outputs=[main_tabs, te_format],
        )
        browse_conv_btn.click(browse_model, outputs=src_path)
        browse_dst_btn.click(browse_and_set_dst, inputs=[src_path, quant_dropdown], outputs=dst_path)
        browse_fix_pad_btn.click(browse_gguf, outputs=fix_pad_src)
        browse_fix_btn.click(browse_gguf, outputs=fix_src)
        browse_extract_src_btn.click(browse_model, outputs=extract_src)
        browse_extract_root_btn.click(browse_models_root, outputs=extract_root)

        src_path.change(_auto_dst, inputs=src_path, outputs=dst_path)
        src_path.change(update_size_estimate, inputs=[src_path, quant_dropdown], outputs=size_info)
        src_path.change(annotate_gguf_choices, inputs=src_path, outputs=quant_dropdown)
        quant_dropdown.change(update_size_estimate, inputs=[src_path, quant_dropdown], outputs=size_info)
        extract_src.change(
            lambda src: str(default_output_root(src)) if src and str(src).strip() else "",
            inputs=extract_src,
            outputs=extract_root,
        )

        browse_llama_btn.click(
            fn=browse_llama_quantize,
            inputs=[exe_path],
            outputs=[exe_path, llama_update_info],
        )
        detect_llama_btn.click(
            fn=detect_llama_quantize_path,
            outputs=[exe_path, llama_update_info],
        )

        conv_event = convert_btn.click(
            fn=run_convert,
            inputs=[src_path, dst_path, quant_dropdown, exe_path, nthreads, keep_intermediate, overwrite_conv],
            outputs=[conv_log, conv_status],
            show_progress="hidden",
        )
        cancel_btn.click(fn=request_cancel, outputs=[conv_status], cancels=[conv_event])

        fix_pad_btn.click(
            fn=run_fix_pad_tokens,
            inputs=[fix_pad_src, fix_pad_dst, overwrite_fix_pad],
            outputs=[pad_log, fix_pad_status],
        )
        fix_btn.click(
            fn=run_fix,
            inputs=[fix_src, fix_dst, fix_file, overwrite_fix],
            outputs=[fix_log, fix_status],
        )
        analyze_components_btn.click(
            fn=analyze_sdxl_components,
            inputs=[extract_src, extract_root],
            outputs=[extract_log, extract_status, extract_root],
        )
        extract_components_btn.click(
            fn=run_extract_components,
            inputs=[
                extract_src,
                extract_root,
                extract_vae,
                extract_clip_l,
                extract_clip_g,
                overwrite_extract,
            ],
            outputs=[extract_log, extract_status],
        )

    return app


# Injected into the page <head> as a plain <script> (global scope, not a module).
# Blocks all programmatic scroll APIs that Gradio fires during streaming updates.
# Manual scrolling via mousewheel / scrollbar is unaffected — those bypass these APIs.
_SCROLL_BLOCK_HEAD = (
    "<script>"
    "window.scrollTo=window.scroll=window.scrollBy=function(){};"
    "Element.prototype.scrollIntoView=function(){};"
    "</script>"
)

if __name__ == "__main__":
    build_app().launch(
        inbrowser=True,
        theme=_THEME,
        css=CSS,
        head=_SCROLL_BLOCK_HEAD,
    )
