"""Web UI for safetensors2GGUF — Gradio frontend for convert, quantize, and fix_5d."""

from __future__ import annotations

import os
import queue
import threading
from pathlib import Path
from typing import Generator

import gradio as gr

from convert import ConversionCancelled, convert_file
from fix_5d_tensors import fix_5d_tensors as _fix_5d
from quantize import (
    ALL_QUANT_CHOICES,
    LLAMA_QUANT_KEYS,
    PYTHON_PRECISIONS,
    SIZE_RATIOS,
    find_exe,
    run_quantize,
)

# ──────────────────────────────────────────────────────────────────────────────
# Cancel support
# ──────────────────────────────────────────────────────────────────────────────

_active_cancel: threading.Event | None = None


def request_cancel() -> str:
    """Signal the active conversion to stop; called by the Cancel button."""
    if _active_cancel is not None:
        _active_cancel.set()
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


def browse_model() -> str:
    """Open a native file dialog filtered to model formats; return the selected path."""
    return _browse(_MODEL_TYPES)


def browse_gguf() -> str:
    """Open a native file dialog filtered to GGUF files; return the selected path."""
    return _browse(_GGUF_TYPES)


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
    """Return a Markdown string estimating the output file size for the selected quant type.

    Returns an empty string when src is absent or not a file.
    """
    if not src or not os.path.isfile(src.strip()):
        return ""
    src_bytes = os.path.getsize(src.strip())
    ratio = SIZE_RATIOS.get(quant_key, 1.0)
    est = src_bytes * ratio
    line = f"Estimated output: **{_fmt_size(est)}**"
    if ratio < 1.0:
        line += f" &nbsp;·&nbsp; {(1 - ratio) * 100:.0f}% smaller than source ({_fmt_size(src_bytes)})"
    elif ratio > 1.0:
        line += f" &nbsp;·&nbsp; source: {_fmt_size(src_bytes)}"
    return line


# ──────────────────────────────────────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────────────────────────────────────

CSS = """
/* ── Prevent Gradio's progress/update events from scrolling the page ── */
html, body {
    overflow-anchor: none !important;
    scroll-behavior: auto !important;
}

/* ── Status bar ── */
#status-wrap {
    position: sticky;
    bottom: 0;
    z-index: 200;
    padding: 6px 14px 4px;
    border-top: 2px solid var(--border-color-accent, #3b82f6);
    background: var(--background-fill-secondary, #1e293b);
    border-radius: 0 0 8px 8px;
    margin-top: 4px;
}
#status-wrap textarea {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    resize: none !important;
    font-family: ui-monospace, monospace;
    font-size: 0.88em;
    font-weight: 600;
    color: var(--body-text-color);
    padding: 0 !important;
}

/* ── Action buttons — equal height ── */
#convert-btn, #cancel-btn, #fix-btn {
    min-height: 52px !important;
    font-size: 1.05em !important;
    letter-spacing: 0.03em;
}

/* ── Cancel button — red ── */
#cancel-btn button, #cancel-btn {
    background: #dc2626 !important;
    border-color: #b91c1c !important;
    color: #fff !important;
}
#cancel-btn button:hover, #cancel-btn:hover {
    background: #b91c1c !important;
}

/* ── Log areas ── */
#conv-log textarea, #fix-log textarea {
    font-family: ui-monospace, monospace;
    font-size: 0.82em;
    line-height: 1.5;
}
#size-info { margin-top: -8px; font-size: 0.9em; }
"""

_THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="sky",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
)

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


# ──────────────────────────────────────────────────────────────────────────────
# Conversion pipeline
# ──────────────────────────────────────────────────────────────────────────────

def _pipeline(
    src: str,
    dst: str,
    quant_key: str,
    exe_path: str,
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
    total_steps = 3 if needs_fix else 2
    step2_end = 0.85 if needs_fix else 0.95

    _log(f"INFO:  [2/{total_steps}] Quantizing to {quant_key} via llama-quantize…")

    # final_dst was already resolved above for the K-quant branch
    if not final_dst:
        final_dst = str(_strip_model_suffix(src)) + f"-{quant_key}.gguf"

    def _prog2(idx, total, key):
        _frac(0.45 + (idx / total * (step2_end - 0.45)) if total else 0.45, f"[2/{total_steps}] {idx}/{total}")

    run_quantize(intermediate, final_dst, quant_key, exe=exe, on_progress=_prog2, on_log=_log, cancel_event=cancel_event)

    if not keep_intermediate and intermediate:
        try:
            Path(intermediate).unlink()
            _log(f"INFO:  Removed intermediate: {intermediate}")
        except OSError:
            pass

    # ── Step 3 (optional): fix 5D tensors ───────────────────────────────────
    if needs_fix:
        _log(f"INFO:  [3/3] Auto-fixing 5D tensors from {fix_path}…")
        fixed_dst = final_dst.replace(".gguf", "-fixed.gguf")

        def _prog3(idx, total, key):
            _frac(step2_end + (idx / total * (1.0 - step2_end)) if total else step2_end, f"[3/3] Tensor {idx}/{total}")

        _fix_5d(final_dst, fixed_dst, fix_path=str(fix_path), overwrite=True, on_progress=_prog3, on_log=_log)
        _log(f"INFO:  5D tensors inserted → {fixed_dst}")
        return fixed_dst

    return final_dst


def run_convert(
    src: str,
    dst: str,
    quant_key: str,
    exe_path: str,
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
                src, dst, quant_key, exe_path,
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


# ──────────────────────────────────────────────────────────────────────────────
# Layout
# ──────────────────────────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    """Construct and return the Gradio Blocks application.

    Builds the Convert and Fix 5D Tensors tabs, wires all events (browse buttons,
    size estimate updates, convert/cancel/fix clicks), and returns the app object
    ready for .launch().
    """
    default_exe = str(find_exe() or "")

    with gr.Blocks(title="safetensors2GGUF") as app:
        gr.Markdown(
            "# safetensors2GGUF\n"
            "Convert model checkpoints to GGUF for **llama.cpp** and **ComfyUI-GGUF**.  "
            "Types marked **[lq]** use the bundled llama-quantize binary."
        )

        with gr.Tabs():

            # ── Tab 1: Convert ─────────────────────────────────────────────
            with gr.Tab("Convert"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=3):
                        src_path = gr.Textbox(
                            label="Source Model",
                            placeholder="/path/to/model.safetensors",
                            info="Supported: .safetensors  .ckpt  .pt  .bin  .pth",
                        )
                        browse_conv_btn = gr.Button("Browse…", size="sm", variant="secondary")

                    with gr.Column(scale=2):
                        dst_path = gr.Textbox(
                            label="Output Path",
                            placeholder="Auto-generated",
                            info="Folder path, full file path, or use {ftype} as placeholder.",
                        )
                        browse_dst_btn = gr.Button("Browse output folder…", size="sm", variant="secondary")
                        overwrite_conv = gr.Checkbox(label="Overwrite existing output", value=False)

                quant_dropdown = gr.Dropdown(
                    choices=ALL_QUANT_CHOICES,
                    value="Q4_K_M",
                    label="Quantization",
                    info="Python-native: F32/F16/BF16/Q8_0 · llama-quantize [lq]: all K-quants",
                )
                size_info = gr.Markdown("", elem_id="size-info")

                with gr.Accordion("Advanced", open=False):
                    exe_path = gr.Textbox(
                        label="llama-quantize path",
                        value=default_exe,
                        placeholder=r"H:\...\llama-quantize.exe",
                        info="Required for [lq] types. Auto-detected from the known ComfyUI location.",
                    )
                    keep_intermediate = gr.Checkbox(
                        label="Keep intermediate F16 GGUF (2-step pipeline only)",
                        value=False,
                    )

                with gr.Row():
                    convert_btn = gr.Button("⚙  Convert", variant="primary", scale=4, elem_id="convert-btn")
                    cancel_btn  = gr.Button("✕  Cancel",  variant="stop",    scale=1, elem_id="cancel-btn")

                with gr.Group(elem_id="status-wrap"):
                    conv_status = gr.Textbox(
                        value="Ready", show_label=False, interactive=False,
                        lines=1, max_lines=1,
                    )
                conv_log = gr.Textbox(
                    label="Log", lines=18, max_lines=18,
                    interactive=False, autoscroll=False, elem_id="conv-log",
                )

            # ── Tab 2: Fix 5D Tensors ──────────────────────────────────────
            with gr.Tab("Fix 5D Tensors"):
                gr.Markdown(
                    "Re-insert 5D tensors into a quantized GGUF file.\n\n"
                    "**Required for HunyuanVideo and Wan models** when running llama-quantize "
                    "manually outside the Convert tab. The Convert tab auto-chains this step."
                )

                with gr.Row(equal_height=False):
                    with gr.Column(scale=3):
                        fix_src = gr.Textbox(
                            label="Source GGUF (quantized)",
                            placeholder="/path/to/model-Q8_0.gguf",
                        )
                        browse_fix_btn = gr.Button("Browse…", size="sm", variant="secondary")

                    with gr.Column(scale=2):
                        fix_dst = gr.Textbox(
                            label="Output Path",
                            placeholder="Auto-generated (source-fixed.gguf)",
                        )
                        fix_file = gr.Textbox(
                            label="Side-car file (optional)",
                            placeholder="fix_5d_tensors_<arch>.safetensors",
                            info="Leave empty to auto-detect in the current directory",
                        )
                        overwrite_fix = gr.Checkbox(label="Overwrite existing output", value=False)

                fix_btn = gr.Button("⚙  Fix 5D Tensors", variant="primary", elem_id="fix-btn")

                with gr.Group(elem_id="status-wrap"):
                    fix_status = gr.Textbox(
                        value="Ready", show_label=False, interactive=False,
                        lines=1, max_lines=1,
                    )
                fix_log = gr.Textbox(
                    label="Log", lines=14, max_lines=14,
                    interactive=False, autoscroll=False, elem_id="fix-log",
                )

        # ── Wire up events ─────────────────────────────────────────────────

        # Browse buttons → native dialogs
        browse_conv_btn.click(browse_model, outputs=src_path)
        browse_fix_btn.click(browse_gguf, outputs=fix_src)
        browse_dst_btn.click(browse_and_set_dst, inputs=[src_path, quant_dropdown], outputs=dst_path)

        # src_path changes (typing or after browse) → refresh dst suggestion + size estimate
        src_path.change(_auto_dst, inputs=src_path, outputs=dst_path)
        src_path.change(update_size_estimate, inputs=[src_path, quant_dropdown], outputs=size_info)

        # Quantization change → refresh size estimate
        quant_dropdown.change(update_size_estimate, inputs=[src_path, quant_dropdown], outputs=size_info)

        conv_event = convert_btn.click(
            fn=run_convert,
            inputs=[src_path, dst_path, quant_dropdown, exe_path, keep_intermediate, overwrite_conv],
            outputs=[conv_log, conv_status],
            show_progress="hidden",  # prevents Gradio's auto-loader from calling scrollIntoView
        )
        cancel_btn.click(
            fn=request_cancel,
            outputs=[conv_status],
            cancels=[conv_event],
        )
        fix_btn.click(
            fn=run_fix,
            inputs=[fix_src, fix_dst, fix_file, overwrite_fix],
            outputs=[fix_log, fix_status],
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
    build_app().launch(inbrowser=True, theme=_THEME, css=CSS, head=_SCROLL_BLOCK_HEAD)
