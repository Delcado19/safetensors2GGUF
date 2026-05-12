"""Web UI for safetensors2GGUF — Gradio frontend for convert, quantize, and fix_5d."""

from __future__ import annotations

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

from convert import ConversionCancelled, convert_file
from fix_5d_tensors import fix_5d_tensors as _fix_5d
from fix_pad_tokens import fix_pad_tokens as _fix_pad
from quantize import (
    ALL_QUANT_CHOICES,
    LLAMA_QUANT_KEYS,
    PYTHON_PRECISIONS,
    estimate_output_size,
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


# ──────────────────────────────────────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────────────────────────────────────

CSS = """
/* ── Scroll prevention ──────────────────────────────────────────────────── */
html, body { overflow-anchor: none !important; scroll-behavior: auto !important; }

/* ── Page width ─────────────────────────────────────────────────────────── */
.gradio-container { max-width: 1100px !important; margin: 0 auto !important; padding-top: 8px !important; }

/* ── App header ─────────────────────────────────────────────────────────── */
#app-header {
    background: linear-gradient(120deg, #4f46e5 0%, #0284c7 100%);
    border-radius: 12px;
    padding: 18px 24px;
    margin-bottom: 14px;
    line-height: 1.4;
}
#app-title {
    font-size: 1.35em; font-weight: 700; color: #fff; margin: 0 0 4px 0;
    letter-spacing: -0.01em;
}
#app-sub { font-size: 0.87em; color: rgba(255,255,255,0.82); margin: 0; }
#app-sub strong { color: rgba(255,255,255,0.97); }

/* ── Input card ─────────────────────────────────────────────────────────── */
.card {
    border: 1.5px solid var(--border-color-primary, #e2e8f0) !important;
    border-radius: 10px !important;
    padding: 16px !important;
    background: var(--background-fill-primary, #fff) !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05) !important;
    margin-bottom: 10px !important;
}

/* ── Browse button alignment ────────────────────────────────────────────── */
.browse-btn { margin-top: 22px !important; }

/* ── Size estimate ──────────────────────────────────────────────────────── */
#size-info { font-size: 0.85em; color: var(--body-text-color-subdued); padding-top: 8px; }

/* ── Status bar ─────────────────────────────────────────────────────────── */
#status-wrap {
    position: sticky; bottom: 0; z-index: 200;
    padding: 5px 12px 3px;
    border-top: 2px solid #4f46e5;
    background: var(--background-fill-secondary, #f8fafc);
    border-radius: 0 0 10px 10px;
    margin-top: 6px;
}
#status-wrap textarea {
    background: transparent !important; border: none !important;
    box-shadow: none !important; resize: none !important;
    font-family: ui-monospace, monospace; font-size: 0.85em;
    font-weight: 600; color: var(--body-text-color); padding: 0 !important;
}

/* ── Action buttons ─────────────────────────────────────────────────────── */
#convert-btn, #fix-pad-btn, #fix-5d-btn {
    min-height: 44px !important; font-size: 1em !important; font-weight: 600 !important;
}
#cancel-btn { min-height: 44px !important; }
#cancel-btn button, #cancel-btn {
    background: #ef4444 !important; border-color: #dc2626 !important; color: #fff !important;
    border-radius: 8px !important;
}
#cancel-btn button:hover, #cancel-btn:hover { background: #dc2626 !important; }

/* ── Log areas ──────────────────────────────────────────────────────────── */
#conv-log textarea, #pad-log textarea, #fix-log textarea {
    font-family: ui-monospace, monospace; font-size: 0.8em; line-height: 1.45;
}
"""

_THEME = gr.themes.Default(
    primary_hue="indigo",
    secondary_hue="sky",
    neutral_hue="slate",
    font=["Inter", "ui-sans-serif", "sans-serif"],
    font_mono=["JetBrains Mono", "ui-monospace", "monospace"],
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

    run_quantize(intermediate, final_dst, quant_key, exe=exe, on_progress=_prog2, on_log=_log, cancel_event=cancel_event)

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


# ──────────────────────────────────────────────────────────────────────────────
# Layout
# ──────────────────────────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    """Construct and return the Gradio Blocks application."""
    default_exe = str(find_exe() or "")

    with gr.Blocks(title="safetensors → GGUF") as app:
        gr.HTML(_HEADER_HTML)

        with gr.Tabs():

            # ── Convert ────────────────────────────────────────────────────
            with gr.Tab("Convert"):
                with gr.Group(elem_classes=["card"]):
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=3):
                            with gr.Row():
                                src_path = gr.Textbox(
                                    label="Source model",
                                    placeholder="model.safetensors / .ckpt / .pt / .bin / .pth",
                                    scale=9,
                                )
                                browse_conv_btn = gr.Button(
                                    "Browse", size="sm", scale=0, elem_classes=["browse-btn"]
                                )
                            with gr.Row():
                                dst_path = gr.Textbox(
                                    label="Output path",
                                    placeholder="Auto-generated next to source",
                                    scale=9,
                                )
                                browse_dst_btn = gr.Button(
                                    "Browse", size="sm", scale=0, elem_classes=["browse-btn"]
                                )
                            quant_dropdown = gr.Dropdown(
                                choices=ALL_QUANT_CHOICES,
                                value="Q4_K_M",
                                label="Quantization",
                            )
                        with gr.Column(scale=2):
                            size_info = gr.Markdown("", elem_id="size-info")

                with gr.Accordion("Advanced", open=False):
                    exe_path = gr.Textbox(
                        label="llama-quantize path",
                        value=default_exe,
                        placeholder=r"H:\...\llama-quantize.exe",
                        info="Required for [lq] types. Auto-detected from the default ComfyUI path.",
                    )
                    with gr.Row():
                        keep_intermediate = gr.Checkbox(label="Keep F16 intermediate", value=False)
                        overwrite_conv = gr.Checkbox(label="Overwrite existing output", value=False)

                with gr.Row():
                    convert_btn = gr.Button("▶  Convert", variant="primary", scale=5, elem_id="convert-btn")
                    cancel_btn  = gr.Button("✕",          variant="stop",    scale=1, elem_id="cancel-btn")

                with gr.Group(elem_id="status-wrap"):
                    conv_status = gr.Textbox(
                        value="Ready", show_label=False, interactive=False,
                        lines=1, max_lines=1,
                    )
                conv_log = gr.Textbox(
                    label="Log", lines=10, max_lines=10,
                    interactive=False, autoscroll=False, elem_id="conv-log",
                )

            # ── Fix Pad Tokens ─────────────────────────────────────────────
            with gr.Tab("Fix Pad Tokens"):
                gr.Markdown(
                    "Correct `x_pad_token` / `cap_pad_token` shape `[D]` → `[1, D]` in an "
                    "existing **Lumina 2** GGUF.  Required when ComfyUI raises "
                    "*size mismatch for x_pad_token*.  New conversions are not affected."
                )
                with gr.Group(elem_classes=["card"]):
                    with gr.Row():
                        fix_pad_src = gr.Textbox(
                            label="Source GGUF",
                            placeholder="model.gguf",
                            scale=9,
                        )
                        browse_fix_pad_btn = gr.Button(
                            "Browse", size="sm", scale=0, elem_classes=["browse-btn"]
                        )
                    fix_pad_dst = gr.Textbox(
                        label="Output path",
                        placeholder="Auto-generated (source-fixed.gguf)",
                    )
                    overwrite_fix_pad = gr.Checkbox(label="Overwrite existing output", value=False)

                fix_pad_btn = gr.Button("▶  Fix Pad Tokens", variant="primary", elem_id="fix-pad-btn")

                with gr.Group(elem_id="status-wrap"):
                    fix_pad_status = gr.Textbox(
                        value="Ready", show_label=False, interactive=False,
                        lines=1, max_lines=1,
                    )
                pad_log = gr.Textbox(
                    label="Log", lines=8, max_lines=8,
                    interactive=False, autoscroll=False, elem_id="pad-log",
                )

            # ── Fix 5D Tensors ─────────────────────────────────────────────
            with gr.Tab("Fix 5D Tensors"):
                gr.Markdown(
                    "Re-insert 5D tensors into a quantized GGUF.  "
                    "**Required for HunyuanVideo / Wan** when using llama-quantize outside "
                    "the Convert tab — the Convert tab chains this step automatically."
                )
                with gr.Group(elem_classes=["card"]):
                    with gr.Row():
                        fix_src = gr.Textbox(
                            label="Source GGUF (quantized)",
                            placeholder="model-Q8_0.gguf",
                            scale=9,
                        )
                        browse_fix_btn = gr.Button(
                            "Browse", size="sm", scale=0, elem_classes=["browse-btn"]
                        )
                    fix_dst = gr.Textbox(
                        label="Output path",
                        placeholder="Auto-generated (source-fixed.gguf)",
                    )
                    fix_file = gr.Textbox(
                        label="Side-car file",
                        placeholder="fix_5d_tensors_<arch>.safetensors — auto-detected when empty",
                    )
                    overwrite_fix = gr.Checkbox(label="Overwrite existing output", value=False)

                fix_btn = gr.Button("▶  Fix 5D Tensors", variant="primary", elem_id="fix-5d-btn")

                with gr.Group(elem_id="status-wrap"):
                    fix_status = gr.Textbox(
                        value="Ready", show_label=False, interactive=False,
                        lines=1, max_lines=1,
                    )
                fix_log = gr.Textbox(
                    label="Log", lines=8, max_lines=8,
                    interactive=False, autoscroll=False, elem_id="fix-log",
                )

        # ── Events ─────────────────────────────────────────────────────────
        browse_conv_btn.click(browse_model, outputs=src_path)
        browse_dst_btn.click(browse_and_set_dst, inputs=[src_path, quant_dropdown], outputs=dst_path)
        browse_fix_pad_btn.click(browse_gguf, outputs=fix_pad_src)
        browse_fix_btn.click(browse_gguf, outputs=fix_src)

        src_path.change(_auto_dst, inputs=src_path, outputs=dst_path)
        src_path.change(update_size_estimate, inputs=[src_path, quant_dropdown], outputs=size_info)
        quant_dropdown.change(update_size_estimate, inputs=[src_path, quant_dropdown], outputs=size_info)

        conv_event = convert_btn.click(
            fn=run_convert,
            inputs=[src_path, dst_path, quant_dropdown, exe_path, keep_intermediate, overwrite_conv],
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
