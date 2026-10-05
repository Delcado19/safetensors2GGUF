"""Frontend-independent output resolution and GGUF conversion pipeline."""
from __future__ import annotations

import queue
import threading
import uuid
from pathlib import Path

from convert import convert_file
from conversion_io import validate_output
from fix_5d_tensors import fix_5d_tensors as _fix_5d
from fix_pad_tokens import fix_pad_tokens as _fix_pad
from quantize import LLAMA_QUANT_KEYS, PYTHON_PRECISIONS, run_quantize
from safetensors_quant import filename_suffix_for

GUI_TENSOR_LOG_EVERY = 25

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
    """Resolve the user-supplied Safetensors output path; mirrors _resolve_dst.

    The filename suffix comes from filename_suffix_for(), not target_key
    directly, so e.g. FP8 files are named with the external
    "fp8_e4m3fn_scaled" convention a Civitai/ComfyUI user would recognize —
    see safetensors_quant.py's _FILENAME_SUFFIX comment.
    """
    if not dst:
        return None
    dst = dst.strip()
    if not dst:
        return None
    suffix = filename_suffix_for(target_key)

    if dst.endswith(("/", "\\")) or Path(dst).is_dir():
        stem = _strip_model_suffix(src)
        return str(Path(dst) / f"{stem.name}-{suffix}.safetensors")

    if "{ftype}" in dst:
        return dst.replace("{ftype}", suffix)

    return dst


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

    # Check the final output before doing expensive work; honor the GUI checkbox.
    final_dst = _resolve_dst(src, dst_raw, quant_key) or str(_strip_model_suffix(src)) + f"-{quant_key}.gguf"
    validate_output(src, final_dst, overwrite)
    is_kquant = quant_key in LLAMA_QUANT_KEYS
    # Total-step count for k-quants is 2 normally, 3 if a 5D side-car exists.
    # We only know needs_fix after step 1, so step 1 shows "1/2+" as a hint.
    step1_scale = 0.45 if is_kquant else 1.0
    step1_label = "[1/2+]" if is_kquant else "[1/1]"

    # ── Step 1: Python conversion ────────────────────────────────────────────
    _log(f"INFO:  {step1_label} Converting to {'F16 GGUF…' if is_kquant else quant_key + ' GGUF…'}")

    if is_kquant:
        target_qt, _ = PYTHON_PRECISIONS["F16"]
        # Unique intermediate and side-car names isolate concurrent conversions.
        intermediate = str(Path(final_dst).parent / (
            _strip_model_suffix(src).name + f"-F16-tmp-{uuid.uuid4().hex}.gguf"
        ))
        step1_dst: str | None = intermediate
    else:
        target_qt, _ = PYTHON_PRECISIONS[quant_key]
        intermediate = None
        step1_dst = final_dst

    def _prog1(idx, total, key):
        _frac((idx / total * step1_scale) if total else 0.0, f"{step1_label} Tensor {idx}/{total}")

    _, arch = convert_file(
        src, step1_dst,
        interact=False, overwrite=False if is_kquant else overwrite,
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
        fix_path = Path(arch.fix_path)
        if getattr(arch, "_nd_tensors", None):
            _log(
                f"INFO:  5D tensor side-car found: {fix_path}. "
                "Use the 'Fix 5D Tensors' tab to insert them after llama-quantize."
            )
        return final_dst

    # ── Step 2: llama-quantize ───────────────────────────────────────────────
    fix_path = Path(arch.fix_path)
    needs_fix = bool(getattr(arch, "_nd_tensors", None))
    # llama-quantize collapses [1, D] pad token shapes back to [D]; re-apply fix afterwards
    needs_pad_fix = bool(getattr(arch, 'keys_unsqueeze', None))
    total_steps = 2 + (1 if needs_fix else 0) + (1 if needs_pad_fix else 0)
    step2_end = 0.85 if needs_fix or needs_pad_fix else 0.95
    if needs_fix:
        fixed_dst = str(Path(final_dst).with_suffix("")) + "-fixed.gguf"
        validate_output(final_dst, fixed_dst, overwrite)

    _log(f"INFO:  [2/{total_steps}] Quantizing to {quant_key} via llama-quantize…")

    def _prog2(idx, total, key):
        _frac(0.45 + (idx / total * (step2_end - 0.45)) if total else 0.45, f"[2/{total_steps}] {idx}/{total}")

    run_quantize(
        intermediate, final_dst, quant_key,
        exe=exe, on_progress=_prog2, on_log=_log,
        nthreads=nthreads, cancel_event=cancel_event, overwrite=overwrite,
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
        fixed_dst = str(Path(result_path).with_suffix("")) + "-fixed.gguf"

        def _prog3(idx, total, key):
            _frac(step2_end + (idx / total * (step3_end - step2_end)) if total else step2_end,
                  f"[{current_step}/{total_steps}] Tensor {idx}/{total}")

        _fix_5d(result_path, fixed_dst, fix_path=str(fix_path), overwrite=overwrite, on_progress=_prog3, on_log=_log, cancel_event=cancel_event)
        _log(f"INFO:  5D tensors inserted → {fixed_dst}")
        result_path = fixed_dst
        current_step += 1

    # ── Step N (optional): re-fix pad token shapes collapsed by llama-quantize
    if needs_pad_fix:
        _log(f"INFO:  [{current_step}/{total_steps}] Re-fixing pad token shapes ([1, D] collapsed by llama-quantize)…")
        padfix_tmp = result_path + ".padfix.tmp"
        _fix_pad(result_path, padfix_tmp, overwrite=True, on_log=_log, cancel_event=cancel_event)
        Path(padfix_tmp).replace(Path(result_path))

    return result_path
