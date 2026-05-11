"""Subprocess wrapper for llama-quantize and quantization-type registry.

Defines which types can be produced natively by the Python gguf library
(PYTHON_PRECISIONS) and which require the external llama-quantize binary
(LLAMA_QUANT_KEYS).  The combined ALL_QUANT_CHOICES list is consumed by the
GUI to populate the quantization dropdown.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import gguf

from convert import ConversionCancelled

# ──────────────────────────────────────────────────────────────────────────────
# Type registry
# ──────────────────────────────────────────────────────────────────────────────

PYTHON_PRECISIONS: dict[str, tuple[gguf.GGMLQuantizationType, gguf.LlamaFileType]] = {
    "F32":  (gguf.GGMLQuantizationType.F32,  gguf.LlamaFileType.ALL_F32),
    "F16":  (gguf.GGMLQuantizationType.F16,  gguf.LlamaFileType.MOSTLY_F16),
    "BF16": (gguf.GGMLQuantizationType.BF16, gguf.LlamaFileType.MOSTLY_BF16),
    "Q8_0": (gguf.GGMLQuantizationType.Q8_0, gguf.LlamaFileType.MOSTLY_Q8_0),
}

# Keys that require llama-quantize (verified against the binary's help output)
LLAMA_QUANT_KEYS: frozenset[str] = frozenset({
    "Q6_K",
    "Q5_K_M",
    "Q4_K_M", "Q4_K_S",
    "Q3_K_M",
    "Q2_K",
})

# Ordered choices for the UI dropdown: (display label, key)
# Covers the types that are practical for ComfyUI-GGUF diffusion models.
ALL_QUANT_CHOICES: list[tuple[str, str]] = [
    ("F32  — Full precision",                              "F32"),
    ("F16  — Half precision · standard",                   "F16"),
    ("BF16 — Brain float 16",                             "BF16"),
    ("Q8_0 — 8-bit · very high quality",                  "Q8_0"),
    ("Q6_K — 6-bit · very high quality  [lq]",            "Q6_K"),
    ("Q5_K_M — 5-bit · high quality  [lq]",               "Q5_K_M"),
    ("Q4_K_M — 4-bit · recommended ★  [lq]",              "Q4_K_M"),
    ("Q4_K_S — 4-bit small  [lq]",                        "Q4_K_S"),
    ("Q3_K_M — 3-bit · moderate quality  [lq]",           "Q3_K_M"),
    ("Q2_K  — 2-bit · smallest  [lq]",                    "Q2_K"),
]

# Approximate output-size ratio relative to an F16 source file.
# Derived from llama-quantize reference output for Llama-3-8B (F16 = 14.00 GB).
SIZE_RATIOS: dict[str, float] = {
    "F32":   2.00,
    "F16":   1.00,
    "BF16":  1.00,
    "Q8_0":  round(7.96  / 14.00, 3),  # 0.569
    "Q6_K":  round(6.14  / 14.00, 3),  # 0.439
    "Q5_K_M": round(5.33 / 14.00, 3),  # 0.381
    "Q4_K_M": round(4.58 / 14.00, 3),  # 0.327
    "Q4_K_S": round(4.37 / 14.00, 3),  # 0.312
    "Q3_K_M": round(3.74 / 14.00, 3),  # 0.267
    "Q2_K":  round(2.96  / 14.00, 3),  # 0.211
}

# ──────────────────────────────────────────────────────────────────────────────
# Binary discovery
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_EXE = Path(r"H:\ComfyUI-Easy-Install\Add-Ons\Tools\llama.cpp\llama-quantize.exe")

_PROGRESS_RE = re.compile(r"\[\s*(\d+)/\s*(\d+)\]")


def find_exe() -> Path | None:
    """Return DEFAULT_EXE if it exists on disk, else None."""
    return DEFAULT_EXE if DEFAULT_EXE.is_file() else None


# ──────────────────────────────────────────────────────────────────────────────
# Subprocess runner
# ──────────────────────────────────────────────────────────────────────────────

def run_quantize(
    src: str | Path,
    dst: str | Path,
    quant_type: str,
    exe: str | Path | None = None,
    on_progress=None,
    on_log=None,
    nthreads: int | None = None,
    cancel_event=None,
) -> None:
    """Run llama-quantize and stream its output through the callbacks.

    Args:
        src: Input GGUF (F16 or F32 recommended).
        dst: Output GGUF path.
        quant_type: Type string accepted by llama-quantize (e.g. "Q4_K_M").
        exe: Path to llama-quantize binary.  Auto-detected when None.
        on_progress: Callback(idx, total, desc) fired on each progress line.
        on_log: Callback(msg) fired for every output line.
        nthreads: Optional thread count forwarded to llama-quantize.
        cancel_event: Optional threading.Event; terminates the subprocess and raises
            ConversionCancelled when set.

    Raises:
        FileNotFoundError: Binary not found or path is wrong.
        RuntimeError: Process exited with non-zero return code.
    """
    if exe is None:
        exe = find_exe()
    if exe is None or not Path(exe).is_file():
        raise FileNotFoundError(
            "llama-quantize.exe not found — set the path in Advanced settings."
        )

    cmd = [str(exe), str(src), str(dst), quant_type]
    if nthreads:
        cmd.append(str(nthreads))

    def _emit(msg: str) -> None:
        if on_log:
            on_log(msg)

    _emit(f"INFO:  $ {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    for line in proc.stdout:
        if cancel_event is not None and cancel_event.is_set():
            proc.terminate()
            proc.wait()
            raise ConversionCancelled()

        line = line.rstrip()
        if not line:
            continue
        _emit(line)
        m = _PROGRESS_RE.search(line)
        if m and on_progress:
            on_progress(int(m.group(1)), int(m.group(2)), line[:72])

    proc.wait()
    if proc.returncode not in (0, -15):  # -15 = SIGTERM from terminate()
        raise RuntimeError(
            f"llama-quantize exited with code {proc.returncode}"
        )
