"""Subprocess wrapper for llama-quantize and quantization-type registry.

Defines which types can be produced natively by the Python gguf library
(PYTHON_PRECISIONS) and which require the external llama-quantize binary
(LLAMA_QUANT_KEYS).  The combined ALL_QUANT_CHOICES list is consumed by the
GUI to populate the quantization dropdown.
"""

from __future__ import annotations

import json
import os
import re
import struct
import subprocess
import tempfile
import time
from dataclasses import dataclass
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
# Size-savings percentages are derived from SIZE_RATIOS just above (1 -
# ratio, rounded to the nearest percent) so the two never drift apart.
ALL_QUANT_CHOICES: list[tuple[str, str]] = [
    ("F32  — Full precision · 2× F16 size",                                  "F32"),
    ("F16  — Half precision · standard",                                     "F16"),
    ("BF16 — Brain float 16 · same size as F16",                             "BF16"),
    ("Q8_0 — 8-bit · very high quality · 43% smaller than F16",              "Q8_0"),
    ("Q6_K — 6-bit · very high quality · 56% smaller than F16  [lq]",        "Q6_K"),
    ("Q5_K_M — 5-bit · high quality · 62% smaller than F16  [lq]",           "Q5_K_M"),
    ("Q4_K_M — 4-bit · recommended ★ · 67% smaller than F16  [lq]",          "Q4_K_M"),
    ("Q4_K_S — 4-bit small · 69% smaller than F16  [lq]",                    "Q4_K_S"),
    ("Q3_K_M — 3-bit · moderate quality · 73% smaller than F16  [lq]",       "Q3_K_M"),
    ("Q2_K  — 2-bit · smallest · 79% smaller than F16  [lq]",                "Q2_K"),
]

# ── File-level size ratios (used as fallback for non-safetensors sources) ──────
# Ratio of output file size to F16 source file size.
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

# ── Per-element bytes for *quantizable* tensors ────────────────────────────────
# These apply only to tensors that actually get quantized (2D+, > 1024 elements,
# not in keys_hiprec).  Derived from the same Llama-3-8B reference run:
#   bytes_per_elem = (output_GB / 14.0) * 2   [since F16 = 2 bytes/elem]
_QUANT_BYTES_PER_ELEM: dict[str, float] = {
    "F32":   4.000,
    "F16":   2.000,
    "BF16":  2.000,
    "Q8_0":  round(7.96 / 14.0 * 2, 4),   # 1.1371
    "Q6_K":  round(6.14 / 14.0 * 2, 4),   # 0.8771
    "Q5_K_M": round(5.33 / 14.0 * 2, 4),  # 0.7614
    "Q4_K_M": round(4.58 / 14.0 * 2, 4),  # 0.6543
    "Q4_K_S": round(4.37 / 14.0 * 2, 4),  # 0.6243
    "Q3_K_M": round(3.74 / 14.0 * 2, 4),  # 0.5343
    "Q2_K":  round(2.96 / 14.0 * 2, 4),   # 0.4229
}

# Safetensors dtype strings that map to high-precision PyTorch dtypes
# (torch.float32 / torch.bfloat16).  Only for these does convert.py force
# 1D tensors and small tensors to F32.
_HIGH_PREC_DTYPES = {"F32", "BF16"}

# Must match convert.QUANTIZATION_THRESHOLD (kept in sync manually)
_QUANTIZATION_THRESHOLD = 1024


def _read_safetensors_header(path: str) -> dict:
    """Return the metadata dict from a safetensors file header.

    Reads only the header bytes — no tensor data is loaded.

    Raises ValueError or OSError on parse errors.
    """
    with open(path, "rb") as fh:
        raw_len = fh.read(8)
        if len(raw_len) < 8:
            raise ValueError("File too short for safetensors header")
        header_len = struct.unpack("<Q", raw_len)[0]
        if header_len > 200_000_000:
            raise ValueError(f"Header length {header_len} suspiciously large")
        return json.loads(fh.read(header_len).decode("utf-8", errors="replace"))


def _estimate_from_safetensors(path: str, quant_key: str) -> int:
    """Estimate GGUF output size by analysing the safetensors header.

    Separates F32-forced tensors (1D, ≤1024 elements, >4D, or BF16/F32 source
    that is 1D/small) from quantizable tensors and applies per-element byte
    counts for each group.
    """
    header = _read_safetensors_header(path)
    bpe = _QUANT_BYTES_PER_ELEM.get(quant_key, 2.0)

    f32_elems = 0
    quant_elems = 0

    for name, meta in header.items():
        if name == "__metadata__":
            continue
        shape = meta.get("shape")
        if not shape and shape != []:
            continue
        n_elems = 1
        for d in shape:
            n_elems *= d
        ndim = len(shape)
        high_prec = meta.get("dtype", "F16") in _HIGH_PREC_DTYPES

        if ndim > 4:
            # 5D+: exported to side-car, re-inserted as F32
            f32_elems += n_elems
        elif high_prec and (ndim == 1 or n_elems <= _QUANTIZATION_THRESHOLD):
            # BF16/F32 source + 1D or small → always stored as F32
            f32_elems += n_elems
        else:
            quant_elems += n_elems

    if quant_key == "F32":
        return int((f32_elems + quant_elems) * 4)

    f32_bytes = f32_elems * 4
    quant_bytes = int(quant_elems * bpe)
    return f32_bytes + quant_bytes


def estimate_output_size(src_path: str, quant_key: str) -> int | None:
    """Estimate GGUF output size in bytes.

    For .safetensors sources the safetensors header is parsed (no tensor data
    loaded) and each tensor's output format is simulated.  This gives accurate
    results even for models with a high proportion of F32-forced tensors (bias,
    norm, positional embeddings) such as diffusion models.

    For all other source formats the estimate falls back to
    ``source_size × SIZE_RATIOS[quant_key]``.

    Returns None when the source file does not exist or cannot be read.
    """
    src_path = (src_path or "").strip()
    if not src_path or not os.path.isfile(src_path):
        return None

    if src_path.lower().endswith(".safetensors"):
        try:
            return _estimate_from_safetensors(src_path, quant_key)
        except Exception:
            pass  # fall through to ratio method

    ratio = SIZE_RATIOS.get(quant_key, 1.0)
    return int(os.path.getsize(src_path) * ratio)


# ──────────────────────────────────────────────────────────────────────────────
# Binary discovery
# ──────────────────────────────────────────────────────────────────────────────

EASY_INSTALL_ROOT_ENV = "COMFYUI_EASY_INSTALL_HOME"
EASY_INSTALL_QUANTIZE_RELATIVE = Path("Add-Ons") / "Tools" / "llama.cpp" / "llama-quantize.exe"
DEFAULT_EASY_INSTALL_ROOT = Path(r"H:\ComfyUI-Easy-Install")
EASY_INSTALL_DRIVE_ROOTS = tuple(Path(f"{drive}:\\ComfyUI-Easy-Install") for drive in "CDEFGHIJKLMNOPQRSTUVWXYZ")
DEFAULT_EXE = DEFAULT_EASY_INSTALL_ROOT / EASY_INSTALL_QUANTIZE_RELATIVE

_PROGRESS_RE = re.compile(r"\[\s*(\d+)/\s*(\d+)\]")


@dataclass(frozen=True)
class QuantizeBenchmarkResult:
    """Timing result for one llama-quantize binary."""

    exe: str
    quant_type: str
    seconds: float
    output_size: int
    error: str | None = None


def _easy_install_roots() -> list[Path]:
    """Return conservative ComfyUI Easy-Install roots to probe."""
    roots: list[Path] = []
    env_root = os.environ.get(EASY_INSTALL_ROOT_ENV)
    if env_root:
        roots.append(Path(env_root))

    roots.append(DEFAULT_EASY_INSTALL_ROOT)

    roots.extend(EASY_INSTALL_DRIVE_ROOTS)

    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root).lower()
        if key not in seen:
            unique.append(root)
            seen.add(key)
    return unique


def _find_easy_install_quantize() -> Path | None:
    """Return the Easy-Install bundled llama-quantize.exe when available."""
    for root in _easy_install_roots():
        candidate = root / EASY_INSTALL_QUANTIZE_RELATIVE
        if candidate.is_file():
            return candidate
    return None


def find_exe() -> Path | None:
    """Return a patched llama-quantize binary path, if one is discoverable.

    Discovery is intentionally conservative.  Upstream llama.cpp release
    binaries are not auto-selected because ComfyUI-GGUF image models require the
    City96 patch.  Windows prefers the Easy-Install bundled binary; non-Windows
    users can provide a locally built City96-patched binary through PATH,
    LLAMA_QUANTIZE_PATH, or the GUI file picker.
    """
    env_path = os.environ.get("LLAMA_QUANTIZE_PATH")
    if env_path and Path(env_path).is_file():
        return Path(env_path)

    easy_install_exe = _find_easy_install_quantize()
    if easy_install_exe is not None:
        return easy_install_exe

    import shutil

    for name in ("llama-quantize", "llama-quantize.exe"):
        found = shutil.which(name)
        if found:
            return Path(found)
    return None


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


def benchmark_quantize_binary(
    exe: str | Path,
    src: str | Path,
    quant_type: str,
    nthreads: int | None = None,
    work_dir: str | Path | None = None,
    on_log=None,
) -> QuantizeBenchmarkResult:
    """Benchmark one llama-quantize binary against a source GGUF.

    The benchmark writes a temporary output GGUF and removes it after timing.
    Use an F16/BF16/F32/Q8_0 source GGUF for meaningful results.
    """
    exe = Path(exe)
    src = Path(src)
    if not src.is_file():
        raise FileNotFoundError(f"Source GGUF not found: {src}")
    if not exe.is_file():
        raise FileNotFoundError(f"llama-quantize binary not found: {exe}")

    tmp_parent = Path(work_dir) if work_dir else src.parent
    tmp_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="llama-quantize-bench-", dir=tmp_parent) as tmp:
        dst = Path(tmp) / f"{src.stem}-{quant_type}.gguf"
        start = time.perf_counter()
        try:
            run_quantize(src, dst, quant_type, exe=exe, nthreads=nthreads, on_log=on_log)
            error = None
        except Exception as exc:
            error = str(exc)
        seconds = time.perf_counter() - start
        output_size = dst.stat().st_size if dst.exists() else 0

    return QuantizeBenchmarkResult(
        exe=str(exe),
        quant_type=quant_type,
        seconds=seconds,
        output_size=output_size,
        error=error,
    )


def benchmark_quantize_binaries(
    exes: list[str | Path],
    src: str | Path,
    quant_type: str,
    nthreads: int | None = None,
    work_dir: str | Path | None = None,
    on_log=None,
) -> list[QuantizeBenchmarkResult]:
    """Benchmark multiple llama-quantize binaries on the same source GGUF."""
    return [
        benchmark_quantize_binary(
            exe, src, quant_type,
            nthreads=nthreads, work_dir=work_dir, on_log=on_log,
        )
        for exe in exes
    ]
