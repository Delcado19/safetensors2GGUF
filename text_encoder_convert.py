"""Text-encoder (LLM/T5) checkpoint -> GGUF conversion.

Bare single-file text-encoder safetensors (as ComfyUI's models/text_encoders/
folder holds — no accompanying config.json/tokenizer files) cannot be
converted by llama.cpp's convert_hf_to_gguf.py directly: that script hard-
requires config.json + tokenizer files (see llama.cpp ModelBase.load_hparams
and its --remote fetch list). There is no tensor-shape-only fallback in
either llama.cpp or ComfyUI-GGUF's public tooling.

Workflow implemented here:
  1. User supplies the local weights file + a HuggingFace repo ID for the
     *base* model (manual field -- no auto-detection).
  2. Download that base repo's config.json + tokenizer files via
     huggingface_hub.
  3. Assemble a temp directory: downloaded config/tokenizer + the local
     weights renamed to what convert_hf_to_gguf.py expects.
  4. Run convert_hf_to_gguf.py as a subprocess of THIS tool's own Python
     interpreter (sys.executable) -- no ComfyUI installation required.
     llama.cpp itself is auto-cloned into a local cache directory on first
     use (its convert_hf_to_gguf.py imports from a sibling `conversion/`
     package, so it can't be vendored as a single file without also
     vendoring and tracking ~90 per-architecture modules; a shallow git
     clone stays in sync with upstream for free). transformers/
     sentencepiece/protobuf are declared as this project's own dependencies
     (pyproject.toml) so the clone's script runs in our venv.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from huggingface_hub import hf_hub_download

from convert_safetensors import convert_to_safetensors
from models.architectures import ModelTemplate
from quantize import LLAMA_QUANT_KEYS, run_quantize

LLAMA_CPP_HOME_ENV = "S2G_LLAMA_CPP_HOME"
LLAMA_CPP_REPO_URL = "https://github.com/ggml-org/llama.cpp"
DEFAULT_LLAMA_CPP_CACHE = Path(__file__).resolve().parent / ".llama.cpp"

# GGUF outtype strings convert_hf_to_gguf.py accepts directly via --outtype.
_GGUF_DIRECT_OUTTYPES: dict[str, str] = {
    "F32": "f32", "F16": "f16", "BF16": "bf16", "Q8_0": "q8_0",
}

# Safetensors-quant formats — see safetensors_quant.SAFETENSORS_DTYPE_CHOICES.
# No llama.cpp / HuggingFace download involved for these at all.
TEXT_ENCODER_SAFETENSORS_FORMATS: frozenset[str] = frozenset({"FP8", "FP8_MIXED", "NVFP4", "NVFP4_MIXED"})

TEXT_ENCODER_OUTTYPES: list[tuple[str, str]] = [
    ("F32", "f32"),
    ("F16", "f16"),
    ("BF16", "bf16"),
    ("Q8_0", "q8_0"),
]

# Full format dropdown for the GUI: GGUF direct outtypes, GGUF K-quants (via a
# plain llama-quantize second pass — see ensure_plain_llama_quantize), and
# safetensors-quant formats (FP8/NVFP4, no HF download needed).
TEXT_ENCODER_FORMAT_CHOICES: list[tuple[str, str]] = [
    ("F32  — Full precision",                   "F32"),
    ("F16  — Half precision · standard",        "F16"),
    ("BF16 — Brain float 16",                  "BF16"),
    ("Q8_0 — 8-bit · very high quality",       "Q8_0"),
    ("Q6_K — 6-bit · very high quality  [lq]", "Q6_K"),
    ("Q5_K_M — 5-bit · high quality  [lq]",    "Q5_K_M"),
    ("Q4_K_M — 4-bit · recommended ★  [lq]",   "Q4_K_M"),
    ("Q4_K_S — 4-bit small  [lq]",             "Q4_K_S"),
    ("Q3_K_M — 3-bit · moderate quality  [lq]", "Q3_K_M"),
    ("Q2_K  — 2-bit · smallest  [lq]",         "Q2_K"),
    ("FP8 — float8_e4m3fn scaled (safetensors)",             "FP8"),
    ("FP8 mixed — FP8 scaled, hiprec tensors stay F32",       "FP8_MIXED"),
    ("NVFP4 — Nvidia 4-bit blockscaled (safetensors)",        "NVFP4"),
    ("NVFP4 mixed — NVFP4, hiprec tensors stay F32",          "NVFP4_MIXED"),
]


def _llama_cpp_dir() -> Path:
    """Return where the llama.cpp clone lives (override via S2G_LLAMA_CPP_HOME)."""
    env_dir = os.environ.get(LLAMA_CPP_HOME_ENV)
    return Path(env_dir) if env_dir else DEFAULT_LLAMA_CPP_CACHE


def ensure_llama_cpp(on_log=None) -> Path:
    """Return the llama.cpp checkout directory, cloning it on first use.

    Raises RuntimeError if git isn't available or the clone fails.
    """
    def _log(msg):
        if on_log:
            on_log(msg)

    clone_dir = _llama_cpp_dir()
    if (clone_dir / "convert_hf_to_gguf.py").is_file():
        return clone_dir

    _log(f"INFO:  Cloning llama.cpp into {clone_dir} (first run only)…")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", LLAMA_CPP_REPO_URL, str(clone_dir)],
            check=True, capture_output=True, text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "git is required to fetch llama.cpp's convert_hf_to_gguf.py but was not found on PATH"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Failed to clone llama.cpp: {exc.stderr or exc}") from exc

    return clone_dir


def find_convert_script() -> Path:
    """Return the path to convert_hf_to_gguf.py, cloning llama.cpp if needed."""
    return ensure_llama_cpp() / "convert_hf_to_gguf.py"


def _quantize_build_dir() -> Path:
    return _llama_cpp_dir() / "build-quantize"


def _find_built_quantize_binary(build_dir: Path) -> Path | None:
    for candidate in (
        build_dir / "bin" / "llama-quantize.exe",
        build_dir / "bin" / "llama-quantize",
        build_dir / "bin" / "Release" / "llama-quantize.exe",
        build_dir / "bin" / "Debug" / "llama-quantize.exe",
    ):
        if candidate.is_file():
            return candidate
    return None


def ensure_plain_llama_quantize(on_log=None) -> Path:
    """Return a plain (unpatched) llama-quantize binary, building it once from
    the auto-cloned llama.cpp checkout via cmake.

    Deliberately separate from the City96-patched llama-quantize used for
    diffusion-model GGUFs (see docs/building-llama-quantize.md) — that patch is
    documented as unsafe for LLM/text GGUFs (tuned for diffusion tensor
    layouts). Requires `cmake` and a C++ compiler toolchain on PATH; raises
    RuntimeError with guidance if either is missing or the build fails.
    """
    def _log(msg):
        if on_log:
            on_log(msg)

    build_dir = _quantize_build_dir()
    existing = _find_built_quantize_binary(build_dir)
    if existing is not None:
        return existing

    llama_cpp_dir = ensure_llama_cpp(on_log=on_log)

    _log("INFO:  Building plain llama-quantize from source (first run only, needs cmake + a C++ compiler)…")
    try:
        subprocess.run(
            ["cmake", "-B", str(build_dir), "-S", str(llama_cpp_dir)],
            check=True, capture_output=True, text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "cmake (and a C++ compiler) are required to build a plain llama-quantize "
            "for text-encoder K-quants but cmake was not found on PATH"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Failed to configure llama.cpp build: {exc.stderr or exc}") from exc

    proc = subprocess.Popen(
        ["cmake", "--build", str(build_dir), "--config", "Release", "--target", "llama-quantize"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    for line in proc.stdout:
        _log(line.rstrip())
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"cmake --build (llama-quantize) exited with code {proc.returncode}")

    binary = _find_built_quantize_binary(build_dir)
    if binary is None:
        raise RuntimeError(f"llama-quantize build finished but the binary was not found under {build_dir}")
    return binary


_MANDATORY_FILES = ("config.json",)
_OPTIONAL_TOKENIZER_FILES = (
    "tokenizer.json", "tokenizer_config.json", "tokenizer.model", "special_tokens_map.json",
)


def fetch_base_config_files(repo_id: str, dest_dir: Path, on_log=None) -> list[str]:
    """Download config.json + whichever tokenizer files exist for repo_id into dest_dir.

    config.json is mandatory (RuntimeError if missing); tokenizer files are
    best-effort since repos vary in which ones they ship.
    """
    def _log(msg):
        if on_log:
            on_log(msg)

    downloaded: list[str] = []
    for filename in _MANDATORY_FILES:
        try:
            hf_hub_download(repo_id, filename, local_dir=str(dest_dir))
            downloaded.append(filename)
        except Exception as exc:
            raise RuntimeError(f"Required file {filename!r} not found in {repo_id!r}: {exc}") from exc

    for filename in _OPTIONAL_TOKENIZER_FILES:
        try:
            hf_hub_download(repo_id, filename, local_dir=str(dest_dir))
            downloaded.append(filename)
            _log(f"INFO:  Downloaded {filename}")
        except Exception:
            continue  # optional — not every repo ships every tokenizer variant

    return downloaded


def convert_text_encoder(
    weights_path: str,
    base_repo_id: str,
    dst_path: str | None = None,
    outtype: str = "f16",
    on_log=None,
    cancel_event=None,
) -> str:
    """Convert a bare single-file text-encoder checkpoint to GGUF.

    Downloads config.json/tokenizer files for base_repo_id, assembles a temp
    HF-style model directory with the local weights, then runs
    convert_hf_to_gguf.py (auto-cloned from llama.cpp) with this tool's own
    Python interpreter.
    """
    def _log(msg):
        if on_log:
            on_log(msg)
        else:
            print(msg)

    script = find_convert_script()

    if dst_path is None:
        dst_path = f"{weights_path.rsplit('.', 1)[0]}-{outtype}.gguf"

    with tempfile.TemporaryDirectory(prefix="s2g_text_encoder_") as tmpdir:
        tmp_path = Path(tmpdir)
        _log(f"INFO:  Fetching config/tokenizer for {base_repo_id}…")
        fetch_base_config_files(base_repo_id, tmp_path, on_log=_log)

        weights_dst = tmp_path / "model.safetensors"
        shutil.copy2(weights_path, weights_dst)

        cmd = [
            sys.executable, str(script), str(tmp_path),
            "--outfile", dst_path,
            "--outtype", outtype,
        ]
        _log(f"INFO:  $ {' '.join(cmd)}")

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        for line in proc.stdout:
            if cancel_event is not None and cancel_event.is_set():
                proc.terminate()
                proc.wait()
                raise RuntimeError("cancelled")
            _log(line.rstrip())
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"convert_hf_to_gguf.py exited with code {proc.returncode}")

    return dst_path


def convert_text_encoder_kquant(
    weights_path: str,
    base_repo_id: str,
    dst_path: str | None = None,
    quant_key: str = "Q4_K_M",
    on_log=None,
    cancel_event=None,
) -> str:
    """Convert a text-encoder checkpoint to a K-quant GGUF (e.g. Q4_K_M).

    Two-pass: convert_text_encoder() produces an F16 intermediate, then a
    plain (unpatched) llama-quantize — built via ensure_plain_llama_quantize,
    never the City96-patched diffusion-model binary — quantizes it to
    quant_key. The F16 intermediate is deleted afterward.
    """
    def _log(msg):
        if on_log:
            on_log(msg)
        else:
            print(msg)

    if dst_path is None:
        dst_path = f"{weights_path.rsplit('.', 1)[0]}-{quant_key}.gguf"

    with tempfile.TemporaryDirectory(prefix="s2g_te_kquant_") as tmpdir:
        intermediate = str(Path(tmpdir) / "intermediate-f16.gguf")
        convert_text_encoder(
            weights_path, base_repo_id, dst_path=intermediate, outtype="f16",
            on_log=on_log, cancel_event=cancel_event,
        )

        exe = ensure_plain_llama_quantize(on_log=on_log)
        _log(f"INFO:  Quantizing to {quant_key} with plain llama-quantize…")
        run_quantize(
            intermediate, dst_path, quant_key, exe=exe,
            on_log=on_log, cancel_event=cancel_event,
        )

    return dst_path


def convert_text_encoder_to_safetensors(
    weights_path: str,
    dst_path: str | None = None,
    target_key: str = "FP8",
    on_log=None,
    cancel_event=None,
) -> str:
    """Convert a text-encoder checkpoint to a quantized .safetensors file
    (FP8/FP8_MIXED/NVFP4/NVFP4_MIXED).

    No HuggingFace download or llama.cpp involved — ComfyUI's text-encoder
    loaders build models from fixed config presets rather than inferring
    hyperparameters from checkpoint tensor shapes (unlike the diffusion-model
    DiT architectures), so the generic ModelTemplate() safety rules (1D-skip,
    16-multiple fallback) are sufficient; no per-architecture keys_hiprec/
    keys_shape_critical list is needed here.
    """
    dst, _ = convert_to_safetensors(
        weights_path, dst_path=dst_path, target_key=target_key, overwrite=True,
        on_log=on_log, cancel_event=cancel_event, model_arch=ModelTemplate(),
    )
    return dst


def convert_text_encoder_any(
    weights_path: str,
    base_repo_id: str,
    dst_path: str | None,
    format_key: str,
    on_log=None,
    cancel_event=None,
) -> str:
    """Dispatch to the right text-encoder conversion backend for format_key.

    format_key is one of TEXT_ENCODER_FORMAT_CHOICES' keys: a GGUF direct
    outtype (F32/F16/BF16/Q8_0), a GGUF K-quant (Q6_K..Q2_K, LLAMA_QUANT_KEYS),
    or a safetensors-quant format (FP8/FP8_MIXED/NVFP4/NVFP4_MIXED — base_repo_id
    is ignored for these, no HF download needed).
    """
    if format_key in TEXT_ENCODER_SAFETENSORS_FORMATS:
        return convert_text_encoder_to_safetensors(
            weights_path, dst_path=dst_path, target_key=format_key,
            on_log=on_log, cancel_event=cancel_event,
        )
    if format_key in LLAMA_QUANT_KEYS:
        return convert_text_encoder_kquant(
            weights_path, base_repo_id, dst_path=dst_path, quant_key=format_key,
            on_log=on_log, cancel_event=cancel_event,
        )
    outtype = _GGUF_DIRECT_OUTTYPES.get(format_key)
    if outtype is None:
        raise ValueError(f"Unknown text-encoder format: {format_key!r}")
    return convert_text_encoder(
        weights_path, base_repo_id, dst_path=dst_path, outtype=outtype,
        on_log=on_log, cancel_event=cancel_event,
    )
