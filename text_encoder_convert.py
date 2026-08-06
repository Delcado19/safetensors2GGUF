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

LLAMA_CPP_HOME_ENV = "S2G_LLAMA_CPP_HOME"
LLAMA_CPP_REPO_URL = "https://github.com/ggml-org/llama.cpp"
DEFAULT_LLAMA_CPP_CACHE = Path(__file__).resolve().parent / ".llama.cpp"

TEXT_ENCODER_OUTTYPES: list[tuple[str, str]] = [
    ("F32", "f32"),
    ("F16", "f16"),
    ("BF16", "bf16"),
    ("Q8_0", "q8_0"),
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
