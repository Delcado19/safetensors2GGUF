"""Text-encoder (LLM/T5) checkpoint -> GGUF conversion.

Bare single-file text-encoder safetensors (as ComfyUI's models/text_encoders/
folder holds — no accompanying config.json/tokenizer files) cannot be
converted by llama.cpp's convert_hf_to_gguf.py directly: that script hard-
requires config.json + tokenizer files (see llama.cpp ModelBase.load_hparams
and its --remote fetch list). There is no tensor-shape-only fallback in
either llama.cpp or ComfyUI-GGUF's public tooling.

Workflow implemented here:
  1. User supplies the local weights file + a HuggingFace repo ID for the
     *base* model (manual field -- no auto-detection, see plan header).
  2. Download that base repo's config.json + tokenizer files via
     huggingface_hub.
  3. Assemble a temp directory: downloaded config/tokenizer + the local
     weights renamed to what convert_hf_to_gguf.py expects.
  4. Run convert_hf_to_gguf.py as a subprocess, using the ComfyUI-Easy-
     Install embedded Python interpreter (it already has transformers /
     torch / mistral_common installed; this repo's own venv does not need
     those heavy deps as a result).
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download

from quantize import _easy_install_roots

CONVERT_SCRIPT_RELATIVE = Path("python_embeded") / "Lib" / "site-packages" / "llama_cpp" / "bin" / "convert_hf_to_gguf.py"
EMBEDDED_PYTHON_RELATIVE = Path("python_embeded") / "python.exe"

TEXT_ENCODER_OUTTYPES: list[tuple[str, str]] = [
    ("F32", "f32"),
    ("F16", "f16"),
    ("BF16", "bf16"),
    ("Q8_0", "q8_0"),
]


def find_convert_script() -> Path | None:
    """Return the path to convert_hf_to_gguf.py under a discoverable
    ComfyUI-Easy-Install root, or None if not found."""
    for root in _easy_install_roots():
        candidate = root / CONVERT_SCRIPT_RELATIVE
        if candidate.is_file():
            return candidate
    return None


def find_embedded_python() -> Path | None:
    """Return the path to the ComfyUI-Easy-Install embedded python.exe,
    or None if not found."""
    for root in _easy_install_roots():
        candidate = root / EMBEDDED_PYTHON_RELATIVE
        if candidate.is_file():
            return candidate
    return None


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
