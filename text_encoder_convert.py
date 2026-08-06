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
