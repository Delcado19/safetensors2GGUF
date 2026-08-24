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
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from huggingface_hub import hf_hub_download

from convert import load_state_dict
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
# No llama.cpp / HuggingFace download involved for these at all. INT8/
# INT8_MIXED added 2026-08-13: convert_text_encoder_to_safetensors() already
# delegates to the same quantize_tensor_st() machinery that implements
# int8_tensorwise/ConvRot for diffusion models (_TARGET_TO_QUANT_FORMAT in
# convert_safetensors.py already mapped these keys) -- this dropdown simply
# never listed them. ConvRot's block-Hadamard rotation only depends on a
# tensor's in_features being divisible by CONVROT_GROUP_SIZE, not on
# architecture, and text encoders carry no keys_hiprec (only
# keys_shape_critical), so INT8 and INT8_MIXED produce identical output --
# same rule already established for keys_hiprec-less diffusion architectures
# in model_support.support_level(). Now render-tested (same day) and
# VERIFIED clean for both qwen3-8b and qwen3-4b -- IMPORTANT: this format
# only loads correctly through ComfyUI's native `CLIPLoader` node. Loading
# it through ComfyUI-GGUF's `CLIPLoaderGGUF` node instead (easy mistake --
# that's the node most Z-Image workflows already use for the GGUF text
# encoder slot) produces full-image structured-noise garbage: that node
# expects an actual .gguf file and has no ConvRot/INT8-safetensors decode
# path. See model_support._TE_RENDER_CONFIRMED_BAD's docstring for the
# render-test history that uncovered this.
TEXT_ENCODER_SAFETENSORS_FORMATS: frozenset[str] = frozenset(
    {"FP8", "FP8_MIXED", "INT8", "INT8_MIXED", "NVFP4", "NVFP4_MIXED", "F16_ST"}
)

# GUI-facing dropdown key -> real safetensors_quant.py target_key. Only needed
# for F16_ST: "F16" is already taken in TEXT_ENCODER_FORMAT_CHOICES by the
# GGUF outtype, so the safetensors variant needs its own dropdown key -- but
# quantize_tensor_st()/convert_safetensors.py's per-tensor log line and
# default output filename both print target_key verbatim, so passing "F16_ST"
# straight through would log "-> F16_ST" and write "...-F16_ST.safetensors",
# neither of which is a real safetensors_quant format. Translate to the real
# "F16" key -- deliberately NOT "F16_MIXED", and NOT interchangeable with it
# despite text encoders carrying no keys_hiprec: quantize_tensor_st's `mixed`
# branch keeps 1D tensors (biases, norm weights) at their ORIGINAL on-disk
# dtype unconditionally (is_hiprec_st's `n_dims == 1` check fires regardless
# of keys_hiprec), so F16_MIXED output would still carry BF16 bias/norm
# tensors through to the file -- plain "F16" casts every tensor including 1D
# ones to real float16. That distinction is exactly what this key exists to
# get right: this option was added because a BF16-native checkpoint's weights
# were numerically unstable in BF16 (see CHANGELOG), so leaving any BF16
# tensors behind via F16_MIXED would undermine the fix.
_TEXT_ENCODER_SAFETENSORS_TARGET_KEY: dict[str, str] = {"F16_ST": "F16"}

TEXT_ENCODER_OUTTYPES: list[tuple[str, str]] = [
    ("F32", "f32"),
    ("F16", "f16"),
    ("BF16", "bf16"),
    ("Q8_0", "q8_0"),
]

# Full format dropdown for the GUI: GGUF direct outtypes, GGUF K-quants (via a
# plain llama-quantize second pass — see ensure_plain_llama_quantize), and
# safetensors-quant formats (FP8/INT8/NVFP4, no HF download needed).
#
# Grouped GGUF-first, then safetensors — matching TEXT_ENCODER_SAFETENSORS_
# FORMATS/LLAMA_QUANT_KEYS/_GGUF_DIRECT_OUTTYPES' own dispatch split in
# convert_text_encoder_any() below, so this ordering IS the membership test,
# not just cosmetic. Gradio 6.x's Dropdown choices are a flat (label, value)
# list with no separator/disabled-option support (checked against the
# installed gradio's Dropdown.__init__ signature, 2026-08-14) — a fake
# separator entry would be selectable and error out downstream, so none is
# added here; GGUF keys stay uppercase and safetensors labels start lowercase
# as the grouping cue instead.
TEXT_ENCODER_FORMAT_CHOICES: list[tuple[str, str]] = [
    # ── GGUF ────────────────────────────────────────────────────────────
    ("F32 · full precision, 2× f16 size", "F32"),
    ("F16 · half precision, standard", "F16"),
    ("BF16 · brain float16", "BF16"),
    ("Q8_0 · 8-bit", "Q8_0"),
    ("Q6_K · 6-bit", "Q6_K"),
    ("Q5_K_M · 5-bit", "Q5_K_M"),
    ("Q4_K_M · 4-bit, recommended ★", "Q4_K_M"),
    ("Q4_K_S · 4-bit, small", "Q4_K_S"),
    ("Q3_K_M · 3-bit", "Q3_K_M"),
    ("Q2_K · 2-bit", "Q2_K"),
    # ── Safetensors ─────────────────────────────────────────────────────
    # "F16_ST" (not "F16" -- already taken by the GGUF outtype above, and not
    # "F16_MIXED" -- see _TEXT_ENCODER_SAFETENSORS_TARGET_KEY's docstring for
    # why that logged/named wrong) -- listed under TEXT_ENCODER_SAFETENSORS_
    # FORMATS above so it routes to convert_text_encoder_to_safetensors() and
    # writes a real .safetensors file, not a GGUF. Added 2026-08-14:
    # converting a BF16-native checkpoint to F16 safetensors (not GGUF) fixed
    # a real render-corruption bug where the checkpoint's weights were
    # numerically unstable under BF16 compute but stable under F16 -- see
    # CHANGELOG's "Fixed (root cause: user checkpoint, not this tool)".
    # Casts every tensor (including 1D biases/norms) to real float16 -- no
    # separate "F16 mixed" entry, unlike FP8/INT8/NVFP4 below: their _MIXED
    # variants exist to avoid quantization loss on hiprec/small tensors, but
    # F16 is a precision cast already (nothing lossy to protect against), and
    # a mixed variant here would leave some tensors at their original BF16 --
    # exactly the thing this option exists to eliminate (see
    # _TEXT_ENCODER_SAFETENSORS_TARGET_KEY's docstring above).
    ("f16 · half precision", "F16_ST"),
    ("fp8 · float8_e4m3fn scaled", "FP8"),
    ("fp8 mix · fp8, hiprec stays F32", "FP8_MIXED"),
    ("int8 · tensorwise, ConvRot-rotated where possible", "INT8"),
    ("int8 mix · int8/ConvRot, hiprec stays F32", "INT8_MIXED"),
    ("nvfp4 · NVIDIA FP4, needs Blackwell GPU", "NVFP4"),
    ("nvfp4 mix · nvfp4, hiprec stays F32, needs Blackwell GPU", "NVFP4_MIXED"),
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


def _locate_msvc_build_env(on_log=None) -> dict[str, str] | None:
    """On Windows, when cmake isn't already on PATH, locate an installed
    Visual Studio C++ toolchain via vswhere and return an environment dict
    with vcvarsall.bat's variables merged in (PATH, INCLUDE, LIB, ...) — the
    same trick setuptools/distutils use to invoke MSVC from a plain script.

    Handles the common case where Visual Studio (with the "Desktop
    development with C++" workload, which bundles its own cmake) is
    installed, but its tools were never added to the user's regular PATH —
    they're normally only available in a "Developer Command Prompt".

    Returns None (the caller then falls back to the inherited environment
    and cmake's own "not found" error) if this isn't Windows, cmake is
    already reachable, or no suitable Visual Studio installation is found.
    """
    if sys.platform != "win32" or shutil.which("cmake") is not None:
        return None

    def _log(msg):
        if on_log:
            on_log(msg)

    program_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    vswhere = Path(program_files_x86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.is_file():
        return None

    try:
        result = subprocess.run(
            [
                str(vswhere), "-latest", "-products", "*",
                "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property", "installationPath",
            ],
            capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    install_path = result.stdout.strip()
    if not install_path:
        return None

    vcvarsall = Path(install_path) / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"
    if not vcvarsall.is_file():
        return None

    _log(f"INFO:  cmake/a C++ compiler not on PATH — found Visual Studio at {install_path}, using it instead…")
    # Run via a temp .bat file rather than `cmd /c "call \"...\" x64 && set"`
    # directly: subprocess's list2cmdline re-quotes an argument that itself
    # contains spaces AND quotes (the vcvarsall path), mangling the nested
    # quotes so cmd.exe can't find the batch file at all (returns exit 1,
    # "command not found" for the literal quoted path). A batch file sidesteps
    # that double-quoting entirely — cmd.exe parses its own contents directly.
    try:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".bat", delete=False, encoding="utf-8"
        ) as bat_file:
            bat_file.write(f'@echo off\r\ncall "{vcvarsall}" x64\r\nset\r\n')
            bat_path = bat_file.name
        try:
            result = subprocess.run(
                ["cmd", "/c", bat_path], capture_output=True, text=True, check=True,
            )
        finally:
            os.unlink(bat_path)
    except (OSError, subprocess.CalledProcessError):
        return None

    env = dict(os.environ)
    for line in result.stdout.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            env[key] = value
    return env


def ensure_plain_llama_quantize(on_log=None) -> Path:
    """Return a plain (unpatched) llama-quantize binary, building it once from
    the auto-cloned llama.cpp checkout via cmake.

    Deliberately separate from the City96-patched llama-quantize used for
    diffusion-model GGUFs (see docs/building-llama-quantize.md) — that patch is
    documented as unsafe for LLM/text GGUFs (tuned for diffusion tensor
    layouts). Requires `cmake` and a C++ compiler toolchain — either already
    on PATH, or (Windows only) discoverable as an installed Visual Studio via
    _locate_msvc_build_env(). Raises RuntimeError with guidance if neither is
    available or the build fails.
    """
    def _log(msg):
        if on_log:
            on_log(msg)

    build_dir = _quantize_build_dir()
    existing = _find_built_quantize_binary(build_dir)
    if existing is not None:
        return existing

    llama_cpp_dir = ensure_llama_cpp(on_log=on_log)
    build_env = _locate_msvc_build_env(on_log=on_log)
    # On Windows, subprocess's env= only sets the CHILD process's environment
    # block — CreateProcess still resolves a bare "cmake" against the CALLING
    # process's own PATH, not build_env's, so a cmake found only via the
    # located Visual Studio would otherwise raise FileNotFoundError despite
    # being right there in build_env["PATH"]. Resolve the absolute path
    # ourselves against build_env's PATH to sidestep that.
    cmake_exe = "cmake"
    if build_env is not None:
        cmake_exe = shutil.which("cmake", path=build_env.get("PATH")) or "cmake"

    _log("INFO:  Building plain llama-quantize from source (first run only, needs cmake + a C++ compiler)…")
    try:
        subprocess.run(
            [cmake_exe, "-B", str(build_dir), "-S", str(llama_cpp_dir)],
            check=True, capture_output=True, text=True, env=build_env,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "cmake (and a C++ compiler) are required to build a plain llama-quantize "
            "for text-encoder K-quants but cmake was not found on PATH"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Failed to configure llama.cpp build: {exc.stderr or exc}") from exc

    proc = subprocess.Popen(
        [cmake_exe, "--build", str(build_dir), "--config", "Release", "--target", "llama-quantize"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1,
        env=build_env,
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
    "tekken.json",  # Mistral/Ministral repos ship this instead of tokenizer.json
    "spiece.model",  # T5/UMT5 repos ship this SentencePiece file instead of tokenizer.model
)

# Config/tokenizer files vendored under text_encoder_configs/ (see docs/architecture.md
# "Vendored Text-Encoder Configs") for the base repos this tool's documented candidate
# families use. Avoids a HuggingFace round-trip on every conversion and keeps the tool
# working if a repo is later gated/pulled. Keyed by exact repo_id as typed into the GUI's
# base-repo field.
_VENDORED_CONFIGS_DIR = Path(__file__).parent / "text_encoder_configs"
_VENDORED_REPOS = {
    "openai/clip-vit-large-patch14": "clip-l",
    "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k": "clip-bigg",
    "google/t5-v1_1-xxl": "t5-xxl",
    "Qwen/Qwen3-4B": "qwen3-4b",
    "Qwen/Qwen3-8B": "qwen3-8b",
    "Qwen/Qwen2.5-VL-7B-Instruct": "qwen2.5-vl-7b",
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506": "mistral-small-3.2-24b",
}

# (hidden_size, num_hidden_layers, vocab_size) read straight off a checkpoint's own
# embedding tensor + layer count -> vendored family short name (text_encoder_configs/<name>/).
# Unlike models/architectures.py's detect_arch() (exact key-list matching), key NAMES alone
# can't disambiguate here: Qwen3/Mistral/Ministral3 are all Llama-style decoders with nearly
# identical key names regardless of size. The shape triple is unique across every vendored
# family though (verified against each family's real config.json), so it doubles as an
# architecture+size fingerprint. See docs/architecture.md "Vendored Text-Encoder Configs".
_FAMILY_SIGNATURES = {
    (768, 12, 49408): "clip-l",
    (1280, 32, 49408): "clip-bigg",
    (4096, 24, 32128): "t5-xxl",
    (4096, 24, 256384): "umt5-xxl",  # UMT5-XXL, Wan 2.1/2.2's text encoder (multilingual T5, larger vocab than t5-xxl)
    (2560, 36, 151936): "qwen3-4b",
    (4096, 36, 151936): "qwen3-8b",
    (3584, 28, 152064): "qwen2.5-vl-7b",
    (5120, 40, 131072): "mistral-small-3.2-24b",
    (3072, 26, 131072): "ernie-image-pe",  # Ministral3 prompt-enhancer, baidu/ERNIE-Image's pe/ subfolder
    (4096, 32, 128256): "llama-3.1-8b",  # Llama-3.1-8B-Instruct, HiDream-I1's 4th text encoder
}

_LAYER_IDX_RE = re.compile(r"\.(?:layers|block)\.(\d+)\.")
_EMBED_KEY_SUFFIXES = ("embed_tokens.weight", "token_embedding.weight", "shared.weight")

# Families whose GGUF conversion is not just untested but structurally
# impossible with this tool's llama.cpp-based pipeline: convert_hf_to_gguf.py
# has no CLIPModel/CLIPTextModel converter at all (confirmed by grepping the
# vendored .llama.cpp checkout for either class name -- zero matches), unlike
# the LLM-style decoder architectures (Qwen3, Mistral) and T5 it does support.
# ComfyUI-GGUF's CLIPLoaderGGUF node has no CLIP-L/bigG decode path either, so
# this isn't a "some other outtype might work" situation -- every GGUF
# outtype and every K-quant is equally impossible. Found 2026-08-14 when a
# user's clip_g -> F16 GGUF conversion ran the full config-fetch + subprocess
# dance before failing with llama.cpp's own opaque "Model CLIPModel is not
# supported" -- this guard fails fast with an actionable message instead.
_GGUF_UNSUPPORTED_FAMILIES: frozenset[str] = frozenset({"clip-l", "clip-bigg"})


def _reject_if_gguf_unsupported(family: str | None) -> None:
    if family in _GGUF_UNSUPPORTED_FAMILIES:
        raise RuntimeError(
            f"'{family}' (CLIP-L/OpenCLIP-bigG) cannot be converted to GGUF -- "
            "llama.cpp's convert_hf_to_gguf.py has no CLIPModel converter, and "
            "ComfyUI-GGUF's CLIPLoaderGGUF node has no CLIP-L/bigG decode path "
            "either, so no GGUF outtype or K-quant will ever work here. Use a "
            "safetensors format (FP8/FP8_MIXED/INT8/INT8_MIXED/NVFP4/"
            "NVFP4_MIXED/F16 safetensors) instead."
        )


def detect_text_encoder_family(state_dict) -> str | None:
    """Identify which vendored family a bare text-encoder checkpoint's weights match.

    Reads (hidden_size, num_hidden_layers, vocab_size) off the checkpoint itself
    (no config.json needed) and looks it up in _FAMILY_SIGNATURES. Returns the
    vendored family short name, or None if nothing matches (unknown/uncommon base
    model — caller falls back to requiring a manual repo ID).
    """
    keys = list(state_dict.keys())
    embed_key = next((k for k in keys if k.endswith(_EMBED_KEY_SUFFIXES)), None)
    if embed_key is None:
        return None

    shape_of = getattr(state_dict, "shape_of", None)
    vocab_size, hidden_size = shape_of(embed_key) if shape_of else tuple(state_dict[embed_key].shape)
    num_layers = len({int(m.group(1)) for k in keys if (m := _LAYER_IDX_RE.search(k))})
    return _FAMILY_SIGNATURES.get((hidden_size, num_layers, vocab_size))


def _copy_vendored_family(short_name: str, dest_dir: Path, on_log=None) -> list[str]:
    """Copy a vendored family's config/tokenizer files into dest_dir. RuntimeError if missing."""
    def _log(msg):
        if on_log:
            on_log(msg)

    vendored_dir = _VENDORED_CONFIGS_DIR / short_name
    if not vendored_dir.is_dir():
        raise RuntimeError(f"No vendored config directory for {short_name!r} ({vendored_dir})")
    _log(f"INFO:  Using vendored config for {short_name} ({vendored_dir})")
    copied = [src.name for src in vendored_dir.iterdir() if src.is_file()]
    for name in copied:
        shutil.copy2(vendored_dir / name, dest_dir / name)
    if "config.json" not in copied:
        raise RuntimeError(f"Vendored config for {short_name!r} is missing config.json ({vendored_dir})")
    return copied


def fetch_base_config_files(repo_id: str, dest_dir: Path, on_log=None) -> list[str]:
    """Copy vendored (or download) config.json + tokenizer files for repo_id into dest_dir.

    Checks _VENDORED_REPOS first so common base repos need no network access;
    falls back to a live HuggingFace download for anything not vendored.
    config.json is mandatory (RuntimeError if missing); tokenizer files are
    best-effort since repos vary in which ones they ship.
    """
    def _log(msg):
        if on_log:
            on_log(msg)

    if repo_id in _VENDORED_REPOS:
        return _copy_vendored_family(_VENDORED_REPOS[repo_id], dest_dir, on_log=on_log)

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


# Tensor-key suffixes that must never reach convert_hf_to_gguf.py: real
# weight/scale-sidecar/sentinel tensors llama.cpp's converter can't map to
# any GGUF tensor name (ValueError: Can not map tensor '<key>').
# "scaled_fp8" is a harmless legacy-format marker (see convert_safetensors.py's
# _scan_quantized_layers() -- already stripped from every safetensors output
# by convert_to_safetensors(), listed here again in case weights_path is a
# raw, never-passed-through-this-tool checkpoint). "spiece_model" is the
# opposite case: load-bearing for ComfyUI's own safetensors loading (see that
# same docstring for why it's deliberately NOT stripped there) but equally
# fatal for llama.cpp's converter, which needs the tokenizer as an external
# vendored .model file instead -- so it's filtered only here, in the copy
# feeding the GGUF conversion, never in the shared F16/FP8/etc. output file
# a user might also load directly in ComfyUI. Found 2026-08-19: an earlier
# version of this fix stripped spiece_model unconditionally in
# convert_safetensors.py, breaking every safetensors output of a self-
# contained-tokenizer checkpoint (ComfyUI: `ValueError: invalid tokenizer`).
_GGUF_INCOMPATIBLE_TENSOR_SUFFIXES = ("scaled_fp8", "spiece_model", "tekken_model")
# "tekken_model" (FLUX.2 dev's mistral_3_small_flux2 encoder's embedded
# Tekken tokenizer JSON) is the same load-bearing-but-GGUF-fatal case --
# added defensively 2026-08-23 alongside dequantize.py's
# _PASSTHROUGH_TENSOR_SUFFIXES fix, even though this family's GGUF path is
# currently blocked earlier (family auto-detection fails outright, see
# model_support.py's mistral-small-3.2-24b GGUF entry) so this never
# actually gets exercised yet.


def _copy_weights_for_gguf(weights_path: str, dst: Path) -> None:
    """Copy a safetensors file into dst, dropping any tensor whose key ends
    in _GGUF_INCOMPATIBLE_TENSOR_SUFFIXES. Streams one tensor at a time via
    load_state_dict's lazy view (same memory profile convert_to_safetensors()
    already uses for every conversion) -- never a raw byte-for-byte copy once
    filtering is needed, but also never holds more than the final output
    dict, matching this project's existing OOM-safety bar."""
    state_dict = load_state_dict(weights_path, strip_prefixes=False)
    keys = state_dict.keys()
    if not any(k.endswith(_GGUF_INCOMPATIBLE_TENSOR_SUFFIXES) for k in keys):
        shutil.copy2(weights_path, dst)
        return
    from safetensors.torch import save_file
    out = {
        k: v for k, v in state_dict.items()
        if not k.endswith(_GGUF_INCOMPATIBLE_TENSOR_SUFFIXES)
    }
    save_file(out, str(dst))


def convert_text_encoder(
    weights_path: str,
    base_repo_id: str | None = None,
    dst_path: str | None = None,
    outtype: str = "f16",
    on_log=None,
    cancel_event=None,
) -> str:
    """Convert a bare single-file text-encoder checkpoint to GGUF.

    Fetches config.json/tokenizer files for the base model, assembles a temp
    HF-style model directory with the local weights, then runs
    convert_hf_to_gguf.py (auto-cloned from llama.cpp) with this tool's own
    Python interpreter.

    ``base_repo_id`` is optional: if omitted (or blank), the checkpoint's own
    weights are fingerprinted via detect_text_encoder_family() and matched
    against a vendored family (see docs/architecture.md "Vendored Text-Encoder
    Configs"). Raises RuntimeError if that fails to match and no base_repo_id
    was given — auto-detection only covers this tool's documented candidate
    families, not arbitrary base models.
    """
    def _log(msg):
        if on_log:
            on_log(msg)
        else:
            # See convert_safetensors.py's identical guard: a Windows console
            # on a legacy codepage (cp1252) raises UnicodeEncodeError on
            # non-ASCII log text (e.g. "->" as U+2192) and would otherwise
            # abort mid-conversion.
            try:
                print(msg)
            except UnicodeEncodeError:
                enc = sys.stdout.encoding or "ascii"
                print(msg.encode(enc, errors="replace").decode(enc))

    if base_repo_id and base_repo_id.strip():
        _reject_if_gguf_unsupported(_VENDORED_REPOS.get(base_repo_id.strip()))
    else:
        state_dict = load_state_dict(weights_path, strip_prefixes=False)
        family = detect_text_encoder_family(state_dict)
        if family is None:
            raise RuntimeError(
                "Could not auto-detect the base model family from these weights, and no "
                "base repo ID was given. Enter the base model's HF repo ID manually."
            )
        _reject_if_gguf_unsupported(family)

    # find_convert_script() may clone llama.cpp (network + subprocess) --
    # deferred until after the cheap, offline family check above so a
    # rejected CLIP family fails fast instead of triggering a clone first.
    script = find_convert_script()

    if dst_path is None:
        dst_path = f"{weights_path.rsplit('.', 1)[0]}-{outtype}.gguf"

    with tempfile.TemporaryDirectory(prefix="s2g_text_encoder_") as tmpdir:
        tmp_path = Path(tmpdir)
        if base_repo_id and base_repo_id.strip():
            _log(f"INFO:  Fetching config/tokenizer for {base_repo_id}…")
            fetch_base_config_files(base_repo_id.strip(), tmp_path, on_log=_log)
        else:
            _log(f"INFO:  Auto-detected base model family: {family}")
            _copy_vendored_family(family, tmp_path, on_log=_log)

        weights_dst = tmp_path / "model.safetensors"
        _copy_weights_for_gguf(weights_path, weights_dst)

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
            # See convert_safetensors.py's identical guard: a Windows console
            # on a legacy codepage (cp1252) raises UnicodeEncodeError on
            # non-ASCII log text (e.g. "->" as U+2192) and would otherwise
            # abort mid-conversion.
            try:
                print(msg)
            except UnicodeEncodeError:
                enc = sys.stdout.encoding or "ascii"
                print(msg.encode(enc, errors="replace").decode(enc))

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


# Generic (non-architecture-specific) model_arch used for the safetensors
# quant path below, since text encoders aren't in models.architectures.arch_list.
# keys_shape_critical protects embedding-lookup tables from NVFP4 packing:
# NVFP4 halves a tensor's on-disk last dimension (2 values/byte), which is
# fine for weights consumed through a dequant-aware quantized-Linear wrapper
# (what ComfyUI's MixedPrecisionOps provides for the transformer's own
# Linear layers) but breaks a plain nn.Embedding, loaded via a vanilla
# `load_state_dict` with no such unpacking -- confirmed against a live
# ComfyUI install: Qwen3-4B's NVFP4 output raised "size mismatch for
# model.embed_tokens.weight: ... torch.Size([151936, 1280]) ... the shape
# in current model is torch.Size([151936, 2560])" (1280 == 2560 // 2).
# Substrings are the common tied/untied embedding-table key names across
# this project's documented text-encoder families (Qwen3/Mistral, T5/UMT5,
# CLIP) -- matched the same way keys_hiprec/keys_ignore are elsewhere,
# open to extending if another architecture's embed key isn't covered yet.
_TEXT_ENCODER_MODEL_ARCH = ModelTemplate()
_TEXT_ENCODER_MODEL_ARCH.keys_shape_critical = [
    "embed_tokens", "shared", "token_embedding", "wte", "lm_head",
    # CLIP-L/bigG's position_embedding: comfy/clip_model.py's
    # CLIPEmbeddings.forward()/CLIPTextModel_.forward() both read
    # `self.embeddings.position_embedding.weight` via a bare attribute access
    # (`comfy.ops.cast_to(...)`), not a module call -- bypasses whatever
    # quantized-dequant machinery the Embedding class has entirely. Found
    # 2026-08-14: NVFP4 crashed outright ("size of tensor a (768) must match
    # tensor b (384)" -- NVFP4 halves the on-disk last dim), and FP8/INT8
    # silently produced corrupted (black-image) output for the same
    # underlying reason -- see safetensors_quant.py's FP8-branch
    # keys_shape_critical check, added alongside this entry.
    "position_embedding",
    # CLIP-G/bigG's text_projection: the ONLY weight feeding SDXL's global
    # pooled "y" conditioning vector (adm_in_channels=2816) -- CLIP-L
    # contributes no pooled/projected output to SDXL at all (its extracted
    # file has no text_projection.weight), only per-token hidden states.
    # Unlike position_embedding above, comfy/clip_model.py's CLIPTextModel
    # calls this as a proper module (`self.text_projection(x[2])`), so it
    # should in principle dequantize correctly -- but a single global vector
    # (not per-token, so no error-averaging across many tokens) is uniquely
    # exposed to a small quantization error skewing the ENTIRE image, not
    # just local detail. Found 2026-08-14: clip_g FP8 (with position_
    # embedding already fixed) still rendered solid black; this was the only
    # remaining unprotected 2D tensor outside the per-layer transformer
    # blocks.
    "text_projection",
    # T5/UMT5's relative_attention_bias: a small [num_buckets, num_heads]
    # lookup table (e.g. HiDream's T5-XXL: [32, 64]) read via a bare
    # nn.Embedding indexing op (comfy/text_encoders/t5.py's
    # T5Attention.forward -> self.relative_attention_bias(...)), same
    # unpacking gap as position_embedding above. Found 2026-08-18: HiDream
    # T5-XXL NVFP4/NVFP4_MIXED crashed loading in ComfyUI with "size
    # mismatch ... torch.Size([32, 32]) ... current model is
    # torch.Size([32, 64])" (NVFP4 halves the on-disk last dim).
    "relative_attention_bias",
]


def convert_text_encoder_to_safetensors(
    weights_path: str,
    dst_path: str | None = None,
    target_key: str = "FP8",
    on_log=None,
    cancel_event=None,
) -> str:
    """Convert a text-encoder checkpoint to a quantized .safetensors file
    (FP8/FP8_MIXED/INT8/INT8_MIXED/NVFP4/NVFP4_MIXED).

    No HuggingFace download or llama.cpp involved — ComfyUI's text-encoder
    loaders build models from fixed config presets rather than inferring
    hyperparameters from checkpoint tensor shapes (unlike the diffusion-model
    DiT architectures), so no per-architecture keys_hiprec list is needed
    here. keys_shape_critical IS needed, though (see _TEXT_ENCODER_MODEL_ARCH
    below) — an earlier version of this docstring claimed otherwise; a live
    ComfyUI test with NVFP4 on a real Qwen3-4B checkpoint proved that wrong.

    strip_prefixes=False: convert_to_safetensors()'s default "model."-prefix
    stripping assumes that prefix wraps a diffusion UNet inside a larger
    checkpoint. A standalone text-encoder file has no such wrapper -- "model."
    is its own genuine module path (e.g. Qwen3's "model.layers.0..."), and
    stripping it breaks ComfyUI's own text-encoder architecture detection
    (comfy/sd.py's detect_te_model looks for literal "model.layers.0...."
    keys) on the output file, silently falling back to the wrong text-encoder
    class at load time.
    """
    dst, _ = convert_to_safetensors(
        weights_path, dst_path=dst_path, target_key=target_key, overwrite=True,
        on_log=on_log, cancel_event=cancel_event, model_arch=_TEXT_ENCODER_MODEL_ARCH,
        strip_prefixes=False,
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
    or a safetensors-quant format (TEXT_ENCODER_SAFETENSORS_FORMATS —
    base_repo_id is ignored for these, no HF download needed).
    """
    if format_key in TEXT_ENCODER_SAFETENSORS_FORMATS:
        real_target_key = _TEXT_ENCODER_SAFETENSORS_TARGET_KEY.get(format_key, format_key)
        return convert_text_encoder_to_safetensors(
            weights_path, dst_path=dst_path, target_key=real_target_key,
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
