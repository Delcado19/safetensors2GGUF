"""Convert safetensors/ckpt model files to GGUF format.

Supports FLUX, SD3, SDXL, SD1, HiDream, HunyuanVideo, Wan, LTXV, Cosmos,
AuraFlow, and Lumina2 architectures. Safetensors-only architectures may be
detected by models.architectures but are rejected here if City96's GGUF patch
has no llm_arch entry for them.

Usage:
    python convert.py --src model.safetensors [--dst model.gguf]
"""

import os
import json
import struct
import logging
import argparse

import gguf
import torch
from tqdm import tqdm
from safetensors import safe_open

from models.architectures import (
    detect_arch,
    MAX_TENSOR_NAME_LENGTH,
    MAX_TENSOR_DIMS,
    QUANTIZATION_THRESHOLD,
    REARRANGE_THRESHOLD,
)
from dequantize import _scan_quantized_layers, dequantize_weight

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


_GGUF_UNSUPPORTED_ARCHITECTURES: dict[str, str] = {
    "ernie_image": (
        "ERNIE-Image GGUF is unsupported: city96/ComfyUI-GGUF's lcpp.patch "
        "has no ernie_image llm_arch entry. Use safetensors output instead."
    ),
    "krea2": (
        "Krea 2 GGUF is unsupported in this project: it needs an unofficial "
        "city96/ComfyUI-GGUF fork/PR (#459), not the lcpp.patch version this "
        "tool targets. Use safetensors output instead."
    ),
}


def _reject_if_gguf_unsupported(model_arch) -> None:
    reason = _GGUF_UNSUPPORTED_ARCHITECTURES.get(model_arch.arch)
    if reason:
        raise RuntimeError(reason)


class ConversionCancelled(Exception):
    """Raised when the user cancels a running conversion."""


# Float8 dtypes — resolved once at import time; empty tuple on older PyTorch builds.
_FLOAT8_DTYPES: tuple = tuple(
    d for d in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2",   None),
    )
    if d is not None
)


# Map safetensors header dtype strings to torch dtypes — used by _LazyStateDict
# to report tensor dtypes without materializing the tensor data itself.
_ST_DTYPE_MAP: dict = {
    "F32":     torch.float32,
    "F16":     torch.float16,
    "BF16":    torch.bfloat16,
    "F64":     torch.float64,
    "F8_E4M3": getattr(torch, "float8_e4m3fn", None),
    "F8_E5M2": getattr(torch, "float8_e5m2", None),
    "I8":      torch.int8,
    "I16":     torch.int16,
    "I32":     torch.int32,
    "I64":     torch.int64,
    "U8":      torch.uint8,
    "BOOL":    torch.bool,
}

# Reverse of the above -- torch dtype -> safetensors header dtype string.
# Used by the streaming safetensors writer (convert_safetensors.py) and
# safetensors_quant.plan_tensor_output() to build header entries for
# tensors that keep their original dtype (mixed-precision hiprec passthrough,
# passthrough sentinel blobs) instead of one of quantize_tensor_st's own
# fixed output dtypes.
_TORCH_TO_ST_DTYPE: dict = {
    v: k for k, v in _ST_DTYPE_MAP.items() if v is not None
}


def _read_safetensors_header(path: str) -> dict:
    """Return the JSON header of a safetensors file (no tensor data loaded)."""
    with open(path, "rb") as f:
        (hdr_len,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(hdr_len))


def _build_key_map(keys, strip_prefixes: bool = True) -> dict:
    """Return a {logical_key: on_disk_key} mapping after applying prefix rules.

    Mirrors strip_prefix's behaviour but on key lists, so the same prefix logic
    can drive both the eager (dict) and lazy (safetensors-streaming) paths.

    ``strip_prefixes=False`` bypasses all of this (identity map): the "model."
    stripping rule below assumes "model." is a wrapper artifact around a
    diffusion UNet (e.g. a full checkpoint's "model.diffusion_model.*"), but
    a bare HF Transformers text-encoder state dict also genuinely uses
    "model." for its own real module path (e.g. Qwen3's
    "model.layers.0.self_attn.q_proj.weight") — stripping it there destroys
    the key structure ComfyUI's own text-encoder architecture detection
    (comfy/sd.py's detect_te_model) depends on. Callers that know their
    source is already a standalone text-encoder checkpoint, not a
    diffusion-model-shaped one, must pass False.
    """
    keys = [k for k in keys if k != "__metadata__"]
    if not strip_prefixes:
        return {k: k for k in keys}

    # Mixed state dicts (VAE + UNet merged) — keep only keys carrying the prefix.
    for pfx in ["model.diffusion_model.", "model."]:
        if any(k.startswith(pfx) for k in keys):
            logging.info(f"Stripping mixed prefix '{pfx}'")
            return {k.replace(pfx, ""): k for k in keys if pfx in k}

    # Uniform prefix (all keys start with it).
    for pfx in ["net."]:
        if all(k.startswith(pfx) for k in keys):
            logging.info(f"Stripping uniform prefix '{pfx}'")
            return {k[len(pfx):]: k for k in keys}

    return {k: k for k in keys}


def strip_prefix(state_dict, strip_prefixes: bool = True):
    """Remove known state-dict prefixes used by various checkpoint formats.

    Eager path: returns a new dict containing the renamed tensors.  Kept for
    non-safetensors sources and existing tests; safetensors sources go through
    _LazyStateDict which applies the same prefix rules via _build_key_map.
    """
    key_map = _build_key_map(state_dict.keys(), strip_prefixes=strip_prefixes)
    return {logical: state_dict[on_disk] for logical, on_disk in key_map.items()}


class _LazyStateDict:
    """Dict-like lazy view over a safetensors file.

    Materializes tensors one at a time via ``safe_open().get_tensor()`` so peak
    RAM stays bounded to a single tensor instead of the full state-dict.
    Required to convert checkpoints larger than physical RAM (e.g. 19 GiB FP8
    Qwen-Image-Edit on a 16 GiB system) without triggering Windows page-in
    failures (KERNEL_DATA_INPAGE_ERROR) or OOM-driven segfaults.

    Implements the subset of the dict interface used by detect_arch (membership,
    iteration) and handle_tensors (``items()``, ``len()``), plus a ``dtype_of()``
    accessor that convert_file uses for dominant-dtype detection without
    materializing any tensor data.

    Note: ``.values()`` is intentionally not implemented because iterating it
    would defeat the streaming purpose by loading every tensor at once.
    """

    def __init__(self, path: str, key_map: dict, header: dict):
        self._path = path
        self._key_map = key_map      # logical_key -> on_disk_key
        self._header = header        # raw safetensors JSON header
        self._fh = safe_open(path, framework="pt", device="cpu")

    def __contains__(self, key):
        return key in self._key_map

    def __iter__(self):
        return iter(self._key_map)

    def __len__(self):
        return len(self._key_map)

    def keys(self):
        return list(self._key_map.keys())

    def __getitem__(self, key):
        return self._fh.get_tensor(self._key_map[key])

    def items(self):
        """Yield (logical_key, tensor) one at a time — bounded peak RAM."""
        for logical, on_disk in self._key_map.items():
            yield logical, self._fh.get_tensor(on_disk)

    def dtype_of(self, key):
        """Return the torch dtype for ``key`` from the header — no tensor load."""
        st_dtype = self._header[self._key_map[key]]["dtype"]
        return _ST_DTYPE_MAP.get(st_dtype)

    def shape_of(self, key):
        """Return the tensor shape for ``key`` from the header — no tensor load."""
        return tuple(self._header[self._key_map[key]]["shape"])

    def file_metadata(self) -> dict:
        """Return the safetensors file-level ``__metadata__`` dict (e.g. this
        tool's own ``_quantization_metadata``), or {} if none was written."""
        return self._header.get("__metadata__") or {}


def load_state_dict(path, strip_prefixes: bool = True):
    """Load a model checkpoint from disk and return a (possibly lazy) state-dict.

    For ``.safetensors`` sources, returns a ``_LazyStateDict`` that streams
    tensors one at a time — keeps peak RAM bounded for checkpoints larger than
    physical memory.  For ``.ckpt``/``.pt``/``.bin``/``.pth`` sources, returns
    a normal eager dict (torch.load has no streaming API, and these formats
    are rarely >RAM in practice).

    ``strip_prefixes=False`` for standalone text-encoder checkpoints — see
    ``_build_key_map``'s docstring for why the default "model." stripping
    rule is wrong for those.
    """
    if any(path.endswith(ext) for ext in (".ckpt", ".pt", ".bin", ".pth")):
        sd = torch.load(path, map_location="cpu", weights_only=True)
        for subkey in ("model", "module"):
            if subkey in sd:
                sd = sd[subkey]
                break
        if len(sd) < 20:
            raise RuntimeError(f"Unexpected checkpoint structure (keys: {list(sd.keys())})")
        return strip_prefix(sd, strip_prefixes=strip_prefixes)

    # safetensors: lazy streaming (avoids OOM on >RAM checkpoints)
    header = _read_safetensors_header(path)
    key_map = _build_key_map(header.keys(), strip_prefixes=strip_prefixes)
    return _LazyStateDict(path, key_map, header)


def _quant_type_for(data, key, model_arch, old_dtype, target_quant=None):
    """Return the GGML quantization type to use for a given tensor.

    1D tensors, tensors with ≤ QUANTIZATION_THRESHOLD elements, and keys listed
    in model_arch.keys_hiprec are always stored as F32 regardless of target_quant.
    All other tensors use target_quant when provided, otherwise fall back to the
    source dtype (BF16 → BF16, everything else → F16).
    """
    n_dims = len(data.shape)
    n_params = data.numel()

    # F16 belongs in this gate too, not just F32/BF16 -- see
    # safetensors_quant.py's is_hiprec_st() comment (2026-08-18, aura_flow_0.3
    # is F16-native, unlike the BF16 checkpoints this mechanism was validated
    # against). Without it, keys_hiprec is silently inert for any F16-native
    # source checkpoint's GGUF conversion too, same bug class.
    if old_dtype in (torch.float32, torch.bfloat16, torch.float16):
        if n_dims == 1:
            return gguf.GGMLQuantizationType.F32
        if n_params <= QUANTIZATION_THRESHOLD:
            return gguf.GGMLQuantizationType.F32
        if any(x in key for x in model_arch.keys_hiprec):
            return gguf.GGMLQuantizationType.F32

    if target_quant is not None:
        return target_quant

    if old_dtype == torch.bfloat16:
        return gguf.GGMLQuantizationType.BF16
    return gguf.GGMLQuantizationType.F16


def handle_tensors(
    writer,
    state_dict,
    model_arch,
    on_progress=None,
    on_log=None,
    target_quant=None,
    cancel_event=None,
    log_tensor_every: int = 1,
    apply_unsqueeze: bool = True,
):
    """Write all tensors from state_dict into the GGUF writer.

    Applies dtype coercion, nan/inf clamping, quantization type selection,
    shape reshaping for SD1/SDXL, and 5D tensor side-car export.

    When on_progress/on_log are provided the function skips tqdm entirely and
    routes all output through the callbacks — used by the GUI.

    Args:
        writer: GGUFWriter instance to write tensors into.
        state_dict: Dict mapping tensor name to torch.Tensor.
        model_arch: Architecture instance from models.architectures.
        on_progress: Optional callback(idx, total, key) called per tensor.
        on_log: Optional callback(msg) for log messages; uses tqdm.write when None.
        target_quant: GGMLQuantizationType override for non-hiprec tensors.
        cancel_event: Optional threading.Event; raises ConversionCancelled when set.
        log_tensor_every: Emit every Nth tensor conversion log line. Progress
            callbacks still fire for every tensor. Values below 1 are treated as 1.
        apply_unsqueeze: Reshape architecture-specific 1D tensors to [1, D].
            K-quant intermediates disable this because llama-quantize expects
            the original 1D Lumina2 pad-token tensors and the GUI fixes them
            after quantization.
    """
    name_lengths = sorted(
        ((k, len(k)) for k in state_dict), key=lambda x: x[1], reverse=True
    )
    if not name_lengths:
        return
    max_name_len = name_lengths[0][1]
    if max_name_len > MAX_TENSOR_NAME_LENGTH:
        too_long = [f"{k!r} ({n})" for k, n in name_lengths if n > MAX_TENSOR_NAME_LENGTH]
        raise ValueError(
            f"Tensor names exceed {MAX_TENSOR_NAME_LENGTH} chars: {', '.join(too_long)}"
        )

    def _emit(msg):
        if on_log:
            on_log(msg)
        else:
            tqdm.write(msg)

    # Pre-scan for an already-quantized source checkpoint (e.g. a community
    # fp8_mixed Wan 2.2 file) so its ".weight_scale"/".comfy_quant" sidecars
    # get dequantized into the real weight instead of being written straight
    # to GGUF as-is — those sidecars are 0-dim tensors, and llama-quantize
    # crashes with `GGML_ASSERT(n_dims >= 1 && n_dims <= GGML_MAX_DIMS)` on
    # them (found 2026-08-20, see _scan_quantized_layers() in dequantize.py).
    quant_formats, quant_skip_keys = _scan_quantized_layers(state_dict)
    if quant_formats:
        _emit(
            f"INFO:  Dequantizing {len(quant_formats)} already-quantized "
            f"layer(s) before GGUF conversion (formats: "
            f"{sorted(set(quant_formats.values()))})"
        )

    # Iterate state_dict directly — never materialize all tensors into a list,
    # otherwise _LazyStateDict's streaming would degrade to eager load.
    total = len(state_dict)
    iter_items = state_dict.items()
    if on_progress is None:
        iter_items = tqdm(iter_items, total=total)
    log_tensor_every = max(1, int(log_tensor_every or 1))

    for idx, (key, data) in enumerate(iter_items):
        if cancel_event is not None and cancel_event.is_set():
            raise ConversionCancelled()

        if on_progress is not None:
            on_progress(idx + 1, total, key)

        if key in quant_skip_keys:
            continue

        if key in quant_formats:
            # Reconstruct the approximate float original from its scale
            # sidecar before the normal dtype coercion/GGUF path below.
            data = dequantize_weight(state_dict, key, quant_formats[key], data)

        old_dtype = data.dtype

        if any(x in key for x in model_arch.keys_ignore):
            _emit(f"  skip (ignored): {key}")
            continue

        # Coerce dtypes that nan_to_num or numpy cannot handle:
        #   float8   → float16  (nan_to_num rejects float8)
        #   bfloat16 → float32  (numpy has no bfloat16 type)
        if _FLOAT8_DTYPES and data.dtype in _FLOAT8_DTYPES:
            data = data.to(torch.float16)
        elif data.dtype == torch.bfloat16:
            data = data.to(torch.float32)

        # Reshape 1D pad tokens to [1, D] for architectures that require it (e.g. Lumina2)
        if apply_unsqueeze and model_arch.keys_unsqueeze and data.dim() == 1 and key in model_arch.keys_unsqueeze:
            data = data.unsqueeze(0)

        # Clamp inf/nan to prevent llama-quantize validation failures
        data = torch.nan_to_num(data, nan=0.0, posinf=65504.0, neginf=-65504.0)
        data = data.numpy()

        if len(data.shape) > MAX_TENSOR_DIMS:
            model_arch.handle_nd_tensor(key, data)
            continue

        data_qtype = _quant_type_for(
            torch.as_tensor(data), key, model_arch, old_dtype, target_quant
        )

        # Reshape for SD1/SDXL: flatten non-256-aligned dims to (n//256, 256)
        if (
            model_arch.shape_fix
            and len(data.shape) > 1
            and data.size >= REARRANGE_THRESHOLD
            and (data.size / 256).is_integer()
            and not (data.shape[-1] / 256).is_integer()
        ):
            orig_shape = data.shape
            data = data.reshape(data.size // 256, 256)
            writer.add_array(
                f"comfy.gguf.orig_shape.{key}",
                tuple(int(d) for d in orig_shape),
            )

        try:
            data = gguf.quants.quantize(data, data_qtype)
        except (AttributeError, gguf.QuantError) as exc:
            _emit(f"  quantize fallback to F16 ({exc}): {key}")
            data_qtype = gguf.GGMLQuantizationType.F16
            data = gguf.quants.quantize(data, data_qtype)

        shape_str = "{" + ", ".join(str(d) for d in reversed(data.shape)) + "}"
        should_log_tensor = (
            log_tensor_every == 1
            or idx == 0
            or idx + 1 == total
            or (idx + 1) % log_tensor_every == 0
        )
        if should_log_tensor:
            _emit(
                f"  {'%-*s' % (max_name_len + 2, key)}"
                f"  {str(old_dtype):<20} -> {data_qtype.name:<8}  {shape_str}"
            )
        writer.add_tensor(key, data, raw_dtype=data_qtype)


_QUANT_FTYPE: dict[gguf.GGMLQuantizationType, tuple[str, gguf.LlamaFileType]] = {
    gguf.GGMLQuantizationType.F16:  ("F16",  gguf.LlamaFileType.MOSTLY_F16),
    gguf.GGMLQuantizationType.BF16: ("BF16", gguf.LlamaFileType.MOSTLY_BF16),
    gguf.GGMLQuantizationType.Q8_0: ("Q8_0", gguf.LlamaFileType.MOSTLY_Q8_0),
    gguf.GGMLQuantizationType.Q4_0: ("Q4_0", gguf.LlamaFileType.MOSTLY_Q4_0),
    gguf.GGMLQuantizationType.F32:  ("F32",  gguf.LlamaFileType.ALL_F32),
}


def convert_file(
    path,
    dst_path=None,
    interact=True,
    overwrite=False,
    on_progress=None,
    on_log=None,
    target_quant=None,
    cancel_event=None,
    log_tensor_every: int = 1,
    apply_unsqueeze: bool = True,
):
    """Convert a model checkpoint to GGUF and write it to disk.

    Args:
        path: Source model file path.
        dst_path: Output path; auto-generated when None. Use {ftype} as a
            placeholder that is replaced by the quantization name.
        interact: When True and the output exists, prompt the user interactively.
        overwrite: Skip existence check entirely.
        on_progress: Callback(idx, total, key) called for each tensor.
        on_log: Callback(msg) for informational messages. Uses logging when None.
        target_quant: GGMLQuantizationType to use for non-hiprec tensors.
            Auto-detected from source dtype when None.
        cancel_event: Optional threading.Event; raises ConversionCancelled when set.
        log_tensor_every: Emit every Nth tensor conversion log line while still
            reporting progress for each tensor.
        apply_unsqueeze: Reshape architecture-specific 1D tensors to [1, D].

    Returns:
        (dst_path, model_arch)
    """
    def _info(msg):
        if on_log:
            on_log(f"INFO:  {msg}")
        else:
            logging.info(msg)

    state_dict = load_state_dict(path)
    model_arch = detect_arch(state_dict)
    _info(f"Architecture: {model_arch.arch}")
    _reject_if_gguf_unsupported(model_arch)

    # Determine GGUF file type from target_quant or dominant source dtype
    if target_quant is not None and target_quant in _QUANT_FTYPE:
        ftype_name, ftype_gguf = _QUANT_FTYPE[target_quant]
    else:
        # Pull dtypes from the safetensors header (no tensor materialization)
        # for lazy state-dicts; fall back to ``.values()`` for eager dicts.
        if isinstance(state_dict, _LazyStateDict):
            dtypes = [state_dict.dtype_of(k) for k in state_dict.keys()]
            dtypes = [d for d in dtypes if d is not None]
        else:
            dtypes = [v.dtype for v in state_dict.values()]
        main_dtype = max(set(dtypes), key=dtypes.count)
        if main_dtype == torch.bfloat16:
            ftype_name, ftype_gguf = "BF16", gguf.LlamaFileType.MOSTLY_BF16
        else:
            ftype_name, ftype_gguf = "F16", gguf.LlamaFileType.MOSTLY_F16

    if dst_path is None:
        dst_path = f"{os.path.splitext(path)[0]}-{ftype_name}.gguf"
    elif "{ftype}" in dst_path:
        dst_path = dst_path.replace("{ftype}", ftype_name)

    if os.path.isfile(dst_path) and not overwrite:
        if interact:
            input(f"Output exists: {dst_path}\nPress Enter to overwrite or Ctrl-C to abort.")
        else:
            raise OSError(f"Output exists and overwrite is disabled: {dst_path}")

    _info(f"Output: {dst_path}  [{ftype_name}]")

    # use_temp_file=True spills converted tensors to a tempfile-backed
    # SpooledTemporaryFile (256 MiB RAM buffer, rest on disk) instead of
    # accumulating every F16/BF16 tensor in self.tensors.  Required for the
    # same reason _LazyStateDict is required on the read side: keeps peak RAM
    # bounded when converting checkpoints larger than physical memory.
    writer = gguf.GGUFWriter(path=None, arch=model_arch.arch, use_temp_file=True)
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    writer.add_file_type(ftype_gguf)

    try:
        handle_tensors(
            writer, state_dict, model_arch,
            on_progress=on_progress, on_log=on_log,
            target_quant=target_quant, cancel_event=cancel_event,
            log_tensor_every=log_tensor_every,
            apply_unsqueeze=apply_unsqueeze,
        )
    except ConversionCancelled:
        # Release writer buffers (may hold GBs of tensor data)
        del writer
        del state_dict
        raise

    _info("Writing GGUF file…")
    writer.write_header_to_file(path=dst_path)
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=on_log is None)
    writer.close()

    fix_path = f"./fix_5d_tensors_{model_arch.arch}.safetensors"
    if os.path.isfile(fix_path):
        _info(
            f"5D tensor fix file found: '{fix_path}'. "
            "Run Fix 5D Tensors after quantization with llama-quantize."
        )

    _info(f"Done → {dst_path}")
    return dst_path, model_arch


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Convert safetensors/ckpt model to F16/BF16 GGUF"
    )
    parser.add_argument("--src", required=True, help="Source model file")
    parser.add_argument("--dst", help="Output GGUF path (default: <src>-F16.gguf)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output")
    args = parser.parse_args()
    if not os.path.isfile(args.src):
        parser.error(f"Source file not found: {args.src}")
    return args


if __name__ == "__main__":
    args = _parse_args()
    out, arch = convert_file(args.src, args.dst, interact=True, overwrite=args.overwrite)
    logging.info(f"Written: {out}  (arch={arch.arch})")
