"""Convert safetensors/ckpt model files to GGUF format.

Supports FLUX, SD3, SDXL, SD1, HiDream, HunyuanVideo, Wan, LTXV, Cosmos,
AuraFlow, and Lumina2 architectures.

Usage:
    python convert.py --src model.safetensors [--dst model.gguf]
"""

import os
import logging
import argparse

import gguf
import torch
from tqdm import tqdm
from safetensors.torch import load_file

from models.architectures import (
    detect_arch,
    MAX_TENSOR_NAME_LENGTH,
    MAX_TENSOR_DIMS,
    QUANTIZATION_THRESHOLD,
    REARRANGE_THRESHOLD,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


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


def strip_prefix(state_dict):
    """Remove known state-dict prefixes used by various checkpoint formats."""
    # Mixed state dicts (VAE + UNet merged)
    for pfx in ["model.diffusion_model.", "model."]:
        if any(k.startswith(pfx) for k in state_dict):
            logging.info(f"Stripping mixed prefix '{pfx}'")
            return {
                k.replace(pfx, ""): v
                for k, v in state_dict.items()
                if pfx in k
            }

    # Uniform prefix (all keys start with it)
    for pfx in ["net."]:
        if all(k.startswith(pfx) for k in state_dict):
            logging.info(f"Stripping uniform prefix '{pfx}'")
            return {k[len(pfx):]: v for k, v in state_dict.items()}

    return state_dict


def load_state_dict(path):
    """Load a model checkpoint from disk and return a clean state dict."""
    if any(path.endswith(ext) for ext in (".ckpt", ".pt", ".bin", ".pth")):
        sd = torch.load(path, map_location="cpu", weights_only=True)
        for subkey in ("model", "module"):
            if subkey in sd:
                sd = sd[subkey]
                break
        if len(sd) < 20:
            raise RuntimeError(f"Unexpected checkpoint structure (keys: {list(sd.keys())})")
    else:
        sd = load_file(path)

    return strip_prefix(sd)


def _quant_type_for(data, key, model_arch, old_dtype, target_quant=None):
    """Return the GGML quantization type to use for a given tensor.

    1D tensors, tensors with ≤ QUANTIZATION_THRESHOLD elements, and keys listed
    in model_arch.keys_hiprec are always stored as F32 regardless of target_quant.
    All other tensors use target_quant when provided, otherwise fall back to the
    source dtype (BF16 → BF16, everything else → F16).
    """
    n_dims = len(data.shape)
    n_params = data.numel()

    if old_dtype in (torch.float32, torch.bfloat16):
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

    items = list(state_dict.items())
    total = len(items)
    iter_items = tqdm(items) if on_progress is None else items
    log_tensor_every = max(1, int(log_tensor_every or 1))

    for idx, (key, data) in enumerate(iter_items):
        if cancel_event is not None and cancel_event.is_set():
            raise ConversionCancelled()

        if on_progress is not None:
            on_progress(idx + 1, total, key)

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

    # Determine GGUF file type from target_quant or dominant source dtype
    if target_quant is not None and target_quant in _QUANT_FTYPE:
        ftype_name, ftype_gguf = _QUANT_FTYPE[target_quant]
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

    writer = gguf.GGUFWriter(path=None, arch=model_arch.arch)
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
