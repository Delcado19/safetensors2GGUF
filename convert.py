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


def _quant_type_for(data, key, model_arch, old_dtype):
    """Return the GGML quantization type to use for a given tensor."""
    n_dims = len(data.shape)
    n_params = data.numel()

    if old_dtype == torch.bfloat16:
        base_qtype = gguf.GGMLQuantizationType.BF16
    else:
        base_qtype = gguf.GGMLQuantizationType.F16

    if old_dtype in (torch.float32, torch.bfloat16):
        if n_dims == 1:
            return gguf.GGMLQuantizationType.F32
        if n_params <= QUANTIZATION_THRESHOLD:
            return gguf.GGMLQuantizationType.F32
        if any(x in key for x in model_arch.keys_hiprec):
            return gguf.GGMLQuantizationType.F32

    return base_qtype


def handle_tensors(writer, state_dict, model_arch):
    """Write all tensors from state_dict into the GGUF writer.

    Applies dtype coercion, nan/inf clamping, quantization type selection,
    shape reshaping for SD1/SDXL, and 5D tensor side-car export.
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

    for key, data in tqdm(state_dict.items()):
        old_dtype = data.dtype

        if any(x in key for x in model_arch.keys_ignore):
            tqdm.write(f"  skip (ignored): {key}")
            continue

        # Clamp inf/nan before any dtype conversion to prevent llama-quantize failures
        data = torch.nan_to_num(data, nan=0.0, posinf=65504.0, neginf=-65504.0)

        # Convert to numpy — float8 variants go via float16
        if data.dtype == torch.bfloat16:
            data = data.to(torch.float32).numpy()
        elif data.dtype in (
            getattr(torch, "float8_e4m3fn", None),
            getattr(torch, "float8_e5m2", None),
        ):
            data = data.to(torch.float16).numpy()
        else:
            data = data.numpy()

        if len(data.shape) > MAX_TENSOR_DIMS:
            model_arch.handle_nd_tensor(key, data)
            continue

        data_qtype = _quant_type_for(
            torch.as_tensor(data), key, model_arch, old_dtype
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
            tqdm.write(f"  quantize fallback to F16 ({exc}): {key}")
            data_qtype = gguf.GGMLQuantizationType.F16
            data = gguf.quants.quantize(data, data_qtype)

        shape_str = "{" + ", ".join(str(d) for d in reversed(data.shape)) + "}"
        tqdm.write(
            f"  {'%-*s' % (max_name_len + 2, key)}"
            f"  {str(old_dtype):<20} → {data_qtype.name:<8}  {shape_str}"
        )
        writer.add_tensor(key, data, raw_dtype=data_qtype)


def convert_file(path, dst_path=None, interact=True, overwrite=False):
    """Convert a model checkpoint to GGUF and write it to disk.

    Returns (dst_path, model_arch).
    """
    state_dict = load_state_dict(path)
    model_arch = detect_arch(state_dict)
    logging.info(f"Architecture detected: {model_arch.arch}")

    # Determine dominant dtype → sets GGUF file type
    dtypes = [v.dtype for v in state_dict.values()]
    dtype_counts = {d: dtypes.count(d) for d in set(dtypes)}
    main_dtype = max(dtype_counts, key=dtype_counts.get)

    if main_dtype == torch.bfloat16:
        ftype_name = "BF16"
        ftype_gguf = gguf.LlamaFileType.MOSTLY_BF16
    else:
        ftype_name = "F16"
        ftype_gguf = gguf.LlamaFileType.MOSTLY_F16

    if dst_path is None:
        dst_path = f"{os.path.splitext(path)[0]}-{ftype_name}.gguf"
    elif "{ftype}" in dst_path:
        dst_path = dst_path.replace("{ftype}", ftype_name)

    if os.path.isfile(dst_path) and not overwrite:
        if interact:
            input(f"Output exists: {dst_path}\nPress Enter to overwrite or Ctrl-C to abort.")
        else:
            raise OSError(f"Output exists and overwrite is disabled: {dst_path}")

    writer = gguf.GGUFWriter(path=None, arch=model_arch.arch)
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    writer.add_file_type(ftype_gguf)

    handle_tensors(writer, state_dict, model_arch)

    writer.write_header_to_file(path=dst_path)
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()

    fix_path = f"./fix_5d_tensors_{model_arch.arch}.safetensors"
    if os.path.isfile(fix_path):
        logging.warning(
            f"5D tensor fix file found at '{fix_path}'. "
            "Run fix_5d_tensors.py after quantization."
        )

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
