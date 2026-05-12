"""Fix shape mismatch for Lumina2 pad tokens in an existing GGUF file.

ComfyUI's NextDiT model expects x_pad_token and cap_pad_token with shape
[1, D], but older checkpoints store them as [D].  This script rewrites the
GGUF with the corrected shapes without touching any other tensor.

Usage:
    python fix_pad_tokens.py --src model.gguf --dst model-fixed.gguf
"""

import os
import argparse
import logging

import gguf
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

PAD_TOKEN_NAMES = {"x_pad_token", "cap_pad_token"}


def _read_arch(reader):
    field = reader.get_field("general.architecture")
    if field is None:
        raise ValueError("GGUF file has no 'general.architecture' field.")
    return str(field.parts[field.data[-1]], encoding="utf-8")


def _read_file_type(reader):
    field = reader.get_field("general.file_type")
    if field is None:
        return None
    return gguf.LlamaFileType(int(field.parts[field.data[-1]].flat[0]))


def fix_pad_tokens(src_path, dst_path, overwrite=False, on_progress=None, on_log=None):
    """Rewrite a Lumina2 GGUF correcting 1D pad token shapes to [1, D].

    Args:
        src_path: Source GGUF file.
        dst_path: Output GGUF path.
        overwrite: Skip output existence check.
        on_progress: Callback(idx, total, key) per tensor.
        on_log: Callback(msg) for informational messages. Uses logging when None.

    Returns:
        List of tensor names that were reshaped.
    """
    def _info(msg):
        if on_log:
            on_log(f"INFO:  {msg}")
        else:
            logging.info(msg)

    if not os.path.isfile(src_path):
        raise FileNotFoundError(f"Source GGUF not found: {src_path}")
    if os.path.isfile(dst_path) and not overwrite:
        raise OSError(f"Output exists, use --overwrite: {dst_path}")

    reader = gguf.GGUFReader(src_path)
    arch = _read_arch(reader)
    file_type = _read_file_type(reader)
    _info(f"Source arch: '{arch}'  file_type: {file_type}")

    writer = gguf.GGUFWriter(path=None, arch=arch)
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    if file_type is not None:
        writer.add_file_type(file_type)

    fixed = []
    all_tensors = list(reader.tensors)
    total = len(all_tensors)
    iter_tensors = tqdm(all_tensors, desc="Copying tensors") if on_progress is None else all_tensors

    for idx, tensor in enumerate(iter_tensors):
        if on_progress is not None:
            on_progress(idx + 1, total, tensor.name)

        data = tensor.data
        qtype = tensor.tensor_type

        if tensor.name in PAD_TOKEN_NAMES and data.ndim == 1:
            data = np.expand_dims(data, axis=0)
            _info(f"  reshaped {tensor.name}: {tensor.data.shape} → {data.shape}")
            fixed.append(tensor.name)

        writer.add_tensor(tensor.name, data, raw_dtype=qtype)

    if not fixed:
        _info("No pad tokens needed reshaping — GGUF may already be correct.")

    writer.write_header_to_file(path=dst_path)
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=on_log is None)
    writer.close()

    _info(f"Written: {dst_path}  (reshaped: {fixed})")
    return dst_path


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Fix Lumina2 pad token shapes in a GGUF (1D → [1, D])"
    )
    parser.add_argument("--src", required=True, help="Source GGUF")
    parser.add_argument("--dst", required=True, help="Output GGUF path")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not os.path.isfile(args.src):
        parser.error(f"Source file not found: {args.src}")
    return args


if __name__ == "__main__":
    args = _parse_args()
    fix_pad_tokens(args.src, args.dst, overwrite=args.overwrite)
