"""Re-insert 5D tensors into a GGUF file after llama-quantize.

During conversion, models with 5D tensors (e.g. HunyuanVideo, Wan) export
those tensors to a side-car safetensors file because the GGUF format and
llama-quantize only support up to 4D. After quantizing the main GGUF, run
this script to insert the 5D tensors back as F32.

Usage:
    python fix_5d_tensors.py --src model-Q8_0.gguf --dst model-Q8_0-fixed.gguf
    python fix_5d_tensors.py --src model.gguf --dst out.gguf --fix custom_fix.safetensors
"""

import os
import argparse
import logging

import gguf
import numpy as np
import torch
from tqdm import tqdm
from conversion_io import atomic_output, validate_output
from convert import ConversionCancelled
from safetensors.torch import load_file

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _read_arch(reader):
    field = reader.get_field("general.architecture")
    if field is None:
        raise ValueError("GGUF file has no 'general.architecture' field.")
    return str(field.parts[field.data[-1]], encoding="utf-8")


def _read_file_type(reader):
    field = reader.get_field("general.file_type")
    if field is None:
        return None
    # Use .flat[0] to handle both 0D scalars and size-1 1D arrays (Issue #384)
    return gguf.LlamaFileType(int(field.parts[field.data[-1]].flat[0]))


def _add_5d_tensor(writer, key, data, on_log=None):
    """Write a single 5D tensor as F32 into the GGUF writer."""
    arr = data.numpy() if isinstance(data, torch.Tensor) else data
    arr = arr.astype(np.float32)
    quantized = gguf.quants.quantize(arr, gguf.GGMLQuantizationType.F32)
    writer.add_tensor(key, quantized, raw_dtype=gguf.GGMLQuantizationType.F32)
    msg = f"  inserting 5D tensor: {key} {arr.shape}"
    if on_log:
        on_log(msg)
    else:
        tqdm.write(msg)


def fix_5d_tensors(src_path, dst_path, fix_path=None, overwrite=False, on_progress=None, on_log=None, cancel_event=None):
    """Insert 5D tensors from fix_path into src GGUF and write to dst_path.

    Args:
        src_path: Quantized source GGUF file.
        dst_path: Output GGUF path.
        fix_path: Side-car safetensors file. Defaults to the GGUF metadata
            basename, then <src>.5d.safetensors or the legacy architecture name.
        overwrite: Skip output existence check.
        on_progress: Callback(idx, total, key) for each tensor copied.
        on_log: Callback(msg) for informational messages. Uses logging when None.
        cancel_event: Optional threading.Event; aborts without replacing the output.

    Returns:
        List of inserted tensor keys.
    """
    def _info(msg):
        if on_log:
            on_log(f"INFO:  {msg}")
        else:
            logging.info(msg)

    if not os.path.isfile(src_path):
        raise FileNotFoundError(f"Source GGUF not found: {src_path}")

    reader = gguf.GGUFReader(src_path)
    arch = _read_arch(reader)
    file_type = _read_file_type(reader)
    _info(f"Source arch: '{arch}'  file_type: {file_type}")

    if fix_path is None:
        sidecar = reader.get_field("comfy.gguf.5d_sidecar")
        if sidecar is not None:
            name = str(sidecar.parts[sidecar.data[-1]], encoding="utf-8")
            # The metadata stores a basename, so moving both files still works.
            fix_path = os.path.join(os.path.dirname(src_path), os.path.basename(name))
        else:
            fix_path = str(src_path) + ".5d.safetensors"
            if not os.path.isfile(fix_path):
                # Legacy manually exported side-cars remain supported.
                fix_path = f"./fix_5d_tensors_{arch}.safetensors"
    if not os.path.isfile(fix_path):
        raise FileNotFoundError(f"5D tensor fix file not found: {fix_path}")

    # Clone immediately to release the mmap before writing (Windows)
    fix_sd = {k: v.clone() for k, v in load_file(fix_path).items()}
    _info(f"5D tensors to insert: {list(fix_sd.keys())}")

    validate_output(src_path, dst_path, overwrite)
    validate_output(fix_path, dst_path, overwrite)

    writer = gguf.GGUFWriter(path=None, arch=arch)
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    if file_type is not None:
        writer.add_file_type(file_type)

    inserted = []
    all_tensors = list(reader.tensors)
    total = len(all_tensors)
    iter_tensors = tqdm(all_tensors, desc="Copying tensors") if on_progress is None else all_tensors

    for idx, tensor in enumerate(iter_tensors):
        if cancel_event is not None and cancel_event.is_set():
            raise ConversionCancelled()
        if on_progress is not None:
            on_progress(idx + 1, total, tensor.name)
        writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)

        # Insert 5D tensor after its bias counterpart if present
        candidate = tensor.name.replace(".bias", ".weight")
        if candidate in fix_sd and candidate not in inserted:
            _add_5d_tensor(writer, candidate, fix_sd[candidate], on_log=on_log)
            inserted.append(candidate)

    # Insert any remaining 5D tensors not yet placed
    for key, data in fix_sd.items():
        if cancel_event is not None and cancel_event.is_set():
            raise ConversionCancelled()
        if key not in inserted:
            _add_5d_tensor(writer, key, data, on_log=on_log)
            inserted.append(key)

    with atomic_output(dst_path, overwrite) as temporary:
        try:
            writer.write_header_to_file(path=temporary)
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file(progress=on_log is None)
        finally:
            # Release the Windows handle before temporary-file cleanup.
            writer.close()
        if cancel_event is not None and cancel_event.is_set():
            raise ConversionCancelled()

    _info(f"Written: {dst_path}  (inserted {len(inserted)} 5D tensors)")
    return inserted


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Re-insert 5D tensors into a quantized GGUF file"
    )
    parser.add_argument("--src", required=True, help="Source GGUF (quantized)")
    parser.add_argument("--dst", required=True, help="Output GGUF path")
    parser.add_argument("--fix", help="Side-car safetensors (default: GGUF side-car metadata, then legacy filenames)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not os.path.isfile(args.src):
        parser.error(f"Source file not found: {args.src}")
    return args


if __name__ == "__main__":
    args = _parse_args()
    fix_5d_tensors(args.src, args.dst, fix_path=args.fix, overwrite=args.overwrite)
