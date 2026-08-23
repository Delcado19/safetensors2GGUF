"""Safetensors → Safetensors quantized output.

Sibling to convert.py's GGUF writer: same architecture detection and tensor
loading, different output backend. Unlike GGUF output, no 5D side-car export,
no shape_fix rearrange, no 1D-pad-token unsqueeze — those exist only to
satisfy GGUF/llama-quantize constraints that plain safetensors doesn't have.
"""

from __future__ import annotations

import json
import os
import struct
import sys

import torch

from convert import _TORCH_TO_ST_DTYPE, load_state_dict
from dequantize import (
    _PASSTHROUGH_TENSOR_SUFFIXES,
    _scan_quantized_layers,
    dequantized_shape_of,
    dequantize_weight,
)
from models.architectures import detect_arch
from safetensors_quant import filename_suffix_for, layer_key, plan_tensor_output, quantize_tensor_st

# Float8 dtypes — resolved once at import time; empty tuple on older PyTorch builds.
_FLOAT8_DTYPES: tuple = tuple(
    d for d in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2",   None),
    )
    if d is not None
)

_TARGET_TO_QUANT_FORMAT = {
    "FP8": "float8_e4m3fn", "FP8_MIXED": "float8_e4m3fn",
    "NVFP4": "nvfp4", "NVFP4_MIXED": "nvfp4",
    "INT8": "int8_tensorwise", "INT8_MIXED": "int8_tensorwise",
}


def _iter_output_keys(state_dict, model_arch, quant_skip_keys):
    """Yield (key, is_passthrough) from state_dict in order, applying the
    same skip/passthrough/keys_ignore filtering convert_to_safetensors()'s
    main loop always has -- factored out so the streaming writer's planning
    pass (Pass 1) and quantizing pass (Pass 2) iterate identically. Both
    passes MUST see the exact same keys in the exact same order, or Pass 2's
    tensors won't line up with Pass 1's planned header (see convert_
    to_safetensors()'s Pass 2 assert)."""
    for key in state_dict.keys():
        if key in quant_skip_keys:
            continue
        if key.endswith(_PASSTHROUGH_TENSOR_SUFFIXES):
            yield key, True
            continue
        if any(x in key for x in model_arch.keys_ignore):
            continue
        yield key, False


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    """Raw little-endian bytes for one tensor, in safetensors' flat storage
    convention. Reinterprets through a uint8 view instead of calling
    tensor.numpy() directly -- numpy has no bfloat16/float8_e4m3fn/
    float8_e5m2 support on most builds, and .numpy() raises for those
    dtypes. torch.Tensor.view(dtype) is a pure reinterpret-cast (no data
    copy beyond what .contiguous() already needs)."""
    tensor = tensor.reshape(1) if tensor.dim() == 0 else tensor
    return tensor.contiguous().view(torch.uint8).numpy().tobytes()


def _build_header(
    entries: list[tuple[str, str, tuple[int, ...]]], metadata: dict
) -> tuple[dict, int]:
    """Compute the safetensors JSON header (name -> dtype/shape/data_offsets)
    and total data-section byte size, from a flat, ordered list of (name,
    dtype, shape) tuples. Offsets are assigned in list order -- Pass 2
    (convert_to_safetensors) MUST produce tensors in this same order for
    the file to be valid; this function itself doesn't enforce that, the
    assert in Pass 2 does."""
    from safetensors_quant import _ST_DTYPE_BYTES

    header: dict = {}
    offset = 0
    for name, dtype, shape in entries:
        n_elems = 1
        for d in shape:
            n_elems *= d
        size = n_elems * _ST_DTYPE_BYTES[dtype]
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    if metadata:
        header["__metadata__"] = metadata
    return header, offset


def _write_header(fh, header: dict) -> int:
    """Write the safetensors 8-byte little-endian header-length prefix plus
    the JSON header itself to an already-open binary file handle. Pads the
    JSON body with trailing spaces to a multiple of 8 bytes, matching the
    reference Rust safetensors serializer's convention (not required by the
    format spec for correctness -- readers only ever consult the declared
    length prefix -- but matched here for closest byte-level parity with
    files this project's old save_file()-based writer produced). Returns
    the number of header bytes written (prefix + JSON, informational)."""
    body = json.dumps(header).encode("utf-8")
    pad = (-len(body)) % 8
    body += b" " * pad
    fh.write(struct.pack("<Q", len(body)))
    fh.write(body)
    return 8 + len(body)


def convert_to_safetensors(
    path,
    dst_path=None,
    target_key="FP8",
    overwrite=False,
    on_progress=None,
    on_log=None,
    cancel_event=None,
    model_arch=None,
    log_tensor_every=1,
    full_precision_fp8=True,
    full_precision_nvfp4=True,
    strip_prefixes=True,
):
    """Convert a model checkpoint to a quantized .safetensors file.

    Args:
        path: Source model file path.
        dst_path: Output path; auto-generated as ``<src>-<target_key>.safetensors`` when None.
        target_key: One of safetensors_quant.SAFETENSORS_DTYPE_CHOICES' keys.
        overwrite: Skip existence check when True.
        on_progress: Optional callback(idx, total, key).
        on_log: Optional callback(msg); prints when None.
        log_tensor_every: Emit a per-tensor log line every Nth tensor (plus the
            first/last), mirroring convert.py's handle_tensors so the GUI log
            box shows which tensor is being processed, not just the arch/done
            lines. Values below 1 are treated as 1 (log every tensor).
        cancel_event: Optional threading.Event; raises RuntimeError("cancelled") when set.
        model_arch: Pre-resolved architecture instance (e.g. a bare
            ``models.architectures.ModelTemplate()`` for text-encoder checkpoints,
            which aren't in ``arch_list`` and would otherwise fail ``detect_arch``).
            When None (default), auto-detected via ``detect_arch`` as before.
        full_precision_fp8: When target_key resolves to float8_e4m3fn (FP8/
            FP8_MIXED), write "full_precision_matrix_mult": true into every
            layer's .comfy_quant config (default True). This makes ComfyUI's
            MixedPrecisionOps.Linear.forward() skip the quantized-compute
            branch entirely for that layer — weight is dequantized to the
            model's compute_dtype and a plain full-precision matmul runs,
            matching the safety profile of the "scaled_fp8" checkpoints
            ComfyUI's own convert_old_quants() derives this flag from for
            legacy community checkpoints (comfy/utils.py). This is what
            makes FP8 output architecture-independently safe: unlike INT8's
            keys_hiprec (a per-architecture, per-layer bet on which tensors
            need protection), this flag disables the risky dynamic-
            activation-quantization code path for every layer, unconditionally.
            The tradeoff is no FP8 tensor-core compute speedup — this format
            is storage/VRAM savings only unless a user explicitly opts out.
            See docs/issues_analysis.md #16.
        full_precision_nvfp4: Same mechanism as full_precision_fp8, applied
            to NVFP4/NVFP4_MIXED (default True). comfy/ops.py's
            _load_quantized_module() reads "full_precision_matrix_mult" from
            the per-layer .comfy_quant config generically — the gate at
            MixedPrecisionOps.Linear.forward() (`not self._full_precision_mm`)
            is not format-specific, it applies identically regardless of
            quant_format. docs/issues_analysis.md #16's original "NVFP4 has
            no equivalent safe mode" conclusion was based on ComfyUI's
            legacy-checkpoint convert_old_quants() only synthesizing this
            flag for scaled_fp8 checkpoints — true for checkpoints ComfyUI
            itself upgrades, but this tool controls its own .comfy_quant
            output directly and was simply never setting the flag for nvfp4.
            Verified by reading comfy/ops.py's actual source (a local
            ComfyUI-Easy-Install checkout, comfy_kitchen 0.2.30) — NOT
            yet render-tested in ComfyUI. lumina2 was previously
            _RENDER_CONFIRMED_BAD for NVFP4/NVFP4_MIXED and flux was CAUTION
            (visible composition drift) under the OLD writer that never set
            this flag; both need re-testing against this fix before either
            model_support.py table changes.
        strip_prefixes: Passed through to ``load_state_dict()``. Diffusion
            checkpoints often wrap the UNet in a "model."/"model.diffusion_model."
            prefix that must be stripped for architecture detection and clean
            output keys (default True). Standalone text-encoder checkpoints
            genuinely use "model." as their own real module path (e.g. Qwen3's
            "model.layers.0...") — stripping it there breaks ComfyUI's own
            text-encoder architecture detection on the output file. Callers
            for text-encoder sources must pass False.

    Returns:
        (dst_path, model_arch)
    """
    def _log(msg):
        if on_log:
            on_log(msg)
        else:
            # Log lines contain non-ASCII characters (e.g. "->" as U+2192);
            # a Windows console on a legacy codepage (cp1252) raises
            # UnicodeEncodeError on print() and aborts the conversion before
            # save_file() runs. Fall back to a codepage-safe transliteration
            # instead of crashing mid-run.
            try:
                print(msg)
            except UnicodeEncodeError:
                enc = sys.stdout.encoding or "ascii"
                print(msg.encode(enc, errors="replace").decode(enc))

    state_dict = load_state_dict(path, strip_prefixes=strip_prefixes)
    quant_formats, quant_skip_keys = _scan_quantized_layers(state_dict)
    if quant_formats:
        _log(
            f"INFO:  Dequantizing {len(quant_formats)} already-quantized "
            f"layer(s) before re-quantizing (formats: "
            f"{sorted(set(quant_formats.values()))})"
        )
    if model_arch is None:
        model_arch = detect_arch(state_dict)
    # ModelTemplate's own base-class default ("invalid") is the deliberate
    # sentinel text_encoder_convert.py passes for text encoders (no
    # per-architecture detection applies to them, see models/architectures.py) --
    # not a real failure, so don't log a confusing "Architecture: invalid" line.
    if model_arch.arch != "invalid":
        _log(f"INFO:  Architecture: {model_arch.arch}")

    if dst_path is None:
        # filename_suffix_for() spells out FP8/FP8_MIXED's external
        # "fp8_e4m3fn_scaled" naming (Civitai/Comfy-Org convention) instead
        # of the internal target_key -- see safetensors_quant.py's
        # _FILENAME_SUFFIX comment. Every other key's suffix is unchanged.
        dst_path = f"{os.path.splitext(path)[0]}-{filename_suffix_for(target_key)}.safetensors"

    if os.path.isfile(dst_path) and not overwrite:
        raise OSError(f"Output exists and overwrite is disabled: {dst_path}")

    # --- Pass 1: plan every output tensor's (name, dtype, shape) and the
    # full _quantization_metadata, from shape/dtype metadata alone -- no
    # tensor data touched. Must use the exact same key set/order as Pass 2
    # below (_iter_output_keys is the shared contract that guarantees this).
    shape_of = getattr(state_dict, "shape_of", None)
    dtype_of = getattr(state_dict, "dtype_of", None)

    def _shape_dtype(k):
        if shape_of is not None:
            return tuple(shape_of(k)), dtype_of(k)
        t = state_dict[k]
        return tuple(t.shape), t.dtype

    entries: list[tuple[str, str, tuple]] = []
    layer_formats: dict[str, dict] = {}

    for key, is_passthrough in _iter_output_keys(state_dict, model_arch, quant_skip_keys):
        shape, dtype = _shape_dtype(key)
        if is_passthrough:
            entries.append((key, _TORCH_TO_ST_DTYPE[dtype], shape))
            continue

        if key in quant_formats:
            shape = dequantized_shape_of(quant_formats[key], shape)
            dtype = torch.float32
        if _FLOAT8_DTYPES and dtype in _FLOAT8_DTYPES:
            dtype = torch.float16

        out_entries, layer_conf = plan_tensor_output(
            key, shape, dtype, model_arch, target_key,
            full_precision_fp8, full_precision_nvfp4,
        )
        entries.extend(out_entries)
        if layer_conf is not None:
            layer_formats[layer_key(key)] = layer_conf

    metadata = {} if model_arch.arch == "invalid" else {"comfy.gguf_source_arch": model_arch.arch}
    if layer_formats:
        metadata["_quantization_metadata"] = json.dumps(
            {"format_version": "1.0", "layers": layer_formats}
        )

    header, _total_data_bytes = _build_header(entries, metadata)

    _log(f"INFO:  Writing {len(entries)} tensors -> {dst_path}")

    # --- Pass 2: re-run the REAL quantization per key, streaming each
    # result's bytes to disk the instant it's produced instead of
    # accumulating a dict -- this is the actual RAM fix. Tensors are
    # written strictly in Pass 1's planned order (guaranteed by using the
    # same _iter_output_keys + plan_tensor_output/quantize_tensor_st
    # branching), so no seeking or offset lookup is needed, just sequential
    # appends after the header.
    total = len(state_dict)
    log_tensor_every = max(1, int(log_tensor_every or 1))
    entry_idx = 0
    idx = 0
    with open(dst_path, "wb") as fh:
        _write_header(fh, header)

        for key, is_passthrough in _iter_output_keys(state_dict, model_arch, quant_skip_keys):
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("cancelled")
            if on_progress:
                on_progress(idx + 1, total, key)
            idx += 1

            if is_passthrough:
                data = state_dict[key]
                exp_name, exp_dtype, exp_shape = entries[entry_idx]
                entry_idx += 1
                assert key == exp_name and tuple(data.shape) == exp_shape, (
                    f"Streaming writer plan mismatch for passthrough {key!r}: "
                    f"planned shape {exp_shape}, got {tuple(data.shape)}"
                )
                fh.write(_tensor_bytes(data))
                continue

            data = state_dict[key]
            if key in quant_formats:
                data = dequantize_weight(state_dict, key, quant_formats[key], data)
            old_dtype = data.dtype
            if _FLOAT8_DTYPES and data.dtype in _FLOAT8_DTYPES:
                data = data.to(torch.float16)
            data = torch.nan_to_num(data, nan=0.0, posinf=65504.0, neginf=-65504.0)
            quantized = quantize_tensor_st(data, key, model_arch, target_key)
            if (
                log_tensor_every == 1
                or idx == 1
                or idx == total
                or idx % log_tensor_every == 0
            ):
                _log(f"  {key}  {old_dtype} -> {target_key}")

            for name, tensor in quantized.items():
                exp_name, exp_dtype, exp_shape = entries[entry_idx]
                entry_idx += 1
                assert (
                    name == exp_name
                    and tuple(tensor.shape) == exp_shape
                    and _TORCH_TO_ST_DTYPE[tensor.dtype] == exp_dtype
                ), (
                    f"Streaming writer plan mismatch for {name!r}: planned "
                    f"{exp_dtype}/{exp_shape}, got real "
                    f"{_TORCH_TO_ST_DTYPE[tensor.dtype]}/{tuple(tensor.shape)} -- "
                    "Pass 1 (plan_tensor_output) and Pass 2 (quantize_tensor_st) "
                    "have drifted apart"
                )
                fh.write(_tensor_bytes(tensor))

    _log(f"INFO:  Done -> {dst_path}")

    # Quantization is supposed to shrink the file; scale/zero-point tensors
    # added per layer can outgrow the savings on already-small or oddly-shaped
    # models, silently producing an output that is both bigger AND lower
    # precision than the input. Warn rather than fail the conversion.
    src_size = os.path.getsize(path)
    dst_size = os.path.getsize(dst_path)
    if dst_size > src_size:
        _log(
            f"WARNING:  Output is larger than the input "
            f"({dst_size / 1e6:.1f} MB > {src_size / 1e6:.1f} MB) despite quantizing to "
            f"'{target_key}' — the precision loss bought no size reduction."
        )

    return dst_path, model_arch
