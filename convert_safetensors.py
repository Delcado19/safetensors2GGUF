"""Safetensors → Safetensors quantized output.

Sibling to convert.py's GGUF writer: same architecture detection and tensor
loading, different output backend. Unlike GGUF output, no 5D side-car export,
no shape_fix rearrange, no 1D-pad-token unsqueeze — those exist only to
satisfy GGUF/llama-quantize constraints that plain safetensors doesn't have.
"""

from __future__ import annotations

import json
import os
import sys

import torch
from safetensors.torch import save_file

from convert import load_state_dict
from dequantize import (
    _PASSTHROUGH_TENSOR_SUFFIXES,
    _scan_quantized_layers,
    detect_quantized_weight,
    dequantize_weight,
)
from models.architectures import detect_arch
from safetensors_quant import filename_suffix_for, layer_key, quantize_tensor_st
from safetensors_quant_int8 import CONVROT_GROUP_SIZE

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

    out_tensors: dict[str, torch.Tensor] = {}
    layer_formats: dict[str, dict] = {}
    quant_format = _TARGET_TO_QUANT_FORMAT.get(target_key)

    # Iterate state_dict.items() directly (not list(...)) — load_state_dict
    # returns a lazy _LazyStateDict for .safetensors sources that streams one
    # tensor at a time; materializing it into a list here would defeat that
    # and reintroduce the >RAM OOM crash fixed in commit 72a49dc (review
    # finding #3). len() is cheap — _LazyStateDict.__len__ reads the key map,
    # not tensor data.
    total = len(state_dict)
    log_tensor_every = max(1, int(log_tensor_every or 1))
    for idx, (key, data) in enumerate(state_dict.items()):
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("cancelled")
        if on_progress:
            on_progress(idx + 1, total, key)
        if key in quant_skip_keys:
            continue
        if key.endswith(_PASSTHROUGH_TENSOR_SUFFIXES):
            # Not a weight at all -- e.g. "spiece_model", a raw SentencePiece
            # tokenizer blob some Comfy-Org checkpoints embed (see
            # _scan_quantized_layers()'s docstring). Every other tensor below
            # goes through nan_to_num/quantize_tensor_st, both of which
            # assume floating-point weight data; either would corrupt or
            # dtype-cast an opaque byte blob like this one into something
            # SPieceTokenizer can no longer parse. Copy through completely
            # unchanged -- same dtype, same bytes, no quantization.
            out_tensors[key] = data
            continue
        if any(x in key for x in model_arch.keys_ignore):
            continue

        if key in quant_formats:
            # Reconstruct the approximate float original before feeding this
            # layer through the normal quantization path below — `data` here
            # is still the raw on-disk quantized tensor (int8/float8/nvfp4
            # packed uint8), never treated as a plain weight matrix.
            data = dequantize_weight(state_dict, key, quant_formats[key], data)

        old_dtype = data.dtype

        # Coerce dtypes that nan_to_num cannot handle:
        #   float8   → float16  (nan_to_num rejects float8)
        if _FLOAT8_DTYPES and data.dtype in _FLOAT8_DTYPES:
            data = data.to(torch.float16)

        data = torch.nan_to_num(data, nan=0.0, posinf=65504.0, neginf=-65504.0)
        quantized = quantize_tensor_st(data, key, model_arch, target_key)
        if (
            log_tensor_every == 1
            or idx == 0
            or idx + 1 == total
            or (idx + 1) % log_tensor_every == 0
        ):
            _log(f"  {key}  {old_dtype} -> {target_key}")
        out_tensors.update(quantized)
        if quant_format and len(quantized) > 1:
            # ComfyUI's convert_old_quants() writes the comfy_quant sidecar as
            # "{layer}.comfy_quant" where `layer` is the module prefix (no
            # trailing "weight") — must match the scale-tensor naming in
            # safetensors_quant.layer_key(), or the sidecar never lines up
            # with the tensor ComfyUI's loader is inspecting.
            layer_conf: dict = {"format": quant_format}
            if quant_format == "int8_tensorwise":
                # quantize_tensor_st silently falls back per-tensor between
                # ConvRot (per-row weight_scale, ndim>1) and plain tensor-wise
                # (scalar weight_scale) depending on in_features divisibility
                # — the scale tensor's own shape is the single source of
                # truth for which one actually happened for this layer, so
                # read it back rather than threading a second return value
                # through quantize_tensor_st just for this.
                scale_t = quantized.get(f"{layer_key(key)}.weight_scale")
                if scale_t is not None and scale_t.numel() > 1:
                    layer_conf["convrot"] = True
                    layer_conf["convrot_groupsize"] = CONVROT_GROUP_SIZE
            elif quant_format == "float8_e4m3fn" and full_precision_fp8:
                layer_conf["full_precision_matrix_mult"] = True
            elif quant_format == "nvfp4" and full_precision_nvfp4:
                layer_conf["full_precision_matrix_mult"] = True
            layer_formats[layer_key(key)] = layer_conf

    # Same sentinel as the log-line guard above: omit rather than write the
    # misleading literal string "invalid" into the output file's metadata
    # for text encoders (no architecture detection applies to them).
    metadata = {} if model_arch.arch == "invalid" else {"comfy.gguf_source_arch": model_arch.arch}
    if layer_formats:
        metadata["_quantization_metadata"] = json.dumps(
            {"format_version": "1.0", "layers": layer_formats}
        )

    _log(f"INFO:  Writing {len(out_tensors)} tensors → {dst_path}")
    save_file(out_tensors, dst_path, metadata=metadata)
    _log(f"INFO:  Done → {dst_path}")

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
