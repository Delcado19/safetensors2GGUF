"""Quantized Safetensors → Safetensors output backend.

Companion to convert.py's GGUF writer: reuses the same architecture detection
and high-precision-tensor rule (keys_hiprec / 1D / <=1024 elems / >4D) but
writes a plain .safetensors file instead of a GGUF container. See
docs/superpowers/plans/2026-08-05-safetensors-output-and-text-encoder-support.md
for the ComfyUI-compatibility research this format registry is based on
(comfy/quant_ops.py QUANT_ALGOS in city96/ComfyUI-GGUF).
"""

from __future__ import annotations

import torch

from models.architectures import QUANTIZATION_THRESHOLD

# Ordered choices for the GUI dropdown: (display label, key)
#
# FP8/NVFP4 (scaled float8_e4m3fn, NVFP4 blockscaled) are deliberately not
# offered here even though safetensors_quant_fp8.py/safetensors_quant_nvfp4.py
# still implement them correctly (verified byte-exact against ComfyUI's own
# kernels, docs/issues_analysis.md #10-#13). The remaining image corruption on
# Lumina2/Z-Image traces to QUANT_ALGOS["float8_e4m3fn"/"nvfp4"]["quantize_input"]
# defaulting True in ComfyUI itself — both formats dynamically quantize
# *activations* at inference time, and activation quantization is inherently
# lossy in a way no weight-side keys_hiprec list can compensate for (see
# docs/issues_analysis.md #15, including its Correction note: this is NOT
# attributable to Comfy-Org/ComfyUI#14595, which is a performance-only bug).
# int8_tensorwise sets quantize_input=False (weight-only quantization,
# activations always full precision), sidestepping that path entirely.
SAFETENSORS_DTYPE_CHOICES: list[tuple[str, str]] = [
    ("F16       — Half precision",                                       "F16"),
    ("F16 mixed — Half precision, hiprec tensors stay F32",               "F16_MIXED"),
    ("INT8      — Tensor-wise INT8, ConvRot-rotated where possible",      "INT8"),
    ("INT8 mixed — INT8/ConvRot, hiprec tensors stay F32 · recommended ★", "INT8_MIXED"),
]

# Architectures whose keys_hiprec protection scope has been confirmed by an
# actual convert+load+render cycle in ComfyUI (not just by matching a
# community tool's published blacklist) — see docs/issues_analysis.md #15.
# Everything else with a non-empty keys_hiprec is protected on the strength
# of that cross-reference alone, which format_recommendation() below
# discloses rather than implying the same level of confidence for every
# architecture.
_RENDER_VERIFIED_ARCHES = {"lumina2"}


def format_recommendation(model_arch, target_key: str) -> tuple[str, str]:
    """Return (level, message) GUI guidance for one architecture + target
    format pair. ``level`` is "ok" or "warn" (drives badge styling).

    Mirrors quantize.py's static GGUF "recommended ★" label on Q4_K_M, but
    per-architecture: unlike GGUF K-quants (a uniform quality/size tradeoff
    regardless of architecture), this project's own testing found plain INT8
    measurably corrupts output (wrong poses/identities) on the
    attention-sensitive DiT architectures that need keys_hiprec protection,
    while costing nothing to recommend on architectures that don't
    (docs/issues_analysis.md #15).
    """
    mixed = target_key.endswith("_MIXED")
    base = target_key[: -len("_MIXED")] if mixed else target_key
    arch = model_arch.arch
    sensitive = bool(model_arch.keys_hiprec)

    if base == "F16":
        return "ok", "F16 preserves full precision — safe for any architecture."

    if base != "INT8":
        return "ok", ""

    if sensitive and not mixed:
        return "warn", (
            f"**Plain INT8 is not recommended for `{arch}`** — this architecture "
            "has attention/embedder layers known to be quantization-sensitive; "
            "plain INT8 has shown visible pose/identity corruption in testing. "
            "Use **INT8 mixed** instead."
        )
    if sensitive and mixed:
        caveat = (
            "" if arch in _RENDER_VERIFIED_ARCHES
            else " (protection list matched against a community reference, "
                 "not yet confirmed by a render test on this architecture)"
        )
        return "ok", f"**INT8 mixed is recommended for `{arch}`**{caveat}."
    return "ok", (
        f"No quantization-sensitive layers identified for `{arch}` — plain "
        "**INT8** should work well; INT8 mixed costs extra output size for "
        "no measured benefit here."
    )

def layer_key(key: str) -> str:
    """Return the module-prefix ComfyUI's quantized-op loader expects scale/
    comfy_quant sidecar tensors under.

    ComfyUI's per-layer loader (comfy/ops.py _load_quantized_weight_body)
    looks up scale tensors as ``{module_prefix}weight_scale`` — a SIBLING of
    ``{module_prefix}weight``, not nested under it. For a tensor named
    "foo.weight" that means the scale tensor must be named "foo.weight_scale",
    not "foo.weight.weight_scale". Confirmed against a real ComfyUI 0.29.2
    "unet unexpected" key dump that showed our old ``<key>.weight_scale``
    naming left every scale/comfy_quant tensor unconsumed — FP8 loaded raw
    unscaled bytes (visible as noise), NVFP4 crashed outright.
    """
    return key[: -len(".weight")] if key.endswith(".weight") else key


_MIXED_KEYS = {"F16_MIXED", "FP8_MIXED", "NVFP4_MIXED", "INT8_MIXED"}
_BASE_KEY = {
    "F16": "F16", "F16_MIXED": "F16",
    "FP8": "FP8", "FP8_MIXED": "FP8",
    "NVFP4": "NVFP4", "NVFP4_MIXED": "NVFP4",
    "INT8": "INT8", "INT8_MIXED": "INT8",
}


def is_hiprec_st(key: str, data: torch.Tensor, model_arch, old_dtype: torch.dtype) -> bool:
    """Return True if ``key`` must stay high-precision (F32), mirroring
    convert._quant_type_for's rule so 'mixed' safetensors output matches the
    existing GGUF mixed-precision behaviour exactly."""
    if old_dtype not in (torch.float32, torch.bfloat16):
        return False
    n_dims = data.dim()
    if n_dims == 1:
        return True
    if data.numel() <= QUANTIZATION_THRESHOLD:
        return True
    if any(x in key for x in model_arch.keys_hiprec):
        return True
    return False


def quantize_tensor_st(
    data: torch.Tensor, key: str, model_arch, target_key: str
) -> dict[str, torch.Tensor]:
    """Quantize one tensor for safetensors output.

    Returns a dict of {tensor_name: tensor} — one entry for F16/FP8-unscaled,
    multiple entries (weight + scale tensors) for FP8-scaled and NVFP4.
    """
    old_dtype = data.dtype
    base = _BASE_KEY[target_key]
    mixed = target_key in _MIXED_KEYS

    if mixed and is_hiprec_st(key, data, model_arch, old_dtype):
        # Keep the tensor's ORIGINAL dtype, not a forced F32 upcast: is_hiprec_st
        # only ever returns True when old_dtype is already float32 or bfloat16
        # (its own gate), so this is a no-op for float32 sources but avoids
        # doubling every bfloat16 hiprec tensor's on-disk size for zero
        # precision benefit — ComfyUI casts every loaded weight to its own
        # compute_dtype regardless of what's on disk, so storing bf16 as F32
        # here buys nothing at inference time. Previously invisible because
        # `keys_hiprec` only covered a handful of small tensors per model; once
        # it grew to cover a large fraction of a model (Lumina2's attention +
        # modulation protection, docs/issues_analysis.md #15) the forced F32
        # upcast made INT8_MIXED output *larger* than the unquantized bf16
        # source — defeating the point of quantizing at all.
        return {key: data.to(old_dtype)}

    if base == "F16":
        return {key: data.to(torch.float16)}

    # FP8/NVFP4 scale-tensor conventions only make sense for weight matrices:
    # ComfyUI's scaled-quant loader applies a layer's .weight_scale only to its
    # .weight tensor, so a quantized 1D tensor (bias, norm weight) would load
    # back unscaled and be wrong by up to ~448x. Unlike the `mixed and
    # is_hiprec_st(...)` check above, this must apply unconditionally — not
    # just in *_MIXED mode — because there is no accuracy or size benefit to
    # scale-quantizing tiny 1D tensors either way. Mirrors convert.py's
    # _quant_type_for, which always keeps 1D tensors at F32 regardless of
    # target_quant (review finding #2).
    if data.dim() == 1:
        return {key: data.to(torch.float32)}

    if base == "FP8":
        from safetensors_quant_fp8 import quantize_fp8_scaled
        return quantize_fp8_scaled(data, key)

    if base == "NVFP4":
        from safetensors_quant_nvfp4 import quantize_nvfp4

        # Shape-safety, not precision — applies regardless of `mixed`. NVFP4
        # halves the on-disk last dim (2 values/byte); for tensors ComfyUI's
        # model_detection.py reads raw to infer architecture hyperparameters
        # (see models.architectures.ModelTemplate.keys_shape_critical), that
        # corrupts the inferred config and crashes model loading downstream —
        # confirmed for Lumina2/Z-Image's cap_embedder.1.weight against a live
        # ComfyUI install (docs/issues_analysis.md #9).
        if any(x in key for x in getattr(model_arch, "keys_shape_critical", [])):
            return {key: data.to(torch.float16)}

        try:
            return quantize_nvfp4(data, key)
        except ValueError:
            # Tensor's last dim isn't a multiple of 16 (e.g. 3x3 conv kernels
            # with last dim 3, or some DiT patch-embed layers) — NVFP4's
            # 16-element block packing can't apply. Fall back to a plain F16
            # write for this one tensor instead of crashing the whole
            # conversion after many tensors have already been processed
            # (review finding #1). No on_log hook exists at this call depth,
            # so this fallback is silent by design.
            return {key: data.to(torch.float16)}

    if base == "INT8":
        from safetensors_quant_int8 import quantize_int8_convrot, quantize_int8_tensorwise

        # Shape-safety, not precision — same rationale as the NVFP4 branch
        # above: tensors ComfyUI reads raw (pre-dequant) to infer architecture
        # hyperparameters must never change on-disk shape.
        if any(x in key for x in getattr(model_arch, "keys_shape_critical", [])):
            return {key: data.to(torch.float16)}

        if data.dim() != 2:
            # ConvRot's block-Hadamard rotation only makes sense on a 2D
            # [out_features, in_features] weight; plain tensor-wise INT8 has
            # no such restriction (single scalar scale over the whole tensor).
            return quantize_int8_tensorwise(data, key)

        try:
            return quantize_int8_convrot(data, key)
        except ValueError:
            # in_features not a multiple of CONVROT_GROUP_SIZE (256) — fall
            # back to plain (unrotated) tensor-wise INT8 for this one tensor.
            return quantize_int8_tensorwise(data, key)

    raise ValueError(f"Unknown target_key: {target_key!r}")
