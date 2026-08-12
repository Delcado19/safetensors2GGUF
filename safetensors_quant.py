"""Quantized Safetensors → Safetensors output backend.

Companion to convert.py's GGUF writer: reuses the same architecture detection
and high-precision-tensor rule (keys_hiprec / 1D / <=1024 elems / >4D) but
writes a plain .safetensors file instead of a GGUF container. See
docs/superpowers/plans/2026-08-05-safetensors-output-and-text-encoder-support.md
for the ComfyUI-compatibility research this format registry is based on
(comfy/quant_ops.py QUANT_ALGOS in city96/ComfyUI-GGUF).
"""

from __future__ import annotations

import json
import struct

import torch

from models.architectures import QUANTIZATION_THRESHOLD

# Ordered choices for the GUI dropdown: (display label, key)
#
# FP8 was removed for a session, then re-added once this project's own
# writer started defaulting to full_precision_matrix_mult=true
# (convert_safetensors.py) -- that flag makes ComfyUI skip the risky
# dynamic-activation-quantization compute path entirely for every FP8
# layer, matching the safety profile of the "scaled_fp8" checkpoints
# already circulating on Civitai/HuggingFace (see docs/issues_analysis.md
# #16). NVFP4 has no equivalent verified safe mode yet (see
# safetensors_quant_nvfp4.py and docs/issues_analysis.md #15) and stays
# unoffered here; its writer/tests remain in the codebase.
SAFETENSORS_DTYPE_CHOICES: list[tuple[str, str]] = [
    ("F16       — Half precision",                                       "F16"),
    ("F16 mixed — Half precision, hiprec tensors stay F32",               "F16_MIXED"),
    ("FP8       — Scaled float8_e4m3fn, full-precision compute (safe)",   "FP8"),
    ("FP8 mixed — FP8, hiprec tensors stay F32",                          "FP8_MIXED"),
    ("INT8      — Tensor-wise INT8, ConvRot-rotated where possible",      "INT8"),
    ("INT8 mixed — INT8/ConvRot, hiprec tensors stay F32 · recommended ★", "INT8_MIXED"),
]

# (arch_key, format_key) pairs whose *_MIXED keys_hiprec protection has been
# confirmed by an actual convert+load+render cycle in ComfyUI (not just by
# matching a community tool's published blacklist) -- see
# docs/issues_analysis.md #15 (INT8_MIXED) and #16's second correction
# (FP8_MIXED: same-seed/prompt comparison, 2026-08-11, showed identity/
# composition/outfit preserved -- the only deviation was a single secondary
# prop, judged tolerable quantization variance, not a correctness failure).
# Everything else with a non-empty keys_hiprec is protected on the strength
# of a community-tool cross-reference alone, which format_recommendation()
# below discloses rather than implying the same level of confidence for
# every architecture.
_RENDER_VERIFIED_MIXED: set[tuple[str, str]] = {
    ("lumina2", "INT8_MIXED"),
    ("lumina2", "FP8_MIXED"),
}

# (base_format -> {arch_key}) pairs where the PLAIN (non-mixed) output has
# been directly render-tested and confirmed wrong -- not just predicted by
# analogy from another format's corruption. INT8: docs/issues_analysis.md
# #15. FP8: same-seed/prompt user comparison (2026-08-11) showing the
# standing figure's outfit and the room background changed between the
# unquantized and plain-FP8 renders, only the pose skeleton surviving --
# the same "wrong identity/composition" pattern #15 documented, now
# confirmed for FP8 too rather than just predicted from INT8's mechanism.
_RENDER_CONFIRMED_BAD_PLAIN: dict[str, set[str]] = {
    "INT8": {"lumina2"},
    "FP8": {"lumina2"},
}


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

    Only `lumina2` has actually been render-tested this way — every other
    architecture's keys_hiprec scope was adopted from cross-referencing
    community tools (tritant/Starnodes), never confirmed by converting +
    loading + rendering with THIS tool's output. That distinction must
    survive into both the warn and the ok message: claiming plain INT8
    "has shown visible corruption in testing" for an architecture nobody
    has actually rendered would overstate what we know, not just for the
    positive (INT8_MIXED-is-fine) case but symmetrically for the negative
    (plain-INT8-is-risky) case too.
    """
    mixed = target_key.endswith("_MIXED")
    base = target_key[: -len("_MIXED")] if mixed else target_key
    arch = model_arch.arch
    sensitive = bool(model_arch.keys_hiprec)
    mixed_verified = (arch, target_key) in _RENDER_VERIFIED_MIXED
    # Separate from the above: whether PLAIN (non-mixed) output for this
    # exact base format has itself been directly render-tested and confirmed
    # wrong, vs. only predicted by analogy from another format's corruption.
    plain_confirmed_bad = arch in _RENDER_CONFIRMED_BAD_PLAIN.get(base, set())

    if base == "F16":
        return "ok", "F16 preserves full precision — safe for any architecture."

    if base not in ("FP8", "INT8"):
        return "ok", ""

    # full_precision_matrix_mult only changes ComfyUI's runtime matmul path —
    # it does nothing for precision already lost when a tensor was quantized
    # to e4m3 on disk. quantize_tensor_st() only skips quantizing
    # keys_hiprec-sensitive tensors when `mixed` is True (is_hiprec_st gate),
    # identically for FP8 and INT8, so plain FP8 carries the same
    # sensitive-architecture risk plain INT8 does and needs the same warning.
    fmt_label = "FP8" if base == "FP8" else "INT8"
    mixed_label = "FP8 mixed" if base == "FP8" else "INT8 mixed"

    if sensitive and not mixed:
        if plain_confirmed_bad:
            evidence = f"plain {fmt_label} has shown visible pose/identity corruption in testing"
        else:
            evidence = (
                "the same risk caused visible pose/identity corruption on `lumina2` "
                "with plain INT8 — this hasn't been render-tested on this specific "
                "architecture, but weight-only vs. full precision isn't an "
                "architecture-specific distinction either"
            )
        return "warn", (
            f"**Plain {fmt_label} is not recommended for `{arch}`** — this architecture "
            f"has layers known to be quantization-sensitive; {evidence}. "
            f"Use **{mixed_label}** instead."
        )
    if sensitive and mixed:
        caveat = (
            "" if mixed_verified
            else " (protection list matched against a community reference, "
                 "not yet confirmed by a render test on this architecture)"
        )
        return "ok", f"**{mixed_label} is recommended for `{arch}`**{caveat}."
    if base == "FP8":
        return "ok", (
            "FP8 defaults to full-precision compute (`full_precision_matrix_mult`) "
            "— safe on any architecture, same mechanism as most circulating "
            "Civitai/HuggingFace FP8 checkpoints. No FP8 tensor-core speedup "
            "without an explicit opt-out (docs/issues_analysis.md #16)."
        )
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


_ST_DTYPE_BYTES: dict[str, int] = {
    "F32": 4, "F64": 8, "F16": 2, "BF16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
}
_CONVROT_GROUP_SIZE = 256  # must match safetensors_quant_int8.py's own constant


def estimate_safetensors_output_size(path: str, target_key: str, model_arch) -> int | None:
    """Estimate quantize_tensor_st()'s output size in bytes by analysing the
    safetensors header (no tensor data loaded) -- the safetensors-output
    equivalent of quantize.py's _estimate_from_safetensors() for GGUF.

    Replicates quantize_tensor_st's exact per-tensor branching (is_hiprec_st's
    1D/small/keys_hiprec gate, keys_shape_critical fallback, NVFP4's last-dim%16
    and INT8 ConvRot's last-dim%256 shape requirements) rather than GGUF's
    simpler unconditional "1D always F32" rule -- necessary because the
    *_MIXED variants' size depends on model_arch.keys_hiprec, which GGUF
    output doesn't have an analogous per-format dependency on.

    Returns None if the file can't be read as a safetensors header.
    """
    try:
        with open(path, "rb") as fh:
            raw_len = fh.read(8)
            if len(raw_len) < 8:
                return None
            header_len = struct.unpack("<Q", raw_len)[0]
            header = json.loads(fh.read(header_len).decode("utf-8", errors="replace"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None

    base = _BASE_KEY.get(target_key)
    if base is None:
        return None
    mixed = target_key in _MIXED_KEYS
    hiprec_substrings = list(getattr(model_arch, "keys_hiprec", None) or [])
    shape_critical_substrings = list(getattr(model_arch, "keys_shape_critical", None) or [])

    total = 0
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        shape = meta.get("shape")
        if not shape and shape != []:
            continue
        n_elems = 1
        for d in shape:
            n_elems *= d
        ndim = len(shape)
        src_dtype = meta.get("dtype", "F16")
        src_bytes = _ST_DTYPE_BYTES.get(src_dtype, 2)

        # Mirrors is_hiprec_st(): only F32/BF16 sources are eligible, and 1D
        # or <=QUANTIZATION_THRESHOLD-element tensors qualify unconditionally,
        # not just via a keys_hiprec substring match.
        is_hiprec = src_dtype in ("F32", "BF16") and (
            ndim == 1 or n_elems <= QUANTIZATION_THRESHOLD
            or any(s in name for s in hiprec_substrings)
        )
        if mixed and is_hiprec:
            total += n_elems * src_bytes
            continue

        if base == "F16":
            total += n_elems * 2
            continue

        if ndim == 1:
            total += n_elems * 4  # F32 -- unconditional, matches quantize_tensor_st
            continue

        if base == "FP8":
            total += n_elems * 1 + 4  # float8_e4m3fn + one F32 scale scalar
            continue

        if base == "NVFP4":
            shape_critical = any(s in name for s in shape_critical_substrings)
            block_ok = shape[-1] % 16 == 0
            if shape_critical or not block_ok:
                total += n_elems * 2  # F16 fallback (shape-critical or non-block-aligned)
            else:
                packed_bytes = n_elems // 2  # 2 elems/byte
                k_blocks = shape[-1] // 16
                if ndim == 2:
                    # 2D weights go through _swizzle_block_scale: padded to
                    # 128x4-tile alignment before storage, not the bare
                    # (rows, k_blocks) shape -- can matter a lot for weights
                    # with < 128 output features.
                    m_padded = -(-shape[0] // 128) * 128
                    k_padded = -(-k_blocks // 4) * 4
                    scale_bytes = m_padded * k_padded
                else:
                    # Higher-rank tensors skip the swizzle (see quantize_nvfp4's
                    # `if len(lead) == 1` guard) -- plain (rows, k_blocks) layout.
                    scale_bytes = (n_elems // shape[-1]) * k_blocks
                total += packed_bytes + scale_bytes + 4  # + one F32 global scale scalar
            continue

        if base == "INT8":
            shape_critical = any(s in name for s in shape_critical_substrings)
            if shape_critical:
                total += n_elems * 2  # F16 fallback
            elif ndim == 2 and shape[-1] % _CONVROT_GROUP_SIZE == 0:
                total += n_elems * 1 + shape[0] * 4  # ConvRot: int8 + one F32 scale per row
            else:
                total += n_elems * 1 + 4  # tensor-wise: int8 + one F32 scale scalar
            continue

    return total
