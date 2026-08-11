"""Per-architecture quantization-format support matrix — the data model
behind the GUI's "Model Support" tab (gui.py) and the dynamic ⚠ annotation
of the GGUF/safetensors format dropdowns.

Two things live here, both editorial judgment calls documented inline
rather than hidden in a spreadsheet: which public model names map to each
internal architecture key (MODEL_DISPLAY_NAMES), and a tri-state confidence
level for each (architecture, format) combination (support_level()).
"""

from __future__ import annotations

from safetensors_quant import _RENDER_VERIFIED_ARCHES

# Public display name per models.architectures.*.arch key. Format:
# "<public name(s)> (<arch key>)" -- the arch key always appears verbatim in
# parentheses so the table stays traceable to models/architectures.py without
# a separate lookup, matching the style the user requested ("Z-Image Turbo
# (Lumina)"). Where one architecture covers multiple public model releases
# with identical quantization support (e.g. every FLUX.1/FLUX.2 variant),
# the name says "Family" rather than listing every release. These are this
# project's own editorial choices, not values ComfyUI or any upstream
# project defines -- review and correct freely.
MODEL_DISPLAY_NAMES: dict[str, str] = {
    "qwen_image": "Qwen-Image Family (qwen_image)",
    "flux": "Flux Family (flux)",
    "sd3": "Stable Diffusion 3 / 3.5 (sd3)",
    "aura": "AuraFlow (aura)",
    "hidream": "HiDream-I1 (hidream)",
    "cosmos": "Cosmos-Predict2 (cosmos)",
    "ltxv": "LTX-Video (ltxv)",
    "hyvid": "HunyuanVideo (hyvid)",
    "wan": "Wan Family (wan)",
    "sdxl": "Stable Diffusion XL (sdxl)",
    "sd1": "Stable Diffusion 1.5 (sd1)",
    "lumina2": "Z-Image / Lumina-Image 2.0 Family (lumina2)",
}

# (display label, format key) — GGUF first (collapses every K-quant level
# into one column, per the user's own reasoning: if base F16 GGUF conversion
# works for an architecture, K-quants apply uniformly on top of it via the
# same generic llama-quantize step), then every GUI-selectable safetensors
# format in SAFETENSORS_DTYPE_CHOICES order, then NVFP4 (implemented,
# per-tool-capability, but never offered in the GUI dropdown).
TABLE_FORMATS: list[tuple[str, str]] = [
    ("GGUF", "GGUF"),
    ("F16", "F16"),
    ("F16 mixed", "F16_MIXED"),
    ("FP8", "FP8"),
    ("FP8 mixed", "FP8_MIXED"),
    ("INT8", "INT8"),
    ("INT8 mixed", "INT8_MIXED"),
    ("NVFP4", "NVFP4"),
    ("NVFP4 mixed", "NVFP4_MIXED"),
]

SUPPORT_VERIFIED = "verified"
SUPPORT_CAUTION = "caution"
SUPPORT_BAD = "bad"
SUPPORT_UNKNOWN = "unknown"

SUPPORT_SYMBOL: dict[str, str] = {
    SUPPORT_VERIFIED: "✓",
    SUPPORT_CAUTION: "⚠",
    SUPPORT_BAD: "✗",
    SUPPORT_UNKNOWN: "?",
}

# (arch_key, format_key) pairs with DIRECT negative render-test evidence —
# actually converted+loaded+rendered and visibly wrong (wrong pose/identity,
# black bars, full-image noise), not just "never tried" or "risky by
# analogy". Kept separate from CAUTION (see support_level()'s docstring) so
# the table distinguishes "confirmed broken" from "merely unverified".
_RENDER_CONFIRMED_BAD: set[tuple[str, str]] = {
    # Plain INT8 on lumina2's keys_hiprec-sensitive tensors: the corruption
    # that motivated adding INT8_MIXED's protection list in the first place
    # (docs/issues_analysis.md #15).
    ("lumina2", "INT8"),
    # NVFP4_MIXED on lumina2: full-image noise, tested end-to-end against
    # both a fine-tune and the official zImageBase checkpoint
    # (docs/issues_analysis.md #15). Plain NVFP4 shares the same root cause
    # (ComfyUI dynamically quantizes activations for every NVFP4 layer
    # regardless of which weights keys_hiprec protects, per #15's
    # Investigation) so it's marked bad too, though only the _MIXED variant
    # was directly rendered.
    ("lumina2", "NVFP4"),
    ("lumina2", "NVFP4_MIXED"),
}


def support_level(arch_key: str, keys_hiprec_nonempty: bool, format_key: str) -> str:
    """Return SUPPORT_VERIFIED / SUPPORT_CAUTION / SUPPORT_BAD / SUPPORT_UNKNOWN
    for one (architecture, format) pair.

    The four-state scheme: VERIFIED means actually converted+loaded+rendered
    correctly in ComfyUI; CAUTION means "technically supported but not
    render-tested for this architecture" (no evidence either way); BAD means
    actually render-tested and confirmed wrong (_RENDER_CONFIRMED_BAD);
    UNKNOWN means this combination has never even been attempted. An earlier
    version of this scheme folded CAUTION and BAD into one symbol — split
    apart because "never tried" and "tried and broke" need different user
    reactions (the former is a request for a test report, the latter a
    warning not to use it).

    - GGUF: always VERIFIED. Every models.architectures.arch_list entry is an
      architecture this tool's GGUF pipeline explicitly detects and handles
      (including automated 5D-tensor/pad-token fixes); K-quants are a
      generic post-processing step (llama-quantize) applied uniformly on top
      of a working F16 GGUF conversion, not something that varies per
      architecture the way keys_hiprec-driven precision choices do.
    - F16 / F16_MIXED: always VERIFIED. This is a precision cast, not
      compressed-representation quantization with a runtime scale lookup —
      it never touches ComfyUI's quantized-compute code path
      (MixedPrecisionOps) at all, so it carries none of the
      architecture-dependent risk INT8/FP8/NVFP4 do.
    - FP8 / FP8_MIXED: CAUTION everywhere (never BAD). FP8 now defaults to
      full_precision_matrix_mult=true, which by code reading of ComfyUI's
      comfy/utils.py and comfy/ops.py should make it skip the risky
      quantized-compute branch entirely — but that's a code-reading
      conclusion, not an actual convert+load+render confirmation with this
      tool's own output on any architecture, so it doesn't meet the VERIFIED
      bar. The pre-fix corruption docs/issues_analysis.md #15 directly
      observed on FP8_MIXED/lumina2 predates this flag and isn't proof
      today's default still fails, so it's unverified (CAUTION), not
      confirmed-bad. Plain FP8 additionally carries the same keys_hiprec
      on-disk precision-loss risk plain INT8 does on sensitive architectures
      (format_recommendation() in safetensors_quant.py warns identically).
    - INT8 / INT8_MIXED: depends on keys_hiprec. If the architecture has no
      keys_hiprec at all, plain INT8 and INT8_MIXED produce byte-identical
      output (nothing to protect either way) — both VERIFIED. If it does:
      INT8_MIXED is VERIFIED only for lumina2 (the one architecture actually
      render-tested end-to-end); every other sensitive architecture's
      INT8_MIXED is CAUTION (protection list cross-referenced against
      community tools, never confirmed with this tool's own output). Plain
      INT8 on a sensitive architecture is BAD for lumina2 specifically — the
      one *confirmed-bad* case (docs/issues_analysis.md #15's plain-INT8
      corruption) — and CAUTION (unverified, not confirmed) for every other
      sensitive architecture, since the same class of risk hasn't actually
      been rendered there.
    - NVFP4 / NVFP4_MIXED: BAD for lumina2 (direct render evidence, #15);
      CAUTION everywhere else. The mechanism is the same class of dynamic-
      activation-quantization risk as pre-fix FP8, and unlike FP8 there is
      no verified full_precision_matrix_mult-equivalent safe mode for NVFP4
      yet, so the structural risk generalizes to CAUTION (not UNKNOWN) on
      every other architecture until that's specifically investigated
      (tracked as a future plan, see docs/issues_analysis.md #16).
    """
    if format_key == "GGUF":
        return SUPPORT_VERIFIED
    if format_key in ("F16", "F16_MIXED"):
        return SUPPORT_VERIFIED
    if format_key in ("FP8", "FP8_MIXED"):
        return SUPPORT_CAUTION
    if format_key in ("INT8", "INT8_MIXED"):
        if not keys_hiprec_nonempty:
            return SUPPORT_VERIFIED
        if arch_key in _RENDER_VERIFIED_ARCHES and format_key == "INT8_MIXED":
            return SUPPORT_VERIFIED
        if (arch_key, format_key) in _RENDER_CONFIRMED_BAD:
            return SUPPORT_BAD
        return SUPPORT_CAUTION
    if format_key in ("NVFP4", "NVFP4_MIXED"):
        if (arch_key, format_key) in _RENDER_CONFIRMED_BAD:
            return SUPPORT_BAD
        return SUPPORT_CAUTION
    return SUPPORT_UNKNOWN


# Text encoders (text_encoder_convert.py) have no per-architecture risk model
# the way diffusion models do: they aren't in models.architectures.arch_list,
# and ComfyUI's text-encoder loaders build models from fixed config presets
# rather than inferring hyperparameters from checkpoint tensor shapes, so
# there's no keys_hiprec-style architecture-dependent caution to compute
# (docs/architecture.md's Text-Encoder Conversion Pipeline section). GGUF
# output goes through llama.cpp's own upstream convert_hf_to_gguf.py rather
# than this project's own writer; FP8/FP8_MIXED/NVFP4/NVFP4_MIXED reuse the
# same safetensors_quant*.py backends the diffusion-model table covers. What
# stays true either way: none of these have a documented ComfyUI load+render
# confirmation by this project (the same bar TABLE_FORMATS holds FP8 to
# above), so every format is CAUTION until one is actually verified that way.
# GGUF collapses every direct outtype (F32/F16/BF16/Q8_0) and K-quant
# (Q6_K..Q2_K) into one column, mirroring TABLE_FORMATS' own GGUF column.
TEXT_ENCODER_TABLE_FORMATS: list[tuple[str, str]] = [
    ("GGUF", "GGUF"),
    ("FP8", "FP8"),
    ("FP8 mixed", "FP8_MIXED"),
    ("NVFP4", "NVFP4"),
    ("NVFP4 mixed", "NVFP4_MIXED"),
]

TEXT_ENCODER_SUPPORT: dict[str, str] = {
    format_key: SUPPORT_CAUTION for _, format_key in TEXT_ENCODER_TABLE_FORMATS
}


def build_support_table() -> list[dict]:
    """Return one row per models.architectures.arch_list entry: display
    name plus a support_level() result for every TABLE_FORMATS column."""
    from models.architectures import arch_list

    rows = []
    for cls in arch_list:
        instance = cls()
        sensitive = bool(instance.keys_hiprec)
        # .get() with a fallback, not a bare index: this runs eagerly at
        # gui.build_app() time, so a newly added arch_list entry without a
        # matching MODEL_DISPLAY_NAMES entry must not crash the whole GUI at
        # startup -- just show the raw arch key until someone adds the name.
        row = {
            "arch": instance.arch,
            "display_name": MODEL_DISPLAY_NAMES.get(instance.arch, instance.arch),
        }
        for _, format_key in TABLE_FORMATS:
            row[format_key] = support_level(instance.arch, sensitive, format_key)
        rows.append(row)
    return rows
