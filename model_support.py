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
SUPPORT_UNKNOWN = "unknown"

SUPPORT_SYMBOL: dict[str, str] = {
    SUPPORT_VERIFIED: "✓",
    SUPPORT_CAUTION: "⚠",
    SUPPORT_UNKNOWN: "?",
}



def support_level(arch_key: str, keys_hiprec_nonempty: bool, format_key: str) -> str:
    """Return SUPPORT_VERIFIED / SUPPORT_CAUTION / SUPPORT_UNKNOWN for one
    (architecture, format) pair.

    The three-state scheme (agreed with the user): VERIFIED means actually
    converted+loaded+rendered correctly in ComfyUI; CAUTION means either
    "technically supported but not render-tested for this architecture" or
    "known to have a correctness/quality issue"; UNKNOWN means this
    combination has never even been attempted. CAUTION deliberately folds
    together "unverified" and "known-bad" into one symbol rather than adding
    a fourth state — see docs/superpowers/plans/2026-08-10-model-support-tab.md
    for why, and the per-branch comments below for which case applies where.

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
    - FP8 / FP8_MIXED: CAUTION everywhere, same as NVFP4. FP8 now defaults to
      full_precision_matrix_mult=true, which by code reading of ComfyUI's
      comfy/utils.py and comfy/ops.py should make it skip the risky
      quantized-compute branch entirely — but that's a code-reading
      conclusion, not an actual convert+load+render confirmation with this
      tool's own output on any architecture, so it doesn't meet the VERIFIED
      bar. Plain FP8 additionally carries the same keys_hiprec on-disk
      precision-loss risk plain INT8 does on sensitive architectures
      (format_recommendation() in safetensors_quant.py warns identically).
    - INT8 / INT8_MIXED: depends on keys_hiprec. If the architecture has no
      keys_hiprec at all, plain INT8 and INT8_MIXED produce byte-identical
      output (nothing to protect either way) — both VERIFIED. If it does:
      INT8_MIXED is VERIFIED only for lumina2 (the one architecture actually
      render-tested end-to-end); every other sensitive architecture's
      INT8_MIXED is CAUTION (protection list cross-referenced against
      community tools, never confirmed with this tool's own output). Plain
      INT8 on a sensitive architecture is CAUTION everywhere, including
      lumina2 — that's the one *confirmed-bad* case (docs/issues_analysis.md
      #15's plain-INT8 screenshot), still using the same CAUTION symbol as
      the merely-unverified cases per the agreed three-state scheme.
    - NVFP4 / NVFP4_MIXED: always CAUTION, on every architecture, not just
      lumina2. The mechanism is the same class of dynamic-activation-
      quantization risk as pre-fix FP8 (docs/issues_analysis.md #15), and
      unlike FP8 there is no verified full_precision_matrix_mult-equivalent
      safe mode for NVFP4 yet — direct negative evidence exists only for
      lumina2 (marcorez8's tiered quality ratings), but the structural risk
      generalizes, so CAUTION (not UNKNOWN) is the honest default everywhere
      until that's specifically investigated (tracked as a future plan, see
      docs/issues_analysis.md #16).
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
        return SUPPORT_CAUTION
    if format_key in ("NVFP4", "NVFP4_MIXED"):
        return SUPPORT_CAUTION
    return SUPPORT_UNKNOWN


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
