"""Per-architecture quantization-format support matrix — the data model
behind the GUI's "Model Support" tab (gui.py) and the dynamic ⚠ annotation
of the GGUF/safetensors format dropdowns.

Two things live here, both editorial judgment calls documented inline
rather than hidden in a spreadsheet: which public model names map to each
internal architecture key (MODEL_DISPLAY_NAMES), and a four-state confidence
level for each (architecture, format) combination (support_level()).
"""

from __future__ import annotations

from safetensors_quant import _RENDER_VERIFIED_MIXED, _SIZE_RATIOS

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
    "ernie_image": "ERNIE-Image (ernie_image)",
    "krea2": "Krea 2 (krea2)",
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

def _format_header_label(format_key: str) -> str:
    if format_key == "GGUF":
        return "GGUF¹"
    short = (
        f"{format_key[:-len('_MIXED')].lower()} mix"
        if format_key.endswith("_MIXED")
        else format_key.lower()
    )
    if format_key in _SIZE_RATIOS:
        return f"{short} ~{round((1 - _SIZE_RATIOS[format_key]) * 100)}%"
    return short


# (display label, format key) — GGUF first (collapses every K-quant level
# into one column, per the user's own reasoning: if base F16 GGUF conversion
# works for an architecture, K-quants apply uniformly on top of it via the
# same generic llama-quantize step), then every GUI-selectable safetensors
# format in SAFETENSORS_DTYPE_CHOICES order, then NVFP4 (implemented,
# per-tool-capability, but never offered in the GUI dropdown).
TABLE_FORMATS: list[tuple[str, str]] = [
    (_format_header_label("GGUF"), "GGUF"),
    (_format_header_label("F16"), "F16"),
    (_format_header_label("F16_MIXED"), "F16_MIXED"),
    (_format_header_label("FP8"), "FP8"),
    (_format_header_label("FP8_MIXED"), "FP8_MIXED"),
    (_format_header_label("INT8"), "INT8"),
    (_format_header_label("INT8_MIXED"), "INT8_MIXED"),
    (_format_header_label("NVFP4"), "NVFP4"),
    (_format_header_label("NVFP4_MIXED"), "NVFP4_MIXED"),
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

# (arch_key, format_key) pairs where BAD means "cannot be built at all" --
# an upstream/tooling gap, not a render-tested-and-wrong result. The table's
# ✗ symbol alone doesn't distinguish this from "builds fine but the picture
# is wrong" (e.g. qwen_image's plain NVFP4, lumina2's plain FP8/INT8) --
# support_reason()/text_encoder_support_reason() surface this text as a
# tooltip so the difference is visible without reading source comments.
_STRUCTURALLY_IMPOSSIBLE: dict[tuple[str, str], str] = {
    ("ernie_image", "GGUF"): (
        "No GGUF build exists with this project's tooling -- city96/ComfyUI-GGUF's "
        "lcpp.patch has no ernie_image llm_arch entry. Safetensors only."
    ),
    ("krea2", "GGUF"): (
        "Krea 2 GGUF requires the Krea-capable molbal/ComfyUI-GGUF loader "
        "according to the published model instructions. The source snapshot "
        "is secured at 5a0a3ffa0e3eae5c6af8b0b981b660a24d5fbc04, but our "
        "native city96 conversion path cannot export this architecture. The "
        "Workbench's pinned molbal adapter exports an explicit non-K allowlist, "
        "and Q4_0 has a two-prompt molbal render test with visible drift versus one INT8 ConvRot checkpoint. Loader requirement is "
        "tested for the recorded city96/molbal revisions; other scopes and coexistence with "
        "the city96 workflow remain pending. Existing safetensors conversion "
        "is a separate path. Source: docs/krea2-backend-preparation.md and "
        "https://huggingface.co/molbal/krea2-gguf."
    ),
}

_TE_STRUCTURALLY_IMPOSSIBLE: dict[tuple[str, str], str] = {
    ("clip-l", "GGUF"): (
        "No GGUF build exists or can exist with this project's tooling -- "
        "llama.cpp's convert_hf_to_gguf.py has no CLIPModel converter, and "
        "ComfyUI-GGUF's CLIPLoaderGGUF has no CLIP-L decode path either. "
        "Upstream gap, not a failed/wrong render."
    ),
    ("clip-bigg", "GGUF"): (
        "No GGUF build exists or can exist with this project's tooling -- "
        "llama.cpp's convert_hf_to_gguf.py has no CLIPModel converter, and "
        "ComfyUI-GGUF's CLIPLoaderGGUF has no OpenCLIP-bigG decode path "
        "either. Upstream gap, not a failed/wrong render."
    ),
    ("qwen2.5-vl-7b", "GGUF"): (
        "The GGUF file builds and loads, but is missing its entire vision "
        "tower (llama.cpp's plain converter only exports the language-model "
        "half) -- unusable for any image-conditioning workflow, which is "
        "this family's only real use case. Upstream tooling gap, not a "
        "failed/wrong render. docs/issues_analysis.md #22."
    ),
    ("mistral-small-3.2-24b", "GGUF"): (
        "Cannot be built for the FLUX.2 dev packaging of this encoder "
        "(mistral_3_small_flux2_*.safetensors, 2026-08-23): it's a "
        "bespoke-trimmed variant with only 30 transformer layers and no "
        "final norm/lm_head tensor -- NOT the vendored 40-layer "
        "Mistral-Small-3.2-24B-Instruct-2506 config this family's GGUF path "
        "assumes. detect_text_encoder_family() correctly returns None (layer "
        "count mismatch, not a bug), so no base_repo_id auto-detects; forcing "
        "the vendored config would apply wrong hyperparameters. No known "
        "correct config exists yet -- even upstream (city96/ComfyUI-GGUF#367) "
        "text-encoder GGUF support for FLUX.2 is an open, unresolved feature "
        "request. Unrelated to the actual (untrimmed) Mistral-Small-3.2-24B "
        "checkpoint ERNIE-Image uses, which this family also covers and "
        "isn't known to have this problem -- just not yet tested either."
    ),
}


def support_reason(arch_key: str, format_key: str) -> str | None:
    """Tooltip text for a diffusion-model support-table cell, or None if the
    cell needs no extra explanation beyond its symbol."""
    return (_STRUCTURALLY_IMPOSSIBLE.get((arch_key, format_key)) or
            _SCOPED_RENDER_NOTES.get((arch_key, format_key)))


def text_encoder_support_reason(family: str, format_key: str) -> str | None:
    """Tooltip text for a text-encoder support-table cell, or None if the
    cell needs no extra explanation beyond its symbol."""
    return (_TE_STRUCTURALLY_IMPOSSIBLE.get((family, format_key)) or
            _TE_DRIFT_NOTES.get((family, format_key)))

# (arch_key, format_key) pairs with DIRECT negative render-test evidence —
# actually converted+loaded+rendered and visibly wrong (wrong pose/identity,
# black bars, full-image noise), not just "never tried" or "risky by
# analogy". Kept separate from CAUTION (see support_level()'s docstring) so
# the table distinguishes "confirmed broken" from "merely unverified".
_RENDER_CONFIRMED_BAD: set[tuple[str, str]] = {
    # Safetensors-only new DiTs: no matching llm_arch in this project's
    # targeted City96 lcpp.patch path (Krea 2 only exists in an unofficial PR).
    ("ernie_image", "GGUF"),
    ("krea2", "GGUF"),
    # Plain INT8 on lumina2's keys_hiprec-sensitive tensors: the corruption
    # that motivated adding INT8_MIXED's protection list in the first place
    # (docs/issues_analysis.md #15).
    ("lumina2", "INT8"),
    # Plain FP8 on lumina2: same-seed/prompt comparison (user-provided,
    # 2026-08-11) showed the standing figure's outfit and the entire room
    # background/set dressing changed between the unquantized and plain-FP8
    # renders, with only the two figures' poses staying recognizable --
    # the same "wrong identity/composition, pose skeleton survives" pattern
    # #15 documented for pre-fix FP8_MIXED/plain INT8. Confirms what
    # format_recommendation() already predicted by analogy from INT8's
    # confirmed corruption (full_precision_matrix_mult only fixes the
    # runtime compute path, not on-disk keys_hiprec precision loss) --
    # now with direct evidence instead of just the shared mechanism.
    ("lumina2", "FP8"),
    # NVFP4/NVFP4_MIXED: briefly moved to _RENDER_TESTED_DRIFT (CAUTION) on
    # 2026-08-13 after a full_precision_nvfp4 re-test showed no more garbage/
    # noise, 2 of 3 prompts matching baseline cleanly. Moved BACK here the same
    # day on review: when it does fail (the 3rd prompt), it fails exactly the
    # same way plain FP8/INT8 above do -- full composition/pose/outfit swap,
    # not a minor detail -- and both of those were judged BAD off a *single*
    # failing test each, with no "but 2 other tests were clean" exception
    # granted. Holding NVFP4 to a looser bar than FP8/INT8 for showing the
    # identical failure mode, just because it also had passing samples,
    # wasn't a consistent standard -- a format that produces a full wrong
    # image on some real prompts is "confirmed wrong", not "tolerable
    # deviation", regardless of how often it happens to succeed elsewhere.
    # See _RENDER_TESTED_DRIFT's comment below for the full render-test
    # writeup this classification is based on.
    ("lumina2", "NVFP4"),
    ("lumina2", "NVFP4_MIXED"),
    # qwen_image plain NVFP4 (Qwen-Image-Edit-2511, 2026-08-20 batch,
    # reviewed 2026-08-23): render-tested via the fixed-seed edit workflow
    # (BF16 text encoder held constant) -- severe full-image mosaic/pixel-
    # noise corruption, not just detail loss or a composition swap like other
    # architectures' plain-NVFP4 drift; the whole frame is visibly wrong.
    # NVFP4_MIXED (same batch) is clean -- see safetensors_quant.py's
    # _RENDER_VERIFIED_MIXED -- so keys_hiprec protection is doing its job
    # and this is the expected plain-format tradeoff turning out unusually
    # severe for this architecture, not a NVFP4_MIXED-side bug.
    ("qwen_image", "NVFP4"),
}

# (arch_key, format_key) pairs that HAVE been convert+load+render-tested but
# showed some visible deviation from the uncompressed baseline that isn't
# severe enough to call BAD (no wrong identity, no broken/garbage output) --
# distinct from "never tested" (2026-08-13: previously both states shared
# CAUTION, which made the symbol mean two different things a user needs to
# react to differently — "please go test this" vs. "tested, use with
# awareness of a known quality tradeoff"). flux's NVFP4/NVFP4_MIXED were the
# original motivating case (2026-08-12, visible composition drift in 2 of 3
# seeds) but got root-caused and fixed the next day (missing
# keys_shape_critical entries + full_precision_nvfp4 never being set) -- a
# re-test came back clean, so they moved to safetensors_quant.
# _RENDER_VERIFIED_MIXED instead.
#
# lumina2's NVFP4/NVFP4_MIXED were previously _RENDER_CONFIRMED_BAD ("full-
# image noise") from a pre-fix conversion (docs/issues_analysis.md #15).
# 2026-08-13: re-converted with the same full_precision_nvfp4 fix that cleared
# flux, then render-tested with 3 prompts x 2 seeds via qwen3-4b. Unlike the
# old evidence, the fixed conversion produces coherent, well-formed images for
# every seed -- no garbage/noise. 2 of 3 prompts matched the BF16 baseline
# closely (fine-detail-only variance). The third (the longest, most
# detail-dense prompt) showed the entire composition, pose, and outfit change
# between baseline and both NVFP4 variants, and even varied noticeably
# between the two images of the same NVFP4 batch -- more batch-internal
# variance than baseline showed for the same prompt. Audited
# ModelLumina2.keys_shape_critical against ComfyUI's model_detection.py
# Lumina2/Z-Image branch (G:\ComfyUI-Easy-Install\ComfyUI, comfy_kitchen
# 0.2.30) and found no missing raw-shape read comparable to flux's txt_in/
# vector_in gap -- cap_embedder.1.weight, x_pad_token, and cap_pad_token
# already cover everything read there. So this isn't the same fixable
# architecture-detection bug; it reads as genuine content-dependent
# quantization sensitivity, and it briefly moved to _RENDER_TESTED_DRIFT
# (CAUTION) on the strength of that improvement -- reverted back to
# _RENDER_CONFIRMED_BAD the same day, see that set's comment above for why
# (the one failing prompt is a full wrong-image swap, judged the same way
# FP8/INT8's single failing tests were, not a "2 of 3 were fine" exception).
#
# aura's plain NVFP4 (aura_flow_0.3, 2026-08-18, fixed seed via aura_t5):
# after both fixes (ModelAura.keys_hiprec + is_hiprec_st's F16 dtype-gate,
# see safetensors_quant.py's _RENDER_VERIFIED_MIXED docstring), NVFP4_MIXED
# matched the F16 baseline exactly on the close-up vampire-portrait prompt
# that originally showed drift. Plain NVFP4 on that same prompt still showed
# visible facial-detail loss (missing lace-veil texture, softer skin, no
# freckles) -- keys_hiprec only ever protects *_MIXED, so this is the
# expected plain-format tradeoff, not a bug. No wrong identity, no crash --
# CAUTION, not BAD. (3 other, less face-detail-dependent motifs had plain
# NVFP4 matching cleanly, so this is prompt/content-dependent severity, not
# a guaranteed failure.)
_RENDER_TESTED_DRIFT: set[tuple[str, str]] = {
    ("aura", "NVFP4"),
    # 2026-10-05 same-source cat/dog comparisons supersede older clean samples
    # for these pairs. Usable images with visible detail drift are CAUTION,
    # not a universal failure claim. See docs/quantization-size-audit.md.
    ("qwen_image", "NVFP4_MIXED"),
    ("qwen_image", "INT8_MIXED"),
}

_SCOPED_RENDER_NOTES: dict[tuple[str, str], str] = {
    ('qwen_image', 'GGUF'): (
        'Qwen-Image-Edit-2511 Q4_K_M: converted and rendered successfully on '
        '2026-10-05 with Easy-Install llama-quantize build 3962 (c8c07d658), '
        'ComfyUI 0.38.0 and RTX 5080. The empty __index_timestep_zero__ marker '
        'was preserved via the companion-file repair path. One cat/hat edit, '
        'no LoRA and no same-seed source comparison: quality parity and other '
        'Qwen variants/precisions are not established. The older lcpp.patch '
        'limitation is specific to that binary, not a universal GGUF ban. '
        'Source: docs/conversion-safety-validation.md, Qwen GGUF render smoke test.'),
    ('qwen_image', 'NVFP4_MIXED'): (
        'Qwen-Image-Edit-2511, qwen_edit_2511 profile: cat/hat and dog/scarf edits '
        'remained usable; eyes, whiskers and scarf details visibly differed from '
        'the original FP8 renders. Same seed, text encoder and VAE; no LoRA. '
        'Two-reference smoke test, not evidence for every Qwen model or prompt. '
        'Source: docs/quantization-size-audit.md, 2026-10-05 runtime validation.'),
    ('qwen_image', 'INT8_MIXED'): (
        'Qwen-Image-Edit-2511, qwen_edit_2511 profile, INT8/ConvRot mixed: usable '
        'cat/hat and dog/scarf edits with small visible detail deviations from '
        'the original FP8 renders. Same seed, text encoder and VAE; no LoRA. '
        'Limited two-reference smoke test. '
        'Source: docs/quantization-size-audit.md, 2026-10-05 runtime validation.'),
    ('lumina2', 'NVFP4_MIXED'): (
        'Profile-dependent evidence: Z-Image Turbo with the z_image_turbo '
        'profile produced usable cat/hat and dog/scarf images, with hat, facial '
        'detail and slight framing differences versus BF16 (2026-10-05, two '
        'prompts). This does not clear the older default-policy family tests '
        'that changed composition, pose and outfit substantially; the family '
        'classification remains bad. Source: docs/quantization-size-audit.md '
        'and the earlier render notes in model_support.py.'),
    ('aura', 'NVFP4'): (
        'AuraFlow 0.3: a close-up portrait lost lace-veil texture/freckles and '
        'showed softer skin; identity remained usable. Other tested motifs '
        'matched more closely. Mixed NVFP4 did not show this drift in that '
        'comparison. Source: model_support.py, 2026-08-18 render notes.'),
}

_UNTESTED_SAFETENSORS_ARCHES: frozenset[str] = frozenset({"ernie_image", "krea2"})


def support_level(arch_key: str, keys_hiprec_nonempty: bool, format_key: str) -> str:
    """Return SUPPORT_VERIFIED / SUPPORT_CAUTION / SUPPORT_BAD / SUPPORT_UNKNOWN
    for one (architecture, format) pair.

    The four-state scheme: VERIFIED means actually converted+loaded+rendered
    correctly in ComfyUI, no visible deviation from the uncompressed
    baseline; CAUTION means convert+load+render-TESTED and showing some
    visible-but-tolerable deviation (_RENDER_TESTED_DRIFT) — not "we didn't
    check", but "we checked, and compression changed the output somewhat";
    BAD means actually render-tested and confirmed wrong — broken output or
    lost identity/composition (_RENDER_CONFIRMED_BAD); UNKNOWN means this
    combination has never actually been rendered, no evidence either way.
    Before 2026-08-13, CAUTION covered both "untested" and "tested with
    drift" — split apart because those need different user reactions (the
    former is a request for a test report, the latter a request to judge
    whether the tradeoff is acceptable for a given use case), and "never
    tried" vs. "tried and broke" already got the same split treatment
    earlier for the same reason.

    - GGUF: always VERIFIED, EXCEPT an architecture explicitly listed in
      _RENDER_CONFIRMED_BAD for "GGUF" because no build is possible with
      this project's targeted City96 lcpp.patch path (`ernie_image`, `krea2`).
      Qwen-Image-Edit-2511 Q4_K_M has a successful 2026-10-05 conversion/render
      using the Easy-Install binary; its reason/scope records that one smoke
      test without claiming source fidelity or every GGUF precision.
      This exception was added 2026-08-23 after the blanket-VERIFIED
      reasoning below silently overrode an already-added
      ("qwen_image", "GGUF") _RENDER_CONFIRMED_BAD entry -- the format_key
      == "GGUF" branch used to return unconditionally before ever
      consulting that set. For every other architecture, the reasoning
      below still applies: every models.architectures.arch_list entry is an
      architecture this tool's GGUF pipeline explicitly detects and handles
      (including automated 5D-tensor/pad-token fixes); K-quants are a
      generic post-processing step (llama-quantize) applied uniformly on top
      of a working F16 GGUF conversion, not something that varies per
      architecture the way keys_hiprec-driven precision choices do. This is
      a design-reasoning conclusion, not a per-architecture render-test
      claim -- note llama-quantize's K-quant path has no keys_hiprec
      equivalent at all (quantize.py's own F32-forcing only covers
      1D/small/BF16-or-F32-source tensors uniformly, nothing
      architecture-aware), so an architecture-dependent K-quant defect isn't
      ruled out by design the way it is for the safetensors path. 2026-08-13:
      got actual render evidence for lumina2/Z-Image specifically -- Q6_K
      (5 seeds across 2 prompts via a Qwen3-4B-family text encoder) matched
      the unquantized baseline's composition/identity/pose exactly, batch-
      index-for-batch-index (a "3 women vs. 2 women" difference between the
      two prompt-C batch images looked like a defect at first, until the
      *unquantized baseline itself* showed the same split -- prompt-adherence
      seed variance, not a quantization artifact). No code change from this
      (GGUF was already unconditionally VERIFIED), just an evidence upgrade
      from "assumed safe by design" to "render-confirmed" for this
      architecture. Same day, `flux` got the same treatment (a matching
      render-test folder, qwen3-8b vanilla text encoder):
      Q6_K across 3 prompts x 2 seeds matched the unquantized baseline with
      no visible deviation at all (closer agreement than lumina2's, no
      seed-variance false alarm this time). IMPORTANT: every render-test
      backing the `flux` arch_key (this GGUF note and every safetensors
      entry in safetensors_quant.py's _RENDER_VERIFIED_MIXED) used a
      FLUX.2 Klein 9B checkpoint (qwen3-8b text encoder), not Flux.1/
      Flux.1 Kontext (T5-XXL text encoder) -- found 2026-08-18 while
      auditing text-encoder test coverage. `ModelFlux` detects both by
      shared tensor naming, so the VERIFIED status applies to the
      `arch_key` as coded, but neither Flux.1 itself nor its T5-XXL text
      encoder have ever actually been render-tested against this tool.
      Treat Flux.1-specific claims as UNKNOWN pending real evidence, not as
      covered by the FLUX.2 results above.
      2026-08-24: got a second, independent GGUF data point -- this one
      actually from FLUX.2 **dev** itself (32B, mistral-small-3.2-24b text
      encoder, not the Klein 9B checkpoint above), Q4_K_M quantized from a
      dequantized fp8_mixed source and render-tested with the tool's own
      rebuilt nvfp4_mixed text encoder. Composition, identity, wound
      pattern, armor, and medallion all matched the equivalent fp8_mixed-
      diffusion render exactly; only a slight (expected) softening in
      fine skin/beard detail from the more aggressive 4-bit quantization,
      no corruption. Closes the one remaining untested cell in this
      session's FLUX.2 dev format sweep (safetensors formats + this GGUF
      quant were all render-tested that day; plain diffusion NVFP4/
      NVFP4_MIXED still only carry single-seed evidence, see the
      safetensors_quant.py entries below).
      2026-08-20 caveat: the "design reasoning" above implicitly assumed a
      plain (never-quantized) source checkpoint. `convert.py`'s GGUF path
      had no dequantization pre-pass for an already-quantized source until
      that day — crashed `llama-quantize` (and, unfixed, would have written
      silently-wrong unscaled weights) converting a community `wan` fp8_mixed
      checkpoint. Fixed (docs/issues_analysis.md #18, shared with
      `convert_safetensors.py`'s pre-existing `_scan_quantized_layers()`).
      The blanket VERIFIED-by-design conclusion above still holds for a
      plain-precision source; a pre-quantized source checkpoint is no longer
      an exception now that the fix is in, but was one before this date.
      Same day, `wan` Q4_K_M got actual render evidence backing this design
      reasoning too: one I2V render with both HIGH- and LOW-noise
      `UnetLoaderGGUF` loaded (`wan22_i2v_14b_{HIGH,LOW}-Q4_K_M.gguf`)
      matched the safetensors-format renders' composition/pose exactly.
    - F16 / F16_MIXED: always VERIFIED for established architectures. This
      is a precision cast, not compressed-representation quantization with a
      runtime scale lookup, so it normally carries none of the
      architecture-dependent risk INT8/FP8/NVFP4 do. New scaffolding-only
      architectures (`ernie_image`, `krea2`) stay UNKNOWN until any real
      safetensors render evidence exists. Same 2026-08-13
      render-test as GGUF's note above also covered lumina2's GGUF F16
      (direct, unquantized) and safetensors F16_MIXED -- both matched the
      baseline cleanly, backing this design-reasoning conclusion with actual
      evidence for this architecture too. `flux`'s matching GGUF F16 and
      safetensors F16_MIXED renders (same test run as its Q6_K note above)
      were equally clean.
    - FP8 / FP8_MIXED: BAD for plain FP8 on lumina2 (direct render evidence —
      same-seed/prompt comparison showed the standing figure's outfit and
      the room background changed between the unquantized and plain-FP8
      renders, only the pose skeleton surviving; see _RENDER_CONFIRMED_BAD).
      FP8_MIXED is VERIFIED for lumina2: a second same-seed/prompt
      comparison (2026-08-11) showed composition/identity/outfit preserved,
      the only deviation being a single secondary prop judged tolerable
      quantization variance, not a correctness failure — the same bar
      INT8_MIXED was held to (_RENDER_VERIFIED_MIXED,
      safetensors_quant.py). Everywhere else, FP8/FP8_MIXED are UNKNOWN
      (never actually rendered on that architecture): full_precision_
      matrix_mult=true should, by code reading of ComfyUI's comfy/utils.py
      and comfy/ops.py, make FP8_MIXED skip the risky quantized-compute
      branch entirely on any architecture, but that's a code-reading
      conclusion, not an actual convert+load+render confirmation.
    - INT8 / INT8_MIXED: depends on keys_hiprec. If the architecture has no
      keys_hiprec at all, plain INT8 and INT8_MIXED produce byte-identical
      output (nothing to protect either way) — both VERIFIED. If it does:
      INT8_MIXED is VERIFIED only for lumina2 and flux (_RENDER_VERIFIED_
      MIXED, both render-tested end-to-end); every other sensitive
      architecture's INT8_MIXED is UNKNOWN (never actually rendered).
      Plain INT8 on a sensitive architecture is BAD for lumina2
      specifically — the one *confirmed-bad* case (docs/issues_analysis.md
      #15's plain-INT8 corruption) — VERIFIED for flux (render-tested
      clean), and UNKNOWN for every other sensitive architecture, since the
      same class of risk hasn't actually been rendered there.
      2026-08-20 caveat: every plain-INT8/INT8_MIXED render behind the
      VERIFIED claims above (`wan`, `hidream`, `aura`, `flux`) used a build
      from before `docs/issues_analysis.md` #19's fix (tensor-wise scale
      written as a `(1,)`-shaped tensor instead of a true 0-dim scalar,
      which crashes ComfyUI's low-VRAM dynamic-requantize path under VRAM
      pressure — invisible to a render test that never triggers that path).
      The VERIFIED classification describes output *correctness* when the
      file loads, which the fix doesn't change; it does not mean every
      already-deployed file is crash-safe under low VRAM until rebuilt.
    - NVFP4 / NVFP4_MIXED: VERIFIED for flux (2026-08-12 render test showed
      composition drift, root-caused 2026-08-13 to two bugs — missing
      keys_shape_critical entries and full_precision_matrix_mult never being
      set for nvfp4 — both fixed, re-tested clean, see safetensors_quant.py's
      _RENDER_VERIFIED_MIXED docstring). BAD for lumina2 (same
      full_precision_nvfp4 fix applied and re-tested 2026-08-13; no longer
      the pre-fix full-image noise, but one of three tested prompts still
      showed a full composition/pose/outfit change from baseline — the same
      "confirmed wrong on at least one real prompt" bar plain FP8/INT8 are
      held to, so this stays BAD rather than CAUTION despite the other two
      prompts rendering cleanly — see _RENDER_CONFIRMED_BAD). UNKNOWN
      everywhere else, since neither the corruption mechanism nor the fix
      has actually been render-confirmed
      there yet (docs/issues_analysis.md #16).
    - sd1 (2026-08-14): FP8/FP8_MIXED/NVFP4/NVFP4_MIXED all VERIFIED —
      DreamShaper 8 (civitai.com/models/4384), render-tested clean first try
      (the sdxl fixes below already covered it: the dim()>=3 Conv2d guard,
      and ModelSD1's attn2.to_k.weight shape_critical entry -- SD1 has no
      label_emb/class-conditioning, so no analog to sdxl's label_emb fix was
      needed). 2 prompts (a photoreal portrait, a fantasy character
      portrait) via CLIP-L, composition/identity/outfit matching the F16
      baseline. INT8/INT8_MIXED already VERIFIED via the `not
      keys_hiprec_nonempty` branch (sd1 has no keys_hiprec, same as sdxl).
    - sdxl (2026-08-14): FP8/FP8_MIXED/NVFP4/NVFP4_MIXED all VERIFIED —
      initially FP8/INT8 produced solid-black renders and plain NVFP4/
      NVFP4_MIXED crashed on load, both root-caused to real bugs (a missing
      >=3D-tensor guard letting Conv2d weights get quantized into a format
      ComfyUI can't load, and a missing keys_shape_critical entry), fixed,
      then re-tested clean across 2 prompts (fantasy render + photorealistic
      portrait) with composition/identity/outfit matching the F16 baseline
      exactly — see safetensors_quant.py's _RENDER_VERIFIED_MIXED docstring.
      INT8/INT8_MIXED are VERIFIED too, but via the `not keys_hiprec_nonempty`
      branch below (sdxl has no keys_hiprec) rather than an explicit
      _RENDER_VERIFIED_MIXED entry — same render evidence backs both paths.
    - sd3 (sd3_medium, 2026-08-15/16): FP8/FP8_MIXED/NVFP4/NVFP4_MIXED all
      VERIFIED. F16/F16_MIXED/FP8/FP8_MIXED/INT8/INT8_MIXED/Q4_K_M GGUF
      render-tested clean via CLIP-G first. Plain NVFP4/NVFP4_MIXED then
      crashed on load ('mat1 and mat2 shapes cannot be multiplied') —
      ModelSD3.keys_shape_critical was missing y_embedder.mlp.0.weight, the
      same adm_in_channels-corruption pattern already fixed for sdxl's
      label_emb.0.0.weight (see safetensors_quant.py's
      _RENDER_VERIFIED_MIXED docstring). Fixed, re-tested clean, no visible
      deviation from the F16 baseline. INT8/INT8_MIXED are VERIFIED via the
      `not keys_hiprec_nonempty` branch (sd3 has no keys_hiprec), same as
      sdxl/sd1.
    - hidream (hidream_i1_dev, 2026-08-17): FP8_MIXED/INT8_MIXED VERIFIED --
      render-tested clean via CLIP-L, matching composition/identity on a
      fixed seed. Plain INT8 produced severely corrupted output (structural
      collapse, not just quality drift) and plain NVFP4 crashed on load
      (`size mismatch ... [4, 1280] ... [4, 2560]`) -- both traced to
      `ff_i.gate.weight` (the MoE router) being quantized in plain mode
      when it never goes through ComfyUI's quantized-loading path at all
      (raw state_dict assign, same bug class as the FP8 branch's CLIP
      `position_embedding` example in `safetensors_quant.py`). Fixed by
      adding it (and `img_emb.emb_pos`, same risk, unconfirmed) to
      `ModelHiDream.keys_shape_critical`, which forces F16 in every
      quantization branch regardless of plain/mixed. Plain FP8/INT8/NVFP4
      all regenerated after the fix and re-render-tested clean -- VERIFIED
      across the board. GGUF Q4_K_M rendered clean throughout (unaffected --
      llama-quantize's K-quant path doesn't go through this tensor-level
      protection at all, and never needed to: it's a post-process on a
      full-precision F16 GGUF, not a `quantize_tensor_st()` call).
    - aura (aura_flow_0.3, 2026-08-18): F16/F16_MIXED/FP8/FP8_MIXED/
      INT8/INT8_MIXED/NVFP4_MIXED/Q4_K_M GGUF all VERIFIED -- render-tested
      clean via aura_t5 (Pile-T5-XL), fixed seed, matching composition/
      identity across formats. NVFP4/NVFP4_MIXED initially showed a
      reproducible composition/lighting shift vs. the baseline, traced to
      two compounding bugs: (1) `ModelAura` had no `keys_hiprec` at all --
      the AdaLN modulation Linears (modC/modX/modCX/modF) scale the
      *entire* residual stream per block and were quantized in every
      format including NVFP4_MIXED, which should have been immune. Fixed by
      adding `keys_hiprec` (the same tensor class Flux/Lumina2 already
      protect, see those classes' comments in models/architectures.py). (2)
      Even with `keys_hiprec` populated, the drift persisted: `is_hiprec_st()`
      (and its GGUF mirror `_quant_type_for()`) only granted hiprec
      protection to F32/BF16 sources -- aura_flow_0.3 ships F16-native on
      disk (unlike the BF16 checkpoints this mechanism was validated
      against), so `keys_hiprec` was a silent no-op for it regardless of
      content. Fixed by widening both gates (and the GUI size-estimate
      mirror) to also accept float16. Re-converted and re-render-tested
      clean across 4 motifs (fire-queen portrait, mountain golem, haunted-
      house/blood-moon, and the original close-up vampire portrait that
      first showed the drift) -- NVFP4_MIXED now matches the F16 baseline
      exactly on every one, including facial detail on the portrait prompt.
      Plain NVFP4 is CAUTION, not VERIFIED: `keys_hiprec` only ever protects
      *_MIXED (by design, same tradeoff every other architecture has), and
      on the vampire-portrait prompt specifically, plain NVFP4 still showed
      visible (tolerable, no wrong identity) facial-detail loss vs. the
      baseline -- the 3 other, less face-detail-dependent motifs matched
      cleanly, so this reads as prompt-dependent severity of an accepted
      tradeoff, not a bug.
    """
    if format_key == "GGUF":
        # Check remaining tooling gaps first. Qwen's historical issue #21
        # restriction is superseded by the recorded Easy-Install smoke test.
        if (arch_key, format_key) in _RENDER_CONFIRMED_BAD:
            return SUPPORT_BAD
        return SUPPORT_VERIFIED
    if arch_key in _UNTESTED_SAFETENSORS_ARCHES:
        return SUPPORT_UNKNOWN
    # New drift evidence must outrank older clean samples for the same pair.
    # Keep stronger negative evidence; a scoped successful profile is not a
    # blanket override of a family/default-policy failure.
    if (arch_key, format_key) in _RENDER_TESTED_DRIFT and (arch_key, format_key) not in _RENDER_CONFIRMED_BAD:
        return SUPPORT_CAUTION
    if format_key in ("F16", "F16_MIXED"):
        return SUPPORT_VERIFIED
    if format_key in ("FP8", "FP8_MIXED"):
        if (arch_key, format_key) in _RENDER_VERIFIED_MIXED:
            return SUPPORT_VERIFIED
        if (arch_key, format_key) in _RENDER_CONFIRMED_BAD:
            return SUPPORT_BAD
        if (arch_key, format_key) in _RENDER_TESTED_DRIFT:
            return SUPPORT_CAUTION
        return SUPPORT_UNKNOWN
    if format_key in ("INT8", "INT8_MIXED"):
        if not keys_hiprec_nonempty:
            return SUPPORT_VERIFIED
        if (arch_key, format_key) in _RENDER_VERIFIED_MIXED:
            return SUPPORT_VERIFIED
        if (arch_key, format_key) in _RENDER_CONFIRMED_BAD:
            return SUPPORT_BAD
        if (arch_key, format_key) in _RENDER_TESTED_DRIFT:
            return SUPPORT_CAUTION
        return SUPPORT_UNKNOWN
    if format_key in ("NVFP4", "NVFP4_MIXED"):
        if (arch_key, format_key) in _RENDER_VERIFIED_MIXED:
            return SUPPORT_VERIFIED
        if (arch_key, format_key) in _RENDER_CONFIRMED_BAD:
            return SUPPORT_BAD
        if (arch_key, format_key) in _RENDER_TESTED_DRIFT:
            return SUPPORT_CAUTION
        return SUPPORT_UNKNOWN
    return SUPPORT_UNKNOWN


# Text encoders (text_encoder_convert.py), per vendored base-model family --
# mirrors the diffusion-model table's per-architecture design (build_support_table()
# below) rather than the single generic row this table used before
# detect_text_encoder_family() made per-family identification possible.
# GGUF collapses every direct outtype (F32/F16/BF16/Q8_0) and K-quant
# (Q6_K..Q2_K) into one column, mirroring TABLE_FORMATS' own GGUF column --
# 2026-08-12 same-model testing (Qwen3-4B, F16 vs. Q8_0 vs. Q6_K, 4 seeds
# each) found no format-specific defect distinguishing K-quants from direct
# outtypes, so there's no evidence to split this column the way there once
# seemed to be (see below).
TEXT_ENCODER_TABLE_FORMATS: list[tuple[str, str]] = [
    (_format_header_label("GGUF"), "GGUF"),
    # A plain precision cast (real float16, no quantization/compression), not
    # the GGUF-outtype "F16" collapsed into the column above -- corresponds
    # to TEXT_ENCODER_FORMAT_CHOICES' "F16_ST" dropdown key (routes to
    # convert_text_encoder_to_safetensors(), see text_encoder_convert.py).
    # Missing until 2026-08-14: this table previously had no column for it at
    # all, so a family like clip-l/clip-bigg -- where F16 safetensors is the
    # ONLY working format after every quantized format was confirmed BAD --
    # showed a row of nothing but ✗, with no visible indication that
    # anything works. See text_encoder_support_level()'s unconditional
    # SUPPORT_VERIFIED for this column, mirroring TABLE_FORMATS' own F16
    # column for diffusion models.
    (_format_header_label("F16"), "F16"),
    (_format_header_label("FP8"), "FP8"),
    (_format_header_label("FP8_MIXED"), "FP8_MIXED"),
    (_format_header_label("INT8"), "INT8"),
    (_format_header_label("INT8_MIXED"), "INT8_MIXED"),
    (_format_header_label("NVFP4"), "NVFP4"),
    (_format_header_label("NVFP4_MIXED"), "NVFP4_MIXED"),
]

# Display label per vendored family short name (text_encoder_configs/<name>/,
# _FAMILY_SIGNATURES in text_encoder_convert.py). Format matches
# MODEL_DISPLAY_NAMES above: "<public name> (<short name>)".
TEXT_ENCODER_FAMILY_DISPLAY_NAMES: dict[str, str] = {
    "qwen3-4b": "Qwen3 4B — Z-Image / FLUX.2 klein 4B (qwen3-4b)",
    "qwen3-8b": "Qwen3 8B — FLUX.2 klein 9B (qwen3-8b)",
    "qwen2.5-vl-7b": "Qwen2.5-VL 7B — Qwen-Image / Qwen-Image-Edit (qwen2.5-vl-7b)",
    "mistral-small-3.2-24b": "Mistral Small 3.2 24B — FLUX.2 dev, ERNIE-Image (mistral-small-3.2-24b)",
    "ernie-image-pe": "Ministral3 Prompt-Enhancer — ERNIE-Image (ernie-image-pe)",
    "t5-xxl": "T5-XXL — FLUX.1 / FLUX.1 Kontext / HiDream-I1 (t5-xxl)",
    "clip-l": "CLIP-L — SDXL / FLUX.1 (clip-l)",
    "clip-bigg": "OpenCLIP-bigG — SDXL (clip-bigg)",
    "pile-t5xl": "Pile-T5-XL — AuraFlow (pile-t5xl)",
    "llama-3.1-8b": "Llama-3.1-8B — HiDream-I1 (llama-3.1-8b)",
    "umt5-xxl": "UMT5-XXL — Wan 2.1 / Wan 2.2 (umt5-xxl)",
}

# (family_short_name, format_key) pairs actually convert+load+render-tested in
# ComfyUI by this project (same evidence bar _RENDER_VERIFIED_MIXED/
# _RENDER_CONFIRMED_BAD hold diffusion models to). 2026-08-12: Qwen3-4B tested
# across GGUF (F16/Q8_0/Q6_K, 4 seeds each) and all 4 safetensors formats --
# every format loaded and rendered without any format-specific defect. An
# unintended figure appeared in ~1-in-4 seeds, but at the *same* rate on the
# unquantized F16 baseline too (see CHANGELOG's "Corrected" entry), so it's
# base-model prompt-adherence noise, not evidence against any format here.
_TE_RENDER_VERIFIED: set[tuple[str, str]] = {
    ("qwen3-4b", "GGUF"),
    ("qwen3-4b", "FP8"),
    ("qwen3-4b", "FP8_MIXED"),
    ("qwen3-4b", "NVFP4"),
    ("qwen3-4b", "NVFP4_MIXED"),
    # qwen3-4b INT8/INT8_MIXED, 2026-08-13: 2 checkpoints (Z-Image Base +
    # its juggernautZ fine-tune), loaded with the correct native
    # `CLIPLoader` node -- both clean, no format-specific defect. (An
    # earlier same-day batch of 6 checkpoints appeared to show these
    # formats totally broken -- full-image structured noise -- but that
    # was a workflow error, not a conversion bug: those renders all fed
    # the .safetensors INT8 output through ComfyUI-GGUF's `CLIPLoaderGGUF`
    # node, which expects an actual .gguf file and can't decode a
    # ConvRot-rotated INT8 safetensors tensor. See
    # `_TE_RENDER_CONFIRMED_BAD`'s docstring.)
    ("qwen3-4b", "INT8"),
    ("qwen3-4b", "INT8_MIXED"),
    # qwen3-8b, 2026-08-13 (FLUX.2 Klein 9B's own text encoder): 3 same-
    # seed/prompt comparisons against the unquantized BF16 baseline for
    # FP8/FP8_MIXED/NVFP4/NVFP4_MIXED, 2 for INT8/INT8_MIXED (added to the
    # dropdown later in the same session) -- zero visible deviation on every
    # seed for every format, including NVFP4/NVFP4_MIXED, which DID show
    # drift when quantizing the FLUX.2 Klein DiT itself in the same session
    # (see safetensors_quant.py's _RENDER_VERIFIED_MIXED docstring). Text-
    # encoder quantization only perturbs the conditioning vector fed into an
    # otherwise full-precision DiT, not the sampling trajectory's own
    # numerics -- evidently a lot more forgiving here than quantizing DiT
    # weights directly. GGUF (Q5_K_M) added 2026-08-13, 2 same-seed/prompt
    # comparisons: minor conditioning drift (a face-tattoo detail, a small
    # ornament gap) but same subject/composition on both seeds -- comparable
    # to the conditioning noise qwen3-4b's GGUF K-quants already showed, not
    # a format-specific defect.
    ("qwen3-8b", "GGUF"),
    ("qwen3-8b", "FP8"),
    ("qwen3-8b", "FP8_MIXED"),
    ("qwen3-8b", "INT8"),
    ("qwen3-8b", "INT8_MIXED"),
    ("qwen3-8b", "NVFP4"),
    ("qwen3-8b", "NVFP4_MIXED"),
    # pile-t5xl (AuraFlow's aura_t5, 2026-08-18): GGUF Q4_K_M render-tested
    # clean across 2 motifs (vampire portrait, haunted-house/blood-moon) via
    # both an F16 and an FP8_MIXED aura_flow_0.3 diffusion model, matching
    # the fully unquantized F16/F16 baseline exactly. Unlike FP8/INT8/NVFP4
    # safetensors (_TE_RENDER_CONFIRMED_BAD below -- structurally unsupported,
    # aura_t5.py has no quantization_metadata wiring at all), GGUF goes
    # through ComfyUI-GGUF's own CLIPLoaderGGUF node, an independent
    # quantized-loading path that doesn't depend on that wiring -- confirms
    # this is a real, working alternative for this text-encoder family.
    ("pile-t5xl", "GGUF"),
    # llama-3.1-8b / t5-xxl (HiDream-I1's Llama-3.1-8B-Instruct and T5-XXL
    # encoders, 2026-08-18): FP8/FP8_MIXED/INT8/INT8_MIXED/NVFP4/NVFP4_MIXED
    # all render-tested clean at a fixed seed via a live HiDream-I1-Dev
    # workflow, matching the unquantized baseline's composition/identity --
    # sources were Comfy-Org's own fp8_scaled repackaging, dequantized and
    # re-quantized through this tool (exercises the same `.scale_weight`
    # fix documented in CHANGELOG.md). T5-XXL NVFP4/NVFP4_MIXED initially
    # crashed ComfyUI's `load_state_dict` on `relative_attention_bias.weight`
    # (a [32, 64] lookup table read via a bare nn.Embedding, same dequant-
    # bypass class as CLIP's `position_embedding`) -- fixed by adding it to
    # `_TEXT_ENCODER_MODEL_ARCH.keys_shape_critical` in text_encoder_convert.py,
    # re-converted and re-render-tested clean. A separate, more spatially
    # complex two-subject prompt showed a non-monotonic compositional
    # sensitivity (subject placement relative to a piano prop shifted
    # between formats, not correlated with quantization strength -- F16
    # showed it, FP8 didn't) -- logged as an observation, not a defect, in
    # docs/issues_analysis.md; it didn't reproduce on the simpler prompt
    # used for the pass/fail render test here.
    ("llama-3.1-8b", "FP8"),
    ("llama-3.1-8b", "FP8_MIXED"),
    ("llama-3.1-8b", "INT8"),
    ("llama-3.1-8b", "INT8_MIXED"),
    ("llama-3.1-8b", "NVFP4"),
    ("llama-3.1-8b", "NVFP4_MIXED"),
    ("t5-xxl", "FP8"),
    ("t5-xxl", "FP8_MIXED"),
    ("t5-xxl", "INT8"),
    ("t5-xxl", "INT8_MIXED"),
    ("t5-xxl", "NVFP4"),
    ("t5-xxl", "NVFP4_MIXED"),
    # llama-3.1-8b/t5-xxl GGUF Q4_K_M (2026-08-19): both built via
    # convert_text_encoder_kquant() against the new vendored
    # text_encoder_configs/llama-3.1-8b/ config (see CHANGELOG.md for the
    # pad_token/unk_token vocab-overflow fix that took to build it), loaded
    # through ComfyUI-GGUF's QuadrupleCLIPLoaderGGUF alongside the other 3
    # HiDream encoders, render-tested clean at the same fixed seed against
    # the FP8 baseline -- composition/identity/outfit identical, only
    # cosmetic background-text-glyph variance.
    ("llama-3.1-8b", "GGUF"),
    ("t5-xxl", "GGUF"),
    # umt5-xxl (Wan 2.2's text encoder, 2026-08-19): 8 same-seed I2V
    # generations against a fixed image+prompt, varying only this encoder's
    # format -- diffusion model held constant at wan's verified NVFP4_MIXED
    # (see safetensors_quant.py's _RENDER_VERIFIED_MIXED). Character/pose/
    # outfit/background identical to the original scaled_fp8 checkpoint's
    # own baseline for every format except plain INT8 -- see
    # _TE_RENDER_TESTED_DRIFT below for that one. F16 not listed here --
    # always VERIFIED unconditionally, see text_encoder_support_level()'s
    # docstring. GGUF Q4_K_M added same day, one more render in the same
    # batch/seed/image -- blink-cycle timing matched the baseline (not
    # plain INT8's off-timing above), everything else identical.
    ("umt5-xxl", "GGUF"),
    ("umt5-xxl", "FP8"),
    ("umt5-xxl", "FP8_MIXED"),
    ("umt5-xxl", "INT8_MIXED"),
    ("umt5-xxl", "NVFP4"),
    ("umt5-xxl", "NVFP4_MIXED"),
    # qwen2.5-vl-7b (Qwen2.5-VL-7B-Huihui-Abliterated, Qwen-Image-Edit-2511's
    # text encoder, 2026-08-20 batch, reviewed 2026-08-23): all 6 safetensors
    # formats render-tested clean via the fixed-seed edit workflow (FP8
    # diffusion model held constant, varying only the text-encoder format),
    # matching the unquantized BF16 baseline's composition/identity/outfit
    # exactly -- only trivial background-figure variance across renders.
    # GGUF is NOT here -- see _TE_RENDER_CONFIRMED_BAD above, structurally
    # missing its vision tower and unusable for this family's actual
    # (image-conditioning) use case.
    ("qwen2.5-vl-7b", "FP8"),
    ("qwen2.5-vl-7b", "FP8_MIXED"),
    ("qwen2.5-vl-7b", "INT8"),
    ("qwen2.5-vl-7b", "INT8_MIXED"),
    ("qwen2.5-vl-7b", "NVFP4"),
    ("qwen2.5-vl-7b", "NVFP4_MIXED"),
    # mistral-small-3.2-24b (mistral_3_small_flux2, FLUX.2 dev's text
    # encoder, 2026-08-23): all 6 safetensors formats render-tested clean
    # via the fixed-seed edit workflow (FP8_MIXED diffusion model held
    # constant, varying only the text-encoder format), matching the source
    # fp4_mixed baseline's composition/identity/wounds/medallion exactly --
    # no visible deviation on any format. Source was FLUX.2's bespoke
    # 30-layer-trimmed checkpoint (see the GGUF entry below) -- these are
    # the ONLY known-good render evidence for that specific packaging;
    # unrelated to and don't imply anything about the untrimmed, standard
    # Mistral-Small-3.2-24B-Instruct-2506 checkpoint ERNIE-Image uses under
    # the same family name, still untested. This run is also what
    # discovered and fixed the "tekken_model" sentinel-tensor corruption
    # (docs/issues_analysis.md #23) -- every file here was rebuilt after
    # that fix, not from the first (silently-broken) batch.
    ("mistral-small-3.2-24b", "FP8"),
    ("mistral-small-3.2-24b", "FP8_MIXED"),
    ("mistral-small-3.2-24b", "INT8"),
    ("mistral-small-3.2-24b", "INT8_MIXED"),
    ("mistral-small-3.2-24b", "NVFP4"),
    ("mistral-small-3.2-24b", "NVFP4_MIXED"),
}

# Mirrors _RENDER_CONFIRMED_BAD's role for diffusion models -- (family,
# format) pairs with direct render evidence of producing wrong/broken
# output. 2026-08-13: a batch of
# 8 renders across 6 Z-Image checkpoints briefly looked like qwen3-4b
# INT8/INT8_MIXED were confirmed-broken (full-image structured noise) --
# turned out to be a workflow error, not a conversion bug: the broken
# renders all loaded the .safetensors INT8 output through ComfyUI-GGUF's
# `CLIPLoaderGGUF` node (for .gguf files), not the native `CLIPLoader`
# node this format actually needs. The 3 renders that used `CLIPLoader`
# correctly (Z-Image Base + its juggernautZ fine-tune, INT8 and
# INT8_MIXED) were clean -- see _TE_RENDER_VERIFIED below.
_TE_RENDER_CONFIRMED_BAD: set[tuple[str, str]] = {
    # Not a render defect -- GGUF conversion itself is structurally
    # impossible for these two families: llama.cpp's convert_hf_to_gguf.py
    # has no CLIPModel/CLIPTextModel converter at all (confirmed 2026-08-14
    # by grepping the vendored .llama.cpp checkout, zero matches), and
    # ComfyUI-GGUF's CLIPLoaderGGUF node has no CLIP-L/bigG decode path
    # either -- every GGUF outtype/K-quant fails identically
    # (text_encoder_convert.py's _reject_if_gguf_unsupported() now fails
    # fast with this explanation instead of llama.cpp's opaque "Model
    # CLIPModel is not supported"). BAD is a stretch of its usual "render-
    # tested and confirmed broken" meaning (this never gets far enough to
    # render), but UNKNOWN would wrongly invite retesting something that can
    # never succeed -- BAD is the more useful signal of the four states here.
    ("clip-l", "GGUF"),
    ("clip-bigg", "GGUF"),
    # qwen2.5-vl-7b GGUF: conversion succeeds and the file loads, but it's
    # unusable for Qwen-Image-Edit's actual use case -- found 2026-08-23
    # render-testing Qwen-Image-Edit-2511, crashed with "mat1 and mat2
    # shapes cannot be multiplied (780x1280 and 3840x1280)" inside the
    # vision tower's qkv layer. Root cause: text_encoder_convert.py drives
    # plain llama.cpp's convert_hf_to_gguf.py without --mmproj, which only
    # exports the language-model tower -- verified with gguf.GGUFReader
    # that the built file has 0 of 339 tensors starting with "visual.",
    # versus 650 present in the equivalent NVFP4_MIXED safetensors build.
    # --mmproj mode would export a vision tower, but into a separate mmproj
    # file using llama.cpp's own multimodal tensor naming, which ComfyUI-
    # GGUF's CLIPLoaderGGUF can't consume either (expects the vision tower
    # inline under ComfyUI's own key names). Upstream/tooling gap, not
    # fixable in this project's own code -- see docs/issues_analysis.md #22.
    # Text-only (no image input) use was not tested and might work, but
    # every Qwen-Image-Edit workflow feeds a reference image, so BAD is the
    # correct signal for this family's actual use case.
    ("qwen2.5-vl-7b", "GGUF"),
    # mistral-small-3.2-24b GGUF: not a render defect -- conversion can't
    # even start. FLUX.2 dev's mistral_3_small_flux2_*.safetensors is a
    # bespoke-trimmed variant (30 transformer layers, no final norm/lm_head)
    # that doesn't match this family's vendored 40-layer
    # Mistral-Small-3.2-24B-Instruct-2506 config -- detect_text_encoder_
    # family() correctly returns None (layer-count signature mismatch), and
    # forcing the vendored repo ID would apply the wrong hyperparameters.
    # Found 2026-08-23 sanity-checking before a batch conversion (caught
    # before wasting a run, not from a crashed build). Even upstream
    # (city96/ComfyUI-GGUF#367) has this as an open, unresolved feature
    # request -- no known-correct config exists yet anywhere. See
    # support_reason() for the full tooltip, including the caveat that this
    # is specific to the FLUX.2 packaging, not necessarily the untrimmed
    # checkpoint ERNIE-Image uses under the same family name.
    ("mistral-small-3.2-24b", "GGUF"),
    # Also not a render defect in the usual sense -- a genuine ComfyUI-side
    # gap, found 2026-08-14 render-testing SDXL's clip_g: FP8 loaded without
    # error but rendered solid black, NVFP4 crashed outright ("mat1 and mat2
    # shapes cannot be multiplied", the packed on-disk shape used raw). Two
    # real bugs in this tool were found and fixed along the way (missing
    # keys_shape_critical protection for position_embedding and
    # text_projection -- see text_encoder_convert.py), but the root cause
    # survives both fixes: comfy/sd1_clip.py's CLIPTextModel.__init__() only
    # selects the quantization-aware MixedPrecisionOps when
    # model_options["quantization_metadata"] is set, and comfy/sd.py's
    # load_text_encoder_state_dicts() only ever sets that key inside its
    # CLIPType.MINIMAX branch -- the standard TEModel.CLIP_G/CLIP_L path
    # (every SDXL load) always falls through to plain comfy.ops.manual_cast,
    # which has no .comfy_quant/weight_scale-aware loading at all. This
    # tool's own .comfy_quant sidecar synthesis (comfy.utils.convert_old_
    # quants(), confirmed reading our file-level _quantization_metadata
    # correctly) never even gets consulted, regardless of how correct the
    # output file is. Every FP8/INT8/NVFP4 (plain and mixed) variant fails
    # identically for both CLIP-L and CLIP-G until ComfyUI wires up
    # quantization_metadata for the standard CLIP text-encoder path too.
    #
    # This gap is architectural, not SDXL-specific: HiDream-I1 also uses
    # CLIP-L/CLIP-G as 2 of its 4 text encoders, and comfy/text_encoders/
    # hidream.py's hidream_clip() factory (checked 2026-08-18) only accepts
    # t5_quantization_metadata/llama_quantization_metadata kwargs -- no
    # clip_l_/clip_g_quantization_metadata equivalent exists there either.
    # These clip-l/clip-bigg entries below therefore apply to HiDream-I1's
    # CLIP-L/CLIP-G too, even though HiDream itself was never separately
    # render-tested for this (only Llama-3.1-8B and T5-XXL were, see
    # _TE_RENDER_VERIFIED below) -- HiDream is NOT fully quantizable end to
    # end: F16 safetensors is the only safe format for 2 of its 4 encoders.
    ("clip-l", "FP8"), ("clip-l", "FP8_MIXED"),
    ("clip-l", "INT8"), ("clip-l", "INT8_MIXED"),
    ("clip-l", "NVFP4"), ("clip-l", "NVFP4_MIXED"),
    ("clip-bigg", "FP8"), ("clip-bigg", "FP8_MIXED"),
    ("clip-bigg", "INT8"), ("clip-bigg", "INT8_MIXED"),
    ("clip-bigg", "NVFP4"), ("clip-bigg", "NVFP4_MIXED"),
    # pile-t5xl (AuraFlow's aura_t5, 2026-08-18): render-tested via the
    # verified aura_flow_0.3-NVFP4_MIXED diffusion model (fixed seed) --
    # FP8 and INT8 both produced complete full-image structured noise, no
    # relation to the prompt at all (worse than clip-g/clip-l's "loads but
    # wrong" failure mode above). Root cause is the same ComfyUI-side
    # quantization_metadata gap, but a level further back:
    # comfy/text_encoders/aura_t5.py's AuraT5Model is a plain
    # sd1_clip.SD1ClipModel subclass with no *_quantization_metadata
    # parameter at all -- unlike flux_clip/sd3_clip/hidream_clip/lumina2's
    # te()/qwen_image's te(), which all accept and wire through their own
    # quantization-metadata kwarg (comfy/text_encoders/flux.py,
    # sd3_clip.py, hidream.py, lumina2.py, qwen_image.py). AuraFlow's text
    # encoder never got this wiring added upstream -- structurally
    # unsupported, not something this tool's output can route around, same
    # class of gap as clip-l/clip-bigg above. NVFP4_MIXED render-tested
    # separately (2026-08-18, user report): hard crash, not just silent
    # noise -- "mat1 and mat2 shapes cannot be multiplied (256x2048 and
    # 1024x2048)" inside t5.py's SelfAttention.q Linear (comfy/ops.py's
    # plain forward_comfy_cast_weights path). 1024 == 2048 // 2: NVFP4
    # halves the on-disk last dim when packing, and with no quantization-
    # aware loading wired for this family, the packed bytes get read raw as
    # a plain weight matrix -- the dimension mismatch crashes outright
    # before any garbage output is even possible.
    # _TEXT_ENCODER_MODEL_ARCH (text_encoder_convert.py) has no keys_hiprec
    # for text encoders, only keys_shape_critical (which doesn't cover
    # per-layer attention Linears like SelfAttention.q), so plain and MIXED
    # are equally affected here -- both entries added on the strength of
    # this one crash, consistent with FP8/INT8's identical root cause
    # rather than requiring separate plain-format evidence.
    ("pile-t5xl", "FP8"), ("pile-t5xl", "FP8_MIXED"),
    ("pile-t5xl", "INT8"), ("pile-t5xl", "INT8_MIXED"),
    ("pile-t5xl", "NVFP4"), ("pile-t5xl", "NVFP4_MIXED"),
}

# (family, format_key) pairs render-tested with visible-but-tolerable
# deviation from the uncompressed baseline -- mirrors _RENDER_TESTED_DRIFT's
# role for diffusion models (model_support.py's CAUTION/UNKNOWN split,
# 2026-08-13). Usable but visible conditioning drift is still CAUTION;
# older clean/usable samples must not hide that observation.
_TE_RENDER_TESTED_DRIFT: set[tuple[str, str]] = {
    ('qwen3-8b', 'GGUF'),
    # umt5-xxl plain INT8 (Wan 2.2, 2026-08-19): same 8-render I2V batch as
    # _TE_RENDER_VERIFIED's umt5-xxl entries above -- character/pose/outfit/
    # background identical to every other format's baseline, but the
    # generated motion's blink cycle landed a few frames off (eyes open at
    # the same frame index every other format shows them closed/mid-blink).
    # Not a composition/identity failure, not INT8's usual keys_hiprec-
    # protection gap either (INT8_MIXED, tested in the same batch, matched
    # the baseline exactly) -- most likely ordinary seed-level sensitivity
    # to the small numeric difference INT8 quantization makes to the
    # conditioning vector, the same class of variance qwen3-8b's GGUF
    # drift above was. Real enough to note, tolerable enough not to be BAD.
    ("umt5-xxl", "INT8"),
}

_TE_DRIFT_NOTES = {
    ('qwen3-8b', 'GGUF'): (
        'Qwen3 8B GGUF Q5_K_M: two same-seed/prompt comparisons retained the '
        'subject/composition but changed a face-tattoo detail and a small '
        'ornament gap. Usable conditioning drift; not evidence that every '
        'GGUF precision drifts. Source: model_support.py, 2026-08-13 tests.'),
    ('umt5-xxl', 'INT8'): (
        'UMT5-XXL, Wan 2.2 I2V: character, pose, outfit and background matched '
        'the baseline, but blink timing shifted by a few frames. INT8_MIXED '
        'matched the baseline in that batch. Source: model_support.py, '
        '2026-08-19 tests.'),
}


def text_encoder_support_level(family: str, format_key: str) -> str:
    """Return SUPPORT_VERIFIED / SUPPORT_CAUTION / SUPPORT_BAD / SUPPORT_UNKNOWN
    for one (vendored text-encoder family, format) pair. Same four-state
    evidence bar as support_level() above: VERIFIED means actually
    converted, loaded, and rendered correctly with no visible deviation;
    CAUTION means render-tested with visible-but-tolerable deviation
    (_TE_RENDER_TESTED_DRIFT); BAD means actually render-tested and
    confirmed to produce broken/garbage output (not just "differs from the
    unquantized baseline" -- prompt-adherence variance that also occurs on
    F16 doesn't count, see _TE_RENDER_VERIFIED's docstring); UNKNOWN means
    never actually rendered, no evidence either way.

    F16 (the safetensors precision-cast column, not the GGUF-outtype "F16"
    collapsed into the GGUF column): always VERIFIED, unconditionally, same
    reasoning as support_level()'s F16/F16_MIXED bullet for diffusion models
    -- a plain dtype cast never touches ComfyUI's quantized-compute code path
    at all, so it carries none of the architecture-dependent risk the
    quantized formats do. Checked first, before any per-family evidence
    lookup, so a family can't accidentally end up in _TE_RENDER_CONFIRMED_BAD
    for "F16" and silently flip this.
    """
    if format_key == "F16":
        return SUPPORT_VERIFIED
    if (family, format_key) in _TE_RENDER_CONFIRMED_BAD:
        return SUPPORT_BAD
    if (family, format_key) in _TE_RENDER_TESTED_DRIFT:
        return SUPPORT_CAUTION
    if (family, format_key) in _TE_RENDER_VERIFIED:
        return SUPPORT_VERIFIED
    return SUPPORT_UNKNOWN


def build_text_encoder_support_table() -> list[dict]:
    """Return one row per TEXT_ENCODER_FAMILY_DISPLAY_NAMES entry: display
    name plus a support level per TEXT_ENCODER_TABLE_FORMATS column --
    mirrors build_support_table() below for the diffusion-model table."""
    rows = []
    for family, display_name in TEXT_ENCODER_FAMILY_DISPLAY_NAMES.items():
        row = {"family": family, "display_name": display_name}
        for _, format_key in TEXT_ENCODER_TABLE_FORMATS:
            row[format_key] = text_encoder_support_level(family, format_key)
            row[f"{format_key}__reason"] = text_encoder_support_reason(family, format_key)
        rows.append(row)
    rows.sort(key=lambda r: r["display_name"])
    return rows


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
            row[f"{format_key}__reason"] = support_reason(instance.arch, format_key)
            if instance.arch == 'lumina2' and format_key == 'NVFP4_MIXED':
                row[f'{format_key}__scope'] = 'Profile-dependent drift'
            if instance.arch == 'qwen_image' and format_key == 'GGUF':
                row[f'{format_key}__scope'] = 'Q4_K_M smoke test'
            if instance.arch == 'krea2' and format_key == 'GGUF':
                row[f'{format_key}__scope'] = 'Requires molbal · tests pending'
        rows.append(row)
    rows.sort(key=lambda r: r["display_name"])
    return rows


def build_workbench_support_tables() -> dict:
    """Project registry evidence onto explicit variants without changing export guards.

    Older family-wide results must not certify a different checkpoint/profile.
    Pending labels describe local integration, not failed images or upstream bans.
    Classic tables and converter classifications retain their existing contracts.
    """
    tables = {}
    for kind, formats, source_rows in (
        ('diffusion', TABLE_FORMATS, build_support_table()),
        ('text_encoder', TEXT_ENCODER_TABLE_FORMATS, build_text_encoder_support_table()),
    ):
        rows = []
        for source in source_rows:
            row = dict(source)
            key = row.get('arch', row.get('family'))
            row['id'] = key
            row['display_name'] = row['display_name'].rsplit(' (', 1)[0].replace(' Family', '').replace('Lumina-Image 2.0', 'Lumina 2.0')
            if key == 'krea2':
                # Native city96 stays guarded; evidence belongs to the molbal path.
                row['GGUF'] = SUPPORT_CAUTION
                row['GGUF__scope'] = 'Q4_0 / molbal loader'
                row['GGUF__format'] = 'Q4_0'
                row['GGUF__reason'] = (
                    'Lustify Krea 2, Q4_0 via the pinned molbal adapter/loader: two '
                    'same-seed cat/hat and dog/scarf comparisons produced usable images '
                    'with small changes in eyes, muzzle, fur and scarf detail versus the '
                    'installed INT8 ConvRot source. A temporary reconstructed F16 source '
                    'was used for export; this is not an unquantized-original comparison. '
                    'ComfyUI 0.38.0, Kitchen 0.2.36, RTX 5080; 512px, eight Euler/simple '
                    'steps, CFG 1, seed 1212121. Four successful renders, but both source and '
                    'GGUF had shared mosaic texture. A controlled 52-step source follow-up '
                    'substantially reduced it; the original recipe is not a clean quality '
                    'baseline. Raw/Turbo identity and a corrected quality comparison remain '
                    'pending. Molbal loader '
                    'required; other checkpoints/precisions, LoRA, forced offload and '
                    'Linux remain untested. Source: docs/krea2-render-validation.md.')
            if key == 'lumina2':
                row.update(id='lumina2_base', display_name='Z-Image Base / Lumina 2.0', precision_profile='auto')
                row['NVFP4_MIXED__reason'] = (
                    'Default/Base policy: older comparisons substantially changed composition, '
                    'pose and outfit. The newer usable Turbo-profile result belongs to the '
                    'separate Z-Image Turbo row, not this policy. Source: model_support.py '
                    'and docs/quantization-size-audit.md.')
                row.pop('NVFP4_MIXED__scope', None)
                turbo = {'id': 'lumina2_turbo', 'arch': key,
                         'display_name': 'Z-Image Turbo · Turbo profile',
                         'precision_profile': 'z_image_turbo'}
                for _, fmt in formats:
                    turbo[fmt] = SUPPORT_VERIFIED if fmt in {'F16', 'F16_MIXED'} else SUPPORT_UNKNOWN
                    turbo[fmt + '__reason'] = (
                        'Turbo-profile evidence is scoped to this checkpoint/policy. '
                        'No equivalent render validation is recorded for this format. '
                        'Source: docs/quantization-size-audit.md.')
                    if fmt in {'F16', 'F16_MIXED'}:
                        turbo[fmt + '__reason'] = 'Plain precision-cast path; supported by implementation reasoning, not a new Turbo render test.'
                turbo['NVFP4_MIXED'] = SUPPORT_CAUTION
                turbo['NVFP4_MIXED__scope'] = 'Two-prompt Turbo test'
                turbo['NVFP4_MIXED__reason'] = (
                    'Z-Image Turbo, z_image_turbo profile: usable cat/hat and dog/scarf '
                    'images, with hat, facial detail and slight framing differences versus '
                    'BF16. Two prompts, seed 1212121, nine Euler/simple steps, no LoRA. '
                    'This does not certify the default/Base policy or other quantizations. '
                    'Source: docs/quantization-size-audit.md, 2026-10-05 runtime tests.')
                # One stored single-pass workflow, fixed seeds and two active
                # LoRAs. This evidence must not leak into Base/default policy.
                for fmt in ('GGUF', 'FP8', 'FP8_MIXED', 'INT8', 'INT8_MIXED', 'NVFP4'):
                    turbo[fmt] = SUPPORT_CAUTION
                    if fmt == 'GGUF':
                        turbo['GGUF__format'] = 'Q4_K_M'
                    turbo[fmt + '__scope'] = 'Q4_K_M / 2 LoRAs' if fmt == 'GGUF' else 'Single-pass / 2 LoRAs'
                    turbo[fmt + '__reason'] = (
                        'Z-Image Turbo BF16 source, stored Seed Variance Enhancer single-pass '
                        'workflow: successful still-life render with two active LoRAs, but '
                        'visible changes to object placement, cloth and background details. '
                        '1120x1440, eight dpmpp_sde/ddim_uniform steps, CFG 1, shift 3.5; '
                        'stored sampler/enhancer seeds retained. The repeated source was '
                        'pixel-identical. Safetensors use z_image_turbo protection; GGUF '
                        'evidence covers only Q4_K_M through city96, not other precisions. '
                        'One checkpoint/prompt/workflow on ComfyUI 0.38.0, RTX 5080; '
                        'other contexts remain untested. Source: docs/zit-batch-validation.md.')
                rows.append(turbo)
            if key == 'mistral-small-3.2-24b':
                row.update(id=key + '_flux2_pruned', display_name='Mistral Small 3.2 24B · FLUX.2 pruned')
                row['GGUF__pending'] = 'true'
                row['GGUF__label'] = 'Configuration pending'
                row['GGUF__scope'] = '30-layer packaging'
                row['GGUF__reason'] = (
                    'The FLUX.2 safetensors packaging has 30 layers and omits final norm/lm_head '
                    'weights. Our 40-layer base configuration must not be forced onto it. '
                    'A matching configuration/export contract is pending. Upstream support '
                    'for the full Mistral encoder does not resolve this packaging mismatch. '
                    'Source: docs/unsupported-matrix-audit.md and model_support.py.')
                for _, fmt in formats:
                    if fmt != 'GGUF':
                        row[fmt + '__reason'] = (
                            'Evidence applies to the FLUX.2 pruned packaging, not the full '
                            '40-layer base encoder. Safetensors F16 is a precision cast; '
                            'the six quantized safetensors formats were render-tested in '
                            'the recorded FLUX.2 workflow. Source: model_support.py, 2026-08-23.')
                full = {'id': key + '_full', 'family': key, 'display_name': 'Mistral Small 3.2 24B · Full encoder'}
                for _, fmt in formats:
                    full[fmt] = SUPPORT_VERIFIED if fmt == 'F16' else SUPPORT_UNKNOWN
                    full[fmt + '__reason'] = (
                        'Full 40-layer base encoder: do not inherit the pruned FLUX.2 '
                        'checkpoint results. Local format-specific render validation '
                        'is pending. F16 is supported by precision-cast reasoning.')
                # The existing family-wide GGUF guard remains until the backend
                # distinguishes full/pruned sources. Do not expose a false handoff.
                full['GGUF'] = SUPPORT_BAD
                full['GGUF__pending'] = 'true'
                full['GGUF__label'] = 'Integration pending'
                full['GGUF__scope'] = '40-layer base encoder'
                full['GGUF__reason'] = (
                    'Upstream city96 supports the full Mistral encoder (issue #367 is '
                    'closed). Our family-wide GGUF guard and local validation still '
                    'need separate full/pruned handling. This is a local integration '
                    'status, not an upstream format ban. Source: docs/unsupported-matrix-audit.md.')
                rows.append(full)
            rows.append(row)
        rows.sort(key=lambda row: row['display_name'])
        format_keys = [key for _, key in formats]
        if kind == 'diffusion':
            # Display recorded prototype evidence without exposing a production target.
            fmt = 'INT4_CONVROT_MIXED'
            format_keys.insert(format_keys.index('INT8_MIXED') + 1, fmt)
            for row in rows:
                zit = row['id'] == 'lumina2_turbo'
                row[fmt] = SUPPORT_CAUTION if row['arch'] == 'sdxl' or zit else SUPPORT_UNKNOWN
                row[fmt + '__selectable'] = 'false'
                row[fmt + '__scope'] = ('ZIT / 2 LoRAs' if zit else
                                       'SDXL checkpoint test' if row['arch'] == 'sdxl' else 'Prototype')
                row[fmt + '__reason'] = (
                    'Z-Image Turbo BF16, z_image_turbo protection, 180 native INT4 ConvRot '
                    'Linear layers: successful single-pass still-life render with two active '
                    'LoRAs, but substantial cloth/background and framing drift. Kernel '
                    'Kitchen 0.2.36 on RTX 5080; production/version integration and forced '
                    'offload remain separate. Source: docs/zit-batch-validation.md. '
                    if zit else
                    'RealVisXL V50 Lightning diffusion UNet, mixed precision: nine successful '
                    'renders including LoRA, classic CPU offload and dynamic VRAM offload. '
                    'Visible changes to ears, face, framing and knitting versus the source. '
                    '58.22% smaller than the source UNet. ComfyUI 0.38.0, Kitchen 0.2.36, '
                    'RTX 5080. Evidence covers this checkpoint and policy only. '
                    if row['arch'] == 'sdxl' else
                    'No project render evidence for INT4 + ConvRot mixed precision on this '
                    'model/variant. SDXL results do not establish compatibility elsewhere. '
                ) + ('Production integration and version guards are pending; this prototype '
                     'cannot be selected for conversion. Source: docs/int4-convrot-validation.md.')
        tables[kind] = {'formats': format_keys, 'rows': rows}
    return tables
