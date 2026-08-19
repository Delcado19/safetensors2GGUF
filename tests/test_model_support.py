"""Tests for model_support.py — the per-architecture format support matrix."""

from __future__ import annotations

from model_support import (
    MODEL_DISPLAY_NAMES,
    SUPPORT_BAD,
    SUPPORT_CAUTION,
    SUPPORT_UNKNOWN,
    SUPPORT_VERIFIED,
    TABLE_FORMATS,
    TEXT_ENCODER_FAMILY_DISPLAY_NAMES,
    TEXT_ENCODER_TABLE_FORMATS,
    build_text_encoder_support_table,
    support_level,
    text_encoder_support_level,
)
from models.architectures import arch_list


class TestModelDisplayNames:
    def test_every_arch_list_entry_has_a_display_name(self):
        for cls in arch_list:
            assert cls().arch in MODEL_DISPLAY_NAMES

    def test_display_names_cite_the_internal_arch_key(self):
        # Each display name must include its own internal arch key in
        # parentheses (project convention agreed with the user: "Z-Image
        # Turbo (Lumina)" style) so the table stays traceable to
        # models/architectures.py without a separate lookup.
        for arch_key, name in MODEL_DISPLAY_NAMES.items():
            assert f"({arch_key})" in name


class TestTableFormats:
    def test_covers_gguf_and_all_gui_safetensors_formats(self):
        keys = {key for _, key in TABLE_FORMATS}
        assert keys == {
            "GGUF", "F16", "F16_MIXED", "INT8", "INT8_MIXED",
            "FP8", "FP8_MIXED", "NVFP4", "NVFP4_MIXED",
        }


class TestSupportLevel:
    def test_gguf_always_verified(self):
        assert support_level("lumina2", True, "GGUF") == SUPPORT_VERIFIED
        assert support_level("sdxl", False, "GGUF") == SUPPORT_VERIFIED

    def test_f16_always_verified(self):
        assert support_level("lumina2", True, "F16") == SUPPORT_VERIFIED
        assert support_level("lumina2", True, "F16_MIXED") == SUPPORT_VERIFIED

    def test_fp8_unknown_except_lumina2_and_flux(self):
        # full_precision_matrix_mult=true fixes ComfyUI's compute path by
        # code reading (comfy/utils.py, comfy/ops.py), but no FP8_MIXED
        # output from this tool has ever been convert+load+render confirmed
        # in ComfyUI on architectures other than lumina2/flux (see
        # test_flux_fp8_int8_verified_render_tested below) -- UNKNOWN (never
        # rendered), not VERIFIED, under the same bar INT8_MIXED is held to.
        assert support_level("qwen_image", True, "FP8_MIXED") == SUPPORT_UNKNOWN
        assert support_level("cosmos", False, "FP8") == SUPPORT_UNKNOWN

    def test_flux_fp8_int8_verified_render_tested(self):
        # FLUX.2 Klein 9B, 2026-08-12: 3 same-seed/prompt comparisons (5
        # seeds total incl. batch variants) against the unquantized BF16
        # baseline showed FP8, FP8_MIXED, INT8 and INT8_MIXED all producing
        # output with zero visible deviation -- identity/pose/outfit/
        # background all matched exactly, every seed. See
        # safetensors_quant._RENDER_VERIFIED_MIXED's docstring for the full
        # writeup, including why plain FP8/INT8 clearing this bar isn't
        # architecture-specific luck for FP8 (full_precision_matrix_mult
        # protects it everywhere) but IS real evidence for INT8 (no such
        # runtime flag exists there).
        assert support_level("flux", True, "FP8") == SUPPORT_VERIFIED
        assert support_level("flux", True, "FP8_MIXED") == SUPPORT_VERIFIED
        assert support_level("flux", True, "INT8") == SUPPORT_VERIFIED
        assert support_level("flux", True, "INT8_MIXED") == SUPPORT_VERIFIED
        # NVFP4/NVFP4_MIXED were NOT cleared by this same seed batch -- 2 of 3
        # seeds showed visible composition drift for both variants while
        # FP8/INT8 stayed pixel-identical on those same seeds. Root-caused
        # and fixed 2026-08-13 (missing keys_shape_critical entries +
        # full_precision_matrix_mult never set for nvfp4); a re-test with
        # both fixes active came back clean -- see test_flux_nvfp4_verified
        # below and safetensors_quant._RENDER_VERIFIED_MIXED's docstring.

    def test_flux_nvfp4_verified(self):
        # 2026-08-13: models/architectures.py's ModelFlux.keys_shape_critical
        # was missing txt_in.weight/vector_in.in_layer.weight (NVFP4 halved
        # their on-disk last dim, corrupting ComfyUI's raw-shape hyperparameter
        # inference -- the same #9 bug class img_in.weight was already
        # protected against), and convert_safetensors.py never set
        # full_precision_matrix_mult for nvfp4 layers (FP8-only before this
        # fix), leaving NVFP4 on ComfyUI's dynamic-activation-quantization
        # compute path. Both fixed; 2 same-seed/prompt comparisons against the
        # BF16 baseline then showed only minor palette/decorative variance
        # (earring color, hat ornament) -- composition/identity/pose/outfit
        # all preserved, the same bar FP8/INT8 already cleared.
        assert support_level("flux", True, "NVFP4") == SUPPORT_VERIFIED
        assert support_level("flux", True, "NVFP4_MIXED") == SUPPORT_VERIFIED

    def test_fp8_lumina2_split_bad_plain_verified_mixed(self):
        # lumina2 has been directly render-tested for both FP8 variants:
        # plain FP8 confirmed corrupted (outfit/background drift, same-seed
        # comparison) -- BAD. FP8_MIXED confirmed correct (composition/
        # identity/outfit preserved, only a tolerable single-prop deviation,
        # second same-seed comparison) -- VERIFIED, same bar INT8_MIXED met.
        assert support_level("lumina2", True, "FP8") == SUPPORT_BAD
        assert support_level("lumina2", True, "FP8_MIXED") == SUPPORT_VERIFIED

    def test_int8_verified_when_no_hiprec_layers(self):
        # No keys_hiprec means plain INT8 and INT8_MIXED produce identical
        # output -- no reason to caution either one.
        assert support_level("sdxl", False, "INT8") == SUPPORT_VERIFIED
        assert support_level("sdxl", False, "INT8_MIXED") == SUPPORT_VERIFIED

    def test_int8_mixed_verified_only_for_lumina2_and_flux(self):
        # lumina2 and flux are the only architectures render-tested
        # end-to-end (docs/issues_analysis.md #15; flux: see
        # test_flux_fp8_int8_verified_render_tested above).
        assert support_level("lumina2", True, "INT8_MIXED") == SUPPORT_VERIFIED
        assert support_level("qwen_image", True, "INT8_MIXED") == SUPPORT_UNKNOWN

    def test_plain_int8_bad_on_lumina2_unknown_elsewhere(self):
        # lumina2 has direct render-test evidence of corruption with plain
        # INT8 (docs/issues_analysis.md #15) -- BAD, not merely UNKNOWN.
        # Every other untested sensitive architecture shares the same risk
        # class but has never actually been rendered, so it stays UNKNOWN
        # (flux is the one exception -- render-tested clean, see
        # test_flux_fp8_int8_verified_render_tested above).
        assert support_level("lumina2", True, "INT8") == SUPPORT_BAD
        assert support_level("qwen_image", True, "INT8") == SUPPORT_UNKNOWN

    def test_nvfp4_bad_on_lumina2_verified_on_flux_unknown_elsewhere(self):
        # lumina2's original BAD verdict (full-image noise, #15) was from a
        # pre-fix conversion. 2026-08-13: re-tested with the same
        # full_precision_nvfp4 fix that cleared flux -- no more noise, but one
        # of three tested prompts still showed a full composition/pose/outfit
        # change from baseline. Briefly moved to CAUTION on the strength of
        # "2 of 3 were clean", then back to BAD the same day: FP8/INT8 above
        # were judged BAD off a single failing test each showing the identical
        # failure pattern, so holding NVFP4 to a looser bar just because it
        # also had passing samples wasn't a consistent standard. flux is
        # VERIFIED (see test_flux_nvfp4_verified above). Every other
        # architecture has never actually been rendered with NVFP4 -- UNKNOWN,
        # not a blanket CAUTION/BAD on the strength of the shared mechanism
        # alone.
        assert support_level("lumina2", True, "NVFP4") == SUPPORT_BAD
        assert support_level("lumina2", True, "NVFP4_MIXED") == SUPPORT_BAD
        assert support_level("cosmos", False, "NVFP4_MIXED") == SUPPORT_UNKNOWN
        assert support_level("flux", True, "NVFP4") == SUPPORT_VERIFIED
        assert support_level("flux", True, "NVFP4_MIXED") == SUPPORT_VERIFIED

    def test_sdxl_verified_after_conv_and_label_emb_fixes(self):
        # 2026-08-14: FP8/INT8 initially rendered solid black (>=3D Conv2d
        # weights got quantized into a format ComfyUI can't load) and plain
        # NVFP4/NVFP4_MIXED crashed on load (label_emb.0.0.weight missing
        # from keys_shape_critical). Both fixed and re-tested clean across 2
        # prompts -- see safetensors_quant._RENDER_VERIFIED_MIXED's docstring.
        assert support_level("sdxl", False, "FP8") == SUPPORT_VERIFIED
        assert support_level("sdxl", False, "FP8_MIXED") == SUPPORT_VERIFIED
        assert support_level("sdxl", False, "NVFP4") == SUPPORT_VERIFIED
        assert support_level("sdxl", False, "NVFP4_MIXED") == SUPPORT_VERIFIED
        # INT8/INT8_MIXED were already VERIFIED via the `not
        # keys_hiprec_nonempty` branch (sdxl has no keys_hiprec) -- unaffected
        # by either bug, but same render evidence now backs it too.
        assert support_level("sdxl", False, "INT8") == SUPPORT_VERIFIED
        assert support_level("sdxl", False, "INT8_MIXED") == SUPPORT_VERIFIED

    def test_sd3_verified_after_y_embedder_fix(self):
        # 2026-08-16: plain NVFP4/NVFP4_MIXED crashed on load
        # (y_embedder.mlp.0.weight missing from keys_shape_critical,
        # corrupting adm_in_channels -- same pattern as sdxl's label_emb
        # fix). Fixed and re-tested clean; FP8/FP8_MIXED were already
        # render-tested clean before that -- see
        # safetensors_quant._RENDER_VERIFIED_MIXED's docstring.
        assert support_level("sd3", False, "FP8") == SUPPORT_VERIFIED
        assert support_level("sd3", False, "FP8_MIXED") == SUPPORT_VERIFIED
        assert support_level("sd3", False, "NVFP4") == SUPPORT_VERIFIED
        assert support_level("sd3", False, "NVFP4_MIXED") == SUPPORT_VERIFIED
        # INT8/INT8_MIXED were already VERIFIED via the `not
        # keys_hiprec_nonempty` branch (sd3 has no keys_hiprec).
        assert support_level("sd3", False, "INT8") == SUPPORT_VERIFIED
        assert support_level("sd3", False, "INT8_MIXED") == SUPPORT_VERIFIED

    def test_hidream_mixed_verified_plain_unknown_after_gate_fix(self):
        # 2026-08-17: ff_i.gate.weight (MoE router) wasn't in
        # keys_shape_critical -- plain NVFP4 crashed on load (halved last
        # dim) and plain INT8 produced severely corrupted output, both
        # because this tensor never goes through ComfyUI's quantized-
        # loading path. Fixed by adding it (hidream has keys_hiprec, so
        # sensitive=True). FP8_MIXED/INT8_MIXED were unaffected (already
        # skip quantizing hiprec tensors) and render-tested clean.
        assert support_level("hidream", True, "FP8_MIXED") == SUPPORT_VERIFIED
        assert support_level("hidream", True, "INT8_MIXED") == SUPPORT_VERIFIED
        # Plain FP8/INT8/NVFP4 all regenerated after the fix and
        # re-render-tested clean.
        assert support_level("hidream", True, "FP8") == SUPPORT_VERIFIED
        assert support_level("hidream", True, "INT8") == SUPPORT_VERIFIED
        assert support_level("hidream", True, "NVFP4") == SUPPORT_VERIFIED
        # NVFP4_MIXED (2026-08-19): quantized from the FP8 source (bf16
        # unavailable, user declined a third re-download), 2 same-seed
        # prompts vs. the FP8 baseline showed identical composition/
        # identity/outfit, only minor secondary-detail variance.
        assert support_level("hidream", True, "NVFP4_MIXED") == SUPPORT_VERIFIED

    def test_wan_nvfp4_mixed_verified(self):
        # 2026-08-19: Wan 2.2 I2V 14B ("DaSiWa SnatchKiss v11" fp8_mixed
        # community checkpoint, quantized from that FP8 source -- no bf16
        # available), 8 same-seed I2V renders via a live ComfyUI workflow.
        # This was "wan" arch's first ever render evidence (keys_hiprec/
        # keys_shape_critical were already populated from a published
        # community blacklist, but never actually exercised until now).
        assert support_level("wan", True, "NVFP4_MIXED") == SUPPORT_VERIFIED

    def test_aura_verified_except_plain_nvfp4_caution(self):
        # 2026-08-18: aura_flow_0.3, fixed seed via aura_t5. FP8/FP8_MIXED/
        # INT8/INT8_MIXED render-tested clean. NVFP4/NVFP4_MIXED initially
        # drifted -- root-caused to two compounding bugs: ModelAura had no
        # keys_hiprec (AdaLN modulation Linears quantized in every format),
        # and is_hiprec_st()'s dtype gate excluded float16 sources
        # (aura_flow_0.3 is F16-native), so keys_hiprec was inert even once
        # populated. Both fixed; NVFP4_MIXED re-render confirmed clean
        # across 4 motifs including the original failing vampire-portrait
        # prompt. Plain NVFP4 still showed visible facial-detail loss on
        # that same portrait prompt -- keys_hiprec never protects plain
        # mode, by design (same as every other architecture) -- CAUTION.
        assert support_level("aura", True, "FP8") == SUPPORT_VERIFIED
        assert support_level("aura", True, "FP8_MIXED") == SUPPORT_VERIFIED
        assert support_level("aura", True, "INT8") == SUPPORT_VERIFIED
        assert support_level("aura", True, "INT8_MIXED") == SUPPORT_VERIFIED
        assert support_level("aura", True, "NVFP4") == SUPPORT_CAUTION
        assert support_level("aura", True, "NVFP4_MIXED") == SUPPORT_VERIFIED

    def test_unknown_format_key_returns_unknown(self):
        assert support_level("sdxl", False, "BOGUS") == SUPPORT_UNKNOWN


class TestBuildSupportTable:
    def test_one_row_per_arch_list_entry(self):
        from model_support import build_support_table
        rows = build_support_table()
        assert len(rows) == len(arch_list)

    def test_row_has_display_name_and_every_format_column(self):
        from model_support import build_support_table
        rows = build_support_table()
        lumina2_row = next(r for r in rows if r["arch"] == "lumina2")
        assert lumina2_row["display_name"] == MODEL_DISPLAY_NAMES["lumina2"]
        for _, format_key in TABLE_FORMATS:
            assert format_key in lumina2_row

    def test_lumina2_row_matches_support_level_directly(self):
        from model_support import build_support_table
        rows = build_support_table()
        lumina2_row = next(r for r in rows if r["arch"] == "lumina2")
        assert lumina2_row["INT8"] == support_level("lumina2", True, "INT8")
        assert lumina2_row["INT8_MIXED"] == support_level("lumina2", True, "INT8_MIXED")

    def test_sdxl_row_has_no_hiprec_sensitive_int8_caution(self):
        from model_support import build_support_table
        rows = build_support_table()
        sdxl_row = next(r for r in rows if r["arch"] == "sdxl")
        assert sdxl_row["INT8"] == SUPPORT_VERIFIED
        assert sdxl_row["INT8_MIXED"] == SUPPORT_VERIFIED

class TestTextEncoderSupport:
    def test_covers_gguf_and_safetensors_formats(self):
        keys = {key for _, key in TEXT_ENCODER_TABLE_FORMATS}
        assert keys == {
            "GGUF", "F16", "FP8", "FP8_MIXED", "INT8", "INT8_MIXED", "NVFP4", "NVFP4_MIXED",
        }

    def test_every_family_format_pair_has_a_support_level(self):
        for family in TEXT_ENCODER_FAMILY_DISPLAY_NAMES:
            for _, format_key in TEXT_ENCODER_TABLE_FORMATS:
                assert text_encoder_support_level(family, format_key) in (
                    SUPPORT_VERIFIED, SUPPORT_CAUTION, SUPPORT_BAD, SUPPORT_UNKNOWN,
                )

    def test_untested_family_stays_unknown_on_every_format(self):
        # qwen2.5-vl-7b has no render-test evidence at all -- every cell
        # UNKNOWN, except F16: that's always unconditionally VERIFIED (a
        # plain precision cast, same reasoning as the diffusion-model
        # table's F16/F16_MIXED columns) -- see test_f16_always_verified
        # below. (t5-xxl was this test's example until 2026-08-18, when it
        # gained real render-test evidence via HiDream-I1's T5-XXL encoder
        # -- see _TE_RENDER_VERIFIED's docstring. clip-l/clip-bigg are NOT a
        # good "untested" example here either -- every one of their
        # quantized formats is now a confirmed structural impossibility,
        # not merely untested -- see test_clip_families_gguf_confirmed_bad
        # and test_clip_families_quantized_safetensors_confirmed_bad below.)
        for _, format_key in TEXT_ENCODER_TABLE_FORMATS:
            if format_key == "F16":
                continue
            assert text_encoder_support_level("qwen2.5-vl-7b", format_key) == SUPPORT_UNKNOWN

    def test_f16_always_verified(self):
        # Plain precision cast, never touches ComfyUI's quantized-compute
        # path -- unconditionally VERIFIED regardless of family, even ones
        # with zero other evidence (qwen2.5-vl-7b) or every other format
        # confirmed BAD (clip-l/clip-bigg, where F16 safetensors is the only
        # format that actually works).
        for family in ("qwen2.5-vl-7b", "clip-l", "clip-bigg", "qwen3-4b"):
            assert text_encoder_support_level(family, "F16") == SUPPORT_VERIFIED, family

    def test_clip_families_quantized_safetensors_confirmed_bad(self):
        # Not a render defect either -- a genuine ComfyUI-side gap found
        # 2026-08-14 render-testing SDXL's clip_g: comfy/sd1_clip.py only
        # selects the quantization-aware MixedPrecisionOps when
        # model_options["quantization_metadata"] is set, and comfy/sd.py's
        # load_text_encoder_state_dicts() only ever sets that key inside its
        # CLIPType.MINIMAX branch -- the standard SDXL CLIP-L/CLIP-G path
        # always falls through to plain comfy.ops.manual_cast, which ignores
        # any .comfy_quant/weight_scale sidecar entirely. FP8 loaded without
        # error but rendered solid black; NVFP4 crashed outright (packed
        # on-disk shape used raw). See model_support.py's
        # _TE_RENDER_CONFIRMED_BAD docstring for the full trace.
        for family in ("clip-l", "clip-bigg"):
            for fmt in ("FP8", "FP8_MIXED", "INT8", "INT8_MIXED", "NVFP4", "NVFP4_MIXED"):
                assert text_encoder_support_level(family, fmt) == SUPPORT_BAD, (family, fmt)

    def test_clip_families_gguf_confirmed_bad(self):
        # Not a render defect -- llama.cpp's convert_hf_to_gguf.py has no
        # CLIPModel/CLIPTextModel converter at all, so GGUF conversion fails
        # before it could ever render (see model_support.py's
        # _TE_RENDER_CONFIRMED_BAD comment and
        # text_encoder_convert._reject_if_gguf_unsupported).
        assert text_encoder_support_level("clip-l", "GGUF") == SUPPORT_BAD
        assert text_encoder_support_level("clip-bigg", "GGUF") == SUPPORT_BAD

    def test_pile_t5xl_confirmed_bad(self):
        # 2026-08-18: aura_t5 (Pile-T5-XL) render-tested via the verified
        # aura_flow_0.3-NVFP4_MIXED diffusion model. FP8/INT8 both produced
        # complete full-image structured noise -- no relation to the prompt
        # at all. NVFP4_MIXED crashed outright ("mat1 and mat2 shapes
        # cannot be multiplied (256x2048 and 1024x2048)" -- NVFP4's last-
        # dim packing halving 2048 to 1024). Root cause one level further
        # back than clip-l/clip-bigg's gap: comfy/text_encoders/aura_t5.py's
        # AuraT5Model never got *_quantization_metadata wiring added at all
        # (unlike flux_clip/sd3_clip/hidream_clip/lumina2/qwen_image, which
        # all have their own te() accepting one) -- structurally unsupported
        # by ComfyUI, not something this tool's output can route around.
        for fmt in ("FP8", "FP8_MIXED", "INT8", "INT8_MIXED", "NVFP4", "NVFP4_MIXED"):
            assert text_encoder_support_level("pile-t5xl", fmt) == SUPPORT_BAD, fmt

    def test_pile_t5xl_gguf_verified(self):
        # 2026-08-18: aura_t5 GGUF Q4_K_M render-tested clean across 2
        # motifs via both F16 and FP8_MIXED diffusion models, matching the
        # unquantized baseline. GGUF loads through ComfyUI-GGUF's own
        # CLIPLoaderGGUF node, independent of the quantization_metadata gap
        # that makes every safetensors-quant format BAD for this family.
        assert text_encoder_support_level("pile-t5xl", "GGUF") == SUPPORT_VERIFIED

    def test_qwen3_4b_verified_on_every_format(self):
        # 2026-08-12/13: convert+load+render-tested in ComfyUI across GGUF
        # (F16/Q8_0/Q6_K) and all 6 safetensors formats, no format-specific
        # defect found (see model_support.py's _TE_RENDER_VERIFIED comment).
        # INT8/INT8_MIXED: 2 Z-Image checkpoints, loaded through the correct
        # native `CLIPLoader` node -- clean. (A same-day batch through the
        # wrong node, ComfyUI-GGUF's `CLIPLoaderGGUF`, looked like total
        # breakage; that was a workflow error, not a conversion defect --
        # see _TE_RENDER_CONFIRMED_BAD's docstring.)
        for _, format_key in TEXT_ENCODER_TABLE_FORMATS:
            assert text_encoder_support_level("qwen3-4b", format_key) == SUPPORT_VERIFIED

    def test_qwen3_8b_verified_on_every_format(self):
        # 2026-08-13 (FLUX.2 Klein 9B's own text encoder): 3 same-seed/
        # prompt comparisons against the unquantized BF16 baseline for
        # FP8/FP8_MIXED/NVFP4/NVFP4_MIXED, 2 for INT8/INT8_MIXED -- zero
        # visible deviation on every seed for every format, including
        # NVFP4/NVFP4_MIXED (which DID show drift quantizing the DiT itself
        # in the same session -- text-encoder quantization only perturbs the
        # conditioning vector, not the sampling trajectory). GGUF (Q5_K_M)
        # added same day: 2 comparisons, minor conditioning drift but same
        # subject/composition both times -- verified alongside the rest.
        for _, format_key in TEXT_ENCODER_TABLE_FORMATS:
            assert text_encoder_support_level("qwen3-8b", format_key) == SUPPORT_VERIFIED

    def test_hidream_llama_and_t5xxl_verified_on_every_format(self):
        # 2026-08-18: FP8/FP8_MIXED/INT8/INT8_MIXED/NVFP4/NVFP4_MIXED
        # render-tested clean via a live HiDream-I1-Dev workflow.
        # 2026-08-19: GGUF Q4_K_M added -- built against the new vendored
        # text_encoder_configs/llama-3.1-8b/ config, loaded through
        # ComfyUI-GGUF's QuadrupleCLIPLoaderGGUF alongside the other 3
        # HiDream encoders, render-tested clean against the FP8 baseline.
        for _, format_key in TEXT_ENCODER_TABLE_FORMATS:
            assert text_encoder_support_level("llama-3.1-8b", format_key) == SUPPORT_VERIFIED
            assert text_encoder_support_level("t5-xxl", format_key) == SUPPORT_VERIFIED

    def test_umt5_xxl_verified_except_plain_int8_caution(self):
        # 2026-08-19: Wan 2.2's umt5-xxl encoder, 9 same-seed I2V renders
        # (fixed image+prompt, GGUF Q4_K_M added same day in a follow-up
        # render) against wan's verified NVFP4_MIXED diffusion model,
        # varying only this encoder's format. Character/pose/outfit/
        # background identical across every format except plain INT8, whose
        # generated motion's blink cycle landed a few frames off the other
        # formats' baseline -- not a composition/identity failure, judged
        # CAUTION (see _TE_RENDER_TESTED_DRIFT's comment for why it's not
        # INT8's usual keys_hiprec-protection gap: INT8_MIXED, same batch,
        # matched the baseline exactly).
        assert text_encoder_support_level("umt5-xxl", "F16") == SUPPORT_VERIFIED
        assert text_encoder_support_level("umt5-xxl", "GGUF") == SUPPORT_VERIFIED
        assert text_encoder_support_level("umt5-xxl", "FP8") == SUPPORT_VERIFIED
        assert text_encoder_support_level("umt5-xxl", "FP8_MIXED") == SUPPORT_VERIFIED
        assert text_encoder_support_level("umt5-xxl", "INT8") == SUPPORT_CAUTION
        assert text_encoder_support_level("umt5-xxl", "INT8_MIXED") == SUPPORT_VERIFIED
        assert text_encoder_support_level("umt5-xxl", "NVFP4") == SUPPORT_VERIFIED
        assert text_encoder_support_level("umt5-xxl", "NVFP4_MIXED") == SUPPORT_VERIFIED

    def test_build_text_encoder_support_table_covers_every_family(self):
        rows = build_text_encoder_support_table()
        assert {r["family"] for r in rows} == set(TEXT_ENCODER_FAMILY_DISPLAY_NAMES)

    def test_family_display_names_cite_the_internal_family_key(self):
        for family, name in TEXT_ENCODER_FAMILY_DISPLAY_NAMES.items():
            assert f"({family})" in name


class TestSupportLevelLumina2Table:
    def test_lumina2_row_flags_confirmed_bad_combinations(self):
        from model_support import build_support_table
        rows = build_support_table()
        lumina2_row = next(r for r in rows if r["arch"] == "lumina2")
        assert lumina2_row["INT8"] == SUPPORT_BAD
        assert lumina2_row["FP8"] == SUPPORT_BAD
        # NVFP4/NVFP4_MIXED: briefly CAUTION on 2026-08-13 after a re-test
        # with the full_precision_nvfp4 fix, reverted to BAD the same day for
        # consistency with FP8/INT8's bar -- see model_support.py's
        # _RENDER_CONFIRMED_BAD comment.
        assert lumina2_row["NVFP4"] == SUPPORT_BAD
        assert lumina2_row["NVFP4_MIXED"] == SUPPORT_BAD

    def test_lumina2_row_flags_render_verified_mixed_combinations(self):
        from model_support import build_support_table
        rows = build_support_table()
        lumina2_row = next(r for r in rows if r["arch"] == "lumina2")
        assert lumina2_row["INT8_MIXED"] == SUPPORT_VERIFIED
        assert lumina2_row["FP8_MIXED"] == SUPPORT_VERIFIED
