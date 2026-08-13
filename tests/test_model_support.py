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

    def test_fp8_caution_except_lumina2_and_flux(self):
        # full_precision_matrix_mult=true fixes ComfyUI's compute path by
        # code reading (comfy/utils.py, comfy/ops.py), but no FP8_MIXED
        # output from this tool has ever been convert+load+render confirmed
        # in ComfyUI on architectures other than lumina2/flux (see
        # test_flux_fp8_int8_verified_render_tested below) -- CAUTION, not
        # VERIFIED, under the same bar INT8_MIXED is held to.
        assert support_level("qwen_image", True, "FP8_MIXED") == SUPPORT_CAUTION
        assert support_level("sdxl", False, "FP8") == SUPPORT_CAUTION

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
        # NVFP4/NVFP4_MIXED were NOT cleared by the same seed batch -- 2 of 3
        # seeds showed visible composition drift for both variants while
        # FP8/INT8 stayed pixel-identical on those same seeds, confirming
        # (not just predicting by analogy) the existing CAUTION rating.
        assert support_level("flux", True, "NVFP4") == SUPPORT_CAUTION
        assert support_level("flux", True, "NVFP4_MIXED") == SUPPORT_CAUTION

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
        assert support_level("qwen_image", True, "INT8_MIXED") == SUPPORT_CAUTION

    def test_plain_int8_bad_on_lumina2_caution_elsewhere(self):
        # lumina2 has direct render-test evidence of corruption with plain
        # INT8 (docs/issues_analysis.md #15) -- BAD, not merely CAUTION.
        # Every other untested sensitive architecture shares the same risk
        # class but has never actually been rendered, so it stays CAUTION
        # (flux is the one exception -- render-tested clean, see
        # test_flux_fp8_int8_verified_render_tested above).
        assert support_level("lumina2", True, "INT8") == SUPPORT_BAD
        assert support_level("qwen_image", True, "INT8") == SUPPORT_CAUTION

    def test_nvfp4_bad_on_lumina2_caution_elsewhere(self):
        # Direct render evidence (full-image noise, #15) exists only for
        # lumina2 -- BAD there, CAUTION everywhere else on the strength of
        # the shared dynamic-activation-quantization mechanism alone (no
        # verified safe mode, unlike FP8's full_precision_matrix_mult) --
        # including flux, whose own render test showed visible composition
        # drift on NVFP4/NVFP4_MIXED rather than clearing them (see
        # test_flux_fp8_int8_verified_render_tested above).
        assert support_level("lumina2", True, "NVFP4") == SUPPORT_BAD
        assert support_level("lumina2", True, "NVFP4_MIXED") == SUPPORT_BAD
        assert support_level("sdxl", False, "NVFP4_MIXED") == SUPPORT_CAUTION
        assert support_level("flux", True, "NVFP4") == SUPPORT_CAUTION

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
            "GGUF", "FP8", "FP8_MIXED", "INT8", "INT8_MIXED", "NVFP4", "NVFP4_MIXED",
        }

    def test_every_family_format_pair_has_a_support_level(self):
        for family in TEXT_ENCODER_FAMILY_DISPLAY_NAMES:
            for _, format_key in TEXT_ENCODER_TABLE_FORMATS:
                assert text_encoder_support_level(family, format_key) in (
                    SUPPORT_VERIFIED, SUPPORT_CAUTION, SUPPORT_BAD, SUPPORT_UNKNOWN,
                )

    def test_untested_family_stays_caution_on_every_format(self):
        # clip-l has no render-test evidence at all -- every cell CAUTION.
        for _, format_key in TEXT_ENCODER_TABLE_FORMATS:
            assert text_encoder_support_level("clip-l", format_key) == SUPPORT_CAUTION

    def test_qwen3_4b_verified_on_every_render_tested_format(self):
        # 2026-08-12: convert+load+render-tested in ComfyUI across GGUF
        # (F16/Q8_0/Q6_K) and the FP8/NVFP4 safetensors formats, no
        # format-specific defect found (see model_support.py's
        # _TE_RENDER_VERIFIED comment). INT8/INT8_MIXED were added to the
        # table 2026-08-13 (safetensors_quant already supported them for
        # text encoders, the dropdown just never offered them) but haven't
        # been render-tested yet -- CAUTION, not VERIFIED, same evidence bar
        # as everything else.
        for _, format_key in TEXT_ENCODER_TABLE_FORMATS:
            if format_key in ("INT8", "INT8_MIXED"):
                assert text_encoder_support_level("qwen3-4b", format_key) == SUPPORT_CAUTION
            else:
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
        assert lumina2_row["NVFP4"] == SUPPORT_BAD
        assert lumina2_row["NVFP4_MIXED"] == SUPPORT_BAD

    def test_lumina2_row_flags_render_verified_mixed_combinations(self):
        from model_support import build_support_table
        rows = build_support_table()
        lumina2_row = next(r for r in rows if r["arch"] == "lumina2")
        assert lumina2_row["INT8_MIXED"] == SUPPORT_VERIFIED
        assert lumina2_row["FP8_MIXED"] == SUPPORT_VERIFIED
