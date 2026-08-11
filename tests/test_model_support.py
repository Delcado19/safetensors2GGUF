"""Tests for model_support.py — the per-architecture format support matrix."""

from __future__ import annotations

from model_support import (
    MODEL_DISPLAY_NAMES,
    SUPPORT_BAD,
    SUPPORT_CAUTION,
    SUPPORT_UNKNOWN,
    SUPPORT_VERIFIED,
    TABLE_FORMATS,
    support_level,
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

    def test_fp8_always_caution(self):
        # full_precision_matrix_mult=true fixes ComfyUI's compute path by
        # code reading (comfy/utils.py, comfy/ops.py), but no FP8 output from
        # this tool has ever been convert+load+render confirmed in ComfyUI on
        # any architecture -- that's CAUTION, not VERIFIED, under the same
        # bar INT8_MIXED is held to.
        assert support_level("lumina2", True, "FP8") == SUPPORT_CAUTION
        assert support_level("flux", True, "FP8_MIXED") == SUPPORT_CAUTION
        assert support_level("sdxl", False, "FP8") == SUPPORT_CAUTION

    def test_int8_verified_when_no_hiprec_layers(self):
        # No keys_hiprec means plain INT8 and INT8_MIXED produce identical
        # output -- no reason to caution either one.
        assert support_level("sdxl", False, "INT8") == SUPPORT_VERIFIED
        assert support_level("sdxl", False, "INT8_MIXED") == SUPPORT_VERIFIED

    def test_int8_mixed_verified_only_for_lumina2(self):
        # lumina2 is the only architecture render-tested end-to-end
        # (docs/issues_analysis.md #15).
        assert support_level("lumina2", True, "INT8_MIXED") == SUPPORT_VERIFIED
        assert support_level("flux", True, "INT8_MIXED") == SUPPORT_CAUTION

    def test_plain_int8_bad_on_lumina2_caution_elsewhere(self):
        # lumina2 has direct render-test evidence of corruption with plain
        # INT8 (docs/issues_analysis.md #15) -- BAD, not merely CAUTION.
        # Every other sensitive architecture shares the same risk class but
        # has never actually been rendered, so it stays CAUTION.
        assert support_level("lumina2", True, "INT8") == SUPPORT_BAD
        assert support_level("flux", True, "INT8") == SUPPORT_CAUTION

    def test_nvfp4_bad_on_lumina2_caution_elsewhere(self):
        # Direct render evidence (full-image noise, #15) exists only for
        # lumina2 -- BAD there, CAUTION everywhere else on the strength of
        # the shared dynamic-activation-quantization mechanism alone (no
        # verified safe mode, unlike FP8's full_precision_matrix_mult).
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

    def test_lumina2_row_flags_confirmed_bad_combinations(self):
        from model_support import build_support_table
        rows = build_support_table()
        lumina2_row = next(r for r in rows if r["arch"] == "lumina2")
        assert lumina2_row["INT8"] == SUPPORT_BAD
        assert lumina2_row["NVFP4"] == SUPPORT_BAD
        assert lumina2_row["NVFP4_MIXED"] == SUPPORT_BAD
