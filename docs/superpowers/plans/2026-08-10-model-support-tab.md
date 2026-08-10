# Model Support Tab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Model Support" GUI tab showing which quantization formats (GGUF, F16/F16_MIXED, INT8/INT8_MIXED, FP8/FP8_MIXED, NVFP4/NVFP4_MIXED) this tool supports for each of its 12 detectable architectures, with click-to-apply cells and dynamically-annotated format dropdowns elsewhere in the GUI — and, as a prerequisite, fix and re-enable FP8 output (currently removed from the GUI) by defaulting it to ComfyUI's `full_precision_matrix_mult` safety flag.

**Architecture:** A new `model_support.py` module is the single source of truth for two things: (1) public display names per internal architecture key, and (2) a pure function computing a tri-state support level (`verified` / `caution` / `unknown`) for any (architecture, format) pair, encoded from facts already in `models/architectures.py` (`keys_hiprec` presence) plus this project's own render-testing history. `gui.py` renders that data as an HTML-celled `gr.Dataframe`, wires its `.select()` event to switch tabs and pre-fill the relevant format dropdown, and reuses the same `support_level()` function to annotate the two existing format dropdowns (GGUF quant, safetensors format) with a ⚠ prefix once a source file's architecture is known.

**Tech Stack:** Python 3.11+, Gradio 6.14 (`gr.Dataframe` with `datatype="html"` columns and its `.select()` event, `gr.Tabs(selected=...)` for programmatic tab switching), pytest, uv.

## Global Constraints

- No new dependencies — everything uses stdlib + already-installed Gradio/PyTorch.
- English for all code, comments, docstrings, commit messages, and doc updates (project convention, see CLAUDE.md).
- Every non-trivial function gets at least one test (project convention, see existing `tests/test_*.py`).
- `CHANGELOG.md` gets an `[Unreleased]` entry for both the FP8 fix and the new tab (project convention, enforced by the `docs-agent` pre-commit hook).
- Full `uv run pytest -q` must pass before each commit (enforced by the `test-agent` pre-commit hook, `.claude/settings.json`).
- Display names and the tri-state support data are editorial judgment calls this plan documents explicitly (see Task 3) — the user should review and correct them; they are not meant to be taken as unquestionable fact.
- Out of scope for this plan (explicitly deferred by the user): a VRAM-size input field on the main page to make quantization choices more size-aware. Revisit as a separate plan if wanted later.
- Out of scope for this plan: investigating whether NVFP4 has an equivalent to FP8's `full_precision_matrix_mult` safe mode (see `model_support.py`'s `support_level()` docstring, Task 3) — NVFP4 stays `SUPPORT_CAUTION` everywhere until that's done as its own plan.

---

## File Structure

| File | Change |
|---|---|
| `safetensors_quant_fp8.py` | Modify: `quantize_fp8_scaled()` gains no new params (unchanged) — this file's job stays "quantize one tensor," the safety flag is metadata about the *layer*, written centrally in `convert_safetensors.py` (Task 1) |
| `convert_safetensors.py` | Modify: write `full_precision_matrix_mult: true` into every FP8 layer's `.comfy_quant` config by default, with a new `full_precision_fp8: bool = True` parameter on `convert_to_safetensors()` for programmatic opt-out (Task 1) |
| `safetensors_quant.py` | Modify: re-add `FP8`/`FP8_MIXED` to `SAFETENSORS_DTYPE_CHOICES`; update `format_recommendation()` to return "ok, safe" for FP8 unconditionally (Task 2) |
| `model_support.py` | **Create**: `MODEL_DISPLAY_NAMES`, `TABLE_FORMATS`, `support_level()`, `build_support_table()` (Tasks 3-4) |
| `quantize.py` | Modify: `ALL_QUANT_CHOICES` labels gain "% smaller than F16" text derived from the existing `SIZE_RATIOS` table (Task 5) |
| `gui.py` | Modify: new "Model Support" tab (Task 6), click-to-apply wiring (Task 7), dynamic ⚠ annotation of the two format dropdowns (Task 8) |
| `tests/test_convert_safetensors.py` | Modify: FP8 flag tests (Task 1) |
| `tests/test_safetensors_quant.py` | Modify: FP8 re-enablement + `format_recommendation` tests (Task 2) |
| `tests/test_model_support.py` | **Create**: `support_level()` / `build_support_table()` tests (Tasks 3-4) |
| `tests/test_quantize.py` | Modify: `ALL_QUANT_CHOICES` label tests (Task 5) |
| `tests/test_gui.py` | Modify: new tab presence, click-to-apply, dropdown annotation tests (Tasks 6-8) |
| `README.md`, `docs/architecture.md`, `docs/issues_analysis.md`, `CHANGELOG.md` | Modify: document the FP8 fix and the new tab (Task 9) |

---

### Task 1: FP8 safety flag — `full_precision_matrix_mult`

**Files:**
- Modify: `convert_safetensors.py`
- Test: `tests/test_convert_safetensors.py`

**Interfaces:**
- Produces: `convert_to_safetensors(..., full_precision_fp8: bool = True)` — when the target format resolves to `float8_e4m3fn` and `full_precision_fp8` is True (default), every FP8 layer's entry in the written `_quantization_metadata` JSON gains `"full_precision_matrix_mult": true`.

- [ ] **Step 1: Write the failing test**

The metadata lives in the safetensors file header, read via `safetensors.safe_open(...).metadata()` — not `load_file`'s tensor dict.

```python
# tests/test_convert_safetensors.py — add to TestConvertToSafetensors
class TestFp8FullPrecisionFlag:
    def test_fp8_layer_config_defaults_full_precision_matrix_mult_true(self, tmp_path):
        import json
        from safetensors import safe_open

        src = _write_minimal_flux(tmp_path)
        dst, _ = convert_to_safetensors(str(src), target_key="FP8", overwrite=True)
        with safe_open(dst, framework="pt", device="cpu") as f:
            meta = json.loads(f.metadata()["_quantization_metadata"])
        layer_conf = next(iter(meta["layers"].values()))
        assert layer_conf["format"] == "float8_e4m3fn"
        assert layer_conf["full_precision_matrix_mult"] is True

    def test_fp8_full_precision_flag_can_be_disabled(self, tmp_path):
        import json
        from safetensors import safe_open

        src = _write_minimal_flux(tmp_path)
        dst, _ = convert_to_safetensors(
            str(src), target_key="FP8", overwrite=True, full_precision_fp8=False,
        )
        with safe_open(dst, framework="pt", device="cpu") as f:
            meta = json.loads(f.metadata()["_quantization_metadata"])
        layer_conf = next(iter(meta["layers"].values()))
        assert "full_precision_matrix_mult" not in layer_conf

    def test_full_precision_flag_absent_for_non_fp8_formats(self, tmp_path):
        import json
        from safetensors import safe_open

        src = _write_minimal_flux(tmp_path)
        dst, _ = convert_to_safetensors(str(src), target_key="INT8", overwrite=True)
        with safe_open(dst, framework="pt", device="cpu") as f:
            meta = json.loads(f.metadata()["_quantization_metadata"])
        layer_conf = next(iter(meta["layers"].values()))
        assert "full_precision_matrix_mult" not in layer_conf
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_convert_safetensors.py::TestFp8FullPrecisionFlag -v`
Expected: FAIL — `target_key="FP8"` currently raises nothing but the written metadata has no `full_precision_matrix_mult` key at all (KeyError on `layer_conf["full_precision_matrix_mult"]`), and `full_precision_fp8` isn't a recognized parameter (TypeError).

- [ ] **Step 3: Write minimal implementation**

In `convert_safetensors.py`, add the new parameter and thread it into the layer-config block:

```python
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
):
    """Convert a model checkpoint to a quantized .safetensors file.

    Args:
        ... (existing args unchanged) ...
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
    """
```

Then in the per-layer metadata block:

```python
            layer_conf: dict = {"format": quant_format}
            if quant_format == "int8_tensorwise":
                scale_t = quantized.get(f"{layer_key(key)}.weight_scale")
                if scale_t is not None and scale_t.numel() > 1:
                    layer_conf["convrot"] = True
                    layer_conf["convrot_groupsize"] = CONVROT_GROUP_SIZE
            elif quant_format == "float8_e4m3fn" and full_precision_fp8:
                layer_conf["full_precision_matrix_mult"] = True
            layer_formats[layer_key(key)] = layer_conf
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_convert_safetensors.py::TestFp8FullPrecisionFlag -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: all pass (no regressions — this only adds a new optional key when `quant_format == "float8_e4m3fn"`, a format nothing else currently produces since FP8 isn't in `SAFETENSORS_DTYPE_CHOICES` yet)

- [ ] **Step 6: Commit**

```bash
git add convert_safetensors.py tests/test_convert_safetensors.py
git commit -m "feat: default FP8 output to full_precision_matrix_mult=true

ComfyUI's MixedPrecisionOps.Linear.forward() (comfy/ops.py) skips the
quantized-compute branch entirely for any layer whose .comfy_quant config
has full_precision_matrix_mult=true -- weight is dequantized to the
model's compute_dtype and a plain full-precision matmul runs. ComfyUI's
own convert_old_quants() derives this flag automatically for legacy
community 'scaled_fp8' checkpoints (comfy/utils.py), which is why widely
circulated FP8 Civitai/HuggingFace checkpoints work reliably while this
project's own FP8 output (which never set the flag) hit the same
dynamic-activation-quantization corruption found on Lumina2/Z-Image.
Defaulting to true makes FP8 output architecture-independently safe --
unlike INT8's keys_hiprec (a per-layer bet on what needs protection),
this disables the risky compute path unconditionally for every layer.
full_precision_fp8=False opts back into real FP8 tensor-core speed for
users who understand the risk."
```

---

### Task 2: Re-enable FP8/FP8_MIXED in the GUI dropdown

**Files:**
- Modify: `safetensors_quant.py`
- Test: `tests/test_safetensors_quant.py`

**Interfaces:**
- Consumes: nothing new (uses existing `quantize_tensor_st()` FP8 branch, unchanged)
- Produces: `SAFETENSORS_DTYPE_CHOICES` includes `"FP8"`/`"FP8_MIXED"`; `format_recommendation(model_arch, "FP8")` / `(..., "FP8_MIXED")` return `("ok", <safe message>)` for every architecture.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_safetensors_quant.py — add to TestRegistry
class TestRegistry:
    ...
    def test_fp8_is_offered_again(self):
        keys = {key for _, key in SAFETENSORS_DTYPE_CHOICES}
        assert "FP8" in keys
        assert "FP8_MIXED" in keys

# add to TestFormatRecommendation
class TestFormatRecommendation:
    ...
    def test_fp8_is_always_ok_regardless_of_architecture(self):
        for arch in (ModelLumina2(), ModelFlux(), ModelSDXL()):
            level, msg = format_recommendation(arch, "FP8")
            assert level == "ok"
            assert msg
            level, msg = format_recommendation(arch, "FP8_MIXED")
            assert level == "ok"
            assert msg
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_safetensors_quant.py -k "fp8_is" -v`
Expected: FAIL — `"FP8"` not in `SAFETENSORS_DTYPE_CHOICES`; `format_recommendation(..., "FP8")` currently falls through `if base != "INT8": return "ok", ""` — passes the "ok" check but `msg` is empty, failing `assert msg`.

- [ ] **Step 3: Write minimal implementation**

In `safetensors_quant.py`, replace the choices list and its preceding comment, and add an FP8 branch to `format_recommendation()`:

```python
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
```

```python
    if base == "F16":
        return "ok", "F16 preserves full precision — safe for any architecture."

    if base == "FP8":
        return "ok", (
            "FP8 defaults to full-precision compute (`full_precision_matrix_mult`) "
            "— safe on any architecture, same mechanism as most circulating "
            "Civitai/HuggingFace FP8 checkpoints. No FP8 tensor-core speedup "
            "without an explicit opt-out (docs/issues_analysis.md #16)."
        )

    if base != "INT8":
        return "ok", ""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_safetensors_quant.py -v`
Expected: PASS (all tests, including the pre-existing `test_expected_keys_present`-style registry test — check it doesn't hardcode an exact set excluding FP8; if it does, update it to include `"FP8"`/`"FP8_MIXED"`)

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add safetensors_quant.py tests/test_safetensors_quant.py
git commit -m "feat: re-enable FP8/FP8_MIXED in Convert -> Safetensors dropdown

Now that convert_safetensors.py defaults FP8 output to
full_precision_matrix_mult=true (previous commit), FP8 is safe on any
architecture -- re-added to SAFETENSORS_DTYPE_CHOICES and
format_recommendation() now returns an unconditional 'ok' for it,
unlike INT8's per-architecture keys_hiprec-driven caution."
```

---

### Task 3: `model_support.py` — display names and `support_level()`

**Files:**
- Create: `model_support.py`
- Test: `tests/test_model_support.py`

**Interfaces:**
- Consumes: `models.architectures.arch_list`, each class's `.arch` and `.keys_hiprec` attributes.
- Produces: `MODEL_DISPLAY_NAMES: dict[str, str]`, `TABLE_FORMATS: list[tuple[str, str]]` (display label, format key), `SUPPORT_VERIFIED = "verified"`, `SUPPORT_CAUTION = "caution"`, `SUPPORT_UNKNOWN = "unknown"`, `SUPPORT_SYMBOL: dict[str, str]` (level -> single-char glyph), `support_level(arch_key: str, keys_hiprec_nonempty: bool, format_key: str) -> str`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model_support.py
"""Tests for model_support.py — the per-architecture format support matrix."""

from __future__ import annotations

from model_support import (
    MODEL_DISPLAY_NAMES,
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

    def test_fp8_always_verified(self):
        # Safe unconditionally once full_precision_matrix_mult defaults true
        # (Task 1) -- architecture-independent, unlike INT8.
        assert support_level("lumina2", True, "FP8") == SUPPORT_VERIFIED
        assert support_level("flux", True, "FP8_MIXED") == SUPPORT_VERIFIED

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

    def test_plain_int8_caution_for_every_sensitive_architecture(self):
        # Includes lumina2 itself -- plain INT8 is the confirmed-bad case there.
        assert support_level("lumina2", True, "INT8") == SUPPORT_CAUTION
        assert support_level("flux", True, "INT8") == SUPPORT_CAUTION

    def test_nvfp4_always_caution(self):
        # Same class of dynamic-activation-quantization risk as FP8, but with
        # no verified safe mode (no equivalent full_precision_matrix_mult
        # confirmed for NVFP4 yet) -- caution everywhere, not just lumina2,
        # since the mechanism is architecture-independent even though direct
        # evidence (marcorez8's tiered ratings) only exists for lumina2.
        assert support_level("lumina2", True, "NVFP4") == SUPPORT_CAUTION
        assert support_level("sdxl", False, "NVFP4_MIXED") == SUPPORT_CAUTION

    def test_unknown_format_key_returns_unknown(self):
        assert support_level("sdxl", False, "BOGUS") == SUPPORT_UNKNOWN
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_model_support.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'model_support'`

- [ ] **Step 3: Write minimal implementation**

```python
# model_support.py
"""Per-architecture quantization-format support matrix — the data model
behind the GUI's "Model Support" tab (gui.py) and the dynamic ⚠ annotation
of the GGUF/safetensors format dropdowns.

Two things live here, both editorial judgment calls documented inline
rather than hidden in a spreadsheet: which public model names map to each
internal architecture key (MODEL_DISPLAY_NAMES), and a tri-state confidence
level for each (architecture, format) combination (support_level()).
"""

from __future__ import annotations

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

# Architectures whose *_MIXED keys_hiprec protection scope has actually been
# confirmed by a full convert+load+render cycle in ComfyUI with this tool's
# own output, not just cross-referenced against a community tool's published
# blacklist. See docs/issues_analysis.md #15. Mirrors
# safetensors_quant._RENDER_VERIFIED_ARCHES.
_RENDER_VERIFIED_ARCHES = {"lumina2"}


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
    - FP8 / FP8_MIXED: always VERIFIED. Defaults to
      full_precision_matrix_mult=true (Task 1 of the plan this function was
      introduced in), which makes ComfyUI skip the quantized-compute branch
      entirely for every layer regardless of architecture — the safety
      mechanism itself is architecture-independent, unlike INT8's
      per-architecture keys_hiprec bet.
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
        return SUPPORT_VERIFIED
    if format_key in ("INT8", "INT8_MIXED"):
        if not keys_hiprec_nonempty:
            return SUPPORT_VERIFIED
        if arch_key in _RENDER_VERIFIED_ARCHES and format_key == "INT8_MIXED":
            return SUPPORT_VERIFIED
        return SUPPORT_CAUTION
    if format_key in ("NVFP4", "NVFP4_MIXED"):
        return SUPPORT_CAUTION
    return SUPPORT_UNKNOWN
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_model_support.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add model_support.py tests/test_model_support.py
git commit -m "feat: add model_support.py — per-architecture format support matrix

New single-source-of-truth module for the upcoming Model Support GUI tab:
MODEL_DISPLAY_NAMES maps internal architecture keys to public model
names, support_level() computes a tri-state (verified/caution/unknown)
confidence rating per (architecture, format) pair from facts already in
models/architectures.py (keys_hiprec presence) plus this project's own
render-testing history (lumina2 is the only architecture actually
convert+load+render tested end-to-end)."
```

---

### Task 4: `build_support_table()`

**Files:**
- Modify: `model_support.py`
- Test: `tests/test_model_support.py`

**Interfaces:**
- Consumes: `models.architectures.arch_list`, `MODEL_DISPLAY_NAMES`, `TABLE_FORMATS`, `support_level()` (Task 3)
- Produces: `build_support_table() -> list[dict]` — one dict per architecture: `{"arch": str, "display_name": str, <format_key>: <support level str>, ...}` for every `(_, format_key)` in `TABLE_FORMATS`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model_support.py — add
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_model_support.py::TestBuildSupportTable -v`
Expected: FAIL — `ImportError: cannot import name 'build_support_table'`

- [ ] **Step 3: Write minimal implementation**

Append to `model_support.py`:

```python
def build_support_table() -> list[dict]:
    """Return one row per models.architectures.arch_list entry: display
    name plus a support_level() result for every TABLE_FORMATS column."""
    from models.architectures import arch_list

    rows = []
    for cls in arch_list:
        instance = cls()
        sensitive = bool(instance.keys_hiprec)
        row = {
            "arch": instance.arch,
            "display_name": MODEL_DISPLAY_NAMES[instance.arch],
        }
        for _, format_key in TABLE_FORMATS:
            row[format_key] = support_level(instance.arch, sensitive, format_key)
        rows.append(row)
    return rows
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_model_support.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add model_support.py tests/test_model_support.py
git commit -m "feat: add build_support_table() to model_support.py

One row per models.architectures.arch_list entry, ready for the GUI
table (Task 6) to render directly without re-deriving per-architecture
sensitivity itself."
```

---

### Task 5: GGUF dropdown quality/size labels

**Files:**
- Modify: `quantize.py`
- Test: `tests/test_quantize.py`

**Interfaces:**
- Produces: `ALL_QUANT_CHOICES` labels each state an approximate "% smaller than F16" figure derived from the existing `SIZE_RATIOS` table, in addition to the quality wording already present.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_quantize.py — add
class TestAllQuantChoicesLabels:
    def test_every_lq_and_python_choice_states_size_savings_except_baseline(self):
        from quantize import ALL_QUANT_CHOICES
        for label, key in ALL_QUANT_CHOICES:
            if key in ("F16", "BF16"):
                continue  # baseline / same size as baseline — no % figure
            assert "smaller than F16" in label or "F16 size" in label, (
                f"{key!r} label missing a size-savings figure: {label!r}"
            )

    def test_q4_k_m_still_marked_recommended(self):
        from quantize import ALL_QUANT_CHOICES
        label = next(l for l, k in ALL_QUANT_CHOICES if k == "Q4_K_M")
        assert "recommended" in label
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_quantize.py::TestAllQuantChoicesLabels -v`
Expected: FAIL — current labels have no "smaller than F16"/"F16 size" text.

- [ ] **Step 3: Write minimal implementation**

Replace `ALL_QUANT_CHOICES` in `quantize.py` (values derived from the existing `SIZE_RATIOS` dict just above it: `1 - ratio`, rounded to the nearest percent; F32's `2.00` ratio is stated as "2× F16 size" instead of a negative percentage):

```python
# Ordered choices for the UI dropdown: (display label, key)
# Covers the types that are practical for ComfyUI-GGUF diffusion models.
# Size-savings percentages are derived from SIZE_RATIOS just above (1 -
# ratio, rounded to the nearest percent) so the two never drift apart.
ALL_QUANT_CHOICES: list[tuple[str, str]] = [
    ("F32  — Full precision · 2× F16 size",                                  "F32"),
    ("F16  — Half precision · standard",                                     "F16"),
    ("BF16 — Brain float 16 · same size as F16",                             "BF16"),
    ("Q8_0 — 8-bit · very high quality · 43% smaller than F16",              "Q8_0"),
    ("Q6_K — 6-bit · very high quality · 56% smaller than F16  [lq]",        "Q6_K"),
    ("Q5_K_M — 5-bit · high quality · 62% smaller than F16  [lq]",           "Q5_K_M"),
    ("Q4_K_M — 4-bit · recommended ★ · 67% smaller than F16  [lq]",          "Q4_K_M"),
    ("Q4_K_S — 4-bit small · 69% smaller than F16  [lq]",                    "Q4_K_S"),
    ("Q3_K_M — 3-bit · moderate quality · 73% smaller than F16  [lq]",       "Q3_K_M"),
    ("Q2_K  — 2-bit · smallest · 79% smaller than F16  [lq]",                "Q2_K"),
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_quantize.py -v`
Expected: PASS (all tests — also re-check any pre-existing test asserting exact label strings elsewhere in the file or in `tests/test_gui.py`; update any that hardcode the old label text)

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add quantize.py tests/test_quantize.py
git commit -m "feat: add size-savings percentages to GGUF quant dropdown labels

Derived directly from the existing SIZE_RATIOS table (1 - ratio, rounded
to the nearest percent) so the two can't drift apart. Quality wording
(already present) is unchanged."
```

---

### Task 6: "Model Support" GUI tab

**Files:**
- Modify: `gui.py`
- Test: `tests/test_gui.py`

**Interfaces:**
- Consumes: `model_support.build_support_table()`, `model_support.TABLE_FORMATS`, `model_support.SUPPORT_SYMBOL` (Tasks 3-4)
- Produces: a new tab in `gui.py`'s `build_app()`; `gui._support_table_cell_html(level: str) -> str`; `gui._support_table_rows_for_dataframe() -> list[list[str]]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gui.py — add
def test_model_support_tab_present():
    app = gui.build_app()
    label_texts = [
        getattr(b, "label", None)
        for b in app.blocks.values()
        if isinstance(b, gr.Dataframe)
    ]
    assert "Model Support" in label_texts


def test_support_table_rows_cover_every_architecture():
    from models.architectures import arch_list
    rows = gui._support_table_rows_for_dataframe()
    assert len(rows) == len(arch_list)


def test_support_table_cell_html_uses_the_right_symbol():
    from model_support import SUPPORT_CAUTION, SUPPORT_VERIFIED
    assert "✓" in gui._support_table_cell_html(SUPPORT_VERIFIED)
    assert "⚠" in gui._support_table_cell_html(SUPPORT_CAUTION)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_gui.py -k support_table -v`
Expected: FAIL — `AttributeError: module 'gui' has no attribute '_support_table_cell_html'` (and no "Model Support" tab yet)

- [ ] **Step 3: Write minimal implementation**

Add to `gui.py`, near `update_format_recommendation` (after its definition, before the `# CSS` section):

```python
from model_support import SUPPORT_SYMBOL, TABLE_FORMATS, build_support_table

_SUPPORT_CELL_COLOR = {
    "verified": "var(--s2g-accent)",
    "caution": "var(--s2g-warn)",
    "unknown": "var(--s2g-muted)",
}


def _support_table_cell_html(level: str) -> str:
    """Return the colored-symbol HTML for one Model Support table cell."""
    symbol = SUPPORT_SYMBOL[level]
    color = _SUPPORT_CELL_COLOR[level]
    return f'<span style="color:{color};font-weight:700;font-size:1.1em;">{symbol}</span>'


def _support_table_rows_for_dataframe() -> list[list[str]]:
    """Build the gr.Dataframe row data: [display_name_html, *format_cells]."""
    rows = []
    for row in build_support_table():
        display_html = (
            f'<span>{row["display_name"].split(" (")[0]}</span> '
            f'<span style="color:var(--s2g-muted);font-size:0.85em;">'
            f'({row["arch"]})</span>'
        )
        cells = [display_html]
        for _, format_key in TABLE_FORMATS:
            cells.append(_support_table_cell_html(row[format_key]))
        rows.append(cells)
    return rows


_SUPPORT_TABLE_LEGEND_HTML = """
<div style="font-size: var(--type-small); color: var(--s2g-muted); margin-top: 8px;">
  <strong style="color:var(--s2g-accent);">✓ Verified</strong> — actually converted, loaded, and rendered correctly in ComfyUI with this tool's own output.
  &nbsp;·&nbsp;
  <strong style="color:var(--s2g-warn);">⚠ Caution</strong> — technically supported by this tool, but not render-tested for this architecture, or a known correctness/quality issue.
  &nbsp;·&nbsp;
  <strong style="color:var(--s2g-muted);">? Unknown</strong> — this combination has never been attempted.
</div>
"""
```

Then in `build_app()`, add the new tab after the "Extract Components" tab (before the closing of `with gr.Tabs():`):

```python
            # ── Model Support ──────────────────────────────────────────────
            with gr.Tab("⊞  Model Support", id=6):
                gr.Markdown(
                    "Which quantization formats this tool supports for each "
                    "detectable architecture. Click a cell to jump to the "
                    "matching Convert tab with that format pre-selected.",
                    elem_classes=["intro"],
                )
                support_table = gr.Dataframe(
                    label="Model Support",
                    headers=["Model", *[label for label, _ in TABLE_FORMATS]],
                    datatype="html",
                    value=_support_table_rows_for_dataframe(),
                    interactive=False,
                    wrap=True,
                )
                gr.HTML(_SUPPORT_TABLE_LEGEND_HTML)
```

Give the other five existing tabs explicit `id=0` through `id=5` (in their current top-to-bottom order: Convert → GGUF, Convert → Safetensors, Convert Text Encoder, Fix Pad Tokens, Fix 5D Tensors, Extract Components), and capture the `gr.Tabs()` context manager into a variable — both needed for Task 7's tab-switching:

```python
        with gr.Tabs() as main_tabs:

            # ── Convert → GGUF ─────────────────────────────────────────────
            with gr.Tab("⬡  Convert → GGUF", id=0):
```

(apply `id=1` through `id=5` to the remaining five `gr.Tab(...)` calls in order, and `id=6` to the new Model Support tab already shown above)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_gui.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: all pass

- [ ] **Step 6: Manual visual check**

Launch the app (`uv run python gui.py` or the existing headless-launch pattern used earlier this session) and confirm in a browser: the new tab renders a 12-row table with colored ✓/⚠/? symbols, a legend below it, and doesn't break the existing tabs' layout.

- [ ] **Step 7: Commit**

```bash
git add gui.py tests/test_gui.py
git commit -m "feat: add Model Support tab

New read-only table (gr.Dataframe, datatype=html) showing every
architecture x format support level from model_support.py, colored
per the tri-state scheme (verified/caution/unknown), with a legend.
Gives all six tabs explicit ids and captures the gr.Tabs() context as
main_tabs, needed by the click-to-apply wiring in the next commit."
```

---

### Task 7: Click-to-apply from the support table

**Files:**
- Modify: `gui.py`
- Test: `tests/test_gui.py`

**Interfaces:**
- Consumes: `main_tabs`, `quant_dropdown` (GGUF tab), `st_format_dropdown` (Safetensors tab), `support_table` (Task 6)
- Produces: `gui.apply_support_table_selection(evt: gr.SelectData) -> tuple` wired to `support_table.select(...)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gui.py — add
def test_apply_support_table_selection_gguf_column():
    from unittest.mock import MagicMock
    evt = MagicMock()
    evt.index = (0, 1)  # row 0, first format column after Model = "GGUF"
    tabs_update, quant_update, st_format_update = gui.apply_support_table_selection(evt)
    assert tabs_update.get("selected") == 0
    assert quant_update.get("value") == "Q4_K_M"


def test_apply_support_table_selection_safetensors_column():
    from unittest.mock import MagicMock
    from model_support import TABLE_FORMATS

    # Find the column index for "INT8_MIXED" (Model column is index 0, so
    # +1 for TABLE_FORMATS' own 0-based position).
    col = 1 + next(i for i, (_, key) in enumerate(TABLE_FORMATS) if key == "INT8_MIXED")
    evt = MagicMock()
    evt.index = (0, col)
    tabs_update, quant_update, st_format_update = gui.apply_support_table_selection(evt)
    assert tabs_update.get("selected") == 1
    assert st_format_update.get("value") == "INT8_MIXED"


def test_apply_support_table_selection_model_column_is_a_noop():
    from unittest.mock import MagicMock
    evt = MagicMock()
    evt.index = (0, 0)  # the Model name column itself
    tabs_update, quant_update, st_format_update = gui.apply_support_table_selection(evt)
    assert "selected" not in tabs_update
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_gui.py -k apply_support_table -v`
Expected: FAIL — `AttributeError: module 'gui' has no attribute 'apply_support_table_selection'`

- [ ] **Step 3: Write minimal implementation**

Add to `gui.py`, right after `_support_table_rows_for_dataframe()`:

```python
def apply_support_table_selection(evt: gr.SelectData):
    """Handle a click on the Model Support table: switch to the matching
    Convert tab and pre-select that format, or no-op for the Model column.

    The GGUF column collapses every K-quant level into one cell (see
    model_support.TABLE_FORMATS's comment) — clicking it can't pick a
    specific quant level, so it defaults to Q4_K_M, this tool's own
    "recommended ★" choice, same as GGUF's own dropdown default.
    """
    row, col = evt.index
    if col == 0:
        return gr.update(), gr.update(), gr.update()

    format_key = TABLE_FORMATS[col - 1][1]
    if format_key == "GGUF":
        return gr.Tabs(selected=0), gr.update(value="Q4_K_M"), gr.update()
    return gr.Tabs(selected=1), gr.update(), gr.update(value=format_key)
```

Then, in `build_app()`, after `support_table = gr.Dataframe(...)` from Task 6, wire the event (this needs `quant_dropdown` and `st_format_dropdown`, which are defined earlier in `build_app()` — the event wiring itself can live at the bottom of `build_app()` alongside the other `# ── Events ──` wiring, not inline in the tab body, matching this file's existing convention):

```python
        support_table.select(
            apply_support_table_selection,
            outputs=[main_tabs, quant_dropdown, st_format_dropdown],
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_gui.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: all pass

- [ ] **Step 6: Manual visual check**

In a browser: click a ✓ cell under "INT8_MIXED" for any row — confirm it switches to the Convert → Safetensors tab with "INT8 mixed" selected in the Output format dropdown. Click a cell under "GGUF" — confirm it switches to Convert → GGUF with Q4_K_M selected.

- [ ] **Step 7: Commit**

```bash
git add gui.py tests/test_gui.py
git commit -m "feat: click-to-apply on the Model Support table

Clicking a format cell switches to the matching Convert tab and
pre-selects that format (GGUF column defaults to Q4_K_M, since it
represents every K-quant level collapsed into one cell). Clicking the
Model name column is a no-op."
```

---

### Task 8: Dynamic ⚠ annotation of the format dropdowns

**Files:**
- Modify: `gui.py`
- Test: `tests/test_gui.py`

**Interfaces:**
- Consumes: `model_support.support_level()`, `models.architectures.detect_arch`, `convert.load_state_dict` (already imported in `gui.py`)
- Produces: `gui.annotate_gguf_choices(src: str) -> gr.update`, `gui.annotate_safetensors_choices(src: str) -> gr.update`, wired to `src_path.change` / `st_src_path.change` in addition to the existing handlers already bound there.

- [ ] **Step 1: Write the failing test**

Flux has a non-empty `keys_hiprec` and is not `lumina2`, so both `INT8` and `INT8_MIXED` are `SUPPORT_CAUTION` for it (only `lumina2`'s `INT8_MIXED` is `SUPPORT_VERIFIED`) — both must carry the ⚠ prefix.

```python
# tests/test_gui.py — add
class TestDynamicDropdownAnnotation:
    def test_annotate_safetensors_choices_marks_caution_entries(self, tmp_path):
        import torch
        from safetensors.torch import save_file

        src = tmp_path / "model.safetensors"
        save_file(
            {
                "double_blocks.0.img_attn.proj.weight": torch.randn(8, 8),
                "img_in.weight": torch.randn(8, 8),
            },
            str(src),
        )
        update = gui.annotate_safetensors_choices(str(src))
        labels_by_key = {key: label for label, key in update["choices"]}
        assert labels_by_key["INT8"].startswith("⚠")
        assert labels_by_key["INT8_MIXED"].startswith("⚠")
        assert not labels_by_key["F16"].startswith("⚠")
        assert not labels_by_key["FP8"].startswith("⚠")

    def test_annotate_safetensors_choices_no_source_returns_unmodified(self):
        update = gui.annotate_safetensors_choices("")
        from safetensors_quant import SAFETENSORS_DTYPE_CHOICES
        assert update["choices"] == [tuple(c) for c in SAFETENSORS_DTYPE_CHOICES]

    def test_annotate_gguf_choices_no_source_returns_unmodified(self):
        update = gui.annotate_gguf_choices("")
        from quantize import ALL_QUANT_CHOICES
        assert update["choices"] == [tuple(c) for c in ALL_QUANT_CHOICES]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_gui.py::TestDynamicDropdownAnnotation -v`
Expected: FAIL — `AttributeError: module 'gui' has no attribute 'annotate_safetensors_choices'`

- [ ] **Step 3: Write minimal implementation**

Add to `gui.py`, near `update_format_recommendation`:

```python
def _detected_arch_or_none(src: str):
    src = (src or "").strip()
    if not src or not os.path.isfile(src):
        return None
    try:
        return detect_arch(load_state_dict(src))
    except Exception:
        return None


def annotate_safetensors_choices(src: str):
    """Return a gr.update(choices=...) for the safetensors format dropdown,
    prefixing a ⚠ to any format that's SUPPORT_CAUTION for the detected
    source's architecture. No-op (original choices, unmodified) when no
    architecture can be detected."""
    from model_support import SUPPORT_CAUTION, support_level
    from safetensors_quant import SAFETENSORS_DTYPE_CHOICES

    model_arch = _detected_arch_or_none(src)
    if model_arch is None:
        return gr.update(choices=[tuple(c) for c in SAFETENSORS_DTYPE_CHOICES])

    sensitive = bool(model_arch.keys_hiprec)
    choices = []
    for label, key in SAFETENSORS_DTYPE_CHOICES:
        if support_level(model_arch.arch, sensitive, key) == SUPPORT_CAUTION:
            label = f"⚠ {label}"
        choices.append((label, key))
    return gr.update(choices=choices)


def annotate_gguf_choices(src: str):
    """Same as annotate_safetensors_choices() for the GGUF quant dropdown.
    Every ALL_QUANT_CHOICES key maps to the table's single "GGUF" column
    (model_support.support_level always returns SUPPORT_VERIFIED for it
    today), so this currently never marks anything — implemented as a real
    call through the shared support_level() function rather than skipped
    entirely, so a future architecture-specific GGUF caveat has one place
    to add it without touching the dropdown-wiring code again."""
    from model_support import SUPPORT_CAUTION, support_level
    from quantize import ALL_QUANT_CHOICES

    model_arch = _detected_arch_or_none(src)
    if model_arch is None:
        return gr.update(choices=[tuple(c) for c in ALL_QUANT_CHOICES])

    sensitive = bool(model_arch.keys_hiprec)
    caution = support_level(model_arch.arch, sensitive, "GGUF") == SUPPORT_CAUTION
    choices = [
        (f"⚠ {label}" if caution else label, key)
        for label, key in ALL_QUANT_CHOICES
    ]
    return gr.update(choices=choices)
```

Then wire both, alongside the existing `src_path.change`/`st_src_path.change` bindings already in `build_app()`:

```python
        src_path.change(annotate_gguf_choices, inputs=src_path, outputs=quant_dropdown)
        st_src_path.change(annotate_safetensors_choices, inputs=st_src_path, outputs=st_format_dropdown)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_gui.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: all pass

- [ ] **Step 6: Manual visual check**

In a browser, Convert → Safetensors tab: select a Flux checkpoint as source — confirm "INT8" and "INT8 mixed" entries in the format dropdown now show a ⚠ prefix, while F16/F16_MIXED/FP8/FP8_MIXED don't. Confirm the entries are still selectable despite the marker (matches the user's explicit "still selectable, no one tells anyone what to do" requirement).

- [ ] **Step 7: Commit**

```bash
git add gui.py tests/test_gui.py
git commit -m "feat: dynamically mark risky dropdown entries with a warning icon

Once a source checkpoint's architecture is detected, the safetensors
format dropdown (and, structurally, the GGUF quant dropdown, though it
never fires today since every GGUF entry is currently SUPPORT_VERIFIED)
prefix any SUPPORT_CAUTION entry with a warning icon. Entries stay fully
selectable -- this only adds information, it never blocks a choice."
```

---

### Task 9: Documentation

**Files:**
- Modify: `README.md`, `docs/architecture.md`, `docs/issues_analysis.md`, `CHANGELOG.md`

- [ ] **Step 1: `docs/issues_analysis.md` — new entry #16**

Append a new numbered entry (following the file's existing format: symptom/investigation/conclusion/fix) documenting: the user's real-world observation that FP8 checkpoints work fine on Civitai, the `full_precision_matrix_mult` discovery (with the exact `comfy/ops.py`/`comfy/utils.py` code references gathered during planning), the fix, and re-enablement of FP8/FP8_MIXED. Explicitly flag NVFP4's equivalent investigation as an open, not-yet-done follow-up (referenced by `model_support.py`'s `support_level()` docstring).

- [ ] **Step 2: `README.md`**

- Add FP8/FP8_MIXED back to the safetensors format table (Task 2 changed `SAFETENSORS_DTYPE_CHOICES`), with a short "why FP8 is safe now" note pointing at issues_analysis.md #16 (replacing/updating the existing "why INT8 and not FP8/NVFP4/MXFP8" section's FP8-specific claims — NVFP4/MXFP8 stay in the "not offered" category, FP8 moves out of it).
- Add a short "Model Support tab" paragraph describing the new tab, the tri-state legend, and the click-to-apply behavior.

- [ ] **Step 3: `docs/architecture.md`**

Update the "Why INT8, not FP8/NVFP4/MXFP8" section (FP8 is no longer in that list) and add a short "Model Support Tab" section describing `model_support.py`'s role (data model) versus `gui.py`'s role (rendering + interaction), matching this file's existing style of documenting each module's responsibility.

- [ ] **Step 4: `CHANGELOG.md`**

Add two `[Unreleased]` entries under `### Add`/`### Fix` as appropriate: the FP8 `full_precision_matrix_mult` fix + re-enablement (Tasks 1-2), and the new Model Support tab + dynamic dropdown annotations (Tasks 3-8).

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: all pass (docs-only changes, but confirms nothing broke in the same commit)

- [ ] **Step 6: Commit**

```bash
git add README.md docs/architecture.md docs/issues_analysis.md CHANGELOG.md
git commit -m "docs: document the FP8 fix and the new Model Support tab

docs/issues_analysis.md #16: the full_precision_matrix_mult discovery
that explains why circulating Civitai FP8 checkpoints work fine while
this project's own FP8 output didn't, and the fix. README/architecture
docs updated to reflect FP8's re-enablement and the new tab; NVFP4's
equivalent investigation is flagged as open, not done here."
```

---

### Task 10: Final integration pass

**Files:** none (verification only)

- [ ] **Step 1: Full test suite**

Run: `uv run pytest -q`
Expected: all pass, no skips beyond pre-existing ones

- [ ] **Step 2: App builds cleanly**

Run: `uv run python -c "import gui; gui.build_app(); print('OK')"`
Expected: `OK`, no exceptions

- [ ] **Step 3: Full manual walkthrough in Chrome**

Launch the app and, in a real browser tab:
1. Model Support tab: 12 rows, correct symbols, legend visible, both light and dark mode readable.
2. Click a GGUF cell → lands on Convert → GGUF with Q4_K_M selected.
3. Click an INT8_MIXED cell for `lumina2` → lands on Convert → Safetensors with "INT8 mixed" selected, no ⚠ (verified).
4. Select a Flux `.safetensors` file as source on Convert → Safetensors → confirm INT8/INT8_MIXED dropdown entries gain a ⚠ prefix.
5. GGUF quant dropdown shows the new size-savings percentages on every entry.

- [ ] **Step 4: Commit** (only if Step 3 surfaced fixes; otherwise nothing to commit)

```bash
git add -A
git commit -m "fix: address issues found in Model Support tab manual walkthrough"
```
