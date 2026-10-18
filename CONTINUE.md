# Paused multi-model / vision-language integration

Status: **paused, unfinished WIP snapshot** on 2026-10-18.
The user requested an immediate pause, then documentation and a push of the
current state. That authorizes saving this intermediate branch state; it does
not resume implementation, qualify the new backends, or authorize a release.
Do not treat this snapshot as ready to use. Resume implementation only when the
user requests it. The last preceding committed baseline is `8640e2f`.

## Scope and files

- `compiled_diffusion.py`: shared shipped CPU K writer for twelve image families.
  The earlier 66 schematic application comparisons cover eleven native-supported
  families. Qwen Image has fixture coverage without an old native equivalent.
  These reports predate later routing/protection changes and their source hashes
  no longer qualify the complete current tree.
- `bundled_tools.py`, `text_encoder_convert.py`, `third_party/text-tools`:
  pinned local text converter sources and a separate Windows x86_64 unpatched
  quantizer with static MSVC runtime. Eighteen schematic Llama/T5/Qwen3 comparisons
  and the earlier synthetic one-layer Qwen3 export succeeded. They do not prove
  complete Qwen3-4B/8B or vision-language inference.
- `extended_backend.py`, `models/architectures.py`: work in progress for Ideogram,
  Qwen Image 2.1, LTXV Upscaler, MiniMax H3 and its VAE, MiniMax Music 3, and Krea
  K/ConvRot formats. Uses the complete frozen Molbal archive at `b77b9d8`, isolated
  in a CPU subprocess. The known-clipping upstream NumPy K encoder is replaced by
  the verified bundled block kernel. Q2/Q3 policy extensions are project policy,
  not independently established native presets for these architectures.
- `original_text_gguf.py`: new original-key GGUF layout for CLIP-L/bigG, full
  Qwen2.5-VL/Qwen3-VL, MiniMax Music text, and ERNIE diffusion. Stores original
  shapes and non-floating buffers; it is not a stock city96/llama.cpp layout.
- `comfyui_s2gguf/`: experimental distinct `S2GGUFTextEncoderLoader` and
  `S2GGUFDiffusionLoader` nodes for that layout. They decode weights into CPU RAM
  before native ComfyUI model construction. They are **not low-bit inference
  kernels** and do not establish lower inference RAM/VRAM requirements.
- `web_api.py`, `conversion_service.py`, `frontend/src/main.tsx`, `model_support.py`:
  partially updated dispatch, format choices, source detection and support status.
  Several tests and UI/docs still describe the earlier contracts.

## Vision-language distinction

The pinned llama.cpp sources contain Qwen2.5-VL/Qwen3-VL text and vision/projector
conversion, including DeepStack mapping. Its separate multimodal projector output
is not directly consumable by the existing ComfyUI-GGUF text loader. Historical
Qwen2.5-VL language-only exports lost the vision tower and failed image conditioning.

The new original-key route aims to retain every language and vision tensor in one
GGUF for the new companion loader. Tiny fixtures checked shape/buffer preservation
and a mocked native ComfyUI call, not complete-model inference.

The frozen Molbal Qwen3-VL path is different: it accepts the official BF16 4B
layout and deliberately extracts a **text-only Q8_CR** encoder. The proposed API
format is `Q8_CR_TEXT_ONLY`; it must never be presented as a complete VL export.
Full-checkpoint tests of this special path have not been run.

## Validation evidence and known failures

Historical checkpoints, each with its own scope:

- Earlier multi-family work: 947 tests passed, Ruff and frontend build passed.
  An additional recorded-export check was then run separately. This predates the
  extended/VL work and is **not** the current suite result.
- Initial new adapter/original-layout tests: 56 passed in 80.33 s, before adding
  integer-scalar and empty-buffer cases and later protection/routing edits.
- Initial complete extended suite: 990 passed / 14 failed in 148.40 s. Includes
  expected stale provenance checks, old support/format assertions, and test inputs
  that needed valid safetensors headers after new source preflight.
- Last targeted pre-pause suite: 129 passed / 6 failed in 7.69 s. One substantive
  failure is BF16 encoding of an empty float tensor; the other five concern old
  support-table expectations. See the follow-up snapshot result below when present.
- Frontend TypeScript/Vite build passed after the first extended routing edits.
  Ruff passed before the pause. A plain import of the two companion node classes
  succeeded under the installed ComfyUI Python with an explicit project path.
  The installed ComfyUI files were not modified and the nodes were not deployed.

Known work remaining (not an exhaustive review):

1. Fix BF16 empty-buffer handling in the original-key writer without dropping the
   source buffer. Preserve integer scalar bytes and complete vision tensor sets.
2. Reconcile tests and support descriptions with the new original-key backend;
   retain historical stock-loader failures without presenting them as validation
   of the new loader. Do not change expectations merely to hide real regressions.
3. Requalify changed sources and update provenance through new measured reports.
   Preserve historical report hashes; do not rewrite them or mark unmeasured code
   as previously qualified. Several current provenance tests intentionally fail.
4. Finish source/format validation and frontend wording. In particular the encoder
   help still mentions first-use tool downloads/compiler requirements and
   safetensors-only CLIP; this is stale for the proposed new routes.
5. Review full VL versus explicit text-only dispatch, supported variants, source
   namespace handling and the official-layout restrictions before enabling them
   as qualified behavior. No normal model conversion should compile or fetch tools.
6. Finish companion-node packaging/docs and test actual native model construction
   with complete checkpoints. No new ComfyUI renders were started; do not require
   one render for every quantization level. Use representative cases by family.
7. Complete code review, numerical/format qualification, cancellation/overwrite
   regression checks, clean-machine and non-Windows distribution work. No blanket
   claim that every requested model/precision is implemented and usable is justified.

## Commands

Global `uv` is installed. In this sandbox use a repository-local cache and invoke
pytest through Python (the old venv pytest launcher has a path issue):

```powershell
uv --cache-dir build/uv-cache run --offline --frozen --no-sync python -m pytest --tb=short -q --basetemp .pytest-check-tmp -p no:cacheprovider
uv --cache-dir build/uv-cache run --offline --frozen --no-sync ruff check .
# In frontend:
npm run build
```

User models remain unchanged. Earlier user-authorized cleanup on `I:` is recorded
separately; do not repeat it or remove newly generated artifacts without approval.
No push to master, release, PR, ComfyUI installation change, or new render is part
of this documentation/snapshot request.

## Paused snapshot validation (2026-10-18)

The complete current suite finished with **992 passed and 12 failed** in 154.06 s.
Ruff and the frontend TypeScript/Vite build passed. Subsequent edits only updated
documentation and public method docstrings; implementation remains paused.

Failing tests (retained for the next implementation session):

- `tests/test_bundled_tools.py::test_real_export_smoke_records_current_sources`
- `tests/test_compiled_diffusion.py::test_recorded_diffusion_parity_and_current_sources`
- `tests/test_compiled_flux_writer.py::test_complete_direct_q2_run_reproduces_previous_model_bytes`
- `tests/test_compiled_kernel_package.py::test_current_kernel_levels_record_independent_native_file_parity`
- `tests/test_compiled_kernel_package.py::test_parallel_writer_evidence_retains_native_equality_and_current_sources`
- `tests/test_compiled_kernel_package.py::test_direct_spool_evidence_preserves_bytes_and_reduces_logical_writes`
- `tests/test_model_support.py::TestSupportLevel::test_gguf_bad_for_safetensors_only_new_architectures`
- `tests/test_model_support.py::TestSupportLevel::test_te_gguf_reason_matches_confirmed_bad`
- `tests/test_model_support.py::TestTextEncoderSupport::test_qwen25vl_gguf_confirmed_bad`
- `tests/test_model_support.py::TestTextEncoderSupport::test_clip_families_gguf_confirmed_bad`
- `tests/test_original_text_gguf.py::test_complete_vl_export_keeps_vision_and_original_shapes[BF16]`
- `tests/test_web_api.py::test_support_matrix_uses_shared_classifications_and_reasons`
