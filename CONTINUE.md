# Multi-model / vision-language integration

## FLUX.3 and existing-model tokenizer extension (2026-10-19)

Implementation resumed after the user's reboot pause. FLUX.3 uses a separate
complete Molbal b07c7e0 source archive; older adapters keep their frozen versions.
The interrupted full-model run stopped at 307/359 tensors without publishing a
GGUF. The resumed Q4_K_M run published 3,123,680,032 bytes on I: after complete
359-tensor payload readback. Matching Molbal loading and installed ComfyUI CPU
construction pass for all 359 logical shapes/parameters, with no missing or
unexpected parameters. See `data/flux3-q4km-native-validation-2026-10-19.json` and
`scripts/check_flux3_native.py`. No forward pass or render is qualified.
The existing-model tokenizer tool covers the installed Tekken, UMT5 SentencePiece
and shared native Qwen assets. Embedded repairs create a verified separate model;
runtime repairs require an explicit healthy resource directory, keep the old
directory as backup and roll back failed publication. No live resource repair was
needed: all 14 installed text encoders were checked, 12 valid and the two retained
raw ERNIE originals missing Tekken as expected. Their fixed copies are valid.
Tokenizer/API/support tests pass (88 tests), and 66 schematic existing diffusion
conversions still match native. The first full suite exposed a Windows mmap lock
in the new fixture's file replacement; cloning before rewriting fixes it, and
the targeted projection/application test passes. Final validation: 1125 tests pass
in 264.29 s; Ruff and frontend build pass. The visible uv Workbench is running;
live FLUX.3 inspection exposes the Molbal formats, a schematic API export succeeds,
and the existing Qwen tokenizer job reports valid. See
`data/tokenizer-flux3-app-validation-2026-10-19.json`. No installed weights or native
tokenizer assets were changed by validation. FLUX.3 render quality remains open.

## Artifact cleanup and I: workspace (2026-10-19)

The user requested cleanup and requires agent tests/conversion artifacts on I:.
`build` is now a junction to `I:\safetensors2GGUF\build`; reference tools remain
available at their historical logical paths and the native executable checksum
still verifies. Large generated models were removed; installed G: models and the
user's Pictures files were untouched. Reports/scripts were checksum-verified into
`I:\safetensors2GGUF\archive\2026-10-19`; old I: report folders were consolidated
there. See `data/test-artifact-cleanup-2026-10-19.json` for exact counts and bytes.
Some old temporary directory ACLs prevent deletion. Their remaining directories are
retained in ignored `.build-cleanup-recovery-2026-10-19` and `.pytest-tmp`; no
project-wide permission reset was attempted. Future ad-hoc tests use
`I:\safetensors2GGUF\tests`, conversions use `I:\safetensors2GGUF\conversions`, and
uv uses `UV_CACHE_DIR=I:\safetensors2GGUF\uv-cache`.
Next: expand tokenizer checks/repair for the existing model set, then integrate
FLUX.3 from a separate pinned current Molbal snapshot. The local FLUX.3 BF16 source
is present (359 tensors, 11,097,404,512 bytes); current detector does not support it.

## ERNIE app progress and visible terminal (2026-10-19)

Original-key ERNIE exports now forward per-tensor progress through the shared
application pipeline and report writing/readback phases. Initial zero-progress
preparation remains visibly active; job logs also reach the launch terminal.
The user explicitly requested a visible terminal. The idle Workbench was restarted
through uv in a visible PowerShell window; ComfyUI was not restarted.
The user's preceding app job completed successfully at
`G:\ComfyUI-Easy-Install\ComfyUI\models\unet\BigLoveEryn1_bf16-Q4_K_M.gguf`
(5,299,410,112 bytes). No new image render was started.
New progress-versus-no-progress export fixtures are byte-identical; 66 schematic
application conversions still match the pinned native reference. See
`data/compiled-diffusion-progress-requalification-2026-10-19.json`.
The complete progress gate passes 1100 tests in 254.26 s. The subsequent missing
loader installation exposed a selector-extension bug: companion nodes now use
GGUF-only folder aliases while preserving native loader filters. All 24 companion
tests pass after that follow-up. The two standalone files are installed under
`G:\ComfyUI-Easy-Install\ComfyUI\custom_nodes\comfyui_s2gguf`; an isolated installed
import finds the completed ERNIE GGUF. ComfyUI itself was not restarted: a restart
is required to register **Original Diffusion Model (S2GGUF)** in the live server.

## Tokenizer inspection and explicit repair (2026-10-19)

Workbench inspection and post-download results now expose missing/invalid Tekken
blobs for ERNIE/Ministral 3 3B and Mistral Small 3.2 24B. The user supplies the
exact repository's tokenizer.json or tekken.json; no network request or guessed
tokenizer is used. Repair publishes a separate checkpoint only after validating
token IDs/byte fallback and complete unchanged-weight payload readback. Existing
files cannot be overwritten. Structural checks do not prove tokenizer provenance,
generation quality, other families, or GGUF tokenizer compatibility.
Validation: 1097 tests pass in 263.98 s through uv; Ruff and frontend build pass.
Live API checks against both installed ponpoke originals report missing blobs;
both repaired copies report valid structures. The idle Workbench was restarted
through uv to activate the API; refresh the browser session before using it.
See `data/tokenizer-check-validation-2026-10-19.json`. No render was started.

## ERNIE complete-checkpoint construction passed (2026-10-19)

The user supplied the complete local
`G:\ComfyUI-Easy-Install\ComfyUI\models\diffusion_models\Ernie\BigLoveEryn1_bf16.safetensors`
checkpoint (409 BF16 tensors, 16,067,034,480 bytes). The earlier source blocker
below is historical; no download is needed. Complete Q4_K_M export/readback and
native CPU BF16 source/companion construction passed: all 409 source tensors
survive, the native 410-entry model states match in shape, both load without
missing/unexpected weights, and actual parameters use BF16. The GGUF is
5,299,410,112 bytes; export including internal readback/hash takes 77.144 s.
See `data/companion-ernie-full.json` for source/output SHA256 and precise scope.
The validation
runner now rejects native diffusion missing/unexpected-weight warnings instead
of checking only text warnings. The user's final commit/push/shutdown instruction
remains active after the final test/documentation gates and verified push.
The final ERNIE gate passes **1087 tests** in 267.50 s; Ruff, frontend build
and the documentation check pass. The user additionally asked for Safetensors
conversion. A separate INT8_MIXED output now passes normal native CPU Safetensors
loading with no missing/unexpected weights: 8,221,839,352 bytes, 252 INT8 weights,
157 protected tensors exactly preserved, all 409 source shapes retained (661
output entries including scales). Original source SHA256 is unchanged. See
`data/ernie-int8-mixed-safetensors-validation.json` for artifact path and hashes.
The GGUF checkpoint was committed/pushed as `1534d42`; the next checkpoint records
the additional Safetensors evidence and table notes. After its verified push,
the requested shutdown can proceed. No further conversion work remains in this
session; image/GPU inference is still not qualified.
Original models and the installed ComfyUI remain unchanged; no renders or
inference were started. All three requested complete-model construction points
are now covered at one representative Q4_K_M precision per family. GPU execution,
quality, offload and broader checkpoint/precision coverage remain open.

## Previous CLIP/VL checkpoint (2026-10-19)

The user requested complete CLIP-L/bigG, full-VL and ERNIE loader checks, regular
commits/pushes, and shutdown after completion. CLIP-L/bigG extracted from the full
RealVisXL SDXL checkpoint and the complete Huihui Qwen3-VL-4B source now pass
Q4_K_M export, full source-key/shape readback and actual native CPU model
construction through `S2GGUFTextEncoderLoader`. Counts are 196, 518 and 713
respectively; the VL export retains all 315 vision and 18 DeepStack tensors.
Only CLIP-L has a missing optional projection, identical to its native baseline.

The separately requested explicit text-only Q8_CR source conversion passes and
has its marker plus zero retained vision tensors. The installed flat-language
Qwen3-VL-8B repack exposed a real plain-Qwen3 routing collision; detection and
public-path regressions now fix it. The ordinary synthetic text pipeline was
requalified without modifying historical data. Support-table evidence now
distinguishes CPU construction from still-unqualified inference.

See `docs/companion-loader-validation.md` and the new `data/companion-*-full.json`
reports. At this preceding checkpoint, no complete ERNIE checkpoint had been
found, including a header-based local architecture scan. The later user-supplied
source resolved that blocker; see the completed ERNIE check above. No download
authorization is needed for the supplied local source.
No node installation files or original model files were changed, and no renders
were started. The final full suite passes **1082 tests** in 244.67 s; Ruff,
frontend build and the documentation gate pass. The source/converted-model
SHA256 hashes were recorded after construction with unchanged source stats.

## Resumed 2026-10-19

The user explicitly resumed implementation, authorized regular commits/pushes,
and requested shutdown after a completed larger feature. The paused snapshot
below is historical, not the current execution instruction.

The BF16/Q8 empty-buffer bug is fixed without dropping buffers. Support tests
now distinguish unqualified original-key companion routes from historical
stock-loader failures. Workbench encoder help no longer claims normal conversion
downloads or compiles tools. The focused export/support/API/provenance gate passes
264 tests. Ruff and frontend build pass.

New immutable reports requalify the changed sources: all 66 schematic diffusion
comparisons match the pinned native reference; the real one-layer Qwen3 export
matches all 14 historical tensor payloads and model metadata except the random
temporary-directory display name. Historical performance/full-model measurements
are retained, not rerun or broadened. See the dated requalification JSON files in
`data/`.

### Completed adapter feature and shutdown checkpoint

The extended Molbal worker now independently checks every emitted tensor's name,
count, type, shape and encoded payload checksum after writing and before atomic
publication. Architecture mismatches, duplicate/missing tensors and corrupt bytes
fail the job. Existing output survives failed conversions. This verifies emitted
bytes, not completeness/correctness of upstream source filtering.

All 102 combinations of six extended architectures and all 17 advertised formats
pass schematic export tests; seven readback-damage cases and an architecture
mismatch/overwrite regression pass. The complete final suite passes **1078 tests**
in 245.80 s. Ruff, frontend build and both documentation gates pass. See
`data/extended-adapter-validation-2026-10-19.json` and
`docs/extended-adapter-validation.md`. The fix checkpoint is committed/pushed as
`d09aad8`; the following commit contains this adapter feature and checkpoint.

No models were converted outside repository-local synthetic/smoke fixtures, no
ComfyUI installation files changed and no new renders were started. Original
model files remain unchanged. Full-checkpoint companion-loader construction and
inference (VL/CLIP/ERNIE), package installation, clean-machine and non-Windows
qualification remain open. Next work should use representative complete sources
and distinguish full-VL from explicit text-only paths before claiming runtime
support. Do not silently certify historical performance or render measurements.

The requested larger adapter-validation feature is complete. Save and push this
checkpoint before honoring the user's shutdown request; do not force-close apps
or discard unsaved work.

## Historical paused snapshot

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
