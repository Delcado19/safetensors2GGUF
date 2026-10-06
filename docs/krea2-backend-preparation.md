# Krea 2 molbal backend preparation

Prepared and implemented: 2026-10-06. The Workbench now dispatches Krea GGUF
to the pinned source through `krea_backend.py`. Small synthetic exports are tested;
no ComfyUI installation change or large-model conversion/render was performed.

## Secured backend

- Upstream: `https://github.com/molbal/ComfyUI-GGUF`.
- Pinned commit: `5a0a3ffa0e3eae5c6af8b0b981b660a24d5fbc04`, dated 2026-10-05.
- Complete tracked source: [source.tar.gz](../third_party/molbal/source.tar.gz),
  52 files / 916,243 compressed bytes, with no git submodules.
- SHA-256: `a5bca0e18ab3bed476bdb1d40e3ff12db211d539d7cdc43f68c114673a458558`.
- Apache-2.0 license/copyright material is preserved in the archive; an exact
  [license copy](../third_party/molbal/LICENSE) and
  [manifest](../third_party/molbal/source.json) are tracked alongside it.
- Offline integrity/path/entrypoint verification:
  `uv run python scripts/verify_vendor_sources.py`.

This source snapshot remains available from our own checkout if the original
repository disappears. It does not yet solve offline packaging of Python wheels,
ComfyUI/Kitchen or the other llama.cpp toolchains. No moving upstream HEAD should
be fetched by the runtime adapter.

## Inspected contract and limits

`tools/convert.py` contains `ModelKrea2`, `convert_file()` and
`convert_safetensors_streamed()`. The function accepts explicit `dst_path`,
`interact=False`, `overwrite`, `quant_type_name`, `streamed=True`, a
`progress_callback`, and optional LoRA/device/target-size parameters.
Its CLI accepts `--src`, `--dst`, `--quant-type` and `--streamed`; it does not
expose the function's overwrite parameter. Do not drive interactive prompts.

The converter's top-level imports and `lora.py` imports are Python/Torch/NumPy/
GGUF/safetensors utilities; ComfyUI core is not imported at those entrypoints.
This suggests a subprocess in our uv environment can handle the initial standard
formats. Imports and real small-file exports have now been exercised for every
allowlisted format in the application environment. Custom ConvRot
modes have additional Kitchen/backend requirements and remain outside this step.

Initial Krea capability allowlist:

| Display | Backend key |
| --- | --- |
| FP16 | `F16` |
| BF16 | `BF16` |
| Q8_0 | `Q8_0` |
| Q5_1 / Q5_0 | `Q5_1` / `Q5_0` |
| Q4_1 / Q4_0 | `Q4_1` / `Q4_0` |

The pinned source also exposes K-quant presets in its converter, while its README
does not endorse K-quants for diffusion inference. Therefore do not enable them
for Krea merely because they occur in `QUANT_TYPE_MAP`. Q8_CR/Q4_CR, target-size
mode and LoRA fusion are deferred until their separate loader/runtime validation.
Never silently substitute Q4_0 for an existing Q4_K_M request.

Molbal's Krea protection rules use anchored `^first.`, `^last.`, `^tproj.`,
`^tmlp.`, `^txtmlp.` and `^txtfusion.projector.` patterns; these differ from our
broader substring-based safetensors protection list. Do not equate their sizes or
quality policy. Until a backend-specific estimator is validated, show unavailable
estimates rather than reuse a mismatched planner.

## Implemented adapter contract

1. Verify the tracked archive before extracting into a fresh disposable
   directory using safe archive extraction. Keep upstream source unmodified and our
   adapter separate. Preserve provenance/license files with distributed builds.
2. Add a small isolated Python subprocess runner for the pinned `convert_file`
   entrypoint. This prevents its `sys.path`/top-level `lora` imports from colliding
   with our application. Set a concrete temporary GGUF destination, explicit
   backend key, `interact=False`, and streaming for safetensors sources.
3. Reuse `validate_output`, `atomic_output` and `iter_process_output`. The upstream
   writer opens its target directly; do not pass the user's final output to it.
   If using our precreated atomic scratch file, allow overwrite only for that
   scratch path. Publish only after a zero exit and a validated readable/nonempty
   Krea GGUF; preserve previous output on failure or cancellation.
4. Dispatch Krea GGUF jobs to this adapter after source architecture detection;
   keep existing diffusion K-quants and text-encoder paths intact. Expose only
   the allowlisted formats for the chosen backend. Reuse shared jobs, cancellation
   and output/log state; do not add a second scheduler or custom select control.
5. Communicate the required Krea-capable molbal loader and pinned backend version
   in the result/guide. The matrix handoff remains restricted until full-model
   render evidence exists; synthetic export is not image-quality evidence.

## Loader deployment and next tests

Krea GGUF requires a Krea-capable ComfyUI core plus the molbal loader.
City96 and molbal register overlapping loader node IDs. Do not install both
unchanged into the same running ComfyUI and assume independent behavior.
Use an isolated ComfyUI test profile for initial Krea validation; decide later
whether to retain a separate profile or package the needed loader with distinct
node IDs while preserving the existing city96 workflow path.

The automated synthetic adapter checks cover: explicit
format dispatch, tensor names/dtypes/shapes, source-alias/existing-output guards,
worker failure, malformed/empty output, cancellation, scalar-scaled FP8 values
and atomic publication. Fresh extraction avoids trusting a mutable code cache.
The GGUF reader is closed even when parsing fails, preserving Windows cleanup.

Next select a real installed
Krea checkpoint for baseline versus converted renders, identical seed/prompt/
encoder/VAE, including appropriate Raw/Turbo settings. Verify the exact molbal
loader revision and record output size/visual drift separately from successful
loading. Linux and LoRA/offload behavior require their own evidence.

## Current status

- Complete pinned source and license: **secured and tracked**.
- Offline verification and corruption regression: **implemented**.
- Workbench conversion adapter: **implemented**, seven formats exercised with
  small synthetic sources through the actual pinned subprocess. Classic native
  `convert.py` retains its city96 Krea guard; the shared GUI pipeline dispatches
  allowlisted Krea requests to molbal.
- Matrix GGUF cell: **Validation pending** / **molbal loader required**. Full-model
  runtime validation is still pending; the matrix does not claim verified renders.
- Full-model Krea conversion/render validation: **pending**, no large model downloaded.
- Mistral and ERNIE integration: **separate later work**.

## Workbench usage and restrictions

Select a Krea `.safetensors` source and press **Inspect & estimate**. Inspection
returns the model-specific allowlist without changing the requested precision.
If Q4_K_M was selected, its disabled placeholder asks for an explicit supported
choice; conversion remains disabled until one is chosen. Select FP16, BF16,
Q8_0, Q5_1, Q5_0, Q4_1 or Q4_0. These exports do not use llama-quantize.
Changing precision retains source-specific capabilities; choosing another source
falls back to the normal registry. API job submission independently detects the
source and rejects wrong-backend formats and inapplicable advanced settings.

Only floating-point (FP32/FP16/BF16 or scalar-scaled FP8) Safetensors sources
are enabled. Packed INT8, NVFP4, ConvRot and `.comfy_quant` sources are rejected
before writing rather than being interpreted as ordinary weights. The adapter
checks normalized names/shapes after export; GGUF architecture, readability,
nonempty payload and duplicate names are checked before atomic publication.
Source files and existing destinations survive failure or cancellation.
No source-size estimate is reused from the native converter.
