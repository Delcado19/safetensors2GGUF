# React workbench migration

The React/TypeScript/Vite frontend and FastAPI backend now cover diffusion
models, text encoders, extraction, GGUF repair and Hugging Face downloads. Workflows use existing algorithms
and atomic output writers.
The classic Gradio frontend remains available during migration.

## Run with uv

```bash
uv sync --dev --frozen
cd frontend
npm ci
npm run build
cd ..
uv run python web_api.py
```

Open http://127.0.0.1:8765. The production frontend is served by FastAPI;
Node/npm is only needed to build it. On Windows, `npm.cmd` avoids PowerShell
execution-policy restrictions without changing the policy.

For frontend development, keep the API running and run `npm run dev` in
`frontend`. Vite serves http://127.0.0.1:5173 and proxies `/api` to port 8765.
Use `--port` to change the production API port (update the Vite proxy when
changing the development API port).

## Current scope

- Local file/folder browser; models are never uploaded through the browser.
- GGUF and safetensors conversion with existing format and precision registries.
- On-demand architecture inspection and size estimates; no model loading on
  each keystroke. Changing inputs invalidates the previous estimate.
- Background conversion, progress, bounded technical logs, cooperative cancel,
  output path, and recent job history.
- Light/dark appearance, responsive layouts, native keyboard-accessible controls,
  native modal focus management, reduced motion/transparency and contrast support.
- Advanced executable, thread, intermediate-file and overwrite settings.

The classic Gradio UI remains available as an alternative. INT4 ConvRot is still
excluded from the production format registry.

## Format names and selection

The Workbench uses one shared display formatter for conversion/extraction selects,
compatibility headers/cells, output preview and job history. Numeric format names
are uppercase; lower-case filenames/loader identifiers are not rewritten.

| Workbench display | Internal key | Meaning |
| --- | --- | --- |
| FP16 | `F16`, encoder `F16_ST` | Half-precision float; GGUF calls the type F16 |
| FP32 / BF16 | `F32` / `BF16` | Floating-point GGUF outputs |
| FP8 (E4M3) | `FP8` | Scaled `float8_e4m3fn` safetensors |
| INT8 + ConvRot | `INT8` | Native `int8_tensorwise`, rotated on eligible linear layers |
| NVFP4 | `NVFP4` | NVIDIA FP4 with block scaling |
| … · mixed precision | Existing `*_MIXED` | Model-specific protected weights retain source precision |
| Q4_K_M, Q6_K, etc. | Unchanged GGUF keys | GGUF-specific quantization names, not interchangeable with NVFP4/INT4 |

Native option groups separate floating-point / Q8 / K-quant GGUF choices and
standard / mixed-precision safetensors policies. Options contain only short
names; the selected format's details and protection policy appear underneath,
linked with `aria-describedby`. Even standard policy keeps mandatory loader/
shape-compatible tensors. API format registries supply the allowed keys;
display changes do not alter writers, output suffixes, metadata or API payloads.
The classic Gradio UI retains its legacy descriptive choice labels.

“Mixed” is used by upstream publishers (for example Comfy-Org's
[`fp8mixed` files](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/tree/main/split_files/diffusion_models)
and [`nvfp4_mixed` file](https://huggingface.co/Comfy-Org/Ideogram-4/blob/main/diffusion_models/ideogram4_nvfp4_mixed.safetensors)),
but those names alone do not specify the same protected layers/scaling/runtime
recipe as this tool. Common numeric terms do not make every community suffix a
portable format contract. The native
[INT8 tensorwise/ConvRot format](https://github.com/Comfy-Org/comfy-quants/blob/main/docs/formats/int8_tensorwise.md)
also distinguishes INT8 storage from the optional rotation procedure.
Civitai was not directly accessible during this naming review; no exhaustive
cross-site naming survey is claimed.

The guide has a separate **INT4 + ConvRot — Prototype / not selectable** card.
It records the existing one-checkpoint SDXL render/LoRA/offload evidence and
pending version/model integration. It does not invent production dropdown
entries or support classifications for untested families. INT8 + ConvRot remains
the implemented selectable path. ConvRot itself is not a bit width.

## Interactive compatibility matrix

The Workbench now uses `build_workbench_support_tables()` to project shared
records onto explicit model/profile variants. Model titles omit Family; the
internal architecture/family identifier appears on a separate muted line.
Centered headers split format and policy: `FP16` / `mixed precision`, `INT8` /
`+ConvRot`, and `INT8 + ConvRot` / `mixed precision`. This is presentation only;
API conversion keys, writer metadata and file naming are unchanged.

Z-Image Base / Lumina 2.0 retains the recorded default-policy classifications.
The separate Z-Image Turbo row scopes evidence to `z_image_turbo`: NVFP4_MIXED
shows visible drift from the recorded two-prompt test; the other quantized
formats/GGUF remain untested for that scoped policy, while FP16 precision casts
use existing implementation reasoning. **Use format** transfers the Turbo profile
for safetensors; selecting GGUF resets to auto because these safetensors profiles
do not apply to the GGUF pipeline.

The Mistral table distinguishes the full 40-layer base encoder from the FLUX.2
30-layer pruned packaging. Only the pruned row carries the recorded quantized
safetensors render results; the full row does not inherit them. GGUF remains
blocked by the existing production family guard: full encoder integration and
pruned configuration/export handling are shown as pending, not as universal
upstream format failures. This presentation split does not implement the missing
model detection/configuration work.

Krea GGUF displays **Integration pending** / **molbal loader required**. Pending
cells use a clock icon, visible explanations, unique accessible descriptions
and a separate evidence filter. They are excluded from the unsupported filter;
their format-handoff button stays disabled. Internally their existing blocked
status remains intact. The classical tables/export guard contracts are unchanged.

The [unsupported-entry upstream audit](unsupported-matrix-audit.md) reviews all
negative cells as of 2026-10-06. Some grouped claims are stale or overbroad;
that review does not imply new runtime validation or silently unblock exports.

**Format guide** opens the shared project support matrix first, followed by the
format overview. Choose diffusion models or text encoders, search a public model
name/internal architecture, and optionally filter by evidence classification.
The filter retains rows with at least one matching cell; other cells remain
visible so alternatives can be compared. No matching rows show an explicit
empty state. Tables use native captions and row/column headers, sticky model
names, focusable horizontal scrolling and labeled cell buttons.

Select a cell to read its classification and available support reason in a
visible detail card. The underlying registry's four states remain unchanged: verified,
caution (visible drift), bad (unsupported/tooling gap or negative render evidence),
and unknown (untested). Reasons are reused from `model_support.py`; the UI adds no
new evidence. Verified can reflect render tests or implementation reasoning and
does not imply every release, quantization level or runtime configuration was
tested. The matrix omits legacy theoretical savings percentages; estimates
belong to source inspection. NVFP4 hardware requirements still apply.

Usable images with observed visible drift are **caution**, even if older samples
looked clean. The shared registry now surfaces the documented Qwen-Image-Edit-2511
NVFP4_MIXED/INT8_MIXED detail drift and Qwen3-8B GGUF Q5_K_M conditioning drift.
Reasons state checkpoint, precision policy, baseline and limited test scope.
Z-Image Turbo NVFP4_MIXED under `z_image_turbo` has a **Profile-dependent drift**
marker: its usable two-prompt result does not clear older default-policy family
failures or unlock the generic bad cell. AuraFlow facial detail and UMT5 blink
timing observations are also explicit. These updates reuse recorded evidence;
they are not new render results and do not change quantization/protection math.

Visible drift depends on the checkpoint, layer policy, inputs and sampling
workflow, not bit width alone. Do not extrapolate one model/seed to every family.
Diffusion quantization research identifies architecture/timestep effects
([Q-Diffusion](https://arxiv.org/abs/2302.04304)) and channel/sample/temporal
variation ([Q-DiT](https://arxiv.org/abs/2406.17343)). Those papers support the
general sensitivity explanation, not certification of our NVFP4 implementation.

**Use format** selects the model category and output container, clears the
previous destination/base override, applies the row's profile (auto for GGUF), and focuses the conversion model
type. It never starts conversion or identifies the source. The user must choose
a source matching the selected family. GGUF represents grouped precisions and
selects Q4_K_M; text-encoder F16 selects `F16_ST` safetensors; other safetensors
keys pass through unchanged. Known bad combinations disable this handoff, while
their explanations remain accessible. Unknown/caution selections retain their
existing classification, not an implied endorsement.

`GET /api/support` uses the existing session token/origin guard and returns
`diffusion`/`text_encoder` objects with format keys and scoped rows/reasons.
Rows have unique `id` values, source `arch`/`family`, optional `precision_profile`,
and optional cell `__label`/`__pending`/`__scope` presentation metadata.
The frontend loads it once per mounted guide, without model loading or network
requests to upstream model providers. Reopening the guide retries a failed load.
No new dependency or decorative table/navigation animation was added.

## Hugging Face downloads

Open **Hugging Face**, enter a model repository ID (`organization/model-name`)
and a revision (default `main`), then choose **Inspect repository**. Inspection
reads metadata only: checkpoint folders, safetensors file counts, total shard
bytes when available, and the resolved commit. No tensors download at this step.
Changing the repository or revision invalidates the plan; stale replies cannot
replace it. When several folders contain checkpoints, choose one explicitly.
The empty folder is the repository root, distinct from an unspecified selection.

Choose a local **Download folder** and start **Download checkpoint**. Only the
selected folder's safetensors files download and merge. The merged name is the
folder's last segment, or the repository name for root weights. The final file
is published atomically, and replacement requires the overwrite checkbox.
The existing streaming merger keeps one tensor in memory at a time, preserves
safetensors metadata, and rejects duplicate tensor keys or conflicting metadata.
The UI pins downloads to the inspected commit; API downloads resolving a branch
or tag also pin that revision before downloading shards.

The download and merge need space for both staged shards and the merged output.
Active downloads show the real transfer/merge phase with shard counts and
**Working**, without pretending shard counts are byte percentages. Cancellation
is cooperative: it waits for the current Hub transfer, then checks between shards,
during merge and before publication. Errors/cancellation leave partial shards in
`.hf_download_<repo-id-with-underscores>/` for a retry; success removes the staging
directory. A retry resumes where the Hub's local download metadata permits it.
It is not persisted job recovery; restarting the server clears job history.

Authentication stays server-side: `HF_TOKEN` takes precedence, then
`CODEX_HUGGINGFACE_API_KEY`, otherwise the Hub's stored login (for example
`uv run hf auth login`). The UI exposes only a credential-availability boolean;
it does not accept or return tokens. Gated repositories may require accepting
the model's conditions on Hugging Face. Transport/access errors return controlled
messages without response headers or signed URLs. No environment credentials are
written to configuration or project files.

Protected endpoints:

- `POST /api/hf/inspect`: `source` (repo ID), optional `revision`; returns repo ID,
  resolved revision and folder groups (`subfolder`, `files`, `bytes`).
- `POST /api/hf/download`: same source/revision, required `destination` directory,
  optional `subfolder` (`null` auto-selects only an unambiguous repository; `""`
  selects root explicitly) and `overwrite`; returns a 202 shared job snapshot.
- `/api/jobs` history/state/cancel applies to downloads too. Conversion, extraction,
  repair and download all share the same one-worker limit.

This feature downloads single-checkpoint safetensors weights, not a complete Hub
repository, tokenizer or Diffusers pipeline. GGUF files are not downloaded by this
merger. Multiple complete checkpoints in one folder are unsupported; duplicate
keys are rejected. A successful download is not a new ComfyUI compatibility or
render guarantee; choose an architecture/loader supported by your workflow.

The integration follows the official [Hub API](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api)
and [file-download API](https://huggingface.co/docs/huggingface_hub/package_reference/file_download).
It uses the already installed Python library; no new runtime dependency was added.

## Extraction and repair

Open **Extract & repair**, choose an operation, and select a local source:

| Operation | Source | Destination / behavior |
| --- | --- | --- |
| Extract components | SDXL safetensors checkpoint | Selected VAE to `vae/`, CLIP-L/G to `clip/` under the models root |
| Compare components | SDXL safetensors checkpoint | Read-only comparison against `vae/sdxlVAE.safetensors`, `clip/clip_l.safetensors`, `clip/clip_g.safetensors` |
| Extract diffusion model | Safetensors checkpoint | Existing conversion filters diffusion keys and writes to `diffusion_models/` |
| Repair pad tokens | Older Lumina2 GGUF | Reshape 1D `x_pad_token` / `cap_pad_token` to `[1, D]` in a separate GGUF |
| Restore 5D tensors | Quantized GGUF | Insert original sidecar tensors as F32 in a separate GGUF |

The models root defaults to the nearest ancestor named `models`, otherwise the
checkpoint's directory. Repair defaults to `<source-stem>-fixed.gguf`; a supplied
output folder uses the same filename. 5D restoration accepts an optional explicit
safetensors sidecar, otherwise uses the existing GGUF-metadata/legacy fallback.
Diffusion GGUF K-quants require the detected llama-quantize executable; set up the
converter through the existing installation guide if it is unavailable.

`POST /api/tools` accepts `operation`, `source`, `destination`, `overwrite`,
`components` (`vae`, `clip_l`, `clip_g`), optional `sidecar`, and diffusion
`container` / `format`. It returns the same 202 job snapshots as `/api/jobs` and
uses the same token/origin checks and one-worker limit. Comparison results are
structured under `result`; component paths appear in `outputs` as they publish.
Component export/comparison has no backend percentage callback, so active jobs
show **Working**. Comparison is local reference evidence, not image-quality proof.

All selected component destinations are preflighted before export. Each component
is written to a sibling temporary file and published atomically. Cancellation or
failure preserves the source and existing uncommitted outputs; previously
published components remain and are shown in the job result. The set of components
is deliberately not one transaction. Cancellation is checked between tensors
and before publication; loading/comparing/writing one large tensor may delay it.
Repair reuses the existing GGUF writers and their cooperative cancellation.

## Text-encoder workflow

Select **Model type ? Text encoder**, then choose GGUF or safetensors. Format
choices come from `TEXT_ENCODER_FORMAT_CHOICES`, not the diffusion registry.
F16 safetensors (`F16_ST` in the API) casts every tensor, including BF16 norms.
Native safetensors outputs retain genuine `model.*` keys and protect embedding,
position/projection and relative-attention tables using the existing encoder
architecture contract. Load them with native ComfyUI `CLIPLoader`; GGUF uses
`CLIPLoaderGGUF`.

GGUF auto-detects bundled base families. An optional original-base repo ID in
Advanced supplies tokenizer/config files for other families. The first run may
clone llama.cpp; K-quants use the existing plain llama-quantize builder (CMake
and C++ compiler required). The diffusion executable/settings do not apply.
Known incompatible combinations (including CLIP GGUF) are rejected before
external work. Known manual base overrides must agree with detected weights.
Unverified combinations are labeled as such; passing file conversion does not
establish ComfyUI render quality.

Safetensors estimates preserve encoder prefixes and protected tables. Encoder
GGUF size estimates are unavailable: the diffusion writer's assumptions do not
match llama.cpp. Encoder backends expose logs but no percentage callback, so
running jobs show **Working**, with an indeterminate accessible progress bar,
and completion shows 100%. Progress has no simulated timers or animations.

## Runtime and access contract

This is a local single-user workbench. Bind only to loopback. A random session
token is obtained from the same-origin `/api/config` endpoint and required for
filesystem, inspection, and job requests. Host validation and origin checks
reject unrelated websites. This is not a remote deployment/authentication model.

One conversion runs at a time to bound model RAM use. A second request returns
HTTP 409. Each job owns its cancellation event; stopping a job cooperatively
reaches the converter. Successful atomic publication wins a late cancellation,
so an existing result is never mislabeled as cancelled.

History contains the latest 20 jobs and 400 log lines per job, in process memory.
Restarting clears history and signals active jobs to cancel. It does not delete
successful output files. Keep the server running until a job is finished or
cancelled; forced process termination is not a graceful cancellation.

Estimates are payload estimates, not promises of final file size. GGUF's
existing estimator has format/architecture limitations; safetensors uses the
source-aware shared planning logic. Low or negative savings are shown explicitly.

## Design and motion

The design uses a charcoal workspace, a restrained pale-green action accent,
compact sidebar, numbered task sections and a separate output/status inspector.
The light theme uses warm neutral surfaces. Icons use one consistent Lucide set;
fonts use the local system stack without CDN requests.

Emil design engineering and Apple design/HIG principles guide hierarchy,
immediate feedback, progressive disclosure and accessibility. Animation
opportunity review accepted pointer press feedback (100 ms press / 160 ms
release, `cubic-bezier(0.23, 1, 0.32, 1)`, scale 0.97). Navigation, keyboard
actions, file rows, logs and numeric progress reject decorative motion because
users need stable, immediately readable information. Reduced motion removes
press movement; keyboard focus never scales. No motion dependency is needed.

## Validation

```bash
uv run python -m pytest --tb=short -q --basetemp .pytest-web-tmp -p no:cacheprovider
uv run python -m ruff check .
cd frontend
npm run build
```

The module invocation avoids the Windows uv script-trampoline error seen on
this machine. API tests exercise actual small GGUF/safetensors conversions,
inspection, overwrite/alias/format guards, token/origin/host checks, failed jobs
and concurrent-job/cancellation behavior. Synthetic models verify orchestration
and file writing; they do not constitute new ComfyUI image-quality evidence.

The stack supports Windows and Linux. This milestone is developed and runtime
tested on Windows; Linux execution, external llama-quantize availability and
GPU dependency installation still need testing on a Linux machine. No Linux
runtime success is claimed from cross-platform source code alone.

### Verified milestone results

On Windows, all 468 Python tests and Ruff pass, and TypeScript/Vite build
successfully. Chromium verifies a real synthetic FP8_MIXED conversion, file
selection, activity, format guide, both themes, 1440/768/390 px viewports,
Escape/focus restoration, reduced-motion mode and actual 200% input text scaling.
The second milestone also verifies an actual synthetic text-encoder F16
safetensors conversion through Chromium, plus FP8_MIXED/F16_ST writer behavior
through the API (original prefixes and embedding values). GGUF API dispatch and
cancellation are tested with a mocked encoder worker; existing encoder tests
cover its converter contract. No new end-to-end llama.cpp conversion or ComfyUI
render was performed for this UI migration. No browser page errors were observed.
CI also builds the frontend with Node 22.

The extraction/repair milestone additionally verifies real synthetic component
exports and reference comparison, diffusion extraction to a newly created models
root, pad-token shape/value repair, and 5D sidecar insertion through the API.
Regression checks cover cross-tool concurrency, cancellation, preflight conflicts,
source aliases and atomic component publication. Chromium exercises component
export/comparison and both GGUF repairs, including file browsing and mobile layout.
These are orchestration/file-format checks; no new ComfyUI render is claimed.

The Hugging Face milestone additionally verifies variant selection, pinned revision
propagation, server-only credentials, metadata preservation/conflict rejection,
overwrite/path guards, cross-tool concurrency and cancellation through API/unit
checks. The opt-in Chromium check actually downloaded and merged the public
`hf-internal-testing/tiny-random-bert` root checkpoint (520,212 download bytes),
validated its readable tensors and successful staging cleanup, and checked plan
invalidation, folder picking and desktop/mobile layouts without page errors.

The matrix milestone verifies the scoped Workbench projection of shared registries and
token/origin protection. Chromium checks search/empty states, category/evidence
filters, unsupported CLIP GGUF explanations and disabled handoff, keyboard F16_ST
handoff/focus, grouped GGUF Q4_K_M and diffusion INT8_MIXED mappings, both themes
and 1440/768/390 px layouts. These are UI/data-contract checks, not new renders.

The compact variant-table update also verifies centered two-line headers,
separate model-code lines, unique variant IDs/accessible descriptions, pending
filter separation and disabled handoffs, full/pruned Mistral evidence separation,
and automatic `z_image_turbo` profile selection. No new model render is claimed.

With the API already running, replay the browser check with:

```bash
uv run --with playwright playwright install chromium
uv run --with playwright python scripts/validate_workbench.py
# Optional network check: downloads a public 520 KB test checkpoint.
uv run --with playwright python scripts/validate_workbench.py --hf-live
```

Screenshots and owned synthetic fixtures are stored in the ignored
`.pytest-workbench-browser-tmp` directory. Relative font sizes honor text scaling;
coarse pointers receive at least 44 px button/summary targets. This is focused
browser verification, not a complete screen-reader or assistive-technology audit.

## Next milestones

1. Accessibility audit, release packaging and launcher integration.
2. Linux runtime verification before replacing the classic launcher by default.

References: [Vite backend integration](https://vite.dev/guide/backend-integration.html),
[FastAPI CORS/origin model](https://fastapi.tiangolo.com/tutorial/cors/),
[React effect cleanup](https://react.dev/reference/react/useEffect).
