# React workbench migration

The React/TypeScript/Vite frontend and FastAPI backend now cover diffusion
models, text encoders, extraction and GGUF repair. Workflows use existing algorithms
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

Hugging Face downloads and the interactive
support matrix remain in `uv run python gui.py`.
They are the next migration milestones, not removed functionality. INT4 ConvRot
is still excluded from the production format registry.

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

On Windows, all 457 Python tests and Ruff pass, and TypeScript/Vite build
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

With the API already running, replay the browser check with:

```bash
uv run --with playwright playwright install chromium
uv run --with playwright python scripts/validate_workbench.py
```

Screenshots and owned synthetic fixtures are stored in the ignored
`.pytest-workbench-browser-tmp` directory. Relative font sizes honor text scaling;
coarse pointers receive at least 44 px button/summary targets. This is focused
browser verification, not a complete screen-reader or assistive-technology audit.

## Next milestones

1. Hugging Face download workflow.
2. Interactive compatibility matrix and release packaging.
3. Linux runtime verification before replacing the classic launcher by default.

References: [Vite backend integration](https://vite.dev/guide/backend-integration.html),
[FastAPI CORS/origin model](https://fastapi.tiangolo.com/tutorial/cors/),
[React effect cleanup](https://react.dev/reference/react/useEffect).
