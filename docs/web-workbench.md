# React workbench migration

The first migration milestone adds a local React/TypeScript/Vite frontend and
FastAPI backend. Both diffusion conversion workflows (GGUF and quantized
safetensors) use the existing conversion algorithms and atomic output writers.
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

Text-encoder conversion, component extraction, repair tools, Hugging Face
downloads, and the interactive support matrix remain in `uv run python gui.py`.
They are the next migration milestones, not removed functionality. INT4 ConvRot
is still excluded from the production format registry.

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

On Windows, all 447 Python tests and Ruff pass, and TypeScript/Vite build
successfully. Chromium verifies a real synthetic FP8_MIXED conversion, file
selection, activity, format guide, both themes, 1440/768/390 px viewports,
Escape/focus restoration, reduced-motion mode and actual 200% input text scaling.
No browser page errors were observed. CI also builds the frontend with Node 22.

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

1. Text-encoder conversion with its distinct prefix and model-family contracts.
2. Extraction, repair and Hugging Face workflows with explicit job adapters.
3. Interactive compatibility matrix and release packaging.
4. Linux runtime verification before replacing the classic launcher by default.

References: [Vite backend integration](https://vite.dev/guide/backend-integration.html),
[FastAPI CORS/origin model](https://fastapi.tiangolo.com/tutorial/cors/),
[React effect cleanup](https://react.dev/reference/react/useEffect).
