# Agents & Automation

## Persistent project TODO — load at every session start

**Paused WIP checkpoint (2026-10-18): read `CONTINUE.md` first.** The user stopped
implementation immediately and then requested documentation/push of the current
state. Do not resume implementation automatically. Extended model/VL adapters and
the new original-key companion loaders are unfinished, with known test failures
and stale provenance assertions. Later source changes supersede current-tree
qualification claims in the chronological notes below. No release or complete
VL/CLIP/ERNIE inference qualification is claimed by this snapshot.

Historical multi-model checkpoint (2026-10-18): Workbench/API `compiled_diffusion` extends
the shipped CPU K writer to FLUX, SD1/SDXL, SD3, AuraFlow, HiDream, Cosmos/Anima,
LTXV, HunyuanVideo, Wan, Lumina/Z-Image and Qwen Image. Eleven families match native
on 66 schematic application conversions; Qwen has adapter tests but no old native
reference. Video 5D floats, leading unit axes and Lumina pads are preserved directly.
Legacy FLUX identifiers remain FLUX-only. At that checkpoint Krea kept its adapter
and ERNIE its safetensors-only guard; later unfinished extensions supersede those
restrictions in the paused tree (see `CONTINUE.md`). Text conversion uses the shipped pinned source archive
and static-runtime unpatched Windows CPU executable, without normal-run cloning
or compilation. Eighteen text quantizer fixtures match the pinned previous tool.
An actual synthetic one-layer Qwen3 source also passes the complete pinned
export/Q4_K_M pipeline (14 tensors); see `data/bundled-text-export-smoke.json`.
Additional full-checkpoint inference, clean-machine and non-Windows qualification
remain open. See `docs/compiled-kernel-package.md`,
`data/compiled-diffusion-validation.json` and
`data/bundled-text-quantizer-validation.json`. Native remains the default.

Export qualification (2026-10-18): a fresh Git archive of `8640e2f` passes 55
package/direct-writer tests and 725 kernel goldens with build tools absent from
PATH and process/network audit guards active after OS detection and imports.
Existing host Python dependencies were reused; clean-machine/installer validation
remains open. See `data/compiled-kernel-export-qualification.json`.

Integration update (2026-10-18): Workbench/API `compiled_flux` uses the shipped
Windows x86_64 CPU package for FLUX Q2_K, Q3_K_M, Q4_K_S/M, Q5_K_M and Q6_K,
without runtime compilation. All six K presets match native across 48 schematic
files (1,192 tensor payloads); 725 block goldens pass. Q2/Q3 API fixtures match
native; the Windows C runtime is statically linked. Original four DLL API
fixtures match NumPy. This supersedes older Workbench/binary pending statements
below for this backend/platform only. Native remains the default. Broader
Q2 quality, full-model Q3, other platforms and clean-machine application qualification remain
open. See `docs/compiled-kernel-package.md`.

Full-model conversion update (2026-10-18): FLUX.2 klein Base 9B now completes
Q2_K through the integrated `compiled_flux` pipeline. The 3,356,285,248-byte GGUF
contains 115 Q2_K and 86 protected F32 tensors; all 201 passed internal readback.
Pipeline time was 662.291 s including F16 preparation; source hashing/additional
output hashing are separate. Original source size/mtime are unchanged and the
intermediate is cleaned up. Output/logs are on `I:\safetensors2GGUF-flux2-q2-bundled-2026-10-18`;
see `data/flux2-q2-bundled-conversion.json`. Full-model native parity and ComfyUI
loading/rendering are not yet tested; do not infer image quality from conversion.

Q2 runtime update (2026-10-18): A user-supplied original/Q2 PNG pair supports
ComfyUI inference through UnetLoaderGGUF with the exact converted GGUF (installed
SHA256 matches). Seed 572241167616859, 34 steps, er_sde, CFG 4, 768x1024 and
prompts/VAE/text encoder match; both LoRAs are disabled. Visible drift affects
pose/leg placement, face and car composition; Q2 looks less natural in this pair,
without gross image collapse. See `data/flux-q2-user-render-comparison.json`.
This supersedes the Q2 loading/rendering-untested note above for this setup only.
Native Q2 render parity, broader quality, LoRA and offload remain untested.

Additional distilled-model evidence (2026-10-18): The supplied Snofs distilledV12Fp8
Q2_K and Q2_K-Python GGUFs match all 201 tensor payloads/types/shapes and interpreted
metadata; whole-file hashes differ. Their export-command provenance is not verified
by filenames. Three supplied PNGs use different seeds (...788 original, ...789 Q2,
...790 Python), so visible differences do not establish quantization/backend drift.
Both Q2 active graphs use UnetLoaderGGUF; LoRA is disabled. See
`data/snofs-q2-user-render-comparison.json`. Repeat with one fixed seed before
claiming render parity or a quality difference.

Same-seed Snofs follow-up (2026-10-18): Replacement original/Q2/Python-Q2 PNGs
all use seed 312380090853791 and matching saved sampling settings, no active LoRA.
The two Q2 results are pixel-identical by direct decoded RGB comparison; installed
GGUF hashes still match the earlier complete 201-tensor comparison. Original/Q2
shows shared Visible drift in face/hand/composition. This resolves the earlier
seed mismatch for this setup; there is no additional difference between the two
Q2 exports here. Precision and loader still change together versus the original.
See `data/snofs-q2-same-seed-render-comparison.json`; no new render was initiated.

Artifact cleanup (2026-10-18): User approved removal of all 600 generated model/
raw-Q6 artifacts on `I:` (77.73 GiB), including prepared-F16.gguf. All 357 JSON/log
files remain and their hashes were verified unchanged. Original FLUX.2 source on
`G:` remains present. Earlier `I:` model paths are historical; recreate verified
preparation before performance experiments. Exact audit:
`data/i-drive-cleanup-2026-10-18.json`.

CPU performance update (2026-10-18): Compiled FLUX writing now uses up to eight
bounded concurrent C calls, preserving ordered output and 65,536-value per-call
bounds. Six real-weight slice presets are file-identical to serial writing and
match verified patched native payloads; full slice-writer medians improve
1.53-2.86x (Q2_K 2.69x) on this Windows host. This is not a full-model speed claim.
Phase logs separate kernel checks, F16 preparation/write, encode/write, tensor
readback and output hashing. The roughly 19.9 GB intermediate/spool-copy overhead
was the next I/O candidate at this stage; the direct-spool update below supersedes
that pending item. No new render or large model conversion was performed for this
slice benchmark, and nothing was regenerated on cleaned `I:` during it. See `scripts/benchmark_compiled_writer.py`,
`data/compiled-kernel-performance-validation.json` and the package guide.

Direct-spool update (2026-10-18): Default compiled FLUX jobs without a Keep
intermediate request now reuse shared float preparation and spool only final
K-quant/protected bytes. No full F16 GGUF is written/re-read; the final spool is
still copied atomically. Keeping an intermediate retains the old route. Six
real-weight slice presets remain file-identical to that route and native-equal;
F32/F16/BF16/raw/scaled FP8 and failure/cancel/readback guards pass. A complete
FLUX.2 klein Base 9B Q2_K run takes 115.863 s and exactly reproduces the earlier
source and output SHA256 (201 tensors). The earlier 662.291 s run is historical,
not a controlled speed ratio. No new render was performed. New output/logs are
on `I:\safetensors2GGUF-flux2-q2-direct-2026-10-18`; no F16 model intermediate
was created. See `data/compiled-kernel-direct-preparation.json`,
`data/flux2-q2-direct-conversion.json` and the package guide. Full-model peak
memory and other platforms remain open.

CPU performance follow-up (2026-10-16): read `docs/compiled-kernel-pilot.md`.
Direct Python calls to the archived b3962 Q4_K/Q5_K/Q6_K C kernels pass 498
golden comparisons and run within about 2% of the same directly called kernel on
one prepared FLUX slice. This is an isolated research DLL, not an application
backend. Explicit research-writer integration now passes 32 actual DLL fixture
comparisons and cancellation/atomic-output checks. Full FLUX.2 klein Base 9B
Q4_K_M matches all 201 native payloads and the earlier Python whole-file hash;
write/readback/hash took 461.685 s versus the historical NumPy run's 1539.879 s.
Read `docs/flux-compiled-kernel-full-model-validation.md` for timing boundaries.
Full-model Q5_K_M and Q6_K now also match all 201 native payloads each and their
earlier Python whole-file hashes; write/readback/hash took 410.520 s and 315.509 s.
The same validation guide records all 603 tensor comparisons and timing scopes.
At this historical stage binary packaging and Workbench integration were pending;
the 2026-10-18 Windows integration above supersedes those items. Linux/macOS,
other architectures and full Q4_K_S remain open. Application defaults are unchanged.
An explicit offline package builder/loader is now available; read
`docs/compiled-kernel-package.md`. A fresh Windows package passes platform/source/
file verification, 435 goldens and ten C guards; all four actual package CLI
presets match NumPy. The later Windows integration ships and explicitly selects
the verified package; no runtime build or upstream download is performed.
Non-Windows and clean-machine application qualification remain open.

**Open: Build one unified model converter.** Preserve the tested diffusion and
text-encoder GGUF paths while integrating the additional model/format capabilities
of `molbal/ComfyUI-GGUF`.

**Prerequisite: Own the critical dependencies.** Essential conversion features
must remain installable/buildable if an upstream repository disappears. Preserve
complete pinned source snapshots (including required bundled dependencies,
licenses and provenance) under project-controlled distribution, alongside tested
platform binaries or reproducible build instructions and checksums. A commit ID,
submodule URL or ignored local cache alone does not satisfy this requirement.
Normal conversion must use the shipped/pinned tools; upstream access belongs to
explicit maintenance/update work, not an implicit fetch of moving HEAD on first
use. Separate this requirement from optional user-requested model downloads.

Analysis started (2026-10-12) on `research/unified-converter-inventory`.
Read `docs/converter-consolidation-inventory.md` and the frozen source manifest in
`third_party/converter_references`. Complete tracked-source references, the old
Kompute gitlink, license/hash verification and static model rules are preserved.
Normal conversion still uses the existing backends; offline builds, dependency
packaging, numerical parity, new loader work and language selection remain open.

Reference analysis updated (2026-10-15): read
`docs/converter-reference-analysis.md` for resolved static inheritance, precision
policy differences and 96 small encoder probes. Frozen Molbal Q4_K/Q5_K math
clips positive-only ramps; do not adopt it unchanged. Follow-up native CPU probes
cover 360 cases: all 95 valid encodings per llama.cpp source are byte-identical
across b3962 and both frozen text refs. Read
`docs/converter-native-reference-comparison.md`. Full model/policy parity and
Linux native execution remain open; production dispatch is unchanged.

Policy follow-up (2026-10-15): read `docs/converter-policy-comparison.md`.
84 schematic native CLI conversions record tensor-order sensitivity, image
fallbacks and T5 mixed-policy differences; 41 installed safetensors headers were
audited without loading weights. The user designated
`G:\ComfyUI-Easy-Install\ComfyUI\models` as local sources and `I:\` for test model
conversion artifacts. Recheck free space before large runs; preserve originals.
Full-model conversion/render fidelity and Linux native execution remain open.

Shared-math pilot (2026-10-15): read `docs/shared-q6-pilot.md`. A research-only
NumPy Q6_K encoder matches all three frozen native references on 166 inputs
(498 byte comparisons), including 21 small weight slices from seven installed
floating-point families. Bounded chunk iteration also preserves bytes in 332
additional comparisons and supports consumer-driven early close. It is not
a default production backend; the scoped experimental integration is noted
below. Three complete installed matrices additionally
match in 912 native chunk comparisons, with atomic raw `.q6k` artifacts on `I:\`;
these are not loadable GGUF models. Per-run encoding times are recorded.
The same three payloads also pass a combined research-only GGUF container
roundtrip (shape/type/alignment/checksum); its mixed-source research architecture
is not a model loader target. Read `data/q6-gguf-container.json` for evidence.
An explicit FLUX F16/F32 GGUF-to-Q6_K research path additionally matches patched
b3962 on eight schematic files (196 tensor comparisons), with metadata/protection
and row-fallback checks. Read `docs/flux-q6-research-path.md`. It does not add a
production route or prove complete-model loading/rendering.
Full-checkpoint numeric follow-up: one installed FLUX.2 klein Base 9B checkpoint
passes all 201 tensor payload/type/shape comparisons and metadata against native
b3962 after common float preparation. The full run exposed a Windows NumPy uint32
alignment/large-offset overflow; shared research reader normalization and >2/4 GiB
regressions fix it. Read `docs/flux-q6-full-model-validation.md`. A user-generated
PNG pair confirms inference via `UnetLoaderGGUF` with two enabled LoRAs and
matching settings against the original; main-subject quality remains close in
this pair. Workflows/hashes are retained in `data/flux-q6-user-render-comparison.json`.
Offload, Linux, isolated memory/throughput and broader coverage remain open.
An explicit experimental Workbench/API backend `python_flux_q6` now reuses F16
preparation for FLUX safetensors/Q6_K only, with overwrite, cancellation/progress
and intermediate cleanup. Native remains the default; no implicit error fallback
or global Python precision entry is added. See `docs/flux-q6-research-path.md`.
Measured peak memory/comparative throughput,
extreme-range policy, other encoders and full-model integration remain
open; preserve separate model/preset rules and the implementation-language choice.

Chunk-size follow-up: `docs/q6-chunk-benchmark.md` records isolated Windows
CPU slice timings/peak RSS and native-prefix parity for five sizes. Keep the
current 256-block default; larger tested chunks were slower. Full-model memory
and native-versus-Python throughput are still unmeasured.

Shared Q5_K math follow-up: `docs/shared-q5-pilot.md` records 166 inputs and
498 exact native byte comparisons plus 332 chunk checks. Positive-ramp saturation
is absent; signed-zero header behavior matches native. This is a block encoder,
now followed by explicit research FLUX Q5_K_M file export (eight schematic
files, 196 exact native tensor comparisons), plus complete FLUX.2 klein Base 9B
parity: all 201 tensor payloads/types/shapes and metadata match. Read
`docs/flux-q5-full-model-validation.md`. Supplied Python/native Q5 PNGs are
pixel-identical with two enabled LoRAs. A follow-up original with matching seed
and settings shows Visible drift in pose/composition, without obvious corruption.
Broader quality remains open. The experimental `python_flux` Workbench/API
backend now accepts Q6_K/Q5_K_M; `python_flux_q6` stays a Q6-only compatibility
alias. Native defaults, source validation, cleanup and atomic output are retained. Existing Q6_K/native
paths and the Krea allowlist remain intact.

Shared Q4_K math follow-up: `docs/shared-q4-pilot.md` records 166 inputs,
498 exact native comparisons and 332 chunk checks. Q4/Q5 share affine fitting,
packing and bounded streaming with distinct native search grids. Q5 was rerun
and remains exact. FLUX Q4_K_S/Q4_K_M research file policies now match patched b3962 on 16
schematic files (400 tensors), including ordered V promotion and row-fallback
counter behavior. Read `docs/flux-q4-research-path.md`. Complete FLUX.2 klein
Base 9B Q4_K_M also matches all 201 native tensor payloads/types/shapes and
metadata; read `docs/flux-q4m-full-model-validation.md`. Supplied Q4_K_M
Python/native PNGs are pixel-identical with two LoRAs; the matching-seed original
shows Visible drift in pose/composition. A second no-LoRA triplet is also
Python/native pixel-identical; original-versus-Q4 differences persist without
active LoRAs. Full Q4_K_S, broader Q4 quality and
application integration remain open.

Recorded roadmap (the final consolidation strategy remains open):
- First provide one application with a shared format registry, model checks,
  precision protection, progress/cancellation and safe output handling.
- Initially dispatch to the existing patched diffusion llama.cpp toolchain,
  plain text-encoder llama.cpp toolchain, and an integrated molbal backend,
  using pinned, traceable versions and matching model/runtime tests.
- Then evaluate selectively consolidating implementations or maintaining forks
  of all three sources into one converter. A wholesale merge remains an
  alternative to assess.
- Consider persistent Linux credential storage through Freedesktop Secret
  Service where available. Retain session-only entry as the distribution-neutral
  fallback and never silently fall back to plaintext storage.
- Add a **New folder** action to the local source/output browser so users can
  create a destination directory without leaving the Workbench.
- Keep working conversion formats available throughout; do not replace a tested
  path solely because another backend offers more features.
- Idea (2026-10-12): Develop shared ComfyUI loader nodes that can load both
  city96-compatible and molbal-specific GGUF models. Evaluate explicit backend
  selection or validated architecture/format dispatch, with unique node IDs to
  avoid registration collisions. Preserve existing city96 workflows; do not
  assume either backend can interpret every format. Implementation and runtime
  validation are future work, not authorized by recording this idea.

Krea preparation completed: the complete molbal source at commit
`5a0a3ffa0e3eae5c6af8b0b981b660a24d5fbc04` is tracked in `third_party/molbal`
with license/provenance/checksum and an offline verifier. Read
`docs/krea2-backend-preparation.md` for the concrete adapter contract and remaining
work. The Workbench adapter is implemented and synthetic exports pass for its
seven-format allowlist. Full-model Q4_0 cat/dog renders with molbal passed with
visible drift versus one installed INT8 ConvRot checkpoint. Stock city96 rejected
the exact GGUF. The initial eight-step recipe also has shared mosaic artifacts;
52-step source sampling substantially reduces them. Confirm Raw/Turbo provenance
and repeat a clean source/GGUF quality comparison before endorsing quality.
Read `docs/krea2-render-validation.md`; other checkpoints/levels,
loader coexistence, LoRA, forced offload and Linux remain pending.

Read and retain this open TODO at the start of every project session. Do not
start implementing it merely because a session begins; use it when planning
relevant work. Update its status when implemented or explicitly superseded.

This project uses Claude Code Hooks and sub-agents that run automatically before every `git commit`.

---

## Commit Pipeline

```
git commit
    │
    ▼
┌─────────────┐     failed                ┌──────────────────────────┐
│  docs-agent │ ──────────────────────────▶│ Commit blocked           │
│             │                            │ (docs incomplete)        │
└──────┬──────┘                            └──────────────────────────┘
       │ OK
       ▼
┌─────────────┐     failed                ┌──────────────────────────┐
│ test-agent  │ ──────────────────────────▶│ Commit blocked           │
│ (command,   │                            │ (tests failing)          │
│  no repair) │                            │                          │
└──────┬──────┘                            └──────────────────────────┘
       │ OK
       ▼
   Commit ✓
```

---

## docs-agent

**File:** `.claude/agents/docs-agent.md`
**Trigger:** `PreToolUse` on `git commit *`
**Timeout:** 120 seconds

### What it checks
| Area | Action |
|---|---|
| Python docstrings | Adds missing docstrings for public API |
| `CHANGELOG.md` | Logs staged changes under `[Unreleased]` |
| `README.md` | Updates on API or installation changes |
| `AGENTS.md` | Updates when new agents/hooks are added |

---

## test-agent

**Config:** `.claude/settings.json` → `hooks.PreToolUse`
**Trigger:** `PreToolUse` on `git commit *`
**Timeout:** 300 seconds
**Type:** `command` (deterministic — not an LLM agent, unlike docs-agent above)

### What it does
1. Runs `uv run pytest --tb=short -q --basetemp .pytest-tmp -p no:cacheprovider`
   (output captured to `.pytest-hook-last.log`, gitignored)
2. Exit code `0` or `5` ("no tests collected") → allows the commit
3. Any other exit code → blocks the commit with the exit code and a pointer
   to `.pytest-hook-last.log`/`uv run pytest` for details

This was originally a `type: "agent"` hook that ran a subagent to analyze and
attempt to repair failing tests (up to 3 tries) before blocking. Converted to
a plain `command` hook because the subagent itself needs Bash-tool permission
to run pytest, which some environments' `dontAsk` permission mode denies by
default (no interactive prompt available to a headless hook) — the
deterministic exit-code check needs no tool permission at all, since it runs
directly as the hook's own subprocess. Test failures now block the commit
outright rather than being auto-repaired; fix them in the normal editing flow
before retrying the commit.

The repo-local `.pytest-tmp` base directory avoids Windows temp-folder permission
issues in automated commit hooks. Pytest's cache provider is disabled for the
hook so stale or locked `.pytest_cache` files cannot block a commit.

---

## Configuration File

Both hooks are configured in `.claude/settings.json`.
The agent definition for docs-agent is in `.claude/agents/docs-agent.md`.
Local Claude permissions are stored in `.claude/settings.local.json`; that file is ignored by Git.

---

## GitHub CI

**File:** `.github/workflows/ci.yml`
**Trigger:** pushes and pull requests targeting `master`, plus manual `workflow_dispatch`

The CI workflow runs on `windows-latest` and mirrors the local validation gates:

1. `uv sync --dev --frozen`
2. `uv run pytest --tb=short -q --basetemp .pytest-tmp -p no:cacheprovider`
3. `uv run ruff check .`
4. Node 22: `npm ci` and `npm run build` in `frontend` (TypeScript and Vite).

---

## Running Manually

```bash
# Run tests only
uv run pytest --tb=short -q --basetemp .pytest-tmp -p no:cacheprovider

# Trigger documentation check manually
# (start agent via Claude Code)
```

## Runtime test defaults

Native `INT4_CONVROT_MIXED` is a production diffusion safetensors target.
Keep eligibility/planning/writing consistent and preserve source precision on
fallback. Use `scripts/check_int4_native.py` with the installed ComfyUI Python
to check Kitchen storage compatibility without model inference. Runtime evidence
is checkpoint/workflow-specific; do not start new ComfyUI renders unless requested.
Latest additional Base evidence: read `docs/zimage-base-render-validation.md`
for the 2026-10-09 ten-format portrait batch, including plain NVFP4 anatomy failure.
The user requested deletion of the Base batch images on 2026-10-10; only
workflow/API snapshots, metrics, logs and the written findings remain.

### Standard visual comparison prompt

Use this user-approved prompt for future quantization image comparisons unless
the user specifies another prompt:

> A gorgeous adult woman leaning against a motel doorway at night, full body,
> both hands visible, wearing a short black dress and heels, neon sign glowing
> behind her, wet pavement reflections, smoky cinematic atmosphere, sultry
> expression, dark noir sensuality, realistic skin, premium cover art, masterpiece.

Keep the same workflow and fixed seeds across reference and quantized runs.
Assess face, expression, pose, anatomy and image quality. Small changes in
reflections, texture or lighting alone do not justify a Visible drift warning.
Record meaningful observed changes and limit conclusions to the tested setup.
The user deferred the ZIT portrait comparison to avoid reconversion costs;
do not resume it without a new request. Existing still-life evidence remains valid.

Prefer an existing matching workflow from the installed user workflow folder,
using a named isolated copy and preserving sampler/steps/CFG/resolution/LoRAs.
Change prompt and necessary model/loader/output references only. A grouped GGUF
Untested cell means one representative precision (Q4_K_M by default), not all
GGUF precisions, unless the user explicitly requests them.
