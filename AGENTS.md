# Agents & Automation

## Persistent project TODO — load at every session start

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

Recorded roadmap (the final consolidation strategy remains open):
- First provide one application with a shared format registry, model checks,
  precision protection, progress/cancellation and safe output handling.
- Initially dispatch to the existing patched diffusion llama.cpp toolchain,
  plain text-encoder llama.cpp toolchain, and an integrated molbal backend,
  using pinned, traceable versions and matching model/runtime tests.
- Then evaluate selectively consolidating implementations or maintaining forks
  of all three sources into one converter. A wholesale merge remains an
  alternative to assess.
- Keep working conversion formats available throughout; do not replace a tested
  path solely because another backend offers more features.

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
Retained local gallery: `runtime-results/zimage-base-2026-10-09/index.html`.

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
