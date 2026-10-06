# Agents & Automation

## Persistent project TODO — load at every session start

**Open: Build one unified model converter.** Preserve the tested diffusion and
text-encoder GGUF paths while integrating the additional model/format capabilities
of `molbal/ComfyUI-GGUF`.

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
