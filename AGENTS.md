# Agents & Automation

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
│             │                            │ (tests failing)          │
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

### What it does
1. Runs `uv run pytest --tb=short -q`
2. On failure: analyzes and repairs (max. 3 attempts)
   - Outdated tests → fix tests
   - Real bug → fix source code
3. Only lets the commit through when all tests pass

---

## Configuration File

Both hooks are configured in `.claude/settings.json`.
The agent definition for docs-agent is in `.claude/agents/docs-agent.md`.
Local Claude permissions are stored in `.claude/settings.local.json`; that file is ignored by Git.

---

## Running Manually

```bash
# Run tests only
uv run pytest --tb=short

# Trigger documentation check manually
# (start agent via Claude Code)
```
