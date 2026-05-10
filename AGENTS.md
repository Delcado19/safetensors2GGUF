# Agenten & Automatisierung

Dieses Projekt nutzt Claude Code Hooks und Sub-Agenten, die automatisch vor jedem `git commit` ausgeführt werden.

---

## Commit-Pipeline

```
git commit
    │
    ▼
┌─────────────┐     fehlgeschlagen     ┌──────────────────────────┐
│  docs-agent │ ──────────────────────▶│ Commit blockiert         │
│             │                        │ (Doku unvollständig)     │
└──────┬──────┘                        └──────────────────────────┘
       │ OK
       ▼
┌─────────────┐     fehlgeschlagen     ┌──────────────────────────┐
│ test-agent  │ ──────────────────────▶│ Commit blockiert         │
│             │                        │ (Tests rot)              │
└──────┬──────┘                        └──────────────────────────┘
       │ OK
       ▼
   Commit ✓
```

---

## docs-agent

**Datei:** `.claude/agents/docs-agent.md`
**Trigger:** `PreToolUse` auf `git commit *`
**Timeout:** 120 Sekunden

### Was er prüft
| Bereich | Aktion |
|---|---|
| Python-Docstrings | Ergänzt fehlende Docstrings für öffentliche API |
| `CHANGELOG.md` | Trägt gestagte Änderungen unter `[Unreleased]` ein |
| `README.md` | Aktualisiert bei API- oder Installationsänderungen |
| `AGENTS.md` | Aktualisiert bei neuen Agenten/Hooks |

---

## test-agent

**Konfiguration:** `.claude/settings.local.json` → `hooks.PreToolUse`
**Trigger:** `PreToolUse` auf `git commit *`
**Timeout:** 300 Sekunden

### Was er tut
1. Führt `python -m pytest --tb=short -q` aus
2. Bei Fehlern: analysiert und repariert (max. 3 Versuche)
   - Veraltete Tests → Tests anpassen
   - Echter Bug → Quellcode reparieren
3. Lässt Commit nur durch wenn alle Tests grün sind

---

## Konfigurationsdatei

Beide Hooks sind in `.claude/settings.local.json` konfiguriert.
Die Agenten-Definition des docs-agent liegt in `.claude/agents/docs-agent.md`.

---

## Manuelles Ausführen

```bash
# Nur Tests laufen lassen
python -m pytest --tb=short

# Dokumentation manuell prüfen lassen
# (Agent über Claude Code starten)
```
