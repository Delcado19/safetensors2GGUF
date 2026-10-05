# Conversion Safety Validation

Validated on Windows on 2026-10-05 after the Graphify-led project review.

- Full suite: **433 tests passed**, including 29 new regression cases.
- Ruff: all checks passed. `git diff --check`: clean.
- Real Python subprocesses exercised cancellation with silent stdout, closed
  stdout before process exit, and caller/log-callback failure; children were reaped.
- File-backed checks exercised original-output preservation, empty-output
  rejection, no-overwrite publication races, hard-link source aliases, and
  Windows handle cleanup after GGUF write errors.
- Synthetic Wan conversion exercised job-specific 5D export and reinsertion.
  Failed writes and failed final replacement preserved the previous GGUF and
  companion. Metadata-based companion discovery was verified.
- HF tests used local fake downloads: dtype/value round trips, header/data
  ordering, duplicate rejection, cancellation/resume preservation, and weak
  references proving earlier tensor wrappers were released before the next load.

The local `uv` launcher failed with `uv trampoline failed to canonicalize script
path`. Validation used the existing project environment directly:

```powershell
.venv/Scripts/python.exe -m pytest --tb=short -q --basetemp .pytest-tmp -p no:cacheprovider
.venv/Scripts/ruff.exe check .
```

Native llama-quantize and llama.cpp conversion were mocked in pipeline tests.
Large-model peak RAM, live Hub downloads, and ComfyUI rendering were not tested;
the model support/quality classifications therefore remain unchanged.
