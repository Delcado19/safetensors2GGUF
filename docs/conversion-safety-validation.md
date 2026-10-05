# Conversion Safety Validation

Validated on Windows on 2026-10-05 after the Graphify-led project review.

- Full suite: **434 tests passed**, including the Qwen 2511 marker regression.
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

Native llama-quantize and llama.cpp conversion were mocked in unit tests.
An additional real Qwen Image Edit 2511 FP8 checkpoint conversion completed
with the Easy-Install llama-quantize build 3962 (c8c07d658), using eight threads:

- Source: 20,448,075,055 bytes; source size and modification time unchanged.
- F16 intermediate: 47,747,889,824 bytes, generated in 110.90 seconds.
- Q4_K_M output with restored variant marker: 13,146,749,536 bytes.
- Native quantization: 136.01 seconds; full successful pipeline: 263.33 seconds.
- All 1,934 tensor names and shapes match the source after removing source
  quantization sidecars. Output types: 1,094 F32, 580 Q4_K, 232 Q6_K, 28 Q5_K.
- The empty `__index_timestep_zero__` marker originally triggered the native
  zero-dimension assertion. It now travels through the existing companion-file
  mechanism and is restored with its exact empty shape after quantization.
- Sampled Windows process counters observed 19.78 GiB peak working set for the
  Python conversion process and at least 33.27 GiB for llama-quantize. Sampling
  was incomplete; these are not a measured whole-pipeline peak RAM bound.

The output and conversion logs/report were saved in the requested Downloads
directory; temporary intermediates were removed. Live Hub downloads and ComfyUI
rendering were not tested; model support/quality classifications remain unchanged.
