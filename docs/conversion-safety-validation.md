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

The output was saved in the requested Downloads directory; conversion logs,
reports and temporary intermediates were subsequently removed. Live Hub downloads were not
tested; model support/quality classifications remain unchanged.

## Qwen Image Edit 2511 GGUF render smoke test

The exact converted Downloads file completed a real edit on the existing local
ComfyUI 0.38.0 server (Python 3.12.10, PyTorch 2.9.1+cu130, RTX 5080 16 GB).
A temporary same-volume hard link and model-directory junction exposed the file
without copying or changing it; the temporary registration was removed afterward.

- Prompt ID: `a1d7aafd-a85a-47d9-9e1f-94fe7d1645f5`; status: success.
- Input: the installed `Cat.jpeg`, resized to 512 x 512.
- Edit: add a realistic bright red knitted hat while retaining the cat and background.
- Qwen Edit Plus conditioning, installed Qwen2.5-VL abliterated FP8 text encoder,
  installed Qwen VAE; no LoRA. Euler/simple, 20 steps, CFG 3, shift 3.1,
  denoise 1, seed 1212121.
- Server execution: 84.39 seconds, including approximately 70.8 seconds sampling.
- Model loaded successfully: 1,094 F32, 580 Q4_K, 232 Q6_K and 28 Q5_K tensors;
  approximately 1,212.74 MB initially offloaded to CPU. No execution errors.
- Output: a decoded and visually inspected 512 x 512 PNG showing the requested
  red knitted hat, recognizable cat, green eyes and gray background. The test
  image and API prompt, history and report JSONs were removed from Downloads
  at the user's request after validation; the converted GGUF was retained.

This proves one end-to-end edit with this converted file and installed components.
It is not a same-seed comparison against the source checkpoint, and does not
establish general quality equivalence or change the model-support ratings.


## Source-aware mixed quantization validation (2026-10-05)

443 tests pass; Ruff and GUI construction pass. Ten additional fixed-seed
ComfyUI renders compare Qwen 2511 INT8/NVFP4 mixed against its FP8 source and
Z-Image Turbo NVFP4 mixed against BF16. Sizes, exact prompt IDs, visual findings,
profile limits and Kitchen packing parity are recorded in
[the quantization size audit](quantization-size-audit.md#implemented-corrections-and-runtime-evidence-2026-10-05).
The temporary ComfyUI model junction was removed after the queue became idle;
original checkpoints and the user's Downloads GGUF were preserved.
