# Streaming Safetensors Writer — Design

## Problem

`convert_to_safetensors()` (`convert_safetensors.py`) quantizes tensors
one at a time from a lazy, mmap-based `_LazyStateDict` (bounded read-side
RAM), but accumulates every quantized result into an `out_tensors` dict
held entirely in memory, then calls `safetensors.save_file(out_tensors,
dst_path)` exactly once at the end. Peak RAM therefore scales with the
full quantized output size, not with any single tensor.

Found 2026-08-23 quantizing FLUX.2 dev's diffusion model (32B params, 50GB
system RAM): INT8 hit a `MemoryError` inside `save_file()`'s `_tobytes()`
call; INT8_MIXED then crashed harder (SIGSEGV) once retried in the same
long-lived process. NVFP4 additionally has its own per-tensor 16x
temporary-buffer spike (`safetensors_quant_nvfp4.py`'s
`_nearest_e2m1_index()` broadcasts against all 16 E2M1 codebook values at
once) — compounding on top of the accumulation problem, not a separate
bug. See `[[project_future_streaming_write_resume]]` (session memory) for
the full incident history and the investigation that ruled out swapping
to a different safetensors library (`syoyo/safetensors-cpp`,
`carsonpo/safetensors.cpp`, and the installed Python `safetensors` 0.7.0
package all buffer the complete file before one write — none of them
stream).

**Reference implementation:** `leejet/stable-diffusion.cpp`'s
`src/model_io/streaming_writer.h` / `safetensors_io.h` — an abstract
`StreamingModelWriter` with `write_metadata()` (writes the full header
from a pre-computed plan before any tensor is quantized), `write_tensor()`
(appends one tensor's finished bytes), and `file_size()` (known upfront,
used to preallocate the output file). Worth reading directly during
implementation; not reusable as code (different language, different type
system), but the architecture this design follows.

## Scope

**Streaming only.** Resume-from-checkpoint is a deliberately separate,
smaller follow-up once this lands and is tested — resume only makes sense
built on top of a streaming writer, and bundling both would make this
change harder to review and roll back independently. See
`[[project_future_streaming_write_resume]]` for the resume design notes,
unchanged by this spec.

Out of scope: the GGUF output path (`convert.py`/`quantize.py`) — that
writer already goes through `llama-quantize.exe` as a subprocess and
doesn't hold the full state dict in Python-side RAM the way this path
does.

## Architecture

Two passes over the same tensor iteration `convert_to_safetensors()`
already does, replacing the single accumulate-then-`save_file()` pass:

### Pass 1 — Plan

Iterate `state_dict` with the exact same skip logic already in place
(`quant_skip_keys` from `_scan_quantized_layers`, `_PASSTHROUGH_TENSOR_
SUFFIXES`, `model_arch.keys_ignore`) — this iteration order and filter
set must stay byte-for-byte identical between passes, since Pass 2 relies
on producing tensors in the same order Pass 1 planned them in. Both
passes should share one filtering helper/generator rather than duplicating
the skip conditions, to make that invariant structural instead of
convention-only.

For each surviving key, determine the tensor(s) it will produce —name,
safetensors dtype string, shape — **without quantizing any data**:

- Passthrough tensors (`spiece_model`, `tekken_model`, ...): same name,
  same dtype, same shape as source (byte-copy, not the shape-derivation
  path).
- Everything else: call a new `plan_tensor_output(key, shape, dtype,
  model_arch, target_key) -> list[(name, st_dtype, shape)]`.

`plan_tensor_output()` is extracted from `estimate_safetensors_output_
size()`'s existing per-tensor branching (`safetensors_quant.py`), which
already replicates `quantize_tensor_st()`'s exact decision tree (is_hiprec_
st's 1D/small/keys_hiprec gate, the >=3D Conv fallback, keys_shape_
critical fallback, NVFP4's last-dim%16 and INT8 ConvRot's last-dim%256
requirements) purely from shape/dtype metadata. Today that logic computes
a byte *count*; this refactor makes it also return the per-tensor *name/
dtype/shape* list, and both `estimate_safetensors_output_size()` (unchanged
external behavior, still a GUI size estimate) and the new streaming write
path call the same function — one source of truth for "what will this
tensor become," not two copies that can drift.

**Already-quantized sources (dequant pre-pass):** `_scan_quantized_layers()`
already identifies these keys and their on-disk quant format up front,
cheaply, before this loop starts. For FP8/INT8-sourced keys, dequantizing
doesn't change shape (weight + sidecar scale merge back to the original
float shape) — the raw header shape is already correct for planning. For
NVFP4-sourced keys, the on-disk last dimension is packed (2 values/byte,
halved) — a new small helper, `dequantized_shape_of(fmt, packed_shape) ->
shape`, in `dequantize.py`, inverts that specific packing rule so Pass 1
can plan against the true post-dequant shape instead of the packed one.
This is the one place shape math has to be duplicated (forward pack in
`safetensors_quant_nvfp4.py`, inverse unpack here) — keep both next to a
comment cross-referencing the other.

From this pass, compute:
- The complete safetensors JSON header (tensor name → dtype/shape/byte
  offset), and therefore the exact final file size.
- `layer_formats` / the `_quantization_metadata` sidecar JSON — today
  built alongside quantization in the main loop; nothing in it is actually
  data-dependent (ConvRot's per-layer bool is `in_features % group_size`,
  `full_precision_matrix_mult` is a static flag from `target_key`/the
  `full_precision_fp8`/`full_precision_nvfp4` parameters) — verified
  against the current `convert_to_safetensors()` body, this is a genuine
  no-op move, not new logic.

### Write header, preallocate

Write the 8-byte little-endian header-length prefix + JSON header to
`dst_path`. Preallocate the file to its final known size (truncate/seek-
extend) so Pass 2 only ever appends sequentially — no filesystem
fragmentation concern, and it surfaces a disk-full error immediately
rather than partway through a multi-hour run.

### Pass 2 — Stream

Iterate `state_dict` again, same order and filters as Pass 1. For each
key, run the **real** `quantize_tensor_st()` (or the passthrough copy)
exactly as today, but instead of `out_tensors.update(quantized)`, write
each resulting tensor's bytes directly to `dst_path` at the offset Pass 1
computed for it — sequential appends in planned order, no seeking. Before
writing, assert the actually-produced dtype/shape matches what Pass 1
planned for that tensor name; this is a correctness invariant (no data-
dependent branching should exist in `quantize_tensor_st()` beyond the
already-shape-based fallbacks it has today), not expected to ever fire,
but cheap insurance against the two passes silently drifting apart in a
future edit.

Progress/log callbacks (`on_progress`, `on_log`, `log_tensor_every`,
`cancel_event`) move to Pass 2, which is where the actual (slow) work
happens — Pass 1 is expected to be fast enough not to need its own
progress reporting, but should still check `cancel_event` between tensors
since a very large model's planning pass is not instant either.

## Error handling

- **Pass 1 failure:** raises the same exceptions as today (bad key,
  unknown `target_key`, etc.), before any byte of `dst_path` has been
  touched — strictly safer than the current behavior (today's single-pass
  design can partially quantize into `out_tensors` before failing; nothing
  is written either way, so no behavior change from the caller's
  perspective).
- **Pass 2 failure / kill:** leaves a preallocated-but-incomplete file at
  `dst_path`. No resume in this step (out of scope, see above) — same
  net outcome as today's OOM/kill (the run must be redone), but now also
  leaves a partial file on disk that a caller should treat as invalid
  (`overwrite=True` on a retry naturally replaces it; a future resume
  feature would read the progress checkpoint discussed in
  `[[project_future_streaming_write_resume]]` to skip already-written
  tensors instead of overwriting). Worth a doc note in this function's
  docstring; no code needed for that alone.

## Files touched

- `convert_safetensors.py` — core loop split into plan/write-header/stream,
  replacing the current single loop + trailing `save_file()` call.
- `safetensors_quant.py` — extract `plan_tensor_output()` from
  `estimate_safetensors_output_size()`'s branching; both functions call it.
- `dequantize.py` — new `dequantized_shape_of(fmt, packed_shape)` helper
  for the NVFP4-packed-source planning case.
- No change to `text_encoder_convert.py` or any other caller — this is a
  write-path-internal refactor; `convert_to_safetensors()`'s signature and
  return value (`(dst_path, model_arch)`) are unchanged.

## Testing

- **Byte-equivalence regression test:** convert a small fixture model
  through both the (temporarily kept, or reconstructed for the test) old
  accumulate-then-`save_file()` path and the new streaming path, for each
  of the 6 safetensors target formats, and assert the outputs are
  equivalent — load both via `safetensors.torch.load_file()` and compare
  tensor-by-tensor with `torch.equal()` (not a raw byte diff, since header
  key ordering is allowed to differ between the two approaches).
- **`dequantized_shape_of()` unit test:** a synthetic NVFP4-packed source
  fixture (packed last-dim, sidecar scale tensors) — assert the inferred
  shape matches the known pre-quantization shape used to build the
  fixture.
- **Existing test suite** (`tests/test_convert_safetensors.py` and
  friends) must stay green unmodified in behavior — these tests assert on
  `convert_to_safetensors()`'s output content, not its internal write
  mechanism, so they double as regression coverage for this refactor.
- No new test infrastructure (no mocking framework, no fixtures directory
  restructure) — reuse the existing fixture-building patterns already in
  `tests/test_convert_safetensors.py`.

## Explicitly not pursued

- **Chunked/buffered partial flushing** (collect N tensors, flush
  periodically) — considered and rejected: reduces peak RAM but not to
  the O(1)-per-tensor target this design achieves, and multi-file
  sharding to keep `save_file()`'s all-at-once API would introduce
  compatibility risk with ComfyUI's loaders (which expect a single
  `.safetensors` file, not a sharded index) for no correctness benefit
  over the two-pass approach.
- **Swapping to a different safetensors-writing library** — already
  investigated and ruled out (see Problem section above); no existing
  library streams writes, so this has to be hand-rolled regardless of
  which library route was considered.
- **Resume/checkpoint** — deliberately deferred, see Scope.
