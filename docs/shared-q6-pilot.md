# Shared Q6_K mathematical pilot

Date: 2026-10-15. Branch: `research/unified-converter-inventory`.
Implementation: [`scripts/reference_q6_k.py`](../scripts/reference_q6_k.py).
Evidence: [`data/q6-pilot-comparison.json`](../data/q6-pilot-comparison.json).

## What is implemented

One research-only NumPy encoder implements the unweighted GGML Q6_K reference
algorithm. It accepts finite float32 tensors with nonempty rows divisible by
256 and emits the standard 210-byte block layout as uint8, preserving leading
dimensions. It does not mutate inputs. FP16/BF16 source values are explicitly
expanded to float32 by the sample checker; the mathematical encoder itself
rejects other input dtypes rather than silently converting them.

The port follows `nearest_int`, `make_qx_quants` with `rmse_type=1` and no external
importance weights, and `quantize_row_q6_K_ref` from frozen b3962
`c8c07d658a6cefc5a50cfdf6be7d726503612303`. Its local scale search uses x-squared
weights, as upstream does; "unweighted" here means no separate importance matrix.
The module preserves the upstream MIT copyright/license reference. The original
notice is tracked in `third_party/converter_references/llama-b3962-LICENSE`.

Important details retained from native GGML:

- The first value with maximum absolute magnitude determines scale sign.
- Scale fitting keeps C's sequential float32 accumulation order. A generic
  NumPy reduction can change rounding and therefore packed bytes.
- The fitted scales are stored in FP16/int8, then quantization codes are
  recomputed using those stored scales.
- If FP16 scale storage underflows to zero, native code retains the initial
  fitted codes. Zero groups and entirely zero blocks are distinct cases.
- Low/high bit planes and little-endian FP16 scale bytes use GGML's exact layout.

The functions need only NumPy, already installed by the project. They do not
call an external executable, import Molbal, choose a model policy or write a file.
The separate checker invokes native executables solely as validation oracles.
Production conversion dispatch and all existing format allowlists are unchanged.
Python is a reference implementation choice; the final converter language and
execution backends remain open.

## Measured results

**166 input tensors matched all three native references exactly: 498 byte
comparisons, zero differing bytes.** Native binaries are checked against their
retained SHA256 values, and their commits/archive hashes must match the frozen
manifest. An incomplete reference set, empty probe set or changed provenance
cannot establish a passing comparison. A byte mismatch produces a report and a
nonzero checker exit status.

The same 166 tensors also match with chunk limits of one and three blocks:
332 additional chunk-boundary comparisons against the default 256-block path.
These checks are recorded per input in `chunk_checks`; they are included in the
checker's overall pass/fail result.

| Input group | Tensors | Coverage |
| --- | --- | --- |
| Initial aligned vectors | 9 | Zeros, constants, signed/one-sided ramps, outliers, rounding and one-block input |
| Additional edge cases | 8 | FP16 scale underflow, epsilon boundary, tied extrema, mixed zero blocks/groups, strided and 3D input |
| Deterministic random vectors | 128 | Signed/positive/negative values, widths 256/512, scale factors across eight decades |
| Installed weight slices | 21 | Three 2x256 slices from each of seven plain floating-point model families |

The native references are b3962, the frozen installed text llama.cpp and the
newer frozen text llama.cpp from the [native reference comparison](converter-native-reference-comparison.md).
The checker calls their raw Q6_K reference kernels, without model/preset dispatch.
This is exact agreement on the recorded inputs, not exhaustive floating-point
domain equivalence, model quality evidence or a performance measurement.

## Installed weights sampled read-only

Samples cover FLUX, Lumina2/Z-Image, SDXL, Qwen3-4B, Qwen3-8B, CLIP-L and CLIP-bigG.
They come from the [installed header audit](converter-policy-comparison.md).
Only files whose audited dtypes are F16/BF16/F32 and which have no raw-byte or
`.comfy_quant` markers are sampled. Current canonical header hashes must still
match the audit. File size/modification time are checked around sampling.

Each sample records model path, tensor name, source dtype/shape, slice bounds,
canonical header checksum and input/payload checksums. Only the tiny slices are
materialized on CPU; weight values are not stored in the report or repository.
Merged checkpoints are restricted to diffusion keys, excluding their CLIP/VAE
components. Source files under `G:\ComfyUI-Easy-Install\ComfyUI\models` are unchanged.
No complete checkpoint was converted, no model output file was written and no
ComfyUI render was started. Future test model outputs still belong under `I:\`.

**CLIP matrix arithmetic passing does not add CLIP GGUF conversion or loader
support.** These are raw numeric blocks, not complete encoded model files.

## Reproduce

Use the existing locked Python environment. Build the native reference programs
as documented in the native comparison first; their paths and hashes are read
from `data/converter-native-comparison.json`. The checker never downloads or
silently substitutes a different binary.

```bash
# Synthetic vectors only: use a separate report filename.
python -m scripts.check_q6_pilot --output q6-synthetic-report.json

# Reproduce the complete local evidence, including the audited model slices.
python -m scripts.check_q6_pilot --models-root "G:/ComfyUI-Easy-Install/ComfyUI/models" --output data/q6-pilot-comparison.json
```

`--references PATH` can select a newly generated native build report, but all
three pinned source revisions and executable checksums are still required.
The pilot report records the NumPy version, random seed and encoder source hash.
The encoder source uses LF on all checkouts so its recorded hash remains stable.

Pytest recomputes all synthetic inputs and compares their packed bytes with the
recorded native results, without requiring a compiler, external executables or
installed model files. Separate tests exercise invalid dtype/shape/non-finite
input, read-only views, numerical overflow and comparison failure handling.

## Bounded mathematical processing

`iter_q6_k(data, chunk_blocks=256)` yields independent `(blocks, 210)` uint8 arrays
in logical C order. Each block contains 256 values. The default copies at most
256 KiB of float32 input per iteration; fitting and packing working arrays scale
with that fixed chunk size. Flat slicing copies only the current window even for
reversed/transposed NumPy views, avoiding a full contiguous source copy. This is
a structural working-set bound, not a measured process peak-RSS claim. Source
storage and chunks retained by the consumer are additional memory.

`quantize_q6_k` retains its existing result shape and input contracts, and uses
the same iterator internally. It still allocates the entire packed result.
Consumers that need bounded output memory must consume/release iterator chunks
instead of collecting them. No full-model writer has been added.

Iteration validates source values per chunk. An invalid later chunk can fail
after earlier chunks were returned; consumers must use atomic file publication
and discard incomplete output. Consumers can stop/close between chunks to cancel;
there is no mid-chunk interruption. Do not mutate input during iteration. Tests
cover chunk boundaries, independent buffers, reversed/transposed views, late
failure, early close and restoring NumPy error settings before yielding.

## Complete installed-tensor streaming check

`scripts/check_q6_tensor_stream.py` reads one explicitly selected plain
F16/BF16/F32 matrix from safetensors, expands row batches to float32 and packs
Q6_K. Each batch is compared byte-for-byte with **all three verified native
reference binaries** before writing. It does not select model/preset rules.

The output is a raw `.q6k` block stream, **not a GGUF model**. Shape, dtype,
checksums, source size/modification time, reference provenance, timings and
reconstruction metrics live in the separate JSON report. The raw format has no
embedded tensor header and cannot be passed to a ComfyUI model loader.

Three complete matrices passed on 2026-10-15: **19,808,256 values, 304 chunks,
912 native byte comparisons, zero mismatches**. Artifacts total 16,248,960 bytes
under `I:\safetensors2GGUF-q6-stream-2026-10-15`; installed sources are unchanged.

| Source / full matrix | Stored dtype | Shape | Native checks | RMSE | Encode time |
| --- | --- | --- | --- | --- | --- |
| FLUX.2 klein Base 9B / image attention projection | BF16 | 4096 x 4096 | 768 | 0.00037183 | 1.824 s |
| Qwen3-4B Engineer V6 / layer 0 key projection | F16 | 1024 x 2560 | 123 | 0.00043972 | 0.272 s |
| SDXL artaix v30 / embedding projection | F16 | 320 x 1280 | 21 | 0.00040688 | 0.037 s |

Exact tensor names and metrics are retained in
[`q6-stream-flux.json`](../data/q6-stream-flux.json),
[`q6-stream-qwen.json`](../data/q6-stream-qwen.json) and
[`q6-stream-sdxl.json`](../data/q6-stream-sdxl.json).
Encoding alone measured approximately 35–42 MiB/s of expanded float32 input on
this Windows CPU run. Those are single-run observations, excluding source
reconstruction, native comparisons and disk I/O; not a warmed benchmark or a
comparison with native throughput. Total wall time includes all those checks.

Source batches contain at most 65,536 values or one full row, whichever is
larger. The encoder's internal chunk bound still applies. Whole-file mapping
and OS page residency are not a measured peak-memory guarantee.

The writer reuses `validate_output` and `atomic_output`: it rejects originals,
aliases and existing destinations, writes beside the destination and publishes
only after checking expected byte count, disk readback SHA256, unchanged source
size/mtime and cancellation. Late invalid values, native mismatch, I/O errors,
changed source, readback corruption and cancellation remove the temporary output.
Sources with `.comfy_quant` markers are rejected rather than reinterpreted.

```bash
python -m scripts.check_q6_tensor_stream --source "path/to/model.safetensors" --tensor "exact.weight.name" --output "I:/tensor.q6k" --report tensor-report.json
```

The source/output/report parent directories must already exist. Outputs and
reports never overwrite existing files; choose fresh names when reproducing.
Report publication is separate from the raw artifact, so a failed report write
can leave a successfully verified raw output. Regression tests use tiny local
fixtures and stub native execution; the committed reports record the actual
installed-model/native-program runs.

## GGUF container roundtrip

`scripts/check_q6_gguf_container.py` packages the three previously verified raw
streams into one research GGUF using the installed GGUF writer's tensor-info,
metadata and padding API. Payloads are copied through fixed 1 MiB buffers,
without constructing full tensor arrays. No numerical re-encoding occurs.

The three-source local run passed on 2026-10-15. Readback with `GGUFReader`
confirmed all tensor names, logical shapes, Q6_K types, byte counts, 32-byte
alignment and payload SHA256 values. The combined payload contains 16,248,960
bytes; the GGUF adds 800 bytes of header/metadata. Evidence is retained in
[`q6-gguf-container.json`](../data/q6-gguf-container.json), and the actual file is
`I:\safetensors2GGUF-q6-stream-2026-10-15\q6-research-tensors-portable.gguf`.

**This is a container-layout test, not a complete model.** It deliberately mixes
three source models and uses `general.architecture = s2gguf_q6_research` plus an
explicit research-only flag. It must not be offered to a model loader. The raw
stream reports provide trusted source semantics; container readback establishes
layout/payload preservation, not independent revalidation of their native runs.
The GGUF embeds SHA256 values of the input report text after CRLF-to-LF
normalization. Report LF endings are also pinned in Git. Normalizing at the read
boundary is necessary because a Windows-generated working file can still contain
CRLF before checkout, even when its Git blob uses LF. A regression test proves
that LF/CRLF forms produce identical container bytes. The portable evidence was
regenerated from the same three verified payloads; earlier artifacts remain on I:.

The tool rejects failing reports, incompatible shapes, duplicate names,
truncated/extended/corrupt raw files and existing output files. Copy-time byte
counts/checksums detect source changes after preflight. Publication is atomic,
after readback and cancellation checks; errors close writer handles and reader
mappings before temporary-file cleanup, including a malformed GGUF header on
Windows. Tests also exercise odd block counts requiring inter-tensor padding,
dequantization after parsing, cancellation and unchanged input files.

```bash
python -m scripts.check_q6_gguf_container --inputs data/q6-stream-flux.json data/q6-stream-qwen.json data/q6-stream-sdxl.json --output "I:/q6-research-tensors.gguf" --report container-report.json
```

The output/report directories must exist; choose new filenames for reproduction.
Raw paths in input reports must exist locally (relative paths resolve against
their report directory). JSON report publication is separate from successful
GGUF publication. The reader maps the container; fixed copy buffers do not
establish a peak-RSS guarantee or eliminate OS page-cache residency.

## Remaining integration gates

An explicit [FLUX Q6_K research file path](flux-q6-research-path.md) now connects
the encoder to the prepared float-GGUF boundary. Eight schematic files match
patched b3962 on 196 tensor payload/type/shape decisions. The subsequent
[full FLUX checkpoint comparison](flux-q6-full-model-validation.md) also passes
all 201 tensor payloads/types/shapes; runtime gates below remain open.

The current explicit overflow guard rejects extreme finite inputs if native-style scale-search
arithmetic overflows. It does not silently pick a different precision or claim
support for the entire float32 range.

Model eligibility, source reconstruction, protection/fallback and ordered S/M
policies remain separate from this encoder. The [policy probes](converter-policy-comparison.md)
must remain compatible before enabling any production path. The prototype has
not been wired into the Workbench or conversion services.

Other K encoders, IQ, ConvRot/safetensors formats, importance-matrix variants,
production streaming/cancellation integration, broader full-model coverage/loading/render fidelity,
Linux native execution, comparative throughput benchmarking and measured peak
memory remain outside this pilot's evidence.
Do not infer those capabilities from matching Q6_K block bytes.
