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

## Remaining integration gates

The current explicit overflow guard rejects extreme finite inputs if native-style scale-search
arithmetic overflows. It does not silently pick a different precision or claim
support for the entire float32 range.

Model eligibility, source reconstruction, protection/fallback and ordered S/M
policies remain separate from this encoder. The [policy probes](converter-policy-comparison.md)
must remain compatible before enabling any production path. The prototype has
not been wired into the Workbench or conversion services.

Other K encoders, IQ, ConvRot/safetensors formats, importance-matrix variants,
production streaming/cancellation integration, full-model files/loading/render fidelity, Linux native
execution, throughput and peak memory remain outside this pilot's evidence.
Do not infer those capabilities from matching Q6_K block bytes.
