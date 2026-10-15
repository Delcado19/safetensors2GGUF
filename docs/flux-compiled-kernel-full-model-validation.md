# FLUX.2 klein Base 9B: compiled CPU kernel full-model validation

Date: 2026-10-16. Branch: `research/unified-converter-inventory`.

## Scope and inputs

The existing research FLUX writer now accepts an explicitly selected compiled
block encoder. Python still owns model policy, metadata, protected tensors,
bounded file I/O, progress, cancellation, readback and atomic publication.
The preserved b3962 Q4_K/Q5_K/Q6_K reference math runs through the small C ABI
described in [the kernel pilot](compiled-kernel-pilot.md). No application backend
or default has changed; this does not choose a final implementation language.

The full-model test reuses the prepared source and independent native output
from [the earlier Q4_K_M validation](flux-q4m-full-model-validation.md):

- Prepared source: `I:\safetensors2GGUF-flux-q6-full-2026-10-15\prepared-F16.gguf`,
  19,939,776,800 bytes, SHA256
  `dfb03aff891ec8f1cfdd8b4ea9cd4f7500053903155f719718fe7c1a0f8d67cc`.
- Native reference: `I:\safetensors2GGUF-flux-q4m-full-2026-10-15\native-Q4_K_M.gguf`,
  SHA256 `82c406e8b6c0772ebece26d188cb6a4f38c8ac273f3325308ee0cd0b940e9470`.

Both complete input hashes were checked immediately before conversion. The
source archive, City96 patch and native executable checksums were also verified.
The DLL hash and its recorded build/source provenance are retained with the
new report. Free space on I was 189,564,538,880 bytes before the experiment.
All existing models and earlier artifacts were preserved.

## Output and validation

New output:
`I:\safetensors2GGUF-flux-compiled-q4m-full-2026-10-16\compiled-Q4_K_M.gguf`.

Size: 5,560,654,144 bytes. SHA256:
`3239e9be7414fd07735091ea3e83b4592913bd5253244f08a8351c37925250b0`.

This is the same complete file checksum as the earlier Python Q4_K_M output.
All 201 tensor names, shapes, types and full payloads match the independent
native reference: 99 Q4_K, 16 Q5_K and 86 F32. Typed metadata and effective
alignment also match. The native file is 32 bytes shorter because of the
existing explicit-default-alignment header difference; it is not a weight
difference. Source size/mtime remained unchanged after conversion/comparison.

The writer expanded at most 65,536 values per call. It checked cancellation
before and after each call, verified returned byte dtype/contiguity/length,
read back every output tensor and computed a complete output checksum before
publishing the temporary file atomically. Peak process RSS was not measured.

`data/flux-compiled-q4m-full-model-comparison.json` retains all 201 tensor hashes,
complete file hashes, input/build/codec provenance, timing scopes and live checks.
The I-folder also retains the writer result, small fixtures and CLI result.

## Safety and CLI checks

Before the large conversion, the actual DLL produced **32 complete files
identical to NumPy**: all four FLUX presets (Q6_K, Q5_K_M, Q4_K_S, Q4_K_M),
F16/F32 sources, both tensor orders and aligned/fallback rows. The Q4 fallback
V tensor consumes its native policy counter in the same order.

Real DLL calls also precede three failure checks: cancellation, a late NaN
tensor and an injected exception after successful native encoding. Each retained
the original source and existing destination and removed the temporary output.
These checks exercise handled errors, not process termination or a native crash.

The explicit CLI was executed on a small Q4_K_M fixture with the actual DLL and
its SHA256; its entire output equals the NumPy file and its JSON identifies the
library. Unit tests also reject incomplete library/hash argument pairs before
creating output, and preserve existing files on malformed callback output.

## Timings and limits

The full compiled writer took **461.685 seconds (7 min 42 s)**, including tensor
encoding, writing, complete tensor readback and whole-file hashing. Source and
reference preflight hashing and the subsequent independent comparison are
outside this duration. This is one sequential Windows CPU run, not a controlled
cross-backend benchmark; host/cache state was uncontrolled and short unit checks
ran in the same overall session without CPU/cache isolation.

Historical observations on the same prepared input:

| Run | Observed time | Scope |
|---|---:|---|
| Earlier NumPy Q4_K_M | 1,539.879 s | Write, full readback and checksum; before the layout optimization |
| Current compiled writer | 461.685 s | Write, full readback and checksum |
| Earlier native executable, one worker | 395.384 s | Native executable run; excludes Python readback/checksum |

The current compiled run is about 3.3 times faster than the earlier NumPy run
and is much closer to the historical native time. This does not establish a
same-run speed ratio versus native. The separately controlled single-slice
comparison remains the evidence for roughly 0.6-1.6% Python-call overhead.

Reproduce with a fresh output using the explicit `--kernel-library` and
`--kernel-sha256` command in [the pilot guide](compiled-kernel-pilot.md), then
run `python -m scripts.compare_gguf_outputs --help` for the bounded comparator.
Never overwrite the source or a previous test artifact for this experiment.

No additional image generation was performed. Existing Q4_K_M visual findings
remain applicable to these identical weights; this does not improve the
previously observed original-versus-Q4 image drift. Full-model compiled Q5/Q6,
other architectures, Linux/macOS, packaged binaries and Workbench integration
remain unvalidated. There is no implicit build, search or encoder fallback.
