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
previously observed original-versus-Q4 image drift. Full-model Q4_K_S,
other architectures, Linux/macOS, packaged binaries and Workbench integration
remain unvalidated. There is no implicit build, search or encoder fallback.

## Q5_K_M and Q6_K full-model follow-up

Both additional full-model runs completed on 2026-10-16 using the same compiled
library, writer and prepared FLUX input. They ran sequentially in a fresh folder:
`I:\safetensors2GGUF-flux-compiled-q5-q6-full-2026-10-16`.
Free space was 183,996,428,288 bytes before starting. The complete source SHA256
was verified again before these runs; each native reference's complete checksum
was verified immediately before its conversion. Source size/mtime remained stable
through both conversions and comparisons. No earlier artifacts were replaced.

| Format | Output bytes | Tensor types | Write/readback/checksum |
|---|---:|---|---:|
| Q5_K_M | 6,688,921,920 | 99 Q5_K + 16 Q6_K + 86 F32 | 410.520 s (6 min 51 s) |
| Q6_K | 7,774,067,008 | 115 Q6_K + 86 F32 | 315.509 s (5 min 16 s) |

For **each format**, all 201 tensor names/types/shapes/payloads, typed metadata and
effective alignment match the independent native reference. Both outputs also
match the complete file checksums of the earlier Python exports:

- `compiled-Q5_K_M.gguf`:
  `0ea183f94b052c4e7e00b40fd0bf69f68cfeb75115a3fd3772c99a3b4d96d14b`.
- `compiled-Q6_K.gguf`:
  `9ed4dd03c54e036885f499c8f1b740a2d344115845595f306cfe8c46ce6b1629`.

The new reports `data/flux-compiled-q5-full-model-comparison.json` and
`data/flux-compiled-q6-full-model-comparison.json` retain every tensor checksum,
full-file hashes, output size, native/library/build/source provenance and separate
writer/comparison timings. The corresponding native file checksums equal the
earlier Q5/Q6 reports; the historical 32-byte header difference remains.

As above, timings exclude preflight input/reference hashing and the independent
comparison. Historical NumPy runs took 1,475.984 s for Q5 and 1,136.084 s for Q6,
including readback/hash. Historical native observations were 381.491 s and
441.711 s respectively. These are different runs with uncontrolled cache/load;
the earlier Q6 native run specifically had shared-load conditions. They do not
establish that this library outperforms the native executable. The slice test
is the controlled evidence that Python's bounded-call overhead is small.

Across Q4_K_M, Q5_K_M and Q6_K, **603 full-model tensor comparisons pass** using
one shared C ABI and the unchanged Python model/file policy. No new renders or
application default changes were made. Packaging, other platforms/architectures,
full Q4_K_S and Workbench integration remain open.
