# FLUX Q5_K_M full-model numeric validation

Date: 2026-10-15. Branch: `research/unified-converter-inventory`.

## Result

**PASS: all 201 tensors match patched b3962 in name, type, shape and complete
payload bytes.** Typed metadata and effective alignment match as well. The model
contains 99 Q5_K, 16 Q6_K and 86 F32 tensors. Evidence:
[`data/flux-q5-full-model-comparison.json`](../data/flux-q5-full-model-comparison.json).

A subsequent user-supplied PNG triplet also establishes pixel-identical Python
and native Q5 output in one workflow; see runtime evidence below.

This is the complete FLUX.2 klein Base 9B checkpoint used for the earlier
[Q6 validation](flux-q6-full-model-validation.md), not a distilled variant.
The existing prepared float GGUF was reused after a fresh full SHA256 check:

- Input: `I:/safetensors2GGUF-flux-q6-full-2026-10-15/prepared-F16.gguf`.
- Size: 19,939,776,800 bytes; SHA256:
  `dfb03aff891ec8f1cfdd8b4ea9cd4f7500053903155f719718fe7c1a0f8d67cc`.
- It contains 105 F32 and 96 F16 tensors prepared from the installed BF16 source.
  Original-source provenance and preparation details remain in the Q6 report.
- Source size/mtime remained stable through both conversions and comparison.
  No original model was modified and no new preparation or download was needed.

## Artifacts and provenance

Both outputs and the native log are in
`I:/safetensors2GGUF-flux-q5-full-2026-10-15`. Free space was checked before writing.

| Output | Bytes | SHA256 |
| --- | ---: | --- |
| Python Q5_K_M | 6,688,921,920 | `0ea183f94b052c4e7e00b40fd0bf69f68cfeb75115a3fd3772c99a3b4d96d14b` |
| Native Q5_K_M | 6,688,921,888 | `eb33989ff624436c76395517d63ecf5d8c3c193085759b52eebfbe41b72fb423` |

Whole-file hashes differ despite identical tensor payloads and typed metadata:
the Python writer explicitly records default alignment, giving 32 extra header
bytes. The comparison verifies effective alignment rather than requiring that
optional header key to exist in both files. The Python Q5 file is 1,085,145,088
bytes smaller than the earlier Python Q6 file for the same prepared source.

Before running, `verified_flux_reference` verified the preserved source/patch
archives and native executable SHA256 against the existing policy report.
The full report records those pinned hashes, exact native command, both encoder
source hashes, writer hash, per-tensor checksums, sizes and whole-file checksums.
Both runs completed successfully on their first attempt. Python published only
after full layout/payload readback; comparison subsequently read all tensors in
bounded windows and checked typed metadata. `python-result.json` is also retained
with the local model artifacts. No native rebuild was required.

Native conversion took 381.49 seconds with one worker; Python took 1,475.98 seconds
including readback and output hashing. These sequential single runs have unequal
measurement scope and uncontrolled host/cache conditions. They are observations,
not a controlled throughput comparison. Peak memory was not measured.

## Reproduce

Use a new destination for each command; existing outputs are protected by default.
The executable path and full invocation used locally are in the evidence report.

```bash
python -m scripts.requantize_flux_q6 --source prepared-F16.gguf --output python-Q5_K_M.gguf --preset Q5_K_M
/path/to/pinned/llama-quantize prepared-F16.gguf native-Q5_K_M.gguf Q5_K_M 1
python -m scripts.compare_gguf_outputs --left python-Q5_K_M.gguf --right native-Q5_K_M.gguf --output fresh-comparison.json
```

The historical script name preserves existing Q6 callers. Q6 remains its default.

## User-supplied runtime evidence

The user supplied three renders on 2026-10-15. The complete saved API prompts,
workflows, file checksums, PNG IHDR and decompressed scanline checksums are retained
in [`data/flux-q5-user-render-comparison.json`](../data/flux-q5-user-render-comparison.json).
All PNG chunk CRCs validate; the original images remain unchanged.

| Render | Saved file | RandomNoise seed |
| --- | --- | ---: |
| Original safetensors | `2026-10-15_21-23-15.png` | 993648672917054 |
| Python Q5_K_M | `2026-10-15_21-25-38.png` | 572241167616859 |
| Native Q5_K_M | `2026-10-15_21-28-21.png` | 572241167616859 |

**Python and native are pixel-identical.** Their IHDR values and decompressed
filtered scanline bytes match exactly (SHA256
`5b61cc41afc533afbfa4fddcec84ab402de593025a050bc65a654dfdc85b9d11`).
For these matching 768x1024, 8-bit, noninterlaced RGB PNGs, identical filtered
scanlines imply identical reconstructed pixels. This comparison does not rely
on a subjective visual match or whole-PNG hashes, which include differing metadata.

The saved executed Python/native prompts differ only in node 115's model filename:
`python-Q5_K_M.gguf` versus `native-Q5_K_M.gguf`, both through `UnetLoaderGGUF`.
Both use er_sde, Flux2Scheduler with 34 steps, CFG 4, the same Qwen3-8B heretic
FP8 encoder and FLUX.2 VAE. The two enabled rgthree LoRAs are Lenovo UltraReal and
NiceGirls UltraReal, each at strength 0.7. Prompts and remaining executed nodes
match. This provides scoped loading, inference and LoRA evidence for Q5.

The initial original uses `UNETLoader` and a different seed; retain it as history,
not as the controlled quality reference. The user subsequently supplied
`2026-10-15_21-33-33.png` from the original safetensors with seed
**572241167616859**. Its full prompt/workflow/checksums are retained under
`same_seed_original` in the render report. All chunk CRCs validate. Compared with
Python Q5, only nodes 108/115 (loader replacement) and node 113's model connection
differ; the seed, prompts, sampler, scheduler, resolution, encoder, VAE and both
LoRAs match.

Visual assessment against this same-seed original: the face/subject appearance,
black leather clothing, green/orange lighting and gas-station/car scene remain
similar and coherent, without obvious mosaic artifacts or gross corruption.
However, the Q5 pair raises the gun/arm, changes a wider stance to crossed legs,
and changes car framing/details and ceiling-light positions. These are meaningful
pose/composition differences, beyond texture or lighting alone. Classify this
pair as **Visible drift**, with usable-looking output in this scene; do not claim
near-identical original fidelity from this example.

Python/native remain pixel-identical, so this observed drift is shared by both
exports, not evidence of an extra Python encoder error. This end-to-end comparison
also changes the loader path with two active LoRAs; it cannot isolate quantization
math from loader/LoRA execution differences. No new render was initiated here.
Saved workflows reference model filenames but do not fingerprint the loaded
weight files or pin the loader implementation version.

## Limits and next gate

This establishes complete numeric parity for one prepared FLUX checkpoint on
Windows CPU and pixel-identical Python/native Q5 inference in one supplied
workflow with two LoRAs. A same-seed original comparison shows Visible drift in
pose/composition. Broader quality, isolated loader/LoRA attribution, forced offload,
Linux execution, other architectures, peak memory and importance matrices remain
untested. Q5 is available through the explicit experimental Workbench/API
`python_flux` backend; native remains the default. The Workbench shows the
observed Q5 drift next to the converter choice. Synthetic real-preparation/API
tests cover conversion, native-tool avoidance, cancellation, write failures and
intermediate cleanup. No additional complete 9B app conversion was run after
integration.
