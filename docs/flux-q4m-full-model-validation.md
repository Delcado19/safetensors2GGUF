# FLUX Q4_K_M full-model numeric validation

Date: 2026-10-15. Branch: `research/unified-converter-inventory`.
Implementation commit: `110d06d36bf3c526329a7a24e92bf17b0ffe2666`.

## Result

**PASS: all 201 tensors match the pinned City96-patched b3962 reference in name,
type, shape and complete payload bytes.** Typed metadata and effective alignment
match. The complete FLUX.2 klein Base 9B output contains 99 Q4_K, 16 Q5_K and
86 F32 tensors. Evidence:
[`data/flux-q4m-full-model-comparison.json`](../data/flux-q4m-full-model-comparison.json).

This extends the schematic [Q4 preset tests](flux-q4-research-path.md) to the
complete prepared checkpoint. It validates Q4_K_M only at full-model scale;
Q4_K_S retains its separate schematic evidence, not a full-checkpoint claim.
No ComfyUI render was started by the agent; user-supplied runtime evidence is
recorded below.

## Input and source protection

Both conversions use the same previously prepared F16/F32 input from the
[Q6 full-model experiment](flux-q6-full-model-validation.md):
`I:/safetensors2GGUF-flux-q6-full-2026-10-15/prepared-F16.gguf`.
Its complete SHA256 was freshly verified before starting:
`dfb03aff891ec8f1cfdd8b4ea9cd4f7500053903155f719718fe7c1a0f8d67cc`;
size 19,939,776,800 bytes. It contains 105 F32 and 96 F16 tensors prepared from
the installed BF16 checkpoint. Original-source hashes and exact preparation
provenance remain in the Q6 report. No preparation, source copy or download was
repeated; originals and earlier Q5/Q6 artifacts remain unchanged.

Source size/mtime remained stable through both conversions and final comparison.
Free space was checked before writing. The pinned native executable, source
archive and City96 patch hashes were verified through `verified_flux_reference`.
The evidence retains these hashes, exact native command, all three encoder
source hashes, writer hash, per-tensor hashes, file hashes and timings.

## Outputs and observations

Artifacts belong to `I:/safetensors2GGUF-flux-q4m-full-2026-10-15`.
The native log and Python per-tensor result JSON are retained alongside both GGUFs.

| Output | Bytes | SHA256 |
| --- | ---: | --- |
| Python Q4_K_M | 5,560,654,144 | `3239e9be7414fd07735091ea3e83b4592913bd5253244f08a8351c37925250b0` |
| Native Q4_K_M | 5,560,654,112 | `82c406e8b6c0772ebece26d188cb6a4f38c8ac273f3325308ee0cd0b940e9470` |

The Python file is 1,128,267,776 bytes smaller than the earlier Python Q5_K_M
file from the same preparation. Its 32 extra bytes versus native are header
representation: Python explicitly writes default alignment. Tensor payloads,
typed metadata values and effective alignment are equal; whole-file hashes
are retained as provenance rather than used as a semantic equality test.

Both conversions passed on the first attempt. Python published atomically only
after full layout/payload readback; the subsequent comparison read every tensor
in bounded windows, checked typed metadata and hashed both complete outputs.
The source code was not changed during either conversion.

Native took 395.38 seconds with one worker; Python took 1,539.88 seconds including
readback and output hashing. These sequential single runs have unequal timing
scope and uncontrolled host/cache conditions; this is not a controlled speed
benchmark. Peak memory was not measured.

## Reproduce

Use fresh output paths. The exact locally verified native invocation is in the
report; no upstream download or native rebuild was required for this run.

```bash
python -m scripts.requantize_flux_q6 --source prepared-F16.gguf --output python-Q4_K_M.gguf --preset Q4_K_M
/path/to/pinned/llama-quantize prepared-F16.gguf native-Q4_K_M.gguf Q4_K_M 1
python -m scripts.compare_gguf_outputs --left python-Q4_K_M.gguf --right native-Q4_K_M.gguf --output fresh-comparison.json
```

The historical script name and Q6_K default preserve existing callers.

## User-supplied runtime comparison

The supplied folders contain the matching renders:

| Model | Saved PNG | RandomNoise seed |
| --- | --- | ---: |
| Original safetensors | `2026-10-15_21-33-33.png` | 572241167616859 |
| Python Q4_K_M | `2026-10-15_23-34-41.png` | 572241167616859 |
| Native Q4_K_M | `2026-10-15_23-32-57.png` | 572241167616859 |

Full local paths, file hashes, complete saved prompts/workflows, IHDR values and
filtered scanline hashes are retained in
[`data/flux-q4m-user-render-comparison.json`](../data/flux-q4m-user-render-comparison.json).
All PNG chunk CRCs validate. Originals remain unchanged.

**Python and native are pixel-identical:** IHDR and decompressed filtered
scanlines match (SHA256
`f3fde798a645429ff452881c2ea201046c51f838c2693707e249799dff768747`).
Both are noninterlaced 768x1024 8-bit RGB PNGs; identical filter data implies
identical reconstructed pixels. The executed prompts differ only in node 116's
GGUF filename. Both use `UnetLoaderGGUF`, er_sde, Flux2Scheduler/34 steps, CFG 4,
the same Qwen3-8B heretic FP8 encoder and FLUX.2 VAE, and two enabled rgthree
LoRAs (Lenovo UltraReal and NiceGirls UltraReal) at 0.7 each.

The original shares seed and all other executed settings. Only the loader
replacement (108/116) and model connection in LoRA node 113 differ. Visual
assessment: similar subject appearance, black leather clothing, green/orange
lighting and gas-station/car scene; output remains coherent without obvious
mosaic corruption. However, arm/gun direction, crossed versus wider leg stance,
car framing/details and ceiling-light positions change noticeably. Classify this
comparison as **Visible drift in pose/composition**, not near-identical fidelity.
Python/native show the same result, providing no evidence of added Python error.
As with Q5, the end-to-end original-versus-GGUF comparison includes different
loader/LoRA execution paths; their contribution is not isolated. Saved workflows
reference filenames without fingerprinting loaded weights or pinning loader code.

## Follow-up with both LoRAs disabled

The user supplied a second triplet: original `2026-10-15_23-40-10.png`, native
`2026-10-15_23-41-53.png` and Python `2026-10-15_23-42-43.png`. Complete saved
prompts/workflows, file hashes, IHDR, scanline hashes and all-chunk CRC validation
are retained in
[`data/flux-q4m-user-render-no-lora.json`](../data/flux-q4m-user-render-no-lora.json).
Both rgthree entries explicitly have `on: false` in all three executed prompts;
nonzero retained strength widgets do not mean the entries are enabled.

Python/native are again pixel-identical, with matching IHDR and filtered-scanline
SHA256 `8ab0db73fcf6f721beb903643e1f899cb9d9313fa175f0355c05310b76afccaa`.
The saved seed and all generation settings match across the three. Besides the
model/loader replacement and connection, original versus Python also changes
node 112's downstream save mode from Preview to Save. All other save-node inputs
and its image source match; this is outside the sampler graph.

Visually, frontal stance and scene structure stay closer in this pair than in
the enabled-LoRA pair. Subject appearance, black leather outfit, gas station/car
and green/orange palette remain similar and coherent. Differences remain in
face/expression, arm/gun direction, car proportions/details, overhead lights and
flare distribution. There is no obvious mosaic corruption or catastrophic
quality failure. These visible differences persist without LoRAs: active LoRAs
are not required for original-versus-Q4 divergence. This single pair does not
measure how much LoRAs amplify drift, nor isolate quantization from preparation,
loader precision/execution or repeat-run nondeterminism.

## Limits and next gate

Numeric parity and pixel-identical Python/native rendering are established for
one complete prepared checkpoint and supplied workflows with both LoRAs enabled
and disabled. Original-versus-Q4 differences persist without active LoRAs.
Broader Q4 quality, isolated loader/LoRA attribution, forced offload, Linux,
other architectures and importance matrices remain untested. Q4_K_S still has
schematic evidence only. Q4 remains a research file path; the experimental
Workbench/API Python route still accepts Q6_K/Q5_K_M only. Native Q4 defaults and
the Krea allowlist remain unchanged. Next assess a narrowly scoped explicit
Q4_K_M app option with the observed drift clearly documented.
