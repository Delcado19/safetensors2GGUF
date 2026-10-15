# Effective rules and initial encoder probes

Date: 2026-10-15. Research branch: `research/unified-converter-inventory`.
Follow-up: [native reference block comparison](converter-native-reference-comparison.md)
now records the three native builds and 360 cases. This document preserves the
initial Python findings and their original evidence boundary.
Production dispatch, vendored production Molbal and native tools are unchanged.

## Reproduce offline

Use the project's installed environment (or sync the locked environment first).
The commands themselves need no network, weights, native compiler or ComfyUI:

```bash
python -m scripts.converter_reference_rules --output data/converter-effective-rules.json
python -m scripts.converter_reference_vectors --output data/converter-reference-vectors.json
```

Both scripts verify all frozen archive checksums and licenses first. The rules
script parses ASTs without executing upstream code. The vector script deliberately
executes only five reviewed K-math function definitions from frozen Molbal
`b77b9d8`, with NumPy and installed gguf-py supplied explicitly. It does not import
the upstream CLI or any model, loader or GPU initialization. No dependency added.

## Effective model rules

The [effective-rule report](../data/converter-effective-rules.json) resolves Python
C3 inheritance, including relative imports and aliases. It records field
expressions and their defining class/file/line; lists replace inherited lists,
they are not concatenated. Effective methods retain their defining source spans.
Unknown bases and dynamic class-body statements are explicit. Expressions,
constructors, decorators, method bodies and runtime configuration are not
evaluated. This resolves declaration ownership, not every runtime decision.

Coverage includes all City96/Molbal diffusion model-template descendants and our
local architecture classes. Text coverage follows the application's GGUF families:
T5/UMT5, Qwen3, Qwen2.5-VL, Llama and Mistral/Ministral3. `Mistral3Model` registers
`Ministral3ForCausalLM` and contains nested backbone classes, which are resolved
under qualified names; the separate `MistralModel` handles Mistral-format input. Missing
requested classes in old b3962 are reported rather than invented. CLIP-L/bigG
remain safetensors-only in our application; Gemma is not an offered family here.

Examples requiring special handling:

| Family | Effective behavior / consolidation requirement |
| --- | --- |
| Wan | Inherits `handle_nd_tensor` from Hunyuan Video; a declaration-only view misses the sidecar path for unsupported dimensions. Modulation shape corrections also live in the C++ patch. |
| SD1/SDXL | Inherit empty protection lists but enable `shape_fix`; reshape policy remains executable adapter logic. |
| Krea | Molbal's `^first.`, `^tmlp.` and related patterns are anchored prefixes, not our general substring rules. Preserve `txtfusion.projector` dimensions and protection. |
| LTX upscaler / MiniMax VAE | `preserve_nd_shapes` records original dimensions while storage combines dimensions. A generic reshape without metadata is insufficient. |
| Qwen3 | Inherits transformations from `Qwen2Model` and the text/base hierarchy; a short subclass does not mean no transformation. |
| Mistral/Ministral3 | Mistral-format input inherits across modules from `LlamaModel`; the HF `Mistral3Model` wrapper inherits `TextModel` and selects a nested backbone (including `Ministral3Model`, derived from `LlamaModel`). Preserve dispatch and constructor/config distinctions. |
| T5/UMT5 | Tokenizer/config staging and relative-position-bias protection must survive consolidation. |

The report also locates and hashes exact C++ policy spans in the frozen archives:
image selection/exclusions, text selection and, in modern llama.cpp, tensor
categories, eligibility and shape fallback. These are source locators, not a C++
interpreter or an exhaustive executable policy translation.

## Precision policies are not interchangeable

The following summarizes the inspected image rules. Non-matching tensors retain
the preset's base type; ignore/protection/eligibility rules run separately.

| Image tensor category | City96 patched b3962 | Molbal Python K planner |
| --- | --- | --- |
| Attention V, Q4_K_S | First four matches become Q5_K; later matches remain Q4_K | Every matching V becomes Q5_K |
| Attention V, Q4_K_M / Q5_K_M | Q6_K | Q6_K |
| Fused QKV, Q4_K_M / Q5_K_M | Q5_K / Q6_K | Q5_K / Q6_K, with a different category pattern set |
| FFN down, Q4_K_S / Q4_K_M / Q5_K_M | Q5_K / Q6_K / Q6_K | Same promotions for its own FFN pattern set |
| Attention output | No separate image promotion branch | Q5_K for Q4_K_S; Q6_K for Q4_K_M and Q5_K_M |
| Embedding/output | Architecture-specific exclusions may prevent quantization | Matching eligible tensors promoted to Q6_K for mixed presets |
| Incompatible K row width | F16 fallback | Planner/encoder error unless protected; SD shape-fix can make a row eligible |

City96's image V counter follows tensor processing order. Molbal sorts keys and
uses category assignments independent of insertion order. Its K-only pattern
matcher also extends trailing-dot wildcard patterns; the legacy matcher does
not. Do not normalize these rules to one glob/regex interpretation silently.

Molbal's K planner separately protects 1D/small/hiprec tensors as F32, preserves
F16/BF16 for noquant/conv weights, skips specified metadata/scalars and rejects
unprotected misaligned rows. This is not the same as the City96 C++ gate, which
only quantizes eligible 2D image weights after architecture-specific exclusions.

Text llama.cpp uses layer-aware `use_more_bits` for selected V/FFN promotions:
first/last eighth and every third middle layer. Output/tied-embedding rules,
head/expert information, explicit tensor overrides and `pure` settings also
affect selection. Modern shape fallback can use Q5_0/Q5_1/Q8_0 for Q4_K/Q5_K/Q6_K
before resorting to F16. It must not be replaced by image rules or Molbal's
uniform category planner. Complete policy parity still needs actual native runs.

## Initial mathematical evidence

The [probe report](../data/converter-reference-vectors.json) contains 96 cases:
12 input vectors across eight types. Q8_0/Q4_0/Q4_1/Q5_0/Q5_1 use installed
gguf-py encoders; Q4_K/Q5_K/Q6_K use frozen Molbal math. All use installed gguf-py
as one common decoder, with package versions and decoder checksum recorded.
These are raw tensor encoders, bypassing model policy and its small-tensor gate.

Inputs include zeros, positive/negative constants, signed and one-sided ramps,
isolated outliers, a rounding probe, one 256-value block and incompatible rows.
Input/payload hashes, byte counts, reconstructed extrema, RMSE and maximum
absolute error are recorded. Valid cases check deterministic bytes and finite,
shape-correct reconstruction. Incompatible cases must reject; unexpected errors
on aligned inputs fail the run. `encoded` means these checks succeeded, not that
the error is acceptable for model use. No S/M preset is a mathematical encoder.

For a positive ramp from 1 to 2 (two rows of 256 float32 values):

| Encoder | RMSE | Maximum absolute error | Reconstructed range |
| --- | --- | --- | --- |
| gguf-py Q8_0 | 0.003512 | 0.007904 | 1.002502 to 1.999878 |
| Molbal Q4_K | 1.468111 | 1.939336 | 0.060664 to 0.060664 |
| Molbal Q5_K | 1.468125 | 1.939351 | 0.060649 to 0.060649 |
| Molbal Q6_K | 0.014219 | 0.028687 | 0.993769 to 1.999333 |

**Blocker for directly adopting Molbal Q4_K/Q5_K math:** the one-sided positive
ramp saturates. In `_quantize_k_affine`, scale is based on `(max-min)/levels`,
while offset is `max(-min, 0)`. When a nonconstant sub-block has positive minimum,
the offset becomes zero but scale covers only the range width, so the positive
baseline cannot fit and values clip. The corresponding negative ramp has much
smaller errors (RMSE 0.001833 / 0.001555). This is a measured synthetic limitation
of the frozen implementation; it is not evidence about every model or later
upstream revisions. The production Krea allowlist does not use these K encoders.
The regression tests also inspect the packed codes directly: every Q4_K/Q5_K
code saturates at its maximum, independently of the common decoder.

Q6_K's smaller errors on these probes are not byte parity with llama.cpp, a
general quality guarantee or a performance result. FP16 stored scales introduce
rounding differences; the rounding probe alone does not establish the complete
rounding contract. The subsequent independent native comparison is linked above.

## Next comparison

1. Run these exact inputs through pinned native b3962 and both text references,
   separating pure block encoding from model/preset dispatch. Preserve source
   revisions, build flags and executable checksums with the results.
2. Compare decoded values and bytes; isolate optimizer/rounding differences from
   layout errors. Investigate the affine clipping before adopting or correcting
   that implementation. Do not change preserved upstream archives.
3. Add policy decision vectors (tensor order, names, dtype, shapes, layer counts)
   before replacing any S/M preset or fallback. Include Q2_K/Q3_K next.

The initial Python phase did not perform native builds. The subsequent native
probe phase is documented separately above. No new model renders, model downloads
or production migrations were performed. Offline dependency packaging and
language selection remain open.
