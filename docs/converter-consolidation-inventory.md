# Unified converter: frozen source inventory

Date: 2026-10-12. Branch: `research/unified-converter-inventory`.
Status: source preservation and static inventory complete. Effective declaration
resolution and initial small encoder probes were added on 2026-10-15; see
[the reference analysis](converter-reference-analysis.md). A subsequent
[native block comparison](converter-native-reference-comparison.md) found identical
bytes across all three llama.cpp refs for 95 valid probes each. Complete converter
parity, backend replacement and hardware optimization remain open.
The subsequent [policy probes](converter-policy-comparison.md) record 84 real
quantizer executions on schematic inputs and 41 installed-model header summaries.
The first [shared Q6_K math pilot](shared-q6-pilot.md) matches the three native
references on 166 inputs and three complete streamed matrices; it remains
separate from production dispatch. Full-tensor checks compare every chunk with
the native references and publish raw research artifacts atomically.
An isolated [compiled CPU kernel pilot](compiled-kernel-pilot.md) now calls pinned
b3962 Q4_K/Q5_K/Q6_K kernels directly from Python. All 498 golden inputs match;
bounded-call slice timings are within about 2% of the same directly called C
kernel. Full-model writer integration, packaging and non-Windows execution remain
open; this does not replace the production backends or settle the final language.

## Objective and boundaries

Build one project-controlled conversion implementation with shared quantization
algorithms, explicit precision policies, model adapters and format writers.
Implementation language is undecided. CPU is the baseline; NVIDIA, AMD and Intel
Arc acceleration are later execution backends, subject to measured benefit and
platform availability. Tensor cores are relevant to eligible matrix operations,
not automatically to rounding, scale search or bit packing.

GGUF and safetensors loader development is a separate companion workstream.
Whether ComfyUI exposes one combined loader or separate loader nodes remains
open. Existing workflows and validated conversion paths must remain usable.

## Frozen references

Full upstream tracked-source archives and root licenses are preserved in
[`third_party/converter_references`](../third_party/converter_references/README.md).
The machine-readable manifest contains authoritative full commit IDs and hashes.

| Source | Revision | Role |
| --- | --- | --- |
| llama.cpp b3962 | `c8c07d658a6cefc5a50cfdf6be7d726503612303` | Diffusion quantizer base, 2024-10-22 |
| City96 ComfyUI-GGUF | `6ea2651e7df66d7585f6ffee804b20e92fb38b8a` | Diffusion tensor rules and custom llama.cpp patch |
| Existing text llama.cpp | `687e7789271ec1276e3470f158428e11a4f80b6f` | Actual local checkout, 2026-08-09 |
| Current llama.cpp | `8bb3cf0d216a5ae48360179f2931daf284be4d01` | New upstream comparison, 2026-10-12 |
| Current Molbal | `b77b9d8124656aa1a548d95c617da4d16cab1a2e` | Additional adapters, Python K encoders and loader rules |
| b3962 Kompute submodule | `4565194ed7c32d1d2efa32ceab4d3c6cae006306` | Complete separately pinned optional GPU source |

Production `third_party/molbal` remains at `5a0a3ff`. The current Molbal reference
is not automatically used by Krea jobs. Neither llama.cpp reference changes the
existing text-encoder cache or its first-use network behavior.

## Responsibilities to consolidate

| Layer | Existing project / City96 path | Molbal reference | Consolidation decision |
| --- | --- | --- | --- |
| Source decoding | Safetensors/CKPT, native-quant reconstruction | Scaled INT8/ConvRot metadata reconstruction, streamed Safetensors | Preserve reconstruction before re-quantization; reject unsupported packed layouts |
| Diffusion adapters | `models/architectures.py`, shape handling, protected keys | `Model*` classes in `tools/convert.py` | Merge explicit rules, retain transformations as adapter code |
| Text adapters | Config/tokenizer staging then llama.cpp converter | Selected Qwen/MiniMax paths | Preserve full config/tokenizer/projector semantics; do not infer export support from loader support |
| Precision selection | Mandatory fallbacks, architecture protection, mixed profiles; City96 C++ preset rules | `plan_k_quantization`, target-size planning, `keys_noquant`/`keys_hiprec` | Separate mathematical encoders from model-level S/M and mixed policies |
| Quantization math | gguf-py / PyTorch and two llama-quantize variants | Standard gguf-py plus local K encoders and CR transformations | Share one encoder per exact format; compare rounding and optimization before replacement |
| Output | Atomic GGUF and safetensors publication, validated shape/metadata | Upstream GGUF output wrapped by our Krea adapter | Keep common cancellation, validation, overwrite and atomic-publish contracts |
| Runtime | ComfyUI native safetensors and City96 loaders | Additional architectures, CR/Dynamic VRAM and experimental K decoding | Separate storage encoding from runtime kernels; require loader-specific proof |

## Format coverage and gaps

| Family | Current application | Molbal Python reference | Work required |
| --- | --- | --- | --- |
| Float GGUF | F32/F16/BF16 | F16/BF16 advertised map | Preserve all current float types |
| Standard GGUF | Q8_0; Krea also Q4_0/Q4_1/Q5_0/Q5_1 | Q8_0/Q4_0/Q4_1/Q5_0/Q5_1 | Establish encoder reference vectors |
| K GGUF | Q2_K/Q3_K_M/Q4_K_S/Q4_K_M/Q5_K_M/Q6_K | Q4_K/Q5_K/Q6_K plus Q4_K_S/M and Q5_K_S/M | Q2_K/Q3_K encoder gap; compare all preset rules rather than just output labels |
| IQ GGUF | Not offered | Not in the converter's explicit quantization map | Future option, not current required coverage; decoding support is not an encoder |
| Native safetensors | FP16 and mixed; FP8/INT8/NVFP4 and mixed; diffusion INT4 ConvRot mixed | No corresponding safetensors export path | Keep native math, metadata and source precision protection |
| CR GGUF | Not offered by our Krea allowlist | Q8_CR/Q4_CR/Q4_CR_W4A4 | Distinct storage/runtime contracts; not aliases for standard Q8_0/Q4_0 |

Molbal K encoder entrypoints are `quantize_k`, `_quantize_k_affine` and
`_quantize_k_q6`. Source CR math uses `quantize_int8_convrot` and
`quantize_int4_cr_w4a4`. Target-size and LoRA fusion are separate optional
features, not prerequisites for a shared mathematical core.

## Models and special rules

The generated [JSON inventory](../data/converter-source-inventory.json) records
model class names, source lines, base classes, registrations, declared tensor
patterns and transformation method names. It does not execute third-party code
or resolve inheritance. An empty declared rule does not imply no inherited rule.

Initial static counts: City96 has 12 model declarations including its template;
Molbal has 22 including its template. Old llama.cpp has 49 registered/declarative
classes; the installed and current text references have 200 and 247 respectively.
These are code declaration counts, not counts of compatible models or validated
checkpoints. Modern llama.cpp adapters live in `conversion/*.py`; preserving just
the `convert_hf_to_gguf.py` CLI would miss those implementations.

Prioritize diffusion Krea, Ideogram, Qwen Image 2.1 and MiniMax adapters alongside
existing FLUX/SD/Wan/LTX families. For text encoders, compare only the application's
offered families first, then optional new families. Protection patterns differ:
Molbal's Krea rules are anchored and must not be silently substituted for our
broader substring policies. Qwen3-VL's special Q8_CR export drops the vision tower;
it is not a full multimodal replacement for the llama.cpp path.

## Validation gates and next work

Completed:

- Hash/file-count/license verification for all six archives, with path/link checks.
- Preservation of the b3962 gitlink source; no other gitlinks found at these refs.
- Read-only City96 patch compatibility check against archived b3962 files.
- Offline static model/format inventory and corruption/parser regression tests.

Next, before replacing a backend:

The first architecture-specific [FLUX Q6_K research path](flux-q6-research-path.md)
now passes eight schematic native-versus-Python file comparisons (196 tensors).
It preserves the existing float-GGUF boundary and remains outside production
dispatch; this is not full-model or loader validation.
The subsequent [full FLUX checkpoint comparison](flux-q6-full-model-validation.md)
passes all 201 tensor payload/type/shape comparisons and metadata for one installed
9B checkpoint. A subsequent user-generated PNG comparison confirms inference
with two enabled LoRAs for this checkpoint and matching saved settings. The
full-model guide retains evidence and the concrete service integration boundary;
offload, Linux and broader model coverage remain open. An opt-in Workbench/API
`python_flux_q6` backend now integrates the existing F16 preparation with this
writer; default/native routes and format registries remain unchanged.

The subsequent [Q5_K block pilot](shared-q5-pilot.md) matches all three native
references in 498 comparisons and 332 chunk-size checks. Q5_K_M model policies
and full-model integration remain separate gates.

1. Extract explicit effective rules for the supported model families, including
   inherited rules, tensor transformations and C++ preset choices.
   Static inheritance/method ownership and policy source spans are now recorded;
   executable policy-decision parity remains open.
2. Build small encoder/dequantizer vectors for zero, constant, outlier, aligned
   and incompatible-shape tensors. Specify rounding, byte layout and fallback
   behavior; equality claims need measured results.
   The initial 96-case probe report exists. Molbal Q4_K/Q5_K clips positive-only
   ramps and must not be adopted unchanged. The subsequent 360-case native block
   comparison includes Q2_K/Q3_K; full model/policy parity remains open.
3. Compare Molbal Q4_K/Q5_K/Q6_K with patched b3962 and both text references.
   Compare S/M policies separately from each individual tensor encoding.
4. Preserve source precision, tokenizer/metadata and atomic output/cancellation
   behavior through the proposed shared interfaces.
5. Implement one common path and compare files, loading, memory and speed before
   enabling production dispatch. New model renders require an explicit request.

No new ComfyUI models or nodes were installed, no model weights were downloaded,
and no inference or native build was started. Optional SDK/download dependencies,
offline installation, native binaries, full-source algorithm parity and the
implementation-language choice remain open.

FLUX Q5_K_M research file export matches patched b3962 in eight schematic
comparisons (196 tensors) and the complete FLUX.2 klein Base 9B checkpoint
(all 201 tensor payloads/types/shapes and metadata). See [full Q5 validation](flux-q5-full-model-validation.md).
User-supplied Python/native Q5 renders are pixel-identical with two enabled LoRAs.
A subsequent original render with matching seed/settings shows Visible drift in
pose/composition, with coherent output in this scene. Broader quality remains open. Choose Q5_K_M and **Advanced settings > GGUF
converter > Python - FLUX only (experimental)** to use the opt-in backend; API
clients send `"gguf_backend": "python_flux"`. Native remains the default.

The [Q4_K block pilot](shared-q4-pilot.md) also passes 166 inputs against three native
references, sharing affine mathematics with Q5. FLUX Q4_K_S/Q4_K_M file policies
also match native on 16 schematic files (400 tensors); see [Q4 file validation](flux-q4-research-path.md).
The [complete FLUX Q4_K_M comparison](flux-q4m-full-model-validation.md) now also matches all 201 native
tensor payloads/types/shapes and metadata. Supplied Python/native Q4_K_M renders
are pixel-identical with two LoRAs; the same-seed original shows Visible drift in
pose/composition. Full Q4_K_S, broader quality and app integration remain open; native Q4 defaults are unchanged.
