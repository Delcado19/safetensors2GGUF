# Quantization Size and ComfyUI Compatibility Audit

Audited on 2026-10-05 against repository commit `65b0660`, current primary
upstream sources, and the installed ComfyUI 0.38.0 / comfy-kitchen 0.2.36 /
PyTorch 2.9.1+cu130 environment on an RTX 5080. No quantization policy or
algorithm was changed by this audit.

## Size findings

The concern is confirmed. The main excess comes from layer selection and
retained floating-point dtypes, rather than the storage cost of ConvRot.

The following are planned **tensor payload sizes**, in GiB (2^30 bytes),
computed from local checkpoint headers with the writer's actual rules:
`load_state_dict`, `_scan_quantized_layers`, `_iter_output_keys`,
`dequantized_shape_of`, and `plan_tensor_output`. Source quantization sidecars
are excluded and reconstructed weights are accounted for as F32, exactly as
the writer does. Header/JSON overhead is omitted; these are not newly generated
multi-GB checkpoints or measured render-quality results.

| Local source | Source | FP8 | FP8 mixed | INT8/ConvRot | INT8 mixed | NVFP4 | NVFP4 mixed |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen Image Edit 2511 FP8 | 19.044 | 19.043 | 28.648 | 19.064 | 28.664 | 10.721 | 21.726 |
| Z-Image Turbo BF16 | 11.464 | 5.744 | 8.169 | 5.751 | 8.171 | 3.241 | 6.727 |
| Flux.2 klein Base 9B BF16 | 16.910 | 8.502 | 9.285 | 8.509 | 9.291 | 4.824 | 5.949 |

Sources are the installed `qwen_image_edit_2511_fp8.safetensors`,
`z_image_turbo_bf16.safetensors`, and `flux-2-klein-base-9b.safetensors` under
`G:/ComfyUI-Easy-Install/ComfyUI/models/diffusion_models`.

### Reconstructed F32 retention

`convert_safetensors.py` reconstructs already-quantized weights as F32 before
both planning and writing. `quantize_tensor_st` then preserves that dtype for
mixed protected weights. The comment about preserving the original dtype
there refers to the reconstructed tensor's dtype, not the checkpoint's
pre-quantization BF16/F16 dtype, which cannot be recovered from FP8 bytes.

Qwen's current mixed output retains 12.828 GiB of float payload. Approximately
12.657 GiB is attributed to `img_mod.1` alone. Promoting approximate FP8 values
to F32 does not recover the precision lost when the source was quantized.
Selecting BF16 storage for reconstructed protected weights would reduce this
overhead, but entails another rounding step and requires explicit policy and
render validation. Genuine F32 source parameters should not be indiscriminately
downcast.

A file-backed small-source reproduction confirmed the behavior: a 33,456-byte
FP8 checkpoint becomes 131,588 bytes for each mixed target because the protected
128 x 256 weight is written as F32. This is actual writer output, not an estimate.

### Variant-specific layer selection

`ModelQwenImage.keys_hiprec` unions the 2511 and 2512 community blacklists,
including every `img_mod.1`. The current
[Kitchen converter](https://github.com/tritant/ComfyUI_Kitchen_nvfp4_Converter/blob/main/convert_to_nvfp4_node.py)
excludes these layers for 2512, but not 2511. The official
[2511 INT8 recipe](https://github.com/Comfy-Org/comfy-quants/blob/main/configs/qwen_image_edit_2511_int8_tensorwise.yaml)
and [2511 NVFP4 recipe](https://github.com/Comfy-Org/comfy-quants/blob/main/configs/qwen_image_edit_2511_nvfp4.yaml)
quantize image/text modulation, except image modulation in block zero.

Removing only the broad `img_mod.1` exclusion, leaving all other current
writer rules unchanged, plans Qwen FP8 mixed at 19.156 GiB, INT8 mixed at
19.176 GiB and NVFP4 mixed at 10.850 GiB. This is a policy simulation, not
implementation parity with the upstream converter and not a quality approval.

`ModelLumina2` uses the community Z-Image **Base** blacklist for Turbo too,
protecting all attention and AdaLN modulation. The current community Turbo
profile excludes embedders/refiners/final output, but not all these transformer
weights. Removing those two broad exclusions in the simulation plans Turbo
INT8 mixed at 6.417 GiB and NVFP4 mixed at 4.200 GiB. Existing render evidence
motivated the conservative profile; do not remove it without matched renders.

### The displayed estimate is wrong for quantized sources

`estimate_safetensors_output_size` plans raw header tensors without the writer's
source dequantization and sidecar filtering. FP8 source dtypes fail the mixed
high-precision predicate, whereas reconstructed F32 weights pass it. For Qwen
INT8 mixed it displays approximately 19.064 GiB versus 28.664 GiB planned by
the writer; NVFP4 mixed displays 10.721 GiB versus 21.726 GiB. The small
reproduction similarly estimates 33,296 bytes for INT8 mixed but writes
131,588 bytes including its header. The planner must use the same logical
source tensors, shapes and effective dtypes as the streaming writer.

### ConvRot does not explain the excess

The project stores INT8 weights at one byte per parameter, with F32 scales per
output row when rotated. It stores no dense rotation matrix. Qwen's complete
plain INT8 scale payload is 21.153 MiB, approximately 0.11% of weight payload;
mixed scales are 16.875 MiB. Changing to a different INT8 converter cannot
provide 4-bit storage without changing the format.

FP8 to INT8 is an 8-bit-to-8-bit conversion: little size reduction is expected.
Compare like-for-like components, source precision and retained-layer policies.
The official [Qwen model listing](https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/tree/main/split_files/diffusion_models)
lists the 2511 BF16 file at 40.9 GB and FP8 mixed / INT8 ConvRot at 20.5 GB
(decimal units). These are different precision baselines, not evidence of
INT8 compressing an already-FP8 source by another factor of two.

NVFP4 uses two packed values per byte plus one FP8 block scale per 16 values:
approximately 0.5625 bytes per quantized parameter before alignment and scalar
metadata. Its required block-scale payload is real storage, not accidental
duplication. See the official [NVFP4 format](https://github.com/Comfy-Org/comfy-quants/blob/main/docs/formats/nvfp4.md).

## Compatibility and numerical checks

Small real CUDA forwards through the installed `comfy.ops.mixed_precision_ops`
Linear passed for the project's FP8, INT8/ConvRot and NVFP4 tensors. Each was
loaded with its actual scales/marker and tested with full-precision matmul
enabled and disabled. All six paths produced finite 32 x 128 outputs. These
checks establish loader/kernel interoperability, not full-model image quality.

- ConvRot INT8 uses the same rotation and scales as installed Kitchen. On a
  seeded 128 x 256 test, F32 codes matched exactly; BF16 differed in 3,144 of
  32,768 codes and F16 in 389. The project divides in F32, whereas Kitchen's
  current recipe divides in weight dtype. This changes numerical parity, not
  file size. Reference: [official INT8 format](https://github.com/Comfy-Org/comfy-quants/blob/main/docs/formats/int8_tensorwise.md).
- NVFP4 block scales matched Kitchen. Most byte differences were signed-zero
  encodings, which decode identically. After normalizing signed zero, the
  three tests differed in 0, 1 and 3 codes. Exact midpoint tests showed the
  project's nearest-codebook `argmin` chooses the earlier code, unlike Kitchen's
  round-to-nearest-even. This should be corrected for numerical parity, not
  advertised as a file-size optimization. Reference:
  [Kitchen eager quantization](https://github.com/Comfy-Org/comfy-kitchen/blob/main/comfy_kitchen/backends/eager/quantization.py).
- The earlier statement that native INT8 never quantizes activations is
  incorrect on this installation. `quantize_input=False` skips wrapping inputs
  in `MixedPrecisionOps`; the dispatched Kitchen `int8_linear` still dynamically
  quantizes activations, and rotates them for ConvRot. Full-precision matmul
  explicitly enables a dequantized alternative. README and the INT8 module
  description were corrected. This distinction matters for quality and
  performance, but does not alter checkpoint size.
- Installed ComfyUI also registers `convrot_w4a4`. A small Kitchen-produced
  INT4 ConvRot weight passed its native ComfyUI CUDA forward: 16,384 weight bytes
  plus 512 scale bytes versus 65,536 source BF16 bytes. This is a genuine 4-bit
  candidate, separate from the project's current INT8 option. It has no full
  diffusion-model render approval here. Reference:
  [Kitchen ConvRot W4A4 layout](https://github.com/Comfy-Org/comfy-kitchen/blob/main/comfy_kitchen/tensor/convrot_w4a4.py).

## Implementation priorities

1. Correct source-aware size estimates and disclose quantized/retained payload
   proportions before conversion; distinguish size savings from compute changes.
2. Define storage dtype for reconstructed protected weights without pretending
   to recover their lost precision; preserve genuine high-precision inputs.
3. Separate Qwen 2511/2512 and Z-Image Base/Turbo precision policies. Existing
   conservative profiles remain until fixed-seed source/output renders validate
   narrower selection across multiple prompts/reference images.
4. Align INT8 and NVFP4 rounding with installed Kitchen, and correct runtime
   activation-quantization descriptions.
5. Evaluate native INT4 ConvRot as a separate, version-gated target using
   Kitchen's existing packer and loader contract. Packed shape/detection,
   low-VRAM offload and LoRA paths require full-model tests before release.

Do not remove convolution or raw-parameter fallbacks merely to make outputs
smaller. Their ComfyUI loading behavior must be checked independently. Do not
equate native loader support, finite small-layer output, and visual equivalence.


## Implemented corrections and runtime evidence (2026-10-05)

The historical measurements above describe the old writer. The current writer
and GUI now share a source-aware plan. Consumed source scale/metadata tensors
are removed, prefixes are normalized consistently, and packed source shapes are
expanded before planning. Protected reconstructed weights are stored as BF16;
genuine F32/F16/BF16 tensors retain their precision. Estimates exclude JSON
header bytes and expose quantized/retained payload shares plus scale overhead.

Qwen Edit 2511's marker selects the narrower policy automatically: only block
zero's image modulation remains protected, following the official Comfy Quants
2511 recipe linked above. Unmarked Qwen variants retain conservative protection.
Z-Image Base/Turbo cannot be distinguished by the common tensor signatures;
Turbo requires an explicit selection, and auto remains conservative.

| Source / target | Old planned output (GiB) | New output (GiB) | Measurement |
|---|---:|---:|---|
| Qwen 2511 FP8 mixed | 28.648 | 19.133 | New source-aware payload plan |
| Qwen 2511 INT8 mixed | 28.664 | 19.154 | Actual file: 20,566,767,496 bytes |
| Qwen 2511 NVFP4 mixed | 21.726 | 10.851 | Actual file: 11,651,454,724 bytes |
| Z-Image Turbo NVFP4 mixed | 6.727 | 4.200 | Actual file: 4,509,509,064 bytes |

Qwen NVFP4 is 43.0% smaller than the installed FP8 source (19.044 GiB).
Its new payload estimate is exactly 11,650,909,724 bytes; the remaining 545,000
bytes are the JSON header. INT8 output remains 0.58% larger than this FP8 source:
fixing mixed precision overhead cannot halve weights that already use eight bits.
The UI explicitly flags savings below 5%.

### Numerical and native-loader checks

With installed Kitchen 0.2.36, INT8 ConvRot quantized codes and scales matched
exactly for all 32,768 tested weights at F32, BF16 and F16. NVFP4 packed bytes and
block scales also matched for the tested tensors. INT8 now divides in the source
dtype with an underflow guard; NVFP4 encodes nearest-even ties and signed zero.
The encoder uses bucket boundaries instead of allocating sixteen candidate
differences per value. Six ComfyUI MixedPrecisionOps CUDA forwards passed
(FP8, INT8, NVFP4, each with full-precision matmul enabled and disabled).
These small checks establish packing/kernel compatibility, not universal model
quality or calibration equivalence to community checkpoints.

### Full-model fixed-seed comparison

Existing ComfyUI 0.38.0, Torch 2.9.1+cu130, Kitchen 0.2.36, RTX 5080 16 GiB.
All ten prompts completed successfully. Preview outputs, workflows and complete
API histories are retained locally in ignored `.pytest-quant-runtime-tmp/`;
no generated files were placed in Downloads. Source checkpoints were preserved.

Qwen: seed 1212121, 512x512, 20 Euler/simple steps, CFG 3, shift 3.1, denoise 1,
identical source FP8 text encoder and VAE, no LoRA. Two reference images (`Cat.jpeg`
and `Dog.jpeg`) requested a red knitted hat and a blue scarf respectively.

| Diffusion format | Cat prompt ID | Dog prompt ID |
|---|---|---|
| Original FP8 | `de528a38-6c1f-4e6e-b05d-18126749b5b3` | `c65d23e5-e118-4a47-ab4b-d1c30f7f75b4` |
| New NVFP4 mixed | `6381c758-fb04-4015-8ec5-b797f3da7568` | `578c885e-b57f-4726-8fa3-5105b162ccab` |
| New INT8 mixed / ConvRot | `702bbe3d-4a3d-4f8e-b85b-610a3c5e3858` | `f0385f04-86c6-4741-8d4e-bdf6a492f46f` |

Visual inspection: both edits were applied, subject and framing remained close
to the source renders. NVFP4 changed fine details (eyes, whiskers and scarf);
INT8 also showed small detail deviations. This is limited smoke-test evidence,
not a general claim of lossless output or LoRA compatibility.

Z-Image Turbo: seed 1212121, 512x512, nine Euler/simple steps, CFG 1, shift 3,
Vanilla Qwen3-4B text encoder and standard `ae.safetensors` VAE. Two text prompts
requested the same cat/hat and dog/scarf subjects.

| Diffusion format | Cat prompt ID | Dog prompt ID |
|---|---|---|
| Original BF16 | `82d8585b-68b8-4e0b-981b-382a2a41755b` | `8cdc40fc-83bd-471f-acdb-751b170c3ba2` |
| New NVFP4 mixed / Turbo profile | `2ae753b7-a4fd-4e3e-a215-188b4aa7643d` | `4ebe3008-fe24-43fe-8fdc-e5f651e2a185` |

Visual inspection: correct recognizable subjects, accessories and colors; hat,
facial details and slight framing differences remained. FP8/INT8 with the narrower
Turbo profile still lack equivalent full-model evidence, so the selectable
Turbo profile stays explicitly experimental rather than changing Base defaults.

Native INT4 ConvRot now has full-model SDXL evidence: nine successful renders
cover baseline/INT4, model-only and full LoRA, classic full CPU offload, and
dynamic offload with 1,769 observed native INT4 requantizations. The 591-layer
checkpoint is 2.00 GiB versus 4.78 GiB source UNet weights. See
[the full validation report](int4-convrot-validation.md) for exact packing,
prompt IDs, visual differences and cleanup. This supersedes the earlier
single-layer-only status; it does not establish cross-family quality or add
an INT4 entry to the production format dropdown.

A transient Windows sharing violation occurred while unlinking Qwen NVFP4's
completed temporary hardlink. The final file had already been published; its
header, offsets and full payload were verified before rendering, and the leftover
link was removed. Atomic-output cleanup now retries only Windows sharing
violations for up to one second and still propagates other errors.
