# INT4 ConvRot full-model validation

Date: 2026-10-05. Scope: native ComfyUI SDXL diffusion loading, normal inference,
classic CPU offload, dynamic VRAM offload, and LoRA patch/requantization.
This is full-model runtime evidence for one SDXL checkpoint, not an approval
for every model family or LoRA.

## Checkpoint and native packing

Source: installed `SDXL 1.0/realvisxlV50_v50LightningBakedvae.safetensors`.
Only its diffusion UNet was extracted; CLIP and VAE were loaded from the original
checkpoint for every render. Comparing against the entire checkpoint would
incorrectly count omitted text-encoder/VAE weights as quantization savings.

- Original UNet payload: 5,134,927,368 bytes (4.782 GiB).
- Converted file: 2,145,210,296 bytes (1.998 GiB), 58.22% smaller.
- Destination: `C:/Users/Delcado/Downloads/realvisxlV50-lightning-INT4_CONVROT_MIXED-test.safetensors`.
- SHA-256: `054694f23d564c997f6e612747551b8f777cd599821ac4a70f02e36bb89f9661`.
- 591 eligible Linear weights quantized using installed Kitchen's
  `TensorCoreConvRotW4A4Layout.quantize(weight)`; conversion took 25.88 seconds.
- Native metadata: `format=convrot_w4a4`, `convrot_groupsize=256`,
  `linear_dtype=int4`. Layout parameter `quant_group_size=64` is preserved.
- Actual installed storage: packed I8 `[N,K/2]` weights and **row-wise F32 `[N]`
  scales**. The layout parameter 64 does not imply a `[K/64,N]` scale matrix.
  An initial planning assertion caught that wrong assumption before publication;
  the corrected plan matched every output tensor's name, shape and dtype.

Eligibility required a two-dimensional `.weight`, K divisible by 256 and N
by 16, excluding the existing SDXL mixed-precision protection and
`keys_shape_critical` lists. All other tensors retained their source dtype;
convolution weights were not packed. In particular, `attn2.to_k.weight` and
`label_emb.0.0.weight` remained readable for ComfyUI architecture detection.
The file was streamed through the existing atomic-output/header helpers.

Reference: [Kitchen native ConvRot layout](https://github.com/Comfy-Org/comfy-kitchen/blob/main/comfy_kitchen/tensor/convrot_w4a4.py).
Tests used the installed 0.2.36 implementation; source behavior can change
between versions. No quantization dependency was installed or upgraded.

## Runtime configuration

ComfyUI 0.38.0, Torch 2.9.1+cu130, Kitchen 0.2.36, RTX 5080 16 GiB.
512x512, seed 1212121, eight Euler/simple steps, CFG 2, denoise 1.
Positive prompt: photograph of a gray tabby cat with green eyes wearing a red
knitted hat, neutral gray background, detailed fur. Negative: blurry, noise,
distorted. The final output was decoded with the original checkpoint VAE.

Normal runs used the existing idle server on port 8188. Forced-offload runs
used a separate hidden server on 8189 with custom/API nodes disabled and
`--reserve-vram 14`. The classic phase also used
`--lowvram --disable-dynamic-vram`. The dynamic phase used default dynamic VRAM
and an explicit in-memory database (`--database-url sqlite:///:memory:`).
The original running server and its launch settings were preserved.

LoRA: installed `SDXL 1.0/SDXL10 - Detail Tweaker XL.safetensors`, strength 0.6.
The first normal/classic comparisons used `LoraLoaderModelOnly`: 722 model
patches were attached, while CLIP LoRA keys were intentionally unconsumed and
logged as such. The final dynamic comparisons used full `LoraLoader` with
model and CLIP strengths both 0.6: 722 model patches and 264 CLIP patches,
with no unloaded-LoRA-key warnings in that phase.

## Results

All nine submitted prompts completed with `status_str=success`.

| Case | Prompt ID | Client elapsed seconds |
|---|---|---:|
| source | `da957572-8124-4025-ae0c-b489368fd83f` | 6.09 |
| int4 | `39fd1ed2-defc-4240-a9c7-dd69076ea788` | 6.06 |
| source-lora | `c6d13310-59e0-4fb1-9514-e44b6e906d80` | 6.60 |
| int4-lora | `537c87f0-4e55-487a-bc06-c19bd8afd6f9` | 6.10 |
| offload-int4 | `e8e64420-c670-4e01-84fe-1a401da3d16d` | 12.07 |
| offload-int4-lora | `4a57efa7-86fc-42c3-aef1-b46a1cf23a98` | 12.06 |
| dynamic-source-lora | `eefe970a-c132-4df6-93c6-f384543ec493` | 9.12 |
| dynamic-int4 | `8a80343d-8619-49c5-931c-e77b0244795e` | 6.04 |
| dynamic-int4-lora | `1791650a-b9dd-4d33-92cd-d5dcefa75627` | 9.06 |

Classic offload log: **0.00 MiB loaded / 2,045.49 MiB offloaded** for the INT4
UNet; the LoRA variant reported **722 low-VRAM patches**. Thus offload was
actually exercised rather than merely enabled by a flag.

The dynamic server wrapped `QuantizedTensor.requantize_from_float` only to
observe its returned layout and shapes; it did not change quantization math.
It recorded **1,769 successful INT4 requantizations**, all preserving rotation
size 256, layout parameter 64, `linear_dtype=int4`, and packed width K/2.
This exercises the dynamic cast/LoRA path that the earlier single-layer test
had not covered.

Visual inspection: decoded images contained the requested recognizable cat,
knitted hat and neutral background. INT4 changed ears, face, framing and knitting
relative to the source; it is not lossless. LoRA produced visible changes in both
source and INT4 outputs. Neither offload path produced black images, noise-only
outputs, shape errors or a kernel crash. The final full-LoRA comparison retained
a clear subject and accessory with visible quantization differences.

Normal INT4 versus classic offload: mean absolute pixel difference
4.320/255, PSNR 28.90 dB. Versus dynamic offload:
6.597/255, PSNR 25.15 dB. These are same-format offload
comparisons, not a perceptual quality score against the unquantized source.

## Cleanup and limits

The requested converted model remains in Downloads. Temporary model hardlink
and install junction, owned test server, preview images and test scratch files
were removed after recording results. Original checkpoints and LoRAs were not
modified. The earlier Qwen GGUF in Downloads was preserved.

Two harness startup problems were corrected during testing: the first isolated
server warned about the original server's locked default database, so subsequent
startup explicitly used an in-memory database; the observation wrapper needed
ComfyUI's CUDA allocator initialization before importing Kitchen/Torch.
Neither was an INT4 render failure. No existing server was stopped or restarted.

This establishes native compatibility for this checkpoint, selected precision
policy and LoRA on this installation. The production diffusion dropdown now
offers `INT4_CONVROT_MIXED`. The streaming writer shares eligibility rules with
size planning, preserves source dtype on fallback, and supports reconversion.
No additional ComfyUI render tests were run for this integration.
The previous blanket statement that full-model
Offload/LoRA evidence was missing is now superseded for SDXL.

## Production storage contract (2026-10-06)

The PyTorch writer reuses the existing regular Hadamard rotation. It emits
signed [-7,7] codes, low nibble first, packed I8 `[N,K/2]` plus F32 `[N]` scales.
Metadata uses `format=convrot_w4a4`, rotation 256 and `linear_dtype=int4`;
Kitchen's fixed quantization group size is 64. Unsupported group/layout metadata
is rejected on reconversion. Protected and ineligible weights retain source dtype.
FP16 zero rows receive a representable scale floor to avoid division by zero.

`scripts/check_int4_native.py` compared packing, scales and reconstruction against
installed Kitchen 0.2.37: identical for FP32, BF16 and FP16 random weights.
Streaming roundtrip tests cover payload estimation, shape-critical protection,
unaligned/conv fallback, zero weights and INT4-to-FP16 reconversion.
These are codec/converter checks, not new model-quality evidence.

The app displays runtime requirements instead of inferring support from a version
number alone: the target ComfyUI must register `convrot_w4a4` and Kitchen must
provide `TensorCoreConvRotW4A4Layout`; native compute needs NVIDIA SM 7.5+.
ComfyUI is a separate environment, so the converter cannot certify its loader
or GPU from its own Python version. Older incompatible runtimes need updating.
