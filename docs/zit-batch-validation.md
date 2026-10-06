# Z-Image Turbo single-pass batch validation

Date: 2026-10-06. Final scope: the six Untested Safetensors/prototype cells and
**one representative GGUF, Q4_K_M**, as explicitly requested by the user.
All seven rendered successfully with visible differences from the BF16 reference.
This is one checkpoint/prompt/workflow, not a universal quality guarantee.

## Preserved workflow and inputs

Source workflow: `G:/ComfyUI-Easy-Install/ComfyUI/user/default/workflows/Z-Image Turbo/Z-Image Turbo - T2I - Seed Variance Enhancer.json`.
SHA-256: `c86cb3051f040d6afe15541fa29bd61b141c2d1d679d200e77580c49446230f8`.
An isolated working copy retained all inference values and both active LoRAs.
Only prompt, model/necessary loader reference and output saving changed.
The workflow's JibMix file was already FP8, so the existing unquantized
`z_image_turbo_bf16.safetensors` (12,309,866,400 bytes) provided the reference.

- 1120x1440, one KSampler, eight `dpmpp_sde` / `ddim_uniform` steps, CFG 1,
  denoise 1, AuraFlow shift 3.5, sampler seed 852571988447841.
- SeedVarianceEnhancer: 50%, strength 20, beginning-step insertion at 20%,
  seed 363294462245669, mask beginning/0%; saved seed values fixed across runs.
- Both saved LoRAs retained at strength 0.8: BRKN BDSM and Deepthroat - K3nk.
  Runtime observed 208 attached model patches.
- Saved Josiefied Qwen3-4B v1 encoder (`lumina2`, CPU) and UltraFlux v1 VAE.
- Prompt: photographic still life of a matte teal ceramic teapot, a red apple
  and a folded cream linen napkin on a wooden table, soft window light, plain
  warm gray background, sharp realistic details, no people.
- ComfyUI 0.38.0 / Torch 2.9.1+cu130 / Kitchen 0.2.36, RTX 5080 16 GiB;
  isolated profile with copied city96/rgthree/SeedVarianceEnhancer nodes.
- Final runs used 3 GiB reserved VRAM. The BF16 source repeat was pixel-identical,
  including across the reserve change. No two-pass/upscale branch was used.

## Results

| Format | Bytes | Savings vs BF16 | Prompt ID | Client seconds |
| --- | ---: | ---: | --- | ---: |
| FP8 | 6,167,959,452 | 49.89% | 473d95f9-d977-491f-b7f3-6d37226124d9 | 24.07 |
| FP8_MIXED | 6,883,529,712 | 44.08% | d51f7745-c51b-4b13-81e7-bc7d342cefb7 | 20.22 |
| INT8 | 6,175,344,484 | 49.83% | 9933d945-50b0-4b7b-8fab-6eaaf9b5f28e | 14.15 |
| INT8_MIXED | 6,890,136,072 | 44.03% | 8b81394a-71a7-46d2-b817-1d688b3cc6d1 | 16.02 |
| NVFP4 | 3,479,984,484 | 71.73% | 7d2e79ab-5028-4df7-b386-39a229288f78 | 22.02 |
| INT4_CONVROT_MIXED | 4,176,950,776 | 66.07% | f3252ceb-0ea6-4ed7-af5d-a598012e848b | 16.02 |
| Q4_K_M | 6,436,497,472 | 47.71% | c8548b83-df82-4412-982c-616fb860455a | 38.03 |

All images retain the requested objects. FP8/INT8 variants visibly change object
placement, cloth folds and wood detail; NVFP4 and native INT4 show stronger
framing/background/material changes. Q4_K_M remains recognizable with object and
cloth/background differences. These are observed full-chain deviations, including
loader/compute/LoRA effects, not an isolated numerical quantizer score.
The Workbench Turbo row records Visible drift for these formats. Base/default
classifications and unrelated model evidence remain unchanged.

[Reference](images/zit-validation/source.png).
[FP8](images/zit-validation/FP8.png).
[FP8_MIXED](images/zit-validation/FP8_MIXED.png).
[INT8](images/zit-validation/INT8.png).
[INT8_MIXED](images/zit-validation/INT8_MIXED.png).
[NVFP4](images/zit-validation/NVFP4.png).
[INT4_CONVROT_MIXED](images/zit-validation/INT4_CONVROT_MIXED.png).
[Q4_K_M](images/zit-validation/Q4_K_M.png).

INT4 is a test-only native prototype: 180 eligible Linear weights, Kitchen
`TensorCoreConvRotW4A4Layout`, group rotation 256, layout group 64, packed I8
[N,K/2] with F32 [N] scales, mixed Turbo protections. It remains nonselectable
until production/version integration; this test does not prove forced offload.
Safetensors used the existing converter with `z_image_turbo`; GGUF used the
existing patched llama-quantize build 3962 (c8c07d658) and stock city96 loader.

## Scope correction and harness issues

The initial batch incorrectly expanded the grouped GGUF cell into all ten
precisions. The user corrected this to one Q4 test. No further GGUF tests were
started after that correction. Those surplus conversion files are discarded;
only Q4_K_M enters the compatibility evidence in this report.
An incidental F32 render took about 1,040 seconds with default memory headroom
and 160 seconds with 3 GiB reserved VRAM, producing pixel-identical images.
It is an auxiliary resource observation, not a requested additional precision
validation or a controlled format-speed benchmark.

UTF-8 fixed a console log error; already published float files were checked
before continuing. Later interrupted tool pipes caused a tqdm stdout OSError,
so the sole Q4 retry used persistent file logs and succeeded. Neither error was
classified as a model/quantization failure.

Original model/workflow/install remain intact. Retain these small image records;
remove owned test model files, copied nodes and the isolated server after tests.
No Linux validation or other-checkpoint/seed claims are made.
