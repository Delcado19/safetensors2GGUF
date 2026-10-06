# Krea 2 Q4_0 full-model render validation

Date: 2026-10-06. Scope: actual installed checkpoint, pinned molbal converter and
loader, two prompt comparisons in an isolated ComfyUI profile. All four renders
completed successfully. This does not certify every Krea model or GGUF precision.

## Source and conversion

Installed diffusion model: `Krea 2/lustifyNSFWCheckpoint_v10Krea2.safetensors`.
The source is **already INT8 ConvRot**, with file-level quantization metadata,
256 packed weights and 256 scales. It is not the original unquantized checkpoint.

The existing Safetensors converter reconstructed its rotated/scaled weights to
a temporary F16 file, then the new pinned Krea adapter exported Q4_0. Neither
the installed model nor ComfyUI custom nodes were modified.

| Artifact | Bytes |
| --- | ---: |
| Installed INT8 ConvRot source | 13,148,974,712 |
| Temporary reconstructed F16 source | 25,640,193,208 |
| Q4_0 GGUF | 8,316,720,160 |

GGUF is **36.75% smaller than the installed INT8 file**, and 67.56% smaller than
the float staging file. Compare against the actual installed source when deciding
whether conversion is worthwhile. The complete staging/export chain took 150.10
seconds; staging took 60.15 seconds. GGUF contains 430 tensors: 256 Q4_0 and 174
F32, with validated normalized names and logical shapes.

SHA-256: `d169cb9fa2591f965d2aab391a378a0bae09f39617a59c3cf0418f7321dc80de`.
Molbal converter/loader: `5a0a3ffa0e3eae5c6af8b0b981b660a24d5fbc04` from our
verified project-owned archive; upstream source was not changed.

## Runtime and matched inputs

- ComfyUI 0.38.0, commit `6b747c0428c343e1417219641db93a4fb7cb69ae`.
- Torch 2.9.1+cu130, Kitchen 0.2.36, RTX 5080 16 GiB.
- Separate loopback server on 8189, disposable base/custom-node/user/output
  directories and in-memory database; only pinned molbal custom nodes loaded.
- Encoder: installed `Krea 2/Huihui-Qwen3-4b-VL.safetensors`, CLIP type `krea2`.
- VAE: installed `WAN 2.2/wan_2.1_vae.safetensors`.
- 512×512, seed 1212121, eight Euler/simple steps, CFG 1, denoise 1.
- Negative conditioning: zeroed positive conditioning. No LoRA.
- Source uses native `UNETLoader`; GGUF uses molbal `UnetLoaderGGUF`.
  Runtime logs report BF16 compute for the source and FP16 for the default GGUF
  path. Thus differences cover the complete export/loader chain, including float
  reconstruction/casting and compute policy, not weight quantization alone.

Cat prompt: photograph of a gray tabby cat with green eyes wearing a red knitted
hat, neutral gray background, detailed fur.

Dog prompt: photograph of a golden retriever wearing a blue wool scarf, sitting
in a snowy park, natural winter light, detailed fur.

| Case | Prompt ID | ComfyUI execution seconds |
| --- | --- | ---: |
| Source cat | `13e17419-389c-4c41-ae50-b631e9b233ee` | 11.00 |
| Q4_0 cat | `3d000c00-953d-4b9e-b59d-0ae620c09ee3` | 9.34 |
| Source dog | `18d0a378-634b-4017-b4f0-72eaa8c1149e` | 7.26 |
| Q4_0 dog | `85dd9e8c-ee87-43e1-ae4b-f6e406e991b7` | 9.52 |

These include different loading/cache work and are not a controlled performance
benchmark. GGUF fully loaded at approximately 8,123 MiB; forced offload was not
tested. Dynamic VRAM was enabled in core, but this does not prove a GGUF offload
cycle or LoRA patching.

## Visual result and loader requirement

Both motifs remain recognizable, with closely matching framing and accessories.
Small relative differences affect cat eyes/muzzle/fur and dog/scarf texture.
All four images also have a shared raster/mosaic texture. This was overlooked
in the initial review; they are loading/relative-drift smoke evidence, **not an
approval of absolute image quality**. See the controlled follow-up below.
No black/noise-only images, shape failures or kernel crashes occurred. The matrix
records **Visible drift**, scoped to this checkpoint/Q4_0/molbal combination.

| Motif | Installed INT8 reference | Converted Q4_0 |
| --- | --- | --- |
| Cat | [Source](images/krea2-validation/source-cat.png) | [GGUF](images/krea2-validation/q4-cat.png) |
| Dog | [Source](images/krea2-validation/source-dog.png) | [GGUF](images/krea2-validation/q4-dog.png) |

The installed stock city96 loader at
`4dfc025e4be06ba9c210bd1189546f7af67f445d` was separately invoked with the exact
same GGUF, in an isolated Python process. It rejected the file with
`ValueError: Unexpected architecture type in GGUF file: 'krea2'`.
The molbal loader requirement is therefore tested for these exact revisions;
no shared loader installation was replaced or coinstalled.

Other quantization levels, Krea checkpoints, full-precision original comparisons,
LoRA, forced offload and Linux remain untested. Existing packed sources still
require an explicit float reconstruction step before the Krea adapter; its
packed-source rejection remains in place to prevent incorrect interpretation.

## Cleanup

The four small preview PNGs are retained as reviewable evidence. The owned float
staging/GGUF files and isolated server/profile were removed after validation.
Original models, the normal ComfyUI server and the user's earlier Downloads test
models were preserved.

## Shared mosaic artifact follow-up

The user correctly identified a common raster texture in all four images.
The original test used eight steps without verifying which Raw/Turbo variant the
installed shared filename represents. The
[official ComfyUI Krea guide](https://github.com/Comfy-Org/docs/blob/main/tutorials/image/krea/krea-2.mdx)
uses the Qwen-Image VAE, distinguishes 52-step Raw from eight-step Turbo, and
places the standard output range at 1K?2K. The installed filename/metadata does
not reliably identify the variant; do not infer Turbo from a working eight-step
render. The original Wan VAE choice was copied from a local workflow.

On 2026-10-06 a fresh isolated core-only ComfyUI profile tested the same installed
INT8 model, encoder, cat prompt and seed:

| Controlled change | Prompt ID | Observation |
| --- | --- | --- |
| Same sampled 512px latent decoded with Wan and Qwen VAEs | `1ec24d05-746c-4225-8bec-fd19cf758672` | Raster remains with both; VAE alone does not explain it. |
| Qwen VAE, 1024px, 8 steps, CFG 1 | `414888a3-29ff-432b-bbd1-37ea6b7b24bc` | Raster remains; resolution alone does not resolve it. |
| Qwen VAE, 512px, 52 steps, CFG 1 | `6ccdd232-d95b-48e9-b7be-cfe5b7532f36` | Sharper fur/hat/face and substantially reduced raster, with the step count as the isolated change. |
| Qwen VAE, 1024px, 52 steps, CFG 4 | `6fc5e651-fd80-48f1-bd06-8d10b200fd1c` | Severe oversaturation/artifacts; changing several parameters together does not establish a correct Raw recipe. |

[512px / 8 steps](images/krea2-validation/qwen-512-8.png),
[512px / 52 steps](images/krea2-validation/qwen-512-52.png),
[1024px / 8 steps](images/krea2-validation/qwen-1024-8.png).

The short denoising recipe strongly contributes to the common artifact. This
follow-up does not establish every remaining pixel pattern's mathematical cause,
identify the exact Raw/Turbo variant, or prove an unquantized source is clean.
A quality approval needs confirmed checkpoint provenance and a repeated
source/GGUF comparison with an appropriate recipe. The existing Q4_0 export,
loader success, size savings and stock-city96 rejection remain measured facts.
The core-only test server/scratch files are cleaned up; diagnostic previews stay.
