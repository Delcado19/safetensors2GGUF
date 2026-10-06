# Audit of unsupported compatibility entries

Reviewed: 2026-10-06. Repository baseline: `218ff5a`. Scope: all 30 current
`bad` cells (8 diffusion, 22 text-encoder), grouped below. This is an upstream
source/current-code audit, not a new conversion or render test. No installed
software, model files, export code or matrix classifications were changed.

## Findings

| Existing negative cells | Current evidence | Assessment |
| --- | --- | --- |
| Qwen-Image GGUF (1) | The project already converted Qwen-Image-Edit-2511 Q4_K_M and rendered a real edit on 2026-10-05; upstream loader explicitly accepts `qwen_image`. Unsloth publishes 2511 GGUF files. | The blanket conversion-impossible claim is contradicted by our own runtime record. Support depends on the quantizer build/type, and quality parity remains untested for that smoke test. |
| ERNIE-Image GGUF (1) | Unsloth publishes ERNIE-Image GGUF. The targeted upstream `lcpp.patch` and stock city96 loader still lack `ernie_image`. | A tooling/loader integration limit, not universal impossibility of ERNIE GGUF. No evidence of this project's export/stock-loader compatibility. |
| Krea 2 GGUF (1) | Producer molbal publishes Krea 2 GGUF and explicitly requires its ComfyUI-GGUF fork. PR #459 remains open and includes Krea work; stock loader lacks `krea2`. | Restriction remains valid for the targeted stock path. External fork support exists. Producer offers non-K formats, so the grouped GGUF cell must not imply our Q4_K_M path is supported. |
| Qwen-Image plain NVFP4 (1) | Our plain-policy renders recorded severe mosaic/noise; current official 2511 NVFP4 recipes protect selected layers. Our revised mixed recipe renders usable images with detail drift. | Keep the negative result for the tested plain policy. Published NVFP4 does not prove that identical layer/scaling/calibration policy works. |
| Z-Image/Lumina2 FP8, INT8, NVFP4, NVFP4_MIXED (4) | Older project comparisons changed outfit/composition substantially. Published Z-Image Turbo NVFP4 uses a different mixed recipe; our `z_image_turbo` NVFP4_MIXED profile has a newer usable two-prompt result with visible detail/framing drift. | Family/default-policy failures should not be extrapolated to every Z-Image variant/profile. Preserve recorded failures, expose the scoped Turbo result, and retest exact plain-policy outputs before lifting those restrictions. |
| CLIP-L/OpenCLIP-bigG GGUF (2) | Current city96 text architecture allowlist lacks CLIP text models; llama.cpp's text conversion map does not register CLIPModel/CLIPTextModel. Vision CLIP/mmproj is a different path. | No supported CLIP text GGUF export/decode path found for the current toolchain. Do not confuse CLIPLoaderGGUF's ability to load regular safetensors alongside a GGUF T5 with CLIP text GGUF support. |
| CLIP-L/OpenCLIP-bigG FP8, FP8_MIXED, INT8, INT8_MIXED, NVFP4, NVFP4_MIXED (12) | Project renders/loads failed with missing quantization-metadata routing. Fresh upstream `SDClipModel` can use MixedPrecisionOps when model options provide quantization metadata, but standard CLIP family routing does not establish the missing per-encoder metadata handoff. | The old explanation is too absolute: the basic operation exists. No end-to-end evidence clears our outputs through the standard loader. Keep the restriction pending targeted metadata-routing work/tests. |
| Pile-T5-XL/AuraFlow FP8, FP8_MIXED, INT8, INT8_MIXED, NVFP4, NVFP4_MIXED (6) | Current AuraT5Model delegates model options to SD1ClipModel; this base now has quantization-aware operations. Standard T5_XL selection still does not establish automatic metadata handoff. Older outputs produced noise or dimension errors. | A concrete loader-routing issue, not an intrinsic ban on low-bit T5. No new success evidence for this project's native artifacts/standard path. Retest after routing changes. |
| Qwen2.5-VL 7B GGUF (1) | Current city96 `gguf_clip_loader` calls `gguf_mmproj_loader` for qwen2vl, finds a matching companion file, maps vision tensors and merges them. QuantStack documents the text encoder + mmproj pair. Our exporter only produces the language GGUF. | The claim that the upstream loader cannot consume a separate vision export is obsolete/incorrect for current upstream. This application's missing mmproj export remains a real integration limit; a language-only file is still insufficient for reference-image editing. |
| Mistral Small 3.2 24B GGUF (1) | Issue #367 is closed. On 2025-11-29 the maintainer reported adding Mistral-small loading/tokenizer reconstruction and testing standard llama.cpp-converted Unsloth files. Current loader reconstructs Tekken for the matching llama embedding shape. | The open-issue/upstream-no-support explanation is stale. The project's separate 30-layer FLUX.2 packaging/config mismatch remains; it must not classify the full 40-layer base encoder as universally unsupported. Split model packaging/tool integration from upstream format support. |

All 30 cells are covered by the counts above. Existing successful web publications
do not prove this converter's fidelity, loader compatibility, LoRA or offload
behavior. This audit does not silently turn blocked cells green.

## Sources and scope

- [Project Qwen conversion and render record](conversion-safety-validation.md#qwen-image-edit-2511-gguf-render-smoke-test): one successful converted Q4_K_M edit, no source-baseline comparison; quantizer build 3962 (`c8c07d658`).
- [Current city96 loader](https://github.com/city96/ComfyUI-GGUF/blob/6ea2651e7df66d7585f6ffee804b20e92fb38b8a/loader.py): image/text allowlists, mmproj pairing and Tekken handling.
- [Targeted lcpp.patch](https://github.com/city96/ComfyUI-GGUF/blob/6ea2651e7df66d7585f6ffee804b20e92fb38b8a/tools/lcpp.patch): still no qwen_image/ernie_image/krea2 quantizer architecture registration; this is narrower than all available binaries/float exports.
- [Qwen 2511 GGUF producer](https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF), [ERNIE GGUF producer](https://huggingface.co/unsloth/ERNIE-Image-GGUF), [Krea producer and required fork](https://huggingface.co/molbal/krea2-gguf).
- [Krea-containing PR #459](https://github.com/city96/ComfyUI-GGUF/pull/459): open at review; updated 2026-10-05. Its current title refers to Ideogram, but it also includes Krea commits.
- [FLUX.2 issue #367](https://github.com/city96/ComfyUI-GGUF/issues/367): closed since 2025-11-30; Mistral support/test statement posted 2025-11-29.
- [Qwen encoder/mmproj instructions](https://huggingface.co/QuantStack/Qwen-Image-Edit-GGUF/discussions/6).
- [Official Qwen 2511 NVFP4 recipe](https://github.com/Comfy-Org/comfy-quants/blob/main/configs/qwen_image_edit_2511_nvfp4.yaml), [Z-Image Turbo NVFP4 producer recipe](https://huggingface.co/InsecureErasure/Z-Image-Turbo-NVFP4), and [project comparisons](quantization-size-audit.md#implemented-corrections-and-runtime-evidence-2026-10-05).
- [Current ComfyUI SDClipModel](https://github.com/Comfy-Org/ComfyUI/blob/d49e888586dd8ae012c0667b33466b815fee07f7/comfy/sd1_clip.py), [standard text loader routing](https://github.com/Comfy-Org/ComfyUI/blob/d49e888586dd8ae012c0667b33466b815fee07f7/comfy/sd.py), [AuraFlow wrapper](https://github.com/Comfy-Org/ComfyUI/blob/d49e888586dd8ae012c0667b33466b815fee07f7/comfy/text_encoders/aura_t5.py).
- [llama.cpp text/MM converter registry](https://github.com/ggml-org/llama.cpp/blob/master/conversion/__init__.py): text CLIP is distinct from multimodal vision conversion.

GitHub API content was checked for ComfyUI because cached raw web snapshots lacked
the MixedPrecisionOps code present in fresh API/local files. ComfyUI upstream
head at review: `d49e888586dd8ae012c0667b33466b815fee07f7` (2026-10-06);
city96 head: `6ea2651e7df66d7585f6ffee804b20e92fb38b8a` (2026-01-12).
The installed source was also read without modifying it. Static constructor
capability alone is not a successful standard-loader/render test.

## Required follow-up to the matrix

1. Correct the demonstrably contradicted Qwen GGUF restriction using the existing
   local conversion/render record, without claiming unmeasured source fidelity.
2. Distinguish a project export limitation, required external loader/fork, recorded
   render failure and untested quality. They are not one universal unsupported state.
3. Separate the full Mistral base encoder from the pruned FLUX.2 packaging, and
   language-only Qwen-VL export from the complete language + mmproj workflow.
4. Preserve dated negative render evidence and profile-specific positive results;
   retest matching outputs before lifting native CLIP/AuraFlow/plain-policy guards.

No large model was downloaded, converted or rendered during this audit.
