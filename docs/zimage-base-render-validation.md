# Z-Image Base converted-model render comparison

Date: 2026-10-09. Status: all ten existing conversions and the independent
repeated source completed; 36 PNGs retained. Visual assessment is recorded below.
No models are converted or deleted during this task.

## Inputs and scope

- Original: `G:/ComfyUI-Easy-Install/ComfyUI/models/diffusion_models/Z-Image Base/zImageBase.safetensors`.
- Ten existing conversions: `I:/zImageBase-*` (nine safetensors outputs and one
  Q4_K_M GGUF). Safetensors record the conservative precision profile.
- Workflow: `G:/ComfyUI-Easy-Install/ComfyUI/user/default/workflows/Z-Image Base/Z-Image Base - FameGrid Revolution - with formula.json`.
  SHA256: `c744f38e29651cae7d947e41e2878147d710a092807ae70fb1ec9527aa3fddbf`.
- Prompt: the user-approved adult motel/noir portrait; saved negative prompt retained.
- Base: 1280x1568, 23 steps, `exponential/res_2s`, `beta57`, CFG 3.5, denoise 1,
  eta 0.5, bongmath enabled, AuraFlow shift 4.
- Saved Pixaroma seed: 3626416340707206, fixed across every run.
- Turbo refinement: saved JibMix ZIT v20, `exponential/res_4s_krogstad` / beta,
  denoise 0.15, three steps from `min(max(round(23 * 0.15), 2), 4)`,
  CFG 0.7 from `3.5 * 0.2`, shift 3.5. No active LoRAs in either loader.
- Encoder: saved Josiefied Qwen3 4B abliterated v1; VAE: saved UltraFlux v1.
- Preserve UltimateSDUpscale, SeedVR2 7B Q4_K_M, sharpening, film grain and
  SkinContrast stages. Final SeedVR2 output is 3344x4096.
- ComfyUI 0.39.0, Kitchen 0.2.37, Torch 2.9.1+cu130, RTX 5080 16 GiB;
  isolated port 8189, 3 GiB reserved VRAM and separate output/user/temp folders.

## Reproducibility and artifacts

Only positive prompt, necessary loader/model references, RNG mode and image-saving
nodes change. Materialize the saved Use Everywhere dependencies in the isolated
working copy. The installed frontend exports the custom widgets/dynamic inputs.
Pin Pixaroma's backend `runSeed` explicitly because frontend import replaces old
saved state. Resolve existing model filename aliases only when their normalized
names uniquely match (Windows separators and spaces in upscaler names).
The first export-only/harness attempts produced no images due to invalid filenames;
they are not counted as model evidence. Require all three PNG stages even when
ComfyUI reports success, since valid numeric output nodes alone can succeed.

For each format retain:

- `base`: raw Base decode before sharpening or Turbo refinement.
- `refined`: the stored Turbo pass.
- `final`: complete stored upscaling/postprocessing chain.

All local artifacts live under `runtime-results/zimage-base-2026-10-09/`:
full-resolution PNGs in `images/`, clickable `index.html`, exported workflow/API
JSON, prompt histories and logs. The folder is deliberately retained and ignored
by Git; no originals or models are copied into it. The original source workflow
is unchanged. `scripts/validate_zimage_base_batch.py --resume` skips complete
three-stage results; it never initiates conversion or shutdown.
`base-contact-sheet.jpg`, `refined-contact-sheet.jpg` and `final-contact-sheet.jpg`
provide labeled overviews. `scripts/summarize_zimage_base_batch.py` reproduces
these sheets and descriptive metrics without changing any original PNG.

The repeated original controls reproducibility. Numeric image differences are
descriptive only; major face/pose/anatomy/composition/quality changes are assessed
visually. Separate Base deviations from changes introduced by Turbo/upscalers.
One prompt/seed/checkpoint/workflow does not certify other models or settings.

## Results

Visual review of raw Base and final outputs:

| Format | Base-stage observations versus original | Final-stage observations | Assessment |
| --- | --- | --- | --- |
| Source BF16 | Cyan/red motel doorway, long wavy hair, parted lips, both arms braced on the doorway, overlapping/crossed legs and heels. Leg/heel overlap is already ambiguous in the reference. | Refinement/upscaling smooth skin, alter small facial/shoe details and intensify wall/pavement texture. | Reference itself is not an anatomy-perfect or pixel-invariant quality target. |
| FP16 | Shorter hair, a more closed mouth, altered facial appearance and hand grip, and one leg clearly bent back instead of the original lower-leg arrangement. Background doorway/window layout also changes. | Same FP16 pose/composition persists after refinement; scene remains sharp and usable, no black/noise-only output. | Meaningful motif differences already occur with a BF16-to-FP16 cast. They do not, by themselves, prove a broken low-bit algorithm or degraded visual quality. |
| FP16 mixed | Long hair and parted lips are closer to the source than plain FP16, but the bent-back leg arrangement differs from the source. Raised-hand grip and background windows also change. Localized irregular fabric at the upper waist. | Refinement keeps the altered leg pose and turns the waist texture into a noticeable oval fabric patch. No global corruption. | Usable image with pose/detail changes and a localized clothing artifact; mixed precision does not guarantee an identical trajectory. |
| FP8 E4M3 scaled | Face turns toward the camera with raised chin instead of looking down; facial appearance, dress neckline/cup construction and leg crossing change. Background becomes a motel walkway with bollards rather than the reference window/door arrangement. | These motif changes persist, with darker blue facial lighting and stronger masonry texture. Full scene remains coherent and sharp. | Pronounced motif/composition variation, but no black/noise-only output or obvious global quantization corruption. |
| FP8 E4M3 scaled mixed | Retains the downward gaze and doorway motif, but shorter hair, deeper dress neckline, hand placement and leg crossing differ. | Postprocessing amplifies a localized yellow-green patch near the right underarm and adds illegible background-wall lettering. Subject and scene stay coherent. | Usable with localized color/detail artifacts; final-chain findings must not be attributed solely to Base weight packing. |
| INT4 + ConvRot mixed | Strong pose change: legs apart with one knee bent, more upright body, shorter/smoother hair, more open gaze and a largely empty warm yellow/brown background. Red/yellow sign and gold pavement reflections replace much of the reference pink/cyan background. | Keeps the changed pose and warmer scene; smoothing and tiny nail/color details change through postprocessing. No global corruption. | Pronounced motif/lighting variation, but successful native W4A4 loading, sampling and complete postprocessing on this Base checkpoint. |
| INT8 + ConvRot | Much more crouched/raised-leg posture; face turns into a right-facing profile. Lighting becomes flatter gray/green, dress neckline changes, and background signage becomes illegible words instead of the reference MOTEL sign. | The profile/crouched pose and gray/green atmosphere remain. Postprocessing smooths skin and changes wall labels but does not restore the reference composition. | **Rejected by the user as unusable due to severe visual drift**, across Base, refined and final images. Successful rendering does not imply acceptable fidelity. |
| INT8 + ConvRot mixed | Much closer to the reference cyan/pink doorway atmosphere than plain INT8. Shorter hair, changed mouth/face details and a bent-back leg differ from the original crossed/overlapping lower-leg arrangement. | Keeps those changes; no large color blotch or global corruption observed. | Usable; in this one test protection reduces the large atmosphere/composition shift seen with plain INT8, but does not reproduce the reference pose. |
| NVFP4 | Severe body/pose breakdown: the subject folds into a contorted almost inverted shape, with implausible neck/torso connections and apparent extra/fused leg structures. Viewpoint and entire motel facade/sign layout also change. | Upscaling preserves and emphasizes the impossible body geometry; the Turbo pass does not repair it. | **Unusable anatomy in this test**, despite technically successful loading and full workflow execution. This is a quality failure, not merely ordinary texture or lighting variance. |
| NVFP4 mixed | Restores a coherent standing/leaning woman rather than plain NVFP4's contorted body. Crossed ankles, bra-style dress construction and a side-on row of motel rooms differ from the reference. Cyan/red atmosphere remains. | Normal body geometry persists; the changed dress, background perspective and facial details remain. | Usable in this test; the conservative mixed policy avoids the severe body breakdown observed with plain NVFP4. |
| GGUF Q4_K_M | Legs are apart, body more upright, face/hair and dress folds differ. Large sign and background layout shift, with muted skin lighting and green/yellow reflections. | Keeps the pose/composition changes while producing a coherent sharp photograph. | Successful native GGUF loading and full workflow; pronounced motif variation but no major global corruption. Only this GGUF precision is covered. |

### Controlled repeat

The original was recomputed at the end. History confirms the Base loader, sampler
and decode were not in the cached-node set. Its Base and Turbo-refined pixels
are **exactly identical** to the first original. The final image differs slightly
(mean absolute RGB difference 0.8132 on the 0–255 scale), with unchanged motif.
The later upscaling/postprocessing chain, including unseeded grain, is therefore
not strictly pixel-invariant. This repeat is a reproducibility check, not a
requirement that quantized images must match pixels.

All 12 jobs (original, ten conversions, repeated original) completed successfully
and produced all three stages. Technical success is not a quality pass: plain
NVFP4 has severe anatomy failure in this prompt/seed. On 2026-10-10 the user also
rejected plain INT8's three stages due to severe visual drift. Both plain INT8
and plain NVFP4 are recorded as **Severe visual drift** with a failed-support
classification in the Workbench Base row. The eight other conversions remain
usable within this assessment, with the larger deviations listed above. Do not infer universal
rankings or support for other checkpoints/workflows from this one sample.

Descriptive raw-Base mean absolute RGB differences versus source are retained
in `metrics.json`. They are not quality scores: plain INT8 has a larger numeric
difference than the visibly worse plain NVFP4 because composition/palette change
also contributes. No black or noise-only result occurred.

Windows entered standby during the FP16 run and resumed at 17:12 local time.
Wall-clock duration therefore does not measure conversion-format performance.
A temporary native Windows sleep-inhibition process now protects the remaining
authorized render/review task, without changing permanent energy settings.
Shutdown is requested only after saving the final findings and verifying the
retained gallery. Model files on I: and the source workflow remain unchanged.
