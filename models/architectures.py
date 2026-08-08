"""Model architecture definitions and detection logic for safetensors-to-GGUF conversion."""

import os
import torch
from tqdm import tqdm
from safetensors.torch import load_file, save_file

QUANTIZATION_THRESHOLD = 1024
REARRANGE_THRESHOLD = 512
MAX_TENSOR_NAME_LENGTH = 127
MAX_TENSOR_DIMS = 4


class ModelTemplate:
    """Base class for all supported model architectures."""

    arch = "invalid"
    shape_fix = False
    keys_detect = []
    keys_banned = []
    keys_hiprec = []
    keys_ignore = []
    keys_unsqueeze = []  # 1D tensors that must be reshaped to [1, D] before writing
    # Tensors whose raw on-disk shape ComfyUI's model_detection.py reads directly
    # to infer architecture hyperparameters (e.g. Lumina2's cap_feat_dim from
    # cap_embedder.1.weight.shape[1], Flux's in_channels from img_in.weight.shape[1]),
    # BEFORE any dequantization happens. NVFP4 packs 2 values per uint8 byte, halving
    # the on-disk last dimension, which corrupts that inference and crashes model
    # loading (see docs/issues_analysis.md #9). Must be excluded from NVFP4 packing
    # unconditionally — not just in *_MIXED mode, since this is a shape-safety
    # constraint, not a precision one (FP8 is unaffected: it changes dtype, not shape).
    keys_shape_critical = []

    def handle_nd_tensor(self, key, data):
        """Handle tensors exceeding MAX_TENSOR_DIMS dimensions.

        Default implementation raises; subclasses that produce 5D tensors
        must override this to write a side-car fix file.
        """
        raise NotImplementedError(
            f"Tensor with unsupported dimensionality ({key} @ {data.shape}). "
            "Override handle_nd_tensor() for this architecture."
        )


class ModelFlux(ModelTemplate):
    arch = "flux"
    keys_detect = [
        ("transformer_blocks.0.attn.norm_added_k.weight",),
        ("double_blocks.0.img_attn.proj.weight",),
    ]
    keys_banned = ["transformer_blocks.0.attn.norm_added_k.weight"]
    # ComfyUI's model_detection.py infers in_channels from img_in.weight.shape[1]
    keys_shape_critical = ["img_in.weight"]


class ModelSD3(ModelTemplate):
    arch = "sd3"
    keys_detect = [
        ("transformer_blocks.0.attn.add_q_proj.weight",),
        ("joint_blocks.0.x_block.attn.qkv.weight",),
    ]
    keys_banned = ["transformer_blocks.0.attn.add_q_proj.weight"]
    # ComfyUI's model_detection.py infers context_embedder_config from context_embedder.weight.shape[1]
    keys_shape_critical = ["context_embedder.weight"]


class ModelAura(ModelTemplate):
    arch = "aura"
    keys_detect = [
        ("double_layers.3.modX.1.weight",),
        ("joint_transformer_blocks.3.ff_context.out_projection.weight",),
    ]
    keys_banned = ["joint_transformer_blocks.3.ff_context.out_projection.weight"]
    # ComfyUI's model_detection.py infers cond_seq_dim from cond_seq_linear.weight.shape[1]
    keys_shape_critical = ["cond_seq_linear.weight"]


class ModelHiDream(ModelTemplate):
    arch = "hidream"
    keys_detect = [
        (
            "caption_projection.0.linear.weight",
            "double_stream_blocks.0.block.ff_i.shared_experts.w3.weight",
        )
    ]
    keys_hiprec = [".ff_i.gate.weight", "img_emb.emb_pos"]
    # Audited against ComfyUI's model_detection.py: HiDream's unet_config is all
    # hardcoded constants, no shape-based hyperparameter inference — safe as-is,
    # no keys_shape_critical needed.


class CosmosPredict2(ModelTemplate):
    arch = "cosmos"
    keys_detect = [
        (
            "blocks.0.mlp.layer1.weight",
            "blocks.0.adaln_modulation_cross_attn.1.weight",
        )
    ]
    keys_hiprec = ["pos_embedder"]
    keys_ignore = ["_extra_state", "accum_"]
    # ComfyUI's model_detection.py infers in_channels/model_channels from
    # x_embedder.proj.1.weight.shape[1]/.shape[0] (a genuine nn.Linear inside the
    # PatchEmbed Sequential, index 1 after the Rearrange at index 0)
    keys_shape_critical = ["x_embedder.proj.1.weight"]


class ModelQwenImage(ModelTemplate):
    """Qwen-Image and Qwen-Image-Edit DiT (incl. 2509 multi-image edit variant).

    Shares tensor names with Flux/SD3 (e.g. transformer_blocks.0.attn.norm_added_k.weight,
    transformer_blocks.0.attn.add_q_proj.weight). Must be probed before ModelFlux/ModelSD3
    in arch_list, otherwise the banned-key heuristic there misclassifies Qwen-Image as
    a reference-format Flux/SD3 and aborts conversion.

    keys_detect mirrors upstream ComfyUI-GGUF tools/convert.py ModelQwenImage so that
    abliterated 2509 single-file checkpoints (e.g. jiangchengchengNLP/Qwen-Edit-2509-abliterated)
    convert cleanly without producing the false-positive "reference format" assertion.
    """

    arch = "qwen_image"
    keys_detect = [
        (
            "time_text_embed.timestep_embedder.linear_2.weight",
            "transformer_blocks.0.attn.norm_added_q.weight",
            "transformer_blocks.0.img_mlp.net.0.proj.weight",
        )
    ]
    # ComfyUI's model_detection.py infers in_channels from img_in.weight.shape[1]
    # (a Linear layer, same convention as Flux's img_in.weight)
    keys_shape_critical = ["img_in.weight"]


class ModelHyVid(ModelTemplate):
    arch = "hyvid"
    keys_detect = [
        (
            "double_blocks.0.img_attn_proj.weight",
            "txt_in.individual_token_refiner.blocks.1.self_attn_qkv.weight",
        )
    ]
    # ComfyUI's model_detection.py infers context_in_dim from
    # txt_in.input_embedder.weight.shape[1] (a genuine nn.Linear); img_in.proj.weight
    # is a Conv and safe — only shape[-1] (kernel width, never a multiple of 16) is
    # touched by our NVFP4 packing, and shape[1]/shape[2:] (in_channels/patch_size)
    # read by ComfyUI there are untouched.
    keys_shape_critical = ["txt_in.input_embedder.weight"]

    def handle_nd_tensor(self, key, data):
        """Write 5D tensor to a side-car safetensors file for later re-insertion.

        Appends to an existing fix file so models with multiple 5D tensors
        (e.g. Wan VACE) don't crash on the second occurrence.
        """
        path = f"./fix_5d_tensors_{self.arch}.safetensors"
        existing = {}
        if os.path.isfile(path):
            # Clone to release the mmap before overwriting the file (Windows)
            existing = {k: v.clone() for k, v in load_file(path).items()}
        existing[key] = torch.from_numpy(data)
        save_file(existing, path)
        tqdm.write(f"5D tensor exported for manual fix: {key} {data.shape}")


class ModelWan(ModelHyVid):
    arch = "wan"
    keys_detect = [
        (
            "blocks.0.self_attn.norm_q.weight",
            "text_embedding.2.weight",
            "head.modulation",
        )
    ]
    # nn.Parameter — cannot load from BF16 source
    keys_hiprec = [".modulation"]
    # ComfyUI's model_detection.py infers dim from head.modulation.shape[-1] — a 3D
    # nn.Parameter (1, 2, dim), NOT 1D, so the unconditional 1D-skip doesn't cover it.
    keys_shape_critical = ["head.modulation"]


class ModelLTXV(ModelTemplate):
    arch = "ltxv"
    keys_detect = [
        (
            "adaln_single.emb.timestep_embedder.linear_2.weight",
            "transformer_blocks.27.scale_shift_table",
            "caption_projection.linear_2.weight",
        )
    ]
    # nn.Parameter — cannot load from BF16 base quant
    keys_hiprec = ["scale_shift_table"]
    # ComfyUI's model_detection.py infers cross_attention_dim from
    # transformer_blocks.0.attn2.to_k.weight.shape[1]
    keys_shape_critical = ["transformer_blocks.0.attn2.to_k.weight"]


class ModelSDXL(ModelTemplate):
    arch = "sdxl"
    shape_fix = True
    keys_detect = [
        ("down_blocks.0.downsamplers.0.conv.weight", "add_embedding.linear_1.weight"),
        (
            "input_blocks.3.0.op.weight",
            "input_blocks.6.0.op.weight",
            "output_blocks.2.2.conv.weight",
            "output_blocks.5.2.conv.weight",
        ),
        ("label_emb.0.0.weight",),
    ]
    # NOT AUDITED for keys_shape_critical: ComfyUI's model_detection.py infers
    # context_dim from the first attn2.to_k.weight.shape[1] it finds by scanning
    # block indices dynamically, rather than a single fixed tensor name — unlike
    # the DiT architectures above, pinning the exact key requires resolving that
    # scan for this specific block layout. See docs/issues_analysis.md #9.


class ModelSD1(ModelTemplate):
    arch = "sd1"
    shape_fix = True
    keys_detect = [
        ("down_blocks.0.downsamplers.0.conv.weight",),
        (
            "input_blocks.3.0.op.weight",
            "input_blocks.6.0.op.weight",
            "input_blocks.9.0.op.weight",
            "output_blocks.2.1.conv.weight",
            "output_blocks.5.2.conv.weight",
            "output_blocks.8.2.conv.weight",
        ),
    ]
    # NOT AUDITED for keys_shape_critical — see ModelSDXL comment above; same
    # dynamic attn2.to_k.weight scan applies. See docs/issues_analysis.md #9.


class ModelLumina2(ModelTemplate):
    arch = "lumina2"
    keys_detect = [
        ("cap_embedder.1.weight", "context_refiner.0.attention.qkv.weight")
    ]
    # nn.Parameter pads — BF16 causes size-doubling on load (Issue #419)
    keys_hiprec = ["x_pad_token", "cap_pad_token"]
    # ComfyUI NextDiT expects shape [1, D]; older checkpoints store them as [D]
    keys_unsqueeze = ["x_pad_token", "cap_pad_token"]
    # ComfyUI's model_detection.py infers cap_feat_dim from cap_embedder.1.weight.shape[1]
    keys_shape_critical = ["cap_embedder.1.weight"]


arch_list = [
    # ModelQwenImage must precede ModelFlux/ModelSD3 — Qwen-Image shares
    # transformer_blocks.0.attn.norm_added_k.weight and add_q_proj.weight
    # with those architectures, which would otherwise trigger their
    # reference-format keys_banned guard and abort conversion.
    ModelQwenImage,
    ModelFlux,
    ModelSD3,
    ModelAura,
    ModelHiDream,
    CosmosPredict2,
    ModelLTXV,
    ModelHyVid,
    ModelWan,
    ModelSDXL,
    ModelSD1,
    ModelLumina2,
]


def is_model_arch(model, state_dict):
    """Return True if state_dict matches the given architecture class.

    Raises AssertionError if the model is detected but in a banned format
    (e.g. reference format instead of diffusers format).
    """
    matched = False
    invalid = False
    for match_list in model.keys_detect:
        if all(key in state_dict for key in match_list):
            matched = True
            invalid = any(key in state_dict for key in model.keys_banned)
            break
    assert not invalid, (
        "Model architecture not allowed for conversion. "
        "Use diffusers format, not reference format."
    )
    return matched


def detect_arch(state_dict):
    """Detect and return the matching architecture instance for state_dict.

    Raises AssertionError if no supported architecture is found.
    """
    for arch in arch_list:
        if is_model_arch(arch, state_dict):
            return arch()
    raise AssertionError(
        f"Unknown model architecture. Checked: {[a.arch for a in arch_list]}"
    )
