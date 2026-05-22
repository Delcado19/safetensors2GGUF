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


class ModelSD3(ModelTemplate):
    arch = "sd3"
    keys_detect = [
        ("transformer_blocks.0.attn.add_q_proj.weight",),
        ("joint_blocks.0.x_block.attn.qkv.weight",),
    ]
    keys_banned = ["transformer_blocks.0.attn.add_q_proj.weight"]


class ModelAura(ModelTemplate):
    arch = "aura"
    keys_detect = [
        ("double_layers.3.modX.1.weight",),
        ("joint_transformer_blocks.3.ff_context.out_projection.weight",),
    ]
    keys_banned = ["joint_transformer_blocks.3.ff_context.out_projection.weight"]


class ModelHiDream(ModelTemplate):
    arch = "hidream"
    keys_detect = [
        (
            "caption_projection.0.linear.weight",
            "double_stream_blocks.0.block.ff_i.shared_experts.w3.weight",
        )
    ]
    keys_hiprec = [".ff_i.gate.weight", "img_emb.emb_pos"]


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


class ModelHyVid(ModelTemplate):
    arch = "hyvid"
    keys_detect = [
        (
            "double_blocks.0.img_attn_proj.weight",
            "txt_in.individual_token_refiner.blocks.1.self_attn_qkv.weight",
        )
    ]

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


class ModelLumina2(ModelTemplate):
    arch = "lumina2"
    keys_detect = [
        ("cap_embedder.1.weight", "context_refiner.0.attention.qkv.weight")
    ]
    # nn.Parameter pads — BF16 causes size-doubling on load (Issue #419)
    keys_hiprec = ["x_pad_token", "cap_pad_token"]
    # ComfyUI NextDiT expects shape [1, D]; older checkpoints store them as [D]
    keys_unsqueeze = ["x_pad_token", "cap_pad_token"]


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
