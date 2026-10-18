"""City96 diffusion policies over the shipped CPU block kernels.

Architecture exclusions are ported from the preserved Apache-2.0 City96 patch
at 6ea2651; block math and ordered mixed-preset rules reuse the qualified writer.
Qwen Image uses explicit conservative project protections, since the
frozen native reference predates those architectures. Krea keeps its own adapter.
"""
from functools import partial
from pathlib import Path
import uuid
import tempfile

import gguf
import numpy as np
from safetensors.torch import load_file

from compiled_flux_writer import CompiledFluxWriter
from conversion_io import atomic_output
from convert import convert_file
from models.architectures import ModelQwenImage
from scripts import requantize_flux_q6 as quant
from scripts.gguf_research_io import open_gguf


_FLUX = ('txt_in.', 'img_in.', 'time_in.', 'vector_in.', 'guidance_in.', 'final_layer.')
PROTECTED = {
    'flux': _FLUX, 'hyvid': _FLUX,
    'sd1': ('class_embedding.', 'time_embedding.', 'add_embedding.', 'time_embed.',
            'label_emb.', 'conv_in.', 'conv_out.'),
    'sd3': ('final_layer.', 'time_text_embed.', 'context_embedder.', 't_embedder.', 'y_embedder.', 'x_embedder.'),
    'aura': ('t_embedder.', 'init_x_linear.'),
    'ltxv': ('adaln_single.', 'caption_projection.', 'patchify_proj.', 'proj_out.', 'scale_shift_table'),
    'wan': ('modulation.', 'patch_embedding.', 'text_embedding.', 'time_projection.', 'time_embedding.', 'img_emb.', 'head.'),
    'hidream': ('p_embedder.', 't_embedder.', 'x_embedder.', 'final_layer.', '.ff_i.gate.weight', 'caption_projection.'),
    'cosmos': ('p_embedder.', 't_embedder.', 't_embedding_norm.', 'x_embedder.', 'pos_embedder.', 'final_layer.'),
    'lumina2': ('t_embedder.', 'x_embedder.', 'final_layer.', 'cap_embedder.', 'context_refiner.', 'noise_refiner.'),
}
PROTECTED['sdxl'] = PROTECTED['sd1']
# These adapters have no equivalent in the pinned native image CLI.
PROTECTED['qwen_image'] = tuple(ModelQwenImage.keys_hiprec)
EXACT = {'sd1': ('input_blocks.0.0.weight', 'out.2.weight'),
         'sdxl': ('input_blocks.0.0.weight', 'out.2.weight'),
         'sd3': ('proj_out.weight', 'pos_embed'),
         'aura': ('modF.1.weight', 'cond_seq_linear.weight', 'final_linear.weight', 'positional_encoding', 'register_tokens')}
_GENERAL = ('_norm.weight', 'ffn_gate_inp.weight', 'ssm_conv1d.weight', 'time_mix_first.weight',
            'time_mix_w1.weight', 'time_mix_w2.weight', 'time_mix_decay_w1.weight',
            'time_mix_decay_w2.weight', 'attn_rel_b.weight')


def quantizable(arch, name, shape):
    """Apply architecture exclusions before mixed-preset counters or row fallbacks."""
    if arch not in PROTECTED:
        raise ValueError('Unsupported bundled diffusion architecture: ' + arch)
    return (len(shape) == 2 and shape[0] > 1 and name.endswith('weight')
            and name not in EXACT.get(arch, ())
            and not any(part in name for part in (*_GENERAL, *PROTECTED[arch])))


class CompiledDiffusionWriter(CompiledFluxWriter):
    """Preserve model-specific preparation and high-rank float tensors in one GGUF."""
    supported_arches = tuple(PROTECTED)
    allows_special_shapes = True

    def __init__(self, path, arch, use_temp_file, **kwargs):
        self.diffusion_arch = arch
        self.accepts_nd_tensors = arch in ('wan', 'hyvid')
        super().__init__(path, arch, use_temp_file, **kwargs)

    def quantizable(self, name, shape):
        """Return whether this architecture permits quantizing the tensor."""
        return quantizable(self.diffusion_arch, name, shape)

    def tensor_type(self, name, shape, source_type):
        """Choose the mixed-preset type while retaining protected source precision."""
        if len(shape) > 5:
            raise ValueError('Diffusion tensors support at most five dimensions')
        if not self.quantizable(name, shape):
            return source_type
        return quant.image_k_type(name, shape, preset=self.preset,
                                  attention_v_index=self.attention_v_index)


def convert_diffusion(source, destination, preset, kernel, *, keep_intermediate=False,
                      overwrite=False, cancel_event=None, on_log=None, on_progress=None):
    """Convert directly, or retain diagnostic F16 preparation and encode its tensors.

    High-rank video tensors stay F32, and Lumina pad tokens retain [1, D]. No
    repair is applied after publication; failures preserve an existing output.
    """
    factory = partial(CompiledDiffusionWriter, kernel=kernel, preset=preset, source=source,
                      cancel_event=cancel_event, on_log=on_log)
    kwargs = dict(interact=False, target_quant=gguf.GGMLQuantizationType.F16,
                  cancel_event=cancel_event, on_log=on_log, on_progress=on_progress)
    if not keep_intermediate:
        convert_file(source, destination, overwrite=overwrite, writer_factory=factory, **kwargs)
        return destination
    before = Path(source).stat()
    intermediate = Path(destination).parent / f'{Path(source).stem}-F16-tmp-{uuid.uuid4().hex}.gguf'
    _, arch = convert_file(source, str(intermediate), **kwargs)
    after = Path(source).stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Source changed during retained preparation')
    # Consume only this preparation's sidecar; never discover another checkpoint's file.
    with open_gguf(intermediate) as reader, atomic_output(destination, overwrite) as temporary:
        writer = factory(None, arch.arch, True)
        with tempfile.TemporaryFile(dir=Path(destination).parent) as spool:
            writer.temp_file = spool
            writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
            writer.add_file_type(None)
            try:
                for name, field in reader.fields.items():
                    if name in ('general.architecture', 'general.file_type', 'general.quantization_version',
                                'general.alignment', 'comfy.gguf.5d_sidecar') or name.startswith('GGUF.'):
                        continue
                    writer.add_key_value(name, field.contents(), field.types[0],
                                         field.types[1] if len(field.types) == 2 else None)
                for tensor in reader.tensors:
                    shape = tensor.shape.tolist()[::-1]
                    data = np.asarray(tensor.data).reshape(shape)
                    writer.add_tensor(tensor.name, data, raw_dtype=tensor.tensor_type)
                if getattr(arch, '_nd_tensors', None):
                    for name, tensor in load_file(arch.fix_path).items():
                        writer.add_tensor(name, tensor.float().numpy(), raw_dtype=gguf.GGMLQuantizationType.F32)
                writer.write_header_to_file(path=temporary)
                writer.write_kv_data_to_file()
                writer.write_tensors_to_file(progress=False)
            finally:
                writer.close()
    return destination
