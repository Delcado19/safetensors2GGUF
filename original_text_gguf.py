"""Original-key CLIP, multimodal and audio text GGUFs for the paired project loader."""
import hashlib
import json
from pathlib import Path
import tempfile

import gguf
import numpy as np
import torch

from compiled_flux_writer import CompiledFluxWriter
from comfyui_s2gguf.codec import LAYOUT_KEY, TENSORS_KEY
from conversion_io import atomic_output, validate_output
from convert import ConversionCancelled, load_state_dict
from dequantize import _scan_quantized_layers, dequantize_weight
from scripts.gguf_research_io import open_gguf
from scripts.requantize_flux_q6 import FLUX_PRESETS, image_k_type

ORIGINAL_FAMILIES = {'clip-l', 'clip-bigg', 'qwen2.5-vl-7b', 'qwen3-vl', 'minimax-music3'}
ORIGINAL_ARCHES = ORIGINAL_FAMILIES | {'ernie_image'}
LOADER_NOTE = ('Requires Original Text / Vision Encoder (S2GGUF), not the stock GGUF CLIP loader. '
               'All vision tensors are retained. Loading dequantizes weights to CPU RAM; inference is unqualified.')


class OriginalTextKWriter(CompiledFluxWriter):
    """Reuse bounded compiled K encoding while protecting lookup and vision buffers."""
    supported_arches = tuple(ORIGINAL_ARCHES)
    allows_special_shapes = True

    def quantizable(self, name, shape):
        """Protect embeddings, buffers and architecture-sensitive original keys."""
        if self.raw_family == 'ernie_image':
            from models.architectures import ModelErnieImage
            if any(part in name for part in ModelErnieImage.keys_hiprec):
                return False
        return (len(shape) == 2 and min(shape) > 1 and np.prod(shape) > 1024
                and name.endswith('weight') and not any(part in name for part in
                ('embed', 'embedding', 'lm_head', 'norm', 'conv', 'position', 'patch_embed')))

    def tensor_type(self, name, shape, source_type):
        """Map original attention/FFN roles into the shared mixed-preset policy."""
        if not self.quantizable(name, shape):
            return source_type
        policy_name = name.replace('.down_proj.', '.ffn_down.').replace('.v_proj.', '.to_v.')
        return image_k_type(policy_name, shape, preset=self.preset, attention_v_index=self.attention_v_index)

    def __init__(self, path, arch, use_temp_file, **kwargs):
        self.raw_family = arch
        super().__init__(path, arch, use_temp_file, **kwargs)

    def add_tensor(self, name, data, *, raw_dtype=None):
        """Keep integer and empty payloads verbatim in the readback plan."""
        if raw_dtype == gguf.GGMLQuantizationType.I8 or data.size == 0:
            self.plan.append({'name': name, 'shape': list(data.shape), 'type': raw_dtype,
                              'bytes': data.nbytes, 'sha256': hashlib.sha256(data).hexdigest()})
            gguf.GGUFWriter.add_tensor(self, name, data, raw_dtype=raw_dtype)
        else:
            super().add_tensor(name, data, raw_dtype=raw_dtype)


def export_original_text(source, destination, family, preset, *, overwrite=False, cancel_event=None, on_log=None,
                         on_progress=None):
    """Preserve every source component and publish only after complete payload readback."""
    if family not in ORIGINAL_ARCHES or preset not in (*FLUX_PRESETS, 'F16', 'F32', 'BF16', 'Q8_0'):
        raise ValueError('Unsupported original-key text family or precision')
    validate_output(source, destination, overwrite)
    from bundled_tools import validate_text_destination
    validate_text_destination(destination)
    state = load_state_dict(str(source), strip_prefixes=False)
    quantized, skip = _scan_quantized_layers(state)
    total = sum(name not in skip for name in state)
    def progress(fraction, phase, log=False):
        if on_progress:
            on_progress(fraction, phase)
        if log and on_log:
            on_log('INFO: ' + phase)
    # These are completed work stages, not a prediction of elapsed time or bytes.
    progress(0.0, f'Preparing {family} {preset}: {total} tensors', True)
    before = Path(source).stat()
    descriptors, expected = {}, {}
    is_k = preset in FLUX_PRESETS
    if is_k:
        from scripts.compiled_kernel_package import load_package, _platform
        host = _platform()
        kernel, _ = load_package(Path(__file__).parent / 'third_party/cpu-kernels' / (host['system'] + '-' + host['machine']))
        writer = OriginalTextKWriter(None, family, True, kernel=kernel, preset=preset, source=source,
                                     cancel_event=cancel_event, on_log=on_log)
    else:
        writer = gguf.GGUFWriter(None, family, use_temp_file=True)
    def cancelled():
        if cancel_event is not None and cancel_event.is_set():
            raise ConversionCancelled()
    with tempfile.TemporaryFile(dir=Path(destination).resolve().parent) as spool:
        writer.temp_file = spool
        writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
        writer.add_file_type(None if is_k else getattr(gguf.LlamaFileType, {
            'F16': 'MOSTLY_F16', 'BF16': 'MOSTLY_BF16', 'F32': 'ALL_F32', 'Q8_0': 'MOSTLY_Q8_0'}[preset]))
        writer.add_uint32(LAYOUT_KEY, 1)
        writer.add_string('comfy.s2g.loader', 'S2GGUFDiffusionLoader' if family == 'ernie_image' else 'S2GGUFTextEncoderLoader')
        for name, tensor in state.items():
            cancelled()
            if name in skip:
                continue
            if name in quantized:
                tensor = dequantize_weight(state, name, quantized[name], tensor)
            row = {'shape': list(tensor.shape), 'dtype': str(tensor.dtype)}
            if not tensor.is_floating_point():
                values = tensor.numpy()
                row['bytes_dtype'] = values.dtype.str
                data = values.reshape(-1).view(np.uint8).reshape(-1).view(np.int8)
                qtype = gguf.GGMLQuantizationType.I8
            else:
                # Float8 storage has no native NumPy dtype; decode after known scales.
                values = tensor.float().numpy()
                if not np.isfinite(values).all():
                    raise ValueError('Nonfinite original text tensor: ' + name)
                if len(values.shape) > 4:
                    values = values.reshape(*values.shape[:3], -1)
                if not values.shape:
                    values = values.reshape(1)
                qtype = gguf.GGMLQuantizationType.F32
                if preset == 'F16' or (is_k and tensor.dtype == torch.float16):
                    # Preserve finite outliers instead of emitting overflowing half floats.
                    if values.size == 0 or np.max(np.abs(values)) <= np.finfo(np.float16).max:
                        qtype = gguf.GGMLQuantizationType.F16
                elif preset == 'BF16' and values.size:
                    # gguf's grouped BF16 codec cannot reshape empty rows; keep empty buffers as F32.
                    qtype = gguf.GGMLQuantizationType.BF16
                elif preset == 'Q8_0' and values.size and values.ndim == 2 and values.shape[-1] % 32 == 0:
                    qtype = gguf.GGMLQuantizationType.Q8_0
                data = gguf.quants.quantize(values, qtype)
            data = np.ascontiguousarray(data)
            writer.add_tensor(name, data, raw_dtype=qtype)
            descriptors[name] = row
            count = len(descriptors)
            progress(0.8 * (count / total), f'Encoding tensor {count}/{total}: {name}',
                     count == 1 or count % 25 == 0 or count == total)
            if not is_k:
                expected[name] = hashlib.sha256(data).hexdigest()
        if not descriptors:
            raise ValueError('No original text tensors')
        writer.add_string(TENSORS_KEY, json.dumps(descriptors))
        writer.add_uint32('comfy.s2g.vision_tensor_count', sum(
            name.startswith(('model.visual.', 'visual.')) for name in descriptors))
        with atomic_output(destination, overwrite) as temporary:
            progress(0.8, 'Writing GGUF file', True)
            try:
                writer.write_header_to_file(path=temporary)
                writer.write_kv_data_to_file()
                writer.write_tensors_to_file(progress=False)
            finally:
                writer.close()
            cancelled()
            with open_gguf(temporary) as reader:
                progress(0.9, 'Validating GGUF readback', True)
                if {tensor.name for tensor in reader.tensors} != set(descriptors):
                    raise ValueError('Original text output tensor set mismatch')
                for tensor in reader.tensors:
                    cancelled()
                    if not is_k and hashlib.sha256(tensor.data).hexdigest() != expected[tensor.name]:
                        raise ValueError('Original text payload readback mismatch')
            after = Path(source).stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError('Original text source changed during conversion')
    if on_log:
        note = ('Requires Original Diffusion Model (S2GGUF). Loading reconstructs CPU weights; inference is unqualified.'
                if family == 'ernie_image' else LOADER_NOTE)
        on_log('INFO: ' + note)
    progress(1.0, 'GGUF export complete')
    return str(destination)
