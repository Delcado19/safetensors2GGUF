# SPDX-License-Identifier: Apache-2.0
"""Experimental FLUX float-GGUF to Q6_K/Q5_K_M path; native remains the application default.

Policy port adapted from City96/ComfyUI-GGUF lcpp.patch at 6ea2651 and the
MIT-licensed llama.cpp b3962 general exclusions. Preserved license notices:
third_party/converter_references/city96-patch-LICENSE and llama-b3962-LICENSE.
The Python policy port and bounded writer are modifications, not upstream code.
"""
import argparse
import hashlib
import json
from pathlib import Path

import gguf
import numpy as np

from conversion_io import atomic_output, validate_output
from scripts.check_q6_gguf_container import _cancel, _file_hash
from scripts.reference_q6_k import iter_q6_k
from scripts.reference_q5_k import iter_q5_k
from scripts.gguf_research_io import open_gguf


def flux_q6_type(name, shape, source_type, *, preset="Q6_K"):
    """Apply frozen City96/b3962 eligibility and mixed-preset rules to a canonical float tensor.

    This quantizer-stage policy follows lcpp.patch at 6ea2651, plus b3962's
    general exclusions. Source preparation/architecture transformations remain
    the existing converter's responsibility. Only FLUX Q6_K and Q5_K_M are covered.
    """
    if preset not in ("Q6_K", "Q5_K_M"):
        raise ValueError("FLUX research presets are Q6_K and Q5_K_M only")
    protected = ('txt_in.', 'img_in.', 'time_in.', 'vector_in.', 'guidance_in.', 'final_layer.',
                 '_norm.weight', 'ffn_gate_inp.weight', 'ssm_conv1d.weight', 'time_mix_first.weight',
                 'time_mix_w1.weight', 'time_mix_w2.weight', 'time_mix_decay_w1.weight',
                 'time_mix_decay_w2.weight', 'attn_rel_b.weight')
    if len(shape) != 2 or not name.endswith('weight') or any(part in name for part in protected):
        return source_type
    if any(part in name for part in ('model.diffusion_model.', 'first_stage_model.',
                                    'single_transformer_blocks.', 'joint_transformer_blocks.')):
        raise ValueError('FLUX tensor namespace has not been prepared for City96')
    if shape[-1] % 256:
        return gguf.GGMLQuantizationType.F16
    # City96 promotes every matching tensor here; text-model layer heuristics do not apply.
    promoted = ('attn_v.weight', '.to_v.weight', '.v.weight', '.attn.w1v.weight',
                '.attn.w2v.weight', '_attn.v_proj.weight', 'attn_qkv.weight',
                'attn.qkv.weight', 'attention.qkv.weight', 'ffn_down', '.ffn.2.weight',
                '.ff.net.2.weight', '.mlp.layer2.weight', '.adaln_modulation_mlp.2.weight',
                '.feed_forward.w2.weight')
    if preset == 'Q6_K' or any(part in name for part in promoted) or (
            'experts.' in name and '.w2.weight' in name):
        return gguf.GGMLQuantizationType.Q6_K
    return gguf.GGMLQuantizationType.Q5_K


def requantize(source, destination, *, cancel_event=None, on_progress=None, overwrite=False, preset="Q6_K"):
    """Atomically requantize a little-endian, single-file FLUX F16/F32 GGUF.

    Preserve metadata and tensor order; only file type/quantization version are
    updated. Use bounded copies and Q5/Q6 chunks, preserving noneligible bytes.
    Reject quantized/BF16 input, sidecars, shards, empty tensors and leading unit
    dimensions requiring GGML rank canonicalization. No source reconstruction,
    shape repair, tokenizer conversion or runtime compatibility is inferred.
    Existing output requires overwrite=True and is replaced only after readback;
    failure or cancellation preserves it. Source aliases are always rejected.
    """
    if preset not in ("Q6_K", "Q5_K_M"):
        raise ValueError("FLUX research presets are Q6_K and Q5_K_M only")
    source, destination = Path(source), Path(destination)
    validate_output(source, destination, overwrite)
    before = source.stat()
    with open_gguf(source) as reader:
        if reader.endianess != gguf.GGUFEndian.LITTLE:
            raise ValueError('Research FLUX path requires little-endian input')
        arch = reader.get_field('general.architecture')
        if arch is None or arch.contents() != 'flux':
            raise ValueError('Research Q6_K path supports FLUX only')
        if any(name.startswith('split.') or name == 'comfy.gguf.5d_sidecar' for name in reader.fields):
            raise ValueError('Shards and tensor sidecars require separate integration')
        if not reader.tensors:
            raise ValueError('Source has no tensors')
        plan = []
        for tensor in reader.tensors:
            shape = list(reversed(tensor.shape.tolist()))
            if (tensor.tensor_type not in (gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32)
                    or not shape or len(shape) > 4 or min(shape) <= 0 or (len(shape) > 1 and shape[0] == 1)):
                raise ValueError('Require canonical nonempty F16/F32 tensors; rank repair is not implemented')
            target = flux_q6_type(tensor.name, shape, tensor.tensor_type, preset=preset)
            size = tensor.n_elements // 256 * gguf.GGML_QUANT_SIZES[target][1] if target in (
                gguf.GGMLQuantizationType.Q6_K, gguf.GGMLQuantizationType.Q5_K) else (
                tensor.n_elements * 2 if target == gguf.GGMLQuantizationType.F16 else tensor.n_bytes)
            plan.append({'name': tensor.name, 'shape': shape, 'source_type': tensor.tensor_type.name,
                         'type': target.name, 'bytes': size})
        _cancel(cancel_event)
        with atomic_output(destination, overwrite=overwrite) as temporary:
            writer = gguf.GGUFWriter(temporary, 'flux')
            try:
                writer.add_custom_alignment(reader.alignment)
                for name, field in reader.fields.items():
                    if name.startswith('GGUF.') or name in ('general.architecture', 'general.file_type',
                                                           'general.quantization_version', 'general.alignment'):
                        continue
                    if len(field.types) > 2:
                        raise ValueError('Nested GGUF metadata arrays are not supported')
                    writer.add_key_value(name, field.contents(), field.types[0],
                                         field.types[1] if len(field.types) == 2 else None)
                writer.add_file_type(gguf.LlamaFileType.MOSTLY_Q6_K if preset == "Q6_K"
                                     else gguf.LlamaFileType.MOSTLY_Q5_K_M)
                writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
                for row in plan:
                    writer.add_tensor_info(row['name'], row['shape'], np.dtype(np.float32), row['bytes'],
                                           raw_dtype=gguf.GGMLQuantizationType[row['type']])
                writer.write_header_to_file()
                writer.write_kv_data_to_file()
                writer.write_ti_data_to_file()
                if writer.fout is None or len(writer.fout) != 1:
                    raise ValueError('Research path requires a single output stream')
                output = writer.fout[0]
                writer.write_padding(output, output.tell())
                for index, (tensor, row) in enumerate(zip(reader.tensors, plan)):
                    digest, written = hashlib.sha256(), 0
                    # Flattening a reader memmap is a view; only each window is expanded.
                    values = tensor.data.reshape(-1)
                    for start in range(0, values.size, 65536):
                        _cancel(cancel_event)
                        chunk = values[start:start + 65536]
                        if not np.isfinite(chunk).all():
                            raise ValueError('Source contains nonfinite values')
                        if row['type'] == 'Q6_K':
                            payloads = iter_q6_k(chunk.astype(np.float32))
                        elif row['type'] == 'Q5_K':
                            payloads = iter_q5_k(chunk.astype(np.float32))
                        elif row['type'] != row['source_type']:
                            with np.errstate(over='raise', invalid='raise'):
                                payloads = [chunk.astype('<f2')]
                        else:
                            payloads = [chunk]
                        for packed in payloads:
                            _cancel(cancel_event)
                            payload = packed.tobytes()
                            output.write(payload)
                            digest.update(payload)
                            written += len(payload)
                    if written != row['bytes']:
                        raise ValueError('Incomplete tensor write')
                    row['payload_sha256'] = digest.hexdigest()
                    writer.write_padding(output, written)
                    if on_progress is not None:
                        on_progress(index + 1, len(plan), tensor.name)
            finally:
                writer.close()
            after = source.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError('Source changed during requantization')
            # Mixed-precision readback must preserve float bytes as well as Q6 blocks.
            with open_gguf(temporary) as check:
                if len(check.tensors) != len(plan):
                    raise ValueError('GGUF tensor count mismatch')
                for tensor, row in zip(check.tensors, plan):
                    if (tensor.name != row['name'] or tensor.tensor_type.name != row['type']
                            or tensor.shape.tolist()[::-1] != row['shape'] or tensor.n_bytes != row['bytes']
                            or tensor.data_offset % check.alignment):
                        raise ValueError('GGUF tensor layout mismatch')
                    digest = hashlib.sha256()
                    data = tensor.data.view(np.uint8).reshape(-1)
                    for start in range(0, data.size, 1024 * 1024):
                        _cancel(cancel_event)
                        digest.update(data[start:start + 1024 * 1024].tobytes())
                    if digest.hexdigest() != row['payload_sha256']:
                        raise ValueError('GGUF tensor readback mismatch')
            checksum = _file_hash(temporary, cancel_event)
            _cancel(cancel_event)
    return {'status': f'Experimental FLUX {preset} requantizer; native remains the default',
            'output': str(destination), 'source': str(source), 'file_sha256': checksum, 'tensors': plan}


def main():
    """Run the explicit research-only float-GGUF conversion and print its report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--preset', choices=('Q6_K', 'Q5_K_M'), default='Q6_K')
    parser.add_argument('--overwrite', action='store_true', help='Atomically replace output after successful validation')
    args = parser.parse_args()
    print(json.dumps(requantize(args.source, args.output, overwrite=args.overwrite, preset=args.preset), indent=2))


if __name__ == '__main__':
    main()
