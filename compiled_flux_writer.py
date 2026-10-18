"""Final-payload GGUF spool fed by the existing FLUX float-preparation path."""
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import hashlib
from pathlib import Path
from time import perf_counter

import gguf
import numpy as np

from scripts import requantize_flux_q6 as quant
from scripts.check_q6_gguf_container import _cancel, _file_hash
from scripts.gguf_research_io import open_gguf


class CompiledFluxWriter(gguf.GGUFWriter):
    """Encode prepared F16/F32 tensors before spooling, validate before publication.

    Used only by convert_file's internal writer factory. It retains that function's
    dtype coercion, FP8 reconstruction, clamping, ordering and precision protection.
    At most one encoded tensor plus bounded parallel input batches are retained.
    """

    def __init__(self, path, arch, use_temp_file, *, kernel, preset, source, cancel_event=None, on_log=None):
        if arch != 'flux' or preset not in quant.FLUX_PRESETS or not use_temp_file:
            raise ValueError('Direct compiled writer requires FLUX, a supported K preset and a final spool')
        super().__init__(path, arch, use_temp_file=use_temp_file)
        # Match the existing requantizer's explicit alignment and metadata order.
        self.add_custom_alignment(gguf.GGUF_DEFAULT_ALIGNMENT)
        self.kernel, self.preset = kernel, preset
        self.source = Path(source)
        self.before = self.source.stat()
        self.cancel_event, self.on_log = cancel_event, on_log
        self.plan = []
        self.attention_v_index = 0
        self.prepared_bytes = self.final_bytes = 0
        self.started = perf_counter()

    def add_quantization_version(self, version):
        """Defer version insertion to preserve the legacy K-container byte layout."""
        self._quant_version = version

    def add_file_type(self, _prepared_type):
        """Describe the final K preset rather than the internal F16 preparation."""
        super().add_file_type(getattr(gguf.LlamaFileType, 'MOSTLY_' + self.preset))
        super().add_quantization_version(self._quant_version)

    def add_tensor(self, name, data, *, raw_dtype=None):
        """Consume one canonical prepared tensor and spool its final packed bytes."""
        _cancel(self.cancel_event)
        shape = list(data.shape)
        if (raw_dtype not in (gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32)
                or not shape or len(shape) > 4 or min(shape) <= 0
                or (len(shape) > 1 and shape[0] == 1)):
            raise ValueError('Direct compiled writer requires canonical nonempty F16/F32 tensors')
        target = quant.flux_q6_type(name, shape, raw_dtype, preset=self.preset,
                                   attention_v_index=self.attention_v_index)
        if quant._flux_quantizable(name, shape) and any(part in name for part in quant._ATTENTION_V):
            self.attention_v_index += 1
        self.prepared_bytes += data.nbytes
        values = data.reshape(-1)
        if target.name in ('Q2_K', 'Q3_K', 'Q4_K', 'Q5_K', 'Q6_K'):
            size = gguf.GGML_QUANT_SIZES[target][1]
            packed = np.empty(values.size // 256 * size, dtype=np.uint8)
            offset = 0
            with ThreadPoolExecutor(max_workers=quant._COMPILED_WORKERS) as pool:
                for start in range(0, values.size, 65536 * quant._COMPILED_WORKERS):
                    _cancel(self.cancel_event)
                    chunk = values[start:start + 65536 * quant._COMPILED_WORKERS]
                    if not np.isfinite(chunk).all():
                        raise ValueError('Prepared tensor contains nonfinite values')
                    inputs = [chunk[i:i + 65536].astype(np.float32).reshape(-1, 256)
                              for i in range(0, chunk.size, 65536)]
                    encode = partial(quant._compiled_chunk, self.kernel, int(target.name[1]))
                    results = map(encode, inputs) if len(inputs) == 1 else pool.map(encode, inputs)
                    for result in results:
                        _cancel(self.cancel_event)
                        packed[offset:offset + result.size] = result.reshape(-1)
                        offset += result.size
            if offset != packed.size:
                raise ValueError('Incomplete direct compiled tensor')
            data = packed.reshape(*shape[:-1], shape[-1] // 256 * size)
        else:
            for start in range(0, values.size, 65536):
                _cancel(self.cancel_event)
                if not np.isfinite(values[start:start + 65536]).all():
                    raise ValueError('Prepared tensor contains nonfinite values')
            if target != raw_dtype:
                with np.errstate(over='raise', invalid='raise'):
                    data = data.astype('<f2')
        data = np.ascontiguousarray(data)
        self.plan.append({'name': name, 'shape': shape, 'type': target,
                          'bytes': data.nbytes, 'sha256': hashlib.sha256(data).hexdigest()})
        self.final_bytes += data.nbytes
        super().add_tensor(name, data, raw_dtype=target)

    def close(self):
        """Read back all final tensors while convert_file still owns atomic output."""
        written = self.fout is not None and self.state is gguf.WriterState.WEIGHTS
        super().close()
        if not written:
            return
        if not self.plan:
            raise ValueError('Direct compiled writer has no tensors')
        after = self.source.stat()
        if (self.before.st_size, self.before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('Source changed during direct compiled conversion')
        if self.on_log:
            self.on_log(f'INFO: Direct preparation/encode/spool/final-write: {perf_counter() - self.started:.3f} s')
        started = perf_counter()
        with open_gguf(self.path) as reader:
            if len(reader.tensors) != len(self.plan):
                raise ValueError('Direct GGUF tensor count mismatch')
            for tensor, row in zip(reader.tensors, self.plan):
                _cancel(self.cancel_event)
                if (tensor.name != row['name'] or tensor.tensor_type != row['type']
                        or tensor.shape.tolist()[::-1] != row['shape'] or tensor.n_bytes != row['bytes']
                        or tensor.data_offset % reader.alignment):
                    raise ValueError('Direct GGUF tensor layout mismatch')
                digest = hashlib.sha256()
                values = tensor.data.view(np.uint8).reshape(-1)
                for start in range(0, values.size, 1024 * 1024):
                    _cancel(self.cancel_event)
                    digest.update(values[start:start + 1024 * 1024])
                if digest.hexdigest() != row['sha256']:
                    raise ValueError('Direct GGUF tensor readback mismatch')
        if self.on_log:
            self.on_log(f'INFO: Direct tensor readback: {perf_counter() - started:.3f} s')
        started = perf_counter()
        checksum = _file_hash(self.path, self.cancel_event)
        _cancel(self.cancel_event)
        if self.on_log:
            self.on_log(f'INFO: Direct file hash: {perf_counter() - started:.3f} s; SHA256 {checksum}')
            self.on_log(f'INFO: Prepared bytes {self.prepared_bytes}; final spool bytes {self.final_bytes}; no float GGUF intermediate')
