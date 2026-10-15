"""Package verified raw Q6_K tensors into a research-only GGUF and check readback."""
import argparse
import hashlib
import json
from pathlib import Path

import gguf
import numpy as np

from conversion_io import atomic_output, validate_output
from convert import ConversionCancelled


def _cancel(event):
    if event is not None and event.is_set():
        raise ConversionCancelled('Q6_K container check cancelled')


def _file_hash(path, cancel_event=None):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            _cancel(cancel_event)
            digest.update(chunk)
    return digest.hexdigest()


def _verify_container(path, records, cancel_event):
    # Keep access to the mmap even if parsing fails, so Windows can remove temps.
    reader = gguf.GGUFReader.__new__(gguf.GGUFReader)
    try:
        reader.__init__(str(path))
        if len(reader.tensors) != len(records):
            raise ValueError('GGUF tensor count mismatch')
        results = []
        for tensor, record in zip(reader.tensors, records):
            _cancel(cancel_event)
            report = record['report']
            if (tensor.name != report['tensor'] or list(reversed(tensor.shape)) != report['shape']
                    or tensor.tensor_type != gguf.GGMLQuantizationType.Q6_K
                    or tensor.n_bytes != report['output_bytes'] or tensor.data_offset % reader.alignment):
                raise ValueError('GGUF tensor layout mismatch')
            digest = hashlib.sha256()
            data = tensor.data.reshape(-1)
            for start in range(0, data.size, 1024 * 1024):
                _cancel(cancel_event)
                digest.update(data[start:start + 1024 * 1024].tobytes())
            if digest.hexdigest() != report['payload_sha256']:
                raise ValueError('GGUF tensor payload checksum mismatch')
            results.append({'name': tensor.name, 'shape': report['shape'],
                            'type': tensor.tensor_type.name, 'bytes': tensor.n_bytes,
                            'data_offset': tensor.data_offset, 'payload_sha256': digest.hexdigest()})
        return results
    finally:
        mapping = getattr(reader, 'data', None)
        if mapping is not None:
            mapping._mmap.close()


def package_tensors(report_paths, destination, *, cancel_event=None):
    """Copy verified raw tensor streams into one atomic GGUF with bounded buffers.

    Reports supply trusted tensor semantics; raw byte count and SHA256 are checked
    independently, during copying and after GGUF readback. This mixed-source test
    container deliberately uses an unknown research architecture, not model
    metadata. Never overwrite reports, source tensors or an existing destination.
    Return container layout/checksum evidence without including weights.
    """
    destination = Path(destination)
    if destination.suffix != '.gguf':
        raise ValueError('Container output must use .gguf')
    records, names = [], set()
    for path in map(Path, report_paths):
        _cancel(cancel_event)
        validate_output(path, destination)
        report_bytes = path.read_bytes()
        report = json.loads(report_bytes)
        shape = report['shape']
        if (report['all_equal'] is not True or len(shape) != 2
                or any(type(dimension) is not int or dimension <= 0 for dimension in shape)
                or shape[1] % 256 or report['source_dtype'] not in ('F16', 'BF16', 'F32')
                or report['output_bytes'] != shape[0] * shape[1] // 256 * 210):
            raise ValueError('Require a passing plain-matrix Q6_K stream report')
        if not report['tensor'] or report['tensor'] in names:
            raise ValueError('Tensor names must be nonempty and unique')
        names.add(report['tensor'])
        raw = Path(report['output'])
        if not raw.is_absolute():
            raw = path.parent / raw
        validate_output(raw, destination)
        if raw.stat().st_size != report['output_bytes'] or _file_hash(raw, cancel_event) != report['payload_sha256']:
            raise ValueError('Raw Q6_K input size/checksum mismatch')
        records.append({'report': report, 'raw': raw, 'report_sha256': hashlib.sha256(report_bytes).hexdigest()})
    if not records:
        raise ValueError('At least one verified tensor report is required')
    _cancel(cancel_event)
    with atomic_output(destination) as temporary:
        writer = gguf.GGUFWriter(temporary, 's2gguf_q6_research')
        try:
            writer.add_name('Research Q6_K tensor container; not a complete model')
            writer.add_bool('s2gguf.research_only', True)
            for index, record in enumerate(records):
                report = record['report']
                packed_shape = (report['shape'][0], report['shape'][1] // 256 * 210)
                writer.add_tensor_info(report['tensor'], packed_shape, np.dtype(np.uint8),
                                       report['output_bytes'], raw_dtype=gguf.GGMLQuantizationType.Q6_K)
                writer.add_string(f's2gguf.source_report_sha256.{index}', record['report_sha256'])
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_ti_data_to_file()
            # The library writes whole tensor arrays; use its header/alignment API
            # and copy raw blocks directly to avoid allocating complete tensors.
            if writer.fout is None or len(writer.fout) != 1:
                raise ValueError('Research container requires a single GGUF output stream')
            output = writer.fout[0]
            writer.write_padding(output, output.tell())
            for record in records:
                digest, copied = hashlib.sha256(), 0
                with record['raw'].open('rb') as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                        _cancel(cancel_event)
                        output.write(chunk)
                        digest.update(chunk)
                        copied += len(chunk)
                if copied != record['report']['output_bytes'] or digest.hexdigest() != record['report']['payload_sha256']:
                    raise ValueError('Raw Q6_K input changed during container copy')
                writer.write_padding(output, copied)
        finally:
            writer.close()
        tensors = _verify_container(temporary, records, cancel_event)
        container_hash = _file_hash(temporary, cancel_event)
        _cancel(cancel_event)
    return {'status': 'GGUF container roundtrip only; no model conversion or loader compatibility claim',
            'output': str(destination), 'file_sha256': container_hash,
            'architecture': 's2gguf_q6_research', 'tensor_count': len(tensors), 'tensors': tensors,
            'source_report_sha256': [record['report_sha256'] for record in records], 'all_equal': True,
            'checker_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def main():
    """Package explicit tensor reports and write separate container evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    for source in [args.output, *args.inputs]:
        validate_output(source, args.report)
    # Also protect the raw inputs from a report destination that aliases them.
    for path in args.inputs:
        raw = Path(json.loads(path.read_text(encoding='utf-8'))['output'])
        validate_output(raw if raw.is_absolute() else path.parent / raw, args.report)
    result = package_tensors(args.inputs, args.output)
    with atomic_output(args.report) as temporary:
        Path(temporary).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
