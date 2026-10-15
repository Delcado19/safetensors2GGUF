"""Compare GGUF tensor layouts, metadata and payloads without full-tensor copies."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from conversion_io import atomic_output, validate_output
from scripts.check_q6_gguf_container import _cancel, _file_hash
from scripts.gguf_research_io import open_gguf


def compare_files(left_path, right_path, *, cancel_event=None):
    """Compare ordered tensors with 1 MiB windows and typed metadata values.

    Ignore GGUF bookkeeping and the optional explicit alignment key, but require
    equal effective alignment. File hashes are evidence, not an equality gate:
    metadata ordering and explicit default keys can change whole-file bytes.
    Readers are mapped; bounded copies do not promise bounded process peak RSS.
    """
    paths = [Path(left_path), Path(right_path)]
    before = [path.stat() for path in paths]
    with open_gguf(paths[0]) as left, open_gguf(paths[1]) as right:

        def metadata(reader):
            return {key: (field.types, field.contents()) for key, field in reader.fields.items()
                    if not key.startswith('GGUF.') and key != 'general.alignment'}

        result = {'left': str(paths[0]), 'right': str(paths[1]), 'tensors': [],
                  'comparison_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'metadata_equal': metadata(left) == metadata(right),
                  'alignment_equal': bool(left.alignment == right.alignment),
                  'tensor_counts_equal': len(left.tensors) == len(right.tensors)}
        for a, b in zip(left.tensors, right.tensors):
            _cancel(cancel_event)
            hashes = [hashlib.sha256(), hashlib.sha256()]
            buffers = [tensor.data.view(np.uint8).reshape(-1) for tensor in (a, b)]
            same = (a.name == b.name and a.tensor_type == b.tensor_type and np.array_equal(a.shape, b.shape)
                    and a.n_bytes == b.n_bytes)
            # Read both complete payloads even on mismatch, retaining useful hashes.
            for start in range(0, max(a.n_bytes, b.n_bytes), 1024 * 1024):
                _cancel(cancel_event)
                windows = [data[start:start + 1024 * 1024] for data in buffers]
                for digest, window in zip(hashes, windows):
                    digest.update(window.tobytes())
                same = same and np.array_equal(*windows)
            result['tensors'].append({'name': a.name, 'right_name': b.name, 'type': a.tensor_type.name,
                                      'right_type': b.tensor_type.name, 'shape': a.shape.tolist()[::-1],
                                      'bytes': a.n_bytes, 'left_sha256': hashes[0].hexdigest(),
                                      'right_sha256': hashes[1].hexdigest(), 'equal': bool(same)})
        result['left_sha256'], result['right_sha256'] = [_file_hash(path, cancel_event) for path in paths]
        for path, original in zip(paths, before):
            after = path.stat()
            if (original.st_size, original.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError('GGUF changed during comparison')
        result['all_equal'] = bool(left.tensors) and all(result[key] for key in
                               ('metadata_equal', 'alignment_equal', 'tensor_counts_equal')) and all(
                                   row['equal'] for row in result['tensors'])
        return result


def main():
    """Write comparison evidence atomically and fail on any semantic mismatch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--left', type=Path, required=True)
    parser.add_argument('--right', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    validate_output(args.left, args.output)
    validate_output(args.right, args.output)
    result = compare_files(args.left, args.right)
    with atomic_output(args.output) as temporary:
        Path(temporary).write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    raise SystemExit(0 if result['all_equal'] else 1)


if __name__ == '__main__':
    main()
