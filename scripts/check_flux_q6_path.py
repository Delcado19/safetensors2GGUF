"""Compare the research FLUX Q6_K file path with pinned patched b3962."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np

from scripts.compare_converter_policies import fixture_tensors, write_fixture
from scripts.compare_native_reference_vectors import ROOT, run_command
from scripts.inventory_converter_sources import verify_references, REFERENCES
from scripts.requantize_flux_q6 import requantize
from scripts.compare_gguf_outputs import compare_files


def verified_flux_reference(references):
    """Return the pinned patched b3962 build after archive/executable checks."""
    manifest = verify_references()
    build = next(build for build in references['builds'] if build['id'] == 'llama-b3962')
    source = next(item for item in manifest['sources'] if item['id'] == 'llama-b3962')
    patch = next(item for item in manifest['sources'] if item['id'] == 'city96-patch')
    with tarfile.open(REFERENCES / patch['archive']) as archive:
        member = next(item for item in archive.getmembers() if item.name.endswith('/lcpp.patch'))
        patch_hash = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
    executable = Path(build['executable'])
    if (build['commit'] != source['commit'] or build['archive_sha256'] != source['sha256']
            or build['city96_patch_sha256'] != patch_hash
            or hashlib.sha256(executable.read_bytes()).hexdigest() != build['executable_sha256']):
        raise ValueError('Patched native reference provenance mismatch')
    return build


def compare_path(artifacts, references):
    """Require pinned provenance, then compare types/shapes/bytes and all metadata."""
    build = verified_flux_reference(references)
    executable = Path(build['executable'])
    artifacts.mkdir(parents=True, exist_ok=False)
    report = {'status': 'FLUX quantizer-stage file parity; schematic inputs, not inference models',
              'native': {key: build[key] for key in ('commit', 'archive_sha256', 'city96_patch_sha256',
                                                    'executable_sha256')},
              'encoder_sha256': hashlib.sha256((ROOT / 'scripts/reference_q6_k.py').read_bytes()).hexdigest(),
              'requantizer_sha256': hashlib.sha256((ROOT / 'scripts/requantize_flux_q6.py').read_bytes()).hexdigest(),
              'results': []}
    for dtype in (np.float16, np.float32):
        for reverse in (False, True):
            for odd in (False, True):
                label = f'{np.dtype(dtype).name}-reverse-{reverse}-unaligned-{odd}'
                src, ours, native = [artifacts / f'{label}-{name}.gguf' for name in ('input', 'python', 'native')]
                tensors = fixture_tensors('flux', reverse, odd)
                # Cover every architecture protection and general exclusion used here.
                tensors.update({f'{name}.weight': (8, 256) for name in
                                ('txt_in', 'time_in', 'vector_in', 'guidance_in', 'final_layer',
                                 'blocks.0.attn_norm', 'blocks.0.ffn_gate_inp', 'blocks.0.attn_rel_b')})
                write_fixture(src, 'flux', tensors, dtype=dtype)
                requantize(str(src), str(ours))
                result = run_command([str(executable), str(src), str(native), 'Q6_K', '1'], text=True)
                native.with_suffix('.log').write_text(result.stdout + result.stderr, encoding='utf-8')
                if result.returncode:
                    raise RuntimeError(f'Native conversion failed: {native.with_suffix(".log")}')
                comparison = compare_files(ours, native)
                report['results'].append({'scenario': label, 'tensors': comparison['tensors'],
                                          'input_sha256': hashlib.sha256(src.read_bytes()).hexdigest(),
                                          'python_sha256': comparison['left_sha256'],
                                          'native_sha256': comparison['right_sha256'],
                                          'metadata_equal': comparison['metadata_equal'],
                                          'alignment_equal': comparison['alignment_equal'],
                                          'tensor_counts_equal': comparison['tensor_counts_equal']})
    report['all_equal'] = all(row['metadata_equal'] and row['alignment_equal'] and row['tensor_counts_equal']
                              and all(tensor['equal'] for tensor in row['tensors']) for row in report['results'])
    return report


def main():
    """Write native/Python fixture artifacts into a fresh explicit directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--references', type=Path, default=ROOT / 'data/converter-policy-comparison.json')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new evidence report path')
    report = compare_path(args.artifacts, json.loads(args.references.read_text()))
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    raise SystemExit(0 if report['all_equal'] else 1)


if __name__ == '__main__':
    main()
