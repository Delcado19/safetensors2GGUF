"""Compare all frozen image architectures against the pinned native quantizer."""
import argparse
import hashlib
import json
from pathlib import Path
import queue
import runpy

from safetensors.torch import load_file, save_file

from compiled_diffusion import PROTECTED
from conversion_service import _pipeline
from convert import convert_file
from scripts.check_flux_q6_path import verified_flux_reference
from scripts.compare_gguf_outputs import compare_files
from scripts.compare_native_reference_vectors import run_command
from scripts.requantize_flux_q6 import FLUX_PRESETS


def main():
    """Record schematic application parity; no inference or full-model claims."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.work.exists() or args.output.exists():
        parser.error('Choose fresh work and output paths')
    root = Path(__file__).resolve().parents[1]
    reference = verified_flux_reference(json.loads((root / 'data/converter-policy-comparison.json').read_text()))
    source_file = runpy.run_path(str(root / 'tests/test_compiled_diffusion.py'))['source_file']
    args.work.mkdir(parents=True)
    rows = []
    for arch in PROTECTED:
        if arch == 'qwen_image':
            continue  # The frozen native reference predates this adapter.
        source = args.work / f'{arch}.safetensors'
        source_file(source, arch)
        tensors = load_file(source)
        tensors = {key: tensor.clone() for key, tensor in tensors.items() if tensor.ndim <= 4}
        if arch == 'hyvid':
            tensors.pop('probe.modulation', None)  # Not a native-preserved HyVid role.
        tensors.update({part + 'weight': tensors['probe.linear.weight'].clone() for part in PROTECTED[arch]})
        save_file(tensors, str(source))
        prepared = args.work / f'{arch}-F16.gguf'
        convert_file(str(source), str(prepared), interact=False, apply_unsqueeze=False, on_log=lambda _: None)
        for preset in FLUX_PRESETS:
            native, ours = (args.work / f'{arch}-{preset}-{suffix}.gguf' for suffix in ('native', 'bundled'))
            result = run_command([reference['executable'], str(prepared), str(native), preset, '1'], text=True)
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
            _pipeline(str(source), str(ours), preset, '', 0, False, False, queue.Queue(),
                      gguf_backend='compiled_diffusion')
            comparison = compare_files(ours, native)
            # Native collapses Lumina [1,D] pads; the application repairs that known
            # loader contract after native quantization, while direct writing keeps it.
            if arch == 'lumina2':
                from fix_pad_tokens import fix_pad_tokens
                fixed = native.with_suffix('.fixed.gguf')
                fix_pad_tokens(str(native), str(fixed), on_log=lambda _: None)
                comparison = compare_files(ours, fixed)
            rows.append({'architecture': arch, 'preset': preset, 'comparison': comparison})
            print(arch, preset, comparison['all_equal'], flush=True)
    report = {'scope': 'Schematic source-to-GGUF application parity; not inference or full checkpoints.',
              'native_reference': reference, 'results': rows,
              'all_equal': all(row['comparison']['all_equal'] for row in rows),
              'source_code_sha256': {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in (
                  'compiled_diffusion.py', 'compiled_flux_writer.py', 'convert.py', 'conversion_service.py',
                  'web_api.py', 'frontend/src/main.tsx', 'scripts/requantize_flux_q6.py')}}
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf8')
    raise SystemExit(0 if report['all_equal'] else 1)


if __name__ == '__main__':
    main()
