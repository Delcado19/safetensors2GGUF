"""Probe real quantizer preset decisions with small schematic GGUF fixtures."""
import argparse
import ast
from fnmatch import fnmatchcase
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from types import SimpleNamespace
import tarfile

import gguf
import numpy as np
import torch

from scripts.compare_native_reference_vectors import ROOT, NATIVE_SOURCES, extract_source, run_command
from scripts.inventory_converter_sources import REFERENCES, verify_references


PRESETS = ('Q2_K', 'Q3_K_M', 'Q4_K_S', 'Q4_K_M', 'Q5_K_M', 'Q6_K')


def apply_city96_patch(source, patch_path):
    """Apply outside the parent Git repository and reject a silently skipped patch."""
    env = os.environ.copy()
    env['GIT_CEILING_DIRECTORIES'] = str(source.parent.resolve())
    for arguments in (['--check'], []):
        result = run_command(['git', 'apply', *arguments, str(patch_path.resolve())],
                             cwd=source, text=True, env=env)
        if result.returncode:
            raise RuntimeError(f'City96 patch failed: {result.stderr}')
    if 'static ggml_type img_tensor_get_type' not in (source / 'src/llama.cpp').read_text(encoding='utf-8'):
        raise RuntimeError('City96 patch did not materialize in the isolated source tree')


def build_quantizer(entry, work, cmake, manifest):
    """Build the pinned CLI; apply only the preserved City96 patch to b3962."""
    source, build = work / 'source', work / 'build'
    extract_source(entry, source)
    env = os.environ.copy()
    # Also isolate git apply: otherwise Git can silently skip paths below our repo.
    env['GIT_CEILING_DIRECTORIES'] = str(work.resolve())
    patch_sha = None
    patched_files = {}
    if entry['id'] == 'llama-b3962':
        city = next(item for item in manifest['sources'] if item['id'] == 'city96-patch')
        with tarfile.open(REFERENCES / city['archive']) as archive:
            patch = archive.extractfile(city['archive_root'] + '/tools/lcpp.patch').read()
        patch_path = work / 'city96.patch'
        patch_path.write_bytes(patch)
        patch_sha = hashlib.sha256(patch).hexdigest()
        apply_city96_patch(source, patch_path)
        patched_files = {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                         for name in ('ggml/include/ggml.h', 'ggml/src/ggml.c', 'src/llama.cpp')}
    options = ['BUILD_SHARED_LIBS', 'GGML_NATIVE', 'GGML_AVX', 'GGML_AVX2', 'GGML_AVX512',
               'GGML_FMA', 'GGML_F16C', 'GGML_OPENMP', 'GGML_CUDA', 'GGML_METAL', 'GGML_BLAS',
               'GGML_VULKAN', 'GGML_KOMPUTE', 'GGML_CCACHE', 'LLAMA_CURL', 'LLAMA_OPENSSL',
               'LLAMA_BUILD_TESTS', 'LLAMA_BUILD_SERVER']
    commands = [[cmake, '-S', str(source), '-B', str(build), '-DCMAKE_BUILD_TYPE=Release',
                 *[f'-D{option}=OFF' for option in options]],
                [cmake, '--build', str(build), '--config', 'Release', '--target', 'llama-quantize', '-j', '2']]
    if os.name == 'nt' and entry['id'] == 'llama-b3962':
        # b3962 relied on transitive <chrono> includes removed in MSVC 19.51.
        header = (ROOT / 'tools/converter-reference/compat-chrono.h').as_posix()
        commands[0].append(f'-DCMAKE_CXX_FLAGS=/FI"{header}"')
    for index, command in enumerate(commands):
        result = run_command(command, text=True, env=env)
        log = work / f'build-{index}.log'
        log.write_text(result.stdout + result.stderr, encoding='utf-8')
        if result.returncode:
            raise RuntimeError(f'Policy quantizer build failed: {log}')
    executable = next(path for path in build.rglob('llama-quantize.exe' if os.name == 'nt' else 'llama-quantize')
                      if path.is_file())
    compiler = next(build.glob('CMakeFiles/*/CMakeCCompiler.cmake')).read_text(encoding='utf-8')
    version = re.search(r'set\(CMAKE_C_COMPILER_VERSION "([^"]+)"\)', compiler).group(1)
    cache = (build / 'CMakeCache.txt').read_text(encoding='utf-8')
    return {'id': entry['id'], 'commit': entry['commit'], 'archive_sha256': entry['sha256'],
            'city96_patch_sha256': patch_sha, 'patched_files_sha256': patched_files,
            'compatibility_header_sha256': hashlib.sha256(
                (ROOT / 'tools/converter-reference/compat-chrono.h').read_bytes()).hexdigest()
                if os.name == 'nt' and entry['id'] == 'llama-b3962' else None,
            'compiler_version': version, 'commands': commands,
            'flags': [line for line in cache.splitlines() if '=' in line and line.startswith(
                ('GGML_', 'LLAMA_', 'BUILD_SHARED_LIBS:', 'CMAKE_C_FLAGS', 'CMAKE_CXX_FLAGS'))],
            'executable': str(executable), 'executable_sha256': hashlib.sha256(executable.read_bytes()).hexdigest()}


def fixture_tensors(architecture, reverse=False, incompatible=False):
    """Make ordered tensor-name/shape probes, not an inference-capable model."""
    if architecture in ('flux', 'sd3', 'sdxl', 'wan'):
        tensors = [(f'blocks.{layer}.attn.to_v.weight', (8, 256)) for layer in range(8)]
        tensors += [('blocks.0.attn.qkv.weight', (8, 256)), ('blocks.0.ffn_down.weight', (8, 256)),
                    ('blocks.0.attn_output.weight', (8, 256)), ('img_in.weight', (8, 256)),
                    ('patch_embedding.weight', (8, 256)), ('norm.weight', (256,)),
                    ('small.weight', (4, 256)), ('spatial.weight', (2, 2, 8, 256))]
    elif architecture == 't5':
        tensors = []
        for layer in range(8):
            for side in ('enc', 'dec'):
                tensors.extend((f'{side}.blk.{layer}.{role}.weight', (8, 256))
                               for role in ('attn_v', 'attn_output', 'ffn_down'))
            tensors.append((f'dec.blk.{layer}.cross_attn_v.weight', (8, 256)))
        tensors += [('token_embd.weight', (8, 256)), ('output.weight', (8, 256)),
                    ('enc.blk.0.attn_norm.weight', (256,)), ('enc.blk.0.attn_rel_b.weight', (8, 32))]
    else:
        tensors = []
        for layer in range(8):
            tensors.extend((f'blk.{layer}.{role}.weight', (8, 256))
                           for role in ('attn_v', 'attn_output', 'ffn_down'))
        tensors += [('token_embd.weight', (8, 256)), ('output.weight', (8, 256)),
                    ('blk.0.attn_norm.weight', (256,)), ('blk.0.attn_rel_b.weight', (8, 32))]
    if incompatible:
        tensors.append(('odd.weight', (8, 160)))
    if reverse:
        tensors.reverse()
    return dict(tensors)


def write_fixture(path, architecture, tensors):
    """Write small F16 inputs with enough metadata to exercise CLI policy logic."""
    writer = gguf.GGUFWriter(str(path), architecture)
    writer.add_name('Schematic converter policy probe; not for inference')
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_F16)
    if architecture not in ('flux', 'sd3', 'sdxl', 'wan'):
        writer.add_context_length(512)
        # Old b3962 requires an explicit vocabulary count (or token list).
        writer.add_vocab_size(8)
        writer.add_token_list([f'<probe-{index}>' for index in range(8)])
        writer.add_embedding_length(256)
        writer.add_feed_forward_length(512)
        writer.add_block_count(8)
        writer.add_head_count(8)
        writer.add_head_count_kv(8)
        writer.add_layer_norm_rms_eps(1e-6)
        writer.add_rope_freq_base(10000.0)
        writer.add_key_length(32)
        writer.add_value_length(32)
        if architecture in ('t5', 't5encoder'):
            writer.add_relative_attn_buckets_count(32)
    try:
        for name, shape in tensors.items():
            data = np.linspace(-1, 1, math.prod(shape), dtype=np.float16).reshape(shape)
            writer.add_tensor(name, data)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=False)
    finally:
        writer.close()


def load_molbal_planner(manifest):
    """Execute selected declarations/helpers from an already verified manifest."""
    entry = next(item for item in manifest['sources'] if item['id'] == 'molbal-current')
    with tarfile.open(REFERENCES / entry['archive']) as archive:
        code = archive.extractfile(entry['archive_root'] + '/tools/convert.py').read().decode('utf-8')
    functions = {'key_matches', '_default_qtype', '_is_raw_byte_tensor', '_k_quant_category',
                 '_k_matches_arch_pattern', '_k_quant_effective_last_dim', 'plan_k_quantization'}
    constants = {'QUANTIZATION_THRESHOLD', 'REARRANGE_THRESHOLD', 'MAX_TENSOR_DIMS', 'RAW_BYTE_TENSOR_KEYS',
                 'QUANT_TYPE_MAP', 'K_QUANT_TYPE_NAMES', 'K_QUANT_PRESETS', 'K_QUANT_SELECTIONS', 'K_QUANT_BLOCK_SIZE'}
    classes = {'ModelTemplate', 'ModelFlux', 'ModelSD3', 'ModelSDXL', 'ModelHyVid', 'ModelWan', 'ModelKrea2'}
    nodes = []
    for node in ast.parse(code).body:
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            nodes.append(node)
        elif isinstance(node, ast.ClassDef) and node.name in classes:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and (
                target.id in constants or target.id.startswith('_K_')) for target in node.targets):
            nodes.append(node)
    namespace = {'gguf': gguf, 'torch': torch, 'fnmatchcase': fnmatchcase}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<verified Molbal K planner>', 'exec'), namespace)
    return namespace


def plan_molbal(namespace, architecture, tensors, preset):
    """Compare shape-only F16 inputs without allocating full checkpoint tensors."""
    if preset not in namespace['K_QUANT_SELECTIONS']:
        return {'status': 'unsupported_preset'}
    models = {'flux': 'ModelFlux', 'sd3': 'ModelSD3', 'sdxl': 'ModelSDXL', 'wan': 'ModelWan', 'krea2': 'ModelKrea2'}
    if architecture not in models:
        return {'status': 'no_equivalent_architecture'}
    state = {name: SimpleNamespace(shape=shape, dtype=torch.float16, numel=lambda shape=shape: math.prod(shape))
             for name, shape in tensors.items()}
    try:
        plan = namespace['plan_k_quantization'](state, namespace[models[architecture]](), preset)
    except ValueError as error:
        return {'status': 'rejected', 'error': str(error)}
    return {'status': 'planned', 'types': {name: qtype.name for name, qtype in plan.items()}}


def run_policies(builds, destination, manifest):
    """Run real CLI preset decisions and retain logs/files in the requested folder."""
    destination.mkdir(parents=True, exist_ok=False)
    namespace = load_molbal_planner(manifest)
    molbal = next(item for item in manifest['sources'] if item['id'] == 'molbal-current')
    report = {'status': 'Schematic quantizer-stage policy probes; not full-model or inference validation',
              'molbal': {'commit': molbal['commit'], 'archive_sha256': molbal['sha256'],
                         'file': 'tools/convert.py', 'entrypoint': 'plan_k_quantization'},
              'builds': builds, 'skipped': [], 'results': []}
    scenarios = [('flux', False, False), ('flux', True, False), ('flux', False, True),
                 ('sd3', False, False), ('sdxl', False, False), ('wan', False, False),
                 ('llama', False, False), ('t5', False, False), ('qwen3', False, False)]
    for architecture, reverse, incompatible in scenarios:
        label = architecture + ('-reverse' if reverse else '') + ('-unaligned' if incompatible else '')
        tensors = fixture_tensors(architecture, reverse, incompatible)
        source = destination / (label + '-F16.gguf')
        write_fixture(source, architecture, tensors)
        for build in builds:
            # Modern unpatched llama.cpp has no diffusion metadata/architecture path.
            if architecture in ('flux', 'sd3', 'sdxl', 'wan') and build['id'] != 'llama-b3962':
                report['skipped'].append({'scenario': label, 'source': build['id'],
                                          'reason': 'Unpatched text reference has no diffusion architecture path'})
                continue
            if architecture == 'qwen3' and build['id'] == 'llama-b3962':
                report['skipped'].append({'scenario': label, 'source': build['id'],
                                          'reason': 'Qwen3 architecture is absent from b3962'})
                continue
            for preset in PRESETS:
                output = destination / f'{label}-{build["id"]}-{preset}.gguf'
                command = [build['executable'], str(source), str(output), preset, '1']
                result = run_command(command, text=True)
                log = output.with_suffix('.log')
                log.write_text(result.stdout + result.stderr, encoding='utf-8')
                row = {'scenario': label, 'architecture': architecture, 'source': build['id'], 'preset': preset,
                       'exit_code': result.returncode, 'log': str(log),
                       'input_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                       'molbal': plan_molbal(namespace, architecture, tensors, preset)}
                if result.returncode:
                    row['status'] = 'failed'
                else:
                    reader = gguf.GGUFReader(str(output))
                    row.update(status='converted', types={tensor.name: tensor.tensor_type.name for tensor in reader.tensors},
                               output_bytes=output.stat().st_size, output_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
                report['results'].append(row)
        print(f'Compared policy scenario {label}', flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cmake', default=shutil.which('cmake'))
    parser.add_argument('--artifacts', type=Path, required=True, help='New directory for all synthetic GGUF files/logs')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.cmake:
        parser.error('Supply --cmake or add CMake to PATH')
    if args.artifacts.exists():
        parser.error('Artifact directory must be new; existing files are never overwritten')
    manifest = verify_references()
    root = ROOT / 'build' / 'converter-policy'
    root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='run-', dir=root))
    builds = []
    for sid in NATIVE_SOURCES:
        print(f'Building policy quantizer {sid}', flush=True)
        entry = next(item for item in manifest['sources'] if item['id'] == sid)
        builds.append(build_quantizer(entry, work / sid, args.cmake, manifest))
        (work / 'builds.json').write_text(json.dumps(builds, indent=2) + '\n', encoding='utf-8')
    report = run_policies(builds, args.artifacts, manifest)
    if any(row['status'] != 'converted' for row in report['results']):
        raise RuntimeError('Policy fixture conversion failed; inspect retained per-case logs')
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
