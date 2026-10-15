"""Resolve frozen model declarations and locate precision policies without imports."""
import argparse
import ast
import hashlib
import json
import posixpath
import tarfile
from pathlib import Path

from scripts.inventory_converter_sources import REFERENCES, verify_references


def effective_rules(modules, selected=None):
    """Resolve class attributes/method owners with C3 order across local imports.

    Values remain source expressions. Unknown bases and dynamic class-body writes
    are reported explicitly; this is not execution of constructor or method code.
    ``modules`` maps relative Python filenames to source strings.
    """
    classes, imports = {}, {}

    def register_class(filename, node, parent=''):
        name = f'{parent}.{node.name}' if parent else node.name
        classes[filename, name] = node
        for child in node.body:
            if isinstance(child, ast.ClassDef):
                register_class(filename, child, name)

    for filename, code in modules.items():
        imports[filename] = {}
        for node in ast.parse(code).body:
            if isinstance(node, ast.ImportFrom):
                parent = posixpath.dirname(filename)
                if node.level:
                    for _ in range(node.level - 1):
                        parent = posixpath.dirname(parent)
                    target = posixpath.join(parent, (node.module or '').replace('.', '/') + '.py')
                else:
                    target = (node.module or '').replace('.', '/') + '.py'
                for name in node.names:
                    imports[filename][name.asname or name.name] = (target, name.name)
            elif isinstance(node, ast.ClassDef):
                register_class(filename, node)

    def bases(key):
        filename, _ = key
        result = []
        for base in classes[key].bases:
            name = ast.unparse(base)
            if name in ('object', 'ABC'):
                continue
            # Nested text backbones may refer to siblings in their class scope.
            scope = key[1].rsplit('.', 1)[0] if '.' in key[1] else ''
            scoped = (filename, f'{scope}.{name}')
            result.append(scoped if scope and scoped in classes else (
                (filename, name) if (filename, name) in classes
                else imports[filename].get(name, (filename, name))))
        return result

    cache = {}

    def lineage(key, active=()):
        if key in active:
            raise ValueError(f'Cyclic model inheritance: {key}')
        if key in cache:
            return cache[key]
        if key not in classes:
            return [key]
        parents = bases(key)
        sequences = [lineage(parent, (*active, key))[:] for parent in parents] + [parents[:]]
        order = [key]
        while any(sequences):
            sequences = [seq for seq in sequences if seq]
            candidate = next((seq[0] for seq in sequences
                              if not any(seq[0] in other[1:] for other in sequences)), None)
            if candidate is None:
                raise ValueError(f'Inconsistent model inheritance: {key}')
            order.append(candidate)
            for seq in sequences:
                if seq[0] == candidate:
                    seq.pop(0)
        cache[key] = order
        return order

    records = []
    for key, node in classes.items():
        if selected is not None and node.name not in selected and key[1].split('.')[0] not in selected:
            continue
        order = lineage(key)
        fields, methods, dynamic = {}, {}, []
        for owner in reversed(order):
            if owner not in classes:
                continue
            filename, name = owner
            for item in classes[owner].body:
                origin = {'file': filename, 'class': name, 'line': item.lineno}
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods[item.name] = {**origin, 'end_line': item.end_lineno}
                elif isinstance(item, (ast.Assign, ast.AnnAssign)):
                    targets = item.targets if isinstance(item, ast.Assign) else [item.target]
                    for target in targets:
                        if isinstance(target, ast.Name) and (target.id.startswith('keys_') or target.id in {
                                'arch', 'model_arch', 'shape_fix', 'preserve_nd_shapes'}):
                            if item.value is not None:
                                fields[target.id] = {**origin, 'expression': ast.unparse(item.value)}
                elif not isinstance(item, (ast.Pass, ast.ClassDef)) and not (
                        isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant)
                        and isinstance(item.value.value, str)):
                    dynamic.append(origin)
        records.append({'file': key[0], 'class': key[1], 'line': node.lineno,
                        'nested_classes': [f'{key[1]}.{item.name}' for item in node.body
                                           if isinstance(item, ast.ClassDef)],
                        'registrations': [ast.unparse(dec) for dec in node.decorator_list
                                          if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute)
                                          and dec.func.attr == 'register'],
                        'mro': [f'{file}:{name}' for file, name in order],
                        'unresolved_bases': [f'{file}:{name}' for file, name in order if (file, name) not in classes],
                        'dynamic_class_statements': dynamic,
                        'effective_fields': fields, 'effective_methods': methods})
    return records


def source_section(code, start, end):
    """Locate a pinned policy span, retaining source lines and a content checksum."""
    first = code.index(start)
    last = code.index(end, first + len(start))
    return {'line': code.count('\n', 0, first) + 1,
            'end_line': code.count('\n', 0, last) + 1,
            'sha256': hashlib.sha256(code[first:last].encode()).hexdigest()}


def build_rules_report(folder=REFERENCES):
    """Inventory effective diffusion/text declarations and exact C++ policy spans."""
    manifest = verify_references(folder)
    report = {'status': 'Resolved static declarations; method bodies and C++ policies require separate analysis',
              'sources': []}
    # These are the application's offered GGUF text families; CLIP is safetensors-only.
    text_classes = {'T5Model', 'T5EncoderModel', 'Qwen2VLModel', 'Qwen3Model',
                    'LlamaModel', 'MistralModel', 'Mistral3Model'}
    for entry in manifest['sources']:
        if entry['id'] == 'kompute-b3962':
            continue
        record = {'id': entry['id'], 'commit': entry['commit']}
        with tarfile.open(folder / entry['archive']) as source:
            prefix = entry['archive_root'] + '/'
            modules = {}
            for member in source.getmembers():
                name = member.name[len(prefix):]
                if member.isfile() and (name in ('tools/convert.py', 'convert_hf_to_gguf.py')
                                        or name.startswith('conversion/') and name.endswith('.py')):
                    modules[name] = source.extractfile(member).read().decode('utf-8')
            diffusion = entry['id'] in ('city96-patch', 'molbal-current')
            if diffusion:
                record['models'] = [item for item in effective_rules(modules)
                                    if any(base.endswith(':ModelTemplate') for base in item['mro'])]
            else:
                record['models'] = effective_rules(modules, text_classes)
            if not diffusion:
                record['missing_requested_classes'] = sorted(text_classes - {item['class'] for item in record['models']})
            filename = 'tools/lcpp.patch' if diffusion else (
                'src/llama.cpp' if entry['id'] == 'llama-b3962' else 'src/llama-quant.cpp')
            code = source.extractfile(prefix + filename).read().decode('utf-8')
            record['cpp_policy_file'] = filename
            if diffusion:
                spans = {'image_type_selection': ('+static ggml_type img_tensor_get_type',
                                                  ' static ggml_type llama_tensor_get_type'),
                         'image_exclusions': ('+        // rules for image models',
                                              '         enum ggml_type new_type;')}
            else:
                spans = {'text_type_selection': ('static ggml_type llama_tensor_get_type', '\nstatic ')}
                if entry['id'] != 'llama-b3962':
                    spans.update({
                        'tensor_categories': ('static tensor_category tensor_get_category', '\nstatic '),
                        'tensor_eligibility': ('static bool tensor_allows_quantization', '\nstatic '),
                        'shape_fallback': ('static ggml_type tensor_type_fallback', '\nstatic ')})
            record['cpp_policy_sections'] = {name: source_section(code, *bounds) for name, bounds in spans.items()}
            if diffusion:
                tree = ast.parse(modules['tools/convert.py'])
                names = {'key_matches', '_k_matches_arch_pattern', 'plan_k_quantization',
                         '_k_quant_category', '_k_quant_effective_last_dim', 'handle_tensors'}
                record['python_policy_sections'] = {
                    node.name: {'file': 'tools/convert.py', 'line': node.lineno, 'end_line': node.end_lineno}
                    for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names}
        report['sources'].append(record)
    local = Path(__file__).resolve().parents[1]
    report['project_models'] = effective_rules({
        'models/architectures.py': (local / 'models/architectures.py').read_text(encoding='utf-8')})
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(build_rules_report(), indent=2) + '\n', encoding='utf-8')
