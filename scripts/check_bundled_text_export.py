"""Requalify the pinned Qwen3 exporter on an existing synthetic one-layer fixture."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

from safetensors import safe_open

from scripts.compare_gguf_outputs import compare_files
from scripts.gguf_research_io import open_gguf
from text_encoder_convert import convert_text_encoder_kquant


def main():
    """Inject only fixture config; run the real shipped exporter and quantizer."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.work.exists() or args.output.exists():
        parser.error('Choose fresh output and work paths')
    root = Path(__file__).resolve().parents[1]
    with safe_open(args.source, framework='pt') as source:
        assert source.get_slice('model.embed_tokens.weight').get_shape() == [151936, 256]
        assert len(source.keys()) == 14
    args.work.mkdir(parents=True)
    before = hashlib.sha256(args.source.read_bytes()).hexdigest()

    def fixture_config(repo, destination, on_log=None):
        config = json.loads((root / 'text_encoder_configs/qwen3-4b/config.json').read_text())
        config.update(hidden_size=256, intermediate_size=512, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=2, tie_word_embeddings=False,
                      max_position_embeddings=512)
        for name in ('tokenizer.json', 'tokenizer_config.json'):
            shutil.copyfile(root / 'text_encoder_configs/qwen3-4b' / name, destination / name)
        (destination / 'config.json').write_text(json.dumps(config))

    output = args.work / 'qwen3-Q4_K_M.gguf'
    with patch('text_encoder_convert.fetch_base_config_files', side_effect=fixture_config):
        convert_text_encoder_kquant(str(args.source), 'fixture/local-config', str(output),
                                   on_log=lambda line: print(line, flush=True))
    comparison = compare_files(output, args.reference)
    with open_gguf(output) as current, open_gguf(args.reference) as previous:
        # llama.cpp derives the display name from a random temporary directory.
        def metadata(reader):
            return {key: (field.types, field.contents()) for key, field in reader.fields.items()
                    if not key.startswith('GGUF.') and key != 'general.name'}
        metadata_equal = metadata(current) == metadata(previous)
    with open_gguf(output) as reader:
        types = {}
        for tensor in reader.tensors:
            types[tensor.tensor_type.name] = types.get(tensor.tensor_type.name, 0) + 1
    report = {
        'scope': 'Synthetic one-layer Qwen3, local config injection, real pinned export/quantizer; no inference.',
        'architecture': 'qwen3', 'tensor_count': sum(types.values()), 'tensor_types': types,
        'source_unchanged': before == hashlib.sha256(args.source.read_bytes()).hexdigest(),
        'comparison': comparison,
        'metadata_equal_except_temporary_name': metadata_equal,
        'source_code_sha256': {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                               for name in ('text_encoder_convert.py', 'bundled_tools.py')},
    }
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    assert (report['source_unchanged'] and metadata_equal and comparison['alignment_equal']
            and comparison['tensor_counts_equal'] and all(row['equal'] for row in comparison['tensors'])), report


if __name__ == '__main__':
    main()
