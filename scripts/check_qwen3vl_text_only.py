"""Check the explicit full-checkpoint text-only VL export without running inference."""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.gguf_research_io import open_gguf
from text_encoder_convert import convert_text_encoder_any


def main():
    """Require the text-only marker, absence of vision, and an unchanged source."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.destination.exists() or args.report.exists():
        parser.error('Choose fresh destination and report paths')
    before = args.source.stat()
    convert_text_encoder_any(str(args.source), '', str(args.destination), 'Q8_CR_TEXT_ONLY', on_log=print)
    with open_gguf(args.destination) as reader:
        assert reader.get_field('comfy.gguf.qwen3vl.text_only').contents()
        names = [tensor.name for tensor in reader.tensors]
        assert names and not any('visual.' in name or 'deepstack' in name for name in names)
        report = {
            'scope': 'Complete local 4B BF16 source to explicit text-only Q8_CR; conversion/readback, not inference.',
            'source': str(args.source), 'output': str(args.destination), 'text_only_marker': True,
            'vision_tensor_count': 0, 'tensor_count': len(names),
            'architecture': reader.get_field('general.architecture').contents(),
            'source_stat_unchanged': (before.st_size, before.st_mtime_ns) ==
                                     (args.source.stat().st_size, args.source.stat().st_mtime_ns),
            'source_code_sha256': {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                                   for name in ('text_encoder_convert.py', 'extended_backend.py')},
        }
    assert report['source_stat_unchanged']
    args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf8')
    print('PASS', args.report)


if __name__ == '__main__':
    main()
