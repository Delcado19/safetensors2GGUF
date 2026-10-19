"""Check installed Qwen runtime resources and native Tekken/SentencePiece blobs.

Structural checks cannot establish tokenizer provenance. Embedded repairs require
the exact checkpoint's tokenizer; runtime repairs require matching ComfyUI assets.
"""
import base64
import hashlib
import json
from pathlib import Path
import struct
import shutil
import tempfile
import uuid

from safetensors import safe_open
from conversion_io import atomic_output, validate_output
from convert import ConversionCancelled, _read_safetensors_header, load_state_dict
from convert_safetensors import _write_header
from text_encoder_convert import detect_text_encoder_family

FAMILIES = {'ernie-image-pe', 'mistral-small-3.2-24b'}
MAX_TOKENIZER_BYTES = 64 * 1024 * 1024


def _validate(data, vocab_size):
    """Validate the complete active Tekken vocabulary before native loading."""
    obj = json.loads(data)
    config = obj['config']
    offset = config['default_num_special_tokens']
    if config['default_vocab_size'] != vocab_size or not 0 < offset < vocab_size:
        raise ValueError('Tokenizer vocabulary does not match the checkpoint')
    special = obj['special_tokens']
    if sorted(row['rank'] for row in special) != list(range(offset)):
        raise ValueError('Invalid special-token IDs')
    tokens = [row.get('token_str', '') for row in special]
    if any(not token for token in tokens) or len(set(tokens)) != len(tokens):
        raise ValueError('Invalid special tokens')
    active = [row for row in obj['vocab'] if 0 <= row['rank'] < vocab_size - offset]
    if sorted(row['rank'] for row in active) != list(range(vocab_size - offset)):
        raise ValueError('Incomplete tokenizer vocabulary')
    raw = [base64.b64decode(row['token_bytes'], validate=True) for row in active]
    if any(not token for token in raw) or len(set(raw)) != len(raw):
        raise ValueError('Invalid tokenizer vocabulary bytes')
    if not {bytes([value]) for value in range(256)} <= set(raw):
        raise ValueError('Tokenizer lacks byte fallback tokens')
    return obj


def _context(source):
    state = load_state_dict(str(source), strip_prefixes=False)
    family = detect_text_encoder_family(state)
    if family not in FAMILIES:
        return None
    key = next(key for key in state if key.endswith('embed_tokens.weight'))
    return state.shape_of(key)[0]


def _blob_context(source):
    """Select only embedded formats actually consumed by the native encoder."""
    vocab = _context(source)
    if vocab is not None:
        return 'tekken_model', vocab
    state = load_state_dict(str(source), strip_prefixes=False)
    if detect_text_encoder_family(state) == 'umt5-xxl':
        key = next(key for key in state if key.endswith(('shared.weight', 'embed_tokens.weight')))
        return 'spiece_model', state.shape_of(key)[0]
    return None


def _validate_blob(data, key, vocab):
    """Allow padded embedding rows, but never token IDs outside the model vocabulary."""
    if key == 'tekken_model':
        _validate(data, vocab)
    else:
        import sentencepiece
        tokenizer = sentencepiece.SentencePieceProcessor(model_proto=data)
        if not 0 < tokenizer.vocab_size() <= vocab:
            raise ValueError('SentencePiece vocabulary exceeds checkpoint embeddings')


def _runtime_context(source):
    """Resolve Qwen's shared native assets from the selected installed model path."""
    state = load_state_dict(str(source), strip_prefixes=False)
    family = detect_text_encoder_family(state)
    if family not in {'qwen3-4b', 'qwen3-8b', 'qwen3-vl', 'qwen2.5-vl-7b'}:
        return None
    key = next(key for key in state if key.endswith('embed_tokens.weight'))
    for parent in Path(source).resolve().parents:
        if parent.name == 'models' and (parent.parent / 'comfy/text_encoders').is_dir():
            return parent.parent / 'comfy/text_encoders/qwen25_tokenizer', state.shape_of(key)[0]
    return None


def _validate_runtime(folder, vocab):
    """Use the same tokenizer class and fixed chat token IDs as native ComfyUI."""
    from transformers import Qwen2Tokenizer
    for name in ('vocab.json', 'merges.txt', 'tokenizer_config.json'):
        path = Path(folder) / name
        if not path.is_file() or not 0 < path.stat().st_size <= MAX_TOKENIZER_BYTES:
            raise ValueError('Missing or oversized native tokenizer resource: ' + name)
    tokenizer = Qwen2Tokenizer.from_pretrained(str(folder), local_files_only=True)
    values = list(tokenizer.get_vocab().values())
    if len(values) < 151646 or min(values) < 0 or max(values) >= vocab or len(set(values)) != len(values):
        raise ValueError('Runtime tokenizer vocabulary does not fit checkpoint embeddings')
    for token, expected in (('<|endoftext|>', 151643), ('<|im_start|>', 151644), ('<|im_end|>', 151645)):
        if tokenizer.convert_tokens_to_ids(token) != expected:
            raise ValueError('Runtime tokenizer chat-token IDs do not match native Qwen')
    if not tokenizer.encode('Tokenizer check: English, Deutsch, 中文.', add_special_tokens=False):
        raise ValueError('Runtime tokenizer cannot encode text')


def repair_runtime_tokenizer(source, reference, *, on_log=None, cancel_event=None):
    """Restore explicit healthy runtime assets, retaining a rollback directory."""
    context = _runtime_context(source)
    if context is None:
        raise ValueError('Cannot locate native Qwen tokenizer resources for this model')
    target, vocab = context
    reference = Path(reference).resolve()
    if reference == target.resolve():
        raise ValueError('Choose a separate healthy tokenizer resource directory')
    _validate_runtime(reference, vocab)
    target.parent.mkdir(parents=True, exist_ok=True)
    backup = target.with_name(target.name + '.backup-' + uuid.uuid4().hex)
    with tempfile.TemporaryDirectory(prefix='tokenizer-repair-', dir=target.parent) as scratch:
        staged = Path(scratch) / 'resources'
        staged.mkdir()
        for name in ('vocab.json', 'merges.txt', 'tokenizer_config.json'):
            shutil.copyfile(reference / name, staged / name)
        _validate_runtime(staged, vocab)
        if cancel_event is not None and cancel_event.is_set():
            raise ConversionCancelled('cancelled')
        # Keep the complete old directory; restore it if publishing the validated copy fails.
        existed = target.exists()
        if existed:
            target.rename(backup)
        try:
            staged.rename(target)
        except OSError:
            if existed:
                backup.rename(target)
            raise
    if on_log:
        on_log(f'INFO: Native Qwen tokenizer restored; previous resources retained at {backup if existed else "no previous directory"}. Restart ComfyUI.')
    return str(source)


def check_tokenizer(source):
    """Return scoped structural status without loading model weights or networking."""
    if Path(source).suffix.lower() != '.safetensors':
        return {'status': 'not_checked', 'repairable': False}
    context = _blob_context(source)
    if context is None:
        runtime = _runtime_context(source)
        if runtime is not None:
            folder, vocab = runtime
            result = {'repairable': True, 'repair_kind': 'runtime', 'resource_path': str(folder)}
            try:
                _validate_runtime(folder, vocab)
            except (ValueError, OSError, TypeError, KeyError, IndexError):
                return result | {'status': 'invalid' if folder.exists() else 'missing',
                                 'message': 'Native Qwen resources need repair. Select a healthy qwen25_tokenizer directory from the same ComfyUI version. Shared by Qwen encoders; a backup is retained.'}
            return {'status': 'valid', 'repairable': False, 'resource_path': str(folder),
                    'message': 'Native Qwen runtime tokenizer loads locally and token IDs fit the model. Tokenizer provenance is not verified.'}
        return {'status': 'not_checked', 'repairable': False}
    blob_key, vocab_size = context
    header = _read_safetensors_header(str(source))
    result = {'repairable': True,
              'message': 'Select tokenizer.json or tekken.json from this exact model repository.' if blob_key == 'tekken_model' else 'Select spiece.model from this exact model repository.'}
    if blob_key not in header:
        return result | {'status': 'missing', 'message': f'Native ComfyUI loading requires embedded {blob_key}. ' + result['message']}
    try:
        item = header[blob_key]
        length = item['data_offsets'][1] - item['data_offsets'][0]
        if item['dtype'] != 'U8' or item['shape'] != [length] or not 0 < length <= MAX_TOKENIZER_BYTES:
            raise ValueError('Invalid tokenizer tensor')
        with safe_open(str(source), framework='pt') as checkpoint:
            data = checkpoint.get_tensor(blob_key).numpy().tobytes()
        _validate_blob(data, blob_key, vocab_size)
    except (ValueError, TypeError, KeyError, IndexError, OverflowError, RuntimeError):
        return result | {'status': 'invalid', 'message': 'Embedded tokenizer data is invalid. ' + result['message']}
    return {'status': 'valid', 'repairable': False,
            'message': 'Embedded tokenizer structure and vocabulary size are valid; provenance is not verified.'}


def _tokenizer_bytes(path, vocab_size):
    if Path(path).stat().st_size > MAX_TOKENIZER_BYTES:
        raise ValueError('Tokenizer JSON exceeds 64 MiB')
    obj = json.loads(Path(path).read_bytes())
    if 'model' in obj:
        if obj['model'].get('type') != 'BPE':
            raise ValueError('Expected a Mistral BPE tokenizer')
        special = {row['content']: row['id'] for row in obj['added_tokens']}
        vocab = obj['model']['vocab']
        if sorted(vocab.values()) != list(range(vocab_size)):
            raise ValueError('Tokenizer vocabulary does not match the checkpoint')
        if sorted(special.values()) != list(range(len(special))) or any(vocab.get(token) != rank for token, rank in special.items()):
            raise ValueError('Tokenizer special tokens do not match its vocabulary')
        # Hugging Face ByteLevel vocab strings encode bytes through this reversible map.
        values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        codes = values.copy()
        extra = 0
        for value in range(256):
            if value not in values:
                values.append(value)
                codes.append(256 + extra)
                extra += 1
        decoder = {chr(code): value for value, code in zip(values, codes)}
        offset = len(special)
        obj = {'config': {'default_num_special_tokens': offset, 'default_vocab_size': vocab_size},
               'special_tokens': [{'token_str': token, 'rank': rank} for token, rank in special.items()],
               'vocab': [{'rank': rank - offset, 'token_bytes': base64.b64encode(bytes(decoder[c] for c in token)).decode()}
                         for token, rank in vocab.items() if rank >= offset]}
    data = json.dumps(obj, ensure_ascii=True, separators=(',', ':')).encode()
    _validate(data, vocab_size)
    return data


def repair_tokenizer(source, destination, tokenizer, *, on_log=None, cancel_event=None):
    """Write a new atomic checkpoint, preserving and verifying all weight payloads."""
    validate_output(source, destination, False)
    context = _blob_context(source)
    if context is None:
        raise ValueError('This encoder does not use a supported embedded tokenizer')
    blob_key, vocab_size = context
    if blob_key == 'tekken_model':
        data = _tokenizer_bytes(tokenizer, vocab_size)
    else:
        if Path(tokenizer).stat().st_size > MAX_TOKENIZER_BYTES:
            raise ValueError('Tokenizer exceeds 64 MiB')
        data = Path(tokenizer).read_bytes()
        _validate_blob(data, blob_key, vocab_size)
    before = Path(source).stat()
    def cancelled():
        if cancel_event is not None and cancel_event.is_set():
            raise ConversionCancelled('cancelled')
    with safe_open(str(source), framework='pt'):
        pass  # Validate source storage before trusting header offsets.
    with open(source, 'rb') as original:
        old_size = struct.unpack('<Q', original.read(8))[0]
        old = json.loads(original.read(old_size))
        header = {'__metadata__': old['__metadata__']} if '__metadata__' in old else {}
        entries = sorted(((key, value) for key, value in old.items() if key not in {'__metadata__', blob_key}),
                         key=lambda row: row[1]['data_offsets'][0])
        offset = 0
        for key, value in entries:
            length = value['data_offsets'][1] - value['data_offsets'][0]
            header[key] = value | {'data_offsets': [offset, offset + length]}
            offset += length
        header[blob_key] = {'dtype': 'U8', 'shape': [len(data)], 'data_offsets': [offset, offset + len(data)]}
        with atomic_output(destination) as temporary:
            digest = hashlib.sha256()
            with open(temporary, 'wb') as output:
                _write_header(output, header)
                for _, value in entries:
                    original.seek(8 + old_size + value['data_offsets'][0])
                    remaining = value['data_offsets'][1] - value['data_offsets'][0]
                    while remaining:
                        cancelled()
                        chunk = original.read(min(8 * 1024 * 1024, remaining))
                        if not chunk:
                            raise ValueError('Truncated source tensor')
                        digest.update(chunk)
                        output.write(chunk)
                        remaining -= len(chunk)
                output.write(data)
            copied = hashlib.sha256()
            with open(temporary, 'rb') as output:
                size = struct.unpack('<Q', output.read(8))[0]
                output.seek(8 + size)
                remaining = offset
                while remaining:
                    cancelled()
                    chunk = output.read(min(8 * 1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError('Truncated repair output')
                    copied.update(chunk)
                    remaining -= len(chunk)
            if copied.digest() != digest.digest() or check_tokenizer(temporary)['status'] != 'valid':
                raise ValueError('Tokenizer repair readback failed')
            if (Path(source).stat().st_size, Path(source).stat().st_mtime_ns) != (before.st_size, before.st_mtime_ns):
                raise ValueError('Source changed during tokenizer repair')
            cancelled()
    if on_log:
        on_log('INFO: Tokenizer repaired in a separate checkpoint; all weight bytes verified unchanged.')
    return str(destination)
