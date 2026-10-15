"""Reader boundaries must keep multi-gigabyte file offsets out of NumPy uint32."""
from types import SimpleNamespace

import gguf
import numpy as np
import pytest

from scripts.gguf_research_io import open_gguf


@pytest.fixture
def fake_reader(monkeypatch):
    """Expose the same NumPy alignment scalar as GGUFReader without a huge file."""
    closed = []

    class Reader:
        def __init__(self, path):
            self.data = SimpleNamespace(_mmap=SimpleNamespace(close=lambda: closed.append(path)))
            self.alignment = np.uint32(32)
            if path == 'invalid':
                raise ValueError('invalid header')

    monkeypatch.setattr(gguf, 'GGUFReader', Reader)
    return closed


@pytest.mark.parametrize('offset', [2**31, 2**31 + 1, 2**32, 2**32 + 17, 2**40])
def test_offsets_over_windows_long_and_uint32_limits_remain_exact(fake_reader, offset):
    with open_gguf('large-file') as reader:
        assert type(reader.alignment) is int
        assert offset % reader.alignment == offset & 31
    assert fake_reader == ['large-file']


def test_parse_and_consumer_errors_close_mapping(fake_reader):
    with pytest.raises(ValueError, match='invalid header'):
        with open_gguf('invalid'):
            pytest.fail('invalid reader must not be yielded')
    with pytest.raises(RuntimeError, match='consumer failed'):
        with open_gguf('valid'):
            raise RuntimeError('consumer failed')
    assert fake_reader == ['invalid', 'valid']
