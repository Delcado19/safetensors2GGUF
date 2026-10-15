"""Shared reader lifetime and integer-width guards for research GGUF tools."""
from contextlib import contextmanager

import gguf


@contextmanager
def open_gguf(path):
    """Open a reader, normalize offset arithmetic and close even on parse errors."""
    reader = gguf.GGUFReader.__new__(gguf.GGUFReader)
    try:
        reader.__init__(str(path))
        # Explicit GGUF alignment is NumPy uint32: modulo with large Python file
        # offsets can overflow C long on Windows. Keep file arithmetic in Python.
        reader.alignment = int(reader.alignment)
        yield reader
    finally:
        mapping = getattr(reader, 'data', None)
        if mapping is not None:
            mapping._mmap.close()
