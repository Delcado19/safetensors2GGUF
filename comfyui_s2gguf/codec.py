"""Decode original-key text GGUFs for ComfyUI without dropping vision components."""
import json

import gguf
import numpy as np
import torch

LAYOUT_KEY = 'comfy.s2g.original_text_layout'
TENSORS_KEY = 'comfy.s2g.original_tensors'


def load_original_text(path):
    """Reconstruct a full CPU state dictionary; peak RAM includes dequantized weights.

    This compatibility loader deliberately uses ComfyUI's native model construction.
    It does not provide low-bit inference kernels or imply reduced inference RAM.
    """
    reader = gguf.GGUFReader.__new__(gguf.GGUFReader)
    try:
        reader.__init__(str(path))
        marker, schema = reader.get_field(LAYOUT_KEY), reader.get_field(TENSORS_KEY)
        if marker is None or marker.contents() != 1 or schema is None:
            raise ValueError('This loader requires an original-key safetensors2GGUF text export')
        descriptors = json.loads(schema.contents())
        if set(descriptors) != {tensor.name for tensor in reader.tensors}:
            raise ValueError('Original text tensor manifest mismatch')
        state = {}
        for tensor in reader.tensors:
            row = descriptors[tensor.name]
            shape = tuple(row['shape'])
            if row.get('bytes_dtype'):
                dtype = np.dtype(row['bytes_dtype'])
                if dtype.kind not in 'biu':
                    raise ValueError('Unsupported original byte payload dtype')
                values = np.asarray(tensor.data).view(np.uint8).reshape(-1).view(dtype).copy()
            else:
                values = gguf.dequantize(tensor.data, tensor.tensor_type).copy()
            value = torch.from_numpy(values.reshape(shape))
            if row.get('dtype') == 'torch.bfloat16':
                value = value.to(torch.bfloat16)
            elif row.get('dtype') == 'torch.float16':
                value = value.to(torch.float16)
            state[tensor.name] = value
        return state
    finally:
        mapping = getattr(reader, 'data', None)
        if mapping is not None:
            mapping._mmap.close()
