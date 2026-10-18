"""Distinct ComfyUI loader for original-key GGUF text encoders, including full VL."""
from .codec import load_original_text


class S2GGUFTextEncoderLoader:
    """Use native ComfyUI text construction after complete GGUF dequantization."""
    @classmethod
    def INPUT_TYPES(cls):
        """Expose encoder files, native CLIP types and optional CPU placement."""
        import folder_paths
        import comfy.sd
        choices = folder_paths.get_filename_list('text_encoders')
        return {'required': {'encoder': (choices,),
                             'type': ([item.name.lower() for item in comfy.sd.CLIPType],)},
                'optional': {'encoder_2': (['none', *choices],),
                             'device': (['default', 'cpu'],)}}

    RETURN_TYPES = ('CLIP',)
    FUNCTION = 'load'
    CATEGORY = 'safetensors2GGUF/loaders'
    DESCRIPTION = ('Loads original-key GGUF exports including complete vision towers. '
                   'Dequantizes on CPU before native ComfyUI loading; requires full weight RAM.')

    def load(self, encoder, type, encoder_2='none', device='default'):
        """Decode complete CPU state dictionaries and return a native CLIP object."""
        import folder_paths
        import comfy.sd
        import torch
        clip_type = comfy.sd.CLIPType[type.upper()]
        names = [encoder] + ([encoder_2] if encoder_2 != 'none' else [])
        states = [load_original_text(folder_paths.get_full_path_or_raise('text_encoders', name)) for name in names]
        options = {} if device != 'cpu' else {'load_device': torch.device('cpu'), 'offload_device': torch.device('cpu')}
        return (comfy.sd.load_text_encoder_state_dicts(
            state_dicts=states, embedding_directory=folder_paths.get_folder_paths('embeddings'),
            clip_type=clip_type, model_options=options),)


class S2GGUFDiffusionLoader:
    """Restore original-key diffusion GGUFs through native ComfyUI model construction."""
    @classmethod
    def INPUT_TYPES(cls):
        """Expose files from ComfyUI's diffusion model directories."""
        import folder_paths
        return {'required': {'model': (folder_paths.get_filename_list('diffusion_models'),)}}

    RETURN_TYPES = ('MODEL',)
    FUNCTION = 'load'
    CATEGORY = 'safetensors2GGUF/loaders'

    def load(self, model):
        """Decode original-key weights and return a supported native diffusion model."""
        import folder_paths
        import comfy.sd
        state = load_original_text(folder_paths.get_full_path_or_raise('diffusion_models', model))
        result = comfy.sd.load_diffusion_model_state_dict(state)
        if result is None:
            raise ValueError('Installed ComfyUI does not support this diffusion architecture')
        return (result,)


NODE_CLASS_MAPPINGS = {'S2GGUFTextEncoderLoader': S2GGUFTextEncoderLoader,
                       'S2GGUFDiffusionLoader': S2GGUFDiffusionLoader}
NODE_DISPLAY_NAME_MAPPINGS = {'S2GGUFTextEncoderLoader': 'Original Text / Vision Encoder (S2GGUF)',
                             'S2GGUFDiffusionLoader': 'Original Diffusion Model (S2GGUF)'}
