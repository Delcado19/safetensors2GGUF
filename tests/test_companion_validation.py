"""Native construction must not hide incomplete diffusion-model weight loading."""
import pytest

from scripts.validate_companion_model import _check_warnings


@pytest.mark.parametrize('warning', ['unet missing: [weight]', 'unet unexpected: [weight]', 'clip missing: [weight]'])
def test_native_weight_mismatch_fails_validation(warning):
    with pytest.raises(ValueError, match='missing or unexpected weights'):
        _check_warnings('ernie_image', [warning])


def test_optional_projection_exception_is_limited_to_native_clip_l():
    warning = "clip missing: ['text_projection.weight']"
    _check_warnings('clip-l', [warning])
    with pytest.raises(ValueError):
        _check_warnings('clip-bigg', [warning])
    _check_warnings('ernie_image', [])
