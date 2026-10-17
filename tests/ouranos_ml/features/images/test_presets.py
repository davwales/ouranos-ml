import pytest

from ouranos_ml.features.images.presets import model_id, resolve_defaults
from tests.ouranos_ml.shared.factories.images_factories import make_model_description


@pytest.mark.parametrize(
    ("compat_class", "steps", "cfg_scale"),
    [
        ("flux-2-klein-9b", 10, 1.0),
        ("z-image", 10, 1.0),
        ("stable-diffusion-xl-v1", 30, 6.0),
        ("unknown", None, None),
    ],
)
def test_resolve_defaults_when_compat_class_known_should_apply_its_preset(compat_class, steps, cfg_scale):
    # Arrange
    description = make_model_description(compat_class=compat_class, width=896, height=1152)

    # Act
    defaults = resolve_defaults(description)

    # Assert
    assert (defaults.steps, defaults.cfg_scale) == (steps, cfg_scale)
    assert (defaults.width, defaults.height) == (896, 1152)


def test_resolve_defaults_when_description_sparse_should_return_no_defaults():
    # Act
    defaults = resolve_defaults({"compat_class": None})

    # Assert
    assert defaults.width is None
    assert defaults.height is None
    assert defaults.steps is None


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("flux-2-klein-9b.safetensors", "flux-2-klein-9b"),
        ("OfficialStableDiffusion/sd_xl_base_1.0.safetensors", "OfficialStableDiffusion/sd_xl_base_1.0"),
        ("no-extension", "no-extension"),
    ],
)
def test_model_id_when_swarmui_name_should_drop_extension(name, expected):
    # Act
    result = model_id(name)

    # Assert
    assert result == expected
