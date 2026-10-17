"""Per-model generation defaults, so a prompt alone produces a properly cooked image.

SwarmUI's own defaults (512x512, 20 steps, CFG 7) suit none of the modern models, and SwarmUI does not apply a
model's preferred settings on its own. Presets are keyed by SwarmUI's ``compat_class``, which base and distilled
checkpoints of a family share; the values assume the distilled/turbo checkpoints, so clients running a base
checkpoint should pass ``steps``/``cfg_scale`` explicitly. Models without a preset fall back to SwarmUI's defaults.
"""

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any


@dataclass(frozen=True)
class Preset:
    """Sampling defaults for a model family."""

    steps: int
    cfg_scale: float


_PRESETS: dict[str, Preset] = {
    "flux-2-klein-9b": Preset(steps=10, cfg_scale=1.0),
    "z-image": Preset(steps=10, cfg_scale=1.0),
    "stable-diffusion-xl-v1": Preset(steps=30, cfg_scale=6.0),
}


@dataclass(frozen=True)
class ModelDefaults:
    """The defaults applied to a model when a request leaves them out."""

    width: int | None
    height: int | None
    steps: int | None
    cfg_scale: float | None


def resolve_defaults(description: dict[str, Any]) -> ModelDefaults:
    """Defaults for a model from its SwarmUI description (``DescribeModel`` / ``ListModels`` entry)."""
    preset = _PRESETS.get(description.get("compat_class") or "")
    return ModelDefaults(
        width=description.get("standard_width") or None,
        height=description.get("standard_height") or None,
        steps=preset.steps if preset else None,
        cfg_scale=preset.cfg_scale if preset else None,
    )


def model_id(name: str) -> str:
    """The id clients use for a model SwarmUI names by file (``name`` from SwarmUI, always with an extension).

    The id is the path without the extension, matching SwarmUI's own UI; SwarmUI accepts both forms.
    """
    path = PurePosixPath(name)
    return str(path.with_suffix("")) if path.suffix else name
