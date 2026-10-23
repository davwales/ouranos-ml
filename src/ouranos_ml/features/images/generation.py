"""The SwarmUI generation pipeline shared by the image operations, from model lookup to our progress events."""

import asyncio
import base64
import json
import math
import time
from collections.abc import AsyncGenerator
from contextlib import aclosing
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

import structlog

from ouranos_ml.features.images.presets import ModelDefaults, model_id, resolve_defaults
from ouranos_ml.features.images.schemas import (
    MAX_DIMENSION,
    MIN_DIMENSION,
    CompletedEvent,
    DoneEvent,
    ErrorEvent,
    GeneratedImage,
    ImageGenerationEvent,
    ImageRequest,
    ImagesResponse,
    Lora,
    ProgressEvent,
    StatusEvent,
)
from ouranos_ml.shared.domain.core.settings import get_settings
from ouranos_ml.shared.infra.clients.swarmui_client import SwarmUIClient, get_swarmui_client
from ouranos_ml.shared.infra.swarmui.errors import SwarmUIError, SwarmUIModelNotFoundError, SwarmUITimeoutError

logger = structlog.get_logger(__name__)

_RESERVED_PARAMS = frozenset({"session_id", "images"})
_SIZE_MULTIPLE = 16
_background_tasks: set[asyncio.Task[None]] = set()


@dataclass(frozen=True)
class InitImage:
    """An image to start from (img2img) and an optional white-means-change mask (inpainting), both base64."""

    image: str
    width: int
    height: int
    mask: str | None = None
    creativity: float | None = None


@dataclass(frozen=True)
class _ResolvedLora:
    """A requested LoRA as SwarmUI knows it."""

    name: str
    weight: float


@dataclass
class _Generation:
    """A generation running in its own SwarmUI session (so interrupting it cannot touch anyone else's)."""

    client: SwarmUIClient
    session_id: str
    model: str
    payload: dict[str, Any]
    messages: AsyncGenerator[dict[str, Any]]
    started: float = field(default_factory=time.monotonic)


async def handle(request: ImageRequest, init: InitImage | None = None) -> ImagesResponse:
    """Generate the images and return them all at once."""
    generation = await _open(request, init)
    images: list[GeneratedImage] = []
    created = int(time.time())

    async with aclosing(_generate(generation, _Translator(generation))) as events:
        async for event in events:
            if isinstance(event, CompletedEvent):
                images.append(GeneratedImage(**event.model_dump(include=set(GeneratedImage.model_fields))))
            elif isinstance(event, DoneEvent):
                created = event.created

    return ImagesResponse(created=created, model=generation.model, data=images)


async def open_stream(request: ImageRequest, init: InitImage | None = None) -> AsyncGenerator[ImageGenerationEvent]:
    """Start the generation eagerly, then return an iterator over its progress events.

    Starting eagerly means an unconfigured or unreachable SwarmUI and unknown models or LoRAs raise here, before any
    SSE bytes are sent, so they surface as real HTTP status codes. Later failures become a final ``error`` event.
    """
    generation = await _open(request, init)
    return _stream(generation)


async def _stream(generation: _Generation) -> AsyncGenerator[ImageGenerationEvent]:
    """Relay the generation's events, ending with an ``error`` event instead of raising."""
    translator = _Translator(generation)
    try:
        async with aclosing(_generate(generation, translator)) as events:
            async for event in events:
                yield event
    except SwarmUIError as exc:
        yield ErrorEvent(sequence_number=translator.next_sequence(), code=exc.error_type, message=exc.message)


async def _open(request: ImageRequest, init: InitImage | None) -> _Generation:
    """Resolve the model, its defaults and the LoRAs, then start the generation in a fresh SwarmUI session."""
    model = request.model or get_settings().swarmui_default_model

    logger.info(
        "image generation requested",
        model=model,
        n=request.n,
        size=request.size or "auto",
        stream=request.stream,
        lora_count=len(request.loras),
        init_image=f"{init.width}x{init.height}" if init else None,
        mask=bool(init and init.mask),
    )

    try:
        client = get_swarmui_client()
        session_id = await client.new_session()
        description, *lora_descriptions = await asyncio.gather(
            _describe(client, session_id, model),
            *(_describe(client, session_id, lora.name, subtype="LoRA") for lora in request.loras),
        )
        loras = [
            _resolve_lora(lora, found, description)
            for lora, found in zip(request.loras, lora_descriptions, strict=True)
        ]
        payload = _build_payload(request, model, resolve_defaults(description), loras, init)
        messages = await client.open_generation(session_id, payload)
    except SwarmUIModelNotFoundError as exc:
        logger.warning("image model not found", model=model, error=exc.message)
        raise
    except Exception:
        logger.error("image generation upstream failed", model=model, exc_info=True)
        raise
    return _Generation(
        client=client,
        session_id=session_id,
        model=model_id(description["name"]),
        payload=payload,
        messages=messages,
    )


async def _describe(
    client: SwarmUIClient, session_id: str, name: str, subtype: str = "Stable-Diffusion"
) -> dict[str, Any]:
    """Look up a model or LoRA, which both validates the name and supplies its defaults."""
    try:
        response = await client.call("DescribeModel", session_id, modelName=name, subtype=subtype)
    except SwarmUIError as exc:
        if "not found" in exc.message.lower():
            kind = "LoRA" if subtype == "LoRA" else "Model"
            raise SwarmUIModelNotFoundError(f"{kind} '{name}' not found") from exc
        raise
    return response["model"]


def _resolve_lora(lora: Lora, description: dict[str, Any], model: dict[str, Any]) -> _ResolvedLora:
    """The LoRA's SwarmUI id and weight: the requested weight, else the LoRA's own default, else 1."""
    lora_class, model_class = description.get("compat_class"), model.get("compat_class")
    if lora_class and model_class and lora_class != model_class:
        logger.warning("lora does not match model", lora_compat_class=lora_class, model_compat_class=model_class)

    weight = lora.weight
    if weight is None:
        try:
            weight = float(description.get("lora_default_weight") or 1.0)
        except (TypeError, ValueError):
            weight = 1.0
    return _ResolvedLora(name=model_id(description["name"]), weight=weight)


def _fit_init_size(init: InitImage, defaults: ModelDefaults) -> tuple[int, int]:
    """The init image's aspect ratio at the model's standard pixel count (or the image's own, without one).

    SwarmUI resizes the init image to the requested size, so this keeps it undistorted at a resolution the model
    handles well. Sides are rounded to multiples of 16 and clamped to the request size limits.
    """
    area = defaults.width * defaults.height if defaults.width and defaults.height else init.width * init.height
    scale = math.sqrt(area / (init.width * init.height))

    def fit(side: int) -> int:
        return min(MAX_DIMENSION, max(MIN_DIMENSION, round(side * scale / _SIZE_MULTIPLE) * _SIZE_MULTIPLE))

    return fit(init.width), fit(init.height)


def _build_payload(
    request: ImageRequest,
    model: str,
    defaults: ModelDefaults,
    loras: list[_ResolvedLora],
    init: InitImage | None,
) -> dict[str, Any]:
    """SwarmUI's flat raw input. Precedence: explicit fields, then raw ``params``, then model defaults.

    With an init image, the default size follows the image's aspect ratio rather than the model's standard size.
    """
    payload = {key: value for key, value in (request.params or {}).items() if key not in _RESERVED_PARAMS}

    default_width, default_height = _fit_init_size(init, defaults) if init else (defaults.width, defaults.height)
    model_defaults = {
        "width": default_width,
        "height": default_height,
        "steps": defaults.steps,
        "cfgscale": defaults.cfg_scale,
    }

    for key, value in model_defaults.items():
        if value is not None:
            payload.setdefault(key, value)

    width, height = request.dimensions() or (None, None)

    explicit = {
        "prompt": request.prompt,
        "model": model,
        "images": request.n,
        "width": width,
        "height": height,
        "negativeprompt": request.negative_prompt,
        "steps": request.steps,
        "cfgscale": request.cfg_scale,
        "seed": request.seed,
        "sampler": request.sampler,
        "scheduler": request.scheduler,
        "loras": [lora.name for lora in loras] or None,
        "loraweights": [lora.weight for lora in loras] or None,
        "initimage": init.image if init else None,
        "initimagecreativity": init.creativity if init else None,
        "maskimage": init.mask if init else None,
        "donotsave": not request.save,
        "nopreviews": not request.previews,
    }

    payload.update({key: value for key, value in explicit.items() if value is not None})
    return payload


async def _generate(generation: _Generation, translator: "_Translator") -> AsyncGenerator[ImageGenerationEvent]:
    """Translate SwarmUI messages into events until the generation finishes, fails, or times out.

    If the caller stops early (client disconnect) or anything fails, the session's generation is interrupted:
    SwarmUI keeps generating after its socket closes, so leaving it would waste the GPU on unwanted images.
    """
    timeout = get_settings().swarmui_generation_timeout_seconds
    deadline = asyncio.get_running_loop().time() + timeout
    completed = False

    try:
        async with aclosing(generation.messages) as messages:
            while True:
                try:
                    async with asyncio.timeout_at(deadline):
                        message = await anext(messages)
                except StopAsyncIteration:
                    break
                except TimeoutError:
                    raise SwarmUITimeoutError(f"Image generation did not finish within {timeout:g}s")
                if "error" in message:
                    raise SwarmUIError(str(message["error"]))
                for event in await translator(message):
                    yield event

        done = translator.done()
        completed = True

        logger.info(
            "image generation complete",
            model=generation.model,
            count=done.count,
            duration_ms=round((time.monotonic() - generation.started) * 1000),
        )

        yield done
    except SwarmUIError:
        logger.error("image generation failed", model=generation.model, exc_info=True)
        raise
    finally:
        if not completed:
            _interrupt_in_background(generation)


def _interrupt_in_background(generation: _Generation) -> None:
    """Interrupt the generation without awaiting, since the caller may itself be mid-cancellation."""

    async def _interrupt() -> None:
        try:
            await generation.client.interrupt(generation.session_id)
            logger.info("image generation interrupted", model=generation.model)
        except Exception:
            logger.warning("image generation interrupt failed", model=generation.model, exc_info=True)

    task = asyncio.create_task(_interrupt())
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


class _Translator:
    """Turns SwarmUI WebSocket messages into our events, tracking sequence numbers, stage, and progress."""

    def __init__(self, generation: _Generation) -> None:
        self._generation = generation
        self._sequence_number = 0
        self._stage: str | None = None
        self._percents: dict[int, float] = {}
        self._count = 0

        try:
            batch_size = max(1, int(generation.payload.get("batchsize", 1)))
        except (TypeError, ValueError):
            batch_size = 1

        self._expected = generation.payload["images"] * batch_size

    def next_sequence(self) -> int:
        """Claim the next sequence number."""
        self._sequence_number += 1
        return self._sequence_number - 1

    async def __call__(self, message: dict[str, Any]) -> list[ImageGenerationEvent]:
        """The events for one SwarmUI message (often none: keep-alives, repeated statuses, intermediates)."""
        if isinstance(status := message.get("status"), dict):
            return self._status(status)
        if isinstance(progress := message.get("gen_progress"), dict):
            return self._progress(progress)
        if isinstance(message.get("image"), str):
            return await self._completed(message)
        if message.get("discard_indices"):
            logger.debug("swarmui discarded images", count=len(message["discard_indices"]))
        return []

    def done(self) -> DoneEvent:
        """The terminal success event."""
        return DoneEvent(
            sequence_number=self.next_sequence(),
            created=int(time.time()),
            model=self._generation.model,
            count=self._count,
        )

    def _status(self, status: dict[str, Any]) -> list[ImageGenerationEvent]:
        """Emit a status event when the stage changes; SwarmUI repeats statuses many times per stage."""
        if status.get("loading_models"):
            stage = "loading_model"
        elif status.get("live_gens"):
            stage = "generating"
        elif status.get("waiting_gens"):
            stage = "queued"
        else:
            return []

        if stage == self._stage:
            return []

        self._stage = stage
        return [
            StatusEvent(
                sequence_number=self.next_sequence(),
                stage=stage,
                waiting_gens=status.get("waiting_gens", 0),
                loading_models=status.get("loading_models", 0),
                live_gens=status.get("live_gens", 0),
            )
        ]

    def _progress(self, progress: dict[str, Any]) -> list[ImageGenerationEvent]:
        """Emit progress for one image, never letting its percent go backwards.

        SwarmUI's ``overall_percent`` is not monotonic: preview frames report a higher figure than the frames
        around them.
        """
        index = int(progress.get("batch_index", 0))
        if index < 0:
            return []

        percent = max(self._percents.get(index, 0.0), float(progress.get("overall_percent", 0.0)))
        self._percents[index] = percent
        preview = progress.get("preview")

        return [
            ProgressEvent(
                sequence_number=self.next_sequence(),
                image_index=index,
                percent=percent,
                step_percent=float(progress.get("current_percent", 0.0)),
                total_percent=self._total_percent(),
                preview_b64=preview.split(",", 1)[-1] if isinstance(preview, str) else None,
            )
        ]

    async def _completed(self, message: dict[str, Any]) -> list[ImageGenerationEvent]:
        """Emit a finished image inline as base64, fetching it from SwarmUI when it was saved to a file."""
        index = int(message.get("batch_index", 0))
        if index < 0:
            return []

        b64_json, output_format = await self._inline(message["image"])
        params = _image_params(message.get("metadata"))
        width = params.get("width", self._generation.payload.get("width"))
        height = params.get("height", self._generation.payload.get("height"))
        self._percents[index] = 1.0
        self._count += 1

        return [
            CompletedEvent(
                sequence_number=self.next_sequence(),
                image_index=index,
                b64_json=b64_json,
                seed=params.get("seed"),
                size=f"{width}x{height}" if width and height else None,
                output_format=output_format,
            )
        ]

    async def _inline(self, image: str) -> tuple[str, str]:
        """``(base64 data, format)`` for a SwarmUI image, which is either a data URL or a ``View/`` path."""
        if image.startswith("data:"):
            header, data = image.split(",", 1)
            mime_type = header.removeprefix("data:").split(";", 1)[0]
            return data, mime_type.rsplit("/", 1)[-1]

        content = await self._generation.client.fetch_file(image)
        return base64.b64encode(content).decode("ascii"), PurePosixPath(image).suffix.lstrip(".").lower() or "png"

    def _total_percent(self) -> float:
        """Progress across every image, from 0 to 1."""
        return min(1.0, sum(self._percents.values()) / self._expected)


def _image_params(metadata: object) -> dict[str, Any]:
    """The parameters SwarmUI records in an image's metadata (best-effort: the format is not guaranteed)."""
    if not isinstance(metadata, str):
        return {}
    try:
        params = json.loads(metadata).get("sui_image_params", {})
    except (ValueError, AttributeError):
        return {}
    return params if isinstance(params, dict) else {}
