import asyncio
import base64
from contextlib import aclosing
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry import trace
from opentelemetry.trace import StatusCode

from ouranos_ml.features.images import generation
from ouranos_ml.features.images.generation import InitImage, handle, open_stream
from ouranos_ml.features.images.schemas import (
    CompletedEvent,
    DoneEvent,
    ErrorEvent,
    ProgressEvent,
    StatusEvent,
)
from ouranos_ml.shared.infra.swarmui.errors import SwarmUIError, SwarmUIModelNotFoundError, SwarmUITimeoutError
from tests.ouranos_ml.shared.factories.images_factories import (
    PNG_B64,
    make_generate_images_request,
    make_lora_description,
    make_model_description,
    make_swarm_error,
    make_swarm_image,
    make_swarm_messages,
    make_swarm_progress,
    make_swarm_status,
    make_swarmui_client,
)
from tests.ouranos_ml.shared.otel_mocks import patch_operation_telemetry

GET_CLIENT = "ouranos_ml.features.images.generation.get_swarmui_client"
GET_SETTINGS = "ouranos_ml.features.images.generation.get_settings"


def _settings(*, timeout: float = 600.0) -> MagicMock:
    settings = MagicMock()
    settings.swarmui_default_model = "flux-2-klein-9b"
    settings.swarmui_generation_timeout_seconds = timeout
    return settings


def _sent_payload(client: MagicMock) -> dict:
    return client.open_generation.await_args.args[1]


async def _collect(request, client: MagicMock) -> list:
    with patch(GET_CLIENT, return_value=client), patch(GET_SETTINGS, return_value=_settings()):
        events = await open_stream(request)
        return [event async for event in events]


async def _drain_interrupts() -> None:
    await asyncio.gather(*generation._background_tasks)


@pytest.mark.asyncio
async def test_handle_when_generation_succeeds_should_return_inline_images():
    # Arrange
    client = make_swarmui_client()
    request = make_generate_images_request()

    # Act
    with patch(GET_CLIENT, return_value=client), patch(GET_SETTINGS, return_value=_settings()):
        response = await handle(request)

    # Assert
    assert response.model == "flux-2-klein-9b"
    assert len(response.data) == 1
    image = response.data[0]
    assert image.b64_json == PNG_B64
    assert image.seed == 42
    assert image.size == "1024x1024"
    assert image.output_format == "png"
    client.interrupt.assert_not_awaited()


@pytest.mark.asyncio
async def test_handle_when_only_prompt_should_use_default_model_and_its_preset():
    # Arrange
    client = make_swarmui_client()
    request = make_generate_images_request()

    # Act
    with patch(GET_CLIENT, return_value=client), patch(GET_SETTINGS, return_value=_settings()):
        await handle(request)

    # Assert
    client.call.assert_awaited_once_with(
        "DescribeModel", "session-1", modelName="flux-2-klein-9b", subtype="Stable-Diffusion"
    )
    payload = _sent_payload(client)
    assert payload["model"] == "flux-2-klein-9b"
    assert payload["images"] == 1
    assert (payload["width"], payload["height"]) == (1024, 1024)
    assert (payload["steps"], payload["cfgscale"]) == (10, 1.0)
    assert payload["donotsave"] is False
    assert payload["nopreviews"] is True
    assert "seed" not in payload


@pytest.mark.asyncio
async def test_handle_when_sdxl_model_should_apply_sdxl_preset():
    # Arrange
    description = make_model_description(
        name="OfficialStableDiffusion/sd_xl_base_1.0.safetensors",
        architecture="stable-diffusion-xl-v1-base",
        compat_class="stable-diffusion-xl-v1",
    )

    client = make_swarmui_client(description=description)
    request = make_generate_images_request(model="OfficialStableDiffusion/sd_xl_base_1.0")

    # Act
    with patch(GET_CLIENT, return_value=client), patch(GET_SETTINGS, return_value=_settings()):
        response = await handle(request)

    # Assert
    payload = _sent_payload(client)
    assert (payload["steps"], payload["cfgscale"]) == (30, 6.0)
    assert response.model == "OfficialStableDiffusion/sd_xl_base_1.0"


@pytest.mark.asyncio
async def test_handle_when_model_has_no_preset_should_leave_sampling_to_swarmui():
    # Arrange
    description = make_model_description(name="mystery.safetensors", architecture="mystery", compat_class="mystery")
    client = make_swarmui_client(description=description)
    request = make_generate_images_request(model="mystery")

    # Act
    with patch(GET_CLIENT, return_value=client), patch(GET_SETTINGS, return_value=_settings()):
        await handle(request)

    # Assert
    payload = _sent_payload(client)
    assert "steps" not in payload
    assert "cfgscale" not in payload
    assert (payload["width"], payload["height"]) == (1024, 1024)


@pytest.mark.asyncio
async def test_handle_when_overrides_given_should_prefer_explicit_then_params_then_preset():
    # Arrange
    client = make_swarmui_client()
    request = make_generate_images_request(
        n=2,
        size="768x512",
        steps=6,
        seed=7,
        save=False,
        previews=True,
        params={"steps": 10, "cfgscale": 2.0, "loras": "detail", "session_id": "evil", "images": 50},
    )

    # Act
    with patch(GET_CLIENT, return_value=client), patch(GET_SETTINGS, return_value=_settings()):
        await handle(request)

    # Assert
    payload = _sent_payload(client)
    assert payload["steps"] == 6
    assert payload["cfgscale"] == 2.0
    assert payload["loras"] == "detail"
    assert payload["images"] == 2
    assert "session_id" not in payload
    assert (payload["width"], payload["height"]) == (768, 512)
    assert payload["seed"] == 7
    assert payload["donotsave"] is True
    assert payload["nopreviews"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("message", ["Model not found.", "model NOT FOUND", "Model 'nope' not found!"])
async def test_handle_when_model_not_found_should_raise_model_not_found(message):
    # Arrange
    client = make_swarmui_client()
    client.call.side_effect = SwarmUIError(message)
    request = make_generate_images_request(model="nope")

    # Act & Assert
    with (
        patch(GET_CLIENT, return_value=client),
        patch(GET_SETTINGS, return_value=_settings()),
        pytest.raises(SwarmUIModelNotFoundError),
    ):
        await handle(request)
    client.open_generation.assert_not_awaited()


@pytest.mark.asyncio
async def test_handle_when_describe_fails_otherwise_should_reraise():
    # Arrange
    client = make_swarmui_client()
    client.call.side_effect = SwarmUIError("Invalid session ID.")
    request = make_generate_images_request()

    # Act & Assert
    with (
        patch(GET_CLIENT, return_value=client),
        patch(GET_SETTINGS, return_value=_settings()),
        pytest.raises(SwarmUIError) as exc_info,
    ):
        await handle(request)
    assert not isinstance(exc_info.value, SwarmUIModelNotFoundError)


@pytest.mark.asyncio
async def test_handle_when_swarmui_reports_error_should_raise_and_interrupt():
    # Arrange
    client = make_swarmui_client(messages=[make_swarm_status(waiting_gens=1), make_swarm_error("bad sampler")])
    request = make_generate_images_request()

    # Act & Assert
    with (
        patch(GET_CLIENT, return_value=client),
        patch(GET_SETTINGS, return_value=_settings()),
        pytest.raises(SwarmUIError, match="bad sampler"),
    ):
        await handle(request)
    await _drain_interrupts()
    client.interrupt.assert_awaited_once_with("session-1")


@pytest.mark.asyncio
async def test_open_stream_when_generation_succeeds_should_emit_stages_progress_completed_done():
    # Arrange
    client = make_swarmui_client()
    request = make_generate_images_request(stream=True)

    # Act
    events = await _collect(request, client)

    # Assert
    assert [event.type for event in events] == [
        "image_generation.status",
        "image_generation.status",
        "image_generation.status",
        "image_generation.progress",
        "image_generation.progress",
        "image_generation.completed",
        "image_generation.done",
    ]
    assert [event.stage for event in events if isinstance(event, StatusEvent)] == [
        "queued",
        "loading_model",
        "generating",
    ]
    assert [event.sequence_number for event in events] == list(range(len(events)))
    done = events[-1]
    assert isinstance(done, DoneEvent)
    assert done.count == 1
    assert done.model == "flux-2-klein-9b"


@pytest.mark.asyncio
async def test_open_stream_when_progress_goes_backwards_should_keep_highest_percent():
    # Arrange
    messages = [
        make_swarm_progress(overall_percent=0.8, preview="data:image/jpeg;base64,PREVIEW"),
        make_swarm_progress(overall_percent=0.75, current_percent=0.5),
    ]
    client = make_swarmui_client(messages=messages)
    request = make_generate_images_request(stream=True)

    # Act
    events = await _collect(request, client)

    # Assert
    progress = [event for event in events if isinstance(event, ProgressEvent)]
    assert [event.percent for event in progress] == [0.8, 0.8]
    assert progress[0].preview_b64 == "PREVIEW"
    assert progress[1].preview_b64 is None
    assert progress[1].step_percent == 0.5


@pytest.mark.asyncio
async def test_open_stream_when_several_images_should_report_total_percent_across_them():
    # Arrange
    client = make_swarmui_client(messages=make_swarm_messages(n=2))
    request = make_generate_images_request(stream=True, n=2)

    # Act
    events = await _collect(request, client)

    # Assert
    totals = [event.total_percent for event in events if isinstance(event, ProgressEvent)]
    assert totals == [0.25, 0.5, 0.75, 1.0]
    completed = [event for event in events if isinstance(event, CompletedEvent)]
    assert [(event.image_index, event.seed) for event in completed] == [(0, 42), (1, 43)]


@pytest.mark.asyncio
async def test_open_stream_when_image_saved_to_file_should_fetch_and_inline_it():
    # Arrange
    view_path = "View/local/raw/2026-10-17/fox.webp"
    client = make_swarmui_client(messages=[make_swarm_image(image=view_path)])
    request = make_generate_images_request(stream=True)

    # Act
    events = await _collect(request, client)

    # Assert
    client.fetch_file.assert_awaited_once_with(view_path)
    completed = events[0]
    assert isinstance(completed, CompletedEvent)
    assert completed.b64_json == base64.b64encode(b"\x89PNG").decode()
    assert completed.output_format == "webp"


@pytest.mark.asyncio
async def test_open_stream_when_intermediate_images_should_skip_them():
    # Arrange
    messages = [make_swarm_progress(batch_index=-11), make_swarm_image(batch_index=-11), make_swarm_image()]
    client = make_swarmui_client(messages=messages)
    request = make_generate_images_request(stream=True)

    # Act
    events = await _collect(request, client)

    # Assert
    assert [event.type for event in events] == ["image_generation.completed", "image_generation.done"]


@pytest.mark.asyncio
async def test_open_stream_when_metadata_unparseable_should_fall_back_to_requested_size():
    # Arrange
    image = make_swarm_image()
    image["metadata"] = "not json"
    client = make_swarmui_client(messages=[image])
    request = make_generate_images_request(stream=True, size="512x768")

    # Act
    events = await _collect(request, client)

    # Assert
    completed = events[0]
    assert isinstance(completed, CompletedEvent)
    assert completed.seed is None
    assert completed.size == "512x768"


@pytest.mark.asyncio
async def test_open_stream_when_no_size_known_should_leave_size_unset():
    # Arrange
    image = make_swarm_image()
    image["metadata"] = "not json"
    description = make_model_description(width=0, height=0)
    client = make_swarmui_client(description=description, messages=[image])
    request = make_generate_images_request(stream=True)

    # Act
    events = await _collect(request, client)

    # Assert
    completed = events[0]
    assert isinstance(completed, CompletedEvent)
    assert completed.size is None
    assert "width" not in _sent_payload(client)


@pytest.mark.asyncio
async def test_open_stream_when_swarmui_reports_error_should_end_with_error_event_and_interrupt():
    # Arrange
    client = make_swarmui_client(messages=[make_swarm_status(waiting_gens=1), make_swarm_error("bad sampler")])
    request = make_generate_images_request(stream=True)

    # Act
    events = await _collect(request, client)
    await _drain_interrupts()

    # Assert
    error = events[-1]
    assert isinstance(error, ErrorEvent)
    assert error.message == "bad sampler"
    assert error.code == "upstream_error"
    assert error.sequence_number == 1
    client.interrupt.assert_awaited_once_with("session-1")


@pytest.mark.asyncio
async def test_open_stream_when_caller_stops_early_should_interrupt_generation():
    # Arrange
    client = make_swarmui_client(messages=make_swarm_messages(n=4))
    request = make_generate_images_request(stream=True, n=4)

    # Act
    with patch(GET_CLIENT, return_value=client), patch(GET_SETTINGS, return_value=_settings()):
        events = await open_stream(request)
        async with aclosing(events):
            await anext(events)
    await _drain_interrupts()

    # Assert
    client.interrupt.assert_awaited_once_with("session-1")


@pytest.mark.asyncio
async def test_open_stream_when_generation_completes_should_not_interrupt():
    # Arrange
    client = make_swarmui_client()
    request = make_generate_images_request(stream=True)

    # Act
    await _collect(request, client)
    await _drain_interrupts()

    # Assert
    client.interrupt.assert_not_awaited()


@pytest.mark.asyncio
async def test_open_stream_when_interrupt_fails_should_not_raise():
    # Arrange
    client = make_swarmui_client(messages=[make_swarm_error()])
    client.interrupt.side_effect = SwarmUIError("gone")
    request = make_generate_images_request(stream=True)

    # Act
    events = await _collect(request, client)
    await _drain_interrupts()

    # Assert
    assert isinstance(events[-1], ErrorEvent)
    client.interrupt.assert_awaited_once()


async def _hanging_messages():
    yield make_swarm_status(waiting_gens=1)
    await asyncio.sleep(10)
    yield make_swarm_image()


@pytest.mark.asyncio
async def test_open_stream_when_generation_times_out_should_end_with_timeout_error_and_interrupt():
    # Arrange
    client = make_swarmui_client()
    client.open_generation.return_value = _hanging_messages()
    request = make_generate_images_request(stream=True)

    # Act
    with patch(GET_CLIENT, return_value=client), patch(GET_SETTINGS, return_value=_settings(timeout=0.05)):
        events = [event async for event in await open_stream(request)]
    await _drain_interrupts()

    # Assert
    error = events[-1]
    assert isinstance(error, ErrorEvent)
    assert error.code == "timeout"
    client.interrupt.assert_awaited_once_with("session-1")


@pytest.mark.asyncio
async def test_handle_when_generation_times_out_should_raise_timeout():
    # Arrange
    client = make_swarmui_client()
    client.open_generation.return_value = _hanging_messages()
    request = make_generate_images_request()

    # Act & Assert
    with (
        patch(GET_CLIENT, return_value=client),
        patch(GET_SETTINGS, return_value=_settings(timeout=0.05)),
        pytest.raises(SwarmUITimeoutError),
    ):
        await handle(request)
    await _drain_interrupts()


@pytest.mark.asyncio
async def test_handle_when_loras_given_should_send_resolved_names_and_weights():
    # Arrange
    loras = [
        make_lora_description(name="styles/ink-wash.safetensors"),
        make_lora_description(name="detail.safetensors", default_weight="0.7"),
        make_lora_description(name="glow.safetensors", default_weight="0.7"),
    ]
    client = make_swarmui_client(loras=loras)
    request = make_generate_images_request(
        loras=[{"name": "styles/ink-wash"}, {"name": "detail.safetensors"}, {"name": "glow", "weight": -0.5}]
    )

    # Act
    with patch(GET_CLIENT, return_value=client), patch(GET_SETTINGS, return_value=_settings()):
        await handle(request)

    # Assert
    client.call.assert_any_await("DescribeModel", "session-1", modelName="styles/ink-wash", subtype="LoRA")
    payload = _sent_payload(client)
    assert payload["loras"] == ["styles/ink-wash", "detail", "glow"]
    assert payload["loraweights"] == [1.0, 0.7, -0.5]


@pytest.mark.asyncio
async def test_handle_when_no_loras_should_leave_lora_params_out():
    # Arrange
    client = make_swarmui_client()
    request = make_generate_images_request()

    # Act
    with patch(GET_CLIENT, return_value=client), patch(GET_SETTINGS, return_value=_settings()):
        await handle(request)

    # Assert
    payload = _sent_payload(client)
    assert "loras" not in payload
    assert "loraweights" not in payload


@pytest.mark.asyncio
async def test_handle_when_lora_unknown_should_raise_model_not_found_before_generating():
    # Arrange
    client = make_swarmui_client(loras=[make_lora_description()])
    request = make_generate_images_request(loras=[{"name": "styles/ink-wash"}, {"name": "missing"}])

    # Act & Assert
    with patch(GET_CLIENT, return_value=client), patch(GET_SETTINGS, return_value=_settings()):
        with pytest.raises(SwarmUIModelNotFoundError, match="LoRA 'missing' not found"):
            await handle(request)
    client.open_generation.assert_not_awaited()


@pytest.mark.asyncio
async def test_handle_when_lora_built_for_another_model_should_still_apply_it():
    # Arrange
    client = make_swarmui_client(loras=[make_lora_description(compat_class="stable-diffusion-xl-v1")])
    request = make_generate_images_request(loras=[{"name": "styles/ink-wash"}])

    # Act
    with patch(GET_CLIENT, return_value=client), patch(GET_SETTINGS, return_value=_settings()):
        await handle(request)

    # Assert
    assert _sent_payload(client)["loras"] == ["styles/ink-wash"]


@pytest.mark.asyncio
async def test_handle_when_init_image_given_should_send_image_mask_and_creativity():
    # Arrange
    client = make_swarmui_client()
    request = make_generate_images_request()
    init = InitImage(image="aW1hZ2U=", width=1024, height=1024, mask="bWFzaw==", creativity=0.4)

    # Act
    with patch(GET_CLIENT, return_value=client), patch(GET_SETTINGS, return_value=_settings()):
        await handle(request, init)

    # Assert
    payload = _sent_payload(client)
    assert payload["initimage"] == "aW1hZ2U="
    assert payload["maskimage"] == "bWFzaw=="
    assert payload["initimagecreativity"] == 0.4


@pytest.mark.asyncio
async def test_handle_when_init_image_without_mask_or_creativity_should_leave_them_to_swarmui():
    # Arrange
    client = make_swarmui_client()
    request = make_generate_images_request()

    # Act
    with patch(GET_CLIENT, return_value=client), patch(GET_SETTINGS, return_value=_settings()):
        await handle(request, InitImage(image="aW1hZ2U=", width=1024, height=1024))

    # Assert
    payload = _sent_payload(client)
    assert "maskimage" not in payload
    assert "initimagecreativity" not in payload


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("image_size", "model_size", "expected"),
    [
        ((64, 48), (1024, 1024), (1184, 880)),  # upscaled to the model's pixel count, aspect kept
        ((1536, 1024), (1024, 1024), (1248, 832)),  # downscaled the same way
        ((800, 600), (0, 0), (800, 608)),  # no standard size: the image's own, rounded to multiples of 16
        ((4000, 100), (1024, 1024), (2048, 256)),  # extreme aspect ratios clamp to the size limits
    ],
)
async def test_handle_when_init_image_and_no_size_should_fit_image_aspect_to_model(image_size, model_size, expected):
    # Arrange
    description = make_model_description(width=model_size[0], height=model_size[1])
    client = make_swarmui_client(description=description)
    request = make_generate_images_request()
    init = InitImage(image="aW1hZ2U=", width=image_size[0], height=image_size[1])

    # Act
    with patch(GET_CLIENT, return_value=client), patch(GET_SETTINGS, return_value=_settings()):
        await handle(request, init)

    # Assert
    payload = _sent_payload(client)
    assert (payload["width"], payload["height"]) == expected


@pytest.mark.asyncio
async def test_handle_when_init_image_and_size_given_should_prefer_explicit_size_then_params():
    # Arrange
    init = InitImage(image="aW1hZ2U=", width=64, height=48)
    explicit = make_generate_images_request(size="512x512")
    from_params = make_generate_images_request(params={"width": 640, "height": 640})
    explicit_client, params_client = make_swarmui_client(), make_swarmui_client()

    # Act
    with patch(GET_SETTINGS, return_value=_settings()):
        with patch(GET_CLIENT, return_value=explicit_client):
            await handle(explicit, init)
        with patch(GET_CLIENT, return_value=params_client):
            await handle(from_params, init)

    # Assert
    assert (_sent_payload(explicit_client)["width"], _sent_payload(explicit_client)["height"]) == (512, 512)
    assert (_sent_payload(params_client)["width"], _sent_payload(params_client)["height"]) == (640, 640)


@pytest.fixture
def telemetry():
    """Record the generation spans in memory and mock the metric instruments."""
    with (
        patch_operation_telemetry() as recorded,
        patch.object(generation, "_wait_histogram") as wait,
        patch.object(generation, "_image_counter") as images,
    ):
        yield SimpleNamespace(
            spans=recorded.spans, duration=recorded.durations["image_generation"], wait=wait, images=images
        )


_ATTRIBUTES = {"image.operation": "generate", "image.model": "flux-2-klein-9b"}


@pytest.mark.asyncio
async def test_handle_when_generation_succeeds_should_record_span_and_metrics(telemetry):
    # Arrange
    client = make_swarmui_client(messages=make_swarm_messages(n=2))
    request = make_generate_images_request(n=2)

    # Act
    with patch(GET_CLIENT, return_value=client), patch(GET_SETTINGS, return_value=_settings()):
        await handle(request)

    # Assert
    (span,) = telemetry.spans.get_finished_spans()
    assert span.name == "image_generation"
    assert span.attributes["image.model"] == "flux-2-klein-9b"
    assert (span.attributes["outcome"], span.attributes["image.count"], span.attributes["image.n"]) == (
        "completed",
        2,
        2,
    )
    assert [event.name for event in span.events] == ["queued", "loading_model", "generating"]
    assert span.status.status_code == StatusCode.UNSET
    telemetry.duration.record.assert_called_once()
    assert telemetry.duration.record.call_args.args[1] == {**_ATTRIBUTES, "outcome": "completed"}
    telemetry.wait.record.assert_called_once()
    assert telemetry.wait.record.call_args.args[1] == {**_ATTRIBUTES, "image.model_loaded": True}
    telemetry.images.add.assert_called_once_with(2, _ATTRIBUTES)


@pytest.mark.asyncio
async def test_handle_when_init_image_given_should_record_edit_operation(telemetry):
    # Arrange
    client = make_swarmui_client()
    init = InitImage(image=PNG_B64, width=1024, height=1024)

    # Act
    with patch(GET_CLIENT, return_value=client), patch(GET_SETTINGS, return_value=_settings()):
        await handle(make_generate_images_request(), init)

    # Assert
    (span,) = telemetry.spans.get_finished_spans()
    assert span.attributes["image.operation"] == "edit"
    telemetry.images.add.assert_called_once_with(1, {**_ATTRIBUTES, "image.operation": "edit"})


@pytest.mark.asyncio
async def test_open_when_swarmui_called_should_make_lookups_children_of_generation_span(telemetry):
    # Arrange
    client = make_swarmui_client()
    current_spans = []
    client.new_session.side_effect = lambda: current_spans.append(trace.get_current_span()) or "session-1"

    # Act
    await _collect(make_generate_images_request(stream=True), client)

    # Assert
    (span,) = telemetry.spans.get_finished_spans()
    assert current_spans[0].get_span_context().span_id == span.context.span_id


@pytest.mark.asyncio
async def test_open_stream_when_swarmui_reports_error_should_record_failed_span(telemetry):
    # Arrange
    client = make_swarmui_client(messages=[make_swarm_error("bad sampler")])

    # Act
    await _collect(make_generate_images_request(stream=True), client)
    await _drain_interrupts()

    # Assert
    (span,) = telemetry.spans.get_finished_spans()
    assert span.attributes["outcome"] == "failed"
    assert span.status.status_code == StatusCode.ERROR
    assert span.events[-1].name == "exception"
    assert telemetry.duration.record.call_args.args[1]["outcome"] == "failed"
    telemetry.images.add.assert_not_called()


@pytest.mark.asyncio
async def test_open_stream_when_caller_stops_early_should_record_interrupted_span(telemetry):
    # Arrange
    client = make_swarmui_client(messages=make_swarm_messages(n=4))

    # Act
    with patch(GET_CLIENT, return_value=client), patch(GET_SETTINGS, return_value=_settings()):
        events = await open_stream(make_generate_images_request(stream=True, n=4))
        async with aclosing(events):
            await anext(events)
    await _drain_interrupts()

    # Assert
    (span,) = telemetry.spans.get_finished_spans()
    assert span.attributes["outcome"] == "interrupted"
    assert span.status.status_code == StatusCode.UNSET
    assert telemetry.duration.record.call_args.args[1]["outcome"] == "interrupted"


@pytest.mark.asyncio
async def test_open_stream_when_no_generating_status_should_record_wait_at_first_progress(telemetry):
    # Arrange
    messages = [make_swarm_progress(overall_percent=0.5), make_swarm_image()]
    client = make_swarmui_client(messages=messages)

    # Act
    await _collect(make_generate_images_request(stream=True), client)

    # Assert
    telemetry.wait.record.assert_called_once()
    assert telemetry.wait.record.call_args.args[1]["image.model_loaded"] is False


@pytest.mark.asyncio
async def test_handle_when_model_not_found_should_end_span_with_error_and_record_no_metrics(telemetry):
    # Arrange
    client = make_swarmui_client()
    client.call.side_effect = SwarmUIError("Model not found.")

    # Act
    with (
        patch(GET_CLIENT, return_value=client),
        patch(GET_SETTINGS, return_value=_settings()),
        pytest.raises(SwarmUIModelNotFoundError),
    ):
        await handle(make_generate_images_request(model="nope"))

    # Assert
    (span,) = telemetry.spans.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes["image.model"] == "nope"
    telemetry.duration.record.assert_not_called()
