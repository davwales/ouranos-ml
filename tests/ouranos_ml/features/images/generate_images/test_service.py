import asyncio
import base64
from contextlib import aclosing
from unittest.mock import MagicMock, patch

import pytest

from ouranos_ml.features.images.generate_images import service
from ouranos_ml.features.images.generate_images.schemas import (
    CompletedEvent,
    DoneEvent,
    ErrorEvent,
    ProgressEvent,
    StatusEvent,
)
from ouranos_ml.features.images.generate_images.service import handle, open_stream
from ouranos_ml.shared.infra.swarmui.errors import SwarmUIError, SwarmUIModelNotFoundError, SwarmUITimeoutError
from tests.ouranos_ml.shared.factories.images_factories import (
    PNG_B64,
    make_generate_images_request,
    make_model_description,
    make_swarm_error,
    make_swarm_image,
    make_swarm_messages,
    make_swarm_progress,
    make_swarm_status,
    make_swarmui_client,
)

GET_CLIENT = "ouranos_ml.features.images.generate_images.service.get_swarmui_client"
GET_SETTINGS = "ouranos_ml.features.images.generate_images.service.get_settings"


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
    await asyncio.gather(*service._background_tasks)


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
    client.call.assert_awaited_once_with("DescribeModel", "session-1", modelName="flux-2-klein-9b")
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
async def test_handle_when_model_not_found_should_raise_model_not_found():
    # Arrange
    client = make_swarmui_client()
    client.call.side_effect = SwarmUIError("Model not found.")
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
