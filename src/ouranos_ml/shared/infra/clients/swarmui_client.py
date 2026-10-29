"""Thin transport client for SwarmUI's native API (``/API/{route}`` POSTs and WebSocket routes)."""

import asyncio
import json
from collections.abc import AsyncGenerator
from functools import lru_cache
from typing import Any

import httpx
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosedError, WebSocketException

from ouranos_ml.shared.domain.core.settings import get_settings
from ouranos_ml.shared.infra.swarmui.errors import (
    SwarmUIError,
    SwarmUIInvalidSessionError,
    SwarmUINotConfiguredError,
    SwarmUITimeoutError,
    SwarmUIUnavailableError,
)

_REQUEST_TIMEOUT_SECONDS = 30.0


class SwarmUIClient:
    """Calls SwarmUI routes, raising ``SwarmUIError`` for failures (SwarmUI reports most errors with HTTP 200)."""

    def __init__(self, base_url: str, token: str = "", http: httpx.AsyncClient | None = None) -> None:
        self._base_url = base_url.rstrip("/")
        self._headers = {"Cookie": f"swarm_token={token}"} if token else {}
        self._http = http or httpx.AsyncClient(
            base_url=self._base_url,
            headers=self._headers,
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
        self._shared_session_id: str | None = None
        self._shared_session_lock = asyncio.Lock()

    async def new_session(self) -> str:
        """Start a new SwarmUI session; generations and interrupts are scoped to it."""
        response = await self._post("GetNewSession", {})
        return response["session_id"]

    async def call(self, route: str, session_id: str, **body: Any) -> dict[str, Any]:
        """POST ``/API/{route}`` within a session and return the JSON reply."""
        return await self._post(route, {"session_id": session_id, **body})

    async def call_shared(self, route: str, **body: Any) -> dict[str, Any]:
        """POST ``/API/{route}`` in a session reused across calls, for routes that need no session of their own.

        SwarmUI keeps sessions for a month, so polling with a new session each time would pile them up. The shared
        session is created on first use and replaced once if SwarmUI reports it invalid. Generations still need their
        own session, so ``InterruptAll`` cannot touch anyone else's.
        """
        session_id = await self._shared_session(stale=None)
        try:
            return await self.call(route, session_id, **body)
        except SwarmUIInvalidSessionError:
            return await self.call(route, await self._shared_session(stale=session_id), **body)

    async def _shared_session(self, stale: str | None) -> str:
        """Return the shared session, starting one if there is none yet or it is still the ``stale`` one."""
        async with self._shared_session_lock:
            if self._shared_session_id is None or self._shared_session_id == stale:
                self._shared_session_id = await self.new_session()
            return self._shared_session_id

    async def interrupt(self, session_id: str) -> None:
        """Interrupt every generation running or queued in the session."""
        await self.call("InterruptAll", session_id, other_sessions=False)

    async def fetch_file(self, path: str) -> bytes:
        """GET a saved output, e.g. a ``View/...`` image path from a generation."""
        try:
            response = await self._http.get(f"/{path.lstrip('/')}")
        except httpx.TimeoutException as exc:
            raise SwarmUITimeoutError(f"SwarmUI timed out serving {path}") from exc
        except httpx.TransportError as exc:
            raise SwarmUIUnavailableError(f"SwarmUI is unreachable: {exc}") from exc
        if response.is_error:
            raise SwarmUIError(f"SwarmUI returned {response.status_code} for {path}")
        return response.content

    async def open_generation(self, session_id: str, payload: dict[str, Any]) -> AsyncGenerator[dict[str, Any]]:
        """Connect to ``GenerateText2ImageWS`` and send the request, then return an iterator over its messages.

        Connecting eagerly means an unreachable server raises here, before any caller output is sent.
        The iterator ends when SwarmUI announces it is closing the socket.
        """
        url = f"{self._base_url.replace('http', 'ws', 1)}/API/GenerateText2ImageWS"
        try:
            socket = await connect(url, additional_headers=self._headers, max_size=None)
            await socket.send(json.dumps({"session_id": session_id, **payload}))
        except (OSError, TimeoutError, WebSocketException) as exc:
            raise SwarmUIUnavailableError(f"SwarmUI is unreachable: {exc}") from exc
        return self._receive(socket)

    @staticmethod
    async def _receive(socket: ClientConnection) -> AsyncGenerator[dict[str, Any]]:
        """Yield parsed messages until SwarmUI says it is closing, always closing the socket afterwards."""
        try:
            async for raw in socket:
                message = json.loads(raw)
                if message.get("socket_intention") == "close":
                    return
                yield message
        except ConnectionClosedError as exc:
            raise SwarmUIUnavailableError(f"SwarmUI closed the connection: {exc}") from exc
        finally:
            await socket.close()

    async def _post(self, route: str, body: dict[str, Any]) -> dict[str, Any]:
        """POST to a SwarmUI API route, turning transport failures and ``error`` replies into exceptions."""
        try:
            response = await self._http.post(f"/API/{route}", json=body)
        except httpx.TimeoutException as exc:
            raise SwarmUITimeoutError(f"SwarmUI timed out on {route}") from exc
        except httpx.TransportError as exc:
            raise SwarmUIUnavailableError(f"SwarmUI is unreachable: {exc}") from exc
        try:
            data = response.json()
        except ValueError:
            data = {}
        if isinstance(data, dict) and "error" in data:
            if data.get("error_id") == "invalid_session_id":
                raise SwarmUIInvalidSessionError(str(data["error"]))
            raise SwarmUIError(str(data["error"]))
        if response.is_error or not isinstance(data, dict):
            raise SwarmUIError(f"SwarmUI returned {response.status_code} for {route}")
        return data


@lru_cache
def get_swarmui_client() -> SwarmUIClient:
    """Creates the SwarmUI client, failing when no SwarmUI server is configured."""
    settings = get_settings()
    if not settings.swarmui_base_url:
        raise SwarmUINotConfiguredError("Image generation is not configured (SWARMUI_BASE_URL is unset)")
    return SwarmUIClient(settings.swarmui_base_url, token=settings.swarmui_token)
