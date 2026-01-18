from types import TracebackType

import lmstudio as lms

from ouranos_ml.domain.settings.settings import settings


class LMStudioClient:
    """A client for interacting with the LM Studio API."""

    def __init__(self, base_url: str | None = None) -> None:
        self._base_url = settings.lmstudio_base_url
        self._client = None

    def __enter__(self) -> lms.Client:
        self._client = lms.Client(self._base_url)
        return self._client

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None
    ) -> bool | None:
        if self._client:
            self._client.close()
        self._client = None
        return False
