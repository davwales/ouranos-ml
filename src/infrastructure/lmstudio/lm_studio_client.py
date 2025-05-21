import os
from typing import Optional
import lmstudio as lms


class LMStudioClient:
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or os.getenv("LMSTUDIO_BASE_URL")
        self._client = None

    def __enter__(self):
        try:
            self._client = lms.Client(self.base_url)
            return self._client
        except Exception as e:
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            self._client.close()
        self._client = None
        return False
