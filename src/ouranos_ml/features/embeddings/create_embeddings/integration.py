from collections.abc import Sequence

from ouranos_ml.shared.infra.clients.lm_studio_client import LMStudioClient


def embed(model_key: str, input: str | list[str]) -> Sequence[int | float] | Sequence[Sequence[int | float]]:
    """Creates embeddings using lmstudio for the given request."""
    with LMStudioClient() as client:
        model = client.embedding.model(model_key)
        return model.embed(input)


def count_tokens(model_key: str, input: str | list[str]) -> list[int]:
    """Returns the number of tokens in the input."""
    with LMStudioClient() as client:
        model = client.embedding.model(model_key)

        if isinstance(input, str):
            return [model.count_tokens(input)]

        return [model.count_tokens(x) for x in list(input)]
