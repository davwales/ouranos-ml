from asyncio import gather
from collections.abc import Sequence

from ouranos_ml.shared.infra.clients.lm_studio_client import get_client


async def embed(model_key: str, input: str | list[str]) -> Sequence[int | float] | Sequence[Sequence[int | float]]:
    """Creates embeddings using lmstudio for the given request."""
    async with get_client() as client:
        model = await client.embedding.model(model_key)
        return await model.embed(input)


async def count_tokens(model_key: str, input: str | list[str]) -> list[int]:
    """Returns the number of tokens in the input."""
    async with get_client() as client:
        model = await client.embedding.model(model_key)

        if isinstance(input, str):
            return [await model.count_tokens(input)]

        tasks = [model.count_tokens(x) for x in list(input)]
        return await gather(*tasks)
