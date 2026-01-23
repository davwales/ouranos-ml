from ouranos_ml.shared.infra.clients.lm_studio_client import get_client


async def list_downloaded_models() -> list[str]:
    """Lists all downloaded models in LMStudio."""
    async with get_client() as client:
        models = await client.list_downloaded_models()
        return [m.model_key for m in models]
