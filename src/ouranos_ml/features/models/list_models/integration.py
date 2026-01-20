from ouranos_ml.shared.infra.clients.lm_studio_client import LMStudioClient


def list_downloaded_models() -> list[str]:
    """Lists all downloaded models in LMStudio."""
    with LMStudioClient() as client:
        models = client.list_downloaded_models()
        return [m.model_key for m in models]
