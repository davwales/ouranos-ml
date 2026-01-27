from pydantic import BaseModel


def server_side_event[T: BaseModel](data: T, event: str | None = None) -> str:
    """Helper function to create server-side events."""
    parts: list[str] = []
    if event:
        parts.append(f"event: {event}")
    parts.append(f"data: {data.model_dump_json()}")
    return "\n".join(parts) + "\n\n"


def done_event() -> str:
    """Helper function to create a server-side event indicating completion."""
    return "data: [DONE]\n\n"
