from enum import StrEnum, auto


class Role(StrEnum):
    """Represents the role of an actor within a conversation."""

    SYSTEM = auto()
    USER = auto()
    ASSISTANT = auto()
