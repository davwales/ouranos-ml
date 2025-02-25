from enum import StrEnum, auto

class Role(StrEnum):
    SYSTEM = auto()
    USER = auto()
    ASSISTANT = auto()
