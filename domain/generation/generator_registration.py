from .generator_type import GeneratorType
from application.generation import BaseGenerator
from datetime import datetime

class GeneratorRegistration:
    type: GeneratorType
    memory_reservation: int
    generator: BaseGenerator
    last_used: datetime

    def __init__(self, type: GeneratorType, memory_reservation: int, generator: BaseGenerator):
        self.type = type
        self.memory_reservation = memory_reservation
        self.generator = generator
        self.last_used = datetime.utcnow()
