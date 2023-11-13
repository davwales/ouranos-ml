from .generator_type import GeneratorType
from .base_generator import BaseGenerator

class GeneratorRegistration:
    type: GeneratorType
    memory_reservation: int
    generator: BaseGenerator

    def __init__(self, type: GeneratorType, memory_reservation: int, generator: BaseGenerator):
        self.type = type
        self.memory_reservation = memory_reservation
        self.generator = generator
