import os
from .generator_registration import GeneratorRegistration
from .generator_type import GeneratorType

class GeneratorFactory():
    allocated_memory: int = 0
    available_memory: int = 0
    known_generators: list[GeneratorRegistration] = []
    loaded_generators: list[GeneratorRegistration] = []

    def __init__(self, available_memory: int, generators: list[GeneratorRegistration]):
        self.available_memory = available_memory
        self.known_generators = generators

    def get_generator(self, type: GeneratorType):
        loaded_reservation = self.__find_reservation(type, self.loaded_generators)
        if loaded_reservation is not None:
            return loaded_reservation.generator
        print(f"Loading {type} generator...")
        reservation = self.__load_generator(type)
        return reservation.generator
        
    def __load_generator(self, type: GeneratorType):
        reservation = self.__find_reservation(type, self.known_generators)
        if reservation is None:
            raise Exception("Could not find generator reservation.")
        
        if self.allocated_memory + reservation.memory_reservation > self.available_memory:
            delta = reservation.memory_reservation - (self.available_memory - self.allocated_memory)
            self.__free_memory(delta)

        reservation.generator.load_model()
        self.allocated_memory += reservation.memory_reservation
        self.loaded_generators.append(reservation)
        print(f"Loaded generator: {str(type)}")
        return reservation

    def __unload_generator(self, reservation: GeneratorRegistration):
        reservation.generator.unload_model()
        self.allocated_memory -= reservation.memory_reservation
        self.loaded_generators.remove(reservation)
        print(f"Unloaded generator: {str(reservation.type)}")

    def __find_reservation(self, type: GeneratorType, reservation_list: list[GeneratorRegistration]):
        for r in reservation_list:
            if r.type == type:
                return r
        return None
    
    def __free_memory(self, amount: int):
        if amount > self.available_memory:
            raise Exception("Attempting to free more memory than available.")
        amount_freed = 0
        for g in self.loaded_generators:
            self.__unload_generator(g)
            amount_freed += g.memory_reservation
            if amount_freed >= amount:
                break
