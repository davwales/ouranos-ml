from abc import ABCMeta

class BaseGenerator(metaclass=ABCMeta):
    def load_model(self):
        pass

    def unload_model(self):
        pass
