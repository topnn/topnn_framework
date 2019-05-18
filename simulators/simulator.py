from abc import ABC, abstractmethod


class Simulator(ABC):
    """
    Abstract class to capture the concept of a "simulator".
    """

    def __init__(self, **kwargs):
        self.pipeline = kwargs.get('pipeline')
        # validate(self.pipeline)

    @abstractmethod
    def run(self):
        raise NotImplementedError
