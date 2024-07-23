from abc import abstractmethod


class AbstractModel:
    @abstractmethod
    def calculate(self):
        pass
