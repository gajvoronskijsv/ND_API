import keras
import numpy as np
from .AbstractModel import AbstractModel


class NeuralNetModel(AbstractModel):
    """
    Singleton neural net keras model
    """
    FILEPATH = r"core/models/neural_network.keras"
    instance = None

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls.instance, cls):
            cls.instance = object.__new__(cls, *args, **kwargs)
            cls.neural_net = keras.saving.load_model(cls.FILEPATH,
                                                     custom_objects=None,
                                                     compile=True,
                                                     safe_mode=True)
        return cls.instance

    def calculate(self, x):
        x = np.array([x])
        y = self.neural_net(x)
        return np.array(y[0]), None
