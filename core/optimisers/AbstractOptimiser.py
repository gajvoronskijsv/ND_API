from abc import abstractmethod
import numpy as np


class AbstractOptimiser:
    """Мета-параметры оптимизатора"""

    def __init__(self,
                 input_cNaCl,
                 input_cHClMin,
                 input_cHClMax,
                 input_cNaOHMin,
                 input_cNaOHMax,
                 input_DBLMin,
                 input_DBLMax,
                 input_MODEL):
        self.cNaCl = input_cNaCl
        self.cHClMin = input_cHClMin
        self.cHClMax = input_cHClMax
        self.cNaOHMin = input_cNaOHMin
        self.cNaOHMax = input_cNaOHMax
        self.DBLMin = input_DBLMin
        self.DBLMax = input_DBLMax
        self.MODEL = input_MODEL

    @abstractmethod
    def optimize(self):
        pass

    @staticmethod
    def tf(y):
        """Вычисляет целевую функцию оптимизации"""
        # y0 = kappaTime
        # y1 = kappa
        # y2 = phTime
        # y3 = ph
        dev = 0
        dev += pow(y[0] - y[2], 2)
        # dev += (abs(y[0]) / 200 + abs(y[2]) / 200) / 2
        dev += pow(y[1] - 1, 2)
        dev += pow(y[3] - 7.5, 2)
        # if (abs(y[3] - 7.5) > 1) dev += abs(y[3] - 7.5) / 8
        # else dev += abs(y[3] - 7.5) / 80
        return dev

    def model_tf(self, NaCl, params):
        """Возвращает целевую функцию от результатов моделирования на входных параметрах"""
        x = np.array([NaCl, params[0], params[1], params[2]])
        y, _ = self.MODEL.calculate(x)
        return self.tf(y)

    def eval_params_3d(self, x, shift=0):
        """Проверка параметров на соответствие граничным условиям"""
        # HCl
        if x[0 + shift] < self.cHClMin:
            x[0 + shift] = self.cHClMin
        if x[0 + shift] > self.cHClMax:
            x[0 + shift] = self.cHClMax
        # NaOH
        if x[1 + shift] < self.cNaOHMin:
            x[1 + shift] = self.cNaOHMin
        if x[1 + shift] > self.cNaOHMax:
            x[1 + shift] = self.cNaOHMax
        # DBL
        if x[2 + shift] < self.DBLMin:
            x[2 + shift] = self.DBLMin
        if x[2 + shift] > self.DBLMax:
            x[2 + shift] = self.DBLMax


    def eval_params(self, x):
        """Проверка параметров на соответствие граничным условиям со сдвигом на 1"""
        self.eval_params_3d(x, shift=1)

    def model(self, x):
        """Запускает математическое либо нейросетевое моделирование в зависимости от глобальных настроек"""
        self.eval_params(x)
        y, log = self.MODEL.calculate(x)
        return y

