import numpy as np
from .AbstractOptimiser import AbstractOptimiser


class BoltzmannOptimiser(AbstractOptimiser):
    """Поиск глобального минимума методом Монте-Карло, Больцмановский отжиг"""

    # функция вероятности принятия нового состояния системы
    @staticmethod
    def h(deltaE, T):
        # точное значение
        # return 1 / (1 + np.exp(deltaE / T))
        # приближенное значение
        return np.exp(-deltaE / T)

    # закон изменения температуры
    @staticmethod
    def T(k, To):
        # Больцмановскйи отжиг
        return To / np.log(1 + k)

    # порождающее семейство распределений
    def G(self, x, T):
        max = np.array([self.cHClMax, self.cNaOHMax, self.DBLMax])
        min = np.array([self.cHClMin, self.cNaOHMin, self.DBLMin])
        rng = np.random.default_rng()
        for i in range(1, 4):
            # получаем приближенное нормальное распределение
            S = rng.standard_normal()
            # получаем распределение g(0, T) (N(mu, rho ^ 2))
            S = np.sqrt(T) * S
            # придаем возмущение соотв.элементу вектора
            # x[i] += S
            x[i] = (max[i - 1] - min[i - 1]) / 2 * S + x[i]
        return x

    def optimize(self):
        """Функция поиска глобального минимума методом Больцмановского отжига"""
        x = np.zeros(4)
        x[0] = self.cNaCl
        x[1] = self.cHClMin + (self.cHClMax - self.cHClMin) / 2
        x[2] = self.cNaOHMin + (self.cNaOHMax - self.cNaOHMin) / 2
        x[3] = self.DBLMin + (self.DBLMax - self.DBLMin) / 2

        xMin = np.zeros(4)
        xMin[0] = x[0]
        xMin[1] = x[1]
        xMin[2] = x[2]
        xMin[3] = x[3]

        yMin = self.model(xMin)

        # Энергия системы
        E = self.tf(yMin)
        # Начальная температура
        To = 1 / 12
        rng = np.random.default_rng()
        for k in range(1, int(1e3)):
            t = self.T(k, To)
            x = self.G(x, t)
            y = self.model(x)
            new_E = self.tf(y)
            if rng.uniform() < self.h(new_E - E, t):
                E = new_E,
                xMin[0] = x[0]
                xMin[1] = x[1]
                xMin[2] = x[2]
                xMin[3] = x[3]
                yMin[0] = y[0]
                yMin[1] = y[1]
                yMin[2] = y[2]
                yMin[3] = y[3]
        return yMin, xMin
