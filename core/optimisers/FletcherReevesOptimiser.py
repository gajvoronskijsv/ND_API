import numpy as np
from .AbstractOptimiser import AbstractOptimiser
from tqdm import tqdm


class FletcherReevesOptimiser(AbstractOptimiser):
    """Численный метод оптимизации Флетчера-Ривза"""
    # Глобальыне параметры
    eps1 = 1e-3
    eps2 = 1e-3
    eps3 = 1e-3
    eps4 = 1e-3
    n = 3
    MAX_NUM_OF_ITERS = 5
    deltaBig = 1e-1

    def grad_f(self, x0, x):
        """Численное вычисление градиаента функции"""
        grad = np.zeros(self.n)
        for i in range(0, self.n):
            x[i] += self.deltaBig
            grad[i] = self.model_tf(x0, x)
            x[i] -= 2 * self.deltaBig
            grad[i] -= self.model_tf(x0, x)
            grad[i] /= (2 * self.deltaBig)
            x[i] += self.deltaBig
        return grad

    def norma(self, x1, x2):
        """Вычисляет норму по принципу минимального отклонения по модулю """
        norma = abs(x1[0] - x2[0])
        for i in range(1, self.n):
            if norma < abs(x1[i] - x2[i]): norma = abs(x1[i] - x2[i])
        return norma

    def norma_grad(self, grad):
        """Вычисляет норму градиента"""
        max = abs(grad[0])
        for i in range(1, self.n):
            if abs(grad[i]) > max: max = abs(grad[i])
        return max

    def g(self, a, NaCl, x0, S):
        """Вычисление целевого значения в процессе бинарного поиска"""
        x1 = np.zeros(self.n)
        for i in range(0, self.n):
            x1[i] = x0[i] - a * S[i]
        return self.model_tf(NaCl, x1)

    def local_min(self, x):
        """Вычисление локального минимума методом Флетчера-Ривза"""
        Xcur = np.zeros(self.n)  # Yk
        Xnew = np.zeros(self.n)  # Yk + 1
        S = np.zeros(self.n)
        NaCl = x[0]
        HCl = x[1]
        NaOH = x[2]
        DBL = x[3]
        Xcur[0] = HCl
        Xcur[1] = NaOH
        Xcur[2] = DBL
        # iteration0 (just like classic drop)
        grad = self.grad_f(NaCl, Xcur)
        # saving grad norm for beta
        tmpNorm = self.norma_grad(grad)
        # So = antigrad(F(Xo))
        for i in range(0, self.n):
            S[i] = -grad[i]
        # alfa?
        l = 0
        r = 1
        count = 0
        while abs(l - r) > self.eps4 and count < self.MAX_NUM_OF_ITERS:
            alfa = (l + r) / 2
            if self.g(alfa - self.deltaBig, NaCl, Xcur, S) < self.g(alfa + self.deltaBig, NaCl, Xcur, S):
                r = alfa
            else:
                l = alfa
            count += 1
        alfa = (l + r) / 2
        # X1 = Xo + alfa * So
        for i in range(0, self.n):
            Xnew[i] = Xcur[i] + alfa * S[i]
        grad = self.grad_f(NaCl, Xnew)
        count = 0
        # main cycle
        while (self.norma_grad(grad) > self.eps1 and
               self.norma(Xnew, Xcur) > self.eps2 and
               abs(self.model_tf(NaCl, Xnew) - self.model_tf(NaCl, Xcur)) > self.eps3 and
               count < self.MAX_NUM_OF_ITERS):
            for i in range(0, self.n):
                Xcur[i] = Xnew[i]
            # betak=norm(grad(F(Xk))) / norm(grad(F(Xk-1)))
            beta = pow(self.norma_grad(grad), 2) / pow(tmpNorm, 2)
            tmpNorm = self.norma_grad(grad)
            # Sk=-grad_f(Xk)+betak * Sk-1
            for i in range(0, self.n):
                S[i] = -grad[i] + beta * S[i]
            # alfa?
            l = 0
            r = 1
            inner_count = 0
            while abs(l - r) > self.eps4 and inner_count < self.MAX_NUM_OF_ITERS:
                alfa = (l + r) / 2
                if self.g(alfa - self.deltaBig, NaCl, Xcur, S) < self.g(alfa + self.deltaBig, NaCl, Xcur, S):
                    r = alfa
                else:
                    l = alfa
                inner_count += 1
            alfa = (l + r) / 2
            # Xk + 1 = Xk + alfa * Sk
            for i in range(0, self.n):
                Xnew[i] = Xcur[i] + alfa * S[i]
            self.eval_params_3d(Xnew)
            grad = self.grad_f(NaCl, Xnew)
            count += 1
        x[0] = NaCl
        x[1] = Xnew[0]
        x[2] = Xnew[1]
        x[3] = Xnew[2]
        y = self.model(x)
        return y, x

    def optimize(self):
        """Запуск моделирования с использованием введенных данных"""
        x = np.zeros(4)
        xMin = np.zeros(4)
        yMin = np.zeros(4)
        nodes = 4
        x[0] = self.cNaCl
        x[1] = self.cHClMin
        x[2] = self.cNaOHMin
        x[3] = self.DBLMin
        y, x = self.local_min(x)
        xMin[0] = x[0]
        xMin[1] = x[1]
        xMin[2] = x[2]
        xMin[3] = x[3]
        yMin[0] = y[0]
        yMin[1] = y[1]
        yMin[2] = y[2]
        yMin[3] = y[3]
        for l in range(0, nodes):
            for m in range(0, nodes):
                for d in tqdm(range(0, nodes)):
                    x[1] = self.cHClMin + l * (self.cHClMax - self.cHClMin) / (nodes + 1)
                    x[2] = self.cNaOHMin + m * (self.cNaOHMax - self.cNaOHMin) / (nodes + 1)
                    x[3] = self.DBLMin + d * (self.DBLMax - self.DBLMin) / (nodes + 1)
                    y, x = self.local_min(x)
                    if self.tf(y) < self.tf(yMin):
                        xMin[1] = x[1]
                        xMin[2] = x[2]
                        xMin[3] = x[3]
                        yMin[0] = y[0]
                        yMin[1] = y[1]
                        yMin[2] = y[2]
                        yMin[3] = y[3]
        return yMin, xMin
