import numpy as np


def __random_qpsk__(Nt, T, generator: np.random.Generator):
    real_imag = generator.integers(0, 2, size=(Nt, T, 2)) - 1.0
    return real_imag.view(np.complex128).squeeze()


class QpskPilotGenerator:
    def __init__(self, Nt, T, generator: np.random.Generator, is_constant=True):
        self.Nt = Nt
        self.T = T
        self.generator = generator
        self.is_constant = is_constant
        if is_constant:
            self.X = __random_qpsk__(Nt, T, self.generator)

    def __next__(self):
        if self.is_constant:
            return self.X
        return __random_qpsk__(self.Nt, self.T, self.generator)
