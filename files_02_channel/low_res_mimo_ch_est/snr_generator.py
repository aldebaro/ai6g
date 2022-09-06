import numpy as np


class Constant:
    def __init__(self, snr_db) -> None:
        self.snr_db = snr_db

    def __iter__(self):
        return self

    def __next__(self):
        return self.snr_db


def constant_snr(snr_db):
    return Constant(snr_db)


class Uniform:
    def __init__(self, low, high, rng: np.random.Generator) -> None:
        self.low = low
        self.high = high
        self.rng = rng

    def __iter__(self):
        return self

    def __next__(self):
        return self.rng.uniform(self.low, self.high)


def uniform_snr(low, high, rng: np.random.Generator):
    return Uniform(low, high, rng)


if __name__ == "__main__":
    u = uniform_snr(0, 10, np.random.default_rng())
    print(next(u))
    print(next(u))
    print(next(u))
    print(next(u))
