from tensorflow.keras.utils import Sequence

import numpy as np
from numpy.linalg import norm


def splitRealImagNewAxis(mat):
    return mat[..., np.newaxis].view(np.float64)


def splitRealImagVerticalConcat(mat):
    r, c = mat.shape
    return splitRealImagNewAxis(mat).transpose((2, 0, 1)).reshape((2 * r, c))


def create_complex_noise(size, power, generator: np.random.Generator):
    m, n = size
    noise = generator.standard_normal((m, n, 2)).view(dtype="complex128").squeeze()
    noise *= np.sqrt(power / (m * n * 2))
    return noise


def mimo_simulation(H, X, snr_db, quantizer, generator: np.random.Generator):
    Y = H @ X

    snr_linear = 10 ** (0.1 * snr_db)
    power_signal = norm(Y, "fro")
    power_noise = power_signal / snr_linear

    Nr = H.shape[0]
    T = X.shape[1]

    noise = create_complex_noise((Nr, T), power_noise, generator)
    Y += noise
    Y = quantizer(Y)

    return Y, noise


class XYGenerator(Sequence):
    def __init__(
        self,
        num_batches=100,
        batch_size=32,
        constant_batches=False,
        should_include_x=False,
        x_generator=None,
        channel_generator=None,
        snr_generator=None,
        quantizer=None,
        generator: np.random.Generator = None,
    ):
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.x_generator = x_generator
        self.channel_generator = channel_generator
        self.snr_generator = snr_generator
        self.quantizer = quantizer
        self.constant_batches = constant_batches
        self.should_include_x = should_include_x
        self.generator = generator

        if self.constant_batches:
            self.make_batches()

    def channel_simulation(self):
        H, Hv = next(self.channel_generator)
        X = next(self.x_generator)
        snr_db = next(self.snr_generator)
        Y, dummy_noise = mimo_simulation(H, X, snr_db, self.quantizer, self.generator)

        return tuple(splitRealImagVerticalConcat(mat) for mat in [X, Y, Hv])

    def batch(self):
        sims = [self.channel_simulation() for _ in range(self.batch_size)]

        sims = [
            (np.vstack((X, Y) if self.should_include_x else (Y)).transpose(), H)
            for X, Y, H in sims
        ]

        return tuple(np.stack(d) for d in zip(*sims))

    def make_batches(self):
        self.data = [self.batch() for _ in range(self.num_batches)]

    def set_snr_generator(self, snr_generator):
        self.snr_generator = snr_generator

    def __getitem__(self, index):
        if self.constant_batches:
            return self.data[index]
        return self.batch()

    def __len__(self):
        return self.num_batches


if __name__ == "__main__":

    from channel_generator import random_channel
    from pilot_generator import constant_random_pilot
    from snr_generator import uniform_snr

    Nr, Nt, T = 8, 8, 256

    xy_generator = XYGenerator(
        num_batches=5,
        batch_size=10,
        x_generator=constant_random_pilot(Nt, T),
        channel_generator=lambda: random_channel(Nr, Nt),
        snr_generator=uniform_snr(30, 33),
        quantizer=np.sign,
    )

    input_tensor, output_tensor = xy_generator[4]

    print(len(xy_generator[:]))
    print(input_tensor.shape)
    print(output_tensor.shape)
