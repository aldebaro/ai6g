import abc
import numpy as np
import numpy.linalg as la


def normalize(H: np.ndarray) -> np.ndarray:
    """Normalize channel using Frobenius norm"""

    Nr, Nt = H.shape
    return np.sqrt(Nr * Nt) / la.norm(H, "fro") * H


class ChannelGenerator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __next__(self):
        pass

    @abc.abstractmethod
    def shape(self):
        pass
