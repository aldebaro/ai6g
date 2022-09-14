from numpy import ndarray
from numpy.fft import fft, ifft


def virtual(H: ndarray) -> ndarray:
    """ Convert channel from antenna domain to virtual domain """
    return ifft(fft(H, axis=-2, norm="ortho"), axis=-1, norm="ortho")


def antenna(Hv: ndarray) -> ndarray:
    """ Convert channel from virtual domain to antenna domain """
    return fft(ifft(Hv, axis=-2, norm="ortho"), axis=-1, norm="ortho")
