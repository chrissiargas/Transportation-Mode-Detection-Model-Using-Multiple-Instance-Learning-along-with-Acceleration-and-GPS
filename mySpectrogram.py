from scipy.signal import welch
import numpy as np


class LogBands:
    def __init__(self, n: int, base: float = 2):
        """
        :param n: Length of signal (i.e., number of frequencies)
        :param base: Base of bands
        """
        X = []
        m2 = n
        m1 = int(np.ceil(m2 / base))
        while m2 != 1:
            x = np.zeros((n,))
            x[m1:m2] = 1
            m2 = m1
            m1 = int(np.ceil(m2 / base))
            X.append(x)
        x = np.zeros((n,))
        x[0] = 1
        X.append(x)
        X = np.array(X)
        self._x = X

    def get_axis(self, v):
        return np.matmul(self._x, v) / np.sum(self._x, axis=1)

    def apply(self, x: np.ndarray):
        return np.matmul(self._x, x)


def my_tvs(x, wsize, wstep, nfft):
    start_idx = np.arange(0, len(x) - wsize, wstep)
    m = nfft // 2 + 1
    s = np.zeros((m, len(start_idx)))
    for i, idx in enumerate(start_idx):
        w, s[:, i] = welch(x[idx:idx + wsize], nperseg=nfft, nfft=nfft)

    return start_idx + wsize // 2, w, s


def my_tvs2(x, wsize, num_of_windows, nfft):
    wstep = (len(x) - wsize) // (num_of_windows - 1)
    t, w, Pxx = my_tvs(x, wsize, wstep, nfft)

    return t[:num_of_windows], w, Pxx[:, :num_of_windows]

