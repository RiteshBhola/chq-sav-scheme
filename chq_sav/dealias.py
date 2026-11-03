import numpy as np


def dealias_filter(grid, u, keep_ratio=2.0 / 3.0):
    """Standard 2/3-rule spectral truncation: zero the outer (1-keep_ratio)
    fraction of wavenumbers in each direction."""
    N = u.shape[0]
    kcut = int(np.floor(keep_ratio * (N // 2)))
    idx = np.fft.fftfreq(N, d=1.0 / N)
    keep1d = np.abs(idx) <= kcut
    mask = np.outer(keep1d, keep1d)
    return np.real(np.fft.ifft2(mask * np.fft.fft2(u)))
