"""Binary voxel geometry for the finite-scale wall experiments."""
import numpy as np


def wall_section(n, p=8, p2=16, p3=32, level='H', sign=1):
    """Return the exact x1 by x3 section, with no interpolation."""
    periods = [p] + ([p2] if level in ('A', 'H') else []) + ([p3] if level == 'H' else [])
    if level not in ('plates', 'A', 'H') or any(v < 4 or v % 2 or n % v for v in periods):
        raise ValueError('Use even periods of at least four, each dividing the cell size.')
    i, k = np.arange(n)[:, None], np.arange(n)[None, :]
    plus = (i + sign * k) % p < p // 2
    if level == 'plates':
        return plus
    plus = plus | (k % p2 < p2 // 2)
    if level == 'A':
        return plus
    minus = ((i - sign * k) % p < p // 2) | (k % p2 < p2 // 2)
    return np.where(i % p3 < p3 // 2, plus, minus)


def extrude(section):
    n = section.shape[0]
    return np.broadcast_to(section[:, None, :], (n, n, n)).copy()
