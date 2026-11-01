"""The fixed direction set used by the stored sequential-laminate bases."""
import numpy as np
from .laminate import icosahedral_axes


def seq_dirs():
    """Six icosahedral, three cubic and four octahedral axes, in stored order."""
    octahedral = np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1]], float) / np.sqrt(3)
    return np.vstack([icosahedral_axes(), np.eye(3), octahedral])
