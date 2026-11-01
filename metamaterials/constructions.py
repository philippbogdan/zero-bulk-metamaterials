"""The HS laminate and the limiting slip-wall construction."""
import numpy as np
import mpmath as mp
from .elastic import solid_moduli, iso_tensor, iso_project, iso_defect, hs_upper
from .laminate import sequential_void_core, icosahedral_axes, mp_laminate, to_mp, frame_to_e3

def hs_corner(nu, theta):
    ks, ms = solid_moduli(nu)
    return sequential_void_core(iso_tensor(ks, ms), theta, icosahedral_axes(), np.ones(6) / 6)

def fibre_wall_mp(Cs, m, tw, delta):
    m = np.asarray(m, float)
    m = m / np.linalg.norm(m)
    a = frame_to_e3(m)[0]
    b = np.cross(m, a)
    p = np.sqrt(tw)
    L1 = mp_laminate([Cs, 0 * Cs], [p, 1 - p], a, delta)
    return mp_laminate([L1, mp.zeros(6, 6)], [p, 1 - p], b, delta)

def slipwall_corner(nu, theta, eps, tw=0.5, dps=60, delta='1e-40'):
    mp.mp.dps = dps
    delta = mp.mpf(delta)
    ks, ms = solid_moduli(nu)
    Cs = iso_tensor(ks, ms)
    s = (1 - eps) ** 2 + (1 - eps) + 1
    thB = (theta - eps * tw * s) / (1 - eps) ** 3
    C = to_mp(sequential_void_core(Cs, thB, icosahedral_axes(), np.ones(6) / 6))
    for m in ([1, 0, 0], [0, 1, 0], [1, 1, 1]):
        W = fibre_wall_mp(Cs, m, tw, delta)
        C = mp_laminate([C, W], [1 - eps, eps], m, delta)
    return (np.array(C.tolist(), dtype=float), thB)

def xy_of(C, nu, theta):
    ks, ms = solid_moduli(nu)
    k, m = iso_project(C)
    kh, mh = hs_upper(theta, ks, ms)
    return (k / kh, m / mh, iso_defect(C) if np.linalg.norm(C) > 0 else 0.0)
