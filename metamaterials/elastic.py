"""Mandel tensors, isotropic projection and solid-void Hashin-Shtrikman bounds."""
import numpy as np
SQ2 = np.sqrt(2.0)
V_H = np.array([1, 1, 1, 0, 0, 0], dtype=float) / np.sqrt(3.0)
LAM_H = np.outer(V_H, V_H)
LAM_S = np.eye(6) - LAM_H
NUS = (0.3, 0.45)
THETAS = (0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7)

def solid_moduli(nu, E=1.0):
    return (E / (3 * (1 - 2 * nu)), E / (2 * (1 + nu)))

def iso_tensor(kappa, mu):
    return 3 * kappa * LAM_H + 2 * mu * LAM_S

def iso_project(C):
    C = np.asarray(C, dtype=float)
    h = V_H @ C @ V_H
    return (h / 3.0, (np.trace(C) - h) / 10.0)

def iso_defect(C):
    k, m = iso_project(C)
    return np.linalg.norm(C - iso_tensor(k, m)) / np.linalg.norm(C)

def hs_upper(theta, ks, ms):
    zk = 4 * ms / 3
    zm = ms * (9 * ks + 8 * ms) / (6 * (ks + 2 * ms))
    k = 1 / (theta / (ks + zk) + (1 - theta) / zk) - zk
    m = 1 / (theta / (ms + zm) + (1 - theta) / zm) - zm
    return (k, m)

def normalized(kappa, mu, nu, theta):
    ks, ms = solid_moduli(nu)
    kh, mh = hs_upper(theta, ks, ms)
    return (kappa / kh, mu / mh)

def voigt_to_mandel(Cv):
    W = np.diag([1, 1, 1, SQ2, SQ2, SQ2])
    return W @ np.asarray(Cv, float) @ W

def mandel_to_voigt(Cm):
    W = np.diag([1, 1, 1, 1 / SQ2, 1 / SQ2, 1 / SQ2])
    return W @ np.asarray(Cm, float) @ W

def sym_to_mandel(s):
    s = np.asarray(s, float)
    return np.array([s[0, 0], s[1, 1], s[2, 2], SQ2 * s[1, 2], SQ2 * s[0, 2], SQ2 * s[0, 1]])

def mandel_to_sym(v):
    v = np.asarray(v, float)
    a = 1 / SQ2
    return np.array([[v[0], a * v[5], a * v[4]], [a * v[5], v[1], a * v[3]], [a * v[4], a * v[3], v[2]]])

def rotate_mandel(C, R):
    basis = [mandel_to_sym(e) for e in np.eye(6)]
    Q = np.array([[np.tensordot(R @ basis[j] @ R.T, basis[i], 2) for j in range(6)] for i in range(6)])
    return Q @ C @ Q.T

def cubic_tensor(C11, C12, C44):
    C = np.zeros((6, 6))
    C[:3, :3] = C12
    np.fill_diagonal(C[:3, :3], C11)
    C[3, 3] = C[4, 4] = C[5, 5] = 2 * C44
    return C
