"""Exact partial-inverse lamination and void-core sequential laminates."""
import numpy as np
import mpmath as mp
from .elastic import mandel_to_sym, sym_to_mandel, rotate_mandel, SQ2
P_IDX = [0, 1, 5]
N_IDX = [2, 3, 4]

def frame_to_e3(n):
    n = np.asarray(n, float) / np.linalg.norm(n)
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    t1 = a - a @ n * n
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    return np.array([t1, t2, n])

def gamma_op(n, C):
    n = np.asarray(n, float) / np.linalg.norm(n)
    basis = [mandel_to_sym(e) for e in np.eye(6)]
    K = np.zeros((3, 3))
    for i in range(3):
        for k in range(3):
            Ei = np.zeros((3, 3))
            Ei[i, :] = n
            Ei = 0.5 * (Ei + Ei.T)
            Ek = np.zeros((3, 3))
            Ek[k, :] = n
            Ek = 0.5 * (Ek + Ek.T)
            K[i, k] = sym_to_mandel(Ei) @ C @ sym_to_mandel(Ek)
    Ki = np.linalg.inv(K)
    G = np.zeros((6, 6))
    for a in range(6):
        v = Ki @ (basis[a] @ n)
        M = 0.5 * (np.outer(n, v) + np.outer(v, n))
        G[:, a] = sym_to_mandel(M)
    return 0.5 * (G + G.T)

def sequential_void_core(Cs, theta, dirs, weights):
    M = sum((w * gamma_op(n, Cs) for n, w in zip(dirs, weights)))
    return Cs + (1 - theta) * np.linalg.inv(theta * M - np.linalg.inv(Cs))

def icosahedral_axes():
    g = (1 + np.sqrt(5)) / 2
    v = np.array([[0, 1, g], [0, 1, -g], [1, g, 0], [1, -g, 0], [g, 0, 1], [-g, 0, 1]], float)
    return v / np.linalg.norm(v, axis=1)[:, None]

def _H(C):
    Cpp = C[np.ix_(P_IDX, P_IDX)]
    Cpn = C[np.ix_(P_IDX, N_IDX)]
    Cnp = C[np.ix_(N_IDX, P_IDX)]
    Cnn = C[np.ix_(N_IDX, N_IDX)]
    Ni = np.linalg.inv(Cnn)
    return np.block([[Cpp - Cpn @ Ni @ Cnp, Cpn @ Ni], [-Ni @ Cnp, Ni]])

def _Hinv(H):
    A = H[:3, :3]
    B = H[:3, 3:]
    D = H[3:, :3]
    F = H[3:, 3:]
    Fi = np.linalg.inv(F)
    C = np.zeros((6, 6))
    C[np.ix_(N_IDX, N_IDX)] = Fi
    C[np.ix_(N_IDX, P_IDX)] = -Fi @ D
    C[np.ix_(P_IDX, N_IDX)] = B @ Fi
    C[np.ix_(P_IDX, P_IDX)] = A - B @ Fi @ D
    return 0.5 * (C + C.T)

def laminate(Cs_list, fracs, n, delta=0.0):
    R = frame_to_e3(n)
    Hs = 0
    for C, f in zip(Cs_list, fracs):
        Cr = rotate_mandel(np.asarray(C, float) + delta * np.eye(6), R)
        Hs = Hs + f * _H(Cr)
    return rotate_mandel(_Hinv(Hs), R.T)

def to_mp(A):
    return mp.matrix([[mp.mpf(x) if not isinstance(x, mp.mpf) else x for x in row] for row in np.asarray(A).tolist()]) if not isinstance(A, mp.matrix) else A

def mp_rot6(R):
    s2 = mp.sqrt(2)
    idx = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    w = [1, 1, 1, s2, s2, s2]
    Q = mp.matrix(6, 6)
    for a, (i, j) in enumerate(idx):
        for b, (k, l) in enumerate(idx):
            if b < 3:
                val = R[i, k] * R[j, l]
            else:
                val = R[i, k] * R[j, l] + R[i, l] * R[j, k]
            Q[a, b] = val * w[a] / (w[b] if b >= 3 else 1)
    return Q

def mp_frame_to_e3(n):
    n = [mp.mpf(x) for x in n]
    nn = mp.sqrt(sum((x * x for x in n)))
    n = [x / nn for x in n]
    a = [mp.mpf(1), 0, 0] if abs(n[0]) < 0.9 else [0, mp.mpf(1), 0]
    d = sum((a[i] * n[i] for i in range(3)))
    t1 = [a[i] - d * n[i] for i in range(3)]
    t1n = mp.sqrt(sum((x * x for x in t1)))
    t1 = [x / t1n for x in t1]
    t2 = [n[1] * t1[2] - n[2] * t1[1], n[2] * t1[0] - n[0] * t1[2], n[0] * t1[1] - n[1] * t1[0]]
    return mp.matrix([t1, t2, n])

def _sub(M, rows, cols):
    S = mp.matrix(len(rows), len(cols))
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            S[i, j] = M[r, c]
    return S

def mp_laminate(Cs_list, fracs, n, delta):
    R = mp_frame_to_e3(n)
    Q = mp_rot6(R)
    Hs = mp.matrix(6, 6)
    for C, f in zip(Cs_list, fracs):
        C = to_mp(C) + delta * mp.eye(6)
        Cr = Q * C * Q.T
        Cpp = _sub(Cr, P_IDX, P_IDX)
        Cpn = _sub(Cr, P_IDX, N_IDX)
        Cnp = _sub(Cr, N_IDX, P_IDX)
        Cnn = _sub(Cr, N_IDX, N_IDX)
        Ni = Cnn ** (-1)
        H = mp.matrix(6, 6)
        blocks = [Cpp - Cpn * Ni * Cnp, Cpn * Ni, -(Ni * Cnp), Ni]
        for (r0, c0), Bk in zip([(0, 0), (0, 3), (3, 0), (3, 3)], blocks):
            for i in range(3):
                for j in range(3):
                    H[r0 + i, c0 + j] = Bk[i, j]
        Hs += mp.mpf(f) * H
    A = _sub(Hs, [0, 1, 2], [0, 1, 2])
    B = _sub(Hs, [0, 1, 2], [3, 4, 5])
    D = _sub(Hs, [3, 4, 5], [0, 1, 2])
    F = _sub(Hs, [3, 4, 5], [3, 4, 5])
    Fi = F ** (-1)
    Cr = mp.matrix(6, 6)
    parts = {(tuple(N_IDX), tuple(N_IDX)): Fi, (tuple(N_IDX), tuple(P_IDX)): -(Fi * D), (tuple(P_IDX), tuple(N_IDX)): B * Fi, (tuple(P_IDX), tuple(P_IDX)): A - B * Fi * D}
    for (rows, cols), Bk in parts.items():
        for i, r in enumerate(rows):
            for j, c in enumerate(cols):
                Cr[r, c] = Bk[i, j]
    Cout = Q.T * Cr * Q
    return (Cout + Cout.T) / 2
