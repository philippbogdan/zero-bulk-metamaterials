"""Differentiable float64 algebra for the optional design search."""
import math
import torch
T = torch.float64
S2 = math.sqrt(2.0)
_IDX = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
P_IDX = [0, 1, 5]
N_IDX = [2, 3, 4]
PERM = P_IDX + N_IDX

def _basis():
    B = torch.zeros(6, 3, 3, dtype=T)
    for a, (i, j) in enumerate(_IDX):
        if i == j:
            B[a, i, j] = 1.0
        else:
            B[a, i, j] = B[a, j, i] = 1 / S2
    return B
BASIS = _basis()

def rot6(R):
    RE = torch.einsum('...ik,bkl,...jl->...bij', R, BASIS.to(R), R)
    return torch.einsum('aij,...bij->...ab', BASIS.to(R), RE)

def frame(n):
    n = n / n.norm(dim=-1, keepdim=True)
    ex = torch.tensor([1.0, 0, 0], dtype=n.dtype).expand_as(n)
    ey = torch.tensor([0, 1.0, 0], dtype=n.dtype).expand_as(n)
    a = torch.where(n[..., :1].abs() < 0.9, ex, ey)
    t1 = a - (a * n).sum(-1, keepdim=True) * n
    t1 = t1 / t1.norm(dim=-1, keepdim=True)
    t2 = torch.cross(n, t1, dim=-1)
    return torch.stack([t1, t2, n], dim=-2)

def Hmap(C):
    C = C[..., PERM, :][..., :, PERM]
    A, B, D, F = (C[..., :3, :3], C[..., :3, 3:], C[..., 3:, :3], C[..., 3:, 3:])
    Fi = torch.linalg.inv(F)
    top = torch.cat([A - B @ Fi @ D, B @ Fi], -1)
    bot = torch.cat([-Fi @ D, Fi], -1)
    return torch.cat([top, bot], -2)

def Hinv(H):
    A, B, D, F = (H[..., :3, :3], H[..., :3, 3:], H[..., 3:, :3], H[..., 3:, 3:])
    Fi = torch.linalg.inv(F)
    top = torch.cat([A - B @ Fi @ D, B @ Fi], -1)
    bot = torch.cat([-Fi @ D, Fi], -1)
    Cp = torch.cat([top, bot], -2)
    inv = [PERM.index(i) for i in range(6)]
    Cp = Cp[..., inv, :][..., :, inv]
    return 0.5 * (Cp + Cp.transpose(-1, -2))

def laminate(Cs, fs, n, delta=0.0):
    R = frame(n)
    Q = rot6(R)
    I6 = torch.eye(6, dtype=T)
    H = 0
    for C, f in zip(Cs, fs):
        Cr = Q @ (C + delta * I6) @ Q.transpose(-1, -2)
        H = H + f[..., None, None] * Hmap(Cr)
    Cr = Hinv(H)
    return Q.transpose(-1, -2) @ Cr @ Q
V_H = torch.tensor([1, 1, 1, 0, 0, 0], dtype=T) / math.sqrt(3)

def iso(C):
    h = torch.einsum('i,...ij,j->...', V_H, C, V_H)
    k = h / 3
    m = (torch.diagonal(C, dim1=-2, dim2=-1).sum(-1) - h) / 10
    LH = torch.outer(V_H, V_H)
    Ci = 3 * k[..., None, None] * LH + 2 * m[..., None, None] * (torch.eye(6, dtype=T) - LH)
    d = (C - Ci).flatten(-2).norm(dim=-1) / C.flatten(-2).norm(dim=-1)
    return (k, m, d)

def iso_tensor(k, m):
    LH = torch.outer(V_H, V_H)
    k = torch.as_tensor(k, dtype=T)
    m = torch.as_tensor(m, dtype=T)
    return 3 * k[..., None, None] * LH + 2 * m[..., None, None] * (torch.eye(6, dtype=T) - LH)

def nn6(n):
    n = n / n.norm(dim=-1, keepdim=True)
    M = n[..., :, None] * n[..., None, :]
    return torch.einsum('aij,...ij->...a', BASIS, M)

def free_normal(C, n, eta=0.0):
    v = nn6(n)
    Cv = torch.einsum('...ij,...j->...i', C, v)
    vCv = (v * Cv).sum(-1)
    return C - (1 - eta) * Cv[..., :, None] * Cv[..., None, :] / vCv[..., None, None]

def sym6(n, p):
    n = n / n.norm(dim=-1, keepdim=True)
    p = p / p.norm(dim=-1, keepdim=True)
    M = 0.5 * (n[..., :, None] * p[..., None, :] + p[..., :, None] * n[..., None, :])
    return torch.einsum('aij,...ij->...a', BASIS, M)

def interface_crystal(C, n, p, eta=0.0):
    v = sym6(n, p)
    Cv = torch.einsum('...ij,...j->...i', C, v)
    vCv = (v * Cv).sum(-1)
    return C - (1 - eta) * Cv[..., :, None] * Cv[..., None, :] / vCv[..., None, None]
