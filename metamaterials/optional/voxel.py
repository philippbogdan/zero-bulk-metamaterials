"""Periodic Q1 voxel elasticity with a matrix-free PyTorch solver.

Solid voxels carry stiffness; void voxels carry none. Periodic floating
islands contribute zero energy and are removed. The returned tensor uses
the symmetric energy form. See docs/finite-scale.md for the discretisation.
"""
from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn.functional as F
from ..elastic import mandel_to_sym
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
_OFFS = np.array([[a, b, c] for c in (0, 1) for b in (0, 1) for a in (0, 1)])

def isotropic_voigt(E, nu):
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[np.arange(3), np.arange(3)] = lam + 2 * mu
    D[3, 3] = D[4, 4] = D[5, 5] = mu
    return D

def _bmat(xi):
    B = np.zeros((6, 24))
    for n, (a, b, c) in enumerate(_OFFS):
        fx = xi[0] if a else 1 - xi[0]
        fy = xi[1] if b else 1 - xi[1]
        fz = xi[2] if c else 1 - xi[2]
        sx, sy, sz = (1 if a else -1, 1 if b else -1, 1 if c else -1)
        dN = np.array([sx * fy * fz, fx * sy * fz, fx * fy * sz])
        i = 3 * n
        B[0, i] = dN[0]
        B[1, i + 1] = dN[1]
        B[2, i + 2] = dN[2]
        B[3, i + 1], B[3, i + 2] = (dN[2], dN[1])
        B[4, i], B[4, i + 2] = (dN[2], dN[0])
        B[5, i], B[5, i + 1] = (dN[1], dN[0])
    return B

def hex8_stiffness(E=1.0, nu=0.3):
    D = isotropic_voigt(E, nu)
    g = np.array([0.5 - 0.5 / math.sqrt(3), 0.5 + 0.5 / math.sqrt(3)])
    K = np.zeros((24, 24))
    for x in g:
        for y in g:
            for z in g:
                B = _bmat((x, y, z))
                K += B.T @ D @ B / 8.0
    return 0.5 * (K + K.T)

def macro_displacements():
    U = np.zeros((24, 6))
    for k in range(6):
        Ek = mandel_to_sym(np.eye(6)[k])
        for n, o in enumerate(_OFFS):
            U[3 * n:3 * n + 3, k] = Ek @ o
    return U

def child_prolongations():
    P = np.zeros((8, 24, 24))
    for ci, c in enumerate(_OFFS):
        for n, a in enumerate(_OFFS):
            t = (c + a) / 2.0
            for m, A in enumerate(_OFFS):
                w = np.prod(np.where(A == 1, t, 1 - t))
                for d in range(3):
                    P[ci, 3 * n + d, 3 * m + d] = w
    return P

def _gather_weight():
    W = np.zeros((24, 3, 2, 2, 2))
    for n, (a, b, c) in enumerate(_OFFS):
        for d in range(3):
            W[3 * n + d, d, a, b, c] = 1.0
    return W

def _scatter_weight():
    W = np.zeros((3, 24, 2, 2, 2))
    for n, (a, b, c) in enumerate(_OFFS):
        for d in range(3):
            W[d, 3 * n + d, 1 - a, 1 - b, 1 - c] = 1.0
    return W

def periodic_components(solid):
    from scipy import ndimage
    solid = np.asarray(solid, bool)
    N = solid.shape
    lab, M = ndimage.label(solid, structure=np.ones((3, 3, 3), int))
    lab = lab.astype(np.int64)
    edges = []
    n0 = N[0]
    assert N[0] == N[1] == N[2]
    for k in range(3):
        A = np.moveaxis(lab, k, 0)
        hi, lo = (A[-1], A[0])
        o1, o2 = [j for j in range(3) if j != k]
        ii = np.arange(n0)
        for d1 in (-1, 0, 1):
            for d2 in (-1, 0, 1):
                nb = np.roll(lo, shift=(-d1, -d2), axis=(0, 1))
                sel = (hi > 0) & (nb > 0)
                if not sel.any():
                    continue
                i, j = np.nonzero(sel)
                s = np.zeros((len(i), 3), np.int64)
                s[:, k] = 1
                s[:, o1] = np.where(i + d1 >= n0, 1, np.where(i + d1 < 0, -1, 0))
                s[:, o2] = np.where(j + d2 >= n0, 1, np.where(j + d2 < 0, -1, 0))
                edges.append(np.concatenate([hi[i, j][:, None], nb[i, j][:, None], s], 1))
    parent = np.arange(M + 1)
    pot = np.zeros((M + 1, 3), np.int64)
    cycles = {}

    def find(x):
        path = []
        while parent[x] != x:
            path.append(x)
            x = parent[x]
        root = x
        acc = np.zeros(3, np.int64)
        for y in reversed(path):
            acc = acc + pot[y]
            pot[y] = acc.copy()
            parent[y] = root
        return root
    if edges:
        E = np.unique(np.concatenate(edges, 0), axis=0)
        for l1, l2, *s in E:
            s = np.array(s, np.int64)
            r1, r2 = (find(l1), find(l2))
            p1 = pot[l1] if l1 != r1 else np.zeros(3, np.int64)
            p2 = pot[l2] if l2 != r2 else np.zeros(3, np.int64)
            if r1 == r2:
                cyc = p1 + s - p2
                if np.any(cyc):
                    cycles.setdefault(r1, []).append(cyc)
            else:
                parent[r2] = r1
                pot[r2] = p1 + s - p2
                if r2 in cycles:
                    cycles.setdefault(r1, []).extend(cycles.pop(r2))
    roots = np.array([find(l) for l in range(M + 1)])
    uniq, inv = np.unique(roots[1:], return_inverse=True)
    newlab = np.concatenate([[0], inv + 1])
    K = len(uniq)
    ranks = np.zeros(K + 1, int)
    axes = [None] * (K + 1)
    for i, r in enumerate(uniq):
        cyc = cycles.get(r, [])
        if cyc:
            Cm = np.array(cyc, float)
            ranks[i + 1] = np.linalg.matrix_rank(Cm)
            if ranks[i + 1] == 1:
                v = np.array(cyc[0], np.int64)
                axes[i + 1] = v // np.gcd.reduce(np.abs(v[v != 0]))
    return (newlab[lab].astype(np.int32), ranks, axes)

def _wrap_hi_yz(t):
    t = torch.cat([t, t[..., :1]], -1)
    return torch.cat([t, t[..., :1, :]], -2)

def _wrap_lo_yz(t):
    t = torch.cat([t[..., -1:], t], -1)
    return torch.cat([t[..., -1:, :], t], -2)

class ElemOp:

    def __init__(self, n, device, dtype, groups=None, rho=None, KEc=None, table=None, idx=None, work_bytes=1000000000.0):
        self.n, self.device, self.dtype = (n, torch.device(device), dtype)
        self.work_bytes = work_bytes
        self.Wg = torch.tensor(_gather_weight(), device=device, dtype=dtype)
        self.Ws = torch.tensor(_scatter_weight(), device=device, dtype=dtype)
        n3 = n ** 3
        if KEc is not None:
            self.kind = 'stored'
            self.KEc = KEc.to(device=device, dtype=dtype).reshape(576, n3).contiguous()
            self.diagE = self.KEc[torch.arange(24, device=device) * 25].reshape(1, 24, n, n, n)
        elif table is not None:
            self.kind = 'table'
            self.T = torch.as_tensor(table).to(device=device, dtype=dtype).reshape(-1, 576).contiguous()
            self.idx = idx.to(device=device, dtype=torch.int32).reshape(n3).contiguous()
            dT = self.T[:, torch.arange(24, device=device) * 25]
            self.diagE = dT[self.idx.long()].T.reshape(1, 24, n, n, n)
        else:
            Kg = torch.as_tensor(np.asarray(groups), dtype=torch.float64)
            self.G = Kg.shape[0]
            self.kind = 'scaled' if self.G == 1 else 'groups'
            self.Kg_t = Kg.to(device=device, dtype=dtype)
            Wg64 = torch.as_tensor(_gather_weight())
            W = torch.einsum('gij,jdabc->gidabc', Kg, Wg64).reshape(self.G * 24, 3, 2, 2, 2)
            self.W1 = W.to(device=device, dtype=dtype)
            self.rho = rho.to(device=device, dtype=dtype).reshape(self.G, n, n, n).contiguous()
            dg = torch.diagonal(Kg, dim1=1, dim2=2).to(device=device, dtype=dtype)
            self.diagE = torch.einsum('gi,gxyz->ixyz', dg, self.rho)[None]
        self._diag = None

    @property
    def label(self):
        return self.kind + (str(self.G) if self.kind == 'groups' else '')

    def _slab(self, B, ch):
        per = B * ch * self.n * self.n * torch.finfo(self.dtype).bits // 8 * 3
        return int(max(1, min(self.n, self.work_bytes // max(per, 1))))

    def _scatter_into(self, f, fE, x0):
        s = fE.shape[2]
        P = F.pad(_wrap_lo_yz(fE), (0, 0, 0, 0, 1, 1))
        o = F.conv3d(P, self.Ws)
        idx = torch.arange(x0, x0 + s + 1, device=f.device) % self.n
        f.index_add_(2, idx, o)

    def _gather(self, u, x0, s, W):
        idx = torch.arange(x0, x0 + s + 1, device=u.device) % self.n
        return F.conv3d(_wrap_hi_yz(u.index_select(2, idx)), W)

    def _elem_chunk(self, e0, e1):
        if self.kind == 'stored':
            return self.KEc[:, e0:e1].T.reshape(-1, 24, 24)
        return self.T[self.idx[e0:e1].long()].reshape(-1, 24, 24)

    def apply(self, u):
        return self._apply_torch(u)

    def _apply_torch(self, u):
        B, n = (u.shape[0], self.n)
        f = torch.zeros_like(u)
        if self.kind in ('scaled', 'groups'):
            s = self._slab(B, 24 * self.G + 24)
            for x0 in range(0, n, s):
                ss = min(s, n - x0)
                e = self._gather(u, x0, ss, self.W1)
                r = self.rho[:, x0:x0 + ss]
                if self.G == 1:
                    e = e * r
                else:
                    e = (e.view(B, self.G, 24, ss, n, n) * r[None, :, None]).sum(1)
                self._scatter_into(f, e, x0)
        else:
            s = self._slab(B, 24 * 3 + 2 * 24 * 24 / max(B, 1))
            n2 = n * n
            for x0 in range(0, n, s):
                ss = min(s, n - x0)
                e = self._gather(u, x0, ss, self.Wg)
                e = e.permute(2, 3, 4, 1, 0).reshape(ss * n2, 24, B)
                e = torch.bmm(self._elem_chunk(x0 * n2, (x0 + ss) * n2), e)
                e = e.reshape(ss, n, n, 24, B).permute(4, 3, 0, 1, 2)
                self._scatter_into(f, e, x0)
        return f

    def diag(self):
        if self._diag is None:
            d = torch.zeros(1, 3, self.n, self.n, self.n, device=self.device, dtype=self.dtype)
            s = self._slab(1, 48)
            for x0 in range(0, self.n, s):
                self._scatter_into(d, self.diagE[:, :, x0:x0 + s], x0)
            self._diag = d
        return self._diag

    def element_matrices(self, x0, x1):
        n = self.n
        if self.kind in ('stored', 'table'):
            return self._elem_chunk(x0 * n * n, x1 * n * n).reshape(x1 - x0, n, n, 24, 24)
        r = self.rho[:, x0:x1]
        return torch.einsum('gxyz,gij->xyzij', r, self.Kg_t)

def prolong(c):
    for d in (2, 3, 4):
        odd = 0.5 * (c + c.roll(-1, d))
        c = torch.stack([c, odd], d + 1).flatten(d, d + 1)
    return c

def restrict(f):
    for d in (2, 3, 4):
        ev = f.narrow(d, 0, f.shape[d] // 2 * 2)
        sh = list(ev.shape)
        sh[d:d + 1] = [sh[d] // 2, 2]
        ev = ev.reshape(sh)
        e0 = ev.select(d + 1, 0)
        e1 = ev.select(d + 1, 1)
        f = e0 + 0.5 * (e1 + e1.roll(1, d))
    return f

@dataclass
class Level:
    n: int
    op: ElemOp
    dinv: torch.Tensor = None
    mask: torch.Tensor = None
    lmax: float = 0.0
    dense_inv: torch.Tensor = None
    act_idx: torch.Tensor = None
    free: torch.Tensor = None
    dinvf: torch.Tensor = None

class Multigrid:

    def __init__(self, rho, Ke, device, dtype, coarse='galerkin', nmin=8, cheb_degree=2, store_bytes=2500000000.0, work_bytes=1000000000.0, power_its=20, cheb_ratio=10.0, coarse_smooth_its=40):
        self.device, self.dtype = (device, dtype)
        self.degree, self.cheb_ratio = (cheb_degree, cheb_ratio)
        self.coarse_smooth_its = coarse_smooth_its
        N = rho.shape[0]
        fine = ElemOp(N, device, dtype, groups=Ke[None], rho=rho, work_bytes=work_bytes)
        self.levels = [Level(N, fine)]
        Pc = torch.as_tensor(child_prolongations(), device=device, dtype=torch.float64)
        Ke_t = torch.as_tensor(Ke, device=device, dtype=torch.float64)
        n, l = (N, 0)
        rho64 = rho.to(device=device, dtype=torch.float64)
        binary = bool(((rho64 == 0) | (rho64 == 1)).all())
        while n % 2 == 0 and n > nmin:
            n //= 2
            l += 1
            prev = self.levels[-1].op
            if coarse in ('mean', 'any'):
                r = rho64.reshape(n, 2 ** l, n, 2 ** l, n, 2 ** l)
                r = r.amax((1, 3, 5)) if coarse == 'any' else r.mean((1, 3, 5))
                if coarse == 'any':
                    r = (r > 0).double()
                op = ElemOp(n, device, dtype, groups=2 ** l * Ke[None], rho=r, work_bytes=work_bytes)
            elif coarse == 'galerkin':
                need = n ** 3 * 576 * torch.finfo(dtype).bits // 8
                if l == 1 and (binary or need > store_bytes):
                    Mc = torch.einsum('cai,ab,cbj->cij', Pc, Ke_t, Pc)
                    r = rho64.reshape(n, 2, n, 2, n, 2).permute(5, 3, 1, 0, 2, 4).reshape(8, n, n, n)
                    if binary:
                        bits = torch.as_tensor([[p >> c & 1 for c in range(8)] for p in range(256)], device=device, dtype=torch.float64)
                        T = torch.einsum('pc,cij->pij', bits, Mc)
                        pw = (2 ** torch.arange(8, device=device, dtype=torch.float64))[:, None, None, None]
                        idx = (r * pw).sum(0).round().to(torch.int32)
                        op = ElemOp(n, device, dtype, table=T, idx=idx, work_bytes=work_bytes)
                    else:
                        op = ElemOp(n, device, dtype, groups=Mc.cpu().numpy(), rho=r, work_bytes=work_bytes)
                else:
                    KEc = torch.empty(576, n ** 3, device=device, dtype=dtype)
                    chunk = max(1, int(200000000.0 // (8 * n * n * 576 * 8)))
                    for X0 in range(0, n, chunk):
                        X1 = min(n, X0 + chunk)
                        Kf = prev.element_matrices(2 * X0, 2 * X1).to(torch.float64)
                        acc = 0
                        for ci, (a, b, c) in enumerate(_OFFS):
                            Kch = Kf[a::2, b::2, c::2]
                            acc = acc + Pc[ci].T @ Kch @ Pc[ci]
                        KEc[:, X0 * n * n:X1 * n * n] = acc.reshape(-1, 576).T.to(dtype)
                        del Kf, acc
                    op = ElemOp(n, device, dtype, KEc=KEc, work_bytes=work_bytes)
            else:
                raise ValueError(coarse)
            self.levels.append(Level(n, op))
        for lev in self.levels:
            d = lev.op.diag()
            dmax = d.max()
            lev.mask = (d > 1e-10 * dmax).to(dtype)
            lev.dinv = torch.where(lev.mask > 0, 1.0 / torch.where(d > 0, d, torch.ones_like(d)), torch.zeros_like(d))
        self.power_its = power_its
        self.set_bc(None)

    def set_bc(self, bcs):
        for l, lev in enumerate(self.levels):
            bc = None if bcs is None else bcs[l].to(self.dtype)
            lev.free = lev.mask if bc is None else lev.mask * bc
            lev.dinvf = lev.dinv if bc is None else lev.dinv * bc
        for lev in self.levels[:-1]:
            lev.lmax = self._power(lev, self.power_its)
        self._coarsest(self.levels[-1])

    def _power(self, lev, its):
        g = torch.Generator(device='cpu').manual_seed(0)
        v = torch.rand(1, 3, lev.n, lev.n, lev.n, generator=g).to(self.device, self.dtype) * lev.free
        lam = 0.0
        for _ in range(its):
            v = v / v.norm()
            w = lev.dinvf * lev.op.apply(v)
            lam = float(w.norm())
            v = w
        return 1.1 * lam

    def _coarsest(self, lev):
        n = lev.n
        nd = 3 * n ** 3
        if nd > 7000:
            lev.dense_inv = None
            lev.lmax = self._power(lev, 30)
            return
        A = torch.zeros(nd, nd, dtype=torch.float64, device=self.device)
        Kall = lev.op.element_matrices(0, n).to(torch.float64).reshape(n ** 3, 24, 24)
        X, Y, Z = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing='ij')
        X, Y, Z = (X.ravel(), Y.ravel(), Z.ravel())
        dofs = np.zeros((n ** 3, 24), np.int64)
        for k, (a, b, c) in enumerate(_OFFS):
            vertex = (X + a) % n * n * n + (Y + b) % n * n + (Z + c) % n
            for d in range(3):
                dofs[:, 3 * k + d] = d * n ** 3 + vertex
        dofs = torch.as_tensor(dofs, device=self.device)
        ii = dofs[:, :, None].expand(-1, 24, 24).reshape(-1)
        jj = dofs[:, None, :].expand(-1, 24, 24).reshape(-1)
        A.index_put_((ii, jj), Kall.reshape(-1), accumulate=True)
        act = torch.nonzero(lev.free.reshape(-1) > 0).squeeze(1)
        Aa = A[act][:, act]
        ev, V = torch.linalg.eigh(Aa)
        tol = 1e-09 * ev.abs().max()
        inv = torch.where(ev > tol, 1.0 / ev, torch.zeros_like(ev))
        lev.dense_inv = (V * inv @ V.T).to(self.dtype)
        lev.act_idx = act

    def _cheb(self, lev, b, x=None):
        lmax = lev.lmax
        lmin = lmax / self.cheb_ratio
        theta, delta = (0.5 * (lmax + lmin), 0.5 * (lmax - lmin))
        sigma = theta / delta
        rho_old = 1.0 / sigma
        r = b if x is None else b - lev.op.apply(x)
        d = lev.dinvf * r / theta
        x = d if x is None else x + d
        for _ in range(self.degree - 1):
            r = r - lev.op.apply(d)
            rho_new = 1.0 / (2 * sigma - rho_old)
            d = rho_new * rho_old * d + 2 * rho_new / delta * (lev.dinvf * r)
            x = x + d
            rho_old = rho_new
        return x

    def _solve_coarsest(self, lev, b):
        if lev.dense_inv is None:
            deg = self.degree
            self.degree = self.coarse_smooth_its
            x = self._cheb(lev, b)
            self.degree = deg
            return x
        B = b.shape[0]
        bf = b.reshape(B, -1)[:, lev.act_idx]
        x = torch.zeros_like(b).reshape(B, -1)
        x[:, lev.act_idx] = bf @ lev.dense_inv
        return x.reshape(b.shape)

    def vcycle(self, b, l=0):
        lev = self.levels[l]
        if l == len(self.levels) - 1:
            return self._solve_coarsest(lev, b)
        x = self._cheb(lev, b)
        r = b - lev.op.apply(x)
        ec = self.vcycle(restrict(r) * self.levels[l + 1].free, l + 1)
        x = x + lev.free * prolong(ec)
        return self._cheb(lev, b, x)

@dataclass
class Result:
    C: np.ndarray
    C_stress: np.ndarray
    asymmetry: float
    iterations: list
    rel_residual: list
    volume_fraction: float
    removed_fraction: float
    n_components: int
    n_islands: int
    times: dict = field(default_factory=dict)
    peak_mem_gb: float = float('nan')
    info: dict = field(default_factory=dict)

def _dot(a, b):
    return (a * b).sum((1, 2, 3, 4), dtype=torch.float64)
STAGNATION = dict(window=25, factor=0.5, below=0.0001)

def _pcg(apply, M, b, bn, tol, maxiter, verbose=False, stagnated=None):
    dtype = b.dtype
    B = b.shape[0]
    iters = [None] * B
    x = torch.zeros_like(b)
    r = b.clone()
    z = M(r)
    p = z.clone()
    rz = _dot(r, z)
    done = _dot(b, b).sqrt() / bn < tol
    for i in torch.nonzero(done).flatten().tolist():
        iters[i] = 0
    it = 0
    hist = []
    W, fac, below = (STAGNATION['window'], STAGNATION['factor'], STAGNATION['below'])
    while it < maxiter and (not bool(done.all())):
        q = apply(p)
        pq = _dot(p, q)
        alpha = torch.where(done | (pq <= 0), torch.zeros_like(rz), rz / pq)
        a_ = alpha.to(dtype)[:, None, None, None, None]
        x += a_ * p
        r -= a_ * q
        it += 1
        rel = _dot(r, r).sqrt() / bn
        hist.append(rel)
        stop = rel < tol
        if len(hist) > W:
            stag = (rel < below) & (rel > fac * hist[-1 - W]) & ~stop & ~done
            if bool(stag.any()) and stagnated is not None:
                stagnated.extend(torch.nonzero(stag).flatten().tolist())
            stop = stop | stag
        for i in torch.nonzero(stop & ~done).flatten().tolist():
            iters[i] = it
        done = done | stop
        if verbose and it % 10 == 0:
            print(f'  it {it} rel {rel.cpu().numpy()}', flush=True)
        if bool(done.all()):
            break
        z = M(r)
        rz_new = _dot(r, z)
        beta = torch.where(done | (rz == 0), torch.zeros_like(rz), -alpha * _dot(z, q) / rz)
        rz = rz_new
        p = z + beta.to(dtype)[:, None, None, None, None] * p
    iters = [it if i is None else i for i in iters]
    return (x, iters, it)

def homogenize(image, E=1.0, nu=0.3, fractional=False, device=None, dtype=torch.float32, precond='mg', coarse='galerkin', tol=1e-06, maxiter=2000, batch=None, cheb_degree=2, nmin=8, store_bytes=2500000000.0, work_bytes=1000000000.0, remove_islands=True, verbose=False):
    t0 = time.time()
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    img = np.asarray(image.cpu() if torch.is_tensor(image) else image)
    N = img.shape[0]
    assert img.shape == (N, N, N) and N >= 3
    if fractional:
        rho = np.clip(img.astype(np.float64), 0.0, 1.0)
        solid = rho > 0
    else:
        solid = img > 0.5 if img.dtype != bool else img
        rho = solid.astype(np.float64)
    vf_in = float(rho.mean())
    labels, ranks, axes = periodic_components(solid)
    nlab = len(ranks) - 1
    keep = ranks >= 1
    keep[0] = False
    n_islands = int((~keep[1:]).sum())
    if remove_islands:
        kept = keep[labels]
        rho = np.where(kept, rho, 0.0)
        newid = np.zeros(nlab + 1, np.int64)
        newid[keep] = np.arange(1, int(keep.sum()) + 1)
        labels = newid[labels]
    removed = vf_in - float(rho.mean())
    t_cc = time.time()
    Ke = hex8_stiffness(E, nu)
    U = macro_displacements()
    FE = Ke @ U
    A = U.T @ Ke @ U * rho.sum()
    V = float(N ** 3)
    nk = int(labels.max())
    info = dict(N=N, coarse=coarse, precond=precond, dtype=str(dtype), tol=tol, rank_1_components=int((ranks[1:][keep[1:]] == 1).sum()) if nlab else 0)
    if nk == 0:
        Z = np.zeros((6, 6))
        return Result(Z, Z, 0.0, [0] * 6, [0.0] * 6, vf_in, removed, 0, n_islands, times=dict(total=time.time() - t0), info=info)
    rho_t = torch.as_tensor(rho, device=device, dtype=dtype)
    lab_t = torch.as_tensor(labels, device=device)
    vertex_labels = lab_t.clone()
    for a, b, c in _OFFS[1:]:
        vertex_labels = torch.maximum(vertex_labels, lab_t.roll((a, b, c), (0, 1, 2)))
    vertex_labels = vertex_labels.reshape(-1)
    counts = torch.bincount(vertex_labels, minlength=nk + 1).to(dtype).clamp_min(1)

    def project(v):
        Bv = v.shape[0]
        vf = v.reshape(Bv, 3, -1)
        s = torch.zeros(Bv, 3, nk + 1, device=device, dtype=v.dtype)
        s.scatter_add_(2, vertex_labels.expand(Bv, 3, -1), vf)
        m = s / counts
        m[:, :, 0] = 0
        return (vf - m.gather(2, vertex_labels.expand(Bv, 3, -1))).reshape(v.shape)
    mg = None
    if precond == 'mg':
        mg = Multigrid(rho_t, Ke, device, dtype, coarse=coarse, nmin=nmin, cheb_degree=cheb_degree, store_bytes=store_bytes, work_bytes=work_bytes)
        op = mg.levels[0].op
        mask, dinv = (mg.levels[0].mask, mg.levels[0].dinv)
        M = mg.vcycle
        info['levels'] = [lv.n for lv in mg.levels]
        info['level_kinds'] = [lv.op.label for lv in mg.levels]
        info['lmax'] = [lv.lmax for lv in mg.levels]
    else:
        op = ElemOp(N, device, dtype, groups=Ke[None], rho=rho_t, work_bytes=work_bytes)
        d = op.diag()
        mask = (d > 1e-10 * d.max()).to(dtype)
        dinv = torch.where(mask > 0, 1.0 / torch.where(d > 0, d, torch.ones_like(d)), torch.zeros_like(d))
        M = lambda r: dinv * r
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t_setup = time.time()

    def rhs(cases, dt, rho_):
        fE = torch.as_tensor(FE[:, cases].T, device=device, dtype=dt)
        f = torch.zeros(len(cases), 3, N, N, N, device=device, dtype=dt)
        o = op_dt(dt)
        s = o._slab(len(cases), 48)
        for x0 in range(0, N, s):
            e = fE[:, :, None, None, None] * rho_[None, None, x0:x0 + s]
            o._scatter_into(f, e, x0)
        return f
    _ops = {}

    def op_dt(dt):
        if dt not in _ops:
            _ops[dt] = op if dt == dtype else ElemOp(N, device, dt, groups=Ke[None], rho=rho_t.to(dt), work_bytes=work_bytes)
        return _ops[dt]
    per_case = 3 * N ** 3 * torch.finfo(dtype).bits // 8
    if batch is None:
        free = (torch.cuda.mem_get_info(device)[0] if device.type == 'cuda' else 8000000000.0) - work_bytes
        batch = int(max(1, min(6, free // (per_case * 24))))
    on_host = device.type == 'cuda' and 6 * per_case > 1000000000.0
    W = torch.zeros(6, 3, N, N, N, device='cpu' if on_host else device, dtype=dtype)
    iters = [0] * 6
    stagnated = []
    info['batch_retries'] = 0
    c0 = 0
    while c0 < 6:
        cases = list(range(c0, min(6, c0 + batch)))
        try:
            b = -project(rhs(cases, torch.float64, rho_t.to(torch.float64))).to(dtype)
            fref = torch.as_tensor(np.linalg.norm(FE[:, cases], axis=0), device=device) * math.sqrt(rho.sum())
            bn = torch.maximum(_dot(b, b).sqrt(), 0.0001 * fref)
            stg = []
            x, its, it = _pcg(op.apply, lambda r: project(M(r)), b, bn, tol, maxiter, verbose, stg)
        except torch.cuda.OutOfMemoryError:
            if batch == 1:
                raise
            b = x = None
            torch.cuda.empty_cache()
            batch = max(1, batch // 2)
            info['batch_retries'] += 1
            continue
        stagnated.extend((cases[i] for i in stg))
        for i, c in enumerate(cases):
            iters[c] = its[i]
        W[cases] = x.to(W.device)
        del b, x
        c0 += len(cases)
    info['batch'] = batch
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t_solve = time.time()
    rho64 = rho_t.to(torch.float64)
    op64 = op_dt(torch.float64)
    Fw = np.zeros((6, 6))
    wKw = np.zeros((6, 6))
    resid = []
    for k in range(6):
        Fk = rhs([k], torch.float64, rho64)
        Kw = op64.apply(W[k:k + 1].to(device=device, dtype=torch.float64))
        res = Kw + Fk
        ref = max(float(_dot(Fk, Fk).sqrt()), 0.0001 * float(np.linalg.norm(FE[:, k])) * math.sqrt(rho.sum()))
        resid.append(float(_dot(res, res).sqrt()) / ref)
        del res
        for i in range(6):
            wi = W[i:i + 1].to(device=device, dtype=torch.float64)
            Fw[k, i] = float(_dot(Fk, wi))
            wKw[i, k] = float(_dot(wi, Kw))
        del Fk, Kw
    C = (A + Fw + Fw.T + wKw) / V
    C = 0.5 * (C + C.T)
    Cs = (A + Fw) / V
    asym = float(np.linalg.norm(Cs - Cs.T) / max(np.linalg.norm(C), 1e-300))
    t_end = time.time()
    peak = torch.cuda.max_memory_allocated(device) / 2 ** 30 if device.type == 'cuda' else float('nan')
    info['iters_total_batches'] = it
    info['stagnated_cases'] = stagnated
    if mg is not None:
        info['level_kinds'] = [lv.op.label for lv in mg.levels]
    else:
        info['level_kinds'] = [op.label]
    return Result(C, Cs, asym, iters, resid, vf_in, removed, nk, n_islands, times=dict(components=t_cc - t0, setup=t_setup - t_cc, solve=t_solve - t_setup, assemble=t_end - t_solve, total=t_end - t0), peak_mem_gb=peak, info=info)
