"""Project a free-wall tree onto the compatible-dilation equations."""
import torch
from .torch_algebra import T, sym6, frame, rot6

def perp_proj(m):
    R = frame(m)
    Q = rot6(R)
    Pin = torch.zeros(6, 6, dtype=T)
    Pin[0, 0] = Pin[1, 1] = Pin[5, 5] = 1.0
    return Q.transpose(-1, -2) @ Pin @ Q

def residual(P, D):
    n = P['leaf_n'][:, :, 0]
    p = P['leaf_p'][:, :, 0]
    s = sym6(n, p)
    R, L = s.shape[:2]
    E = torch.zeros(R, L, 6, L, dtype=T)
    idx = torch.arange(L)
    E[:, idx, :, idx] = s.permute(1, 0, 2)
    rows = []
    for l in reversed(range(D)):
        f = torch.sigmoid(P['f%d' % l])
        A_ = E[:, 0::2]
        B_ = E[:, 1::2]
        Pp = perp_proj(P['m%d' % l])
        rows.append((Pp @ (A_ - B_)).reshape(R, -1, L))
        E = f[..., None, None] * A_ + (1 - f[..., None, None]) * B_
    rows.append(E[:, 0])
    A = torch.cat(rows, 1)
    I6 = torch.tensor([1, 1, 1, 0, 0, 0], dtype=T)
    b = torch.zeros(R, A.shape[1], dtype=T)
    b[:, -6:] = I6
    sol = torch.linalg.lstsq(A, b[..., None]).solution
    res = (A @ sol)[..., 0] - b
    return (res ** 2).sum(-1) / 3.0

def _system(P, D):
    n = P['leaf_n'][:, :, 0]
    p = P['leaf_p'][:, :, 0]
    s = sym6(n, p)
    R, L = s.shape[:2]
    E = torch.zeros(R, L, 6, L, dtype=T)
    idx = torch.arange(L)
    E[:, idx, :, idx] = s.permute(1, 0, 2)
    rows = []
    for l in reversed(range(D)):
        f = torch.sigmoid(P['f%d' % l])
        A_ = E[:, 0::2]
        B_ = E[:, 1::2]
        rows.append((perp_proj(P['m%d' % l]) @ (A_ - B_)).reshape(R, -1, L))
        E = f[..., None, None] * A_ + (1 - f[..., None, None]) * B_
    rows.append(E[:, 0])
    A = torch.cat(rows, 1)
    b = torch.zeros(R, A.shape[1], dtype=T)
    b[:, -6:] = torch.tensor([1, 1, 1, 0, 0, 0], dtype=T)
    return (A, b)

def project(design, iters=20, tol=1e-15):
    import copy, numpy as np
    D = design['D']
    keys = ['leaf_n', 'leaf_p'] + ['m%d' % l for l in range(D)] + ['f%d' % l for l in range(D)]
    P0 = {'leaf_n': torch.tensor([design['leaf_n']], dtype=T), 'leaf_p': torch.tensor([design['leaf_p']], dtype=T)}
    for l in range(D):
        P0['m%d' % l] = torch.tensor([design['m'][l]], dtype=T)
        P0['f%d' % l] = torch.logit(torch.tensor([design['f'][l]], dtype=T))
    shapes = [P0[k].shape for k in keys]
    sizes = [P0[k].numel() for k in keys]
    z = torch.cat([P0[k].reshape(-1) for k in keys])
    A, b = _system(P0, D)
    alpha = torch.linalg.lstsq(A, b[..., None]).solution[0, :, 0]
    z = torch.cat([z, alpha])

    def F(zz):
        P = {}
        o = 0
        for k, sh, sz in zip(keys, shapes, sizes):
            P[k] = zz[o:o + sz].reshape(sh)
            o += sz
        A, b = _system(P, D)
        return A[0] @ zz[o:] - b[0]
    hist = []
    best = (float('inf'), z)
    for it in range(iters):
        r = F(z)
        hist.append(float(r.norm()))
        if hist[-1] < best[0]:
            best = (hist[-1], z.clone())
        if hist[-1] < tol:
            break
        J = torch.autograd.functional.jacobian(F, z)
        z = z - torch.linalg.lstsq(J, r[:, None], driver='gelsd').solution[:, 0]
    z = best[1]
    hist.append(best[0])
    out = copy.deepcopy(design)
    o = 0
    vals = {}
    for k, sh, sz in zip(keys, shapes, sizes):
        vals[k] = z[o:o + sz].reshape(sh)[0]
        o += sz
    out['leaf_n'] = vals['leaf_n'].tolist()
    out['leaf_p'] = vals['leaf_p'].tolist()
    out['m'] = [vals['m%d' % l].tolist() for l in range(D)]
    out['f'] = [torch.sigmoid(vals['f%d' % l]).tolist() for l in range(D)]
    out['leaf_eta'] = [[0.0] * len(e) for e in design['leaf_eta']]
    out['zero_mode_alpha'] = z[o:].tolist()
    out['zero_mode_residual_history'] = hist
    return (out, hist[-1])

def _mp_system_residual(design, alpha):
    import mpmath as mp
    from ..laminate import mp_frame_to_e3, mp_rot6
    D = design['D']
    s2 = mp.sqrt(2)

    def sym6mp(n, p):
        n = [mp.mpf(x) for x in n]
        p = [mp.mpf(x) for x in p]
        nn = mp.sqrt(sum((x * x for x in n)))
        pn = mp.sqrt(sum((x * x for x in p)))
        n = [x / nn for x in n]
        p = [x / pn for x in p]
        M = [[(n[i] * p[j] + p[i] * n[j]) / 2 for j in range(3)] for i in range(3)]
        return mp.matrix([M[0][0], M[1][1], M[2][2], s2 * M[1][2], s2 * M[0][2], s2 * M[0][1]])
    E = [alpha[i] * sym6mp(design['leaf_n'][i][0], design['leaf_p'][i][0]) for i in range(2 ** D)]
    res = []
    for l in reversed(range(D)):
        new = []
        for j in range(len(E) // 2):
            m = design['m'][l][j]
            f = mp.mpf(design['f'][l][j])
            Q = mp_rot6(mp_frame_to_e3(m))
            dlt = Q * (E[2 * j] - E[2 * j + 1])
            for c in (2, 3, 4):
                dlt[c] = 0
            dlt = Q.T * dlt
            res += list(dlt)
            new.append(f * E[2 * j] + (1 - f) * E[2 * j + 1])
        E = new
    I = mp.matrix([1, 1, 1, 0, 0, 0])
    res += list(E[0] - I)
    return res

def refine_mp(design, dps=60, iters=8):
    import copy, numpy as np, mpmath as mp
    mp.mp.dps = dps
    D = design['D']
    keys = ['leaf_n', 'leaf_p'] + ['m%d' % l for l in range(D)] + ['f%d' % l for l in range(D)]

    def pack(d):
        v = list(np.ravel(d['leaf_n'])) + list(np.ravel(d['leaf_p']))
        for l in range(D):
            v += list(np.ravel(d['m'][l]))
        for l in range(D):
            v += [float(np.log(f / (1 - f))) for f in d['f'][l]]
        return v
    L = 2 ** D

    def unpack(z):
        d = copy.deepcopy(design)
        o = 0
        d['leaf_n'] = [[[z[o + 3 * i + k] for k in range(3)]] for i in range(L)]
        o += 3 * L
        d['leaf_p'] = [[[z[o + 3 * i + k] for k in range(3)]] for i in range(L)]
        o += 3 * L
        for l in range(D):
            d['m'][l] = [[z[o + 3 * j + k] for k in range(3)] for j in range(2 ** l)]
            o += 3 * 2 ** l
        for l in range(D):
            d['f'][l] = [1 / (1 + mp.exp(-z[o + j])) for j in range(2 ** l)]
            o += 2 ** l
        return (d, z[o:])
    z0 = [mp.mpf(x) for x in pack(design)] + [mp.mpf(a) for a in design['zero_mode_alpha']]
    P = {'leaf_n': torch.tensor([design['leaf_n']], dtype=T), 'leaf_p': torch.tensor([design['leaf_p']], dtype=T)}
    for l in range(D):
        P['m%d' % l] = torch.tensor([design['m'][l]], dtype=T)
        P['f%d' % l] = torch.logit(torch.tensor([design['f'][l]], dtype=T))
    shapes = [P[k].shape for k in keys]
    sizes = [P[k].numel() for k in keys]
    zf = torch.tensor([float(x) for x in z0], dtype=T)

    def F(zz):
        Q = {}
        o = 0
        for k, sh, sz in zip(keys, shapes, sizes):
            Q[k] = zz[o:o + sz].reshape(sh)
            o += sz
        A, b = _system(Q, D)
        return A[0] @ zz[o:] - b[0]
    J = torch.autograd.functional.jacobian(F, zf).numpy()
    Jp = np.linalg.pinv(J)
    z = z0
    hist = []
    for it in range(iters):
        d, a = unpack(z)
        r = _mp_system_residual(d, a)
        nr = mp.sqrt(sum((x * x for x in r)))
        hist.append(float(nr))
        if nr < mp.mpf(10) ** (-dps + 8):
            break
        step = Jp @ np.array([float(x) for x in r])
        z = [z[i] - mp.mpf(step[i]) for i in range(len(z))]
    d, a = unpack(z)
    d['leaf_n'] = [[[float(x) for x in v[0]]] for v in d['leaf_n']]
    d['leaf_p'] = [[[float(x) for x in v[0]]] for v in d['leaf_p']]
    d['m'] = [[[float(x) for x in v] for v in lv] for lv in d['m']]
    d['f'] = [[float(x) for x in lv] for lv in d['f']]
    d['zero_mode_alpha'] = [float(x) for x in a]
    d2 = copy.deepcopy(d)
    r = _mp_system_residual(d2, [mp.mpf(x) for x in d2['zero_mode_alpha']])
    hist.append(float(mp.sqrt(sum((x * x for x in r)))))
    return (d, hist)
