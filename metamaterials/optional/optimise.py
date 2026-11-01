"""Search laminate parameters by maximising shear with isotropy and dilation penalties."""
import json, math, sys, time, argparse
import torch
from .torch_algebra import T, laminate, iso, iso_tensor, interface_crystal
ETA_FLOOR = 1e-09
SEQ_CACHE = {}

def seq_dirs():
    import numpy as np
    from ..laminate import icosahedral_axes
    cub = np.eye(3)
    octa = np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1]], float) / np.sqrt(3)
    return np.vstack([icosahedral_axes(), cub, octa])

def seq_base(P, Cs, theta):
    import numpy as np
    from ..laminate import gamma_op
    key = (float(Cs[0, 0]), float(Cs[0, 1]))
    if key not in SEQ_CACHE:
        Cn = Cs.detach().cpu().numpy()
        SEQ_CACHE[key] = torch.tensor(np.array([gamma_op(n, Cn) for n in seq_dirs()]), dtype=T)
    Gm = SEQ_CACHE[key].to(Cs.device)
    w = torch.softmax(P['leaf_sw'], -1)
    M = torch.einsum('rlj,jab->rlab', w, Gm)
    return Cs + (1 - theta) * torch.linalg.inv(theta * M - torch.linalg.inv(Cs))

def leaf_tensors(Mb, P, base, eta_fixed=None):
    R, nl = P['leaf_w'].shape[:2]
    X = Mb.expand(R, nl, 6, 6)
    if base == 'seq':
        X = seq_base(P, Mb, P['theta'])
    if base == 'plate':
        q = P['leaf_q']
        e = torch.eye(3, dtype=T, device=q.device)
        qn = q / q.norm(dim=-1, keepdim=True)
        a1 = torch.cross(qn, e[0].expand_as(qn), dim=-1) + 0.001 * torch.cross(qn, e[1].expand_as(qn), dim=-1)
        a1 = a1 / a1.norm(dim=-1, keepdim=True)
        a2 = torch.cross(qn, a1, dim=-1)
        for a in (qn, a1, a2):
            X = interface_crystal(X, qn, a, torch.full((R, nl, 1, 1), 1e-12, dtype=T, device=q.device))
    if eta_fixed is None:
        eta = ETA_FLOOR + (1 - ETA_FLOOR) * torch.sigmoid(P['leaf_w'])
    else:
        eta = torch.full_like(P['leaf_w'], eta_fixed)
    for k in range(P['leaf_n'].shape[2]):
        X = interface_crystal(X, P['leaf_n'][:, :, k], P['leaf_p'][:, :, k], eta[:, :, k, None, None])
    return (X, eta)

def build(Mb, P, D, base, eta_fixed=None):
    X, eta = leaf_tensors(Mb, P, base, eta_fixed)
    for l in reversed(range(D)):
        A = X[:, 0::2]
        Bb = X[:, 1::2]
        f = torch.sigmoid(P['f%d' % l])
        X = laminate([A, Bb], [f, 1 - f], P['m%d' % l])
    return (X[:, 0], eta)

def refs(base, nuB):
    kB = 1 / (3 * (1 - 2 * nuB))
    mB = 1 / (2 * (1 + nuB))
    if base == 'B':
        return (kB, mB, iso_tensor(kB, mB))
    if isinstance(base, tuple):
        from ..elastic import hs_upper
        kh, mh = hs_upper(base[1], kB, mB)
        return (kh, mh, iso_tensor(kB, mB))
    zk = 4 * mB / 3
    zm = mB * (9 * kB + 8 * mB) / (6 * (kB + 2 * mB))
    return (kB * zk / (kB + zk), mB * zm / (mB + zm), iso_tensor(kB, mB))

def run(nuB, c, D=5, R=64, iters=3000, seed=0, dmax=0.00095, K=1, base='B', eta_fixed=None, lr=0.03, device='cpu', log=True, init=None, theta=None, extra_pen=None):
    torch.manual_seed(seed)
    kR, mR, Mb = refs(('seq', theta) if base == 'seq' else base, nuB)
    Mb = Mb.to(device)
    nl = 2 ** D
    P = {'leaf_n': torch.randn(R, nl, K, 3, dtype=T, device=device), 'leaf_p': torch.randn(R, nl, K, 3, dtype=T, device=device), 'leaf_w': torch.randn(R, nl, K, dtype=T, device=device) - 1.0, 'leaf_q': torch.randn(R, nl, 3, dtype=T, device=device), 'leaf_sw': torch.zeros(R, nl, 13, dtype=T, device=device)}
    for l in range(D):
        P['m%d' % l] = torch.randn(R, 2 ** l, 3, dtype=T, device=device)
        P['f%d' % l] = torch.zeros(R, 2 ** l, dtype=T, device=device)
    if init is not None:
        for k, v in init.items():
            P[k] = v.to(device).clone()
    train = [k for k in P if not (k == 'leaf_w' and eta_fixed is not None) and (not (k == 'leaf_q' and base != 'plate')) and (not (k == 'leaf_sw' and base != 'seq'))]
    P['theta'] = theta
    for k in train:
        P[k].requires_grad_(True)
    opt = torch.optim.Adam([P[k] for k in train], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=lr * 0.01)
    for it in range(iters):
        frac = it / iters
        pen = 10 ** min(7, 1 + 8 * frac)
        opt.zero_grad()
        C, eta = build(Mb, P, D, base, eta_fixed)
        k, m, d = iso(C)
        J = m / mR - c * k / kR
        Jp = J - pen * torch.relu(d - dmax) ** 2
        if extra_pen is not None:
            Jp = Jp - extra_pen(P, frac)
        (-Jp.sum()).backward()
        opt.step()
        sched.step()
        if log and (it % 500 == 0 or it == iters - 1):
            with torch.no_grad():
                ok = d <= 0.001
                jb = torch.where(ok, J, torch.full_like(J, -1000000000.0))
                i = int(jb.argmax())
                print(f'it {it} best J={float(jb[i]):.5f} y={float(m[i] / mR):.5f} x={float(k[i] / kR):.2e} d={float(d[i]):.2e} ok={int(ok.sum())}/{R}', flush=True)
    with torch.no_grad():
        C, eta = build(Mb, P, D, base, eta_fixed)
        k, m, d = iso(C)
        J = m / mR - c * k / kR
    out = []
    for i in range(R):
        out.append({'J': float(J[i]), 'zero_mode_residual': float(extra_pen.residual(P)[i]) if getattr(extra_pen, 'residual', None) else None, 'x_rel': float(k[i] / kR), 'y_rel': float(m[i] / mR), 'defect': float(d[i]), 'D': D, 'K': K, 'base': base, 'nuB': nuB, 'c': c, 'leaf_n': P['leaf_n'][i].detach().cpu().tolist(), 'leaf_p': P['leaf_p'][i].detach().cpu().tolist(), 'leaf_eta': eta[i].detach().cpu().tolist(), 'leaf_q': P['leaf_q'][i].detach().cpu().tolist() if base == 'plate' else None, 'leaf_sw': torch.softmax(P['leaf_sw'][i], -1).detach().cpu().tolist() if base == 'seq' else None, 'theta': theta, 'm': [P['m%d' % l][i].detach().cpu().tolist() for l in range(D)], 'f': [torch.sigmoid(P['f%d' % l][i]).detach().cpu().tolist() for l in range(D)]})
    return out
