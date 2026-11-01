"""Evaluate stored laminate geometry with arbitrary-precision wall and tree algebra."""
import numpy as np
import mpmath as mp
from .elastic import solid_moduli, iso_tensor, iso_project, iso_defect, hs_upper, sym_to_mandel
from .laminate import mp_laminate, to_mp, sequential_void_core, icosahedral_axes, frame_to_e3, laminate as f_laminate
G = H = 0.5
BETA = np.pi / 4

def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)

def material_M(nu, theta_M):
    ks, ms = solid_moduli(nu)
    Cs = iso_tensor(ks, ms)
    if theta_M >= 1:
        return Cs
    return sequential_void_core(Cs, theta_M, icosahedral_axes(), np.ones(6) / 6)

def _smode(n, p):
    n = _unit(n)
    p = _unit(p)
    return sym_to_mandel(0.5 * (np.outer(n, p) + np.outer(p, n)))

def wall_core(lam, M, Z, n, p, delta, par_tol=1e-06):
    n = _unit(n)
    p = _unit(p)
    if np.linalg.norm(np.cross(n, p)) > par_tol:
        pl = lam([M, Z], [H, 1 - H], p, delta)
        return lam([M, pl], [G, 1 - G], n, delta)
    t = frame_to_e3(n)[0]
    ws = []
    for sgn in (1, -1):
        q = np.cos(BETA) * n + sgn * np.sin(BETA) * t
        pl = lam([M, Z], [H, 1 - H], q, delta)
        ws.append(lam([M, pl], [G, 1 - G], n, delta))
    return lam(ws, [0.5, 0.5], t, delta)

def _flam(Cs, fs, n, delta):
    return f_laminate([np.asarray(C, float) for C in Cs], [float(f) for f in fs], n, delta)

def _NN_compliance_s(W, n, s):
    from .laminate import mp_frame_to_e3, mp_rot6, N_IDX
    Q = mp_rot6(mp_frame_to_e3(_unit(n)))
    Wr = Q * W * Q.T
    sr = Q * to_mp(np.asarray(s).reshape(6, 1))
    CNN = mp.matrix(3, 3)
    sN = mp.matrix(3, 1)
    for i, a in enumerate(N_IDX):
        sN[i] = sr[a]
        for j, b in enumerate(N_IDX):
            CNN[i, j] = Wr[a, b]
    return ((sN.T * CNN ** (-1) * sN)[0, 0], (sN.T * sN)[0, 0])

def fit_bridge(Mmp, core, n, p, eta, eps, delta):
    s = _smode(n, p)
    sM = to_mp(s.reshape(6, 1))
    lam = (1 - mp.mpf(eta)) / (mp.mpf(eta) * (sM.T * Mmp * sM)[0, 0])
    t = frame_to_e3(_unit(n))[0]
    a = mp.mpf('1e-3')
    for _ in range(40):
        W = mp_laminate([core, Mmp], [1 - a, a], t, delta)
        q, s2 = _NN_compliance_s(W, n, s)
        target = lam * s2 ** 2 / eps
        r = q / target
        a_new = min(a * r, mp.mpf('0.999'))
        if abs(r - 1) < mp.mpf('1e-12'):
            break
        a = a_new
    return a

def leaf_volumes(design):
    v = np.ones(1)
    for l in range(design['D']):
        f = np.asarray(design['f'][l], float)
        v = np.stack([v * f, v * (1 - f)], axis=1).ravel()
    return v
ETA_CUT = 1e-12
ETA_UNCUT = 1 - 1e-09

def leaf_base(design, i, nu, theta_M):
    from .directions import seq_dirs
    ks, ms = solid_moduli(nu)
    return sequential_void_core(iso_tensor(ks, ms), theta_M, seq_dirs(), np.asarray(design['leaf_sw'][i], float))

def evaluate(design, M, eps, delta, limit=False, theta_leaf=None, nu=None, theta_M=None):
    D = design['D']
    Mmp = to_mp(M)
    Mf = np.asarray(M, float)
    Z = mp.zeros(6, 6)
    X = []
    wall_void = []
    for i in range(2 ** D):
        if design.get('base', 'B') == 'plate':
            q = design['leaf_q'][i]
            base = mp_laminate([Mmp, Z], [mp.mpf(theta_leaf), 1 - mp.mpf(theta_leaf)], q, delta)
        elif design.get('base', 'B') == 'seq':
            base = to_mp(leaf_base(design, i, nu, theta_M))
        else:
            base = Mmp
        Xi = base
        wv = 0.0
        for n, p, eta in zip(design['leaf_n'][i], design['leaf_p'][i], design['leaf_eta'][i]):
            if eta >= ETA_UNCUT:
                continue
            if limit:
                s = to_mp(_smode(n, p).reshape(6, 1))
                Xs = Xi * s
                e = mp.mpf(0) if eta < ETA_CUT else mp.mpf(eta)
                Xi = Xi - (1 - e) * Xs * Xs.T / (s.T * Xs)[0, 0]
                continue
            Wm = base if design.get('base', 'B') == 'seq' else Mmp
            core = wall_core(mp_laminate, Wm, Z, n, p, delta)
            if eta < ETA_CUT:
                W = core
                a = 0.0
            else:
                a = fit_bridge(Wm, core, n, p, eta, eps, delta)
                t = frame_to_e3(_unit(n))[0]
                W = mp_laminate([core, Wm], [1 - a, a], t, delta)
                a = float(a)
            Xi = mp_laminate([Xi, W], [1 - eps, eps], n, delta)
            wv = 1 - (1 - wv) * (1 - float(eps) * (1 - a) * (1 - G) * (1 - H))
        X.append(Xi)
        wall_void.append(wv)
    v = leaf_volumes(design)
    for l in reversed(range(D)):
        f = design['f'][l]
        m = design['m'][l]
        X = [mp_laminate([X[2 * i], X[2 * i + 1]], [mp.mpf(f[i]), 1 - mp.mpf(f[i])], m[i], delta) for i in range(len(X) // 2)]
    return (X[0], float((v * np.asarray(wall_void)).sum()))

def certify(design, nu, theta, eps_list=(0.0001, 1e-06), deltas=('1e-30', '1e-40'), dps=60):
    mp.mp.dps = dps
    ks, ms = solid_moduli(nu)
    plate = design.get('base', 'B') == 'plate'
    out = {'nu': nu, 'theta': theta, 'runs': []}
    for eps in eps_list:
        for d in deltas:
            if plate:
                M = material_M(nu, 1.0)
                Cm, wv = evaluate(design, M, mp.mpf(eps), mp.mpf(d), theta_leaf=theta)
                th_tot = theta * (1 - wv)
            else:
                thM = 1.0 if theta >= 1 else theta
                for _ in range(3):
                    M = material_M(nu, thM)
                    Cm, wv = evaluate(design, M, mp.mpf(eps), mp.mpf(d), nu=nu, theta_M=thM)
                    if theta >= 1:
                        break
                    thM = theta / (1 - wv)
                th_tot = thM * (1 - wv)
            C = np.array(Cm.tolist(), dtype=float)
            k, m = iso_project(C)
            kh, mh = hs_upper(min(th_tot, 1.0), ks, ms)
            out['runs'].append({'eps': eps, 'delta': d, 'theta_total': float(th_tot), 'x': float(k / kh), 'y': float(m / mh), 'defect': float(iso_defect(C)), 'kappa': float(k), 'mu': float(m)})
    M = material_M(nu, 1.0 if theta >= 1 or plate else theta)
    Cm, _ = evaluate(design, M, 0, mp.mpf(deltas[-1]), limit=True, theta_leaf=theta, nu=nu, theta_M=1.0 if theta >= 1 else theta)
    C = np.array(Cm.tolist(), dtype=float)
    k, m = iso_project(C)
    kh, mh = hs_upper(min(theta, 1.0), ks, ms)
    out['limit'] = {'x': float(k / kh), 'y': float(m / mh), 'defect': float(iso_defect(C)), 'C': C.tolist()}
    return out
