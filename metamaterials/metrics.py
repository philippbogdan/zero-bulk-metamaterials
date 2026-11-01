"""Support gaps and finite-scale comparisons, recomputed from stored evidence."""
import json
import math
import mpmath as mp
import numpy as np
from .elastic import NUS, THETAS, solid_moduli, hs_upper, iso_tensor, rotate_mandel, sym_to_mandel
from .constructions import hs_corner, slipwall_corner, xy_of
from .evidence import DATA, finest_point
from .laminate import mp_laminate


def known_points(nu, theta):
    """HS, disconnected grains and a slip wall at its recorded small wall fraction."""
    ks, ms = solid_moduli(nu)
    kh, mh = hs_upper(theta, ks, ms)
    x, y, defect = xy_of(hs_corner(nu, theta), nu, theta)
    points = [dict(family='void', x=0., y=0., defect=0.),
              dict(family='hs', x=x, y=y, defect=defect)]
    fractions = theta * np.concatenate([np.logspace(-6, -1, 6), np.linspace(.1, 1, 91)[:-1]])
    for fraction in fractions:
        k, m = hs_upper(fraction, ks, ms)
        points.append(dict(family='dust', x=k / kh, y=m / mh, defect=0.))
    wall, _ = slipwall_corner(nu, theta, 1e-8)
    x, y, defect = xy_of(wall, nu, theta)
    points.append(dict(family='slip', x=x, y=y, defect=defect))
    return points


def corner_gaps(records):
    """Compare support in the 89 integer directions from 91 to 179 degrees."""
    certified = [finest_point(r) for r in records]
    certified = [p for p in certified if p['defect'] <= 1e-3]
    angles = np.arange(91, 180)
    u = np.column_stack([np.cos(np.deg2rad(angles)), np.sin(np.deg2rad(angles))])
    box = np.maximum(u, 0).sum(axis=1)
    rows = []
    for nu in NUS:
        for theta in THETAS:
            baseline = known_points(nu, theta)
            accepted = [p for p in certified if p['nu'] == nu and p['theta'] == theta]
            before_xy = np.array([[p['x'], p['y']] for p in baseline])
            all_xy = np.array([[p['x'], p['y']] for p in baseline + accepted])
            before = box - (u @ before_xy.T).max(axis=1)
            values = u @ all_xy.T
            after = box - values.max(axis=1)
            i, j = int(before.argmax()), int(after.argmax())
            winner = (baseline + accepted)[int(values[j].argmax())]
            rows.append(dict(nu=nu, theta=theta, before=float(before[i]), after=float(after[j]),
                             phi_before=int(angles[i]), phi_after=int(angles[j]),
                             attained_by=winner.get('file', winner['family']), accepted=len(accepted),
                             support=[dict(phi=int(a), before=float(b), after=float(c))
                                      for a, b, c in zip(angles, before, after)]))
    return rows


def exact_laminate(tensors, fractions, normal):
    with mp.workdps(40):
        value = mp_laminate([mp.matrix(np.asarray(c).tolist()) for c in tensors],
                            fractions, normal, mp.mpf('1e-30'))
    return np.array(value.tolist(), dtype=float)


def finite_scale_table(directory=None):
    """Reapply the exact formula to the FEM tensors of the preceding level."""
    directory = PathLike(directory)
    sources = {p.name: json.loads(p.read_text()) for p in sorted(directory.glob('*.json'))}
    a_record = sources['q2_finite_herringbone_A.json']
    a_tensors = {c['p2']: np.asarray(c['C_fem']) for c in a_record['cases']}
    reflection = np.diag([-1., 1., 1.])
    rows = []
    for filename, record in sources.items():
        solid = iso_tensor(*solid_moduli(record['nu']))
        for case in record['cases']:
            if case['level'] == 'A':
                comparison = exact_laminate([solid, record['plates']['+']], [.5, .5], [0, 0, 1])
                q = np.array([1., 0., 1.]) / np.sqrt(2)
                v = sym_to_mandel((np.outer([0, 0, 1], q) + np.outer(q, [0, 0, 1])) / 2)
                mode_key = 'mode_s_fem_over_solid'
                formula_key = 'mode_s_formula_over_solid'
                ratio = case['p2'] / record['P']
            else:
                plus = a_tensors[case['p2']]
                minus = rotate_mandel(plus, reflection)
                comparison = exact_laminate([plus, minus], [.5, .5], [1, 0, 0])
                v = np.array([0., 0., 1., 0., 0., 0.])
                mode_key, formula_key = 'nn_fem_over_solid', 'nn_formula_over_solid'
                ratio = case['p3'] / case['p2']
            fem = np.asarray(case['C_fem'])
            error = float(np.linalg.norm(fem - comparison) / np.linalg.norm(comparison))
            rows.append(dict(file=filename, level=case['level'], p2=case['p2'], p3=case.get('p3'),
                             N=case['N'], ratio=ratio, relative_error=error,
                             expected_error=case['rel_err_vs_formula'],
                             mode_fem=float(v @ fem @ v / (v @ solid @ v)),
                             expected_mode_fem=case[mode_key],
                             mode_formula=float(v @ comparison @ v / (v @ solid @ v)),
                             expected_mode_formula=case[formula_key],
                             formula_tensor_difference=float(np.linalg.norm(comparison - case['C_formula']))
                             if 'C_formula' in case else None))
    return rows


def PathLike(directory):
    from pathlib import Path
    return Path(directory) if directory is not None else DATA / 'finite_scale'
