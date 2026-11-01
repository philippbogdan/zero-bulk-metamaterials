"""Build figure tables and voxel geometry from the included evidence files."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'data'
OUT = ROOT / 'figures' / 'data'


def write_csv(name, rows):
    with (OUT / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def hs(theta, nu):
    k, m = 1 / (3 * (1 - 2 * nu)), 1 / (2 * (1 + nu))
    a = 4 * m / 3
    b = m * (9 * k + 8 * m) / (6 * (k + 2 * m))
    return np.array([theta * k * a / (a + (1 - theta) * k),
                     theta * m * b / (b + (1 - theta) * m)])


def main():
    OUT.mkdir(exist_ok=True)
    points, headline = [], []
    for path in sorted((SOURCE / 'certified').glob('*.json')):
        record = json.loads(path.read_text())
        finest = min(r['eps'] for r in record['runs'])
        runs = [r for r in record['runs'] if r['eps'] == finest]
        point = dict(file=path.name, family=record['family'], nu=record['nu'],
                     theta=record['theta'], x=max(r['x'] for r in runs),
                     y=min(r['y'] for r in runs), defect=max(r['defect'] for r in runs),
                     eps=finest, accepted=int(max(r['defect'] for r in runs) <= 1e-3))
        points.append(point)
        if record['family'] == 'zero_bulk_aniso':
            headline.append(point)
    write_csv('certified.csv', points)
    write_csv('headline.csv', headline)
    baseline, gaps, supports = [], [], []
    angles = np.arange(91, 180)
    directions = np.column_stack([np.cos(np.deg2rad(angles)), np.sin(np.deg2rad(angles))])
    for nu, theta in sorted({(p['nu'], p['theta']) for p in points}):
        scale = hs(theta, nu)
        body_fractions = theta * np.concatenate([np.logspace(-6, -1, 6), np.linspace(.1, 1, 91)[:-1]])
        xy = [[0., 0.], [1., 0.], [1., 1.]]
        for t in body_fractions:
            xy.append((hs(t, nu) / scale).tolist())
        for i, (x, y) in enumerate(xy):
            baseline.append(dict(nu=nu, theta=theta, family=['void', 'slip_limit', 'hs'][i] if i < 3 else 'dust', x=x, y=y))
        attained = [[p['x'], p['y']] for p in points if p['nu'] == nu and p['theta'] == theta and p['accepted']]
        before = directions[:, 1] - (directions @ np.asarray(xy).T).max(axis=1)
        after = directions[:, 1] - (directions @ np.asarray(xy + attained).T).max(axis=1)
        gaps.append(dict(nu=nu, theta=theta, before=float(before.max()), after=float(after.max()),
                         phi_before=int(angles[before.argmax()]), phi_after=int(angles[after.argmax()])))
        for a, b, c in zip(angles, before, after):
            supports.append(dict(nu=nu, theta=theta, phi_deg=int(a), gap_before=float(b), gap_after=float(c)))
    write_csv('baseline.csv', baseline)
    write_csv('gaps.csv', gaps)
    write_csv('support.csv', supports)
    finite = []
    for path in sorted((SOURCE / 'finite_scale').glob('*.json')):
        record = json.loads(path.read_text())
        for case in record['cases']:
            finite.append(dict(file=path.name, level=case['level'], P=record['P'], p2=case['p2'],
                               p3=case.get('p3', ''), N=case['N'],
                               ratio=case.get('ratio_p2_P', case.get('ratio_p3_p2')),
                               relative_error=case['rel_err_vs_formula']))
    write_csv('finite_scale.csv', finite)
    record = json.loads((SOURCE / 'finite_scale/q2_finite_herringbone_H_N128.json').read_text())
    case = record['cases'][0]
    n, p, p2, p3 = case['N'], record['P'], case['p2'], case['p3']
    i, k = np.arange(n)[:, None], np.arange(n)[None, :]
    ap = ((i + k) % p < p // 2) | (k % p2 < p2 // 2)
    am = ((i - k) % p < p // 2) | (k % p2 < p2 // 2)
    mask = np.where(i % p3 < p3 // 2, ap, am)
    geometry = dict(source='q2_finite_herringbone_H_N128.json', N=n, P=p, p2=p2, p3=p3,
                    coordinates='i=x1, k=x3; invariant along x2', solid=mask.astype(int).tolist())
    (OUT / 'wall_geometry.json').write_text(json.dumps(geometry, separators=(',', ':')) + '\n')
    path = SOURCE / 'certified/zero_bulk_aniso_nu0.3_th0.1.json'
    (OUT / 'headline_geometry.json').write_text(path.read_text())
    print(f'{len(points)} certified records; {len(headline)} headline records; {len(finite)} FEM comparisons')
    main_gaps = [g for g in gaps if g['theta'] < 1]
    print('Corner gap:', min(g['before'] for g in main_gaps), max(g['before'] for g in main_gaps),
          'to', min(g['after'] for g in main_gaps), max(g['after'] for g in main_gaps))


if __name__ == '__main__':
    main()
