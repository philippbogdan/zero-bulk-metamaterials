"""Read the stored designs and select their conservative finest-wall evaluations."""
import json
from pathlib import Path
import numpy as np

DATA = Path(__file__).resolve().parents[1] / 'data'
FAMILIES = {'zero_bulk_aniso', 'zero_bulk_iso', 'zero_bulk_module', 'tree_aniso', 'tree_iso'}


def read_designs(directory=None):
    records = []
    for path in sorted(Path(directory or DATA / 'certified').glob('*.json')):
        record = json.loads(path.read_text())
        if record['family'] not in FAMILIES or not record['runs']:
            raise ValueError(f'Invalid certified record: {path.name}')
        record['file'] = path.name
        records.append(record)
    if not records:
        raise ValueError('No certified designs found')
    return records


def finest_point(record):
    eps = min(float(r['eps']) for r in record['runs'])
    runs = [r for r in record['runs'] if float(r['eps']) == eps]
    values = [r[k] for r in runs for k in ('x', 'y', 'defect', 'theta_total')]
    if not np.isfinite(values).all():
        raise ValueError(f'Non-finite evaluation: {record["file"]}')
    return dict(file=record['file'], family=record['family'], nu=record['nu'], theta=record['theta'],
                x=max(r['x'] for r in runs), y=min(r['y'] for r in runs),
                defect=max(r['defect'] for r in runs), eps=eps)


def validate_geometry(design):
    depth = design['D']
    leaves = 2 ** depth
    for name in ('leaf_n', 'leaf_p', 'leaf_eta'):
        if len(design[name]) != leaves:
            raise ValueError(f'Invalid {name} count')
    for level in range(depth):
        if len(design['m'][level]) != 2 ** level or len(design['f'][level]) != 2 ** level:
            raise ValueError('Invalid branch count')
        f = np.asarray(design['f'][level])
        if not (np.isfinite(f).all() and (f > 0).all() and (f < 1).all()):
            raise ValueError('Branch fractions must lie strictly between zero and one')
    for n, p, eta in zip(design['leaf_n'], design['leaf_p'], design['leaf_eta']):
        if not len(n) == len(p) == len(eta) == design['K']:
            raise ValueError('Inconsistent wall count')
        if not np.isfinite(n).all() or not np.isfinite(p).all() or not np.isfinite(eta).all():
            raise ValueError('Non-finite wall parameter')
        if (np.linalg.norm(n, axis=1) == 0).any() or (np.linalg.norm(p, axis=1) == 0).any():
            raise ValueError('Wall directions must be nonzero')
        if (np.asarray(eta) < 0).any() or (np.asarray(eta) > 1).any():
            raise ValueError('Wall compliance fraction outside the unit interval')
    for normals in design['m']:
        if not np.isfinite(normals).all() or (np.linalg.norm(normals, axis=1) == 0).any():
            raise ValueError('Invalid lamination normal')
    if design.get('base') == 'seq':
        weights = np.asarray(design['leaf_sw'])
        if weights.shape != (leaves, 13) or not np.isfinite(weights).all() or (weights < 0).any():
            raise ValueError('Invalid sequential-laminate weights')
        if not np.allclose(weights.sum(axis=1), 1, rtol=0, atol=1e-12):
            raise ValueError('Sequential-laminate weights must sum to one')
