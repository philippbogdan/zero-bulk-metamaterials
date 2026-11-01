"""Repeat the direct voxel simulation of one wall or every stored wall case."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from metamaterials.elastic import iso_tensor, solid_moduli
from metamaterials.evidence import DATA
from metamaterials.geometry import wall_section, extrude
from metamaterials.metrics import exact_laminate
from metamaterials.optional.voxel import homogenize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all', action='store_true', help='Repeat every stored case, including the large herringbone cells.')
    parser.add_argument('--level', choices=['A', 'H'], default='A')
    parser.add_argument('--N', type=int, default=32)
    parser.add_argument('--p2', type=int, default=16)
    parser.add_argument('--p3', type=int, default=32)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--output', type=Path, default=Path('results/finite-scale.json'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    specifications = []
    if args.all:
        for path in sorted((DATA / 'finite_scale').glob('*.json')):
            specifications.extend(json.loads(path.read_text())['cases'])
    else:
        specifications = [dict(level=args.level, N=args.N, p2=args.p2, p3=args.p3)]
    cache = {}

    def solve(level, n, p2=16, p3=32, sign=1):
        key = (level, n, p2, p3, sign)
        if key not in cache:
            section = wall_section(n, p2=p2, p3=p3, level=level, sign=sign)
            result = homogenize(extrude(section), nu=.3, dtype=torch.float64, device=args.device,
                                tol=1e-9, maxiter=20000, batch=1, store_bytes=1e9, work_bytes=3e8)
            if not np.isfinite(result.C).all() or max(result.rel_residual) > 2e-8:
                raise RuntimeError('The FEM solve did not reach the required residual.')
            cache[key] = np.asarray(result.C)
        return cache[key]

    output = dict(nu=.3, P=8, cases=[])
    solid = iso_tensor(*solid_moduli(.3))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for spec in specifications:
        level, n, p2 = spec['level'], spec['N'], spec['p2']
        p3 = spec.get('p3', 32)
        if level == 'A':
            children = [solid, solve('plates', 32)]
            normal = [0, 0, 1]
            ratio = p2 / 8
        else:
            children = [solve('A', max(p2, 32), p2, sign=1), solve('A', max(p2, 32), p2, sign=-1)]
            normal = [1, 0, 0]
            ratio = p3 / p2
        formula = exact_laminate(children, [.5, .5], normal)
        tensor = solve(level, n, p2, p3)
        error = float(np.linalg.norm(tensor - formula) / np.linalg.norm(formula))
        output['cases'].append(dict(level=level, N=n, p2=p2, p3=p3 if level == 'H' else None,
                                    ratio=ratio, relative_error=error, C_fem=tensor.tolist(),
                                    C_formula=formula.tolist(), preceding_tensors=[c.tolist() for c in children]))
        args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
        print(f'{level} N={n} p2={p2} p3={p3}: {100 * error:.6f}% tensor error', flush=True)


if __name__ == '__main__':
    main()
