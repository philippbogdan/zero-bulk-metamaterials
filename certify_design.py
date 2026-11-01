"""Evaluate a new design independently of its search estimates."""
import argparse
import json
from pathlib import Path
import mpmath as mp
from metamaterials.certify import certify
from metamaterials.evidence import validate_geometry
from metamaterials.zero_mode import dilation_residual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--nu', type=float)
    parser.add_argument('--theta', type=float)
    parser.add_argument('--output', type=Path, default=Path('results/new-certification.json'))
    args = parser.parse_args()
    record = json.loads(args.input.read_text())
    design = record.get('design', record)
    nu = args.nu if args.nu is not None else record.get('nu', design.get('nu'))
    theta = args.theta if args.theta is not None else record.get('theta', design.get('theta'))
    if nu is None or theta is None:
        parser.error('Supply nu and theta, either in the input or as arguments.')
    validate_geometry(design)
    result = certify(design, nu, theta, eps_list=[1e-12, 1e-14], deltas=['1e-50', '1e-60'], dps=90)
    result['design'] = design
    if 'zero_mode_alpha' in design:
        with mp.workdps(90):
            residual = dilation_residual(design, [mp.mpf(a) for a in design['zero_mode_alpha']])
            result['dilation_residual'] = float(mp.sqrt(sum(v * v for v in residual)))
    result['isotropic_at_finest_eps'] = all(r['defect'] <= 1e-3 for r in result['runs'] if r['eps'] == 1e-14)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    for value in result['runs']:
        print(value)
    return 0 if result['isotropic_at_finest_eps'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
