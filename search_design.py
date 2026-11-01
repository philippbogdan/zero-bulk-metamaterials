"""Search one solid fraction for a new free-wall laminate, then project its dilation mode."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from metamaterials.optional.optimise import run
from metamaterials.optional.project import residual, project, refine_mp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nu', type=float, default=.3)
    parser.add_argument('--theta', type=float, default=.1)
    parser.add_argument('--depth', type=int, default=5)
    parser.add_argument('--restarts', type=int, default=8)
    parser.add_argument('--iterations', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=Path, default=Path('results/new-design.json'))
    args = parser.parse_args()
    if not (0 < args.theta < 1 and -1 < args.nu < .5 and 1 <= args.depth <= 8
            and args.restarts >= 1 and args.iterations >= 1):
        parser.error('Use physical material parameters, positive search sizes and depth from one to eight.')
    torch.set_num_threads(1)

    def penalty(parameters, progress):
        return 10 ** min(7, 1 + 8 * progress) * residual(parameters, args.depth)

    penalty.residual = lambda parameters: residual(parameters, args.depth)
    candidates = run(args.nu, 0., D=args.depth, R=args.restarts, iters=args.iterations,
                     seed=args.seed, base='seq', eta_fixed=0., theta=args.theta,
                     extra_pen=penalty, device='cpu', log=False)
    candidates = [c for c in candidates if np.isfinite(c['y_rel']) and np.isfinite(c['defect'])]
    if not candidates:
        raise RuntimeError('The search returned no finite candidate')
    candidate = min(candidates, key=lambda c: (c['defect'] > 1e-3, c['zero_mode_residual'] > 1e-8, -c['y_rel']))
    design, _ = project(candidate)
    design, history = refine_mp(design, dps=60)
    keep = ('D', 'K', 'base', 'nuB', 'c', 'leaf_n', 'leaf_p', 'leaf_eta',
            'leaf_q', 'leaf_sw', 'theta', 'm', 'f', 'zero_mode_alpha')
    design = {k: design[k] for k in keep}
    output = dict(nu=args.nu, theta=args.theta, design=design, seed=args.seed,
                  search_iterations=args.iterations, search_restarts=args.restarts,
                  dilation_residual=history[-1], certified=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    print(f'Candidate written to {args.output}; certify it before using its stiffness.')
    print(f'Compatible-dilation residual: {history[-1]:.6e}')


if __name__ == '__main__':
    main()
