"""Recompute the published numbers and fail if any comparison does not match."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time
import mpmath as mp
import numpy as np

from metamaterials.certify import certify
from metamaterials.evidence import DATA, read_designs, validate_geometry
from metamaterials.metrics import corner_gaps, finite_scale_table
from metamaterials.zero_mode import dilation_residual

ROOT = Path(__file__).resolve().parent


def compare(label, actual, expected, atol, rtol=0.):
    matched = bool(np.isfinite(actual) and np.isfinite(expected)
                   and abs(actual - expected) <= atol + rtol * abs(expected))
    return dict(quantity=label, actual=float(actual), expected=float(expected),
                absolute_tolerance=atol, relative_tolerance=rtol, passed=matched)


def evaluate_record(payload):
    record, all_runs = payload
    validate_geometry(record['design'])
    selected = record['runs'] if all_runs else [
        r for r in record['runs'] if r['eps'] == 1e-14 and r['delta'] == '1e-60']
    if not selected:
        raise ValueError(f'No requested reference evaluation in {record["file"]}')
    results, checks = [], []
    for expected in selected:
        result = certify(record['design'], record['nu'], record['theta'],
                         eps_list=[expected['eps']], deltas=[expected['delta']], dps=90)
        value = result['runs'][0]
        for key, tolerance in [('x', 2e-11), ('y', 2e-9), ('defect', 2e-9),
                               ('theta_total', 2e-14), ('kappa', 2e-11), ('mu', 2e-10)]:
            checks.append(compare(key, value[key], expected[key], tolerance))
        if record['family'] == 'zero_bulk_aniso':
            checks.append(dict(quantity='isotropy threshold', actual=value['defect'],
                               maximum=1e-3, passed=bool(value['defect'] <= 1e-3)))
        for key in ('x', 'y', 'defect'):
            checks.append(compare('limit ' + key, result['limit'][key], record['limit'][key], 2e-9))
        results.append(result)
    residual = None
    if 'zero_mode_alpha' in record['design']:
        with mp.workdps(90):
            r = dilation_residual(record['design'], [mp.mpf(a) for a in record['design']['zero_mode_alpha']])
            residual = float(mp.sqrt(sum(a * a for a in r)))
        checks.append(dict(quantity='compatible dilation residual', actual=residual,
                           maximum=5e-13, passed=bool(np.isfinite(residual) and residual <= 5e-13)))
    if record['family'].startswith('zero_bulk'):
        for value in (r['runs'][0] for r in results):
            checks.append(compare('zero bulk x', value['x'], 0., 1e-12))
    return dict(file=record['file'], family=record['family'], nu=record['nu'], theta=record['theta'],
                evaluations=results, dilation_residual=residual, checks=checks,
                passed=all(c['passed'] for c in checks))


def quick_checks(records):
    checks = [compare('stored design count', len(records), 556, 0),
              compare('headline design count', sum(r['family'] == 'zero_bulk_aniso' for r in records), 14, 0)]
    manifest = json.loads((DATA / 'manifest.json').read_text())
    for name, expected_hash in manifest['files'].items():
        path = DATA / name
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        checks.append(dict(quantity='data fingerprint: ' + name, actual=actual_hash,
                           expected=expected_hash, passed=actual_hash == expected_hash))
    for record in records:
        validate_geometry(record['design'])
    from metamaterials.evidence import finest_point
    excluded = [finest_point(r) for r in records if finest_point(r)['defect'] > 1e-3]
    checks.append(compare('accepted stored design count', len(records) - len(excluded), 555, 0))
    reference = json.loads((DATA / 'reference.json').read_text())
    gaps = corner_gaps(records)
    for row, expected in zip(gaps, reference['gaps']):
        if (row['nu'], row['theta']) != (expected['nu'], expected['theta']):
            raise ValueError('Reference case order mismatch')
        for key in ('before', 'after'):
            checks.append(compare(f'gap {key}, nu={row["nu"]}, theta={row["theta"]}',
                                  row[key], expected[key], 2e-10))
        checks.append(compare(f'gap angle, nu={row["nu"]}, theta={row["theta"]}',
                              row['phi_after'], expected['phi_after'], 0))
    finite = finite_scale_table()
    checks.append(compare('finite-scale comparison count', len(finite), 11, 0))
    for row in finite:
        identity = f'{row["file"]}, p2={row["p2"]}, p3={row["p3"]}, N={row["N"]}'
        for actual, expected in [('relative_error', 'expected_error'), ('mode_fem', 'expected_mode_fem'),
                                  ('mode_formula', 'expected_mode_formula')]:
            checks.append(compare(f'{actual}: {identity}', row[actual], row[expected], 2e-10))
        if row['formula_tensor_difference'] is not None:
            checks.append(compare(f'formula tensor: {identity}', row['formula_tensor_difference'], 0., 2e-10))
    headline = [r for r in records if r['family'] == 'zero_bulk_aniso']
    summaries = []
    for record in headline:
        fine = next(r for r in record['runs'] if r['eps'] == 1e-14 and r['delta'] == '1e-60')
        coarse = next(r for r in record['runs'] if r['eps'] == 1e-12 and r['delta'] == '1e-60')
        with mp.workdps(90):
            residual = dilation_residual(record['design'], [mp.mpf(a) for a in record['design']['zero_mode_alpha']])
            norm = float(mp.sqrt(sum(r * r for r in residual)))
        summaries.append(dict(file=record['file'], x=fine['x'], y=fine['y'], defect=fine['defect'],
                              dilation_residual=norm, eps_difference=abs(coarse['y'] - fine['y']),
                              limit_difference=abs(fine['y'] - record['limit']['y'])))
        for name, value, maximum in [('dilation residual', norm, 5e-13),
                                      ('eps convergence', abs(coarse['y'] - fine['y']), 1.5e-7),
                                      ('limit convergence', abs(fine['y'] - record['limit']['y']), 1.5e-9)]:
            checks.append(dict(quantity=name + ': ' + record['file'], actual=value, maximum=maximum,
                               passed=bool(np.isfinite(value) and value <= maximum)))
    return dict(gaps=gaps, finite_scale=finite, excluded=excluded, stored_headline=summaries,
                checks=checks, passed=all(c['passed'] for c in checks))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all', action='store_true', help='Re-certify every stored eps and delta for all designs; allow hours.')
    parser.add_argument('--stored-only', action='store_true', help='Check gaps and FEM comparisons only; skip geometry certification.')
    parser.add_argument('--case', metavar='FILENAME', help='Re-certify one stored file instead of the headline set.')
    parser.add_argument('--processes', type=int, default=1, help='Concurrent geometry evaluations; default: one.')
    parser.add_argument('--output', type=Path, default=ROOT / 'results' / 'reproduction.json')
    args = parser.parse_args(argv)
    if args.processes < 1 or sum([args.all, args.stored_only, bool(args.case)]) > 1:
        parser.error('Choose at most one of --all, --stored-only and --case; processes must be positive.')
    started = time.monotonic()
    output = dict(scope='all' if args.all else 'stored-only' if args.stored_only else 'single-case' if args.case else 'headline',
                  precision_digits=90, geometries=[], errors=[], passed=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')

    save()
    try:
        records = read_designs()
        output['stored_checks'] = quick_checks(records)
        for row in output['stored_checks']['gaps']:
            print(f'nu={row["nu"]} theta={row["theta"]}: gap {row["before"]:.8f} -> {row["after"]:.8f} at {row["phi_after"]} degrees', flush=True)
        for row in output['stored_checks']['finite_scale']:
            print(f'{row["level"]} p2={row["p2"]} p3={row["p3"]} N={row["N"]}: error {100 * row["relative_error"]:.6f}%', flush=True)
        chosen = [] if args.stored_only else [r for r in records if args.all or
                 (r['file'] == args.case if args.case else r['family'] == 'zero_bulk_aniso')]
        if not args.stored_only and not chosen:
            raise ValueError('The requested design was not found')
        payloads = [(r, args.all or bool(args.case)) for r in chosen]
        pool = ProcessPoolExecutor(max_workers=args.processes) if args.processes > 1 else None
        try:
            values = pool.map(evaluate_record, payloads) if pool else map(evaluate_record, payloads)
            for result in values:
                output['geometries'].append(result)
                value = result['evaluations'][0]['runs'][0]
                print(f'{result["file"]}: x={value["x"]:.3e} y={value["y"]:.10f} '
                      f'defect={value["defect"]:.6e} theta={value["theta_total"]:.16g} '
                      f'dilation={result["dilation_residual"]} {"PASS" if result["passed"] else "FAIL"}', flush=True)
                save()
        finally:
            if pool:
                pool.shutdown()
        if output['scope'] == 'headline' and len(output['geometries']) == 14:
            values = [r['evaluations'][0]['runs'][0] for r in output['geometries']]
            residuals = [r['dilation_residual'] for r in output['geometries']]
            gaps = output['stored_checks']['gaps']
            output['headline_summary'] = dict(
                shear_fraction_min=min(v['y'] for v in values),
                shear_fraction_max=max(v['y'] for v in values),
                maximum_absolute_bulk_coordinate=max(abs(v['x']) for v in values),
                maximum_isotropy_defect=max(v['defect'] for v in values),
                maximum_dilation_residual=max(residuals),
                baseline_gap_range=[min(g['before'] for g in gaps), max(g['before'] for g in gaps)],
                augmented_gap_range=[min(g['after'] for g in gaps), max(g['after'] for g in gaps)],
                solid_fractions=sorted({r['theta'] for r in output['geometries']}),
                base_poisson_ratios=sorted({r['nu'] for r in output['geometries']}),
                leaves=sorted({2 ** r['design']['D'] for r in chosen}),
                lamination_levels_above_base=sorted({r['design']['D'] + 3 for r in chosen}))
            print('Headline summary:', json.dumps(output['headline_summary'], sort_keys=True), flush=True)
        output['passed'] = output['stored_checks']['passed'] and all(r['passed'] for r in output['geometries'])
    except Exception as error:
        output['errors'].append(f'{type(error).__name__}: {error}')
    output['elapsed_seconds'] = time.monotonic() - started
    save()
    if output['errors']:
        print('\n'.join(output['errors']), file=sys.stderr)
    if not output['passed']:
        for check in output.get('stored_checks', {}).get('checks', []):
            if not check['passed']:
                print('FAIL', check, file=sys.stderr)
    print(f'{"PASS" if output["passed"] else "FAIL"}: {output["scope"]}; results in {args.output}')
    return 0 if output['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
