# Reproduce the evidence

Use Python 3.11 or later. From this directory:

```sh
python -m pip install -r requirements.txt
python reproduce.py
```

The command reads only the included data. It prints every case, writes `results/reproduction.json`, and exits nonzero if any comparison fails. The recorded run of `python reproduce.py --processes 8` passed in about 5 seconds, including all 14 headline geometries at 90 digits. Its complete output is included in [results/reproduction.json](results/reproduction.json). The default `python reproduce.py` uses one process and passed end to end from a fresh clone in 20.5 seconds wall time on a desktop Apple silicon CPU, with peak memory about 70 MB. It reproduced the same results as the recorded 8-process run: all 14 geometries at 90 digits, all gaps and finite-scale comparisons. Runtime depends on the machine.

The default check:

1. Reads all 556 records and validates their geometry. Uses the finest wall fraction, the largest bulk coordinate, smallest shear coordinate and largest isotropy defect across stored void regularisations. Excludes points with defect above 0.001.
2. Recomputes the baseline and augmented corner gap for every headline case. The baseline contains the exact HS laminate, a fixed dust sampling and an explicitly evaluated slip-wall geometry.
3. Reapplies the exact laminate formula to the saved FEM tensors of the preceding level, checking all 11 finite-scale comparisons and their free-mode stiffnesses.
4. Re-evaluates the 14 headline geometries at wall fraction `eps=1e-14`, void regularisation `delta=1e-60` and 90 decimal digits. Compares bulk, shear, isotropy defect, total solid fraction and the wall-fraction limit with each stored evaluation. Independently checks compatible dilation from the stored amplitudes.

The lamination operations use mpmath. The base sequential tensors, unit wall directions and stored geometric parameters retain the float64 conventions of the reference calculation. Thus “90 digits” describes the arithmetic of the singular lamination calculation, not 90 correct digits in the final material properties. This is numerical re-evaluation, not interval arithmetic or a global optimality certificate. See [methods](docs/methods.md).

The output records the scope, every actual and expected value, each tolerance, exclusions and failures. The main comparison tolerances are absolute: 2e-9 for normalised shear and isotropy defect, 2e-11 for normalised bulk, 2e-14 for total fraction, and 2e-10 for gaps and finite-scale errors. Headline bulk must also be within 1e-12 of zero, isotropy defect at most 1e-3, and the dilation residual at most 5e-13. These tolerances accommodate float64 base algebra without relaxing the reported isotropy threshold. The full run's largest dilation residual is about 2.06e-13, for `zero_bulk_aniso_nu0.45_th0.7.json`; the other 13 are about 1e-15.

## Short checks and the complete dataset

```sh
python reproduce.py --stored-only --output results/stored-check.json
python -m unittest discover -s tests
python reproduce.py --case zero_bulk_aniso_nu0.3_th0.1.json
python reproduce.py --all --processes 2 --output results/all-designs.json
```

`--stored-only` checks stored evidence and dilation residuals in seconds. It does **not** re-certify stiffness from geometry. `--case` checks all recorded eps and delta values for one file. `--all` re-evaluates every recorded eps and delta for every design, including the rejected record and the solid-base limits. Allow hours. A rejected point passes a reproduction check if its rejected value is reproduced; it never becomes an accepted point merely because its arithmetic matches. Exact headline membership is the `zero_bulk_aniso` family, not a filename substring chosen by the user.

The small test suite checks closed-form limits, rotation and projection identities, compatibility sensitivity, stored-tensor comparisons and failure detection. It is separate from the full high-precision evidence check.

## From a new design to certification

PyTorch is optional and never imported by the headline path:

```sh
python -m pip install -r requirements-optional.txt
python search_design.py --nu 0.3 --theta 0.1 --seed 0 --output results/new-design.json
python certify_design.py results/new-design.json --output results/new-certification.json
```

The search optimises directions, branch fractions and per-leaf base weights, penalises anisotropy and incompatibility, then projects the compatible-dilation equations. This command operates on one case on the CPU. It is a reproducible route to new candidates, not a promise to rediscover the stored optima or their exact search trajectories. Do not quote its float64 search estimates as certified results. Projection can reduce shear or spoil isotropy; the separate certification command reports that outcome and exits nonzero if the finest evaluations fail the isotropy threshold. Search and projection may take substantial time.

To certify a custom design directly, provide either the stored record format or a design object plus `--nu` and `--theta`. The command checks both eps values and both `delta=1e-50` and `delta=1e-60` at 90 digits.

## Repeat the finite-scale simulation

```sh
python finite_scale.py --level A --N 32 --p2 16
python finite_scale.py --all --device cuda --output results/finite-scale-fresh.json
```

`--all` repeats precisely the stored case specifications, using the binary geometry in `metamaterials/geometry.py`. It also recomputes the preceding-level tensors. The solver uses float64 periodic Q1 hexahedra, a conjugate-gradient residual target of 1e-9 and true zero-stiffness voids. Large cells are much faster on one GPU; the full calculation can require substantial memory and hours. `--device cpu` is supported. The stored-evidence command needs neither a GPU nor PyTorch.

This experiment checks individual wall steps. It does not simulate the full tree at finite scale. Increasing the scale ratio and increasing the resolution are different experiments; the stored duplicate at ratio four checks the latter distinction.

## Figures

The figure scripts and their inputs are included. From the repository root:

```sh
python -m pip install -r requirements-figures.txt
python figures/scripts/make_data.py
MPLBACKEND=Agg python figures/scripts/figures.py
```

Table generation reads only `data/` and takes seconds. It writes `figures/data/`; the plotting command replaces the three PNG and SVG charts in `figures/`. The [figure instructions](figures/scripts/README.md) give the slower ray-tracing commands for `wall.png` and all orbit frames, followed by the exact FFmpeg command for `wall.mp4`: H.264, yuv420p, 24 frames per second and faststart. FFmpeg with libx264 must be installed separately. These figure dependencies are optional for the headline numerical check.

The opening stiffness map uses the accepted records in [data/certified](data/certified/README.md) and the baseline construction formulas. The gap and convergence plots use the same quantities checked by `reproduce.py`. [Methods](docs/methods.md) records the map's selection and display conventions.

The wall image uses the exact binary herringbone mask: a cutaway of the full section beside a magnified crop, with colour identifying orientation within one solid. The video changes only the camera. [Finite-scale geometry](docs/finite-scale.md) gives the mask, crop and cutaway coordinates; no channel widths or layer thicknesses are exaggerated.
