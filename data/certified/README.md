# The stored laminate geometries

| Family | Records | Meaning |
|---|---|---|
| `zero_bulk_aniso` | 14 | Headline zero-bulk trees, a separate anisotropic base in each leaf |
| `zero_bulk_iso` | 16 | Zero-bulk trees on an isotropic HS base, including solid-base limits |
| `zero_bulk_module` | 16 | Simpler zero-bulk module polycrystals |
| `tree_aniso` | 238 | Trees with per-leaf anisotropic bases and varied bulk/shear tradeoffs |
| `tree_iso` | 272 | Trees on an isotropic base with varied bulk/shear tradeoffs |

Each JSON file contains `family`, base Poisson ratio `nu`, nominal solid fraction `theta`, optional support slope `c`, explicit `design`, stored `runs` and the vanishing-wall `limit`. A run records `eps`, `delta`, `theta_total`, normalised `x` and `y`, `defect`, `kappa` and `mu`. The limit additionally contains the effective Mandel tensor `C`.

The design specifies tree depth `D`, walls per leaf `K`, base type, leaf wall normals `leaf_n`, free-mode directions `leaf_p`, retained mode fractions `leaf_eta`, branch normals `m` and fractions `f`. Each branch fraction weights its left child. Levels are listed from root to leaves; children are paired consecutively when evaluating upwards. A `seq` base additionally uses each leaf's `leaf_sw` weights in the order returned by `metamaterials.directions.seq_dirs`. Free-wall zero-bulk trees store `zero_mode_alpha`, the leaf amplitudes giving compatible dilation. Some inherited design fields are unused for a particular base; `leaf_q` is only relevant to plate bases.

Use the smallest eps and the least favourable coordinates over deltas, then apply defect <= 0.001. Exactly one stored design fails that rule: `tree_iso_nu0.45_th0.1_c3.0.json`, with finest-wall defect about 0.001133. It remains here for transparency but is excluded from support calculations. Coarser runs can have larger defects and are preserved as numerical evidence.

At theta=1 the base is the solid, while the explicit wall geometry has actual fraction 1-O(eps). Read `theta_total` rather than treating that nominal value as an exact finite-eps fraction. The headline set only uses theta below one.

The files preserve the supplied numerical values. High-precision evaluation refers to the lamination arithmetic, not to high-precision storage of the parameters or an interval proof. [Precision and selection rules](../../docs/methods.md).
