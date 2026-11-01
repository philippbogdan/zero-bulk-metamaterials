# What finite scale costs

The individual wall constructions were checked by periodic voxel finite elements against the exact laminate formula. The comparisons test scale separation, not the full headline materials.

For voxel indices i=x1 and k=x3, independent of x2, the solid masks are:

```text
P+ : ((i + k) mod P) < P/2
P- : ((i - k) mod P) < P/2
A+ : P+ or ((k mod p2) < p2/2)
A- : P- or ((k mod p2) < p2/2)
H  : A+ if ((i mod p3) < p3/2), otherwise A-
```

Here P=8 and the solid has E=1, nu_s=0.3. The arrays are extruded periodically along x2. The wall image uses the exact H mask with N=128, p2=16 and p3=32: plate period 8, layering period 16 and orientation-flip period 32 voxels. H is the rank-three wall used in the simpler module family; the headline designs use rank-two walls with their own nonparallel directions.

The left view contains the full 128 by 128 section extruded 24 voxels along x2. The upper-right quarter of the front half is cut away: i in [64,128), x2 in [0,12), k in [64,128). The right view is the exact 48 by 32 sub-array i in [0,48), k in [0,32), with the same extrusion, shown at 2.15 times the left view's magnification. These are display cuts of the simulated geometry, not changes to the simulated periodic cell.

Deep blue and copper distinguish opposite plate orientations of one solid; the solid layers are light grey. Every voxel retains the same dimensions. The linked six-second video moves the camera around the unchanged geometry, with azimuth going from 32 to -32 degrees and back. The poster corresponds to its opening view. No deformation field, enlarged channel or extra scale separation is implied.

## Compare one level at a time

For A, compute `lam_e3(solid, FEM(P+); 1/2,1/2)`. For H, compute `lam_e1(FEM(A+), FEM(A-); 1/2,1/2)`. The relative error is

```math
\frac{\|C_{\rm FEM}-C_{\rm formula}\|_F}{\|C_{\rm formula}\|_F}.
```

The stored A+ tensors suffice to reconstruct A-: reflection of x1, followed by a periodic translation by P/2-1 voxels, maps the plus mask to the minus mask. The A layer mask does not change. Therefore reflect the elasticity tensor with `diag(-1,1,1)`; translations leave the homogenised tensor unchanged. This is how the lightweight checker obtains the preceding-level minus tensor. Its resulting H comparison tensors agree with the separately stored formulas to round-off. The full FEM command recomputes both signs directly.

| Wall | Scale ratio | Relative tensor error |
|---|---|---|
| Rank two, p2/P | 2, 4, 8 | 6.296%, 2.621%, 1.277% |
| Herringbone, p2=16, p3/p2 | 2, 4, 8, 16 | 15.490%, 7.779%, 3.848%, 1.914% |
| Herringbone, p2=32, p3/p2 | 2, 4, 8 | 15.045%, 7.695%, 3.878% |

The ratio-four, p2=16 case occurs at both N=128 and N=256, with the same 7.77898% error to the displayed precision. Refining that same geometry does not remove its scale-separation error. The convergence figure includes both observations at the overlapping location, as a filled circle and a larger hollow ring. Thin straight segments join consecutive measurements; they are not fits. The dashed inverse-ratio line is a slope guide with arbitrary normalisation, placed above the data.

The error falls approximately as the reciprocal scale ratio. Using about 1/ratio as an error estimate for each level is useful for intuition, but it is not a rigorous estimate for a complete tree. Errors from repeated levels cannot simply be assumed to cancel or to add independently.

## Solver and scope

The solver uses one periodic trilinear Q1 hexahedron per voxel, exact element integration by tensor-product two-point Gauss quadrature, float64 arithmetic and a conjugate-gradient relative residual target of 1e-9. Solid voxels carry isotropic elasticity. Void voxels have zero stiffness, not a weak substitute. Floating disconnected components contribute no macroscopic elastic energy and are removed. Periodic translations are projected out of the solve.

The returned effective tensor uses the symmetric energy form of the six Mandel load solutions. Its discretisation error and scale-separation error are distinct from the much smaller solver residual. The formula comparisons use mpmath at 40 digits and regularisation 1e-30, then return float64 tensors.

Not simulated here: the complete eight-level trees, the sequential-laminate base, or wall fractions of 1e-14. Their reported properties follow from reiterated exact lamination in the infinite-separation limit. The wall test does not establish printability, finite-size performance or a practical tolerance budget for the full structures.
