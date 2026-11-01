# Read the stiffness map correctly

The solid has Young's modulus E=1 and Poisson ratio nu_s. Its bulk and shear moduli are

$$\kappa_s=\frac{1}{3(1-2\nu_s)},\qquad \mu_s=\frac{1}{2(1+\nu_s)}.$$

An effective elasticity tensor is stored as a symmetric matrix in Mandel order
`(11,22,33,sqrt(2)23,sqrt(2)13,sqrt(2)12)`. Let
$v=(1,1,1,0,0,0)^T/\sqrt3$, $P_h=vv^T$ and $P_s=I_6-P_h$. Then

$$\kappa^*=\frac{v^TC^*v}{3},\quad
\mu^*=\frac{\operatorname{tr}C^*-v^TC^*v}{10},\quad
C_{\rm iso}=3\kappa^*P_h+2\mu^*P_s.$$

The isotropy defect is $\|C^*-C_{\rm iso}\|_F/\|C^*\|_F$. The accepted threshold is 0.001. The reported moduli are those of the isotropic projection even when the defect is small but nonzero.

## Normalisation

For total solid fraction theta and void, write

$$z_\kappa=\frac{4\mu_s}{3},\qquad
z_\mu=\frac{\mu_s(9\kappa_s+8\mu_s)}{6(\kappa_s+2\mu_s)}.$$

The Hashin-Shtrikman bounds are

$$\kappa_{\rm HS}=\frac{\theta\kappa_s z_\kappa}{z_\kappa+(1-\theta)\kappa_s},\qquad
\mu_{\rm HS}=\frac{\theta\mu_s z_\mu}{z_\mu+(1-\theta)\mu_s}.$$

All map coordinates are dimensionless: $x=\kappa^*/\kappa_{\rm HS}$ and $y=\mu^*/\mu_{\rm HS}$. This normalisation compares materials with the same solid fraction and base solid. The bounds apply to the isotropic projection, not to each directional stiffness separately.

## Exact laminate formula

Rotate the layer normal to e3. Split the Mandel indices into in-plane strain components $P=(11,22,12)$ and traction components $N=(33,23,13)$. For nonsingular $C_{NN}$,

$$H(C)=\begin{pmatrix}
C_{PP}-C_{PN}C_{NN}^{-1}C_{NP}&C_{PN}C_{NN}^{-1}\\
-C_{NN}^{-1}C_{NP}&C_{NN}^{-1}
\end{pmatrix}.$$

Continuity of in-plane strain and traction gives $H(C^*)=\sum_i f_iH(C_i)$. Invert the partial transform and rotate back. This is an exact homogenisation formula for separated layers. Void and singular wall tensors are regularised by adding delta times the Mandel identity at each lamination call.

The base with void core and solid coating is

$$C_B=C_s+(1-\theta_B)\left(\theta_B\sum_jw_j\Gamma_s(n_j)-C_s^{-1}\right)^{-1},$$

where $\Gamma_s(n)$ is defined in the [bound proof](bounds.md). Equal weights on the six icosahedral axes give the isotropic HS point. The headline bases use different nonnegative weights in each leaf on the fixed set of six icosahedral, three cubic and four octahedral axes.

## Precision and the meaning of certification

Stored directions, fractions, weights and zero-mode amplitudes are binary64 numbers. The evaluator forms the base tensors and wall-direction normalisations in float64, then performs partial inversion, wall assembly and tree lamination with mpmath at 90 decimal digits. It finally projects the tensor in float64, matching the reference convention. Cancellation can therefore give a tiny negative reported bulk modulus. Such a value is round-off, not negative elastic energy.

For the headlines, wall fractions of 1e-12 and 1e-14 differ in normalised shear by at most 1.5e-7. The finest evaluation differs from the wall-fraction limit by at most 1.5e-9. The main check re-evaluates the finest geometry and the limit; it checks the coarser difference from stored runs. `--all` also recomputes the coarser runs. The headline data use delta=1e-60; the isotropic-base records include the regularisation comparison at delta=1e-50 and 1e-60.

These checks establish numerical stability of the specified laminate calculation. They do not prove global optimality, exact isotropy, an interval enclosure or physical manufacturability. In particular, “exact zero bulk” refers to the compatible-dilation construction before rounding; the supplied rounded designs have small, measured residuals.

The complete headline run in [results/reproduction.json](../results/reproduction.json) re-evaluated all 14 geometries at 90 digits and passed every comparison. Its largest absolute bulk coordinate is 2.84e-17, largest isotropy defect 0.0009451 and largest compatible-dilation residual 2.06e-13. The other 13 dilation residuals are about 1e-15. The tiny bulk values differ slightly from the stored reference values because the final projection is in float64; neither set is an interval bound.

## Display conventions

The stiffness map shows all 555 accepted records, including the solid-base limits, and leaves out the one record above the isotropy threshold. For display only, bulk coordinates with magnitude below 1e-12 are placed at zero. This affects the 46 zero-bulk records, including 17 tiny negative values; their largest absolute stored coordinate is 2.56e-17. The underlying values are unchanged, and overlapping points are not displaced. The solid-fraction colour key is ordinal: equal-sized colour blocks carry the exact tested fractions, not a continuous interpolation.

The outer square is a proved bound. Neither its interior nor the convex hull of the plotted points is shown as an established set of attainable isotropic materials.

## The corner gap

For $u_\phi=(\cos\phi,\sin\phi)$,

$$h_{\rm att}(\phi)=\max_{p\in\mathcal P}u_\phi\cdot p,\qquad
h_{\rm box}(\phi)=\max(0,\cos\phi)+\max(0,\sin\phi).$$

The reported corner gap is the maximum of $h_{\rm box}-h_{\rm att}$ over integer angles from 91 to 179 degrees. It is a sampled support-function comparison, not a continuous-angle Hausdorff distance, a volume fraction or a theorem that all convex combinations are realised by isotropic composites.

For each design, the finest stored eps is selected. Across deltas at that eps, the checker uses the largest x, smallest y and largest defect. Since cos(phi) is negative and sin(phi) positive here, this is conservative for support. The resulting pair is a conservative numerical estimate, not a newly evaluated tensor. All stored records remain available, including the rejected one.

The baseline uses the HS laminate, the origin as a closure point, disconnected-grain additions to lower-fraction HS laminates, and a slip-wall construction at eps=1e-8. Dust body fractions are theta times six logarithmic samples from 1e-6 to 0.1 and 90 linear samples from 0.1 up to but excluding 1. The hero map's baseline uses the limiting slip point (1,0); this point cannot improve support in the upper-left quadrant because the origin dominates it. The checker explicitly evaluates the finite-eps slip construction. Neither choice changes the reported corner gaps at the quoted precision.
