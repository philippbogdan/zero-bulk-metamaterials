# Why quadratic translations cannot couple this corner

For a solid and void, the quadratic translation method applied to the mixed dilation and shear-complementary energy cannot produce a coupled bound below the HS square. This is a statement about a class of bounds, not a solution of the general stiffness problem.

## The mixed energy and the admissible translation

Let compatible strain epsilon have mean eI. Let five divergence-free stresses sigma_j have means tau_j forming an orthonormal deviatoric basis. The mixed effective energy is

```math
W(e;\tau)=e^2I:C^*I+\sum_{j=1}^5\tau_j:(C^*)^{-1}\tau_j.
```

For singular tensors, complementary energy is understood on the supported stresses, or by the positive regularisation limit. A quadratic translation has the form

```math
T(\varepsilon,\sigma)=\varepsilon:A\varepsilon+
2\sum_j\varepsilon:B_j\sigma_j+\sum_{i,j}\sigma_i:D_{ij}\sigma_j.
```

It must be quasiconvex for compatible epsilon and divergence-free sigma, and the local energy minus T must be nonnegative in both phases. Averaging then gives $`W\geq T(eI,\tau)`$.

## The strain block vanishes

In the void all stresses vanish and strain is unconstrained. The local condition is therefore $`\varepsilon:A\varepsilon\leq0`$ for every symmetric epsilon, so A is negative semidefinite.

A compatible plane wave has strain $`\mathrm{sym}(n\otimes a)`$. Quasiconvexity of a quadratic form requires nonnegativity on every such wave with stresses zero. Thus A is also nonnegative on these strains. A negative semidefinite matrix whose quadratic form vanishes on a vector annihilates that vector. The compatible wave strains span the symmetric matrices: diagonal basis matrices use a=n, and off-diagonal ones use distinct coordinate directions. Consequently A=0.

## The cross blocks are scalar Hill pairings

Fix a wave direction n. A divergence-free stress wave satisfies $`\sigma n=0`$. With A=0, the amplitude of the compatible strain can have either sign and arbitrary magnitude while the stress amplitude is held fixed. Nonnegativity of T on this wave forces

```math
\mathrm{sym}(n\otimes a):B_j\sigma=0
\quad\text{for every a whenever }\sigma n=0.
```

Equivalently $`(B_j\sigma)n=0`$ whenever $`\sigma n=0`$. A useful elementary linear-algebra lemma is that a linear map on symmetric matrices with this property is a scalar multiple of the identity. For a rank-one matrix $tt^T$, every n perpendicular to t is in its kernel, so the image must be proportional to $tt^T$. Apply this to the coordinate vectors and their sums and differences. Linearity forces the proportionality constants to agree and sends each symmetric off-diagonal basis matrix to that same multiple of itself. Hence $`B_j=b_jI`$.

The cross terms are therefore $`2b_j\varepsilon:\sigma_j`$. Hill's integration-by-parts identity gives

```math
\langle\varepsilon:\sigma_j\rangle=\langle\varepsilon\rangle:\langle\sigma_j\rangle=eI:\tau_j=0.
```

Thus the translation evaluated at the means contains no dilation term and no strain-stress coupling: $`T(eI,\tau)=\sum_{i,j}\tau_i:D_{ij}\tau_j`$.

## Why the remaining bound cannot improve the corner

The remaining right-hand side is independent of e. At e=0, test its validity on the isotropic HS sequential laminate, which attains $`\mu^*=\mu_{\rm HS}`$ at the prescribed fraction. Every admissible right-hand side is therefore at most $`5/(2\mu_{\rm HS})`$ for the orthonormal shear stresses. For an isotropic effective tensor,

```math
W(e;\tau)=9e^2\kappa^*+\frac5{2\mu^*}.
```

All these inequalities are already satisfied when $`\kappa^*\geq0`$ and $`\mu^*\leq\mu_{\rm HS}`$. They add no coupled restriction below the box. Together with the separate HS bulk bound, the general outer region remains the same square.

This argument concerns quadratic local translations of this mixed energy. It does not rule out non-quadratic constraints, nonlocal arguments or a construction reaching the corner. An analogous coupled-reference HS argument is only a sketch in the underlying analysis and is not asserted as an additional theorem here.
