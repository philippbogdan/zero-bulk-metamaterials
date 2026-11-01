# A compatible dilation with stiff shear

A free wall removes one strain-energy mode while the rest of its base remains stiff. Combining these modes through a laminate tree can produce a macroscopic dilation.

## The geometry

For a base material M, let `lam_n` denote exact lamination with normal n. The rank-two wall is

```math
P_p=\mathrm{lam}_p(M,0;1/2,1/2),\qquad
A(n,p)=\mathrm{lam}_n(M,P_p;1/2,1/2).
```

For nonparallel n and p, its kernel is spanned by $`s=\mathrm{sym}(n\otimes p)`$. To see this, zero energy in a laminate requires zero energy in each phase. The positive-definite M phase must have zero strain; the plate stack permits strains in $`\mathrm{sym}(p\otimes\mathbb R^3)`$. Compatibility across the outer wall also restricts that strain to $`\mathrm{sym}(n\otimes\mathbb R^3)`$. For nonparallel n and p, their intersection is the line spanned by s. The averages of these compatible fields give the wall mode.

When p is parallel to n, use two oppositely tilted rank-two walls, laminated along a tangent t. With $`q_\pm=(n\pm t)/\sqrt2`$, the modes $`s_\pm=\mathrm{sym}(n\otimes q_\pm)`$ have a sum proportional to $`n\otimes n`$ and a difference in $`\mathrm{sym}(t\otimes\mathbb R^3)`$. Thus the rank-three herringbone wall admits free normal opening. The simpler module family uses this construction. The headline designs use nonparallel n and p, so their walls are rank two.

Insert a wall at fraction eps into its leaf base, along normal n. In the free-wall, vanishing-eps limit, the effective tensor is

```math
X=M-\frac{(Ms)(Ms)^T}{s^TMs}.
```

It is positive semidefinite and $Xs=0$. A partly compliant wall retains a fraction eta of this mode by multiplying the subtracted term by $`(1-\eta)`$; the explicit finite-eps geometry adds a small material bridge, fitted by the evaluator. All headline leaves have eta=0.

Each headline has a depth-five binary laminate tree, hence 32 leaves. A path contains five tree laminations, one wall insertion and two wall laminations, above the base's sequential layering. These eight levels require infinite separation in the homogenisation model. No finite spacing is claimed to realise the headline stiffness.

The headline evaluations use wall fraction eps=1e-14. This small value matters: in the simpler module family, eps=1e-4 can still leave normalised shear about 0.015 from its limit, and the partly compliant wall calculations needed eps about 1e-12. Convergence in wall fraction and separation between the internal layer scales are separate requirements. Neither supplies a practical feature-size prescription for a complete structure.

## The zero-mode condition

Assign each leaf the strain $`E_i=\alpha_i s_i`$. At a branch with normal m and fraction f, combine the children as

```math
E=fE_L+(1-f)E_R.
```

A piecewise constant strain is compatible across the flat interface precisely when

```math
E_L-E_R\in\mathrm{sym}(m\otimes\mathbb R^3).
```

Let $`P_m^\perp`$ be the orthogonal projection onto the complement of that subspace. The complete system is

```math
P_m^\perp(E_L-E_R)=0\quad\text{at every branch},\qquad E_{\rm root}=I.
```

For fixed geometry this is linear in the leaf amplitudes. Each leaf strain has zero energy. Inductively, every compatible laminate of its children has zero energy at the mean strain. If the root strain is I, the homogenised dilation energy is zero. Positive semidefiniteness then implies $`C^*I=0`$ and $`\kappa^*=0`$.

```text
leaf free strains
        |
compatible jumps at each branch
        |
weighted mean at the root = I
        |
zero dilation energy
```

A simpler special case is immediate: if all phases share the zero mode I, use the same strain in every phase. Every jump is zero and every laminate preserves that mode.

## What is checked numerically

`metamaterials/zero_mode.py` reconstructs the residual directly from the stored amplitudes, unit directions and branch fractions, independently of the stiffness evaluation. It includes every in-plane compatibility residual and the six root-dilation components. The reported norm is the Euclidean norm of that complete vector, not divided by the number of constraints.

The stored headline residuals range from about 7.3e-16 to 2.1e-13. This finite residual is the effect of the supplied rounded parameters; it is not an exact symbolic certificate. The 90-digit geometry evaluations nevertheless report bulk at float64 round-off and preserve high shear. Both checks are necessary: a tiny projected bulk alone does not establish compatibility.

The formula describes a limit. A finite connected periodic elastic solid with connectivity spanning the cell cannot support an exact dilation at zero energy: zero strain would make its connected solid part rigid, incompatible with a nonzero dilation across the periodic cell. Finite structures therefore need a separate assessment of connectivity, thickness and scale separation. The included wall simulations are such an assessment of individual steps only.
