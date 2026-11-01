# The HS square bounds the isotropic projection

The bound applies even when the effective material is anisotropic. It limits the projected bulk and shear moduli, not every directional modulus. The argument uses periodic elasticity and extends to separated-scale limits by closure.

## Setting

Let the unit-volume periodic cell have solid indicator chi, with mean theta in (0,1). The local tensor is $C(x)=\chi(x)C_s$, where
$C_s=3\kappa_sP_h+2\mu_sP_s$ is positive definite. For a prescribed mean strain $\varepsilon_0$,

$$\varepsilon_0:C^*\varepsilon_0=\inf_{u\ {\rm periodic}}
\langle(\varepsilon_0+e(u)):C(\varepsilon_0+e(u))\rangle,\qquad e(u)=\operatorname{sym}\nabla u.$$

The admissible displacements are periodic H1 fields. The tensor $C^*$ is positive semidefinite, so both projected moduli are nonnegative.

For a unit vector n define the acoustic tensor and the strain Green operator by

$$K(n)=\mu_sI+(\kappa_s+\mu_s/3)n\otimes n,\qquad
\tau:\Gamma_s(n)\tau=(\tau n)\cdot K(n)^{-1}(\tau n).$$

Completing the square in
$(C_s^{-1}\tau-\operatorname{sym}(n\otimes a)):C_s(C_s^{-1}\tau-\operatorname{sym}(n\otimes a))$
and choosing $a=K(n)^{-1}\tau n$ gives
$0\preceq\Gamma_s(n)\preceq C_s^{-1}$.

## A polarisation trial field

For any symmetric field q vanishing in the solid, completing the square in the void gives

$$(1-\chi)\varepsilon:C_s\varepsilon\geq2q:\varepsilon-q:C_s^{-1}q.$$

Subtract this inequality from the full-solid energy and minimise over periodic u. The zero-mean Fourier modes minimise independently. For reciprocal vector k, put $n=k/|k|$; the minimum of the quadratic displacement term is
$-\widehat q(k)^*:\Gamma_s(n)\widehat q(k)$. The acoustic inverse is uniformly bounded, so this minimisation is attained in the appropriate H1 class.

Take $q=(1-\chi)p$ with constant symmetric p. Parseval gives
$\sum_{k\ne0}|\widehat\chi(k)|^2=\theta(1-\theta)$. Define

$$M=\sum_{k\ne0}\frac{|\widehat\chi(k)|^2}{\theta(1-\theta)}\Gamma_s(k/|k|),\qquad
A=C_s^{-1}-\theta M.$$

Thus $A\succeq(1-\theta)C_s^{-1}\succ0$, and the trial-field inequality is

$$\varepsilon_0:C^*\varepsilon_0\leq
\varepsilon_0:C_s\varepsilon_0-2(1-\theta)p:\varepsilon_0+(1-\theta)p:Ap.$$

Minimising over p yields, with $D=C_s-C^*$,

$$D\succeq(1-\theta)A^{-1}\succ0,\qquad
(1-\theta)D^{-1}\preceq C_s^{-1}-\theta M.$$

## Rotation-invariant traces

The two contractions of $\Gamma_s(n)$ are independent of n:

$$g_h=\operatorname{tr}(P_h\Gamma_s)=\frac{1}{3\kappa_s+4\mu_s},\qquad
g_s=\operatorname{tr}(P_s\Gamma_s)=\frac{3(\kappa_s+2\mu_s)}{\mu_s(3\kappa_s+4\mu_s)}.$$

They follow by inserting the acoustic tensor above. The numerical tests verify both identities for several directions and base solids. Taking traces gives

$$(1-\theta)\operatorname{tr}(P_hD^{-1})\leq B_h=\frac1{3\kappa_s}-\theta g_h,$$
$$(1-\theta)\operatorname{tr}(P_sD^{-1})\leq B_s=\frac5{2\mu_s}-\theta g_s.$$

Average D over rotations. Its average is
$D_0=3(\kappa_s-\kappa^*)P_h+2(\mu_s-\mu^*)P_s$.
The matrix inverse is operator convex, so the inverse of the rotation average is bounded above by the average inverse. Taking either invariant trace therefore gives

$$\frac{1-\theta}{3(\kappa_s-\kappa^*)}\leq B_h,\qquad
\frac{5(1-\theta)}{2(\mu_s-\mu^*)}\leq B_s.$$

Rearrangement gives

$$\kappa^*\leq\kappa_s-\frac{1-\theta}{3B_h}=\kappa_{\rm HS}(\theta),\qquad
\mu^*\leq\mu_s-\frac{5(1-\theta)}{2B_s}=\mu_{\rm HS}(\theta).$$

Substitution yields exactly the formulas in [methods](methods.md). At theta=0 and theta=1 the corresponding conclusions follow directly or by closure. Hence $0\leq x\leq1$ and $0\leq y\leq1$.

The trial-field and trace-bound framework is classical. [References](references.md) identify its sources. This proof supplies the outer square used throughout the repository; it makes no stronger claim about the coupled low-bulk corner.
