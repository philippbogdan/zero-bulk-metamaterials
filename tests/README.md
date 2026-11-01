# Small checks with independent answers

The suite tests the HS corner, acoustic trace identities, a homogeneous laminate, the arithmetic and harmonic closed forms for diagonal layers, rotation covariance, a small explicit geometry, sensitivity of the dilation residual, exact wall reflection and reapplication of the FEM comparison formulas. A deliberately corrupted shear value must make the real checker return a nonzero exit status.

These bounded checks need only the headline dependencies. They do not replace re-evaluation of the 14 complete headline designs. Run them with `python -m unittest discover -s tests`.
