# Direct wall simulations

`q2_finite_herringbone_A.json` contains three rank-two wall tensors and the preceding plate-stack tensors. `q2_finite_herringbone_H.json` contains six herringbone tensors at N=256. `q2_finite_herringbone_H_N128.json` contains two herringbone tensors at N=128. The ratio-four overlap is intentional.

Each case identifies its layer periods, resolution, effective Mandel tensor `C_fem`, stored relative error and mode stiffnesses. H cases also retain `C_formula`. These stored formulas are checked against a fresh application of lamination to the preceding-level FEM tensors; they are not used as the primary input to that calculation. Runtime metadata are omitted.

The base solid has E=1 and nu_s=0.3. The exact binary geometry and reflection used to reconstruct the opposite wall orientation are specified in [the finite-scale explanation](../../docs/finite-scale.md). These tensors describe individual wall steps, not the complete headline structures.
