# Search and direct simulation

These modules require PyTorch. They are isolated from the headline checker.

`torch_algebra.py` supplies differentiable float64 lamination. `optimise.py` searches leaf and branch parameters; `project.py` solves the compatible-dilation equations and refines their residual with mpmath. Use the root-level `search_design.py` and `certify_design.py` commands to keep search estimates separate from certified stiffness.

`voxel.py` supplies periodic Q1 voxel elasticity using the reference PyTorch convolution implementation, matrix-free assembly, connected-component handling and geometric multigrid. The root-level `finite_scale.py` fixes the geometry and solver settings for the wall experiment. [Full instructions](../../REPRODUCE.md).
