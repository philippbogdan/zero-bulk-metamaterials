# Geometry and evidence

The data contain 556 explicit laminate geometries with stored high-precision evaluations, plus three finite-scale tensor files. The headline set is the 14-record `zero_bulk_aniso` family. All data are included under CC0 1.0.

[Certified records](certified/README.md) describe exact separated-scale geometries. [Finite-scale records](finite_scale/README.md) contain direct voxel FEM tensors. They answer different questions and are not interchangeable.

`reference.json` holds expected corner gaps from a separate extraction of the stored evidence. `reproduce.py` recomputes those gaps from the complete dataset and the baseline construction formulas. The data manifest lists SHA-256 hashes, family counts and exclusions so that the supplied inputs can be identified precisely.
