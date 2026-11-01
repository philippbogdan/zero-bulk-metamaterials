# Licences and dependencies

The repository's own code, data, figures and text are dedicated to the public domain under [CC0 1.0](LICENSE). The full legal text is the [Creative Commons legal code](https://creativecommons.org/publicdomain/zero/1.0/legalcode.txt), reproduced without editorial changes.

The numerical modules are derived from the same project's research implementation. No external dataset, mesh generator or third-party source code is bundled. Packages are installed by the user and retain their own terms:

| Package | Use | Licence |
|---|---|---|
| NumPy | Tensor arrays and float64 algebra | BSD-3-Clause; distributions include additional licensed components |
| SciPy | Optional FEM connectivity and numerical tools | BSD-3-Clause; distributions include additional notices |
| mpmath | Arbitrary-precision lamination | BSD-3-Clause |
| PyTorch | Optional design search and finite-scale simulation | BSD-3-Clause; distributions include additional notices |
| Matplotlib | Stiffness, gap and convergence plots | Matplotlib licence, based on the PSF licence |
| pandas | Figure-table loading and selection | BSD-3-Clause |
| Numba | Compiled voxel ray tracing | BSD-2-Clause |
| Pillow | PNG output for plots, the wall image and video frames | MIT-CMU |

These licence names come from installed package metadata and the accompanying licence files. The numerical dependencies are in `requirements.txt`, optional search and simulation dependencies in `requirements-optional.txt`, and figure dependencies in `requirements-figures.txt`. Video encoding also needs a separate FFmpeg installation with libx264 support, under that installation's own licence terms. GIF optimisation optionally uses gifsicle, installed separately under its own GPL-2.0 licence. See the [figure commands](figures/scripts/README.md).

This repository does not redistribute these packages or replace their licence notices. CC0 contains no warranty and does not waive patent or trademark rights.
