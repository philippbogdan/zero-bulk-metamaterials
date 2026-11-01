# Exact geometry to stiffness

`laminate.py` implements partial-inverse rank-one lamination in NumPy and mpmath. `certify.py` assembles the explicit walls, accounts for their void fraction, evaluates the tree and computes its vanishing-wall limit. `directions.py` holds the fixed sequential-base direction order; importing it does not import PyTorch.

`elastic.py` defines Mandel notation and the projected moduli. `zero_mode.py` checks compatible dilation independently of stiffness. `evidence.py` validates stored geometry and applies the conservative finest-wall selection. `metrics.py` computes support gaps and rebuilds the finite-scale comparison tensors. `geometry.py` constructs the binary wall sections.

For example, load one geometry and evaluate its finest wall fraction:

```python
import json
from metamaterials.certify import certify

with open('data/certified/zero_bulk_aniso_nu0.3_th0.1.json') as stream:
    record = json.load(stream)
value = certify(record['design'], record['nu'], record['theta'],
                eps_list=[1e-14], deltas=['1e-60'], dps=90)
print(value['runs'][0])
```

See [methods](../docs/methods.md) for the precision convention and [optional modules](optional/README.md) for search and direct simulation.
