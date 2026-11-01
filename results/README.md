# Recomputed values

`reproduce.py` writes its numerical evidence and pass/fail comparisons here. Read the `scope` field: `stored-only` does not re-certify geometry, `single-case` covers one design, `headline` covers the 14 main designs, and `all` covers every stored run.

The included [reproduction.json](reproduction.json) is a completed headline check: all 14 geometries were re-evaluated at 90 digits, every stored-evidence comparison passed, and the error list is empty. Its elapsed time is 4.72 seconds for the eight-process command documented in [REPRODUCE.md](../REPRODUCE.md). The file records the elapsed time but not the process count.

The expected results live with the [data](../data/README.md); they are separate from generated outputs. A successful exit applies only to the scope actually executed. Long calculations can leave a partial file with `passed: false` if interrupted.
