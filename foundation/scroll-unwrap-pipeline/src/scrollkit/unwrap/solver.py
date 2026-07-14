"""Factor-once / solve-many sparse SPD solver with a backend ladder.

Backends (preference order unless pinned): pypardiso, cholmod (if installed), pyamg, scipy-splu.
The production choice is recorded by scripts/bench_solver.py in reports/solver_bench.json and
resolved here when backend='auto'.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

_LADDER = ["pypardiso", "cholmod", "pyamg", "scipy-splu"]


class FactorizedSPD:
    """factorize(A) once; solve(B) many times. A must be SPD (pin the null space first)."""

    def __init__(self, A: sp.spmatrix, backend: str = "auto", repo_root: Path | None = None) -> None:
        if backend == "auto":
            backend = self._resolve_auto(repo_root)
        backend = backend.replace("_", "-")
        if backend == "scipy":
            backend = "scipy-splu"
        self.backend = backend
        A = A.tocsr()
        if backend == "pypardiso":
            import pypardiso

            self._solver = pypardiso.PyPardisoSolver()
            self._A = A
            self._solver.factorize(A)
        elif backend == "cholmod":
            from sksparse.cholmod import cholesky

            self._factor = cholesky(A.tocsc())
        elif backend == "pyamg":
            import pyamg

            self._ml = pyamg.smoothed_aggregation_solver(A.tocsr())
            self._A = A
        elif backend == "scipy-splu":
            self._lu = spla.splu(A.tocsc())
        else:
            raise ValueError(f"unknown backend {backend!r}")

    @staticmethod
    def _resolve_auto(repo_root: Path | None) -> str:
        if repo_root is not None:
            bench = Path(repo_root) / "reports/solver_bench.json"
            if bench.exists():
                chosen = json.loads(bench.read_text()).get("chosen")
                if chosen:
                    return chosen
        for name in _LADDER:
            try:
                if name == "pypardiso":
                    import pypardiso  # noqa: F401
                elif name == "cholmod":
                    from sksparse.cholmod import cholesky  # noqa: F401
                elif name == "pyamg":
                    import pyamg  # noqa: F401
                return name
            except ImportError:
                continue
        return "scipy-splu"

    def solve(self, B: np.ndarray) -> np.ndarray:
        B = np.asarray(B, dtype=np.float64)
        squeeze = B.ndim == 1
        if squeeze:
            B = B[:, None]
        if self.backend == "pypardiso":
            X = self._solver.solve(self._A, B)
        elif self.backend == "cholmod":
            X = self._factor(B)
        elif self.backend == "pyamg":
            cols = [self._ml.solve(B[:, j], tol=1e-10, accel="cg") for j in range(B.shape[1])]
            X = np.stack(cols, axis=1)
        else:
            X = self._lu.solve(B)
        return X[:, 0] if squeeze else X


def pin_vertex(L: sp.spmatrix, pinned: int) -> tuple[sp.csr_matrix, np.ndarray]:
    """Remove row/col `pinned` (Dirichlet). Returns (L_reduced, keep_index_map)."""
    n = L.shape[0]
    keep = np.concatenate([np.arange(pinned), np.arange(pinned + 1, n)])
    L = L.tocsr()
    return L[keep][:, keep].tocsr(), keep
