"""The solver of the geometry steps: a model written as equations over named unknowns, solved robustly.

A calibration step states its model here and nowhere else:

    p = Problem()
    p.unknowns("axis", n_tiles)                      # blocks of unknowns, by name
    p.unknowns("map", n_terms)
    e = p.equations(measured, fit=not_held_out)      # one equation per measurement: measured = sum of terms
    e.add("axis", tile_a, -1.0)                      # each term: a block, which of its unknowns, the coefficient
    e.add("axis", tile_b, +1.0)                      #   (arrays, one value per equation, or one number for all)
    e.add("map", k, -(phi_b - phi_a))
    s = p.solve(huber=0.1)
    s["axis"], s.left(e)                             # the unknowns by name; what each equation leaves

Every equation is a line in the unknowns; a model that is not (an angle that multiplies a place) is written as a
line about the values in hand and solved again. The arithmetic is helirec's own, in C++
(helirec/calibration/solve.hpp: columns scaled to unit length, a Huber loss by reweighting, the normal equations
factored with pivoting; no outside solver). Equations with fit False take no part in the solve and are predicted:
what they leave is the model's test.
"""
from __future__ import annotations

import numpy as np


class Equations:
    """Some equations of a problem (Problem.equations): their terms are added here."""

    def __init__(self, problem, first, n):
        self.problem, self.first, self.n = problem, first, n

    def add(self, block, index, coefficient, where=None):
        """+ coefficient x unknown `index` of `block`, in every equation (index, coefficient: one per equation, or
        one for all); where: only in those equations. A coefficient of zero adds nothing."""
        offset, size = self.problem.blocks[block]
        index = np.broadcast_to(np.asarray(index, np.int64), (self.n,))
        coefficient = np.broadcast_to(np.asarray(coefficient, float), (self.n,))
        keep = coefficient != 0
        if where is not None:
            keep = keep & np.broadcast_to(np.asarray(where, bool), (self.n,))
        if np.any((index[keep] < 0) | (index[keep] >= size)):
            raise ValueError(f"an equation names an unknown outside the {size} of {block!r}")
        self.problem._row.append(self.first + np.flatnonzero(keep))
        self.problem._col.append(offset + index[keep])
        self.problem._value.append(coefficient[keep])
        return self

    @property
    def rows(self):
        return slice(self.first, self.first + self.n)


class Solution:
    def __init__(self, problem, x, residual, weight, spread, rank):
        self.problem, self.x, self.residual, self.weight, self.spread, self.rank = problem, x, residual, weight, spread, rank

    def __getitem__(self, block):
        offset, size = self.problem.blocks[block]
        return self.x[offset:offset + size]

    def left(self, equations):
        """measured minus modelled, per equation of `equations` (fitted or not)"""
        return self.residual[equations.rows]


class Problem:
    def __init__(self):
        self.blocks = {}                 # name -> (offset, size)
        self.n = 0                       # unknowns
        self.m = 0                       # equations
        self._row, self._col, self._value, self._y, self._fit = [], [], [], [], []

    def unknowns(self, name, size):
        if name in self.blocks:
            raise ValueError(f"the unknowns {name!r} are already there")
        self.blocks[name] = (self.n, int(size))
        self.n += int(size)
        return name

    def equations(self, measured, fit=True):
        """One equation per value of `measured`; fit: which of them the solve uses (one for all, or one each)."""
        y = np.atleast_1d(np.asarray(measured, float))
        e = Equations(self, self.m, len(y))
        self._y.append(y)
        self._fit.append(np.broadcast_to(np.asarray(fit, bool), y.shape).copy())
        self.m += len(y)
        return e

    def solve(self, huber=0.0, relative=False, iterations=8, rcond=1e-12):
        """huber: the loss's threshold in the equations' units (0: least squares), or with relative a multiple of
        the residuals' own robust spread. Returns a Solution."""
        from . import _core

        cat = lambda parts, dtype: np.concatenate(parts).astype(dtype) if parts else np.zeros(0, dtype)
        row, col, value = cat(self._row, np.int64), cat(self._col, np.int64), cat(self._value, float)
        y, fit = cat(self._y, float), cat(self._fit, bool)
        # the equations fitted, renumbered; the others are predicted below
        number = np.cumsum(fit) - 1
        use = fit[row]
        got = _core.calibration_solve(np.ascontiguousarray(number[row[use]]), np.ascontiguousarray(col[use]), np.ascontiguousarray(value[use]), np.ascontiguousarray(y[fit]),
                                      self.n, float(huber), bool(relative), int(iterations), float(rcond))
        x = np.asarray(got["x"])
        residual = y - np.bincount(row, weights=value * x[col], minlength=self.m)
        weight = np.zeros(self.m)
        weight[fit] = np.asarray(got["weight"])
        return Solution(self, x, residual, weight, float(got["spread"]), int(got["rank"]))
