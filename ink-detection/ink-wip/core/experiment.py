from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Experiment:
    name: str
    data: Any
    model: Any
    loss: Any
    objective: Any
    runtime: Any
    augment: Any
    stitch: Any
    trainer: Any = None
    evaluator: Any = None
