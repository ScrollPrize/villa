# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in LICENSES/Apache-2.0.txt.
#
# Modified for Dinovol's three-dimensional training pipeline.

import numpy as np


class CosineScheduler:
    def __init__(
            self,
            *,
            base_value: float,
            final_value: float,
            total_iters: int,
            warmup_iters: int = 0,
            start_warmup_value: float = 0.0,
            freeze_iters: int = 0,
    ) -> None:
        self.final_value = float(final_value)
        self.total_iters = int(total_iters)

        if self.total_iters <= 0:
            self.schedule = np.zeros((0,), dtype=np.float64)
            return

        freeze_iters = max(0, min(int(freeze_iters), self.total_iters))
        warmup_iters = max(0, min(int(warmup_iters), self.total_iters - freeze_iters))

        freeze_schedule = np.zeros((freeze_iters,), dtype=np.float64)
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters, dtype=np.float64)

        cosine_iters = self.total_iters - warmup_iters - freeze_iters
        if cosine_iters > 0:
            iters = np.arange(cosine_iters, dtype=np.float64)
            cosine_schedule = final_value + 0.5 * (base_value - final_value) * (
                        1.0 + np.cos(np.pi * iters / len(iters)))
        else:
            cosine_schedule = np.zeros((0,), dtype=np.float64)

        self.schedule = np.concatenate((freeze_schedule, warmup_schedule, cosine_schedule))
        if len(self.schedule) != self.total_iters:
            raise AssertionError("invalid scheduler length")

    def __getitem__(self, step: int) -> float:
        if step >= self.total_iters:
            return self.final_value
        return float(self.schedule[step])


