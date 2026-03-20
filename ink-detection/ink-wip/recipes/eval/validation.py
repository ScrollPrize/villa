from __future__ import annotations

from dataclasses import dataclass, field, replace

from ink.core.types import EvalReport


@dataclass(frozen=True, kw_only=True)
class ValidationEvaluator:
    patch: object | None = None
    stitch: object | None = None
    stage_prefix: str = field(default="val", init=False)

    def build(self, *, data, runtime=None, stitch=None, logger=None) -> ValidationEvaluator:
        bound_patch = None if self.patch is None else self.patch.build(
            data=data,
            runtime=runtime,
            stitch=stitch,
            logger=logger,
        )
        bound_stitch = None if self.stitch is None else self.stitch.build(
            data=data,
            runtime=runtime,
            stitch=stitch,
            logger=logger,
        )
        return replace(self, patch=bound_patch, stitch=bound_stitch)

    def evaluate(self, model, val_loader, *, device=None) -> EvalReport:
        if self.patch is not None and self.stitch is not None:
            self.stitch.begin_epoch()
            patch_report = self.patch.evaluate(
                model,
                val_loader,
                device=device,
                batch_observer=self.stitch.observe_batch,
            )
            stitch_report = self.stitch.finalize_epoch()
            return EvalReport(
                summary={},
                stages={
                    "patch": patch_report,
                    "stitch": stitch_report,
                },
            )

        if self.patch is not None:
            patch_report = self.patch.evaluate(model, val_loader, device=device)
            return EvalReport(
                summary={},
                stages={"patch": patch_report},
            )
        if self.stitch is not None:
            stitch_report = self.stitch.evaluate(model, val_loader, device=device)
            return EvalReport(
                summary={},
                stages={"stitch": stitch_report},
            )
        return EvalReport(summary={})
