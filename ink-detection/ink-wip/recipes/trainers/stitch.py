from __future__ import annotations

from dataclasses import dataclass

from ink.core.types import DataBundle, EvalReport
from ink.recipes.metrics import flatten_eval_report
from ink.recipes.trainers.patch import EvalEpochResult
from ink.recipes.trainers.support.logging import init_wandb_session
from ink.recipes.trainers.support.run_state import apply_init_checkpoint, resolve_checkpoint_path


@dataclass(frozen=True)
class StitchRunResult:
    eval_epoch: EvalEpochResult
    store_root_dir: str | None = None
    segment_prob_paths: dict[str, str] | None = None


@dataclass
class StitchTraining:
    experiment: object
    val_loader: object

    def _logged_eval_report(self, report: EvalReport) -> EvalReport:
        evaluator = getattr(self.experiment, "evaluator", None)
        stage_prefix = getattr(evaluator, "stage_prefix", "")
        return flatten_eval_report(report, stage_prefix=stage_prefix)

    def run(self, *, device=None, run_fs=None) -> StitchRunResult:
        evaluator = getattr(self.experiment, "evaluator", None)
        if evaluator is None or not callable(getattr(evaluator, "evaluate", None)):
            raise TypeError("stitch inference requires evaluator.evaluate(model, val_loader, *, device=None)")

        model = self.experiment.model
        runtime = self.experiment.runtime
        init_ckpt_path = resolve_checkpoint_path(getattr(runtime, "init_ckpt_path", None))
        resume_ckpt_path = resolve_checkpoint_path(getattr(runtime, "resume_ckpt_path", None))
        if init_ckpt_path is not None and resume_ckpt_path is not None:
            raise ValueError("init_ckpt_path and resume_ckpt_path are mutually exclusive")
        ckpt_path = init_ckpt_path if init_ckpt_path is not None else resume_ckpt_path
        if ckpt_path is not None:
            apply_init_checkpoint(model, ckpt_path=ckpt_path)

        wandb_session = init_wandb_session(self.experiment, run_fs=run_fs)
        try:
            raw_report = evaluator.evaluate(model, self.val_loader, device=device)
            if not isinstance(raw_report, EvalReport):
                raise TypeError("evaluator must return EvalReport")
            report = self._logged_eval_report(raw_report)
            eval_epoch = EvalEpochResult(epoch=0, report=report)
            if run_fs is not None:
                run_fs.log_eval_epoch(0, report)
            if wandb_session is not None:
                wandb_session.log_eval_epoch(0, report.summary)

            stitch_eval = getattr(evaluator, "stitch", None)
            store = getattr(stitch_eval, "store", None)
            store_root_dir = None
            segment_prob_paths = None
            if store is not None and getattr(store, "root_dir", None) is not None:
                store_root_dir = str(store.root_dir)
                segment_shapes = dict(getattr(stitch_eval, "segment_shapes", {}) or {})
                if segment_shapes:
                    segment_prob_paths = {
                        str(segment_id): store.write_full_segment_probs(segment_id=str(segment_id))
                        for segment_id in segment_shapes
                    }
            return StitchRunResult(
                eval_epoch=eval_epoch,
                store_root_dir=store_root_dir,
                segment_prob_paths=segment_prob_paths,
            )
        finally:
            if wandb_session is not None:
                wandb_session.finish()


@dataclass(frozen=True)
class StitchTrainer:
    def build(self, *, experiment, data: DataBundle, logger=None) -> StitchTraining:
        del logger
        return StitchTraining(
            experiment=experiment,
            val_loader=data.val_loader,
        )
