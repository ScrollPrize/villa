from __future__ import annotations

from dataclasses import dataclass, field

from ink.core.types import EvalReport


@dataclass(frozen=True)
class MetricReport:
    summary: dict[str, float] = field(default_factory=dict)
    by_group: dict[str, dict[str, float]] = field(default_factory=dict)
    by_segment: dict[str, dict[str, float]] = field(default_factory=dict)


def _merge_metric_map(current: dict[str, float], incoming: dict[str, float], *, where: str) -> dict[str, float]:
    merged = dict(current)
    for key, value in incoming.items():
        key = str(key)
        if key in merged:
            raise ValueError(f"duplicate metric key {key!r} while merging {where}")
        merged[key] = float(value)
    return merged


def _merge_nested_metric_map(
    current: dict[str, dict[str, float]],
    incoming: dict[str, dict[str, float]],
    *,
    where: str,
) -> dict[str, dict[str, float]]:
    merged = {str(entity): dict(metrics) for entity, metrics in current.items()}
    for entity, metrics in incoming.items():
        entity_key = str(entity)
        merged[entity_key] = _merge_metric_map(
            merged.get(entity_key, {}),
            metrics,
            where=f"{where}[{entity_key!r}]",
        )
    return merged


def _merge_stage_reports(
    current: dict[str, EvalReport],
    incoming: dict[str, EvalReport],
) -> dict[str, EvalReport]:
    merged = {str(stage): report for stage, report in current.items()}
    for stage, report in incoming.items():
        stage_key = str(stage)
        if stage_key in merged:
            merged[stage_key] = merge_eval_reports([merged[stage_key], report])
        else:
            merged[stage_key] = report
    return merged


def merge_metric_reports(reports: list[MetricReport]) -> EvalReport:
    summary: dict[str, float] = {}
    by_group: dict[str, dict[str, float]] = {}
    by_segment: dict[str, dict[str, float]] = {}
    for report in reports:
        summary = _merge_metric_map(summary, report.summary, where="summary")
        by_group = _merge_nested_metric_map(by_group, report.by_group, where="by_group")
        by_segment = _merge_nested_metric_map(by_segment, report.by_segment, where="by_segment")
    return EvalReport(summary=summary, by_group=by_group, by_segment=by_segment)


def merge_eval_reports(reports: list[EvalReport]) -> EvalReport:
    summary: dict[str, float] = {}
    by_group: dict[str, dict[str, float]] = {}
    by_segment: dict[str, dict[str, float]] = {}
    stages: dict[str, EvalReport] = {}
    for report in reports:
        summary = _merge_metric_map(summary, report.summary, where="summary")
        by_group = _merge_nested_metric_map(by_group, report.by_group, where="by_group")
        by_segment = _merge_nested_metric_map(by_segment, report.by_segment, where="by_segment")
        stages = _merge_stage_reports(stages, report.stages)
    return EvalReport(summary=summary, by_group=by_group, by_segment=by_segment, stages=stages)


def _prefixed_key(key: str, prefix: str) -> str:
    key = str(key)
    prefix = str(prefix)
    if key.startswith(prefix):
        return key
    return f"{prefix}{key}"


def _flatten_prefixed_stage_report(report: EvalReport, *, prefix: str) -> EvalReport:
    base = EvalReport(
        summary={_prefixed_key(key, prefix): float(value) for key, value in report.summary.items()},
        by_group={
            str(group): {_prefixed_key(key, prefix): float(value) for key, value in metrics.items()}
            for group, metrics in report.by_group.items()
        },
        by_segment={
            str(segment): {_prefixed_key(key, prefix): float(value) for key, value in metrics.items()}
            for segment, metrics in report.by_segment.items()
        },
    )
    nested_reports = [base]
    for stage, stage_report in report.stages.items():
        nested_reports.append(
            _flatten_prefixed_stage_report(
                stage_report,
                prefix=f"{prefix}{str(stage)}/",
            )
        )
    return merge_eval_reports(nested_reports)


def _normalized_stage_prefix(prefix: str) -> str:
    prefix = str(prefix).strip()
    if not prefix:
        return ""
    if prefix.endswith("/"):
        return prefix
    return f"{prefix}/"


def flatten_eval_report(report: EvalReport, *, stage_prefix: str = "") -> EvalReport:
    stage_prefix = _normalized_stage_prefix(stage_prefix)
    base = EvalReport(
        summary={str(key): float(value) for key, value in report.summary.items()},
        by_group={str(group): {str(key): float(value) for key, value in metrics.items()} for group, metrics in report.by_group.items()},
        by_segment={
            str(segment): {str(key): float(value) for key, value in metrics.items()}
            for segment, metrics in report.by_segment.items()
        },
    )
    reports = [base]
    for stage, stage_report in report.stages.items():
        reports.append(
            _flatten_prefixed_stage_report(
                stage_report,
                prefix=f"{str(stage_prefix)}{str(stage)}/",
            )
        )
    return merge_eval_reports(reports)
