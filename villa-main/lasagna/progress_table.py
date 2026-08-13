from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ProgressColumn:
	key: str
	label: str
	description: str
	min_width: int = 5


def format_progress_value(value: float) -> str:
	av = abs(float(value))
	if av != 0.0 and (av >= 1000.0 or av < 1.0e-3):
		return f"{float(value):.1e}"
	if av < 10.0:
		return f"{float(value):.4f}"
	if av < 100.0:
		return f"{float(value):.3f}"
	return f"{float(value):.1f}"


def print_progress_legend(
	*,
	prefix: str,
	items: Iterable[tuple[str, str]],
	cols_per_row: int = 3,
) -> None:
	items_l = list(items)
	if not items_l:
		return
	print(f"{prefix} progress columns", flush=True)
	key_w = max(len(k) for k, _v in items_l)
	desc_w = max(len(v) for _k, v in items_l)
	cell_w = key_w + 3 + desc_w
	header_cell = f"{'col':<{key_w}} : {'meaning':<{desc_w}}"
	header = " | ".join(f"{header_cell:<{cell_w}}" for _ in range(int(cols_per_row)))
	print(f"  {header}", flush=True)
	for i in range(0, len(items_l), int(cols_per_row)):
		cells = [f"{k:<{key_w}} : {v:<{desc_w}}" for k, v in items_l[i:i + int(cols_per_row)]]
		while len(cells) < int(cols_per_row):
			cells.append(" " * cell_w)
		print(f"  {' | '.join(cells)}", flush=True)


def progress_widths(columns: Sequence[ProgressColumn], values: dict[str, str]) -> dict[str, int]:
	return {
		col.key: max(int(col.min_width), len(col.label), len(values.get(col.key, "")))
		for col in columns
	}


def progress_header(columns: Sequence[ProgressColumn], widths: dict[str, int]) -> str:
	return " ".join(f"{col.label:>{int(widths[col.key])}s}" for col in columns)


def progress_row(columns: Sequence[ProgressColumn], widths: dict[str, int], values: dict[str, str]) -> str:
	return " ".join(f"{values.get(col.key, ''):>{int(widths[col.key])}s}" for col in columns)

