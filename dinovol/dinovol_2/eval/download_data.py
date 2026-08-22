from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import re
from urllib.request import urlopen


DATA_ROOT = Path(__file__).resolve().parent / "data"
SURFACE_EVAL_BASE = "https://dl.ash2txt.org/community-uploads/bruniss/labels/surfaces/train_surf_eval"
INK_EVAL_BASE = "https://dl.ash2txt.org/community-uploads/bruniss/labels/ink/train_ink_eval"
INK_MANIFEST_URL = f"{INK_EVAL_BASE}/manifest.json"


def _download(url: str, destination: Path) -> None:
    if destination.exists():
        print(f"skip   {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"fetch  {url}")
    with urlopen(url) as response, destination.open("wb") as output:
        output.write(response.read())
    print(f"wrote  {destination}")


def _list_remote_tifs(url: str) -> list[str]:
    with urlopen(url) as response:
        page = response.read().decode("utf-8", errors="replace")
    names = {
        html.unescape(match)
        for match in re.findall(r'href="([^"]+\.tif)"', page, flags=re.IGNORECASE)
    }
    return sorted(names)


def download_surfaces(*, data_root: Path = DATA_ROOT) -> None:
    image_names = _list_remote_tifs(f"{SURFACE_EVAL_BASE}/images/")
    label_names = set(_list_remote_tifs(f"{SURFACE_EVAL_BASE}/labels/"))
    if not image_names:
        raise ValueError(f"No surface task-eval images found at {SURFACE_EVAL_BASE}/images/")

    missing_labels = [name for name in image_names if name not in label_names]
    if missing_labels:
        raise ValueError(
            f"Missing surface task-eval labels for: {', '.join(missing_labels[:5])}"
        )

    for name in image_names:
        _download(
            f"{SURFACE_EVAL_BASE}/images/{name}",
            data_root / "surfaces" / "images" / name,
        )
        _download(
            f"{SURFACE_EVAL_BASE}/labels/{name}",
            data_root / "surfaces" / "labels" / name,
        )


def download_ink(*, data_root: Path = DATA_ROOT) -> None:
    task_root = Path(data_root) / "ink"
    manifest_path = task_root / "manifest.json"
    _download(INK_MANIFEST_URL, manifest_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    crops = manifest.get("crops")
    if not isinstance(crops, list) or not crops:
        raise ValueError(f"Ink task manifest at {manifest_path} does not contain any crops")

    for crop in crops:
        image_name = str(crop["image_tif"])
        label_name = str(crop.get("label_tif", image_name))
        _download(
            f"{INK_EVAL_BASE}/images/{image_name}",
            task_root / "images" / image_name,
        )
        _download(
            f"{INK_EVAL_BASE}/labels/{label_name}",
            task_root / "labels" / image_name,
        )
        _download(
            f"{INK_EVAL_BASE}/supervision_masks/{image_name}",
            task_root / "supervision_masks" / image_name,
        )


def download_tasks(tasks: str | tuple[str, ...] | list[str] = "both", *, data_root: Path = DATA_ROOT) -> None:
    if isinstance(tasks, str):
        normalized = tasks.strip().lower()
        if normalized in {"", "both"}:
            selected = ("surfaces", "ink")
        else:
            selected = (normalized,)
    else:
        selected = tuple(str(task).strip().lower() for task in tasks)

    if "surfaces" in selected:
        download_surfaces(data_root=Path(data_root))
    if "ink" in selected:
        download_ink(data_root=Path(data_root))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download packaged task-eval data.")
    parser.add_argument("--task", default="both", choices=("both", "surfaces", "ink"))
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    args = parser.parse_args()
    download_tasks(args.task, data_root=args.data_root)


if __name__ == "__main__":
    main()
