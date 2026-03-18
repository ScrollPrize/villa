from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlopen


DATA_ROOT = Path(__file__).resolve().parent / "data"

FIBER_IMAGE_URLS = [
    "https://dl.ash2txt.org/datasets/fiber-skeletons/Dataset004_sk-fibers-binary-20250728/imagesTr/s1_00497_01497_03997_256_0000.tif",
    "https://dl.ash2txt.org/datasets/fiber-skeletons/Dataset004_sk-fibers-binary-20250728/imagesTr/s1_08997_02997_02497_256_0000.tif",
    "https://dl.ash2txt.org/datasets/fiber-skeletons/Dataset004_sk-fibers-binary-20250728/imagesTr/s3_01994_01494_00994_512_0000.tif",
    "https://dl.ash2txt.org/datasets/fiber-skeletons/Dataset004_sk-fibers-binary-20250728/imagesTr/s5_06494_01994_03994_512_0000.tif",
]

FIBER_LABEL_URLS = [
    "https://dl.ash2txt.org/datasets/fiber-skeletons/Dataset004_sk-fibers-binary-20250728/labelsTr/s1_00497_01497_03997_256.tif",
    "https://dl.ash2txt.org/datasets/fiber-skeletons/Dataset004_sk-fibers-binary-20250728/labelsTr/s1_08997_02997_02497_256.tif",
    "https://dl.ash2txt.org/datasets/fiber-skeletons/Dataset004_sk-fibers-binary-20250728/labelsTr/s3_01994_01494_00994_512.tif",
    "https://dl.ash2txt.org/datasets/fiber-skeletons/Dataset004_sk-fibers-binary-20250728/labelsTr/s5_06494_01994_03994_512.tif",
]

SURFACE_STEMS = [
    "sample_00807",
    "sample_00808",
    "sample_00809",
    "sample_00810",
    "sample_00908",
    "sample_00909",
    "sample_00910",
    "sample_00911",
    "sample_00912",
    "sample_00913",
    "sample_00914",
    "sample_00915",
    "sample_00916",
]

SURFACE_IMAGE_BASE = "https://dl.ash2txt.org/community-uploads/bruniss/labels/kds/images"
SURFACE_LABEL_BASE = "https://dl.ash2txt.org/community-uploads/bruniss/labels/kds/labels"


def _download(url: str, destination: Path) -> None:
    if destination.exists():
        print(f"skip   {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"fetch  {url}")
    with urlopen(url) as response, destination.open("wb") as output:
        output.write(response.read())
    print(f"wrote  {destination}")


def _fiber_name(url: str) -> str:
    return Path(url).name.replace("_0000.tif", ".tif")


def download_fibers(*, data_root: Path = DATA_ROOT) -> None:
    for url in FIBER_IMAGE_URLS:
        _download(url, data_root / "fibers" / "images" / _fiber_name(url))

    for url in FIBER_LABEL_URLS:
        _download(url, data_root / "fibers" / "labels" / _fiber_name(url))


def download_surfaces(*, data_root: Path = DATA_ROOT) -> None:
    for stem in SURFACE_STEMS:
        _download(
            f"{SURFACE_IMAGE_BASE}/{stem}.tif",
            data_root / "surfaces" / "images" / f"{stem}.tif",
        )
        _download(
            f"{SURFACE_LABEL_BASE}/{stem}_surf.tif",
            data_root / "surfaces" / "labels" / f"{stem}.tif",
        )


def download_tasks(tasks: str | tuple[str, ...] | list[str] = "both", *, data_root: Path = DATA_ROOT) -> None:
    if isinstance(tasks, str):
        normalized = tasks.strip().lower()
        if normalized in {"", "both", "all"}:
            selected = ("fibers", "surfaces")
        else:
            selected = (normalized,)
    else:
        selected = tuple(str(task).strip().lower() for task in tasks)

    if "fibers" in selected:
        download_fibers(data_root=Path(data_root))
    if "surfaces" in selected:
        download_surfaces(data_root=Path(data_root))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download packaged task-eval data.")
    parser.add_argument("--task", default="both", choices=("both", "fibers", "surfaces"))
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    args = parser.parse_args()
    download_tasks(args.task, data_root=args.data_root)


if __name__ == "__main__":
    main()
