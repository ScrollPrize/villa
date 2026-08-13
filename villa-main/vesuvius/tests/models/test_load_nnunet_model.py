from __future__ import annotations

import sys
from pathlib import Path

import nnunetv2
import pytest

from vesuvius.utils.models import load_nnunet_model


def test_resolve_python_class_exact_finds_4000epochs_variant() -> None:
    trainer_root = Path(nnunetv2.__path__[0]) / "training" / "nnUNetTrainer"

    trainer_class = load_nnunet_model.resolve_python_class_exact(
        str(trainer_root),
        "nnUNetTrainer_4000epochs",
        "nnunetv2.training.nnUNetTrainer",
    )

    assert trainer_class is not None
    assert trainer_class.__name__ == "nnUNetTrainer_4000epochs"
    assert trainer_class.__module__.endswith("variants.training_length.nnUNetTrainer_Xepochs")


def test_resolve_python_class_exact_ignores_unrelated_import_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package_root = tmp_path / "fakepkg"
    trainer_root = package_root / "training" / "nnUNetTrainer"
    (trainer_root / "variants" / "training_length").mkdir(parents=True)
    (trainer_root / "variants" / "loss").mkdir(parents=True)

    for init_path in [
        package_root / "__init__.py",
        package_root / "training" / "__init__.py",
        trainer_root / "__init__.py",
        trainer_root / "variants" / "__init__.py",
        trainer_root / "variants" / "training_length" / "__init__.py",
        trainer_root / "variants" / "loss" / "__init__.py",
    ]:
        init_path.write_text("", encoding="utf-8")

    (trainer_root / "variants" / "training_length" / "good.py").write_text(
        "class TargetTrainer:\n    pass\n",
        encoding="utf-8",
    )
    (trainer_root / "variants" / "loss" / "broken.py").write_text(
        "import definitely_missing_dependency\nclass BrokenTrainer:\n    pass\n",
        encoding="utf-8",
    )

    monkeypatch.syspath_prepend(str(tmp_path))

    imported_modules = []
    real_import_module = load_nnunet_model.importlib.import_module

    def tracked_import(module_name: str):
        imported_modules.append(module_name)
        return real_import_module(module_name)

    monkeypatch.setattr(load_nnunet_model.importlib, "import_module", tracked_import)

    trainer_class = load_nnunet_model.resolve_python_class_exact(
        str(trainer_root),
        "TargetTrainer",
        "fakepkg.training.nnUNetTrainer",
    )

    assert trainer_class is not None
    assert trainer_class.__name__ == "TargetTrainer"
    assert not any(module_name.endswith(".broken") for module_name in imported_modules)


def test_resolve_python_class_exact_does_not_import_from_nnunetv1_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    trainer_root = Path(nnunetv2.__path__[0]) / "training" / "nnUNetTrainer"
    imported_modules = []
    real_import_module = load_nnunet_model.importlib.import_module
    before = {name for name in sys.modules if ".from_nnunetv1." in name}

    def tracked_import(module_name: str):
        imported_modules.append(module_name)
        return real_import_module(module_name)

    monkeypatch.setattr(load_nnunet_model.importlib, "import_module", tracked_import)

    trainer_class = load_nnunet_model.resolve_python_class_exact(
        str(trainer_root),
        "nnUNetTrainer_4000epochs",
        "nnunetv2.training.nnUNetTrainer",
    )

    after = {name for name in sys.modules if ".from_nnunetv1." in name}

    assert trainer_class is not None
    assert trainer_class.__name__ == "nnUNetTrainer_4000epochs"
    assert not any(".from_nnunetv1." in module_name for module_name in imported_modules)
    assert after == before
