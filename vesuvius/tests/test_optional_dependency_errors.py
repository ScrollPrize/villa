"""Names that vanish with optional dependencies should say what to install.

Regression: `vesuvius/__init__.py` bound `list_files`, `list_cubes`,
`update_list`, `is_aws_ec2_instance` and `VCDataset` to None when their
optional dependencies were absent, which a bare ``pip install vesuvius`` and
the ``volume-only`` extra both are. Calling one then raised
"'NoneType' object is not callable", which names neither the missing
dependency nor the extra that provides it.
"""

import pytest

import vesuvius


@pytest.mark.unit
def test_placeholder_raises_importerror_naming_the_extra():
    cause = ModuleNotFoundError("No module named 'aiohttp'")
    stub = vesuvius._requires_extra("list_files", "all", cause)
    with pytest.raises(ImportError) as excinfo:
        stub()
    message = str(excinfo.value)
    assert "list_files" in message
    assert "vesuvius[all]" in message
    assert excinfo.value.__cause__ is cause


@pytest.mark.unit
def test_placeholder_keeps_the_name_it_stands_in_for():
    stub = vesuvius._requires_extra("VCDataset", "models", ImportError("no torch"))
    assert stub.__name__ == "VCDataset"
    assert "models" in (stub.__doc__ or "")


@pytest.mark.unit
@pytest.mark.parametrize(
    "name", ["list_files", "list_cubes", "update_list", "is_aws_ec2_instance", "VCDataset"]
)
def test_exported_names_are_never_none(name):
    """Whether or not the extras are installed, these must be callable."""
    assert name in vesuvius.__all__
    obj = getattr(vesuvius, name)
    assert obj is not None, f"{name} is None; calling it cannot explain itself"
    assert callable(obj)
