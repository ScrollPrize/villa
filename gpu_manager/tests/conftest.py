import pytest


@pytest.fixture(autouse=True)
def isolated_constraints(tmp_path, monkeypatch):
    # Synthetic inventories must never inherit a real host's administrative policy.
    monkeypatch.setenv("VILLA_GPU_MANAGER_CONSTRAINTS", str(tmp_path / "constraints.json"))
