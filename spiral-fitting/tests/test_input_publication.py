"""Inject failures before/after each publication rename, then resume."""

import os
import pytest

from input_publication import Output, PublicationTransaction, fingerprint
from service_http import ApiError


def prepare(tmp_path):
    target = tmp_path / "dataset"
    source = tmp_path / "drafts"
    target.mkdir()
    source.mkdir()
    (target / "same.json").write_bytes(b"original collection")
    (source / "same.json").write_bytes(b"selected revision")
    (target / "fiber.json").write_bytes(b"old fiber")
    (source / "fiber.json").write_bytes(b"new fiber")
    (target / "patch").mkdir()
    (target / "patch" / "x.tif").write_bytes(b"old geometry")
    (source / "patch").mkdir()
    (source / "patch" / "x.tif").write_bytes(b"selected geometry")
    (target / "removed.json").write_bytes(b"delete me")
    outputs = [Output(target / name, source / name, fingerprint(target / name))
               for name in ("same.json", "patch", "fiber.json")]
    outputs.append(Output(target / "removed.json", None, fingerprint(target / "removed.json")))
    return PublicationTransaction.prepare(outputs), target, source


@pytest.mark.parametrize("failure_phase", range(1, 6))
@pytest.mark.parametrize("after_rename", [False, True])
def test_every_rename_failure_resumes_exact_outputs(tmp_path, monkeypatch, failure_phase, after_rename):
    transaction, target, source = prepare(tmp_path)
    replace = os.replace
    count = 0

    def injected(src, dst):
        nonlocal count
        count += 1
        if count == failure_phase and not after_rename:
            raise OSError("injected before rename")
        replace(src, dst)
        if count == failure_phase and after_rename:
            raise OSError("injected after rename")

    with monkeypatch.context() as fault:
        fault.setattr(os, "replace", injected)
        with pytest.raises(OSError, match="injected"):
            transaction.resume()
    with pytest.raises(RuntimeError, match="incomplete"):
        transaction.release()
    # New local edits cannot leak into a retained transaction.
    (source / "same.json").write_bytes(b"newer unsubmitted draft")
    transaction.resume()
    transaction.resume()  # Lost success response.
    assert (target / "same.json").read_bytes() == b"selected revision"
    assert (target / "fiber.json").read_bytes() == b"new fiber"
    assert (target / "patch" / "x.tif").read_bytes() == b"selected geometry"
    assert not (target / "removed.json").exists()
    assert all(entry["phase"] == "published" for entry in transaction.status())
    transaction.release()
    transaction.release()
    assert not list(target.glob(".spiral-publication-*"))


def test_changed_target_rejects_entire_prepared_batch(tmp_path):
    transaction, target, _ = prepare(tmp_path)
    (target / "fiber.json").write_bytes(b"external edit")
    with pytest.raises(ApiError, match="changed"):
        transaction.resume()
    assert (target / "same.json").read_bytes() == b"original collection"
    assert (target / "patch" / "x.tif").read_bytes() == b"old geometry"


def test_external_file_deletion_is_not_mistaken_for_a_completed_backup_rename(tmp_path):
    transaction, target, _ = prepare(tmp_path)
    (target / "same.json").unlink()
    with pytest.raises(ApiError, match="changed"):
        transaction.resume()
    assert not (target / "same.json").exists()
    assert (target / "fiber.json").read_bytes() == b"old fiber"


def test_published_target_is_verified_before_resuming_other_targets(tmp_path, monkeypatch):
    transaction, target, _ = prepare(tmp_path)
    original_replace = os.replace

    def interrupted(src, dst):
        original_replace(src, dst)
        raise OSError("lost response")

    with monkeypatch.context() as fault:
        fault.setattr(os, "replace", interrupted)
        with pytest.raises(OSError):
            transaction.resume()
    (target / "same.json").write_bytes(b"external post-publication edit")
    with pytest.raises(ApiError, match="changed"):
        transaction.resume()
    assert (target / "patch" / "x.tif").read_bytes() == b"old geometry"


def test_addition_and_deletion_of_missing_target_are_resumable(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"new")
    target = tmp_path / "target"
    missing = tmp_path / "missing"
    transaction = PublicationTransaction.prepare([
        Output(target, source, None), Output(missing, None, None)])
    transaction.resume()
    transaction.resume()
    assert target.read_bytes() == b"new"
    transaction.release()


def test_missing_source_preparation_publishes_nothing(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    with pytest.raises(ApiError, match="missing"):
        PublicationTransaction.prepare([
            Output(first, second, fingerprint(first)),
            Output(second, tmp_path / "missing", fingerprint(second))])
    assert first.read_bytes() == b"first"
    assert not list(tmp_path.glob(".spiral-publication-*"))


def test_directory_fingerprint_detects_empty_directory_and_name_changes(tmp_path):
    (tmp_path / "a").mkdir()
    before = fingerprint(tmp_path)
    (tmp_path / "a").rename(tmp_path / "b")
    assert before != fingerprint(tmp_path)


def test_rejects_symlinks_and_overlapping_targets(tmp_path):
    directory = tmp_path / "patch"
    directory.mkdir()
    (directory / "x.tif").write_bytes(b"data")
    with pytest.raises(ApiError, match="overlap"):
        PublicationTransaction.prepare([
            Output(directory, None, fingerprint(directory)),
            Output(directory / "x.tif", None, fingerprint(directory / "x.tif"))])
    (tmp_path / "link").symlink_to(directory)
    with pytest.raises(ApiError, match="symlink"):
        fingerprint(tmp_path / "link")
