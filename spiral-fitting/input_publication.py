"""In-process, resumable publication of prepared dataset outputs.

The caller holds the dataset commit lock during prepare/resume and serializes
all subsequent workspace mutations behind recovery. Targets and sources are
service-resolved paths, never unchecked client paths. Outputs have already
been derived from exact applied revisions and validated against their bases.

There is no crash journal and no claim of atomic multi-file visibility.
"""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
from uuid import uuid4

from service_http import ApiError, sha256_file


def fingerprint(path, *, file_digest=sha256_file):
    """Byte-exact file/directory identity; absence is represented by None."""
    path = Path(path)
    if path.is_symlink():
        raise ApiError(409, f"Managed input is a symlink: {path}")
    if not path.exists():
        return None
    if path.is_file():
        return "file:" + file_digest(path)
    if not path.is_dir():
        raise ApiError(409, f"Managed input is not a regular file or directory: {path}")
    entries = []
    for child in sorted(path.rglob("*")):
        relative = child.relative_to(path).as_posix()
        if child.is_symlink():
            raise ApiError(409, f"Managed input contains a symlink: {child}")
        if child.is_file():
            entries.append((relative, "file", file_digest(child)))
        elif child.is_dir():
            entries.append((relative, "directory"))
        else:
            raise ApiError(409, f"Unsupported managed input: {child}")
    return "directory:" + hashlib.sha256(
        json.dumps(entries, separators=(",", ":")).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Output:
    target: Path
    source: Path | None
    expected: str | None


@dataclass
class _Publication:
    output: Output
    directory: Path
    result: str | None
    phase: str = "prepared"

    @property
    def prepared(self):
        return self.directory / "prepared"

    @property
    def backup(self):
        return self.directory / "original"

    def conflict(self, current):
        raise ApiError(409, "Dataset target changed during publication",
                       payload={"target": str(self.output.target),
                                "expected": self.output.expected, "current": current})

    def reconcile(self):
        """Recognize renames completed before a lost/failed acknowledgement."""
        current = fingerprint(self.output.target)
        if self.phase == "published":
            if current != self.result:
                self.conflict(current)
            return
        expected = self.output.expected
        if self.result is None:
            # Deletion is a rename to retained backup. Never infer successful
            # deletion from absence without the original backup.
            if current is None and (expected is None or fingerprint(self.backup) == expected):
                self.phase = "published"
                return
        elif not self.prepared.exists() and current == self.result:
            self.phase = "published"
            return
        if (current is None and (expected or "").startswith("directory:")
                and fingerprint(self.backup) == expected):
            self.phase = "backed_up"
            return
        if self.phase == "backed_up" or current != expected:
            self.conflict(current)

    def publish(self):
        self.reconcile()
        if self.phase == "published":
            return
        target = self.output.target
        if self.result is None:
            os.replace(target, self.backup)
            self.phase = "published"
            return
        # Recheck prepared bytes too: no retry may publish an externally
        # modified temporary output or silently regenerate it from new input.
        if fingerprint(self.prepared) != self.result:
            raise ApiError(409, "Prepared dataset output changed",
                           payload={"target": str(target)})
        replacing_directory = (self.output.expected or "").startswith("directory:")
        if replacing_directory and self.phase == "prepared":
            os.replace(target, self.backup)
            self.phase = "backed_up"
        os.replace(self.prepared, target)
        self.phase = "published"


class PublicationTransaction:
    def __init__(self, publications):
        self._publications = tuple(publications)
        self.completed = False

    @classmethod
    def prepare(cls, outputs):
        outputs = tuple(Output(Path(o.target).absolute(),
                               Path(o.source).absolute() if o.source is not None else None,
                               o.expected) for o in outputs)
        targets = [output.target.resolve() for output in outputs]
        if any(first == second or first in second.parents or second in first.parents
               for index, first in enumerate(targets) for second in targets[index + 1:]):
            raise ApiError(400, "Publication targets must not overlap")
        publications = []
        try:
            # Validate every base before creating any prepared output.
            for output in outputs:
                current = fingerprint(output.target)
                if current != output.expected:
                    raise ApiError(409, "Dataset target changed before preparation",
                                   payload={"target": str(output.target), "current": current})
            for output in outputs:
                output.target.parent.mkdir(parents=True, exist_ok=True)
                directory = output.target.parent / f".spiral-publication-{uuid4().hex}"
                directory.mkdir()
                publication = _Publication(output, directory, None)
                publications.append(publication)
                if output.source is not None:
                    source_digest = fingerprint(output.source)
                    if source_digest is None:
                        raise ApiError(409, "Prepared output source is missing")
                    if output.source.is_dir():
                        shutil.copytree(output.source, publication.prepared)
                    else:
                        shutil.copy2(output.source, publication.prepared)
                    publication.result = fingerprint(publication.prepared)
                    if publication.result != source_digest:
                        raise ApiError(409, "Output source changed while being prepared")
                    if (output.expected is not None
                            and output.expected.split(":", 1)[0]
                            != publication.result.split(":", 1)[0]):
                        raise ApiError(409, "Managed input cannot change between file and directory")
                # File publication remains one atomic replace. Retain the
                # original before that replace, until the whole batch settles.
                if (output.expected or "").startswith("file:") and output.source is not None:
                    shutil.copy2(output.target, publication.backup)
                    if fingerprint(publication.backup) != output.expected:
                        raise ApiError(409, "Dataset target changed while being prepared")
            transaction = cls(publications)
            transaction._reconcile()
            return transaction
        except BaseException:
            for publication in publications:
                shutil.rmtree(publication.directory, ignore_errors=True)
            raise

    def _reconcile(self):
        for publication in self._publications:
            publication.reconcile()

    def resume(self):
        # Check *all* pending and completed outputs before publishing the next
        # one. A partially published transaction is never remerged on retry.
        self._reconcile()
        for publication in self._publications:
            publication.publish()
        self.completed = True

    def status(self):
        return [{"target": str(p.output.target), "phase": p.phase,
                 "expected": p.output.expected, "result": p.result}
                for p in self._publications]

    def recovery_paths(self):
        return [p.directory for p in self._publications if p.directory.exists()]

    def release(self):
        """Call only after recording persistence of every selected revision."""
        if not self.completed:
            raise RuntimeError("Cannot release an incomplete dataset transaction")
        for publication in self._publications:
            if publication.directory.exists():
                shutil.rmtree(publication.directory)
