"""Immutable generated-data directories published as opaque artifacts.

The registry is self-contained: it knows about directories, digests and
in-flight downloads, and nothing about sessions, uploads or the service
lifecycle. The lifecycle orchestrator supplies the ``kind``/``session_id``
namespacing and the retention policy; the registry only enforces them.
"""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
import os
from pathlib import Path
import secrets
import shutil
import threading
import time

from service_http import ApiError, resolve_inside, sha256_file


MAX_ARTIFACT_FILES = 4096
MAX_PRUNED_IDS_REMEMBERED = 4096


class Artifact:
    __slots__ = ("artifact_id", "kind", "session_id", "generation", "root",
                 "files", "entry_point", "inflight", "pruned",
                 "delete_root_on_prune", "created")

    def __init__(self, artifact_id, kind, session_id, generation, root,
                 files, entry_point, delete_root_on_prune):
        self.artifact_id = artifact_id
        self.kind = kind
        self.session_id = session_id
        self.generation = generation
        self.root = root
        self.files = files
        self.entry_point = entry_point
        self.inflight = 0
        self.pruned = False
        self.delete_root_on_prune = delete_root_on_prune
        self.created = time.time()

    def ref(self):
        return {"id": self.artifact_id, "kind": self.kind,
                "generation": self.generation}

    def manifest(self):
        return {
            "schema_version": 1,
            "id": self.artifact_id,
            "kind": self.kind,
            "session_id": self.session_id,
            "generation": self.generation,
            "entry_point": self.entry_point,
            "files": [
                {"name": name, "size": info["size"], "sha256": info["sha256"]}
                for name, info in sorted(self.files.items())
            ],
        }


class ArtifactRegistry:
    """Immutable generated-data directories addressed by opaque IDs.

    Files are digested once at registration (inside the fitter's pause/export
    window). Pruning never removes an artifact while a download holds an
    in-flight reference; a pruned ID answers ``410 Gone``.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._artifacts = OrderedDict()
        self._pruned_ids = OrderedDict()

    def register_directory(self, kind, session_id, generation, root,
                           entry_point, *, delete_root_on_prune=False,
                           progress=None, hash_workers=1):
        root = Path(root).resolve(strict=True)
        paths = []
        for directory, dirnames, filenames in os.walk(root, followlinks=False):
            dirnames.sort()
            for filename in sorted(filenames):
                path = Path(directory) / filename
                if path.is_symlink() or not path.is_file():
                    continue
                paths.append(path)
                if len(paths) > MAX_ARTIFACT_FILES:
                    raise ApiError(HTTPStatus.INTERNAL_SERVER_ERROR,
                                   "Artifact has too many files to register")

        def digest(path):
            return path, path.stat().st_size, sha256_file(path)

        files = {}
        workers = max(1, min(int(hash_workers), len(paths) or 1))
        with ThreadPoolExecutor(max_workers=workers,
                                thread_name_prefix="spiral-artifact-hash") as executor:
            for index, (path, size, sha256) in enumerate(
                    executor.map(digest, paths), start=1):
                relative = path.relative_to(root).as_posix()
                files[relative] = {"size": size, "sha256": sha256}
                if progress is not None:
                    progress(index, len(paths), relative)
        if entry_point not in files:
            raise ApiError(HTTPStatus.INTERNAL_SERVER_ERROR,
                           f"Artifact entry point {entry_point!r} was not found")
        artifact_id = f"{kind}-{generation}-{secrets.token_hex(8)}"
        artifact = Artifact(artifact_id, kind, session_id, generation, root,
                            files, entry_point, delete_root_on_prune)
        with self._lock:
            self._artifacts[artifact_id] = artifact
        return artifact.ref()

    def _get(self, artifact_id):
        artifact = self._artifacts.get(artifact_id)
        if artifact is None or artifact.pruned:
            if artifact is not None or artifact_id in self._pruned_ids:
                raise ApiError(HTTPStatus.GONE, "Artifact has been pruned")
            raise ApiError(HTTPStatus.NOT_FOUND, "Unknown artifact")
        return artifact

    def manifest(self, artifact_id):
        with self._lock:
            return self._get(artifact_id).manifest()

    def acquire_file(self, artifact_id, relative_name):
        """Return ``(artifact, path, info)`` holding an in-flight reference."""
        with self._lock:
            artifact = self._get(artifact_id)
            info = artifact.files.get(relative_name)
            if info is None:
                raise ApiError(HTTPStatus.NOT_FOUND,
                               "The artifact does not contain this file")
            artifact.inflight += 1
        try:
            path = resolve_inside(artifact.root, relative_name)
        except BaseException:
            self.release(artifact)
            raise
        return artifact, path, info

    def release(self, artifact):
        delete_root = None
        with self._lock:
            artifact.inflight -= 1
            if artifact.pruned and artifact.inflight == 0 and artifact.delete_root_on_prune:
                delete_root = artifact.root
        if delete_root is not None:
            shutil.rmtree(delete_root, ignore_errors=True)

    def retire_root(self, root, timeout=15):
        """Reject new readers and drain existing readers before external cleanup."""
        root = Path(root).resolve()
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                matching = [a for a in self._artifacts.values()
                            if a.root.is_relative_to(root)]
                for artifact in matching:
                    artifact.pruned = True
                if not any(a.inflight for a in matching):
                    for artifact in matching:
                        del self._artifacts[artifact.artifact_id]
                        self._pruned_ids[artifact.artifact_id] = True
                    return
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Artifact readers did not stop; retained {root}")
            time.sleep(0.05)

    def find(self, kind, root, session_id=None):
        """The live artifact of one kind already registered on ``root``.

        Returns its reference, or None; ``session_id`` narrows the search to
        one session. Two artifacts must never both own one directory, since
        pruning either would delete the other's files.
        """
        root = Path(root).resolve(strict=False)
        with self._lock:
            for artifact in self._artifacts.values():
                if (artifact.kind == kind and Path(artifact.root) == root
                        and (session_id is None
                             or artifact.session_id == session_id)):
                    return artifact.ref()
        return None

    def prune(self, kind, session_id, keep, retain=None):
        """Prune all but the newest ``keep`` artifacts of one kind.

        ``retain(artifact)`` may exempt an older artifact from this pass; it
        stays registered and is reconsidered at the next one.
        """
        to_delete = []
        with self._lock:
            matching = [a for a in self._artifacts.values()
                        if a.kind == kind and a.session_id == session_id]
            matching.sort(key=lambda a: a.generation)
            for artifact in matching[:-keep] if keep else matching:
                if retain is not None and retain(artifact):
                    continue
                del self._artifacts[artifact.artifact_id]
                self._pruned_ids[artifact.artifact_id] = True
                while len(self._pruned_ids) > MAX_PRUNED_IDS_REMEMBERED:
                    self._pruned_ids.popitem(last=False)
                artifact.pruned = True
                if artifact.delete_root_on_prune and artifact.inflight == 0:
                    to_delete.append(artifact.root)
        for root in to_delete:
            shutil.rmtree(root, ignore_errors=True)
