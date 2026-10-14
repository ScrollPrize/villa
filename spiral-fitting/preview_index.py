"""On-disk index pairing published preview surfaces with model states.

A published preview is the Lasagna-flattened surface of one exported model
state, and flattening it costs minutes. A checkpoint records that same model
state. Both carry the surface's content digest (``model_state_sha256``), so
when a checkpoint is loaded the service can look the digest up here and
re-show the flattened surface it already holds instead of asking for a new
export and a new flatten.

The index lives beside the published generations, under
``<output>/.spiral-published/``, so it survives the service process and its
in-memory artifact registry. It holds two maps:

``previews``
    digest -> the published generation's manifest, and where it came from.
``pins``
    checkpoint path -> digest. A generation whose digest a pinned checkpoint
    still names is exempt from the fixed-count retention that would
    otherwise delete it; a pin lapses when its checkpoint file disappears,
    and a checkpoint re-saved under the same path (the autosave) re-pins the
    new digest and releases the old one.

Every operation re-reads the file: it is small, it is written rarely, and
reading fresh is what keeps two service processes on the same output root
from clobbering each other's view. The read-modify-write itself is
serialized across threads and processes by an advisory lock file beside
the index, so one writer's update is never lost to another's, and two
writers never hold the temporary file that atomic replacement swaps in.
"""

from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
from pathlib import Path

from service_files import ExclusiveFileLock

INDEX_FILE_NAME = "preview-index.json"
LOCK_FILE_NAME = ".preview-index.lock"
INDEX_SCHEMA = "spiral-preview-index/1"
# The index is small and written briefly, so a real wait for it is short.
# The cap exists only so a holder that is wedged (not crashed: a crashed
# process releases its lock) surfaces as an error instead of a hang.
LOCK_TIMEOUT_SECONDS = 60.0


class PublishedPreviewIndex:
    def __init__(self, output_directory):
        self.path = Path(output_directory) / ".spiral-published" / INDEX_FILE_NAME
        self._lock = threading.Lock()
        # Serializes this instance's operations; the file lock below
        # serializes read-modify-write across every instance and process.
        self._file_lock = ExclusiveFileLock(
            self.path.with_name(LOCK_FILE_NAME))

    @contextmanager
    def _locked(self):
        """One whole read-modify-write cycle, serialized everywhere.

        The thread lock keeps one instance's threads from interleaving
        their cycles; the file lock extends the same critical section
        across every instance and process that shares the index file.
        """
        with self._lock:
            self._file_lock.acquire(timeout=LOCK_TIMEOUT_SECONDS)
            try:
                yield
            finally:
                self._file_lock.release()

    # -- storage ---------------------------------------------------------

    def _load(self):
        try:
            document = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            document = None
        if (not isinstance(document, dict)
                or document.get("schema") != INDEX_SCHEMA):
            document = {"schema": INDEX_SCHEMA, "previews": {}, "pins": {}}
        if not isinstance(document.get("previews"), dict):
            document["previews"] = {}
        if not isinstance(document.get("pins"), dict):
            document["pins"] = {}
        return document

    def _store(self, document):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.incoming")
        try:
            temporary.write_text(
                json.dumps(document, indent=2, sort_keys=True) + "\n",
                encoding="utf-8")
            os.replace(temporary, self.path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    # -- validation ------------------------------------------------------

    @staticmethod
    def _entry_is_live(digest, entry):
        """Whether a recorded generation is still on disk and still itself."""
        if not isinstance(entry, dict):
            return False
        manifest_path = Path(str(entry.get("manifest_path") or ""))
        if not manifest_path.is_file():
            return False
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        if not isinstance(manifest, dict) \
                or manifest.get("model_state_sha256") != digest:
            return False
        surface_path = manifest.get("surface_path")
        if surface_path and not Path(str(surface_path)).is_dir():
            return False
        return True

    def _drop_dead_entries_locked(self, document):
        previews = document["previews"]
        dead = [digest for digest, entry in previews.items()
                if not self._entry_is_live(digest, entry)]
        for digest in dead:
            del previews[digest]
        pins = document["pins"]
        lapsed = [path for path in pins if not Path(path).is_file()]
        for path in lapsed:
            del pins[path]
        return bool(dead or lapsed)

    # -- operations ------------------------------------------------------

    def record(self, digest, *, manifest_path, session_id, generation,
               source_fit_iteration=None):
        """Remember that ``manifest_path`` is the published surface of ``digest``."""
        if not digest:
            return
        with self._locked():
            document = self._load()
            self._drop_dead_entries_locked(document)
            document["previews"][str(digest)] = {
                "manifest_path": str(manifest_path),
                "session_id": str(session_id),
                "generation": int(generation),
                "source_fit_iteration": source_fit_iteration,
            }
            self._store(document)

    def pin(self, checkpoint_path, digest):
        """Record that the checkpoint at ``checkpoint_path`` has ``digest``."""
        if not checkpoint_path or not digest:
            return
        with self._locked():
            document = self._load()
            self._drop_dead_entries_locked(document)
            document["pins"][str(checkpoint_path)] = str(digest)
            self._store(document)

    def lookup(self, digest):
        """The live published entry for ``digest``, or None."""
        if not digest:
            return None
        with self._locked():
            document = self._load()
            if self._drop_dead_entries_locked(document):
                self._store(document)
            entry = document["previews"].get(str(digest))
            return dict(entry) if entry else None

    def pinned_roots(self):
        """Published generation directories a live checkpoint still names."""
        with self._locked():
            document = self._load()
            if self._drop_dead_entries_locked(document):
                self._store(document)
            wanted = set(document["pins"].values())
            roots = set()
            for digest, entry in document["previews"].items():
                if digest in wanted:
                    roots.add(
                        Path(str(entry["manifest_path"])).parent.resolve(
                            strict=False))
            return roots
