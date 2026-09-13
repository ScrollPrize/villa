"""Dataset editing workspace, independent of the resident fit generation.

The coordinator owns acceptance/application/publication. Uploads and status
remain concurrent. Every catalog content path is an immutable workspace copy.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import threading
from uuid import UUID, uuid4, uuid5

from input_publication import Output, PublicationTransaction, fingerprint
from input_workspace import Catalog, Change, Content, InputIdentity, MutationCoordinator, WorkspaceLease
from service_files import ExclusiveFileLock
from service_http import ApiError
from service_uploads import (PCL_ROLE_FILES, UploadEnvironment, UploadManager,
                             collection_has_affected_links, _validate_replacement_document)


class EditingWorkspace:
    def __init__(self, dataset, output, sources, resident, influence=None):
        self.dataset = Path(dataset).resolve()
        self.id = str(uuid4())
        self.root = Path(output) / 'editing-workspaces' / self.id
        self.sources = copy.deepcopy(sources)
        self.resident = resident
        self.influence = influence or (lambda: {})
        self.application_influence = {}
        self.catalog = Catalog()
        self.coordinator = MutationCoordinator()
        self.lease = WorkspaceLease(self.dataset)
        self.seeded = False
        self.transactions = {}
        self.accepted_commands = {}
        self.commit_selections = {}
        self.lifecycle_started = {}
        self.resident_generation = None
        self.reviewed_bases = {}
        self.uploads = UploadManager(UploadEnvironment(
            lock=threading.RLock(), output_root=lambda: self.root,
            session_id=lambda: self.id, ephemeral_dir=lambda: self.root / 'content',
            require_session=lambda: None, immutable_content=True))

    def claim(self, token, command_id):
        def claim(_):
            self.lease.claim(token)
            already_seeded = self.seeded
            if not self.seeded:
                try:
                    self._seed()
                except BaseException:
                    self.catalog = Catalog()
                    shutil.rmtree(self.root, ignore_errors=True)
                    self.lease.close()
                    raise
            if already_seeded:
                self.refresh_clean(f'{command_id}:refresh')
            return {'workspace_id': self.id, 'owned': True}
        return self.coordinator.execute(command_id, 'claim',
            {'token_digest': hashlib.sha256(str(token).encode()).hexdigest()}, claim,
            recoverable=lambda exc: isinstance(exc, TimeoutError))

    def refresh_clean(self, command_id):
        """Reconcile externally changed clean targets on explicit reconnect.

        Local client revisions remain separate; a dirty client will review a
        changed accepted target instead of losing its draft. Unrelated PCL
        changes do not create a new revision for this collection.
        """
        try:
            self.resident()
        except ApiError as exc:
            if int(exc.status) == 409:
                return
            raise
        revisions = self.accepted_commands.get(command_id)
        if revisions is None:
            self.catalog.register_external_bases(self._discover_inputs())
            documents, changes = {}, []
            for entry in self.catalog.entries():
                if entry.accepted != entry.persisted:
                    continue
                identity = entry.identity
                if identity.kind == 'pcl':
                    if identity.source not in documents:
                        target = self._target(identity.source)
                        documents[identity.source] = json.loads(target.read_text()) if target.exists() else {'collections': {}}
                    document = documents[identity.source]
                    current = document.get('collections', {}).get(str(identity.collection_id))
                    previous = self._collection(entry.current, identity) if entry.current.content else None
                else:
                    current, document = self._external(identity)
                    previous = entry.current.content.json()['fingerprint'] if entry.current.content else None
                if current == previous:
                    continue
                content = None
                if current is not None:
                    destination = self.root / 'refreshed' / str(uuid4())
                    content = (self._json_content({**document, 'collections': {str(identity.collection_id): current}},
                                                 destination.with_suffix('.json')) if identity.kind == 'pcl'
                               else self._copy(identity.source, destination))
                changes.append(Change(identity, entry.accepted, content))
            changed = self.catalog.accept(changes) if changes else ()
            changed_ids = {r.id for r in changed}
            revisions = (*changed, *(e.current for e in self.catalog.entries()
                if e.applied < e.accepted and e.accepted == e.persisted
                and e.identity.id not in changed_ids))
            self.accepted_commands[command_id] = revisions
        if revisions and self._apply(command_id, revisions).get('applied'):
            self.catalog.mark_persisted(revisions)

    def require(self, token):
        self.lease.require(token)

    def release(self, token, command_id):
        return self.coordinator.execute(command_id, 'release',
            {'token_digest': hashlib.sha256(str(token).encode()).hexdigest()},
            lambda _: (self.lease.release(token) or {'released': True}))

    def _target(self, value):
        path = Path(value)
        if path.is_symlink():
            raise ApiError(409, 'Managed input is a symlink')
        target = path.resolve()
        if not target.is_relative_to(self.dataset) or target == self.dataset:
            raise ApiError(403, 'Input target is outside the managed dataset')
        return target

    def _copy(self, source, destination):
        before = fingerprint(source)
        if before is None:
            raise ApiError(409, 'Input source disappeared while taking a snapshot')
        destination.parent.mkdir(parents=True, exist_ok=True)
        if Path(source).is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)
        if fingerprint(destination) != before or fingerprint(source) != before:
            raise ApiError(409, 'Input source changed while taking a snapshot')
        return Content.from_json({'path': str(destination), 'fingerprint': before})

    def _json_content(self, document, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(document, ensure_ascii=False, allow_nan=False))
        return Content.from_json({'path': str(destination), 'fingerprint': fingerprint(destination)})

    def _seed(self):
        self.catalog.register_external_bases(self._discover_inputs(), applied=True)
        self.seeded = True

    def _discover_inputs(self):
        """Snapshot unknown dataset targets for both startup and reconnect."""
        known = {(e.identity.source, e.identity.collection_id) for e in self.catalog.entries()}
        discovered = []
        namespace = UUID(self.id)
        snapshot_root = self.root / 'base' / str(uuid4())
        specs = list(self.sources.get('pcl_inputs', []))
        paths = {str(Path(spec['path']).resolve()) for spec in specs}
        specs.extend({'path': str(self.dataset / name), 'role': role}
                     for role, name in PCL_ROLE_FILES.items()
                     if str((self.dataset / name).resolve()) not in paths)
        for spec in specs:
            path = Path(spec['path'])
            if not path.is_file():
                continue
            before = fingerprint(path)
            document = json.loads(path.read_text())
            for key, collection in sorted(document.get('collections', {}).items(), key=lambda item: int(item[0])):
                if (str(path.resolve()), int(key)) in known:
                    continue
                input_id = str(uuid5(namespace, f'{path.resolve()}:{key}'))
                content = self._json_content({**document, 'collections': {key: collection}},
                                             snapshot_root / f'{input_id}.json')
                discovered.append((InputIdentity(input_id, 'pcl', str(path.resolve()),
                                                        spec.get('role'), int(key)), content))
            if fingerprint(path) != before:
                raise ApiError(409, 'PCL source changed while snapshotting the editing workspace')
        resolved = self.sources.get('resolved', {})
        for key, kind, role in [('verified_patches', 'patch', 'verified'),
                                ('unverified_patches', 'patch', 'unverified'),
                                ('fibers', 'fiber', None)]:
            directory = Path(resolved.get(key) or self.dataset / key)
            if not directory.is_dir():
                continue
            for source in sorted(directory.iterdir()):
                if kind == 'patch' and not (source / 'meta.json').is_file():
                    continue
                if kind == 'fiber' and (not source.is_file() or source.suffix != '.json'):
                    continue
                if (str(source.resolve()), None) in known:
                    continue
                input_id = str(uuid5(namespace, str(source.resolve())))
                destination = snapshot_root / (input_id if kind == 'patch' else f'{input_id}.json')
                discovered.append((InputIdentity(input_id, kind, str(source.resolve()), role),
                                   self._copy(source, destination)))
        return discovered

    def status(self):
        return {'workspace_id': self.id, 'ready': self.seeded,
                'inputs': self.catalog.status(),
                'transactions': {key: transaction.status() for key, transaction in tuple(self.transactions.items())}}

    def _identity(self, change):
        if not isinstance(change, dict) or not isinstance(change.get('id'), str):
            raise ApiError(400, 'Each change needs a logical input UUID')
        input_id = change['id']
        try:
            return self.catalog.entry(input_id).identity
        except KeyError:
            kind, role = change.get('kind'), change.get('role')
            name = change.get('name') or input_id
            if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,127}', name):
                raise ApiError(400, 'Input name must be a safe file name')
            if kind == 'pcl' and role in PCL_ROLE_FILES:
                target = self.dataset / PCL_ROLE_FILES[role]
            elif kind == 'patch' and role in {'verified', 'unverified', None}:
                role = role or 'verified'
                target = Path(self.sources.get('resolved', {}).get(f'{role}_patches') or self.dataset / f'{role}_patches') / name
            elif kind == 'fiber':
                target = Path(self.sources.get('resolved', {}).get('fibers') or self.dataset / 'fibers') / f'{name.removesuffix(".json")}.json'
            else:
                raise ApiError(400, 'Invalid input kind or role')
            try:
                return InputIdentity(input_id, kind, str(target), role)
            except (ValueError, TypeError, AttributeError) as exc:
                raise ApiError(400, 'Input id must be a canonical UUID') from exc

    def _changes(self, request):
        result = []
        items = request.get('changes', [])
        if not isinstance(items, list):
            raise ApiError(400, 'Changes must be a list')
        errors = []
        for item in items:
            try:
                result.append(self._change(item))
            except ApiError as exc:
                if int(exc.status) != 400:
                    raise
                errors.append({'field': item.get('id', '') if isinstance(item, dict) else '',
                               'message': exc.message})
        if errors:
            raise ApiError(400, 'Repair the selected input drafts', errors)
        targets = {}
        for entry in self.catalog.entries():
            if entry.identity.kind != 'pcl':
                targets[entry.identity.source] = entry.identity.id
        for change in result:
            identity = change.identity
            if identity.kind != 'pcl':
                if targets.get(identity.source, identity.id) != identity.id:
                    raise ApiError(409, 'Two logical inputs name the same persistence target')
                targets[identity.source] = identity.id
        for source in dict.fromkeys(c.identity.source for c in result
                                    if c.identity.kind == 'pcl' and c.expected == 0):
            target = self._target(source)
            if target.exists():
                document = json.loads(target.read_text())
                self.catalog.reserve_collection_ids(source, [int(key) for key in document.get('collections', {})])
        return result

    def _change(self, item):
        identity = self._identity(item)
        self._target(identity.source)
        if item.get('deleted'):
            content = None
        elif 'restore_revision' in item:
            entry = self.catalog.entry(identity.id)
            number = item['restore_revision']
            if entry.deleted and entry.accepted == entry.persisted:
                raise ApiError(409, 'Committed deletion must be saved as a new input')
            if type(number) is not int or not 1 <= number <= entry.accepted:
                raise ApiError(400, 'Unknown restore revision')
            content = entry.revisions[number - 1].content
            if content is None:
                raise ApiError(400, 'Restore must select a content revision')
        else:
            upload = self.uploads.get(item.get('upload_id'))
            with upload.lock:
                if upload.record is None or upload.kind != identity.kind:
                    raise ApiError(409, 'Finalize matching immutable content before applying')
                if identity.kind == 'pcl' and upload.role != identity.role:
                    raise ApiError(400, 'Upload role does not match the input')
                path = Path(upload.record['path'])
                if identity.kind == 'pcl':
                    document = json.loads(path.read_text())
                    collections = document.get('collections', {})
                    if len(collections) != 1:
                        raise ApiError(400, 'Each logical PCL must contain exactly one collection')
                    key = next(iter(collections))
                    if identity.role in {'same_winding', 'relative'}:
                        _validate_replacement_document(document, key, identity.role)
                content = Content.from_json({'path': str(path), 'fingerprint': fingerprint(path)})
        return Change(identity, item.get('expected_revision'), content)

    def _selection(self, pairs):
        revisions = []
        for pair in pairs:
            try:
                entry = self.catalog.entry(pair['id'])
                number = pair['revision']
                if type(number) is not int or not 1 <= number <= entry.accepted:
                    raise KeyError(number)
                revisions.append(entry.revisions[number - 1])
            except (KeyError, TypeError) as exc:
                raise ApiError(409, 'Unknown selected input revision') from exc
        if not revisions or len({r.id for r in revisions}) != len(revisions):
            raise ApiError(400, 'Select one revision per input')
        return tuple(revisions)

    def _document(self, revision):
        content = revision.content.json()
        if fingerprint(content['path']) != content['fingerprint']:
            raise ApiError(409, 'Immutable input content changed')
        return json.loads(Path(content['path']).read_text())

    def _collection(self, revision, identity):
        collection = copy.deepcopy(next(iter(self._document(revision)['collections'].values())))
        if 'id' in collection:
            collection['id'] = (identity.collection_id if type(collection['id']) is int
                                else str(identity.collection_id))
        return collection

    def _records(self, revisions):
        records = []
        for revision in revisions:
            identity = self.catalog.entry(revision.id).identity
            record = {'id': identity.id, 'kind': identity.kind, 'role': identity.role,
                      'revision': revision.number, 'source_path': identity.source,
                      'source_id': str(identity.collection_id) if identity.kind == 'pcl' else
                                   (Path(identity.source).stem if identity.kind == 'fiber' else Path(identity.source).name),
                      'deleted': revision.content is None}
            if revision.content is not None:
                content = revision.content.json()
                if fingerprint(content['path']) != content['fingerprint']:
                    raise ApiError(409, 'Immutable input content changed')
                record['path'] = content['path']
                if identity.kind == 'pcl':
                    document = self._document(revision)
                    collection = self._collection(revision, identity)
                    path = self.root / 'resident' / identity.id / f'{revision.number}.json'
                    if not path.exists():
                        self._json_content({**document, 'collections': {str(identity.collection_id): collection}}, path)
                    record['path'] = str(path)
            records.append(record)
        return records

    def _apply(self, command_id, revisions):
        for revision in revisions:
            entry = self.catalog.entry(revision.id)
            if revision.number < entry.applied and revision.number not in entry.applied_history:
                raise ApiError(409, 'An unapplied older revision cannot replace a newer resident revision')
        pending = [r for r in revisions if self.catalog.entry(r.id).applied < r.number]
        if not pending:
            return {'applied': True}
        if command_id not in self.application_influence:
            self.application_influence[command_id] = copy.deepcopy(self.influence())
        result = self.resident().apply_input_changes(command_id, self._records(pending),
            influence_config=self.application_influence[command_id])
        if result.get('applied'):
            self.catalog.mark_applied(pending)
        else:
            self.catalog.record_error(pending, 'apply', json.dumps(result.get('errors', {})))
        return result

    def change(self, token, request):
        self.require(token)
        command_id = request.get('command_id')
        def perform(captured):
            self.require(token)
            revisions = self.accepted_commands.get(command_id)
            if revisions is None:
                selected = self._selection(captured['revisions']) if captured.get('revisions') else ()
                changes = self._changes(captured)
                if {r.id for r in selected}.intersection(c.identity.id for c in changes):
                    raise ApiError(400, 'Select each logical input only once')
                try:
                    revisions = (*selected, *self.catalog.accept(changes))
                except ApiError as exc:
                    if exc.payload.get('conflicts'):
                        conflicts = []
                        for conflict in exc.payload['conflicts']:
                            try:
                                entry = self.catalog.entry(conflict['id'])
                            except KeyError:
                                conflicts.append(conflict)
                                continue
                            current, _ = self._external(entry.identity)
                            change = next(c for c in changes if c.identity.id == entry.identity.id)
                            base = (entry.revisions[change.expected - 1].content.json()
                                    if type(change.expected) is int and 0 < change.expected <= entry.accepted
                                    and entry.revisions[change.expected - 1].content else None)
                            conflicts.append({**conflict, 'expected_revision': entry.accepted,
                                'review_token': self._review_token(current), 'base': base, 'current': current,
                                'local': change.content.json() if change.content else None,
                                'actions': ['use_current', 'apply_local_after_review', 'save_as_new']})
                        raise ApiError(409, exc.message, payload={'conflicts': conflicts}) from exc
                    raise
                self.accepted_commands[command_id] = revisions
            result = self._apply(command_id, revisions)
            return {**result, 'workspace_id': self.id, 'revisions': [
                {'id': r.id, 'revision': r.number} for r in revisions], 'catalog': [self.catalog.entry(r.id).status() for r in revisions]}
        return self.coordinator.execute(command_id, 'change', request, perform,
                                         recoverable=lambda exc: isinstance(exc, TimeoutError))

    def apply(self, token, request):
        self.require(token)
        def perform(captured):
            self.require(token)
            return self._apply(captured['command_id'], self._selection(captured.get('revisions', [])))
        return self.coordinator.execute(request.get('command_id'), 'apply', request, perform,
            recoverable=lambda exc: isinstance(exc, TimeoutError))

    def _conflict(self, identity, base, current, local):
        raise ApiError(409, 'The managed input changed outside this workspace', payload={
            'code': 'input_revision_conflict', 'conflicts': [{'id': identity.id,
                'base': base, 'current': current, 'local': local,
                'expected_revision': self.catalog.entry(identity.id).accepted,
                'review_token': self._review_token(current),
                'actions': ['discard_local', 'save_as_new'] if current is None else
                           ['use_current', 'apply_local_after_review', 'save_as_new']}]})

    def _outputs(self, revisions, command_id):
        grouped = {}
        for revision in revisions:
            identity = self.catalog.entry(revision.id).identity
            grouped.setdefault(identity.source, []).append(revision)
        outputs = []
        for source, group in grouped.items():
            target = self._target(source)
            before = fingerprint(target)
            identity = self.catalog.entry(group[0].id).identity
            if identity.kind == 'pcl':
                current = json.loads(target.read_text()) if target.exists() else {
                    'vc_pointcollections_json_version': '1', 'collections': {}}
                merged = copy.deepcopy(current)
                for revision in group:
                    entry = self.catalog.entry(revision.id)
                    key = str(entry.identity.collection_id)
                    persisted = entry.revisions[entry.persisted - 1] if entry.persisted else None
                    base = (self._collection(persisted, entry.identity)
                            if persisted and persisted.content else None)
                    reviewed = self.reviewed_bases.get(entry.identity.id)
                    if reviewed and reviewed[0] == entry.persisted:
                        base = reviewed[1]
                    found = current['collections'].get(key)
                    local = self._collection(revision, entry.identity) if revision.content else None
                    if found != base or (found is not None and collection_has_affected_links(current['collections'], key)):
                        self._conflict(entry.identity, base, found, local)
                    if local is None:
                        merged['collections'].pop(key, None)
                    else:
                        merged['collections'][key] = local
                staged = self.root / 'publication' / hashlib.sha256(command_id.encode()).hexdigest() / f'{group[0].id}.json'
                self._json_content(merged, staged)
            else:
                if len(group) != 1:
                    raise ApiError(409, 'Two logical inputs name the same persistence target')
                revision = group[0]
                entry = self.catalog.entry(revision.id)
                persisted = entry.revisions[entry.persisted - 1] if entry.persisted else None
                base = persisted.content.json()['fingerprint'] if persisted and persisted.content else None
                reviewed = self.reviewed_bases.get(entry.identity.id)
                if reviewed and reviewed[0] == entry.persisted:
                    base = reviewed[1]
                if before != base:
                    self._conflict(identity, base, before, revision.content.json() if revision.content else None)
                staged = Path(revision.content.json()['path']) if revision.content else None
            outputs.append(Output(target, staged, before))
        return outputs

    def commit(self, token, request):
        self.require(token)
        command_id = request.get('command_id')
        def perform(captured):
            self.require(token)
            revisions = self.commit_selections.setdefault(command_id,
                self._selection(captured.get('revisions', [])))
            for revision in revisions:
                if revision.number < self.catalog.entry(revision.id).persisted:
                    raise ApiError(409, 'Cannot persist an older revision over a newer committed revision')
            pending = tuple(r for r in revisions if r.number > self.catalog.entry(r.id).persisted)
            if not pending and command_id not in self.transactions:
                return {'committed': [r.id for r in revisions], 'catalog': [self.catalog.entry(r.id).status() for r in revisions]}
            if command_id not in self.transactions:
                result = self._apply(f'{command_id}:apply', revisions)
                if not result.get('applied'):
                    raise ApiError(409, 'Selected revisions did not apply', payload=result)
            with ExclusiveFileLock(self.dataset / '.spiral-commit.lock') as lock:
                transaction = self.transactions.get(command_id)
                if transaction is None:
                    transaction = PublicationTransaction.prepare(self._outputs(pending, command_id))
                    self.transactions[command_id] = transaction
                transaction.resume()
                self.catalog.mark_persisted(revisions)
                transaction.release()
                del self.transactions[command_id]
            return {'committed': [r.id for r in revisions], 'catalog': [self.catalog.entry(r.id).status() for r in revisions]}
        try:
            return self.coordinator.execute(command_id, 'commit', request, perform,
                recoverable=lambda exc: command_id in self.transactions or isinstance(exc, TimeoutError))
        except Exception as exc:
            revisions = self.commit_selections.get(command_id, ())
            self.catalog.record_error(revisions, 'commit', str(exc))
            if command_id in self.transactions:
                raise ApiError(503, 'Dataset publication needs recovery', payload={
                    'command_id': command_id, 'transaction': self.transactions[command_id].status()}) from exc
            raise

    def discard(self, token, request):
        """Discard selected workspace changes by restoring current dataset bytes."""
        self.require(token)
        command_id = request.get('command_id')
        def perform(captured):
            self.require(token)
            revisions = self.accepted_commands.get(command_id)
            if revisions is None:
                selected = self._selection(captured['revisions']) if captured.get('revisions') else ()
                changes = []
                for revision in selected:
                    entry = self.catalog.entry(revision.id)
                    if entry.accepted != revision.number:
                        raise ApiError(409, 'Select the latest accepted revision when discarding')
                    current, document = self._external(entry.identity)
                    content = None
                    if current is not None:
                        destination = self.root / 'discarded' / str(uuid4())
                        if entry.identity.kind == 'pcl':
                            content = self._json_content({**document, 'collections': {
                                str(entry.identity.collection_id): current}}, destination.with_suffix('.json'))
                        else:
                            content = self._copy(entry.identity.source, destination)
                    changes.append(Change(entry.identity, entry.accepted, content))
                revisions = self.catalog.accept(changes) if changes else ()
                self.accepted_commands[command_id] = revisions
            result = self._apply(command_id, revisions)
            if not result.get('applied'):
                return {**result, 'catalog': [self.catalog.entry(r.id).status() for r in revisions]}
            for revision in revisions:
                entry = self.catalog.entry(revision.id)
                current, _ = self._external(entry.identity)
                expected = (self._collection(revision, entry.identity) if entry.identity.kind == 'pcl'
                            else revision.content.json()['fingerprint']) if revision.content else None
                if current != expected:
                    self._conflict(entry.identity, expected, current, None)
            self.catalog.mark_persisted(revisions)
            return {'discarded': True, 'catalog': [self.catalog.entry(r.id).status() for r in revisions]}
        return self.coordinator.execute(command_id, 'discard', request, perform,
            recoverable=lambda exc: isinstance(exc, TimeoutError))

    @staticmethod
    def _review_token(current):
        return hashlib.sha256(json.dumps(current, sort_keys=True, separators=(',', ':'),
                                         allow_nan=False).encode()).hexdigest()

    def _external(self, identity):
        target = self._target(identity.source)
        if identity.kind == 'pcl':
            document = json.loads(target.read_text()) if target.exists() else {'collections': {}}
            return document.get('collections', {}).get(str(identity.collection_id)), document
        return fingerprint(target), None

    def resolve_conflict(self, token, request):
        self.require(token)
        def perform(captured):
            self.require(token)
            entry = self.catalog.entry(captured.get('id'))
            retained = self.accepted_commands.get(captured['command_id'])
            if retained is None and entry.accepted != captured.get('expected_revision'):
                raise ApiError(409, 'Review the newer accepted revision before resolving this input')
            current, document = self._external(entry.identity)
            if captured.get('review_token') != self._review_token(current):
                self._conflict(entry.identity, None, current, entry.current.content.json() if entry.current.content else None)
            action = captured.get('action')
            if action == 'apply_local_after_review':
                # Retain the exact reviewed comparison base. Another external
                # edit still conflicts; review never means unconditional overwrite.
                self.reviewed_bases[entry.identity.id] = (entry.persisted, current)
            elif action == 'use_current':
                content = None
                if retained is None and current is not None:
                    destination = self.root / 'reviewed' / str(uuid4())
                    if entry.identity.kind == 'pcl':
                        content = self._json_content({**document, 'collections': {
                            str(entry.identity.collection_id): current}}, destination.with_suffix('.json'))
                    else:
                        content = self._copy(entry.identity.source, destination)
                if retained is None:
                    retained = self.catalog.accept([Change(entry.identity, entry.accepted, content)])
                    self.accepted_commands[captured['command_id']] = retained
                revision, = retained
                result = self._apply(captured['command_id'], (revision,))
                if not result.get('applied'):
                    return {**result, 'catalog': [self.catalog.entry(entry.identity.id).status()]}
                latest, _ = self._external(entry.identity)
                if latest != current:
                    self._conflict(entry.identity, current, latest, current)
                self.catalog.mark_persisted((revision,))
                self.reviewed_bases.pop(entry.identity.id, None)
            else:
                raise ApiError(400, 'Choose use_current or apply_local_after_review')
            return {'resolved': True, 'catalog': [self.catalog.entry(entry.identity.id).status()]}
        return self.coordinator.execute(request.get('command_id'), 'resolve_conflict', request, perform,
            recoverable=lambda exc: isinstance(exc, TimeoutError))

    def replay_resident(self, generation):
        """Bind baseline identities and replay desired revisions after a rebuild.

        Adoption shares resident geometry for unchanged baseline inputs and
        retains disabled inputs without requiring them to participate.
        """
        if self.resident_generation == generation:
            return {'applied': True}
        revisions = self.catalog.desired()
        if not revisions:
            self.resident_generation = generation
            return {'applied': True}
        records = self._records(revisions)
        for record, revision in zip(records, revisions):
            record['adopt'] = self.catalog.entry(revision.id).persisted == revision.number
        result = self.resident().apply_input_changes(f'rebuild:{generation}', records)
        if result.get('applied'):
            self.catalog.reset_applied()
            self.catalog.mark_applied(revisions)
            self.resident_generation = generation
        else:
            self.catalog.record_error(revisions, 'apply', json.dumps(result.get('errors', {})))
        return result

    def close(self):
        self.coordinator.shutdown(self.lease.close)
