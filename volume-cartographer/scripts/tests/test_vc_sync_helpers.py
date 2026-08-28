"""Tests for vc_sync content hashing, shadow bases, and conflict analysis.

Pure-local: a temp directory stands in for the sync dir and tracked rows are
written straight into the SQLite DB. No S3 or network access.
"""
import json
import os
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fiber_merge
import vc_sync
from vc_sync import S3SyncManager, SyncAction


@pytest.fixture
def manager(tmp_path):
    """A manager over a temp dir with config pre-seeded (no S3 contact)."""
    config = {
        's3_bucket': 'test-bucket',
        's3_prefix': 'test/prefix',
        'aws_profile': None,
        'last_updated': '2026-01-01T00:00:00',
    }
    (tmp_path / '.s3sync.json').write_text(json.dumps(config))
    return S3SyncManager(str(tmp_path))


def write_local(manager, relpath, content):
    path = os.path.join(manager.local_dir, relpath)
    os.makedirs(os.path.dirname(path) or manager.local_dir, exist_ok=True)
    with open(path, 'w') as f:
        f.write(content)
    return path


def track_row(manager, path, local_size, local_mtime, s3_size, s3_etag,
              local_md5=None):
    with sqlite3.connect(manager.db_file) as conn:
        conn.execute(
            'INSERT OR REPLACE INTO files '
            '(path, local_size, local_mtime, s3_size, s3_mtime, s3_etag, local_md5) '
            'VALUES (?, ?, ?, ?, ?, ?, ?)',
            (path, local_size, local_mtime, s3_size, 0.0, s3_etag, local_md5))


def local_info(manager, relpath):
    path = os.path.join(manager.local_dir, relpath)
    stat = os.stat(path)
    info = {'path': relpath, 'local_size': stat.st_size,
            'local_mtime': stat.st_mtime, 'is_backup': False}
    if manager._should_hash(relpath, stat.st_size):
        info['local_md5'] = manager._file_md5(path)
    return info


def s3_info(size, etag):
    return {'path': 'x', 's3_size': size, 's3_mtime': 0.0,
            's3_etag': etag, 'is_backup': False}


class TestHashPolicy:
    def test_json_small_is_hashed(self):
        assert S3SyncManager._should_hash('fibers/proj/f1.json', 1024)

    def test_json_case_insensitive(self):
        assert S3SyncManager._should_hash('a/B.JSON', 10)

    def test_large_json_not_hashed(self):
        assert not S3SyncManager._should_hash('big.json', vc_sync.HASH_MAX_BYTES + 1)

    def test_non_json_not_hashed(self):
        assert not S3SyncManager._should_hash('volumes/chunk.zarr', 100)
        assert not S3SyncManager._should_hash('img.tif', 100)

    def test_none_size_not_hashed(self):
        assert not S3SyncManager._should_hash('a.json', None)


class TestEtagIsMd5:
    def test_plain_md5(self):
        assert S3SyncManager._etag_is_md5('d41d8cd98f00b204e9800998ecf8427e')

    def test_multipart_rejected(self):
        assert not S3SyncManager._etag_is_md5('abc123-4')

    def test_empty_rejected(self):
        assert not S3SyncManager._etag_is_md5('')
        assert not S3SyncManager._etag_is_md5(None)


class TestIgnoreRules:
    def test_shadow_and_conflict_dirs_are_ignored(self):
        # Load-bearing: the tool's own dirs must never sync
        assert S3SyncManager._is_ignored(
            f'{vc_sync.BASE_DIR_NAME}/fibers/f1.json')
        assert S3SyncManager._is_ignored(
            f'{vc_sync.CONFLICT_DIR_NAME}/fibers/f1.conflict-x-local.json')

    def test_normal_fiber_file_not_ignored(self):
        assert not S3SyncManager._is_ignored('fibers/proj/kb_001.json')


class TestDbMigration:
    def test_local_md5_column_added(self, manager):
        with sqlite3.connect(manager.db_file) as conn:
            cols = {row[1] for row in conn.execute('PRAGMA table_info(files)')}
        assert 'local_md5' in cols

    def test_migration_idempotent(self, manager):
        manager._init_db()
        manager._init_db()


class TestAnalyzeChanges:
    MD5_A = None  # filled per-test from real files

    def test_touch_only_is_not_a_change(self, manager):
        """mtime moved, content identical -> in sync (no false conflict)."""
        write_local(manager, 'f.json', '{"a": 1}')
        info = local_info(manager, 'f.json')
        remote = s3_info(info['local_size'], info['local_md5'])
        # Tracked with same md5 but an mtime far in the past
        track_row(manager, 'f.json', info['local_size'], info['local_mtime'] - 9999,
                  info['local_size'], info['local_md5'], info['local_md5'])
        actions = manager.analyze_changes({'f.json': info}, {'f.json': remote})
        assert actions['f.json'][0] == SyncAction.SKIP

    def test_mtime_preserving_edit_detected(self, manager):
        """Content changed but size+mtime identical -> still an upload."""
        write_local(manager, 'f.json', '{"a": 2}')
        info = local_info(manager, 'f.json')
        old_md5 = '0' * 32
        remote = s3_info(info['local_size'], old_md5)
        track_row(manager, 'f.json', info['local_size'], info['local_mtime'],
                  info['local_size'], old_md5, old_md5)
        actions = manager.analyze_changes({'f.json': info}, {'f.json': remote})
        assert actions['f.json'][0] == SyncAction.UPLOAD

    def test_both_changed_is_conflict(self, manager):
        write_local(manager, 'f.json', '{"a": 3}')
        info = local_info(manager, 'f.json')
        remote = s3_info(info['local_size'] + 5, '1' * 32)
        track_row(manager, 'f.json', 999, 0.0, 888, '2' * 32, '3' * 32)
        actions = manager.analyze_changes({'f.json': info}, {'f.json': remote})
        assert actions['f.json'][0] == SyncAction.CONFLICT

    def test_both_changed_but_converged_skips(self, manager):
        write_local(manager, 'f.json', '{"a": 4}')
        info = local_info(manager, 'f.json')
        remote = s3_info(info['local_size'], info['local_md5'])
        track_row(manager, 'f.json', 999, 0.0, 888, '2' * 32, '3' * 32)
        actions = manager.analyze_changes({'f.json': info}, {'f.json': remote})
        assert actions['f.json'][0] == SyncAction.SKIP
        assert 'converged' in actions['f.json'][1].lower()

    def test_converged_with_record_updates_tracking_and_shadow(self, manager):
        write_local(manager, 'f.json', '{"a": 4}')
        info = local_info(manager, 'f.json')
        remote = s3_info(info['local_size'], info['local_md5'])
        track_row(manager, 'f.json', 999, 0.0, 888, '2' * 32, '3' * 32)
        manager.analyze_changes({'f.json': info}, {'f.json': remote}, record=True)
        with sqlite3.connect(manager.db_file) as conn:
            row = conn.execute(
                'SELECT local_md5, s3_etag FROM files WHERE path = ?',
                ('f.json',)).fetchone()
        assert row[0] == info['local_md5']
        assert row[1] == info['local_md5']
        assert os.path.exists(manager._shadow_path('f.json'))

    def test_untracked_same_size_different_content_is_conflict(self, manager,
                                                               monkeypatch):
        write_local(manager, 'f.json', '{"a": 5}')
        info = local_info(manager, 'f.json')
        remote = s3_info(info['local_size'], 'f' * 32)  # same size, other bytes
        # The mismatching ETag alone is not proof (could be SSE-KMS); the
        # remote content settles it.
        monkeypatch.setattr(manager, '_remote_md5', lambda p: 'f' * 32)
        actions = manager.analyze_changes({'f.json': info}, {'f.json': remote})
        assert actions['f.json'][0] == SyncAction.CONFLICT

    def test_untracked_same_size_same_content_skips(self, manager):
        write_local(manager, 'f.json', '{"a": 6}')
        info = local_info(manager, 'f.json')
        remote = s3_info(info['local_size'], info['local_md5'])
        actions = manager.analyze_changes({'f.json': info}, {'f.json': remote})
        assert actions['f.json'][0] == SyncAction.SKIP

    def test_untracked_opaque_etag_resolved_by_remote_hash(self, manager,
                                                           monkeypatch):
        """A multipart/KMS ETag proves nothing: the remote content is hashed
        to decide, instead of silently assuming in-sync."""
        write_local(manager, 'f.json', '{"a": 7}')
        info = local_info(manager, 'f.json')
        remote = s3_info(info['local_size'], 'abc123-4')

        monkeypatch.setattr(manager, '_remote_md5', lambda p: info['local_md5'])
        actions = manager.analyze_changes({'f.json': info}, {'f.json': remote})
        assert actions['f.json'][0] == SyncAction.SKIP

        monkeypatch.setattr(manager, '_remote_md5', lambda p: '9' * 32)
        actions = manager.analyze_changes({'f.json': info}, {'f.json': remote})
        assert actions['f.json'][0] == SyncAction.CONFLICT

    def test_untracked_unverifiable_content_is_conflict(self, manager,
                                                        monkeypatch):
        """If the remote cannot be fetched for hashing, 'unknown' surfaces
        as a conflict rather than becoming a false baseline."""
        write_local(manager, 'f.json', '{"a": 7}')
        info = local_info(manager, 'f.json')
        remote = s3_info(info['local_size'], 'abc123-4')
        monkeypatch.setattr(manager, '_remote_md5', lambda p: None)
        actions = manager.analyze_changes({'f.json': info}, {'f.json': remote})
        assert actions['f.json'][0] == SyncAction.CONFLICT

    def test_untracked_non_hashed_file_keeps_size_heuristic(self, manager):
        """Files outside the hash scope keep the historical size heuristic."""
        write_local(manager, 'vol2.tif', 'eight ch')
        path = os.path.join(manager.local_dir, 'vol2.tif')
        stat = os.stat(path)
        info = {'path': 'vol2.tif', 'local_size': stat.st_size,
                'local_mtime': stat.st_mtime, 'is_backup': False}
        remote = s3_info(stat.st_size, 'abc123-4')
        actions = manager.analyze_changes({'vol2.tif': info}, {'vol2.tif': remote})
        assert actions['vol2.tif'][0] == SyncAction.SKIP

    def test_md5_backfill_with_record(self, manager):
        """Row predating hashing gets its md5 adopted when stats match."""
        write_local(manager, 'f.json', '{"a": 8}')
        info = local_info(manager, 'f.json')
        remote = s3_info(info['local_size'], info['local_md5'])
        track_row(manager, 'f.json', info['local_size'], info['local_mtime'],
                  info['local_size'], info['local_md5'], None)
        manager.analyze_changes({'f.json': info}, {'f.json': remote}, record=True)
        with sqlite3.connect(manager.db_file) as conn:
            row = conn.execute('SELECT local_md5 FROM files WHERE path = ?',
                               ('f.json',)).fetchone()
        assert row[0] == info['local_md5']

    def test_no_record_leaves_db_untouched(self, manager):
        write_local(manager, 'f.json', '{"a": 9}')
        info = local_info(manager, 'f.json')
        remote = s3_info(info['local_size'], info['local_md5'])
        track_row(manager, 'f.json', info['local_size'], info['local_mtime'],
                  info['local_size'], info['local_md5'], None)
        manager.analyze_changes({'f.json': info}, {'f.json': remote})
        with sqlite3.connect(manager.db_file) as conn:
            row = conn.execute('SELECT local_md5 FROM files WHERE path = ?',
                               ('f.json',)).fetchone()
        assert row[0] is None

    def test_non_json_behavior_unchanged(self, manager):
        """Large/non-json files keep pure size/mtime semantics."""
        write_local(manager, 'vol.tif', 'not-really-a-tif')
        path = os.path.join(manager.local_dir, 'vol.tif')
        stat = os.stat(path)
        info = {'path': 'vol.tif', 'local_size': stat.st_size,
                'local_mtime': stat.st_mtime, 'is_backup': False}
        remote = s3_info(stat.st_size, 'whatever-etag')
        track_row(manager, 'vol.tif', stat.st_size, stat.st_mtime,
                  stat.st_size, 'whatever-etag')
        actions = manager.analyze_changes({'vol.tif': info}, {'vol.tif': remote})
        assert actions['vol.tif'][0] == SyncAction.SKIP


class TestShadow:
    def test_update_and_remove(self, manager):
        write_local(manager, 'fibers/f.json', '{"x": 1}')
        manager._update_shadow('fibers/f.json')
        shadow = manager._shadow_path('fibers/f.json')
        assert os.path.exists(shadow)
        assert open(shadow).read() == '{"x": 1}'
        manager._remove_shadow('fibers/f.json')
        assert not os.path.exists(shadow)
        # empty shadow subdir cleaned up too
        assert not os.path.exists(os.path.dirname(shadow))

    def test_non_hashable_files_get_no_shadow(self, manager):
        write_local(manager, 'vol.tif', 'data')
        manager._update_shadow('vol.tif')
        assert not os.path.exists(manager._shadow_path('vol.tif'))

    def test_shadow_not_scanned(self, manager):
        write_local(manager, 'fibers/f.json', '{"x": 1}')
        manager._update_shadow('fibers/f.json')
        files = manager.scan_local_files()
        assert all(not p.startswith(vc_sync.BASE_DIR_NAME) for p in files)


class TestConflictStash:
    def test_stash_local_copy(self, manager):
        write_local(manager, 'fibers/f.json', '{"mine": true}')
        dst = manager._stash_conflict_copy('fibers/f.json', 'local')
        assert dst and os.path.exists(dst)
        assert open(dst).read() == '{"mine": true}'
        assert vc_sync.CONFLICT_DIR_NAME in dst
        assert '-local' in os.path.basename(dst)
        assert dst.endswith('.json')

    def test_stash_dir_not_scanned(self, manager):
        write_local(manager, 'fibers/f.json', '{"mine": true}')
        manager._stash_conflict_copy('fibers/f.json', 'local')
        files = manager.scan_local_files()
        assert all(vc_sync.CONFLICT_DIR_NAME not in p for p in files)

    def test_divergent_download_target_stashed(self, manager):
        write_local(manager, 'fibers/f.json', '{"edited": "locally"}')
        info = local_info(manager, 'fibers/f.json')
        # Tracked md5 differs from current content
        track_row(manager, 'fibers/f.json', info['local_size'], 0.0,
                  10, 'e' * 32, 'a' * 32)
        manager._stash_divergent_local_copies(['fibers/f.json'])
        stash_root = os.path.join(manager.local_dir, vc_sync.CONFLICT_DIR_NAME)
        stashed = [f for _, _, fs in os.walk(stash_root) for f in fs]
        assert len(stashed) == 1

    def test_clean_download_target_not_stashed(self, manager):
        write_local(manager, 'fibers/f.json', '{"clean": true}')
        info = local_info(manager, 'fibers/f.json')
        track_row(manager, 'fibers/f.json', info['local_size'], 0.0,
                  10, 'e' * 32, info['local_md5'])
        manager._stash_divergent_local_copies(['fibers/f.json'])
        assert not os.path.exists(
            os.path.join(manager.local_dir, vc_sync.CONFLICT_DIR_NAME))

    def test_edit_after_scan_is_still_stashed(self, manager):
        """The divergence check hashes at stash time, not scan time: an
        edit made during the (long) interactive prompt phase must still be
        preserved before the download overwrites it."""
        write_local(manager, 'fibers/f.json', '{"v": 1}')
        info = local_info(manager, 'fibers/f.json')
        track_row(manager, 'fibers/f.json', info['local_size'], 0.0,
                  10, 'e' * 32, info['local_md5'])  # clean at scan time
        write_local(manager, 'fibers/f.json', '{"v": 2}')  # edited afterwards
        manager._stash_divergent_local_copies(['fibers/f.json'])
        stash_root = os.path.join(manager.local_dir, vc_sync.CONFLICT_DIR_NAME)
        stashed = [f for _, _, fs in os.walk(stash_root) for f in fs]
        assert len(stashed) == 1


class TestAutoMerge:
    """_attempt_auto_merge with a stubbed remote fetch — no S3."""

    @staticmethod
    def fiber(cps_z, branches=None, generation=1):
        cps = [[100.0 + 10.0 * i, 200.0, 300.0 + z] for i, z in enumerate(cps_z)]
        return {
            'type': 'vc3d_fiber', 'version': 1, 'filename': 'f.json',
            'generation': generation,
            'control_points': cps,
            'line_points': cps,  # coarse but valid for the merge
            'branches': branches or [], 'tags': [],
        }

    @staticmethod
    def link(target, anchor_cp):
        return {
            'control_point_index': 0, 'branch_fiber_id': 1,
            'branch_control_point_index': 0, 'branch_file': target,
            'control_point_direction': [1.0, 0.0, 0.0],
            'branch_control_point_direction': [0.0, 1.0, 0.0],
            'control_point_position': list(anchor_cp),
            'branch_control_point_position': [9.0, 9.0, 9.0],
            'pending': True,
        }

    def setup_scenario(self, manager, monkeypatch, base, local, remote):
        path = 'fibers/f.json'
        local_path = write_local(manager, path, json.dumps(local))
        # Shadow holds the base; tracked local_md5 records the base content
        # (the local file has diverged from it since the last sync).
        shadow = manager._shadow_path(path)
        os.makedirs(os.path.dirname(shadow), exist_ok=True)
        with open(shadow, 'w') as f:
            json.dump(base, f)
        base_md5 = manager._file_md5(shadow)
        info = local_info(manager, path)
        track_row(manager, path, 10, 0.0, 10, 'a' * 32, base_md5)

        def fake_fetch(p):
            tmp = manager._merge_tmp_path(p, '.remote')
            with open(tmp, 'w') as f:
                json.dump(remote, f)
            return remote, tmp

        monkeypatch.setattr(manager, '_fetch_remote_json', fake_fetch)
        return path, local_path, info

    def test_clean_merge_deferred_until_applied(self, manager, monkeypatch):
        """Planning never touches the local file; _apply_pending_merges (run
        after user confirmation) performs the swap and the stashing."""
        base = self.fiber([0, 0, 0, 0])
        local = self.fiber([0, 0, 0, 0], generation=2)
        local['branches'] = [self.link('kb_b.json', local['control_points'][1])]
        remote = self.fiber([0, 0, 0, 0], generation=2)
        remote['branches'] = [self.link('kb_c.json', remote['control_points'][2])]
        path, local_path, info = self.setup_scenario(
            manager, monkeypatch, base, local, remote)
        before = open(local_path).read()

        plan = manager._attempt_auto_merge(path, info, s3_info(10, 'b' * 32))

        assert isinstance(plan, dict) and os.path.exists(plan['pending'])
        assert open(local_path).read() == before  # untouched until confirmed
        stash_root = os.path.join(manager.local_dir, vc_sync.CONFLICT_DIR_NAME)

        def stashes():
            # Actual conflict copies only — not the .tmp planning scratch
            return [f for _, _, fs_ in os.walk(stash_root)
                    for f in fs_ if '.conflict-' in f]

        assert stashes() == []

        manager._apply_pending_merges([(path, plan)])

        merged = json.load(open(local_path))
        targets = sorted(b['branch_file'] for b in merged['branches'])
        assert targets == ['kb_b.json', 'kb_c.json']
        assert merged['generation'] == 3
        assert len(stashes()) == 2  # local + remote pre-merge copies
        assert not os.path.exists(plan['pending'])
        assert not os.path.exists(plan['remote_tmp'])

    def test_cancelled_merge_discarded_without_touching_local(self, manager,
                                                              monkeypatch):
        base = self.fiber([0, 0, 0, 0])
        local = self.fiber([0, 0, 0, 0], generation=2)
        local['tags'] = ['from-local']
        remote = self.fiber([0, 0, 0, 0], generation=2)
        remote['tags'] = ['from-remote']
        path, local_path, info = self.setup_scenario(
            manager, monkeypatch, base, local, remote)
        before = open(local_path).read()

        plan = manager._attempt_auto_merge(path, info, s3_info(10, 'b' * 32))
        assert isinstance(plan, dict)
        manager._discard_pending_merges([(path, plan)])

        assert open(local_path).read() == before
        assert not os.path.exists(plan['pending'])
        assert not os.path.exists(plan['remote_tmp'])
        stash_root = os.path.join(manager.local_dir, vc_sync.CONFLICT_DIR_NAME)
        assert [f for _, _, fs_ in os.walk(stash_root)
                for f in fs_ if '.conflict-' in f] == []

    def test_dry_run_probes_without_writing(self, manager, monkeypatch):
        base = self.fiber([0, 0, 0, 0])
        local = self.fiber([0, 0, 0, 0], generation=2)
        local['tags'] = ['from-local']
        remote = self.fiber([0, 0, 0, 0], generation=2)
        remote['tags'] = ['from-remote']
        path, local_path, info = self.setup_scenario(
            manager, monkeypatch, base, local, remote)
        before = open(local_path).read()

        outcome = manager._attempt_auto_merge(path, info, s3_info(10, 'b' * 32),
                                              dry_run=True)

        assert outcome and outcome.startswith('would auto-merge')
        assert open(local_path).read() == before
        stash_root = os.path.join(manager.local_dir, vc_sync.CONFLICT_DIR_NAME)
        stashed = [f for _, _, fs_ in os.walk(stash_root) for f in fs_]
        assert stashed == []  # probing writes nothing

    def test_conflicting_merge_stashes_all_three(self, manager, monkeypatch):
        base = self.fiber([0, 0, 0, 0])
        local = self.fiber([0, 5, 0, 0], generation=2)   # both move CP1
        remote = self.fiber([0, -5, 0, 0], generation=2)
        path, local_path, info = self.setup_scenario(
            manager, monkeypatch, base, local, remote)
        before = open(local_path).read()

        outcome = manager._attempt_auto_merge(path, info, s3_info(10, 'b' * 32))

        assert outcome is None
        assert open(local_path).read() == before
        stash_root = os.path.join(manager.local_dir, vc_sync.CONFLICT_DIR_NAME)
        stashed = sorted(f for _, _, fs_ in os.walk(stash_root) for f in fs_)
        assert len(stashed) == 3  # local + remote + base
        assert any('-base' in name for name in stashed)

    def test_same_basename_conflicts_do_not_collide(self, manager, monkeypatch):
        """Two conflicting files sharing a basename in different directories
        must get independent pending/remote temp files; flattening to the
        basename cross-contaminated their merged content."""
        plans = {}
        for pkg, tag in (('pkgA', 'from-a'), ('pkgB', 'from-b')):
            base = self.fiber([0, 0, 0, 0])
            local = self.fiber([0, 0, 0, 0], generation=2)
            local['tags'] = [tag + '-local']
            remote = self.fiber([0, 0, 0, 0], generation=2)
            remote['tags'] = [tag + '-remote']
            path = f'{pkg}/fibers/f.json'
            local_path = write_local(manager, path, json.dumps(local))
            shadow = manager._shadow_path(path)
            os.makedirs(os.path.dirname(shadow), exist_ok=True)
            with open(shadow, 'w') as f:
                json.dump(base, f)
            info = local_info(manager, path)
            track_row(manager, path, 10, 0.0, 10, 'a' * 32,
                      manager._file_md5(shadow))

            def fake_fetch(p, remote_doc=remote):
                tmp = manager._merge_tmp_path(p, '.remote')
                with open(tmp, 'w') as f:
                    json.dump(remote_doc, f)
                return remote_doc, tmp

            monkeypatch.setattr(manager, '_fetch_remote_json', fake_fetch)
            plan = manager._attempt_auto_merge(path, info, s3_info(10, 'b' * 32))
            assert isinstance(plan, dict)
            plans[path] = (local_path, plan)

        pending_paths = [plan['pending'] for _, plan in plans.values()]
        assert len(set(pending_paths)) == 2  # no shared temp file

        manager._apply_pending_merges(
            [(path, plan) for path, (_, plan) in plans.items()])
        for path, (local_path, _) in plans.items():
            merged = json.load(open(local_path))
            marker = 'from-a' if 'pkgA' in path else 'from-b'
            assert any(marker in t for t in merged['tags']), \
                f"{path} received another file's merged content"

    def test_no_base_means_no_merge(self, manager, monkeypatch):
        local = self.fiber([0, 0, 0, 0], generation=2)
        path = 'fibers/f.json'
        write_local(manager, path, json.dumps(local))
        info = local_info(manager, path)
        track_row(manager, path, 10, 0.0, 10, 'not-an-md5-etag-4', 'c' * 32)
        monkeypatch.setattr(manager, '_fetch_remote_json',
                            lambda p: (self.fiber([0, 0, 0, 1]), None))

        assert manager._attempt_auto_merge(path, info, s3_info(10, 'b' * 32)) is None


class TestRecordUntrackedSynced:
    def test_same_size_divergent_content_not_healed(self, manager, monkeypatch):
        write_local(manager, 'f.json', '{"a": 1}')
        info = local_info(manager, 'f.json')
        remote = s3_info(info['local_size'], 'd' * 32)  # same size, other bytes
        monkeypatch.setattr(manager, '_remote_md5', lambda p: 'd' * 32)
        manager._record_untracked_synced({'f.json': info}, {'f.json': remote})
        with sqlite3.connect(manager.db_file) as conn:
            rows = conn.execute('SELECT COUNT(*) FROM files').fetchone()[0]
        assert rows == 0

    def test_unverifiable_pair_not_healed(self, manager, monkeypatch):
        """No content proof (opaque etag, remote unreachable) -> stays
        untracked instead of becoming a false baseline."""
        write_local(manager, 'f.json', '{"a": 1}')
        info = local_info(manager, 'f.json')
        remote = s3_info(info['local_size'], 'abc123-4')
        monkeypatch.setattr(manager, '_remote_md5', lambda p: None)
        manager._record_untracked_synced({'f.json': info}, {'f.json': remote})
        with sqlite3.connect(manager.db_file) as conn:
            rows = conn.execute('SELECT COUNT(*) FROM files').fetchone()[0]
        assert rows == 0

    def test_opaque_etag_healed_via_remote_hash(self, manager, monkeypatch):
        write_local(manager, 'f.json', '{"a": 1}')
        info = local_info(manager, 'f.json')
        remote = s3_info(info['local_size'], 'abc123-4')
        monkeypatch.setattr(manager, '_remote_md5', lambda p: info['local_md5'])
        manager._record_untracked_synced({'f.json': info}, {'f.json': remote})
        with sqlite3.connect(manager.db_file) as conn:
            rows = conn.execute('SELECT COUNT(*) FROM files').fetchone()[0]
        assert rows == 1
        assert os.path.exists(manager._shadow_path('f.json'))

    def test_verified_pair_healed_with_shadow(self, manager):
        write_local(manager, 'f.json', '{"a": 1}')
        info = local_info(manager, 'f.json')
        remote = s3_info(info['local_size'], info['local_md5'])
        manager._record_untracked_synced({'f.json': info}, {'f.json': remote})
        with sqlite3.connect(manager.db_file) as conn:
            row = conn.execute('SELECT local_md5 FROM files').fetchone()
        assert row[0] == info['local_md5']
        assert os.path.exists(manager._shadow_path('f.json'))


class TestBackfillTrackedBases:
    """The 'update' command must seed hashes + merge bases for tracked
    files that predate content hashing (the shadow-base backfill)."""

    def test_tracked_row_without_md5_gets_seeded(self, manager):
        write_local(manager, 'f.json', '{"a": 1}')
        info = local_info(manager, 'f.json')
        remote = s3_info(info['local_size'], info['local_md5'])
        track_row(manager, 'f.json', info['local_size'], info['local_mtime'],
                  info['local_size'], info['local_md5'], None)  # pre-hash row
        manager._backfill_tracked_bases({'f.json': info}, {'f.json': remote})
        with sqlite3.connect(manager.db_file) as conn:
            row = conn.execute('SELECT local_md5 FROM files').fetchone()
        assert row[0] == info['local_md5']
        assert os.path.exists(manager._shadow_path('f.json'))

    def test_divergent_tracked_pair_left_alone(self, manager, monkeypatch):
        write_local(manager, 'f.json', '{"a": 1}')
        info = local_info(manager, 'f.json')
        remote = s3_info(info['local_size'], 'e' * 32)
        track_row(manager, 'f.json', info['local_size'], info['local_mtime'],
                  info['local_size'], 'e' * 32, None)
        monkeypatch.setattr(manager, '_remote_md5', lambda p: 'e' * 32)
        manager._backfill_tracked_bases({'f.json': info}, {'f.json': remote})
        with sqlite3.connect(manager.db_file) as conn:
            row = conn.execute('SELECT local_md5 FROM files').fetchone()
        assert row[0] is None
        assert not os.path.exists(manager._shadow_path('f.json'))

    def test_unverifiable_pair_left_alone(self, manager, monkeypatch):
        write_local(manager, 'f.json', '{"a": 1}')
        info = local_info(manager, 'f.json')
        remote = s3_info(info['local_size'], 'abc123-4')  # opaque ETag
        track_row(manager, 'f.json', info['local_size'], info['local_mtime'],
                  info['local_size'], 'abc123-4', None)
        monkeypatch.setattr(manager, '_remote_md5', lambda p: None)
        manager._backfill_tracked_bases({'f.json': info}, {'f.json': remote})
        assert not os.path.exists(manager._shadow_path('f.json'))

    def test_refresh_tracking_wires_in_the_backfill(self, manager, monkeypatch):
        write_local(manager, 'f.json', '{"a": 1}')
        info = local_info(manager, 'f.json')
        remote = s3_info(info['local_size'], info['local_md5'])
        track_row(manager, 'f.json', info['local_size'], info['local_mtime'],
                  info['local_size'], info['local_md5'], None)
        monkeypatch.setattr(manager, 'scan_s3_files',
                            lambda include_backups=False: {'f.json': remote})
        manager.refresh_tracking()
        assert os.path.exists(manager._shadow_path('f.json'))


class TestProbeSideEffects:
    def test_remote_md5_leaves_sync_dir_untouched(self, manager, monkeypatch):
        """status/--dry-run verification fetches must not create anything
        inside the sync directory."""
        def fake_aws(cmd):
            with open(cmd[-1], 'w') as f:
                f.write('{"x": 1}')
            class Result:
                stdout = ''
            return Result()

        monkeypatch.setattr(manager, '_run_aws_command', fake_aws)
        assert manager._remote_md5('fibers/f.json')
        assert not os.path.exists(
            os.path.join(manager.local_dir, vc_sync.CONFLICT_DIR_NAME))
        assert not os.path.exists(
            os.path.join(manager.local_dir, vc_sync.BASE_DIR_NAME))


class TestLinkConsistency:
    """_plan_link_consistency + _apply_peer_fixes: reciprocal mirroring of
    a merged fiber's link decisions into its peer files."""

    CPS_A = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]]
    CPS_B = [[0.0, 50.0, 0.0], [10.0, 50.0, 0.0]]

    @staticmethod
    def fiber(name, cps, branches=None, generation=1):
        return {'type': 'vc3d_fiber', 'version': 1, 'filename': name,
                'generation': generation,
                'control_points': [list(p) for p in cps],
                'line_points': [list(p) for p in cps],
                'branches': branches or [], 'tags': []}

    @classmethod
    def entry(cls, target, local_cps, local_index, far_cps, far_index):
        return {
            'control_point_index': local_index, 'branch_fiber_id': 3,
            'branch_control_point_index': far_index, 'branch_file': target,
            'control_point_direction':
                fiber_merge.endpoint_tangent(local_cps, local_cps[local_index]),
            'branch_control_point_direction':
                fiber_merge.endpoint_tangent(far_cps, far_cps[far_index]),
            'control_point_position': list(local_cps[local_index]),
            'branch_control_point_position': list(far_cps[far_index]),
            'pending': True,
        }

    def make_plan(self, manager, path, merged_doc, base_doc, peers):
        pending = manager._merge_tmp_path(path, '.merged')
        with open(pending, 'w') as f:
            json.dump(merged_doc, f)
        return {'pending': pending, 'remote_tmp': None, 'summary': 's',
                'merged_doc': merged_doc, 'base_doc': base_doc,
                'peer_files': peers}

    def test_missing_reciprocal_planned_and_applied(self, manager):
        a = self.fiber('a.json', self.CPS_A,
                       branches=[self.entry('b.json', self.CPS_A, 1,
                                            self.CPS_B, 0)])
        b = self.fiber('b.json', self.CPS_B)   # reciprocal missing
        write_local(manager, 'fibers/b.json', json.dumps(b))
        plan = self.make_plan(manager, 'fibers/a.json', a,
                              self.fiber('a.json', self.CPS_A), ['b.json'])

        peer_fixes, demoted = manager._plan_link_consistency(
            [('fibers/a.json', plan)], set())
        assert demoted == []
        assert list(peer_fixes) == ['fibers/b.json']
        reciprocal = peer_fixes['fibers/b.json']['branches'][0]
        assert reciprocal['branch_file'] == 'a.json'
        assert reciprocal['control_point_index'] == 0
        assert reciprocal['branch_control_point_index'] == 1
        assert reciprocal['pending'] is True

        manager._apply_peer_fixes(peer_fixes)
        with open(os.path.join(manager.local_dir, 'fibers/b.json')) as f:
            assert json.load(f) == peer_fixes['fibers/b.json']
        stash_root = os.path.join(manager.local_dir, vc_sync.CONFLICT_DIR_NAME)
        stashed = [name for _, _, files in os.walk(stash_root) for name in files]
        assert any('-peer' in name for name in stashed)

    def test_consistent_peer_needs_no_fix(self, manager):
        a_entry = self.entry('b.json', self.CPS_A, 1, self.CPS_B, 0)
        b_entry = self.entry('a.json', self.CPS_B, 0, self.CPS_A, 1)
        a = self.fiber('a.json', self.CPS_A, branches=[a_entry])
        b = self.fiber('b.json', self.CPS_B, branches=[b_entry])
        write_local(manager, 'fibers/b.json', json.dumps(b))
        plan = self.make_plan(manager, 'fibers/a.json', a,
                              self.fiber('a.json', self.CPS_A), ['b.json'])
        peer_fixes, demoted = manager._plan_link_consistency(
            [('fibers/a.json', plan)], set())
        assert demoted == []
        assert peer_fixes == {}

    def test_missing_peer_demotes_merge(self, manager):
        a = self.fiber('a.json', self.CPS_A,
                       branches=[self.entry('b.json', self.CPS_A, 1,
                                            self.CPS_B, 0)])
        plan = self.make_plan(manager, 'fibers/a.json', a,
                              self.fiber('a.json', self.CPS_A), ['b.json'])
        peer_fixes, demoted = manager._plan_link_consistency(
            [('fibers/a.json', plan)], set())
        assert peer_fixes == {}
        assert demoted and demoted[0][0] == 'fibers/a.json'
        assert 'missing' in demoted[0][1]

    def test_peer_scheduled_for_download_uses_remote_content(self, manager,
                                                             monkeypatch):
        """A peer being downloaded is fixed against its REMOTE content —
        the fix lands after the download and re-uploads."""
        a = self.fiber('a.json', self.CPS_A,
                       branches=[self.entry('b.json', self.CPS_A, 1,
                                            self.CPS_B, 0)])
        remote_b = self.fiber('b.json', self.CPS_B, generation=5)
        write_local(manager, 'fibers/b.json',
                    json.dumps(self.fiber('b.json', [[9.0, 9.0, 9.0]])))

        def fake_fetch(path):
            tmp = manager._merge_tmp_path(path, '.remote')
            with open(tmp, 'w') as f:
                json.dump(remote_b, f)
            return remote_b, tmp

        monkeypatch.setattr(manager, '_fetch_remote_json', fake_fetch)
        plan = self.make_plan(manager, 'fibers/a.json', a,
                              self.fiber('a.json', self.CPS_A), ['b.json'])
        peer_fixes, demoted = manager._plan_link_consistency(
            [('fibers/a.json', plan)], {'fibers/b.json'})
        assert demoted == []
        fixed = peer_fixes['fibers/b.json']
        # Based on the remote doc (gen 5), bumped by the rewrite
        assert fixed['generation'] == 6
        assert fixed['branches'][0]['branch_file'] == 'a.json'


class TestAnalyzeDeleteActions:
    """The tracked delete-vs-download decision — historically the dangerous
    class in this tool — pinned explicitly."""

    def test_tracked_local_deletion_proposes_remote_delete(self, manager):
        track_row(manager, 'f.json', 10, 0.0, 10, 'e' * 32)
        actions = manager.analyze_changes({}, {'f.json': s3_info(10, 'e' * 32)})
        assert actions['f.json'][0] == SyncAction.DELETE_REMOTE

    def test_untracked_s3_only_file_downloads(self, manager):
        actions = manager.analyze_changes({}, {'f.json': s3_info(10, 'e' * 32)})
        assert actions['f.json'][0] == SyncAction.DOWNLOAD

    def test_tracked_s3_deletion_proposes_local_delete(self, manager):
        write_local(manager, 'f.json', '{"a": 1}')
        info = local_info(manager, 'f.json')
        track_row(manager, 'f.json', info['local_size'], info['local_mtime'],
                  info['local_size'], 'e' * 32, info['local_md5'])
        actions = manager.analyze_changes({'f.json': info}, {})
        assert actions['f.json'][0] == SyncAction.DELETE_LOCAL

    def test_untracked_local_only_file_uploads(self, manager):
        write_local(manager, 'f.json', '{"a": 1}')
        info = local_info(manager, 'f.json')
        actions = manager.analyze_changes({'f.json': info}, {})
        assert actions['f.json'][0] == SyncAction.UPLOAD

    def test_deleted_from_both_skips(self, manager):
        track_row(manager, 'f.json', 10, 0.0, 10, 'e' * 32)
        actions = manager.analyze_changes({}, {})
        assert actions['f.json'][0] == SyncAction.SKIP


class TestOldSchemaMigration:
    def test_rows_survive_with_null_md5(self, tmp_path):
        """A genuinely old DB (no local_md5 column, legacy idx_path index)
        must migrate in place: column added, rows intact with NULL md5."""
        config = {'s3_bucket': 'b', 's3_prefix': 'p', 'aws_profile': None,
                  'last_updated': 'x'}
        (tmp_path / '.s3sync.json').write_text(json.dumps(config))
        db_file = tmp_path / '.s3sync.db'
        with sqlite3.connect(db_file) as conn:
            conn.execute('''CREATE TABLE files
                            (path TEXT PRIMARY KEY, local_size INTEGER,
                             local_mtime REAL, s3_size INTEGER,
                             s3_mtime REAL, s3_etag TEXT)''')
            conn.execute('CREATE INDEX idx_path ON files(path)')
            conn.execute('INSERT INTO files VALUES (?, ?, ?, ?, ?, ?)',
                         ('old.json', 5, 1.0, 5, 2.0, 'e' * 32))
        manager = S3SyncManager(str(tmp_path))
        with sqlite3.connect(manager.db_file) as conn:
            cols = {row[1] for row in conn.execute('PRAGMA table_info(files)')}
            row = conn.execute(
                'SELECT path, s3_etag, local_md5 FROM files').fetchone()
        assert 'local_md5' in cols
        assert row == ('old.json', 'e' * 32, None)


class TestStagingPath:
    def test_staged_files_are_ignored_by_scans(self):
        """A crash between staging and os.replace must not leave a file the
        next sync would upload as a new object."""
        staged = S3SyncManager._staging_path('fibers/proj/f.json')
        assert S3SyncManager._is_ignored(staged)


class TestDemotionFixpoint:
    """_plan_conflict_resolutions: demotions from link-consistency planning
    must feed back into the blocked-peer cascade (the #1246 review's M3 —
    a peer fix must never override a user's manual conflict choice)."""

    CPS = TestLinkConsistency.CPS_A

    def make_plan_for(self, manager, path, peers):
        doc = TestLinkConsistency.fiber(os.path.basename(path), self.CPS)
        pending = manager._merge_tmp_path(path, '.merged')
        with open(pending, 'w') as f:
            json.dump(doc, f)
        return {'pending': pending, 'remote_tmp': None, 'summary': 's',
                'merged_doc': doc,
                'base_doc': TestLinkConsistency.fiber(os.path.basename(path),
                                                      self.CPS),
                'peer_files': peers}

    def test_plan_demotion_cascades_to_dependents(self, manager, monkeypatch):
        """A merges with peer B; B's own merge demotes (its peer C is
        missing). B becoming manual must demote A too — no peer fix may be
        computed for B from stale content."""
        plans = {
            'fibers/a.json': self.make_plan_for(manager, 'fibers/a.json',
                                                ['b.json']),
            'fibers/b.json': self.make_plan_for(manager, 'fibers/b.json',
                                                ['c.json']),  # c missing
        }
        write_local(manager, 'fibers/a.json', json.dumps(plans['fibers/a.json']['merged_doc']))
        write_local(manager, 'fibers/b.json', json.dumps(plans['fibers/b.json']['merged_doc']))
        monkeypatch.setattr(manager, '_attempt_auto_merge',
                            lambda path, li, si: plans[path])
        conflicts = [('fibers/a.json', 'both changed'),
                     ('fibers/b.json', 'both changed')]
        pending, peer_fixes, manual = manager._plan_conflict_resolutions(
            conflicts, {}, {}, set(), set(), auto_merge=True)
        assert pending == []
        assert peer_fixes == {}
        assert sorted(p for p, _ in manual) == ['fibers/a.json',
                                                'fibers/b.json']

    def test_manual_peer_blocks_merge_upfront(self, manager, monkeypatch):
        plans = {'fibers/a.json': self.make_plan_for(manager, 'fibers/a.json',
                                                     ['b.json'])}
        write_local(manager, 'fibers/a.json',
                    json.dumps(plans['fibers/a.json']['merged_doc']))
        write_local(manager, 'fibers/b.json', 'not-json')

        def attempt(path, li, si):
            return plans.get(path)

        monkeypatch.setattr(manager, '_attempt_auto_merge', attempt)
        conflicts = [('fibers/a.json', 'both changed'),
                     ('fibers/b.json', 'both changed')]
        pending, peer_fixes, manual = manager._plan_conflict_resolutions(
            conflicts, {}, {}, set(), set(), auto_merge=True)
        assert pending == [] and peer_fixes == {}
        assert {p for p, _ in manual} == {'fibers/a.json', 'fibers/b.json'}

    def test_partial_peer_failure_rolls_back_fixes(self, manager):
        """Two peers, second missing: the first peer's tentative fix must
        not leak into peer_fixes when the merge demotes."""
        a = TestLinkConsistency.fiber(
            'a.json', TestLinkConsistency.CPS_A,
            branches=[TestLinkConsistency.entry(
                          'b.json', TestLinkConsistency.CPS_A, 1,
                          TestLinkConsistency.CPS_B, 0),
                      TestLinkConsistency.entry(
                          'c.json', TestLinkConsistency.CPS_A, 0,
                          TestLinkConsistency.CPS_B, 1)])
        b = TestLinkConsistency.fiber('b.json', TestLinkConsistency.CPS_B)
        write_local(manager, 'fibers/b.json', json.dumps(b))  # c.json missing
        pending = manager._merge_tmp_path('fibers/a.json', '.merged')
        with open(pending, 'w') as f:
            json.dump(a, f)
        plan = {'pending': pending, 'remote_tmp': None, 'summary': 's',
                'merged_doc': a,
                'base_doc': TestLinkConsistency.fiber(
                    'a.json', TestLinkConsistency.CPS_A),
                'peer_files': ['b.json', 'c.json']}
        peer_fixes, demoted = manager._plan_link_consistency(
            [('fibers/a.json', plan)], set())
        assert demoted and demoted[0][0] == 'fibers/a.json'
        assert peer_fixes == {}


class TestDeleteLocalSafety:
    def test_divergent_local_copy_stashed_before_deletion(self, manager):
        """Remote deleted the file, local edited it since the last sync —
        previously the only loss path with no conflict copy."""
        write_local(manager, 'fibers/f.json', '{"edited": "locally"}')
        info = local_info(manager, 'fibers/f.json')
        track_row(manager, 'fibers/f.json', info['local_size'], 0.0,
                  10, 'e' * 32, 'a' * 32)  # tracked hash differs
        manager._stash_divergent_local_copies(['fibers/f.json'])
        stash_root = os.path.join(manager.local_dir, vc_sync.CONFLICT_DIR_NAME)
        stashed = [f for _, _, fs in os.walk(stash_root) for f in fs]
        assert len(stashed) == 1

    def test_delete_vs_edit_surfaced_in_reason(self, manager):
        write_local(manager, 'f.json', '{"edited": true}')
        info = local_info(manager, 'f.json')
        track_row(manager, 'f.json', info['local_size'], info['local_mtime'],
                  10, 'e' * 32, 'a' * 32)  # s3 gone, local hash changed
        actions = manager.analyze_changes({'f.json': info}, {})
        action, reason = actions['f.json']
        assert action == SyncAction.DELETE_LOCAL
        assert 'unsynced edits' in reason


class TestPlannerInputHardening:
    def test_nul_byte_peer_name_demotes_instead_of_crashing(self, manager):
        """A remote-crafted branch_file with an embedded NUL must demote
        the merge, not abort the sync with a ValueError traceback."""
        doc = TestLinkConsistency.fiber('a.json', TestLinkConsistency.CPS_A)
        pending = manager._merge_tmp_path('fibers/a.json', '.merged')
        with open(pending, 'w') as f:
            json.dump(doc, f)
        plan = {'pending': pending, 'remote_tmp': None, 'summary': 's',
                'merged_doc': doc, 'base_doc': doc,
                'peer_files': ['b\x00.json']}
        peer_fixes, demoted = manager._plan_link_consistency(
            [('fibers/a.json', plan)], set())
        assert peer_fixes == {}
        assert demoted and 'invalid linked fiber name' in demoted[0][1]

    def test_nan_in_merged_doc_demotes(self, manager):
        """A NaN smuggled through an opaque subtree would serialize as
        invalid JSON (unloadable by nlohmann); planning demotes it."""
        a = TestLinkConsistency.fiber(
            'a.json', TestLinkConsistency.CPS_A,
            branches=[TestLinkConsistency.entry(
                'b.json', TestLinkConsistency.CPS_A, 1,
                TestLinkConsistency.CPS_B, 0)])
        a['opaque_extra'] = float('nan')
        b = TestLinkConsistency.fiber('b.json', TestLinkConsistency.CPS_B)
        write_local(manager, 'fibers/b.json', json.dumps(b))
        pending = manager._merge_tmp_path('fibers/a.json', '.merged')
        with open(pending, 'w') as f:
            f.write('{}')
        plan = {'pending': pending, 'remote_tmp': None, 'summary': 's',
                'merged_doc': a,
                'base_doc': TestLinkConsistency.fiber(
                    'a.json', TestLinkConsistency.CPS_A),
                'peer_files': ['b.json']}
        peer_fixes, demoted = manager._plan_link_consistency(
            [('fibers/a.json', plan)], set())
        assert peer_fixes == {}
        assert demoted and 'refresh against' in demoted[0][1]


class TestClassifyFibers:
    """hfsync publishes outward, so classify_fibers is a gate: what it puts
    in `tagged` is uploaded to a Hugging Face bucket, and what it puts in
    `untagged` is deleted from one."""

    @staticmethod
    def fiber(tags, **extra):
        doc = {'type': 'vc3d_fiber', 'version': 1, 'filename': 'f.json',
               'control_points': [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
               'line_points': [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
               'generation': 1, 'branches': [], 'tags': tags}
        doc.update(extra)
        return doc

    @staticmethod
    def write(tmp_path, name, doc):
        path = tmp_path / name
        path.write_text(doc if isinstance(doc, str) else json.dumps(doc))
        return path

    def test_tagged_fiber_publishes(self, tmp_path):
        self.write(tmp_path, 'a.json', self.fiber(['reviewed']))
        tagged, untagged, invalid, deferred = vc_sync.classify_fibers(
            str(tmp_path), 'reviewed')
        assert (tagged, untagged, invalid, deferred) == (['a.json'], [], [], [])

    def test_untagged_fiber_is_removable(self, tmp_path):
        self.write(tmp_path, 'a.json', self.fiber(['draft']))
        tagged, untagged, invalid, deferred = vc_sync.classify_fibers(
            str(tmp_path), 'reviewed')
        assert (tagged, untagged, deferred) == ([], ['a.json'], [])

    def test_needs_reoptimization_is_held_back(self, tmp_path):
        """A sync-merged fiber's line is a straight-segment placeholder until
        VC3D re-fits it; publishing it presents unfitted geometry as
        reviewed."""
        self.write(tmp_path, 'a.json',
                   self.fiber(['reviewed', vc_sync.REOPTIMIZE_TAG]))
        tagged, untagged, invalid, deferred = vc_sync.classify_fibers(
            str(tmp_path), 'reviewed')
        assert tagged == []
        # NOT untagged: a held-back local file must not delete the good copy
        # already published.
        assert untagged == []
        assert invalid == []
        assert [name for name, _ in deferred] == ['a.json']
        assert vc_sync.REOPTIMIZE_TAG in deferred[0][1]

    def test_untagged_and_unfitted_is_still_removable(self, tmp_path):
        """Holding back applies only to fibers claiming the publish tag."""
        self.write(tmp_path, 'a.json', self.fiber([vc_sync.REOPTIMIZE_TAG]))
        tagged, untagged, invalid, deferred = vc_sync.classify_fibers(
            str(tmp_path), 'reviewed')
        assert (tagged, untagged, deferred) == ([], ['a.json'], [])

    def test_string_tags_do_not_substring_match(self, tmp_path):
        """`tag in tags` over a string publishes 'unreviewed' as 'reviewed'."""
        self.write(tmp_path, 'a.json', self.fiber('unreviewed'))
        tagged, untagged, invalid, deferred = vc_sync.classify_fibers(
            str(tmp_path), 'reviewed')
        assert tagged == []
        assert [name for name, _ in invalid] == ['a.json']

    def test_null_tags_does_not_crash(self, tmp_path):
        """`tags: null` used to raise TypeError and abort the whole run."""
        self.write(tmp_path, 'a.json', self.fiber(None))
        tagged, untagged, invalid, deferred = vc_sync.classify_fibers(
            str(tmp_path), 'reviewed')
        assert [name for name, _ in invalid] == ['a.json']

    def test_non_object_root_does_not_crash(self, tmp_path):
        """A JSON root that isn't an object used to raise AttributeError."""
        self.write(tmp_path, 'a.json', '[1, 2, 3]')
        tagged, untagged, invalid, deferred = vc_sync.classify_fibers(
            str(tmp_path), 'reviewed')
        assert [name for name, _ in invalid] == ['a.json']

    def test_unloadable_fiber_is_invalid_not_untagged(self, tmp_path):
        """An unloadable doc is quarantined, not treated as a deletion
        request for its published namesake."""
        self.write(tmp_path, 'a.json', self.fiber(['reviewed'], version=2))
        tagged, untagged, invalid, deferred = vc_sync.classify_fibers(
            str(tmp_path), 'reviewed')
        assert (tagged, untagged) == ([], [])
        assert [name for name, _ in invalid] == ['a.json']

    def test_existing_skips_still_apply(self, tmp_path):
        (tmp_path / 'empty.json').write_text('')
        (tmp_path / 'bad.json').write_text('{not json')
        (tmp_path / '.hidden.json').write_text(json.dumps(self.fiber(['reviewed'])))
        (tmp_path / 'notes.txt').write_text('x')
        (tmp_path / 'sub.json').mkdir()
        tagged, untagged, invalid, deferred = vc_sync.classify_fibers(
            str(tmp_path), 'reviewed')
        assert tagged == []
        assert sorted(name for name, _ in invalid) == ['bad.json', 'empty.json']

    def test_unpublishable_reason_without_fiber_merge(self, monkeypatch):
        """The fallback path (fiber_merge import failed) must still refuse the
        shapes that crash the run or mis-tag a fiber."""
        monkeypatch.setattr(vc_sync, 'fiber_merge', None)
        assert vc_sync.unpublishable_reason({'tags': ['reviewed']}) is None
        assert vc_sync.unpublishable_reason({'tags': None})
        assert vc_sync.unpublishable_reason({'tags': 'unreviewed'})
        assert vc_sync.unpublishable_reason([1, 2, 3])


# --- SFTP (ash) sync -------------------------------------------------------

import subprocess


def completed(returncode=0, stdout='', stderr=''):
    return subprocess.CompletedProcess(args=[], returncode=returncode,
                                       stdout=stdout, stderr=stderr)


@pytest.fixture
def sftp_manager(tmp_path, monkeypatch):
    """An SftpSyncManager over a temp dir; rclone probing/obscuring stubbed
    out so construction touches no network and needs no rclone binary."""
    creds_file = tmp_path / 'creds.json'
    creds_file.write_text(json.dumps(
        {'host': 'ash.example.org', 'user': 'user1',
         'pass': 'secret', 'port': 2022}))
    os.chmod(creds_file, 0o600)
    local = tmp_path / 'data'
    local.mkdir()
    (local / vc_sync.ASH_CONFIG_NAME).write_text(json.dumps({
        'remote_path': '/srv/backup',
        'creds_file': str(creds_file),
        'last_updated': '2026-01-01T00:00:00',
    }))
    monkeypatch.setattr(vc_sync.S3SyncManager, '_detect_rclone',
                        lambda self: True)
    monkeypatch.setattr(vc_sync.SftpSyncManager, '_rclone_obscure',
                        staticmethod(lambda password: 'obscured:' + password))
    return vc_sync.SftpSyncManager(str(local))


class TestSftpCredentials:
    def write_creds(self, tmp_path, data):
        path = tmp_path / 'creds.json'
        path.write_text(json.dumps(data))
        os.chmod(path, 0o600)
        return str(path)

    def test_valid_with_default_port(self, tmp_path):
        path = self.write_creds(tmp_path, {'host': 'h', 'user': 'u', 'pass': 'p'})
        creds = vc_sync.load_sftp_credentials(path)
        assert creds == {'host': 'h', 'user': 'u', 'pass': 'p', 'port': 22,
                         'known_hosts_file': 'none'}

    def test_known_hosts_file_passthrough(self, tmp_path):
        path = self.write_creds(tmp_path,
                                {'host': 'h', 'user': 'u', 'pass': 'p',
                                 'known_hosts_file': '~/.ssh/known_hosts'})
        creds = vc_sync.load_sftp_credentials(path)
        assert creds['known_hosts_file'] == '~/.ssh/known_hosts'

    def test_explicit_port(self, tmp_path):
        path = self.write_creds(tmp_path,
                                {'host': 'h', 'user': 'u', 'pass': 'p', 'port': 2022})
        assert vc_sync.load_sftp_credentials(path)['port'] == 2022

    def test_missing_file_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            vc_sync.load_sftp_credentials(str(tmp_path / 'nope.json'))

    def test_missing_key_exits(self, tmp_path):
        path = self.write_creds(tmp_path, {'host': 'h', 'user': 'u'})
        with pytest.raises(SystemExit):
            vc_sync.load_sftp_credentials(path)

    def test_non_string_pass_exits(self, tmp_path):
        path = self.write_creds(tmp_path,
                                {'host': 'h', 'user': 'u', 'pass': 12345})
        with pytest.raises(SystemExit):
            vc_sync.load_sftp_credentials(path)

    def test_non_int_port_exits(self, tmp_path):
        path = self.write_creds(tmp_path,
                                {'host': 'h', 'user': 'u', 'pass': 'p', 'port': '22'})
        with pytest.raises(SystemExit):
            vc_sync.load_sftp_credentials(path)


class TestSftpManagerState:
    def test_state_files_are_independent_of_s3_sync(self, sftp_manager):
        assert sftp_manager.config_file.endswith(vc_sync.ASH_CONFIG_NAME)
        assert sftp_manager.db_file.endswith(vc_sync.ASH_DB_NAME)
        assert sftp_manager.BASE_DIR == vc_sync.ASH_BASE_DIR_NAME
        assert sftp_manager.CONFLICT_DIR == vc_sync.ASH_CONFLICT_DIR_NAME
        assert vc_sync.ASH_BASE_DIR_NAME != vc_sync.BASE_DIR_NAME

    def test_ash_state_files_are_ignored_by_scans(self):
        # Load-bearing: the ash sync's own state must never be synced,
        # by either the S3 sync or the ash sync itself
        for name in (vc_sync.ASH_CONFIG_NAME, vc_sync.ASH_DB_NAME):
            assert S3SyncManager._is_ignored(name)
        for dirname in (vc_sync.ASH_BASE_DIR_NAME, vc_sync.ASH_CONFLICT_DIR_NAME):
            assert S3SyncManager._is_ignored(f'{dirname}/fibers/f1.json')

    def test_rclone_remote_and_env(self, sftp_manager):
        assert sftp_manager._rclone_remote() == ':sftp:/srv/backup'
        env = sftp_manager._rclone_env()
        assert env['RCLONE_SFTP_HOST'] == 'ash.example.org'
        assert env['RCLONE_SFTP_USER'] == 'user1'
        assert env['RCLONE_SFTP_PORT'] == '2022'
        assert env['RCLONE_SFTP_PASS'] == 'obscured:secret'
        assert env['RCLONE_SFTP_KNOWN_HOSTS_FILE'] == 'none'

    def test_display_url(self, sftp_manager):
        assert sftp_manager._get_s3_url('a/b.json') == (
            'sftp://user1@ash.example.org:2022/srv/backup/a/b.json')

    def test_no_versioning_and_no_history_base(self, sftp_manager):
        assert sftp_manager._bucket_versioning_enabled() is False
        assert sftp_manager._fetch_base_from_history(
            'a.json', 'd41d8cd98f00b204e9800998ecf8427e') is None

    def test_aws_cli_calls_are_a_bug(self, sftp_manager):
        with pytest.raises(RuntimeError):
            sftp_manager._run_aws_command(['aws', 's3', 'ls'])

    def test_init_writes_config_roundtrip(self, tmp_path, monkeypatch):
        creds_file = tmp_path / 'creds.json'
        creds_file.write_text(json.dumps(
            {'host': 'h', 'user': 'u', 'pass': 'p'}))
        os.chmod(creds_file, 0o600)
        monkeypatch.setattr(vc_sync.S3SyncManager, '_detect_rclone',
                            lambda self: True)
        monkeypatch.setattr(vc_sync.SftpSyncManager, '_rclone_obscure',
                            staticmethod(lambda password: 'x'))
        local = tmp_path / 'newdir'
        first = vc_sync.SftpSyncManager(str(local), 'backup/scrolls/',
                                        str(creds_file))
        assert first.remote_path == 'backup/scrolls'  # trailing slash stripped
        reloaded = vc_sync.SftpSyncManager(str(local))
        assert reloaded.remote_path == 'backup/scrolls'
        assert reloaded.creds_file == str(creds_file)


class TestSftpRemoteChanged:
    MD5_A = 'a' * 32
    MD5_B = 'b' * 32

    def test_size_change_wins(self, sftp_manager):
        assert sftp_manager._remote_changed(
            {'s3_size': 10, 's3_etag': self.MD5_A, 's3_mtime': 0.0},
            {'s3_size': 11, 's3_etag': self.MD5_A, 's3_mtime': 0.0})

    def test_matching_md5s_beat_mtime_drift(self, sftp_manager):
        assert not sftp_manager._remote_changed(
            {'s3_size': 10, 's3_etag': self.MD5_A, 's3_mtime': 0.0},
            {'s3_size': 10, 's3_etag': self.MD5_A, 's3_mtime': 9999.0})

    def test_differing_md5s_detected_despite_same_mtime(self, sftp_manager):
        assert sftp_manager._remote_changed(
            {'s3_size': 10, 's3_etag': self.MD5_A, 's3_mtime': 0.0},
            {'s3_size': 10, 's3_etag': self.MD5_B, 's3_mtime': 0.0})

    def test_mtime_fallback_without_hashes(self, sftp_manager):
        tracked = {'s3_size': 10, 's3_etag': None, 's3_mtime': 100.0}
        assert not sftp_manager._remote_changed(
            tracked, {'s3_size': 10, 's3_etag': None, 's3_mtime': 101.0})
        assert sftp_manager._remote_changed(
            tracked, {'s3_size': 10, 's3_etag': None, 's3_mtime': 105.0})

    def test_one_sided_hash_falls_back_to_mtime(self, sftp_manager):
        # tracked before server-side hashing was available
        tracked = {'s3_size': 10, 's3_etag': None, 's3_mtime': 100.0}
        assert sftp_manager._remote_changed(
            tracked, {'s3_size': 10, 's3_etag': self.MD5_A, 's3_mtime': 105.0})

    def test_untracked_mtime_is_not_a_change(self, sftp_manager):
        assert not sftp_manager._remote_changed(
            {'s3_size': 10}, {'s3_size': 10, 's3_etag': None, 's3_mtime': 5.0})


class TestSftpTimeParsing:
    def test_nanosecond_rfc3339(self, sftp_manager):
        ts = sftp_manager._parse_rclone_time('2026-08-27T10:00:00.123456789Z')
        assert abs(ts - 1787824800.123456) < 0.001

    def test_offset_and_no_fraction(self, sftp_manager):
        assert (sftp_manager._parse_rclone_time('2026-08-27T12:00:00+02:00') ==
                sftp_manager._parse_rclone_time('2026-08-27T10:00:00Z'))


class TestSftpContentMatch:
    MD5_A = 'a' * 32
    MD5_B = 'b' * 32

    def test_hash_match_is_proof(self, sftp_manager):
        assert sftp_manager._content_matches_remote('f.json', self.MD5_A,
                                                    self.MD5_A)

    def test_hash_mismatch_is_proof_and_seeds_cache(self, sftp_manager):
        assert not sftp_manager._content_matches_remote('f.json', self.MD5_A,
                                                        self.MD5_B)
        # The remote hash is now known without any download
        assert sftp_manager._remote_md5('f.json') == self.MD5_B

    def test_no_remote_hash_downloads(self, sftp_manager, monkeypatch):
        monkeypatch.setattr(sftp_manager, '_remote_md5',
                            lambda path: self.MD5_A)
        assert sftp_manager._content_matches_remote('f.json', self.MD5_A, None)


class TestSftpScanAndHashes:
    def test_scan_parses_listing_applies_ignores_and_attaches_md5s(
            self, sftp_manager, monkeypatch):
        listing = [
            {'Path': 'fibers/f1.json', 'Size': 100,
             'ModTime': '2026-08-27T10:00:00.5Z'},
            {'Path': '.hidden/skipme.json', 'Size': 5,
             'ModTime': '2026-08-27T10:00:00Z'},
            {'Path': 'layers_full/x.tif', 'Size': 5,
             'ModTime': '2026-08-27T10:00:00Z'},
            {'Path': 'volumes/big.tif', 'Size': 12345,
             'ModTime': '2026-08-27T10:00:00Z'},
        ]
        md5 = 'c' * 32
        calls = {}

        def fake_capture(args, timeout=None):
            assert args[0] == 'lsjson'
            return completed(stdout=json.dumps(listing))

        def fake_md5sums(paths):
            calls['paths'] = paths
            return {'fibers/f1.json': md5}

        monkeypatch.setattr(sftp_manager, '_run_rclone_capture', fake_capture)
        monkeypatch.setattr(sftp_manager, '_remote_md5sums', fake_md5sums)

        files = sftp_manager.scan_s3_files()
        assert sorted(files) == ['fibers/f1.json', 'volumes/big.tif']
        assert files['fibers/f1.json']['s3_etag'] == md5
        assert files['fibers/f1.json']['s3_size'] == 100
        assert files['volumes/big.tif']['s3_etag'] is None
        # Only hash-scoped files were sent for hashing
        assert calls['paths'] == ['fibers/f1.json']

    def test_scan_missing_remote_dir_is_empty(self, sftp_manager, monkeypatch):
        monkeypatch.setattr(
            sftp_manager, '_run_rclone_capture',
            lambda args, timeout=None: completed(
                returncode=3, stderr='2026/08/27 error: directory not found'))
        assert sftp_manager.scan_s3_files() == {}

    def test_md5sums_parses_and_skips_unsupported(self, sftp_manager,
                                                  monkeypatch):
        stdout = ('d41d8cd98f00b204e9800998ecf8427e  a.json\n'
                  'UNSUPPORTED  b.json\n'
                  '                                  c.json\n')
        monkeypatch.setattr(sftp_manager, '_run_rclone_capture',
                            lambda args, timeout=None: completed(stdout=stdout))
        md5s = sftp_manager._remote_md5sums(['a.json', 'b.json', 'c.json'])
        assert md5s == {'a.json': 'd41d8cd98f00b204e9800998ecf8427e'}
        assert sftp_manager._remote_hashing_works is True

    def test_md5sums_unsupported_server_disables_hashing(self, sftp_manager,
                                                         monkeypatch):
        ran = []
        monkeypatch.setattr(
            sftp_manager, '_run_rclone_capture',
            lambda args, timeout=None: ran.append(args) or completed(stdout=''))
        assert sftp_manager._remote_md5sums(['a.json']) == {}
        assert sftp_manager._remote_hashing_works is False
        # Later calls short-circuit instead of re-probing the server
        assert sftp_manager._remote_md5sums(['b.json']) == {}
        assert len(ran) == 1


class TestSftpAnalyzeIntegration:
    """analyze_changes over the SFTP manager: the md5-in-the-etag-slot and
    mtime-fallback semantics drive the same engine decisions ETags do on S3."""

    def test_remote_content_change_same_size_is_download(self, sftp_manager):
        path = 'fibers/f1.json'
        write_local(sftp_manager, path, '{"a": 1}')
        info = local_info(sftp_manager, path)
        track_row(sftp_manager, path, info['local_size'], info['local_mtime'],
                  info['local_size'], 'a' * 32, local_md5=info['local_md5'])
        remote = {path: {'path': path, 's3_size': info['local_size'],
                         's3_mtime': 0.0, 's3_etag': 'b' * 32,
                         'is_backup': False}}
        actions = sftp_manager.analyze_changes({path: info}, remote)
        assert actions[path][0] == SyncAction.DOWNLOAD

    def test_unhashed_remote_mtime_change_is_download(self, sftp_manager):
        path = 'volumes/meta.bin'
        write_local(sftp_manager, path, 'binary')
        info = local_info(sftp_manager, path)
        with sqlite3.connect(sftp_manager.db_file) as conn:
            conn.execute(
                'INSERT OR REPLACE INTO files '
                '(path, local_size, local_mtime, s3_size, s3_mtime, s3_etag) '
                'VALUES (?, ?, ?, ?, ?, ?)',
                (path, info['local_size'], info['local_mtime'],
                 info['local_size'], 100.0, None))
        remote = {path: {'path': path, 's3_size': info['local_size'],
                         's3_mtime': 200.0, 's3_etag': None,
                         'is_backup': False}}
        actions = sftp_manager.analyze_changes({path: info}, remote)
        assert actions[path][0] == SyncAction.DOWNLOAD

    def test_both_changed_identical_content_converges(self, sftp_manager):
        path = 'fibers/f1.json'
        write_local(sftp_manager, path, '{"a": 2}')
        info = local_info(sftp_manager, path)
        track_row(sftp_manager, path, 1, 0.0, 1, 'a' * 32, local_md5='b' * 32)
        remote = {path: {'path': path, 's3_size': info['local_size'],
                         's3_mtime': 0.0, 's3_etag': info['local_md5'],
                         'is_backup': False}}
        actions = sftp_manager.analyze_changes({path: info}, remote)
        assert actions[path][0] == SyncAction.SKIP

    def test_both_changed_different_content_is_conflict(self, sftp_manager):
        path = 'fibers/f1.json'
        write_local(sftp_manager, path, '{"a": 2}')
        info = local_info(sftp_manager, path)
        track_row(sftp_manager, path, 1, 0.0, 1, 'a' * 32, local_md5='b' * 32)
        remote = {path: {'path': path, 's3_size': info['local_size'],
                         's3_mtime': 0.0, 's3_etag': 'c' * 32,
                         'is_backup': False}}
        actions = sftp_manager.analyze_changes({path: info}, remote)
        assert actions[path][0] == SyncAction.CONFLICT
