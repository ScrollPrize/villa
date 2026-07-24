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

    def test_untracked_same_size_different_content_is_conflict(self, manager):
        write_local(manager, 'f.json', '{"a": 5}')
        info = local_info(manager, 'f.json')
        remote = s3_info(info['local_size'], 'f' * 32)  # same size, other bytes
        actions = manager.analyze_changes({'f.json': info}, {'f.json': remote})
        assert actions['f.json'][0] == SyncAction.CONFLICT

    def test_untracked_same_size_same_content_skips(self, manager):
        write_local(manager, 'f.json', '{"a": 6}')
        info = local_info(manager, 'f.json')
        remote = s3_info(info['local_size'], info['local_md5'])
        actions = manager.analyze_changes({'f.json': info}, {'f.json': remote})
        assert actions['f.json'][0] == SyncAction.SKIP

    def test_untracked_multipart_etag_keeps_size_heuristic(self, manager):
        """Non-MD5 etags cannot prove divergence -> keep today's SKIP."""
        write_local(manager, 'f.json', '{"a": 7}')
        info = local_info(manager, 'f.json')
        remote = s3_info(info['local_size'], 'abc123-4')
        actions = manager.analyze_changes({'f.json': info}, {'f.json': remote})
        assert actions['f.json'][0] == SyncAction.SKIP

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
        manager._stash_divergent_download_targets(
            ['fibers/f.json'], {'fibers/f.json': info})
        stash_root = os.path.join(manager.local_dir, vc_sync.CONFLICT_DIR_NAME)
        stashed = [f for _, _, fs in os.walk(stash_root) for f in fs]
        assert len(stashed) == 1

    def test_clean_download_target_not_stashed(self, manager):
        write_local(manager, 'fibers/f.json', '{"clean": true}')
        info = local_info(manager, 'fibers/f.json')
        track_row(manager, 'fibers/f.json', info['local_size'], 0.0,
                  10, 'e' * 32, info['local_md5'])
        manager._stash_divergent_download_targets(
            ['fibers/f.json'], {'fibers/f.json': info})
        assert not os.path.exists(
            os.path.join(manager.local_dir, vc_sync.CONFLICT_DIR_NAME))


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

    def test_clean_merge_written_and_stashed(self, manager, monkeypatch):
        base = self.fiber([0, 0, 0, 0])
        local = self.fiber([0, 0, 0, 0], generation=2)
        local['branches'] = [self.link('kb_b.json', local['control_points'][1])]
        remote = self.fiber([0, 0, 0, 0], generation=2)
        remote['branches'] = [self.link('kb_c.json', remote['control_points'][2])]
        path, local_path, info = self.setup_scenario(
            manager, monkeypatch, base, local, remote)

        outcome = manager._attempt_auto_merge(path, info, s3_info(10, 'b' * 32))

        assert outcome == 'merged'
        merged = json.load(open(local_path))
        targets = sorted(b['branch_file'] for b in merged['branches'])
        assert targets == ['kb_b.json', 'kb_c.json']
        assert merged['generation'] == 3
        stash_root = os.path.join(manager.local_dir, vc_sync.CONFLICT_DIR_NAME)
        stashed = [f for _, _, fs_ in os.walk(stash_root) for f in fs_]
        assert len(stashed) == 2  # local + remote pre-merge copies

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
    def test_same_size_divergent_content_not_healed(self, manager):
        write_local(manager, 'f.json', '{"a": 1}')
        info = local_info(manager, 'f.json')
        remote = s3_info(info['local_size'], 'd' * 32)  # same size, other bytes
        manager._record_untracked_synced({'f.json': info}, {'f.json': remote})
        with sqlite3.connect(manager.db_file) as conn:
            rows = conn.execute('SELECT COUNT(*) FROM files').fetchone()[0]
        assert rows == 0

    def test_verified_pair_healed_with_shadow(self, manager):
        write_local(manager, 'f.json', '{"a": 1}')
        info = local_info(manager, 'f.json')
        remote = s3_info(info['local_size'], info['local_md5'])
        manager._record_untracked_synced({'f.json': info}, {'f.json': remote})
        with sqlite3.connect(manager.db_file) as conn:
            row = conn.execute('SELECT local_md5 FROM files').fetchone()
        assert row[0] == info['local_md5']
        assert os.path.exists(manager._shadow_path('f.json'))
