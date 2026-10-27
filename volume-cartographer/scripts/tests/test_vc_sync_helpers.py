"""Tests for vc_sync content hashing and conflict analysis.

Pure-local: a temp directory stands in for the sync dir and tracked rows are
written straight into the SQLite DB. No S3 or network access.
"""
import hashlib
import json
import os
import sqlite3
import sys
import types

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
    def test_conflict_dirs_are_ignored(self):
        # Load-bearing: the tool's own dirs must never sync
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

    def test_converged_with_record_updates_tracking(self, manager):
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

    def test_verified_pair_healed(self, manager):
        write_local(manager, 'f.json', '{"a": 1}')
        info = local_info(manager, 'f.json')
        remote = s3_info(info['local_size'], info['local_md5'])
        manager._record_untracked_synced({'f.json': info}, {'f.json': remote})
        with sqlite3.connect(manager.db_file) as conn:
            row = conn.execute('SELECT local_md5 FROM files').fetchone()
        assert row[0] == info['local_md5']


class TestBackfillTrackedHashes:
    """The update command verifies content before recording missing hashes."""

    def test_tracked_row_without_md5_gets_seeded(self, manager):
        write_local(manager, 'f.json', '{"a": 1}')
        info = local_info(manager, 'f.json')
        remote = s3_info(info['local_size'], info['local_md5'])
        track_row(manager, 'f.json', info['local_size'], info['local_mtime'],
                  info['local_size'], info['local_md5'], None)  # pre-hash row
        manager._backfill_tracked_hashes({'f.json': info}, {'f.json': remote})
        with sqlite3.connect(manager.db_file) as conn:
            row = conn.execute('SELECT local_md5 FROM files').fetchone()
        assert row[0] == info['local_md5']

    def test_divergent_tracked_pair_left_alone(self, manager, monkeypatch):
        write_local(manager, 'f.json', '{"a": 1}')
        info = local_info(manager, 'f.json')
        remote = s3_info(info['local_size'], 'e' * 32)
        track_row(manager, 'f.json', info['local_size'], info['local_mtime'],
                  info['local_size'], 'e' * 32, None)
        monkeypatch.setattr(manager, '_remote_md5', lambda p: 'e' * 32)
        manager._backfill_tracked_hashes({'f.json': info}, {'f.json': remote})
        with sqlite3.connect(manager.db_file) as conn:
            row = conn.execute('SELECT local_md5 FROM files').fetchone()
        assert row[0] is None

    def test_unverifiable_pair_left_alone(self, manager, monkeypatch):
        write_local(manager, 'f.json', '{"a": 1}')
        info = local_info(manager, 'f.json')
        remote = s3_info(info['local_size'], 'abc123-4')  # opaque ETag
        track_row(manager, 'f.json', info['local_size'], info['local_mtime'],
                  info['local_size'], 'abc123-4', None)
        monkeypatch.setattr(manager, '_remote_md5', lambda p: None)
        manager._backfill_tracked_hashes({'f.json': info}, {'f.json': remote})

    def test_refresh_tracking_wires_in_the_backfill(self, manager, monkeypatch):
        write_local(manager, 'f.json', '{"a": 1}')
        info = local_info(manager, 'f.json')
        remote = s3_info(info['local_size'], info['local_md5'])
        track_row(manager, 'f.json', info['local_size'], info['local_mtime'],
                  info['local_size'], info['local_md5'], None)
        monkeypatch.setattr(manager, 'scan_s3_files',
                            lambda include_backups=False: {'f.json': remote})
        manager.refresh_tracking()
        with sqlite3.connect(manager.db_file) as conn:
            assert conn.execute('SELECT local_md5 FROM files').fetchone()[0] == info['local_md5']


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
        assert sftp_manager.CONFLICT_DIR == vc_sync.ASH_CONFLICT_DIR_NAME

    def test_ash_state_files_are_ignored_by_scans(self):
        # Load-bearing: the ash sync's own state must never be synced,
        # by either the S3 sync or the ash sync itself
        for name in (vc_sync.ASH_CONFIG_NAME, vc_sync.ASH_DB_NAME):
            assert S3SyncManager._is_ignored(name)
        for dirname in (vc_sync.ASH_CONFLICT_DIR_NAME,):
            assert S3SyncManager._is_ignored(f'{dirname}/fibers/f1.json')

    def test_rclone_remote_and_env(self, sftp_manager):
        assert sftp_manager._rclone_remote() == ':sftp:/srv/backup'
        env = sftp_manager._rclone_env()
        assert env['RCLONE_SFTP_HOST'] == 'ash.example.org'
        assert env['RCLONE_SFTP_USER'] == 'user1'
        assert env['RCLONE_SFTP_PORT'] == '2022'
        assert env['RCLONE_SFTP_PASS'] == 'obscured:secret'
        # The "none" default must leave the variable UNSET: older rclone
        # treats any value as a literal path and fails to open "none"
        assert 'RCLONE_SFTP_KNOWN_HOSTS_FILE' not in env

    def test_known_hosts_path_reaches_rclone_env(self, sftp_manager):
        sftp_manager._creds['known_hosts_file'] = '~/.ssh/known_hosts'
        env = sftp_manager._rclone_env()
        assert env['RCLONE_SFTP_KNOWN_HOSTS_FILE'] == os.path.expanduser(
            '~/.ssh/known_hosts')

    def test_shell_type_and_hash_commands_are_pinned(self, sftp_manager):
        """Detection must not run at all: rclone's lazy per-connection
        shell/hash probe cannot be persisted for an on-the-fly :sftp:
        backend, and racing it across --checkers connections makes a
        batched md5sum report "hash unsupported" for most files —
        silently degrading change detection to size+mtime."""
        env = sftp_manager._rclone_env()
        assert env['RCLONE_SFTP_SHELL_TYPE'] == 'unix'
        assert env['RCLONE_SFTP_MD5SUM_COMMAND'] == 'md5sum'
        assert env['RCLONE_SFTP_SHA1SUM_COMMAND'] == 'sha1sum'

    def test_pins_override_a_polluted_caller_environment(self, sftp_manager,
                                                         monkeypatch):
        """Like host/user/pass, the pins are enforced, not defaults: a
        stray RCLONE_SFTP_* in the caller's shell must not reintroduce
        the probe race. (Deliberate overrides go through
        VC_SYNC_RCLONE_FLAGS — rclone flags beat its env vars.)"""
        monkeypatch.setenv('RCLONE_SFTP_SHELL_TYPE', 'powershell')
        monkeypatch.setenv('RCLONE_SFTP_MD5SUM_COMMAND', 'certutil')
        env = sftp_manager._rclone_env()
        assert env['RCLONE_SFTP_SHELL_TYPE'] == 'unix'
        assert env['RCLONE_SFTP_MD5SUM_COMMAND'] == 'md5sum'

    def test_display_url(self, sftp_manager):
        assert sftp_manager._get_s3_url('a/b.json') == (
            'sftp://user1@ash.example.org:2022/srv/backup/a/b.json')

    def test_no_versioning(self, sftp_manager):
        assert sftp_manager._bucket_versioning_enabled() is False

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
                                                         monkeypatch, capsys):
        ran = []
        monkeypatch.setattr(
            sftp_manager, '_run_rclone_capture',
            lambda args, timeout=None: ran.append(args) or completed(stdout=''))
        assert sftp_manager._remote_md5sums(['a.json']) == {}
        assert sftp_manager._remote_hashing_works is False
        # The user is told why hashing degraded — once, on first detection
        assert 'hashing unavailable' in capsys.readouterr().out
        # Later calls short-circuit instead of re-probing the server
        assert sftp_manager._remote_md5sums(['b.json']) == {}
        assert len(ran) == 1
        assert 'hashing unavailable' not in capsys.readouterr().out


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


@pytest.mark.parametrize("choice", [SyncAction.UPLOAD, SyncAction.DOWNLOAD, SyncAction.SKIP])
@pytest.mark.parametrize("confirmed", [False, True])
def test_sync_uses_explicit_file_conflict_choice(manager, monkeypatch, choice, confirmed):
    path = "annotation.json"
    write_local(manager, path, '{"value": 1}')
    local = {path: local_info(manager, path)}
    remote = {path: s3_info(local[path]['local_size'], 'b' * 32)}
    manager.use_rclone = False
    monkeypatch.setattr(manager, 'scan_local_files', lambda *args: local)
    monkeypatch.setattr(manager, 'scan_s3_files', lambda *args: remote)
    monkeypatch.setattr(manager, 'analyze_changes',
                        lambda *args, **kwargs: {path: (SyncAction.CONFLICT, 'Both changed')})
    monkeypatch.setattr(manager, '_record_untracked_synced', lambda *args: None)
    monkeypatch.setattr(manager, 'resolve_conflict', lambda *args: choice)
    monkeypatch.setattr(manager, '_stash_divergent_local_copies', lambda *args: None)
    monkeypatch.setattr(vc_sync, 'confirm', lambda *args: confirmed)
    transfers = []
    monkeypatch.setattr(manager, 'perform_upload', lambda p: transfers.append((p, SyncAction.UPLOAD)))
    monkeypatch.setattr(manager, 'perform_download', lambda p, remote: transfers.append((p, SyncAction.DOWNLOAD)))
    manager.sync()
    assert transfers == ([(path, choice)] if confirmed and choice != SyncAction.SKIP else [])


def test_retired_fiber_publishing_command_is_rejected(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['vc_sync.py', 'hfsync', '/unused', '--dry-run'])
    with pytest.raises(SystemExit) as error:
        vc_sync.main()
    assert error.value.code == 2
