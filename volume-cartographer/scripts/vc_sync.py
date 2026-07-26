#!/usr/bin/env python3
"""
AWS S3 Interactive Sync Tool with Conflict Resolution

Transfers run through rclone (parallel, one process per batch) when the rclone
binary is available, and fall back to serial per-file aws CLI calls otherwise.
Neither path needs an rclone config file: credentials come from the standard
AWS credential chain (env vars pasted into the terminal, ~/.aws/credentials,
or EC2 instance roles). Where possible, the credentials the aws CLI resolved
are handed to rclone as static env vars (via `aws configure export-credentials`)
so both tools use the same identity and rclone never resolves its own.

Automatically ignores:
- Hidden files and directories (starting with .)
- Any directory containing 'layers' in its name (e.g., layers/, layers_fullres/, old_layers/)
- The .s3sync.json configuration file and .s3sync.db database
- Files matching backup patterns (see BACKUP_PATTERNS)
- Directories named 'backups' (unless --sync-backups is specified)

Change detection compares file size and mtime locally (1-second tolerance)
and size + ETag on S3, so a local edit that preserves both size and mtime is
invisible until the file changes again.

'update' refreshes tracking non-destructively (records files already in sync,
prunes entries for files gone from both sides) and never hides a pending
difference. 'reset' stamps the current state as the synced baseline, which
discards pending differences — it asks for confirmation.

Usage:
    python s3_sync.py init <directory> <s3_bucket> <s3_prefix> [--profile=<aws_profile>]
    python s3_sync.py status <directory> [--verbose] [--sync-backups]
    python s3_sync.py sync <directory> [--dry-run] [--sync-backups]
    python s3_sync.py update <directory> [--sync-backups]
    python s3_sync.py reset <directory> [--sync-backups]
    python s3_sync.py hfsync <directory> [--dry-run]

Hugging Face sync (hfsync):
    Pushes fiber JSONs carrying a given tag to a Hugging Face storage bucket.
    Opt-in per directory: it only runs where a .hfsync.json exists next to the
    data (the file is never synced to S3 and should not be committed anywhere):

        {
          "hf_bucket_path": "hf://buckets/<org>/<bucket>/<path>",
          "hf_cli": "/path/to/hf",          # optional, defaults to hf on PATH
          "tag": "reviewed"                 # optional, defaults to "reviewed"
        }

    Authentication uses the token stored by `hf auth login`; no credentials
    are read from or written to this script or its config. Upload is additive
    and skips unchanged files; a remote file is only removed when the same
    filename exists locally WITHOUT the tag. Files that exist only remotely
    are never touched.
"""

import os
import sys
import json
import shlex
import shutil
import sqlite3
import hashlib
import argparse
import tempfile
import subprocess
from datetime import datetime, timezone
from enum import Enum
from contextlib import contextmanager

# Optional fiber-aware three-way merge (fiber_merge.py next to this script).
# Everything else works without it; annotation conflicts just stay manual.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import fiber_merge
except ImportError:
    fiber_merge = None


# Backup file patterns - these files are only uploaded, never downloaded or deleted
# Note: This is separate from the backups/ directory filter which is controlled by --sync-backups
BACKUP_PATTERNS = [
    '_backup',
    '.backup',
    '_bak',
    '.bak',
]

# Content hashing is restricted to annotation-sized JSON files: volumes and
# other bulk data keep the cheap size/mtime heuristics.
HASH_SUFFIXES = ('.json',)
HASH_MAX_BYTES = 16 * 1024 * 1024

# Tool-owned directories inside the sync dir. Both are dot-prefixed, which
# _is_ignored and the scan_local_files dir pruning already exclude from
# syncing — that exclusion is a load-bearing dependency.
BASE_DIR_NAME = '.s3sync-base'          # last-synced copies (3-way merge base)
CONFLICT_DIR_NAME = '.s3sync-conflicts'  # versions preserved before overwrite

class SyncAction(Enum):
    UPLOAD = "upload"
    DOWNLOAD = "download"
    CONFLICT = "conflict"
    SKIP = "skip"
    DELETE_LOCAL = "delete_local"
    DELETE_REMOTE = "delete_remote"


def is_backup_file(filename):
    """Check if a file matches backup patterns"""
    return any(pattern in filename.lower() for pattern in BACKUP_PATTERNS)


def prompt_choice(message, choices):
    """Prompt until the user enters one of `choices`; EOF (Ctrl-D) returns None"""
    while True:
        try:
            response = input(message).strip().lower()
        except EOFError:
            print()
            return None
        if response in choices:
            return response
        print(f"Invalid choice. Please enter one of: {', '.join(choices)}.")


def confirm(message):
    """[y/N]-style confirmation; anything but 'y' (including EOF) is no"""
    try:
        return input(message).strip().lower() == 'y'
    except EOFError:
        print()
        return False


class S3SyncManager:
    # Parallel transfer settings for the rclone fast path
    RCLONE_TRANSFERS = 16
    RCLONE_CHECKERS = 32

    def __init__(self, local_dir, s3_bucket=None, s3_prefix=None,
                 aws_profile=None):
        self.local_dir = os.path.abspath(local_dir)
        self.config_file = os.path.join(self.local_dir, '.s3sync.json')
        self.db_file = os.path.join(self.local_dir, '.s3sync.db')

        # Load or create config
        if os.path.exists(self.config_file):
            self._load_config()
        else:
            if not s3_bucket or not s3_prefix:
                raise ValueError("s3_bucket and s3_prefix required for initialization")

            # Create directory if it doesn't exist during init
            os.makedirs(self.local_dir, exist_ok=True)

            self.s3_bucket = s3_bucket
            self.s3_prefix = s3_prefix.rstrip('/')
            self.aws_profile = aws_profile
            self._save_config()

        # Initialize database
        self._init_db()

        self.use_rclone = self._detect_rclone()

        # Per-run cache of remote content hashes (path -> md5 or None),
        # used when an ETag cannot prove content equality.
        self._remote_md5_cache = {}

        # Per-run scratch dir for transient fetch/merge artifacts, created
        # lazily OUTSIDE the sync tree so probing (status, --dry-run)
        # leaves the sync directory untouched. Removed by _cleanup_run_tmp.
        self._run_tmp_dir = None

        # Cached result of the bucket-versioning probe (None = not asked).
        self._bucket_versioning = None

    def _detect_rclone(self):
        """Use rclone only if the binary exists AND can read the sync directory.

        Sandboxed installs (e.g. the Ubuntu snap) may lack access to some
        paths — a snap without the removable-media interface cannot read
        /media, for instance. Probing the actual sync dir catches that.

        Sets self.rclone_unavailable_reason when returning False.
        """
        self.rclone_unavailable_reason = None

        if not shutil.which('rclone'):
            self.rclone_unavailable_reason = "rclone binary not found on PATH"
            return False

        result = subprocess.run(
            ['rclone', 'lsf', self.local_dir, '--max-depth', '1'],
            capture_output=True, text=True)
        if result.returncode != 0:
            detail_lines = (result.stderr or '').strip().splitlines()
            detail = detail_lines[-1] if detail_lines else "unknown error"
            self.rclone_unavailable_reason = (
                f"rclone cannot read {self.local_dir} "
                f"(sandboxed install? a snap needs its home/removable-media "
                f"interfaces connected): {detail}")
            return False

        return True

    def _load_config(self):
        """Load configuration from JSON file"""
        with open(self.config_file, 'r') as f:
            data = json.load(f)

        self.s3_bucket = data['s3_bucket']
        self.s3_prefix = data['s3_prefix']
        self.aws_profile = data.get('aws_profile')

    def _save_config(self):
        """Save configuration to JSON file (just config, not file tracking)"""
        data = {
            'local_dir': self.local_dir,
            's3_bucket': self.s3_bucket,
            's3_prefix': self.s3_prefix,
            'aws_profile': self.aws_profile,
            'last_updated': datetime.now().isoformat()
        }

        with open(self.config_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _init_db(self):
        """Initialize SQLite database for file tracking"""
        conn = sqlite3.connect(self.db_file)
        conn.execute('''
                     CREATE TABLE IF NOT EXISTS files (
                                                          path TEXT PRIMARY KEY,
                                                          local_size INTEGER,
                                                          local_mtime REAL,
                                                          s3_size INTEGER,
                                                          s3_mtime REAL,
                                                          s3_etag TEXT,
                                                          last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                     )
                     ''')

        # path is the PRIMARY KEY and therefore already indexed; drop the
        # redundant secondary index older versions of this script created
        conn.execute('DROP INDEX IF EXISTS idx_path')

        # Lazy migration: databases created before content hashing lack the
        # local_md5 column. Rows keep NULL until the file next syncs or a
        # scan backfills them in analyze_changes.
        columns = {row[1] for row in conn.execute('PRAGMA table_info(files)')}
        if 'local_md5' not in columns:
            try:
                conn.execute('ALTER TABLE files ADD COLUMN local_md5 TEXT')
            except sqlite3.OperationalError:
                pass  # a concurrent first run added it between the check and here

        conn.commit()
        conn.close()

    @contextmanager
    def _get_db(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_file)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except BaseException:
            # Roll back explicitly (and loudly, via re-raise) so an interrupted
            # transaction can't silently discard a whole tracking update
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _track_file(conn, path, local_info, s3_info):
        """Record a file's synced state (the single writer for tracking rows).

        local_info needs 'local_size'/'local_mtime', s3_info needs
        's3_size'/'s3_mtime' (+ optional 's3_etag'); either may be None for a
        file that only exists on one side.
        """
        conn.execute('''
            INSERT OR REPLACE INTO files
            (path, local_size, local_mtime, s3_size, s3_mtime, s3_etag, local_md5)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (path,
              local_info['local_size'] if local_info else None,
              local_info['local_mtime'] if local_info else None,
              s3_info['s3_size'] if s3_info else None,
              s3_info['s3_mtime'] if s3_info else None,
              s3_info.get('s3_etag') if s3_info else None,
              local_info.get('local_md5') if local_info else None))

    def _run_aws_command(self, cmd):
        """Run AWS CLI command with optional profile and better error handling"""
        if self.aws_profile:
            cmd = cmd + ['--profile', self.aws_profile]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return result
        except subprocess.CalledProcessError as e:
            print(f"\n❌ AWS CLI Error:")
            print(f"Command: {' '.join(cmd)}")
            print(f"Exit code: {e.returncode}")
            if e.stdout:
                print(f"Stdout: {e.stdout}")
            if e.stderr:
                print(f"Stderr: {e.stderr}")
            raise

    def _get_s3_url(self, relative_path=None):
        """Get S3 URL for a file or directory"""
        if relative_path:
            return f"s3://{self.s3_bucket}/{self.s3_prefix}/{relative_path}"
        return f"s3://{self.s3_bucket}/{self.s3_prefix}/"

    def _parse_timestamp(self, timestamp_str):
        """Parse AWS timestamp to Unix timestamp"""
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return dt.timestamp()

    def _cleanup_empty_dirs(self, filepath):
        """Remove empty parent directories after file deletion"""
        dirpath = os.path.dirname(filepath)

        while dirpath and dirpath != self.local_dir:
            try:
                if os.path.isdir(dirpath) and not os.listdir(dirpath):
                    print(f"    Removing empty directory: {os.path.relpath(dirpath, self.local_dir)}")
                    os.rmdir(dirpath)
                    dirpath = os.path.dirname(dirpath)
                else:
                    break
            except OSError:
                break

    @staticmethod
    def _is_ignored(relative_path, include_backups=False):
        """Shared ignore rules for local and S3 scans (POSIX-style path).

        Both scans MUST agree on these rules: transfer verification compares
        upload/delete candidates against a fresh S3 listing, so a file one
        scan sees and the other skips would never verify.
        """
        parts = relative_path.split('/')
        filename = parts[-1]

        if filename.startswith('.') or filename.endswith('.obj'):
            return True

        for part in parts[:-1]:
            if part.startswith('.') or 'layers' in part.lower():
                return True
            if not include_backups and part == 'backups':
                return True

        return False

    @staticmethod
    def _should_hash(relative_path, size):
        """Content-hash policy: annotation-sized JSON files only."""
        if size is None or size > HASH_MAX_BYTES:
            return False
        return relative_path.lower().endswith(HASH_SUFFIXES)

    @staticmethod
    def _file_md5(filepath):
        digest = hashlib.md5()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _etag_is_md5(etag):
        """True when an S3 ETag has plain-MD5 shape (single-part upload).

        Shape alone is not proof of content MD5 (SSE-KMS/SSE-C ETags are
        also 32-hex but opaque), so callers may use a matching ETag as
        POSITIVE evidence of equality only — never a mismatch as proof of
        difference."""
        return bool(etag) and '-' not in etag

    def _content_matches_remote(self, path, md5, etag):
        """Verified content equality between a local hash and the remote
        object: a plain-MD5 ETag match is positive proof; otherwise the
        remote content is fetched and hashed (False on fetch failure)."""
        if not md5:
            return False
        if self._etag_is_md5(etag or '') and md5 == etag:
            return True
        return self._remote_md5(path) == md5

    def _bucket_versioning_enabled(self):
        """Whether the target bucket keeps object versions (cached)."""
        if self._bucket_versioning is None:
            try:
                result = self._run_aws_command(
                    ['aws', 's3api', 'get-bucket-versioning',
                     '--bucket', self.s3_bucket])
                data = json.loads(result.stdout or '{}')
                self._bucket_versioning = data.get('Status') == 'Enabled'
            except Exception:
                self._bucket_versioning = False
        return self._bucket_versioning

    def _remote_md5(self, path):
        """Definitive remote content hash: download the object to a temp
        file and hash it. Cached per run; returns None when the fetch
        fails. Only used for hash-scoped (annotation-sized JSON) files
        whose ETag cannot prove equality."""
        if path in self._remote_md5_cache:
            return self._remote_md5_cache[path]
        print(f"  Fetching remote copy of {path} to verify content...")
        tmp = self._merge_tmp_path(path, '.hashcheck')
        md5 = None
        try:
            self._run_aws_command(['aws', 's3', 'cp', self._get_s3_url(path), tmp])
            md5 = self._file_md5(tmp)
        except Exception:
            md5 = None
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass
        self._remote_md5_cache[path] = md5
        return md5

    # --- shadow copies: the file content as of the last successful sync ---
    # These are the base versions for three-way merging of divergent
    # annotation files. Scope matches _should_hash.

    def _shadow_path(self, path):
        return os.path.join(self.local_dir, BASE_DIR_NAME, path)

    def _update_shadow(self, path):
        """Record the current local file as the last-synced (base) version."""
        src = os.path.join(self.local_dir, path)
        try:
            stat = os.stat(src)
        except OSError:
            return
        if not self._should_hash(path, stat.st_size):
            return
        dst = self._shadow_path(path)
        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            tmp = dst + '.tmp'
            shutil.copy2(src, tmp)
            os.replace(tmp, dst)
        except OSError as e:
            print(f"  ⚠️  Could not update merge base for {path}: {e}")

    def _remove_shadow(self, path):
        dst = self._shadow_path(path)
        try:
            os.remove(dst)
        except OSError:
            return
        shadow_root = os.path.join(self.local_dir, BASE_DIR_NAME)
        dirpath = os.path.dirname(dst)
        while dirpath != shadow_root and dirpath.startswith(shadow_root + os.sep):
            try:
                os.rmdir(dirpath)
            except OSError:
                break
            dirpath = os.path.dirname(dirpath)

    # --- conflict copies: versions preserved before they would be lost ---

    def _stash_conflict_copy(self, path, side, source=None):
        """Preserve a version of `path` that is about to be overwritten.

        side is 'local', 'remote', or 'base'. source is a filepath to copy;
        None with side='remote' fetches the current S3 object. Stashing must
        never block the sync itself, so failures only warn.
        """
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        stem, ext = os.path.splitext(os.path.basename(path))
        dst = os.path.join(self.local_dir, CONFLICT_DIR_NAME,
                           os.path.dirname(path),
                           f"{stem}.conflict-{stamp}-{side}{ext}")
        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if source is None and side == 'remote':
                self._run_aws_command(['aws', 's3', 'cp', self._get_s3_url(path), dst])
            else:
                shutil.copy2(source or os.path.join(self.local_dir, path), dst)
        except Exception as e:
            print(f"  ⚠️  Could not save {side} conflict copy for {path}: {e}")
            return None
        print(f"  Saved {side} conflict copy: {os.path.relpath(dst, self.local_dir)}")
        return dst

    def _stash_divergent_download_targets(self, download_paths):
        """Stash local copies a download would overwrite when their content
        diverges from the tracked synced state (or was never tracked).

        Hashes are taken NOW, not from the scan: the interactive prompt
        phase can be arbitrarily long, and an edit made during it is
        exactly what this last line of defense exists to preserve."""
        if not download_paths:
            return
        with self._get_db() as conn:
            cursor = conn.execute('SELECT path, local_md5 FROM files')
            tracked_md5 = {row['path']: row['local_md5'] for row in cursor}
        for path in download_paths:
            local_path = os.path.join(self.local_dir, path)
            try:
                stat = os.stat(local_path)
            except OSError:
                continue  # no local file to lose
            if not self._should_hash(path, stat.st_size):
                continue
            try:
                current_md5 = self._file_md5(local_path)
            except OSError:
                continue
            if tracked_md5.get(path) != current_md5:
                self._stash_conflict_copy(path, 'local')

    # --- fiber-aware three-way merge (see fiber_merge.py) ---

    def _merge_tmp_path(self, path, suffix):
        # Mirror the relative path — flattening to the basename would let
        # two same-named files in different directories overwrite each
        # other's pending merge data mid-run.
        if self._run_tmp_dir is None:
            self._run_tmp_dir = tempfile.mkdtemp(prefix='vc_sync-')
        tmp = os.path.join(self._run_tmp_dir, path + suffix)
        os.makedirs(os.path.dirname(tmp), exist_ok=True)
        return tmp

    def _cleanup_run_tmp(self):
        """Remove the per-run scratch dir (it can hold fiber content)."""
        if self._run_tmp_dir:
            shutil.rmtree(self._run_tmp_dir, ignore_errors=True)
            self._run_tmp_dir = None

    @staticmethod
    def _staging_path(target):
        """Same-directory (same-filesystem, so os.replace is atomic)
        staging name for an atomic write of `target`. Dot-prefixed so a
        crash-orphaned staging file is ignored by the scans."""
        return os.path.join(os.path.dirname(target),
                            '.' + os.path.basename(target) + '.s3sync-staged')

    def _load_base(self, path, tracked):
        """Return the parsed last-synced (base) version of a file, or None.

        Prefers the local shadow copy, verified against the tracked md5 it
        was recorded with; falls back to fetching the exact version from S3
        version history by the tracked ETag."""
        shadow = self._shadow_path(path)
        tracked_md5 = tracked.get('local_md5')
        if tracked_md5 and os.path.exists(shadow):
            try:
                if self._file_md5(shadow) == tracked_md5:
                    with open(shadow) as f:
                        return json.load(f)
            except (OSError, json.JSONDecodeError):
                pass
        etag = tracked.get('s3_etag')
        if etag and self._etag_is_md5(etag):
            return self._fetch_base_from_history(path, etag)
        return None

    def _fetch_base_from_history(self, path, etag):
        """Fetch the object version whose ETag matches the tracked baseline
        from the (versioned) bucket. Returns parsed JSON or None."""
        key = f"{self.s3_prefix}/{path}"
        try:
            result = self._run_aws_command(['aws', 's3api', 'list-object-versions',
                                            '--bucket', self.s3_bucket,
                                            '--prefix', key])
            data = json.loads(result.stdout or '{}')
        except Exception:
            return None
        version_id = None
        for version in data.get('Versions', []):
            if (version.get('Key') == key and
                    version.get('ETag', '').strip('"') == etag):
                version_id = version.get('VersionId')
                break
        if not version_id:
            return None
        tmp = self._merge_tmp_path(path, '.base')
        try:
            self._run_aws_command(['aws', 's3api', 'get-object',
                                   '--bucket', self.s3_bucket,
                                   '--key', key,
                                   '--version-id', version_id,
                                   tmp])
            with open(tmp) as f:
                return json.load(f)
        except Exception:
            return None
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass

    def _fetch_remote_json(self, path):
        """Fetch the current remote object. Returns (parsed_json, temp_path)
        — the caller stashes/removes temp_path — or (None, None)."""
        tmp = self._merge_tmp_path(path, '.remote')
        try:
            self._run_aws_command(['aws', 's3', 'cp', self._get_s3_url(path), tmp])
            with open(tmp) as f:
                return json.load(f), tmp
        except Exception:
            try:
                os.remove(tmp)
            except OSError:
                pass
            return None, None

    def _attempt_auto_merge(self, path, local_info, s3_info, dry_run=False):
        """Try a fiber-aware three-way merge for a conflicting file.

        The local file is NEVER modified here: a clean merge is written to
        a pending temp file and returned as
        {'pending': ..., 'remote_tmp': ..., 'summary': ...} for the caller
        to apply with _apply_pending_merges() AFTER the user confirms the
        sync (or discard with _discard_pending_merges() on cancellation).
        Returns a preview string in dry-run mode, or None when the file is
        not eligible, no base is available, or the merge has genuine
        conflicts (all three versions are stashed in that case and the
        caller falls back to the interactive prompt).
        """
        if fiber_merge is None or not local_info or not s3_info:
            return None
        if not self._should_hash(path, local_info.get('local_size')):
            return None
        local_path = os.path.join(self.local_dir, path)
        try:
            with open(local_path) as f:
                local_doc = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
        if not fiber_merge.is_fiber_doc(local_doc):
            return None

        with self._get_db() as conn:
            row = conn.execute('SELECT * FROM files WHERE path = ?',
                               (path,)).fetchone()
        tracked = dict(row) if row else {}
        base_doc = self._load_base(path, tracked)
        if base_doc is None or not fiber_merge.is_fiber_doc(base_doc):
            if not dry_run:
                print(f"  (no merge base available for {path})")
            return None

        remote_doc, remote_tmp = self._fetch_remote_json(path)
        if remote_doc is None or not fiber_merge.is_fiber_doc(remote_doc):
            if remote_tmp:
                try:
                    os.remove(remote_tmp)
                except OSError:
                    pass
            return None

        keep_remote_tmp = False
        try:
            # A merger bug on malformed input must degrade to manual
            # resolution, never abort the whole sync.
            try:
                result = fiber_merge.merge_fibers(base_doc, local_doc, remote_doc)
            except Exception as ex:
                print(f"  ⚠️  merge failed for {path} ({ex}); manual resolution")
                return None
            summary = fiber_merge.summarize(result)
            if dry_run:
                return f"would auto-merge ({summary})" if result['ok'] else None

            if not result['ok']:
                print(f"  ✗ cannot auto-merge {path}:")
                for conflict in result['conflicts']:
                    print(f"      {conflict}")
                # Preserve all three versions for a later manual 3-way merge
                self._stash_conflict_copy(path, 'local')
                self._stash_conflict_copy(path, 'remote', source=remote_tmp)
                shadow = self._shadow_path(path)
                if os.path.exists(shadow):
                    self._stash_conflict_copy(path, 'base', source=shadow)
                return None

            # Plan only: the local file is untouched until the user confirms
            pending = self._merge_tmp_path(path, '.merged')
            with open(pending, 'w') as f:
                json.dump(result['merged'], f, indent=2)
                f.write('\n')
            print(f"  ✓ will auto-merge {path}: {summary}")
            for note in result['notes']:
                print(f"      note: {note}")
            print("      (applied after you confirm the sync)")
            keep_remote_tmp = True
            return {'pending': pending, 'remote_tmp': remote_tmp,
                    'summary': summary,
                    # For the link-consistency pass (_plan_link_consistency)
                    'merged_doc': result['merged'],
                    'base_doc': base_doc,
                    'peer_files': result.get('peer_files', [])}
        finally:
            if remote_tmp and not keep_remote_tmp:
                try:
                    os.remove(remote_tmp)
                except OSError:
                    pass

    def _apply_pending_merges(self, pending_merges):
        """Apply confirmed merges: stash the versions being replaced, then
        atomically swap the merged result into place."""
        for path, plan in pending_merges:
            self._stash_conflict_copy(path, 'local')
            if plan['remote_tmp']:
                self._stash_conflict_copy(path, 'remote', source=plan['remote_tmp'])
            target = os.path.join(self.local_dir, path)
            # The pending file lives in the run temp dir (possibly another
            # filesystem); stage next to the target so the swap is atomic.
            # Dot-prefixed so a crash-orphaned staging file is ignored by
            # the scans instead of uploaded as a new object.
            staged = self._staging_path(target)
            shutil.copyfile(plan['pending'], staged)
            os.replace(staged, target)
            print(f"  ✓ applied auto-merge: {path}")
            for temp in (plan['pending'], plan['remote_tmp']):
                if temp:
                    try:
                        os.remove(temp)
                    except OSError:
                        pass

    @staticmethod
    def _discard_pending_merges(pending_merges):
        for _, plan in pending_merges:
            for temp in (plan['pending'], plan['remote_tmp']):
                if temp:
                    try:
                        os.remove(temp)
                    except OSError:
                        pass

    def _demote_merge(self, path, plan, reason):
        """Turn a planned auto-merge back into a manual conflict, preserving
        all three versions for later manual resolution."""
        print(f"  ✗ cannot auto-merge {path}: {reason}")
        self._stash_conflict_copy(path, 'local')
        if plan['remote_tmp'] and os.path.exists(plan['remote_tmp']):
            self._stash_conflict_copy(path, 'remote', source=plan['remote_tmp'])
        shadow = self._shadow_path(path)
        if os.path.exists(shadow):
            self._stash_conflict_copy(path, 'base', source=shadow)
        self._discard_pending_merges([(path, plan)])

    def _rewrite_pending(self, plan, doc):
        with open(plan['pending'], 'w') as f:
            json.dump(doc, f, indent=2)
            f.write('\n')
        plan['merged_doc'] = doc

    def _plan_link_consistency(self, pending_merges, download_paths):
        """Mirror each auto-merged fiber's link decisions into its peer
        files (fiber_merge.refresh_pair_links) so VC3D's index-exact
        cross-file reciprocity holds on the post-sync state.

        Pure planning over the files' FUTURE content (pending merge
        results, remote content for scheduled downloads, local content
        otherwise); nothing on disk is touched. Returns (peer_fixes,
        demoted): peer_fixes maps non-merge peer paths to their fixed
        docs (to write after downloads and upload), demoted lists
        (path, reason) merges that must fall back to manual resolution.
        Pending files of merges whose far-side fields were refreshed are
        rewritten in place (they are plan artifacts, not user files).
        """
        if fiber_merge is None:
            return {}, []
        plans = dict(pending_merges)
        future_docs = {path: plan['merged_doc'] for path, plan in pending_merges}
        changed = set()
        demoted = []

        def future_doc(path):
            if path in future_docs:
                return future_docs[path]
            if path in download_paths:
                doc, tmp = self._fetch_remote_json(path)
                if tmp:
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
            else:
                try:
                    with open(os.path.join(self.local_dir, path)) as f:
                        doc = json.load(f)
                except (OSError, json.JSONDecodeError):
                    doc = None
            future_docs[path] = doc
            return doc

        for path, plan in pending_merges:
            peers = plan.get('peer_files') or []
            a_name = os.path.basename(path)
            a_dir = os.path.dirname(path)
            a_doc = future_docs[path]
            a_changed = False
            # Tentative view: committed only if every peer of this merge
            # resolves, so a demotion cannot leak partial fixes.
            saved_docs = dict(future_docs)
            saved_changed = set(changed)
            failure = None
            for peer_name in peers:
                if peer_name == a_name:
                    # A self-referential link cannot be mirrored; never
                    # producible by VC3D, so demote rather than guess.
                    failure = "self-referential link"
                    break
                if '/' in peer_name or os.sep in peer_name or \
                        peer_name in ('.', '..'):
                    # Defense in depth: peer names are loader-normalized
                    # basenames; anything else must not become a path.
                    failure = f"invalid linked fiber name {peer_name!r}"
                    break
                peer_path = os.path.join(a_dir, peer_name) if a_dir else peer_name
                b_doc = future_doc(peer_path)
                if b_doc is None or not fiber_merge.is_fiber_doc(b_doc):
                    failure = f"linked fiber {peer_name} is missing or unreadable"
                    break
                # A refresh bug on malformed input must demote this merge,
                # never abort the whole sync mid-planning.
                try:
                    refreshed = fiber_merge.refresh_pair_links(
                        a_doc, b_doc, a_name, peer_name,
                        base_doc=plan.get('base_doc'))
                except Exception as ex:
                    failure = f"refresh against {peer_name} failed ({ex})"
                    break
                if not refreshed['ok']:
                    failure = '; '.join(refreshed['conflicts'])
                    break
                a_doc = refreshed['a_doc']
                a_changed = a_changed or refreshed['a_changed']
                if refreshed['b_changed']:
                    future_docs[peer_path] = refreshed['b_doc']
                    changed.add(peer_path)
            if failure:
                future_docs = saved_docs
                changed = saved_changed
                demoted.append((path, f"link consistency: {failure}"))
                continue
            future_docs[path] = a_doc
            if a_changed:
                changed.add(path)

        if demoted:
            return {}, demoted

        for path in changed & set(plans):
            self._rewrite_pending(plans[path], future_docs[path])
        peer_fixes = {path: future_docs[path]
                      for path in changed if path not in plans}
        return peer_fixes, demoted

    def _apply_peer_fixes(self, peer_fixes):
        """Write link-consistency fixes into peer files, stashing the prior
        versions. Runs IMMEDIATELY after the merges land, before any
        network transfer: a merged file and its peer fixes must hit disk
        together, or an interrupt leaves a cross-file inconsistency that no
        future sync heals (the merged file would re-upload as a plain local
        edit). Fix docs already embody the peer's post-sync content —
        including the remote content of peers whose download they
        supersede."""
        for path in sorted(peer_fixes):
            target = os.path.join(self.local_dir, path)
            if os.path.exists(target):
                self._stash_conflict_copy(path, 'peer')
            staged = self._staging_path(target)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(staged, 'w') as f:
                json.dump(peer_fixes[path], f, indent=2)
                f.write('\n')
            os.replace(staged, target)
            print(f"  ✓ link consistency fix applied: {path}")

    def _plan_conflict_resolutions(self, conflicts, local_files, s3_files,
                                   download_paths, delete_local_paths,
                                   auto_merge):
        """Split conflicts into planned auto-merges (with their peer fixes)
        and manual conflicts.

        ONE fixpoint drives both demotion causes — peers that are manual
        conflicts or pending local deletion, and peers the link-consistency
        planner cannot resolve — so a demotion from either stage re-runs
        the other. This is what guarantees a peer fix is never computed
        for (or applied over) a file whose fate the user later decides at
        the interactive prompt.

        Returns (pending_merges, peer_fixes, manual_conflicts).
        """
        pending_merges = []
        manual_conflicts = []
        for path, reason in conflicts:
            if auto_merge:
                plan = self._attempt_auto_merge(path, local_files.get(path),
                                                s3_files.get(path))
                if isinstance(plan, dict):
                    pending_merges.append((path, plan))
                    continue
            manual_conflicts.append((path, reason))

        def peer_paths(path, plan):
            directory = os.path.dirname(path)
            return [os.path.join(directory, name) if directory else name
                    for name in plan.get('peer_files') or []
                    if name != os.path.basename(path)]

        def demote(path, plan, reason):
            self._demote_merge(path, plan, reason)
            manual_conflicts.append(
                (path, "Both local and S3 modified since last sync"))
            manual_paths.add(path)

        manual_paths = {p for p, _ in manual_conflicts} | set(delete_local_paths)
        peer_fixes = {}
        while True:
            # Stage 1: cascade blocked-peer demotions to a fixpoint.
            progressed = True
            while progressed:
                progressed = False
                still_pending = []
                for path, plan in pending_merges:
                    blocked = sorted(os.path.basename(p)
                                     for p in peer_paths(path, plan)
                                     if p in manual_paths)
                    if blocked:
                        demote(path, plan,
                               "linked fiber(s) with unresolved conflicts "
                               "or pending deletion: " + ", ".join(blocked))
                        progressed = True
                    else:
                        still_pending.append((path, plan))
                pending_merges = still_pending
            if not pending_merges:
                peer_fixes = {}
                break
            # Stage 2: plan the cross-file link consistency; its demotions
            # feed back into stage 1 on the next iteration.
            peer_fixes, demoted = self._plan_link_consistency(
                pending_merges, download_paths)
            if not demoted:
                break
            demoted_reasons = dict(demoted)
            still_pending = []
            for path, plan in pending_merges:
                if path in demoted_reasons:
                    demote(path, plan, demoted_reasons[path])
                else:
                    still_pending.append((path, plan))
            pending_merges = still_pending

        return pending_merges, peer_fixes, manual_conflicts

    def scan_local_files(self, include_backups=False):
        """Scan local directory for files"""
        print(f"Scanning local directory: {self.local_dir}")
        files = {}

        for root, dirs, filenames in os.walk(self.local_dir):
            # Prune ignored directories early so huge layers/ trees aren't walked;
            # _is_ignored below stays the authority on what counts as ignored
            dirs[:] = [d for d in dirs if not d.startswith('.') and
                       'layers' not in d.lower() and
                       (include_backups or d != 'backups')]

            for filename in filenames:
                filepath = os.path.join(root, filename)
                relative_path = os.path.relpath(filepath, self.local_dir).replace(os.sep, '/')

                if self._is_ignored(relative_path, include_backups):
                    continue

                stat = os.stat(filepath)
                info = {
                    'path': relative_path,
                    'local_size': stat.st_size,
                    'local_mtime': stat.st_mtime,
                    'is_backup': is_backup_file(filename)
                }
                if self._should_hash(relative_path, stat.st_size):
                    try:
                        info['local_md5'] = self._file_md5(filepath)
                    except OSError:
                        pass
                files[relative_path] = info

        print(f"Found {len(files)} local files")
        return files

    def scan_s3_files(self, include_backups=False):
        """Scan S3 bucket for files with pagination support"""
        print(f"Scanning S3: s3://{self.s3_bucket}/{self.s3_prefix}/")
        files = {}
        continuation_token = None
        page_count = 0

        # Trailing slash keeps sibling prefixes (e.g. "<prefix>_old") out of the listing
        list_prefix = f"{self.s3_prefix}/" if self.s3_prefix else ""

        while True:
            cmd = [
                'aws', 's3api', 'list-objects-v2',
                '--bucket', self.s3_bucket,
                '--prefix', list_prefix
            ]

            if continuation_token:
                cmd.extend(['--continuation-token', continuation_token])

            result = self._run_aws_command(cmd)

            if not result.stdout:
                print("No files found in S3")
                break

            data = json.loads(result.stdout)

            if 'Contents' not in data:
                if page_count == 0:
                    print("No files found in S3")
                break

            for obj in data['Contents']:
                # Skip if it's just the prefix itself
                if obj['Key'] == list_prefix:
                    continue

                relative_path = obj['Key'][len(list_prefix):]

                if self._is_ignored(relative_path, include_backups):
                    continue

                files[relative_path] = {
                    'path': relative_path,
                    's3_size': obj['Size'],
                    's3_mtime': self._parse_timestamp(obj['LastModified']),
                    's3_etag': obj.get('ETag', '').strip('"'),
                    'is_backup': is_backup_file(os.path.basename(relative_path))
                }

            page_count += 1

            if not data.get('IsTruncated'):
                break

            continuation_token = data.get('NextContinuationToken')
            if not continuation_token:
                break

            if page_count % 10 == 0:
                print(f"  Scanned {len(files)} files so far...")

        print(f"Found {len(files)} S3 files")
        return files

    def _prune_stale_tracking(self, current_paths, include_backups=False):
        """Drop tracking rows for paths that no longer exist locally or on S3.

        Rows that are invisible to this run's scans only because backups/
        directories were excluded are left alone — the scans carry no
        information about them. Rows for permanently ignored paths (hidden,
        layers, .obj) are still pruned.
        """
        def invisible_backup(path):
            return (not include_backups and
                    self._is_ignored(path) and
                    not self._is_ignored(path, include_backups=True))

        with self._get_db() as conn:
            cursor = conn.execute('SELECT path FROM files')
            stale = [row['path'] for row in cursor
                     if row['path'] not in current_paths
                     and not invisible_backup(row['path'])]
            for path in stale:
                conn.execute('DELETE FROM files WHERE path = ?', (path,))

        for path in stale:
            self._remove_shadow(path)

        if stale:
            print(f"Pruned {len(stale)} tracking entries for files that no longer exist anywhere")
        return len(stale)

    def refresh_tracking(self, include_backups=False):
        """Non-destructive tracking refresh (the 'update' command).

        Records files that are already identical on both sides and prunes
        entries for files gone from both. Tracking for files with pending
        differences is left untouched, so 'sync' still proposes them.
        For annotation files this also backfills content hashes and merge
        bases onto rows that predate content hashing — run `update` once
        while in sync to seed the three-way merge bases.
        """
        print("\nRefreshing file tracking...")

        local_files = self.scan_local_files(include_backups)
        s3_files = self.scan_s3_files(include_backups)

        self._record_untracked_synced(local_files, s3_files)
        self._backfill_tracked_bases(local_files, s3_files)
        self._prune_stale_tracking(set(local_files) | set(s3_files), include_backups)

        print("File tracking refreshed")

    def _backfill_tracked_bases(self, local_files, s3_files):
        """Seed content hashes and shadow (merge-base) copies for
        already-tracked files that predate content hashing.

        Only files whose local content is VERIFIED identical to the remote
        (plain-MD5 ETag match, or hashing the fetched remote content when
        the ETag proves nothing) are touched; divergent or unverifiable
        pairs keep their pending differences for `sync` to propose."""
        with self._get_db() as conn:
            rows = conn.execute('SELECT path, local_md5 FROM files').fetchall()
        tracked_md5 = {row['path']: row['local_md5'] for row in rows}

        # Verification (which may fetch remote content) runs before the
        # tracking transaction opens.
        seeded = []
        for path, known_md5 in tracked_md5.items():
            local_info = local_files.get(path)
            s3_info = s3_files.get(path)
            if not local_info or not s3_info:
                continue
            md5 = local_info.get('local_md5')
            if not md5:
                continue  # not hash-scoped
            if known_md5 == md5 and os.path.exists(self._shadow_path(path)):
                continue  # already seeded
            if self._content_matches_remote(path, md5, s3_info.get('s3_etag')):
                seeded.append(path)

        with self._get_db() as conn:
            for path in seeded:
                self._track_file(conn, path, local_files[path], s3_files[path])

        for path in seeded:
            self._update_shadow(path)
        if seeded:
            print(f"Backfilled content hash + merge base for {len(seeded)} "
                  f"tracked file(s)")

    def reset_tracking(self, include_backups=False):
        """Stamp the current local and S3 state as the synced baseline.

        Destructive: files that currently differ between local and S3 are
        recorded as in sync, so their pending differences will never be
        proposed. The 'reset' command confirms before calling this.
        """
        print("\nResetting file tracking...")

        local_files = self.scan_local_files(include_backups)
        s3_files = self.scan_s3_files(include_backups)
        current_paths = set(local_files) | set(s3_files)

        with self._get_db() as conn:
            for path in current_paths:
                self._track_file(conn, path, local_files.get(path), s3_files.get(path))

        # Reset stamps "in sync" even for files that differ, so a shadow copy
        # is only trustworthy as a merge base where content is verified equal;
        # anywhere else it would poison future 3-way merges.
        for path in current_paths:
            local_info = local_files.get(path)
            s3_info = s3_files.get(path)
            md5 = local_info.get('local_md5') if local_info else None
            etag = (s3_info.get('s3_etag') or '') if s3_info else ''
            if md5 and self._etag_is_md5(etag) and md5 == etag:
                self._update_shadow(path)
            else:
                self._remove_shadow(path)

        self._prune_stale_tracking(current_paths, include_backups)

        print("File tracking reset")

    def analyze_changes(self, local_files, s3_files, record=False):
        """Analyze what needs to be synced and detect conflicts.

        With record=True (sync, not dry-run), verified ground truth
        discovered during analysis is written back to tracking: MD5 backfills
        for rows predating content hashing, and re-tracking of files whose
        local and remote content converged independently.
        """
        actions = {}
        md5_backfills = []   # (md5, path) for rows predating content hashing
        converged = []       # paths where both sides changed to identical content

        with self._get_db() as conn:
            # Get all tracked files
            cursor = conn.execute('SELECT * FROM files')
            tracked_files = {row['path']: dict(row) for row in cursor}

        # Get all paths
        all_paths = set(tracked_files.keys()) | set(local_files.keys()) | set(s3_files.keys())

        for path in all_paths:
            local_info = local_files.get(path)
            s3_info = s3_files.get(path)
            tracked_info = tracked_files.get(path, {})

            # Check if this is a backup file
            is_backup = (local_info and local_info.get('is_backup')) or \
                        (s3_info and s3_info.get('is_backup'))

            # Backup files: only upload, never download or delete
            if is_backup:
                if local_info and not s3_info:
                    actions[path] = (SyncAction.UPLOAD, "Backup file (new)")
                elif local_info and s3_info:
                    # Check if local backup changed
                    local_changed = (tracked_info.get('local_size') != local_info['local_size'] or
                                     (tracked_info.get('local_mtime') and
                                      abs(tracked_info['local_mtime'] - local_info['local_mtime']) > 1))
                    if local_changed:
                        actions[path] = (SyncAction.UPLOAD, "Backup file (modified)")
                    else:
                        actions[path] = (SyncAction.SKIP, "Backup file (in sync)")
                elif s3_info and not local_info:
                    # Backup exists on S3 but not locally - skip (never download backups)
                    actions[path] = (SyncAction.SKIP, "Backup file (S3 only, not downloading)")
                continue

            # Regular file logic (non-backup)
            # File only exists locally
            if local_info and not s3_info:
                if tracked_info.get('s3_size') is not None:
                    actions[path] = (SyncAction.DELETE_LOCAL, "S3 file was deleted")
                else:
                    actions[path] = (SyncAction.UPLOAD, "New local file")

            # File only exists on S3
            elif s3_info and not local_info:
                if tracked_info.get('local_size') is not None:
                    actions[path] = (SyncAction.DELETE_REMOTE, "Local file was deleted")
                else:
                    actions[path] = (SyncAction.DOWNLOAD, "New S3 file")

            # File exists in both places
            elif local_info and s3_info:
                current_md5 = local_info.get('local_md5')
                s3_etag = s3_info.get('s3_etag') or ''
                if tracked_info:
                    # We have tracking history
                    stat_changed = (tracked_info.get('local_size') != local_info['local_size'] or
                                    (tracked_info.get('local_mtime') and
                                     abs(tracked_info['local_mtime'] - local_info['local_mtime']) > 1))

                    tracked_md5 = tracked_info.get('local_md5')
                    if current_md5 and tracked_md5:
                        # Content is authoritative for hashed files: a
                        # touch/re-save with identical bytes is not a change,
                        # and a size+mtime-preserving edit is.
                        local_changed = current_md5 != tracked_md5
                    else:
                        local_changed = stat_changed
                        if current_md5 and not tracked_md5 and not stat_changed:
                            # Row predates content hashing and the stats say
                            # the file is unchanged since last sync: adopt the
                            # current content as the tracked content.
                            md5_backfills.append((current_md5, path))

                    s3_changed = (tracked_info.get('s3_size') != s3_info['s3_size'] or
                                  tracked_info.get('s3_etag') != s3_info['s3_etag'])

                    if local_changed and s3_changed:
                        if (current_md5 and
                                self._content_matches_remote(path, current_md5,
                                                             s3_etag)):
                            # Both sides changed but ended up with identical
                            # content (e.g. both machines synced a third copy)
                            actions[path] = (SyncAction.SKIP,
                                             "Both sides converged to identical content")
                            converged.append(path)
                        else:
                            actions[path] = (SyncAction.CONFLICT,
                                             "Both local and S3 modified since last sync")
                    elif local_changed:
                        actions[path] = (SyncAction.UPLOAD, "Local file modified")
                    elif s3_changed:
                        actions[path] = (SyncAction.DOWNLOAD, "S3 file modified")
                    else:
                        actions[path] = (SyncAction.SKIP, "Files are in sync")
                else:
                    # No tracking history
                    if local_info['local_size'] != s3_info['s3_size']:
                        actions[path] = (SyncAction.CONFLICT, "Files differ (no sync history)")
                    elif not current_md5:
                        actions[path] = (SyncAction.SKIP, "Files appear to be in sync")
                    elif self._content_matches_remote(path, current_md5, s3_etag):
                        # A plain-MD5 ETag match, or the remote content
                        # hashed when the ETag proves nothing — "unknown"
                        # must not silently become "in sync".
                        actions[path] = (SyncAction.SKIP,
                                         "Files verified identical (no sync history)")
                    elif self._remote_md5(path) is None:  # cached, no refetch
                        actions[path] = (SyncAction.CONFLICT,
                                         "Content could not be verified (no sync history)")
                    else:
                        actions[path] = (SyncAction.CONFLICT,
                                         "Same size but different content (no sync history)")

            # File deleted from both
            elif path in tracked_files and not local_info and not s3_info:
                actions[path] = (SyncAction.SKIP, "File deleted from both")

        if record and (md5_backfills or converged):
            with self._get_db() as conn:
                if md5_backfills:
                    conn.executemany('UPDATE files SET local_md5 = ? WHERE path = ?',
                                     md5_backfills)
                for path in converged:
                    self._track_file(conn, path, local_files[path], s3_files[path])
            for path in converged:
                self._update_shadow(path)

        return actions

    def _record_untracked_synced(self, local_files, s3_files):
        """Record tracking for size-matched files that have no sync history.

        Without a tracking row, a file later deleted locally looks like a new
        S3 file and gets re-downloaded instead of proposed for remote
        deletion. This self-heals tracking lost to an interrupted init/update.
        """
        with self._get_db() as conn:
            tracked = {row['path'] for row in conn.execute('SELECT path FROM files')}

        # Backup-pattern files are excluded: they are upload-only, and
        # analyze_changes deliberately re-uploads them when untracked
        # (even on a size match). The upload itself records tracking.
        # Hashable files must additionally be verified identical — by a
        # matching plain-MD5 ETag, or by hashing the remote content when
        # the ETag proves nothing. Unverified pairs stay untracked rather
        # than becoming a false baseline. Verification (network) runs
        # OUTSIDE the tracking transaction.
        def content_verified(path):
            md5 = local_files[path].get('local_md5')
            if not md5:
                return True  # not hash-scoped: size heuristic as before
            return self._content_matches_remote(
                path, md5, s3_files[path].get('s3_etag'))

        healed = [path for path in local_files.keys() & s3_files.keys()
                  if path not in tracked and
                  not local_files[path].get('is_backup') and
                  local_files[path]['local_size'] == s3_files[path]['s3_size'] and
                  content_verified(path)]

        with self._get_db() as conn:
            for path in healed:
                self._track_file(conn, path, local_files[path], s3_files[path])

        for path in healed:
            if local_files[path].get('local_md5'):
                # Content-verified identical: safe to adopt as a merge base
                self._update_shadow(path)

        if healed:
            print(f"Recorded {len(healed)} in-sync file(s) that had no tracking history")

    def resolve_conflict(self, path, reason, local_info, s3_info):
        """Interactively resolve a conflict"""
        print(f"\n⚠️  CONFLICT: {path}")
        print(f"Reason: {reason}")

        if local_info and s3_info:
            print(f"  Local:  Size={local_info['local_size']:,} bytes, "
                  f"Modified={datetime.fromtimestamp(local_info['local_mtime'])}")
            print(f"  S3:     Size={s3_info['s3_size']:,} bytes, "
                  f"Modified={datetime.fromtimestamp(s3_info['s3_mtime'])}")

            if "both" in reason.lower():
                print("  ⚠️  Both files have been modified since last sync!")

            response = prompt_choice(
                "\nChoose: [l]ocal → remote, [r]emote → local, [s]kip? ",
                ('l', 'r', 's'))

            # Preserve whichever version is about to be discarded
            if response == 'r':
                self._stash_conflict_copy(path, 'local')
            elif response == 'l':
                if self._should_hash(path, s3_info['s3_size']):
                    self._stash_conflict_copy(path, 'remote')
                elif self._bucket_versioning_enabled():
                    print("  (previous remote version remains available "
                          "through S3 bucket versioning)")
                else:
                    print("  ⚠️  Bucket versioning is NOT enabled: the "
                          "previous remote version will be overwritten "
                          "without a preserved copy")

            return {'l': SyncAction.UPLOAD,
                    'r': SyncAction.DOWNLOAD}.get(response, SyncAction.SKIP)

        return SyncAction.SKIP

    def perform_upload(self, path, local_files):
        """Upload a single file to S3 and update tracking"""
        local_path = os.path.join(self.local_dir, path)
        s3_path = self._get_s3_url(path)

        # Re-stat immediately before upload: interactive prompts can leave a long
        # window between scan and upload, during which the file may have changed
        pre_stat = os.stat(local_path)

        print(f"  Uploading: {path} → remote")

        cmd = ['aws', 's3', 'cp', local_path, s3_path]
        self._run_aws_command(cmd)

        post_stat = os.stat(local_path)
        if (post_stat.st_size, post_stat.st_mtime) != (pre_stat.st_size, pre_stat.st_mtime):
            print(f"  ⚠️  {path} changed while it was being uploaded; "
                  f"the S3 copy may be incomplete and will be re-uploaded on next sync")

        print(f"  ✓ Uploaded: {path}")

        # Get fresh S3 info
        cmd = ['aws', 's3api', 'head-object', '--bucket', self.s3_bucket,
               '--key', f"{self.s3_prefix}/{path}"]
        result = self._run_aws_command(cmd)

        data = json.loads(result.stdout)
        s3_info = {
            's3_size': data['ContentLength'],
            's3_mtime': self._parse_timestamp(data['LastModified']),
            's3_etag': data.get('ETag', '').strip('"'),
        }

        # Track the pre-upload stats: if the file changed during upload, the next
        # scan will see a local difference and schedule a re-upload.
        with self._get_db() as conn:
            hashed = self._track_transfer(conn, path, s3_info,
                                          expect_size=pre_stat.st_size,
                                          expect_mtime=pre_stat.st_mtime)
        if hashed:
            self._update_shadow(path)

        return True

    def _track_transfer(self, conn, path, s3_info,
                        expect_size=None, expect_mtime=None):
        """Record tracking for a just-transferred file; returns True when a
        content hash was recorded (caller then seeds the shadow copy).

        With expect_* given (uploads), the hash is recorded only if the
        file is still bit-for-bit what was sent, so the tracked hash
        matches what actually reached S3. ETags are never consulted for
        this (opaque under SSE-KMS); the CLI checksums the transfer
        itself."""
        local_path = os.path.join(self.local_dir, path)
        stat = os.stat(local_path)
        size = stat.st_size if expect_size is None else expect_size
        mtime = stat.st_mtime if expect_mtime is None else expect_mtime
        local_info = {'local_size': size, 'local_mtime': mtime}
        stable = (expect_size is None or
                  (stat.st_size, stat.st_mtime) == (expect_size, expect_mtime))
        if stable and self._should_hash(path, stat.st_size):
            try:
                local_info['local_md5'] = self._file_md5(local_path)
            except OSError:
                pass
        self._track_file(conn, path, local_info, s3_info)
        return 'local_md5' in local_info

    def perform_download(self, path, s3_files):
        """Download a single file from S3 and update tracking"""
        local_path = os.path.join(self.local_dir, path)
        s3_path = self._get_s3_url(path)

        # Create directory if needed
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        print(f"  Downloading: remote → {path}")

        cmd = ['aws', 's3', 'cp', s3_path, local_path]
        self._run_aws_command(cmd)

        print(f"  ✓ Downloaded: {path}")

        # The file just written IS the synced content (the CLI checksums
        # transfers), so its own hash is the tracking/base truth.
        with self._get_db() as conn:
            hashed = self._track_transfer(conn, path, s3_files[path])
        if hashed:
            self._update_shadow(path)

        return True

    def perform_delete_local(self, path):
        """Delete a local file and update tracking"""
        local_path = os.path.join(self.local_dir, path)

        print(f"  Deleting local: {path}")
        os.remove(local_path)
        print(f"  ✓ Deleted local: {path}")

        # Clean up empty directories
        self._cleanup_empty_dirs(local_path)

        # Remove from database
        with self._get_db() as conn:
            conn.execute('DELETE FROM files WHERE path = ?', (path,))
        self._remove_shadow(path)

        return True

    def perform_delete_remote(self, path):
        """Delete a file from S3 and update tracking"""
        s3_path = self._get_s3_url(path)

        print(f"  Deleting from S3: {path}")

        cmd = ['aws', 's3', 'rm', s3_path]
        self._run_aws_command(cmd)

        print(f"  ✓ Deleted from S3: {path}")

        # Remove from database
        with self._get_db() as conn:
            conn.execute('DELETE FROM files WHERE path = ?', (path,))
        self._remove_shadow(path)

        return True

    def _rclone_remote(self):
        """On-the-fly rclone remote for the S3 target (no rclone.conf needed).

        env_auth=true uses the standard AWS credential chain, so this works
        both with credentials pasted into the terminal (env vars) and with
        EC2 instance roles — same sources as the aws CLI.

        no_head=true skips rclone's post-upload verification HEAD. On a
        versioned bucket that HEAD targets the new object's versionId, which
        needs the s3:GetObjectVersion permission — credentials without it
        (but with Get/Put/List) see every upload 403 on rclone's first
        attempt even though the PUT itself succeeded. The batch methods below
        verify every upload themselves against fresh S3 metadata, so
        rclone's HEAD is redundant here.
        """
        return (f":s3,provider=AWS,env_auth=true,no_check_bucket=true,"
                f"no_head=true:{self.s3_bucket}/{self.s3_prefix}")

    def _rclone_env(self):
        """Environment for rclone subprocesses, mirroring aws CLI credential/region config.

        When the aws CLI can export the credentials it resolved (cached SSO /
        assume-role sessions, instance-role creds from IMDS, pasted env vars),
        they are injected as static env vars so rclone uses the exact identity
        the scan phase just used. rclone resolving credentials on its own is a
        known source of transient 403s (flaky IMDS fetches, fresh-token
        propagation races) and hard failures (SSO profiles on older rclone).
        """
        env = os.environ.copy()
        if self.aws_profile:
            env['AWS_PROFILE'] = self.aws_profile

        creds = self._export_aws_credentials()
        if creds:
            # Drop any stale session token so it can't be combined with the
            # exported key pair (long-lived keys export no session token)
            env.pop('AWS_SESSION_TOKEN', None)
            env.update(creds)

        # State the credential mode once per process: essential when
        # debugging 403s from machines we can't see
        if not getattr(self, '_creds_mode_printed', False):
            self._creds_mode_printed = True
            if creds:
                print("  rclone credentials: injected from aws CLI")
            else:
                print("  rclone credentials: resolving independently "
                      "(aws CLI could not export them)")

        region = env.get('AWS_REGION') or env.get('AWS_DEFAULT_REGION')
        if region:
            env.setdefault('RCLONE_S3_REGION', region)
        return env

    def _export_aws_credentials(self):
        """Credentials as resolved by the aws CLI, as env vars for rclone.

        Returns {} when `aws configure export-credentials` is unavailable
        (aws CLI v1 / old v2) or fails for any reason; rclone then resolves
        credentials itself via env_auth, exactly as before.
        """
        cmd = ['aws', 'configure', 'export-credentials', '--format', 'env-no-export']
        if self.aws_profile:
            cmd += ['--profile', self.aws_profile]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            return {}
        if result.returncode != 0:
            return {}

        wanted = ('AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_SESSION_TOKEN')
        creds = {}
        for line in result.stdout.splitlines():
            key, sep, value = line.strip().partition('=')
            if sep and key in wanted and value:
                creds[key] = value

        # A partial key pair must not shadow rclone's own resolution
        if 'AWS_ACCESS_KEY_ID' not in creds or 'AWS_SECRET_ACCESS_KEY' not in creds:
            return {}
        return creds

    def _run_rclone(self, args, paths):
        """Run one rclone command over a --files-from list of relative paths"""
        # The list file lives inside the sync dir (hidden, so scans skip it):
        # sandboxed rclone installs (e.g. snap) often cannot read /tmp, but
        # must be able to read the sync dir for transfers to work at all
        with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False,
                                         dir=self.local_dir, prefix='.s3sync-files-') as f:
            f.write('\n'.join(paths) + '\n')
            list_path = f.name

        cmd = ['rclone'] + args + [
            '--files-from', list_path,
            '--transfers', str(self.RCLONE_TRANSFERS),
            '--checkers', str(self.RCLONE_CHECKERS),
            '--stats-one-line', '--stats', '15s',
        ]

        # Debugging escape hatch, e.g. to capture which HTTP request a 403
        # comes from: VC_SYNC_RCLONE_FLAGS='-vv --dump headers' (rclone
        # redacts the Authorization header in dumps)
        extra_flags = os.environ.get('VC_SYNC_RCLONE_FLAGS')
        if extra_flags:
            cmd += shlex.split(extra_flags)

        try:
            result = subprocess.run(cmd, env=self._rclone_env())
            return result.returncode == 0
        finally:
            try:
                os.unlink(list_path)
            except OSError:
                pass

    # Verify batches up to this size with per-file head-object calls; larger
    # batches use one full prefix listing instead
    VERIFY_HEAD_MAX = 8

    def _fetch_s3_info(self, paths, include_backups=False):
        """Fresh S3 metadata for the given paths, for post-transfer verification.

        Returns {path: s3_info} containing only paths that exist remotely.
        Small batches use per-file head-object calls (cheaper than listing a
        large prefix); anything bigger reuses the full listing scan. Errors
        other than a clean 404 keep the path in the result with unknown size,
        so callers treat it conservatively (upload unverified / delete not
        confirmed) rather than concluding the file is gone.
        """
        if len(paths) > self.VERIFY_HEAD_MAX:
            print("  Verifying against a fresh S3 listing...")
            return self.scan_s3_files(include_backups)

        print(f"  Verifying {len(paths)} file(s) against S3...")
        info = {}
        for path in paths:
            cmd = ['aws', 's3api', 'head-object', '--bucket', self.s3_bucket,
                   '--key', f"{self.s3_prefix}/{path}"]
            if self.aws_profile:
                cmd += ['--profile', self.aws_profile]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                if '404' in (result.stderr or ''):
                    continue  # confirmed absent
                print(f"  ⚠️  Could not verify {path}: {(result.stderr or '').strip()}")
                info[path] = {'path': path, 's3_size': None, 's3_mtime': None}
                continue

            data = json.loads(result.stdout)
            info[path] = {
                'path': path,
                's3_size': data['ContentLength'],
                's3_mtime': self._parse_timestamp(data['LastModified']),
                's3_etag': data.get('ETag', '').strip('"'),
            }
        return info

    def perform_uploads_batch(self, paths, include_backups=False):
        """Upload files to S3 in one parallel rclone run and update tracking"""
        if not paths:
            return 0

        # Stat immediately before the transfer so mid-upload changes are detectable
        pre_stats = {}
        for path in paths:
            try:
                stat = os.stat(os.path.join(self.local_dir, path))
                pre_stats[path] = (stat.st_size, stat.st_mtime)
            except OSError as e:
                print(f"  ❌ Cannot read {path}: {e}")

        if not pre_stats:
            return 0

        print(f"  Uploading {len(pre_stats)} files ({self.RCLONE_TRANSFERS} parallel transfers)...")
        ok = self._run_rclone(['copy', self.local_dir, self._rclone_remote()],
                              list(pre_stats))
        if not ok:
            # Don't advance tracking on a partial failure: a same-size stale
            # remote object would pass the size check below and the file would
            # then be recorded as in sync, permanently skipping the re-upload.
            # Leaving tracking unchanged re-schedules every file next sync;
            # rclone skips the ones that did transfer, so the retry is cheap.
            print(f"  ❌ rclone reported errors during upload; tracking left "
                  f"unchanged for all {len(pre_stats)} files, will retry on next sync")
            return 0

        for path, (size, mtime) in pre_stats.items():
            try:
                stat = os.stat(os.path.join(self.local_dir, path))
                if (stat.st_size, stat.st_mtime) != (size, mtime):
                    print(f"  ⚠️  {path} changed while it was being uploaded; "
                          f"the S3 copy may be incomplete and will be re-uploaded on next sync")
            except OSError:
                print(f"  ⚠️  {path} disappeared during upload")

        fresh_s3 = self._fetch_s3_info(list(pre_stats), include_backups)

        success_count = 0
        shadow_updates = []
        with self._get_db() as conn:
            for path, (size, mtime) in pre_stats.items():
                s3_info = fresh_s3.get(path)
                if not s3_info or s3_info['s3_size'] != size:
                    print(f"  ❌ Upload not verified for {path}; will retry on next sync")
                    continue
                try:
                    if self._track_transfer(conn, path, s3_info,
                                            expect_size=size,
                                            expect_mtime=mtime):
                        shadow_updates.append(path)
                except OSError:
                    print(f"  ⚠️  {path} disappeared after upload")
                    continue
                success_count += 1

        for path in shadow_updates:
            self._update_shadow(path)

        print(f"  ✓ Uploaded {success_count}/{len(paths)} files")
        return success_count

    def perform_downloads_batch(self, paths, s3_files):
        """Download files from S3 in one parallel rclone run and update tracking"""
        if not paths:
            return 0

        print(f"  Downloading {len(paths)} files ({self.RCLONE_TRANSFERS} parallel transfers)...")
        ok = self._run_rclone(['copy', self._rclone_remote(), self.local_dir], paths)
        if not ok:
            # Mirror of the upload case: a same-size stale local file would
            # pass the size check and be recorded as in sync with the new
            # remote ETag, permanently skipping the re-download
            print(f"  ❌ rclone reported errors during download; tracking left "
                  f"unchanged for all {len(paths)} files, will retry on next sync")
            return 0

        success_count = 0
        shadow_updates = []
        with self._get_db() as conn:
            for path in paths:
                local_path = os.path.join(self.local_dir, path)
                try:
                    stat = os.stat(local_path)
                except OSError:
                    print(f"  ❌ Download not verified for {path}; will retry on next sync")
                    continue
                if stat.st_size != s3_files[path]['s3_size']:
                    print(f"  ❌ Size mismatch for {path}; will retry on next sync")
                    continue
                if self._track_transfer(conn, path, s3_files[path]):
                    shadow_updates.append(path)
                success_count += 1

        for path in shadow_updates:
            self._update_shadow(path)

        print(f"  ✓ Downloaded {success_count}/{len(paths)} files")
        return success_count

    def perform_deletes_remote_batch(self, paths, include_backups=False):
        """Delete files from S3 in one rclone run and update tracking"""
        if not paths:
            return 0

        print(f"  Deleting {len(paths)} files from S3...")
        ok = self._run_rclone(['delete', self._rclone_remote()], paths)
        if not ok:
            print("  ⚠️  rclone reported errors during deletion; verifying what was deleted")

        # rclone delete exits 0 even when --files-from matched nothing on the
        # remote, so the exit code alone can't confirm deletion. Verify against
        # fresh S3 metadata and only drop tracking for files that are actually
        # gone — a cleared row for a still-present file would make the next
        # sync propose re-downloading it instead of retrying the delete.
        fresh_s3 = self._fetch_s3_info(paths, include_backups)

        success_count = 0
        deleted_paths = []
        with self._get_db() as conn:
            for path in paths:
                if path in fresh_s3:
                    print(f"  ❌ {path} is still present on S3; will retry on next sync")
                    continue
                conn.execute('DELETE FROM files WHERE path = ?', (path,))
                deleted_paths.append(path)
                success_count += 1

        for path in deleted_paths:
            self._remove_shadow(path)

        print(f"  ✓ Deleted {success_count}/{len(paths)} files from S3")
        return success_count

    def _print_file_preview(self, files, title, max_files=50):
        """Print preview of files to be processed"""
        if not files:
            return

        print(f"\n{title} ({len(files)} total):")
        for i, (path, reason) in enumerate(sorted(files)[:max_files], 1):
            print(f"  {i}. {path}")
            if reason:
                print(f"     └─ {reason}")

        if len(files) > max_files:
            print(f"  ... and {len(files) - max_files} more files")

    @staticmethod
    def _print_sync_summary(uploads, downloads, deletes_local, deletes_remote,
                            conflicts_label, conflicts_count):
        """Summary of pending operations, printed right before the user decides"""
        print(f"\nSync Summary:")
        print(f"  Uploads pending:    {len(uploads)}")
        print(f"  Downloads pending:  {len(downloads)}")
        print(f"  Local deletions:    {len(deletes_local)}")
        print(f"  Remote deletions:   {len(deletes_remote)}")
        print(f"  {conflicts_label + ':':<20}{conflicts_count}")

    def _validate_upload_candidates(self, paths):
        """Flag zero-byte files and unparseable JSON among upload candidates"""
        flagged = []
        for path in paths:
            local_path = os.path.join(self.local_dir, path)
            try:
                if os.path.getsize(local_path) == 0:
                    flagged.append((path, "zero-byte file"))
                    continue
                if path.lower().endswith('.json'):
                    with open(local_path, 'r') as f:
                        json.load(f)
            except json.JSONDecodeError as e:
                flagged.append((path, f"unparseable JSON ({e.msg} at line {e.lineno})"))
            except (OSError, UnicodeDecodeError) as e:
                flagged.append((path, f"unreadable ({e})"))
        return flagged

    def sync(self, dry_run=False, include_backups=False, auto_merge=True):
        """Perform interactive sync operation"""
        if not include_backups:
            print("Note: Ignoring backups/ directories (use --sync-backups to include them)")

        if self.use_rclone:
            print("Transfer backend: rclone (parallel transfers)")
        else:
            print("Transfer backend: aws CLI (serial)")
            print(f"⚠️  rclone unavailable, syncs will be slower: {self.rclone_unavailable_reason}")

        print("\nAnalyzing changes...")

        local_files = self.scan_local_files(include_backups)
        s3_files = self.scan_s3_files(include_backups)

        actions = self.analyze_changes(local_files, s3_files, record=not dry_run)

        # Self-heal tracking so a future local deletion of these files is
        # proposed as a remote delete rather than a re-download
        if not dry_run:
            self._record_untracked_synced(local_files, s3_files)

        # Separate actions by type
        uploads = []
        downloads = []
        deletes_local = []
        deletes_remote = []
        conflicts = []

        for path, (action, reason) in sorted(actions.items()):
            if action == SyncAction.UPLOAD:
                uploads.append((path, reason))
            elif action == SyncAction.DOWNLOAD:
                downloads.append((path, reason))
            elif action == SyncAction.DELETE_LOCAL:
                deletes_local.append((path, reason))
            elif action == SyncAction.DELETE_REMOTE:
                deletes_remote.append((path, reason))
            elif action == SyncAction.CONFLICT:
                conflicts.append((path, reason))

        if not any([uploads, downloads, deletes_local, deletes_remote, conflicts]):
            print("\n✓ Everything is in sync!")
            return

        # Show preview of files
        self._print_file_preview(uploads, "Files to Upload")
        self._print_file_preview(downloads, "Files to Download")
        self._print_file_preview(deletes_local, "Files to Delete Locally")
        self._print_file_preview(deletes_remote, "Files to Delete from S3")
        self._print_file_preview(conflicts, "Conflicts to Resolve")

        # Flag suspect upload candidates (zero-byte files, unparseable JSON)
        invalid_uploads = self._validate_upload_candidates([p for p, _ in uploads])

        if dry_run:
            if invalid_uploads:
                print(f"\n⚠️  {len(invalid_uploads)} upload candidate(s) look invalid:")
                for path, problem in invalid_uploads:
                    print(f"  {path}: {problem}")
            if conflicts and auto_merge and fiber_merge is not None:
                print("\nMerge preview for conflicts (fetches remote copies "
                      "to test-merge; local files are not touched):")
                for path, reason in conflicts:
                    probe = self._attempt_auto_merge(path,
                                                     local_files.get(path),
                                                     s3_files.get(path),
                                                     dry_run=True)
                    print(f"  {path}: {probe or 'manual resolution required'}")
            self._print_sync_summary(uploads, downloads, deletes_local, deletes_remote,
                                     "Conflicts", len(conflicts))
            print("\n--dry-run mode: No changes will be made")
            return

        # Process conflicts first: plan auto-merges (and their cross-file
        # link-consistency fixes) for what we safely can, prompt for the
        # rest. Merges are only applied after confirmation.
        pending_merges, peer_fixes, manual_conflicts = \
            self._plan_conflict_resolutions(conflicts, local_files, s3_files,
                                            {p for p, _ in downloads},
                                            {p for p, _ in deletes_local},
                                            auto_merge)

        for path, _ in pending_merges:
            uploads.append((path, "Auto-merged local + remote changes"))
        # A peer fix supersedes any scheduled download of the same file:
        # the fix doc was computed on the remote content and is written
        # locally together with the merges, then uploaded.
        if peer_fixes:
            downloads = [(p, r) for p, r in downloads if p not in peer_fixes]
        for path in sorted(peer_fixes):
            if all(existing != path for existing, _ in uploads):
                uploads.append((path, "Link consistency for merged fiber"))

        resolved_actions = []
        for path, reason in manual_conflicts:
            action = self.resolve_conflict(path, reason, local_files.get(path),
                                           s3_files.get(path))
            if action != SyncAction.SKIP:
                resolved_actions.append((path, action))
        resolved_download_paths = {p for p, a in resolved_actions
                                   if a == SyncAction.DOWNLOAD}

        merged_count = len(pending_merges)
        if merged_count:
            print(f"\n✓ {merged_count} conflicting file(s) will be auto-merged "
                  f"and uploaded after confirmation")
        if peer_fixes:
            print(f"✓ {len(peer_fixes)} linked fiber file(s) will receive "
                  f"reciprocal link fixes and be uploaded")

        # Let the user decide what to do with suspect upload candidates
        invalid_uploads += self._validate_upload_candidates(
            [p for p, a in resolved_actions if a == SyncAction.UPLOAD])

        if invalid_uploads:
            print(f"\n⚠️  {len(invalid_uploads)} upload candidate(s) look invalid:")
            for path, problem in invalid_uploads:
                print(f"  {path}: {problem}")

            skip_paths = set()
            for path, problem in invalid_uploads:
                response = prompt_choice(
                    f"\n{path} ({problem}) — [u]pload anyway, [s]kip? ", ('u', 's'))
                if response != 'u':
                    skip_paths.add(path)

            if skip_paths:
                uploads = [(p, r) for p, r in uploads if p not in skip_paths]
                resolved_actions = [(p, a) for p, a in resolved_actions if p not in skip_paths]
                print(f"\nSkipping {len(skip_paths)} invalid file(s)")

        # Merge resolved conflicts into the main action lists so the summary
        # below reflects exactly what will run
        for path, action in resolved_actions:
            if action == SyncAction.UPLOAD:
                uploads.append((path, "Resolved conflict"))
            elif action == SyncAction.DOWNLOAD:
                downloads.append((path, "Resolved conflict"))
            elif action == SyncAction.DELETE_LOCAL:
                deletes_local.append((path, "Resolved conflict"))
            elif action == SyncAction.DELETE_REMOTE:
                deletes_remote.append((path, "Resolved conflict"))

        # The summary sits right next to the confirmation prompt: with a long
        # file list above, this is what makes the decision readable
        self._print_sync_summary(uploads, downloads, deletes_local, deletes_remote,
                                 "Conflicts skipped",
                                 len(conflicts) - len(resolved_actions) - merged_count)

        total_operations = (len(uploads) + len(downloads) +
                            len(deletes_local) + len(deletes_remote))

        print(f"\n{total_operations} operations will be performed.")
        if not confirm("Continue? [y/N]: "):
            self._discard_pending_merges(pending_merges)
            print("Sync cancelled. No local files were modified (conflict "
                  "copies and verified tracking backfills from the analysis "
                  "are kept).")
            return

        # Perform operations
        print("\nSyncing...")
        success_count = 0

        # Confirmed: swap the planned merge results into place (stashing the
        # pre-merge local and remote versions first), immediately followed
        # by their peer link fixes — the pair must land together BEFORE any
        # network transfer, or an interrupt leaves a cross-file
        # inconsistency that no future sync heals.
        self._apply_pending_merges(pending_merges)
        self._apply_peer_fixes(peer_fixes)

        # Last line of defense before downloads overwrite local files: stash
        # any local copy whose content diverges from its tracked state (or
        # was never tracked). Interactive conflict resolutions stashed at
        # prompt time already.
        self._stash_divergent_download_targets(
            [p for p, _ in downloads if p not in resolved_download_paths])

        if self.use_rclone:
            success_count += self.perform_downloads_batch([p for p, _ in downloads], s3_files)
            success_count += self.perform_uploads_batch([p for p, _ in uploads], include_backups)
        else:
            for path, reason in downloads:
                try:
                    self.perform_download(path, s3_files)
                    success_count += 1
                except Exception as e:
                    print(f"  ❌ Failed to download {path}: {e}")

            for path, reason in uploads:
                try:
                    self.perform_upload(path, local_files)
                    success_count += 1
                except Exception as e:
                    print(f"  ❌ Failed to upload {path}: {e}")

        # Process local deletions (no network involved, always per-file)
        for path, reason in deletes_local:
            try:
                self.perform_delete_local(path)
                success_count += 1
            except Exception as e:
                print(f"  ❌ Failed to delete local {path}: {e}")

        # Process remote deletions
        if self.use_rclone:
            success_count += self.perform_deletes_remote_batch([p for p, _ in deletes_remote], include_backups)
        else:
            for path, reason in deletes_remote:
                try:
                    self.perform_delete_remote(path)
                    success_count += 1
                except Exception as e:
                    print(f"  ❌ Failed to delete remote {path}: {e}")

        print(f"\n✓ Sync complete: {success_count}/{total_operations} operations successful")

    def show_status(self, verbose=False, include_backups=False):
        """Show sync status"""
        print(f"S3 Sync Status")
        print(f"Local directory: {self.local_dir}")
        print(f"S3 location: s3://{self.s3_bucket}/{self.s3_prefix}/")

        if self.aws_profile:
            print(f"AWS Profile: {self.aws_profile}")

        if self.use_rclone:
            print("Transfer backend: rclone (parallel transfers)")
        else:
            print("Transfer backend: aws CLI (serial)")
            print(f"⚠️  rclone unavailable, syncs will be slower: {self.rclone_unavailable_reason}")

        if not include_backups:
            print("Note: Ignoring backups/ directories (use --sync-backups to include them)")

        # Get database stats
        with self._get_db() as conn:
            cursor = conn.execute('SELECT COUNT(*) as count FROM files')
            tracked_count = cursor.fetchone()['count']
            print(f"Tracked files: {tracked_count}")

        print("\nAnalyzing changes...")

        local_files = self.scan_local_files(include_backups)
        s3_files = self.scan_s3_files(include_backups)
        actions = self.analyze_changes(local_files, s3_files)

        # Count actions
        action_counts = {}
        for path, (action, reason) in actions.items():
            action_counts[action] = action_counts.get(action, 0) + 1

        print(f"\nSummary:")
        print(f"  Files to upload:     {action_counts.get(SyncAction.UPLOAD, 0)}")
        print(f"  Files to download:   {action_counts.get(SyncAction.DOWNLOAD, 0)}")
        print(f"  Files to delete (S3): {action_counts.get(SyncAction.DELETE_REMOTE, 0)}")
        print(f"  Files to delete (local): {action_counts.get(SyncAction.DELETE_LOCAL, 0)}")
        print(f"  Conflicts:           {action_counts.get(SyncAction.CONFLICT, 0)}")
        if action_counts.get(SyncAction.CONFLICT) and fiber_merge is not None:
            # Cheap offline check: a conflict is a merge candidate when it is
            # a hashed annotation file with a local merge base. Whether the
            # merge actually succeeds is only known at sync time.
            candidates = sum(
                1 for path, (action, _) in actions.items()
                if action == SyncAction.CONFLICT and
                local_files.get(path, {}).get('local_md5') and
                os.path.exists(self._shadow_path(path)))
            print(f"    of which auto-merge candidates: {candidates}")
        print(f"  In sync:             {action_counts.get(SyncAction.SKIP, 0)}")

        if verbose:
            # Show detailed file list
            for action in [SyncAction.UPLOAD, SyncAction.DOWNLOAD, SyncAction.DELETE_REMOTE,
                           SyncAction.DELETE_LOCAL, SyncAction.CONFLICT]:
                files = [(p, r) for p, (a, r) in actions.items() if a == action]
                if files:
                    print(f"\n{action.value.replace('_', ' ').title()} ({len(files)} files):")
                    for path, reason in sorted(files):
                        print(f"  {path}: {reason}")


HFSYNC_CONFIG_NAME = '.hfsync.json'


def load_hfsync_config(local_dir):
    """Load the per-directory Hugging Face sync opt-in config, or None if absent"""
    config_path = os.path.join(local_dir, HFSYNC_CONFIG_NAME)
    if not os.path.exists(config_path):
        return None

    with open(config_path, 'r') as f:
        config = json.load(f)

    bucket_path = config.get('hf_bucket_path', '')
    if not bucket_path.startswith('hf://buckets/'):
        raise ValueError(f"{config_path}: 'hf_bucket_path' must start with hf://buckets/")
    config['hf_bucket_path'] = bucket_path.rstrip('/')

    hf_cli = shutil.which(config.get('hf_cli') or 'hf')
    if not hf_cli:
        raise ValueError(
            f"{config_path}: hf CLI not found "
            f"(install huggingface_hub>=1.0 or set 'hf_cli' to the binary's path)")
    config['hf_cli'] = hf_cli

    config.setdefault('tag', 'reviewed')
    return config


def classify_fibers(local_dir, tag):
    """Split the directory's fiber JSONs into tagged / untagged / invalid"""
    tagged, untagged, invalid = [], [], []

    for name in sorted(os.listdir(local_dir)):
        if name.startswith('.') or not name.endswith('.json'):
            continue
        path = os.path.join(local_dir, name)
        if not os.path.isfile(path):
            continue

        try:
            if os.path.getsize(path) == 0:
                invalid.append((name, "zero-byte file"))
                continue
            with open(path, 'r') as f:
                tags = json.load(f).get('tags', [])
        except json.JSONDecodeError as e:
            invalid.append((name, f"unparseable JSON ({e.msg} at line {e.lineno})"))
            continue
        except (OSError, UnicodeDecodeError) as e:
            invalid.append((name, f"unreadable ({e})"))
            continue

        if tag in tags:
            tagged.append(name)
        else:
            untagged.append(name)

    return tagged, untagged, invalid


def hf_sync(local_dir, dry_run=False):
    """Sync tagged fibers to the Hugging Face bucket configured in .hfsync.json"""
    local_dir = os.path.abspath(local_dir)

    config = load_hfsync_config(local_dir)
    if config is None:
        print(f"Hugging Face sync is not configured for {local_dir}")
        print(f"To enable it, create {os.path.join(local_dir, HFSYNC_CONFIG_NAME)}:")
        print('  {')
        print('    "hf_bucket_path": "hf://buckets/<org>/<bucket>/<path>",')
        print('    "hf_cli": "/path/to/hf",          (optional, defaults to hf on PATH)')
        print('    "tag": "reviewed"                 (optional)')
        print('  }')
        return

    hf_cli = config['hf_cli']
    bucket_path = config['hf_bucket_path']
    tag = config['tag']

    print(f"Hugging Face sync: {local_dir} → {bucket_path}")
    if dry_run:
        print("--dry-run mode: No changes will be made")

    tagged, untagged, invalid = classify_fibers(local_dir, tag)
    print(f"\nLocal fibers: {len(tagged)} tagged '{tag}', "
          f"{len(untagged)} without the tag, {len(invalid)} invalid")
    for name, problem in invalid:
        print(f"  ⚠️  Skipping {name}: {problem}")

    # Upload (additive): stage tagged files with mtimes preserved so
    # `hf buckets sync` transfers only new or changed ones
    if tagged:
        staging = tempfile.mkdtemp(prefix='hfsync-')
        try:
            for name in tagged:
                shutil.copy2(os.path.join(local_dir, name), staging)

            cmd = [hf_cli, 'buckets', 'sync', staging, bucket_path]
            if dry_run:
                cmd.append('--dry-run')
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print("❌ hf buckets sync failed; aborting before removals")
                return
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    # Removals: only filenames that exist locally WITHOUT the tag and are
    # present remotely. Files that exist only remotely are never touched.
    result = subprocess.run([hf_cli, 'buckets', 'list', bucket_path, '-R', '-q'],
                            capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Could not list {bucket_path}: {(result.stderr or '').strip()}")
        return
    remote_names = {line.rsplit('/', 1)[-1]
                    for line in result.stdout.splitlines() if line.strip()}

    to_remove = [name for name in untagged if name in remote_names]
    removed = 0
    for name in to_remove:
        if dry_run:
            print(f"  Would remove from HF (no longer tagged '{tag}'): {name}")
            continue
        rm = subprocess.run([hf_cli, 'buckets', 'rm', f"{bucket_path}/{name}", '--yes'],
                            capture_output=True, text=True)
        if rm.returncode == 0:
            print(f"  ✓ Removed from HF (no longer tagged '{tag}'): {name}")
            removed += 1
        else:
            print(f"  ❌ Failed to remove {name}: {(rm.stderr or '').strip()}")

    if dry_run:
        print(f"\n--dry-run: {len(tagged)} upload candidates (unchanged files "
              f"are skipped, see plan above), {len(to_remove)} would be removed")
    else:
        print(f"\n✓ Hugging Face sync complete: {len(tagged)} tagged fibers synced, "
              f"{removed} removed, {len(invalid)} skipped as invalid")


def main():
    parser = argparse.ArgumentParser(description='AWS S3 interactive sync with conflict resolution')
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Init command
    init_parser = subparsers.add_parser('init', help='Initialize sync configuration')
    init_parser.add_argument('directory', help='Local directory to sync')
    init_parser.add_argument('s3_bucket', help='S3 bucket name')
    init_parser.add_argument('s3_prefix', help='S3 prefix (path within bucket)')
    init_parser.add_argument('--profile', help='AWS profile to use')

    # Status command
    status_parser = subparsers.add_parser('status', help='Show sync status')
    status_parser.add_argument('directory', help='Local directory')
    status_parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed file list')
    status_parser.add_argument('--sync-backups', action='store_true', help='Include backups/ directories in sync')

    # Sync command
    sync_parser = subparsers.add_parser('sync', help='Perform interactive sync')
    sync_parser.add_argument('directory', help='Local directory')
    sync_parser.add_argument('--dry-run', action='store_true', help='Show what would be synced without doing it')
    sync_parser.add_argument('--sync-backups', action='store_true', help='Include backups/ directories in sync')
    sync_parser.add_argument('--no-auto-merge', action='store_true',
                             help='Disable the fiber-aware three-way merge for '
                                  'conflicting annotation files')

    # Update command
    update_parser = subparsers.add_parser(
        'update',
        help='Refresh file tracking without hiding pending changes '
             '(records in-sync files, prunes stale entries)')
    update_parser.add_argument('directory', help='Local directory')
    update_parser.add_argument('--sync-backups', action='store_true', help='Include backups/ directories in tracking')

    # Reset command
    reset_parser = subparsers.add_parser(
        'reset',
        help='Reset sync tracking (marks ALL files as synced, discarding pending differences)')
    reset_parser.add_argument('directory', help='Local directory')
    reset_parser.add_argument('--sync-backups', action='store_true', help='Include backups/ directories in reset')

    # Hugging Face sync command
    hfsync_parser = subparsers.add_parser(
        'hfsync', help='Sync tagged fibers to a Hugging Face bucket (requires .hfsync.json)')
    hfsync_parser.add_argument('directory', help='Local directory')
    hfsync_parser.add_argument('--dry-run', action='store_true',
                               help='Show what would be synced without doing it')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == 'init':
        # Initialize new sync configuration. No transfers happen here: files
        # identical on both sides are recorded as in sync, and everything else
        # (S3-only, local-only, size mismatches) is left for 'sync' to resolve.
        manager = S3SyncManager(args.directory, args.s3_bucket, args.s3_prefix, args.profile)
        print(f"Initialized sync configuration in {args.directory}")
        print(f"S3 location: s3://{args.s3_bucket}/{args.s3_prefix}/")

        print("\nScanning initial state...")
        local_files = manager.scan_local_files(include_backups=False)
        s3_files = manager.scan_s3_files(include_backups=False)

        manager._record_untracked_synced(local_files, s3_files)

        s3_only = sum(1 for path in s3_files if path not in local_files)
        local_only = sum(1 for path in local_files if path not in s3_files)
        if s3_only:
            print(f"{s3_only} file(s) exist only on S3 and can be downloaded with 'sync'")
        if local_only:
            print(f"{local_only} file(s) exist only locally and can be uploaded with 'sync'")

        print("\n✓ Initialization complete!")
        print("Use 'sync' to transfer differences and 'status' to see current sync state")

    elif args.command == 'hfsync':
        # Independent of the S3 sync configuration; gated only on .hfsync.json
        hf_sync(args.directory, args.dry_run)

    else:
        # Check for existing configuration
        config_file = os.path.join(args.directory, '.s3sync.json')

        if not os.path.exists(config_file):
            print(f"Error: No sync configuration found in {args.directory}")
            print("Run 'init' command first to set up sync configuration")
            sys.exit(1)

        manager = S3SyncManager(args.directory)

        try:
            if args.command == 'status':
                manager.show_status(args.verbose, args.sync_backups)

            elif args.command == 'sync':
                manager.sync(args.dry_run, args.sync_backups,
                             auto_merge=not args.no_auto_merge)

            elif args.command == 'update':
                manager.refresh_tracking(args.sync_backups)

            elif args.command == 'reset':
                print("Resetting sync tracking...")
                print("This will mark ALL current files as synced, discarding any pending differences.")
                print("Files that currently differ between local and S3 will also lose their "
                      "merge base until the next successful sync.")
                if not args.sync_backups:
                    print("Note: Excluding backups/ directories (use --sync-backups to include them)")

                if confirm("Continue? [y/N]: "):
                    manager.reset_tracking(args.sync_backups)
                    print("✓ Sync tracking reset. All files marked as in sync.")
                else:
                    print("Reset cancelled.")
        finally:
            # Fetched remote copies / pending merge artifacts may hold
            # annotation content; never leave them in the system tmp dir.
            manager._cleanup_run_tmp()


if __name__ == "__main__":
    main()