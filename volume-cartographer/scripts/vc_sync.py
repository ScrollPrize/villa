#!/usr/bin/env python3
"""
AWS S3 Interactive Sync Tool with Conflict Resolution

Transfers run through rclone (parallel, one process per batch) when the rclone
binary is available, and fall back to serial per-file aws CLI calls otherwise.
Neither path needs an rclone config file: credentials come from the standard
AWS credential chain (env vars pasted into the terminal, ~/.aws/credentials,
or EC2 instance roles).

Automatically ignores:
- Hidden files and directories (starting with .)
- Any directory containing 'layers' in its name (e.g., layers/, layers_fullres/, old_layers/)
- The .s3sync.json configuration file and .s3sync.db database
- Files matching backup patterns (see BACKUP_PATTERNS)
- Directories named 'backups' (unless --sync-backups is specified)

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
import shutil
import sqlite3
import argparse
import tempfile
import subprocess
from datetime import datetime
from enum import Enum
from contextlib import contextmanager


# Backup file patterns - these files are only uploaded, never downloaded or deleted
# Note: This is separate from the backups/ directory filter which is controlled by --sync-backups
BACKUP_PATTERNS = [
    '_backup',
    '.backup',
    '_bak',
    '.bak',
]

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

        # Create index for faster lookups
        conn.execute('CREATE INDEX IF NOT EXISTS idx_path ON files(path)')
        conn.commit()
        conn.close()

    @contextmanager
    def _get_db(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_file)
        conn.row_factory = sqlite3.Row
        yield conn
        conn.commit()
        conn.close()

    def _run_aws_command(self, cmd):
        """Run AWS CLI command with optional profile and better error handling"""
        if self.aws_profile:
            cmd.extend(['--profile', self.aws_profile])

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

    def scan_local_files(self, include_backups=False):
        """Scan local directory for files"""
        print(f"Scanning local directory: {self.local_dir}")
        files = {}

        for root, dirs, filenames in os.walk(self.local_dir):
            # Skip hidden directories, directories containing 'layers', and backups (unless requested)
            dirs[:] = [d for d in dirs if not d.startswith('.') and
                       'layers' not in d.lower() and
                       (include_backups or d != 'backups')]

            for filename in filenames:
                # Skip hidden files, sync config, and database
                if filename.startswith('.') or filename in ['.s3sync.json', '.s3sync.db'] or filename.endswith('.obj'):
                    continue

                filepath = os.path.join(root, filename)
                relative_path = os.path.relpath(filepath, self.local_dir)

                # Skip files in directories containing 'layers'
                path_parts = relative_path.split(os.sep)
                if any('layers' in part.lower() for part in path_parts[:-1]):
                    continue

                # Skip files in backups directories unless explicitly requested
                if not include_backups and 'backups' in path_parts[:-1]:
                    continue

                stat = os.stat(filepath)
                files[relative_path] = {
                    'path': relative_path,
                    'local_size': stat.st_size,
                    'local_mtime': stat.st_mtime,
                    'is_backup': is_backup_file(filename)
                }

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

                # Skip hidden files
                filename = os.path.basename(relative_path)
                if filename.startswith('.') or filename.endswith('.obj'):
                    continue

                # Skip files in hidden directories or directories containing 'layers'
                path_parts = relative_path.split('/')
                if any(part.startswith('.') for part in path_parts[:-1]):
                    continue
                if any('layers' in part.lower() for part in path_parts[:-1]):
                    continue

                # Skip backups directories unless explicitly requested
                if not include_backups and 'backups' in path_parts[:-1]:
                    continue

                files[relative_path] = {
                    'path': relative_path,
                    's3_size': obj['Size'],
                    's3_mtime': self._parse_timestamp(obj['LastModified']),
                    's3_etag': obj.get('ETag', '').strip('"'),
                    'is_backup': is_backup_file(filename)
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

    def update_files(self, include_backups=False):
        """Update file tracking with current state"""
        print("\nUpdating file tracking...")

        local_files = self.scan_local_files(include_backups)
        s3_files = self.scan_s3_files(include_backups)

        with self._get_db() as conn:
            # Get all tracked paths
            cursor = conn.execute('SELECT path FROM files')
            tracked_paths = set(row['path'] for row in cursor)

            # Get all current paths
            current_paths = set(local_files.keys()) | set(s3_files.keys())

            # Update or insert files
            for path in current_paths:
                local_info = local_files.get(path)
                s3_info = s3_files.get(path)

                conn.execute('''
                    INSERT OR REPLACE INTO files 
                    (path, local_size, local_mtime, s3_size, s3_mtime, s3_etag)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    path,
                    local_info['local_size'] if local_info else None,
                    local_info['local_mtime'] if local_info else None,
                    s3_info['s3_size'] if s3_info else None,
                    s3_info['s3_mtime'] if s3_info else None,
                    s3_info.get('s3_etag') if s3_info else None
                ))

            # Remove files that no longer exist anywhere
            for path in tracked_paths - current_paths:
                conn.execute('DELETE FROM files WHERE path = ?', (path,))

        print("File tracking updated successfully")

    def analyze_changes(self, local_files, s3_files):
        """Analyze what needs to be synced and detect conflicts"""
        actions = {}

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
                if tracked_info:
                    # We have tracking history
                    local_changed = (tracked_info.get('local_size') != local_info['local_size'] or
                                     (tracked_info.get('local_mtime') and
                                      abs(tracked_info['local_mtime'] - local_info['local_mtime']) > 1))

                    s3_changed = (tracked_info.get('s3_size') != s3_info['s3_size'] or
                                  tracked_info.get('s3_etag') != s3_info['s3_etag'])

                    if local_changed and s3_changed:
                        actions[path] = (SyncAction.CONFLICT, "Both local and S3 modified since last sync")
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
                    else:
                        actions[path] = (SyncAction.SKIP, "Files appear to be in sync")

            # File deleted from both
            elif path in tracked_files and not local_info and not s3_info:
                actions[path] = (SyncAction.SKIP, "File deleted from both")

        return actions

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

            while True:
                response = input("\nChoose: [l]ocal → remote, [r]emote → local, [s]kip? ").strip().lower()
                if response == 'l':
                    return SyncAction.UPLOAD
                elif response == 'r':
                    return SyncAction.DOWNLOAD
                elif response == 's':
                    return SyncAction.SKIP
                else:
                    print("Invalid choice. Please enter 'l', 'r', or 's'.")

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
        s3_mtime = self._parse_timestamp(data['LastModified'])
        s3_etag = data.get('ETag', '').strip('"')

        # Track the pre-upload stats: if the file changed during upload, the next
        # scan will see a local difference and schedule a re-upload
        with self._get_db() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO files
                (path, local_size, local_mtime, s3_size, s3_mtime, s3_etag)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                path,
                pre_stat.st_size,
                pre_stat.st_mtime,
                pre_stat.st_size,
                s3_mtime,
                s3_etag
            ))

        return True

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

        # Get the actual mtime of the downloaded file
        stat = os.stat(local_path)
        actual_local_mtime = stat.st_mtime
        actual_local_size = stat.st_size

        # Update database with actual file stats
        with self._get_db() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO files 
                (path, local_size, local_mtime, s3_size, s3_mtime, s3_etag)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                path,
                actual_local_size,
                actual_local_mtime,  # Use actual mtime from filesystem
                s3_files[path]['s3_size'],
                s3_files[path]['s3_mtime'],
                s3_files[path].get('s3_etag')
            ))

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

        return True

    def _rclone_remote(self):
        """On-the-fly rclone remote for the S3 target (no rclone.conf needed).

        env_auth=true uses the standard AWS credential chain, so this works
        both with credentials pasted into the terminal (env vars) and with
        EC2 instance roles — same sources as the aws CLI.
        """
        return f":s3,provider=AWS,env_auth=true,no_check_bucket=true:{self.s3_bucket}/{self.s3_prefix}"

    def _rclone_env(self):
        """Environment for rclone subprocesses, mirroring aws CLI credential/region config"""
        env = os.environ.copy()
        if self.aws_profile:
            env['AWS_PROFILE'] = self.aws_profile
        region = env.get('AWS_REGION') or env.get('AWS_DEFAULT_REGION')
        if region:
            env.setdefault('RCLONE_S3_REGION', region)
        return env

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

        try:
            result = subprocess.run(cmd, env=self._rclone_env())
            return result.returncode == 0
        finally:
            try:
                os.unlink(list_path)
            except OSError:
                pass

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

        # One listing pass for fresh S3 metadata instead of a head-object per file
        fresh_s3 = self.scan_s3_files(include_backups)

        success_count = 0
        with self._get_db() as conn:
            for path, (size, mtime) in pre_stats.items():
                s3_info = fresh_s3.get(path)
                if not s3_info or s3_info['s3_size'] != size:
                    print(f"  ❌ Upload not verified for {path}; will retry on next sync")
                    continue
                conn.execute('''
                    INSERT OR REPLACE INTO files
                    (path, local_size, local_mtime, s3_size, s3_mtime, s3_etag)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (path, size, mtime,
                      s3_info['s3_size'], s3_info['s3_mtime'], s3_info.get('s3_etag')))
                success_count += 1

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
                conn.execute('''
                    INSERT OR REPLACE INTO files
                    (path, local_size, local_mtime, s3_size, s3_mtime, s3_etag)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (path, stat.st_size, stat.st_mtime,
                      s3_files[path]['s3_size'], s3_files[path]['s3_mtime'],
                      s3_files[path].get('s3_etag')))
                success_count += 1

        print(f"  ✓ Downloaded {success_count}/{len(paths)} files")
        return success_count

    def perform_deletes_remote_batch(self, paths):
        """Delete files from S3 in one rclone run and update tracking"""
        if not paths:
            return 0

        print(f"  Deleting {len(paths)} files from S3...")
        ok = self._run_rclone(['delete', self._rclone_remote()], paths)
        if not ok:
            print("  ❌ rclone reported errors during deletion; tracking left unchanged, will retry on next sync")
            return 0

        with self._get_db() as conn:
            for path in paths:
                conn.execute('DELETE FROM files WHERE path = ?', (path,))

        print(f"  ✓ Deleted {len(paths)} files from S3")
        return len(paths)

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

    def sync(self, dry_run=False, include_backups=False):
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

        actions = self.analyze_changes(local_files, s3_files)

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

        # Summary
        print(f"\nSync Summary:")
        print(f"  Uploads pending:    {len(uploads)}")
        print(f"  Downloads pending:  {len(downloads)}")
        print(f"  Local deletions:    {len(deletes_local)}")
        print(f"  Remote deletions:   {len(deletes_remote)}")
        print(f"  Conflicts:          {len(conflicts)}")

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
            print("\n--dry-run mode: No changes will be made")
            return

        # Process conflicts first
        resolved_actions = []
        for path, reason in conflicts:
            local_info = local_files.get(path)
            s3_info = s3_files.get(path)

            action = self.resolve_conflict(path, reason, local_info, s3_info)
            if action != SyncAction.SKIP:
                resolved_actions.append((path, action))

        # Let the user decide what to do with suspect upload candidates
        invalid_uploads += self._validate_upload_candidates(
            [p for p, a in resolved_actions if a == SyncAction.UPLOAD])

        if invalid_uploads:
            print(f"\n⚠️  {len(invalid_uploads)} upload candidate(s) look invalid:")
            for path, problem in invalid_uploads:
                print(f"  {path}: {problem}")

            skip_paths = set()
            for path, problem in invalid_uploads:
                while True:
                    response = input(f"\n{path} ({problem}) — [u]pload anyway, [s]kip? ").strip().lower()
                    if response == 'u':
                        break
                    elif response == 's':
                        skip_paths.add(path)
                        break
                    else:
                        print("Invalid choice. Please enter 'u' or 's'.")

            if skip_paths:
                uploads = [(p, r) for p, r in uploads if p not in skip_paths]
                resolved_actions = [(p, a) for p, a in resolved_actions if p not in skip_paths]
                print(f"\nSkipping {len(skip_paths)} invalid file(s)")

        # Confirm before proceeding
        total_operations = (len(uploads) + len(downloads) + len(deletes_local) +
                            len(deletes_remote) + len(resolved_actions))

        print(f"\n{total_operations} operations will be performed.")
        response = input("Continue? [y/N]: ").strip().lower()

        if response != 'y':
            print("Sync cancelled.")
            return

        # Merge resolved conflicts into the main action lists
        for path, action in resolved_actions:
            if action == SyncAction.UPLOAD:
                uploads.append((path, "Resolved conflict"))
            elif action == SyncAction.DOWNLOAD:
                downloads.append((path, "Resolved conflict"))
            elif action == SyncAction.DELETE_LOCAL:
                deletes_local.append((path, "Resolved conflict"))
            elif action == SyncAction.DELETE_REMOTE:
                deletes_remote.append((path, "Resolved conflict"))

        # Perform operations
        print("\nSyncing...")
        success_count = 0

        # Process uploads and downloads
        if self.use_rclone:
            success_count += self.perform_uploads_batch([p for p, _ in uploads], include_backups)
            success_count += self.perform_downloads_batch([p for p, _ in downloads], s3_files)
        else:
            for path, reason in uploads:
                try:
                    self.perform_upload(path, local_files)
                    success_count += 1
                except Exception as e:
                    print(f"  ❌ Failed to upload {path}: {e}")

            for path, reason in downloads:
                try:
                    self.perform_download(path, s3_files)
                    success_count += 1
                except Exception as e:
                    print(f"  ❌ Failed to download {path}: {e}")

        # Process local deletions (no network involved, always per-file)
        for path, reason in deletes_local:
            try:
                self.perform_delete_local(path)
                success_count += 1
            except Exception as e:
                print(f"  ❌ Failed to delete local {path}: {e}")

        # Process remote deletions
        if self.use_rclone:
            success_count += self.perform_deletes_remote_batch([p for p, _ in deletes_remote])
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

    # Update command
    update_parser = subparsers.add_parser('update', help='Update file tracking with current state')
    update_parser.add_argument('directory', help='Local directory')
    update_parser.add_argument('--sync-backups', action='store_true', help='Include backups/ directories in tracking')

    # Reset command
    reset_parser = subparsers.add_parser('reset', help='Reset sync tracking (mark all as synced)')
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
        # Initialize new sync configuration
        manager = S3SyncManager(args.directory, args.s3_bucket, args.s3_prefix, args.profile)
        print(f"Initialized sync configuration in {args.directory}")
        print(f"S3 location: s3://{args.s3_bucket}/{args.s3_prefix}/")

        # Initial sync: download any S3 files that don't exist locally
        print("\nChecking for files to download from S3...")
        local_files = manager.scan_local_files(include_backups=False)  # Don't include backups by default
        s3_files = manager.scan_s3_files(include_backups=False)

        # Find files that exist in S3 but not locally
        files_to_download = []
        for path in s3_files:
            if path not in local_files:
                files_to_download.append(path)

        if files_to_download:
            print(f"\nFound {len(files_to_download)} files in S3 that don't exist locally.")
            print("Note: Excluding backups/ directories. Use --sync-backups if needed.")
            response = input("Download all files? [y/N]: ").strip().lower()

            if response == 'y':
                print(f"\nDownloading {len(files_to_download)} files using aws s3 sync (this is much faster)...")
                print(f"S3 location: s3://{args.s3_bucket}/{args.s3_prefix}/")
                print(f"Local directory: {args.directory}\n")

                # Use aws s3 sync for bulk download - much faster!
                cmd = [
                    'aws', 's3', 'sync',
                    f"s3://{args.s3_bucket}/{args.s3_prefix}/",
                    args.directory,
                    '--exclude', '.*',  # Exclude hidden files
                    '--exclude', '*.obj',  # Exclude .obj files
                ]

                # Add excludes for layers and backups directories
                cmd.extend(['--exclude', '*layers*/*'])
                cmd.extend(['--exclude', '*/backups/*'])

                if args.profile:
                    cmd.extend(['--profile', args.profile])

                # Run without capture_output so we see live progress
                try:
                    subprocess.run(cmd, check=True)
                    print(f"\n✓ Download complete!")
                except subprocess.CalledProcessError as e:
                    print(f"\n❌ Download failed with exit code {e.returncode}")
                    sys.exit(1)
        else:
            print("✓ All S3 files already exist locally")

        # Do initial tracking update after downloads (exclude backups by default)
        manager.update_files(include_backups=False)

        print("\n✓ Initialization complete!")
        print("Use 'status' command to see current sync state")

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

        if args.command == 'status':
            manager.show_status(args.verbose, getattr(args, 'sync_backups', False))

        elif args.command == 'sync':
            manager.sync(args.dry_run, getattr(args, 'sync_backups', False))

        elif args.command == 'update':
            manager.update_files(getattr(args, 'sync_backups', False))

        elif args.command == 'reset':
            print("Resetting sync tracking...")
            print("This will mark all current files as synced.")
            if not getattr(args, 'sync_backups', False):
                print("Note: Excluding backups/ directories (use --sync-backups to include them)")
            response = input("Continue? [y/N]: ").strip().lower()

            if response == 'y':
                manager.update_files(getattr(args, 'sync_backups', False))
                print("✓ Sync tracking reset. All files marked as in sync.")
            else:
                print("Reset cancelled.")


if __name__ == "__main__":
    main()