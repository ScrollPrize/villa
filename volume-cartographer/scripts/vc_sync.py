#!/usr/bin/env python3
"""
AWS S3 Interactive Sync Tool with Conflict Resolution

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
"""

import os
import sys
import json
import csv
import tempfile
import sqlite3
import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path
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

# Report configuration
REPORT_S3_BUCKET = "philodemos"
REPORT_S3_PREFIX = "david/reports"


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

        while True:
            cmd = [
                'aws', 's3api', 'list-objects-v2',
                '--bucket', self.s3_bucket,
                '--prefix', self.s3_prefix
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

            prefix_len = len(self.s3_prefix) + 1 if self.s3_prefix else 0

            for obj in data['Contents']:
                # Skip if it's just the prefix itself
                if obj['Key'] == self.s3_prefix + '/':
                    continue

                relative_path = obj['Key'][prefix_len:]

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

    def _get_report_csv_path(self):
        """Return the S3 URL and filename for the report based on the volume package name"""
        pkg_name = Path(self.local_dir).resolve().parent.name
        if pkg_name.endswith(".volpkg"):
            pkg_name = pkg_name[:-7]
        filename = f"{pkg_name}.csv"
        return f"s3://{REPORT_S3_BUCKET}/{REPORT_S3_PREFIX}/{filename}", filename

    def _collect_segment_areas(self):
        """Collect segment names and area_cm2 values from meta.json files"""
        segments = {}
        for root, dirs, filenames in os.walk(self.local_dir):
            dirs[:] = [d for d in dirs if not d.startswith('.') and 'layers' not in d.lower() and d != 'backups']
            if 'meta.json' in filenames:
                meta_path = os.path.join(root, 'meta.json')
                try:
                    with open(meta_path, 'r') as f:
                        meta_data = json.load(f)
                    segment_name = meta_data.get('uuid', os.path.basename(root))
                    area_cm2 = meta_data.get('area_cm2', 0.0)
                    segments[segment_name] = area_cm2
                except (json.JSONDecodeError, OSError) as e:
                    print(f"  Warning: Could not read {meta_path}: {e}")
                    continue
        return segments

    def generate_segment_report(self):
        """Generate or update the segment area CSV report and upload to S3"""
        print("\nGenerating report...")
        s3_csv_path, report_filename = self._get_report_csv_path()
        current_datetime = datetime.now().strftime("%Y%m%d%H%M%S")

        current_segments = self._collect_segment_areas()
        if not current_segments:
            print("  No segments found with meta.json files")
            return

        print(f"  Found {len(current_segments)} segments")

        existing_data = {}
        existing_datetimes = []

        # Try to download existing CSV
        download_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        download_tmp_path = download_tmp.name
        download_tmp.close()

        try:
            cmd = ['aws', 's3', 'cp', s3_csv_path, download_tmp_path]
            if self.aws_profile:
                cmd.extend(['--profile', self.aws_profile])
            subprocess.run(cmd, capture_output=True, text=True, check=True)

            with open(download_tmp_path, 'r', newline='') as f:
                reader = csv.reader(f)
                rows = list(reader)

            if rows:
                existing_datetimes = rows[0][1:] if len(rows[0]) > 1 else []
                for row in rows[1:]:
                    if not row:
                        continue
                    segment_name = row[0]
                    values = [float(val) if val else 0.0 for val in row[1:]]
                    # Pad to header length if necessary
                    if len(values) < len(existing_datetimes):
                        values.extend([0.0] * (len(existing_datetimes) - len(values)))
                    existing_data[segment_name] = values

                print(f"  Loaded existing report ({report_filename})")
        except subprocess.CalledProcessError:
            print(f"  No existing report found at {s3_csv_path}, creating new one")
        except Exception as e:
            print(f"  Warning: Could not read existing report: {e}")
        finally:
            try:
                os.unlink(download_tmp_path)
            except OSError:
                pass

        # Only include segments that currently exist (remove deleted segments)
        all_segments = set(current_segments.keys())
        header_row = ["Segment Name"] + existing_datetimes + [current_datetime]
        csv_rows = [header_row]

        for segment_name in sorted(all_segments):
            row = [segment_name]
            previous_values = existing_data.get(segment_name, [0.0] * len(existing_datetimes))
            if len(previous_values) < len(existing_datetimes):
                previous_values.extend([0.0] * (len(existing_datetimes) - len(previous_values)))
            row.extend(previous_values)
            row.append(current_segments.get(segment_name, 0.0))
            csv_rows.append(row)

        # Write new CSV
        upload_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        upload_tmp_path = upload_tmp.name
        try:
            with open(upload_tmp_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(csv_rows)

            cmd = ['aws', 's3', 'cp', upload_tmp_path, s3_csv_path]
            if self.aws_profile:
                cmd.extend(['--profile', self.aws_profile])
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"  ✓ Report uploaded")
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Failed to upload report: {e}")
        finally:
            try:
                os.unlink(upload_tmp_path)
            except OSError:
                pass

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

        print(f"  Uploading: {path} → remote")

        cmd = ['aws', 's3', 'cp', local_path, s3_path]
        self._run_aws_command(cmd)

        print(f"  ✓ Uploaded: {path}")

        # Get fresh S3 info
        cmd = ['aws', 's3api', 'head-object', '--bucket', self.s3_bucket,
               '--key', f"{self.s3_prefix}/{path}"]
        result = self._run_aws_command(cmd)

        data = json.loads(result.stdout)
        s3_mtime = self._parse_timestamp(data['LastModified'])
        s3_etag = data.get('ETag', '').strip('"')

        # Update database with actual local file info
        with self._get_db() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO files 
                (path, local_size, local_mtime, s3_size, s3_mtime, s3_etag)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                path,
                local_files[path]['local_size'],
                local_files[path]['local_mtime'],
                local_files[path]['local_size'],
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

    def sync(self, dry_run=False, include_backups=False):
        """Perform interactive sync operation"""
        if not include_backups:
            print("Note: Ignoring backups/ directories (use --sync-backups to include them)")

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

        if dry_run:
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

        # Confirm before proceeding
        total_operations = (len(uploads) + len(downloads) + len(deletes_local) +
                            len(deletes_remote) + len(resolved_actions))

        print(f"\n{total_operations} operations will be performed.")
        response = input("Continue? [y/N]: ").strip().lower()

        if response != 'y':
            print("Sync cancelled.")
            return

        # Perform operations
        print("\nSyncing...")
        success_count = 0

        # Process uploads
        for path, reason in uploads:
            try:
                self.perform_upload(path, local_files)
                success_count += 1
            except Exception as e:
                print(f"  ❌ Failed to upload {path}: {e}")

        # Process downloads
        for path, reason in downloads:
            try:
                self.perform_download(path, s3_files)
                success_count += 1
            except Exception as e:
                print(f"  ❌ Failed to download {path}: {e}")

        # Process deletions
        for path, reason in deletes_local:
            try:
                self.perform_delete_local(path)
                success_count += 1
            except Exception as e:
                print(f"  ❌ Failed to delete local {path}: {e}")

        for path, reason in deletes_remote:
            try:
                self.perform_delete_remote(path)
                success_count += 1
            except Exception as e:
                print(f"  ❌ Failed to delete remote {path}: {e}")

        # Process resolved conflicts
        for path, action in resolved_actions:
            try:
                if action == SyncAction.UPLOAD:
                    self.perform_upload(path, local_files)
                elif action == SyncAction.DOWNLOAD:
                    self.perform_download(path, s3_files)
                elif action == SyncAction.DELETE_LOCAL:
                    self.perform_delete_local(path)
                elif action == SyncAction.DELETE_REMOTE:
                    self.perform_delete_remote(path)
                success_count += 1
            except Exception as e:
                print(f"  ❌ Failed to process {path}: {e}")

        print(f"\n✓ Sync complete: {success_count}/{total_operations} operations successful")

    def show_status(self, verbose=False, include_backups=False):
        """Show sync status"""
        print(f"S3 Sync Status")
        print(f"Local directory: {self.local_dir}")
        print(f"S3 location: s3://{self.s3_bucket}/{self.s3_prefix}/")

        if self.aws_profile:
            print(f"AWS Profile: {self.aws_profile}")

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
            if not args.dry_run:
                manager.generate_segment_report()

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