#!/usr/bin/env python3
"""
AWS S3 Interactive Sync Tool with Conflict Resolution

Automatically ignores:
- Hidden files and directories (starting with .)
- Any directory containing 'layers' in its name (e.g., layers/, layers_fullres/, old_layers/)
- The .s3sync.json configuration file and .s3sync.db database

Usage:
    python s3_sync.py init <directory> <s3_bucket> <s3_prefix> [--profile=<aws_profile>]
    python s3_sync.py status <directory> [--verbose]
    python s3_sync.py sync <directory> [--dry-run]
    python s3_sync.py update <directory>
"""

import os
import sys
import json
import sqlite3
import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from enum import Enum
from contextlib import contextmanager


class SyncAction(Enum):
    UPLOAD = "upload"
    DOWNLOAD = "download"
    CONFLICT = "conflict"
    SKIP = "skip"
    DELETE_LOCAL = "delete_local"
    DELETE_REMOTE = "delete_remote"


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
        """Run AWS CLI command with optional profile"""
        if self.aws_profile:
            cmd.extend(['--profile', self.aws_profile])
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result

    def _get_s3_url(self, relative_path=None):
        """Get S3 URL for a file or directory"""
        if relative_path:
            return f"s3://{self.s3_bucket}/{self.s3_prefix}/{relative_path}"
        return f"s3://{self.s3_bucket}/{self.s3_prefix}/"

    def _parse_timestamp(self, timestamp_str):
        """Parse AWS timestamp to Unix timestamp"""
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return dt.timestamp()

    def scan_local_files(self):
        """Scan local directory for files"""
        print(f"Scanning local directory: {self.local_dir}")
        files = {}

        for root, dirs, filenames in os.walk(self.local_dir):
            # Skip hidden directories and directories containing 'layers'
            dirs[:] = [d for d in dirs if not d.startswith('.') and 'layers' not in d.lower()]

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

                stat = os.stat(filepath)
                files[relative_path] = {
                    'path': relative_path,
                    'local_size': stat.st_size,
                    'local_mtime': stat.st_mtime
                }

        print(f"Found {len(files)} local files")
        return files

    def scan_s3_files(self):
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

                files[relative_path] = {
                    'path': relative_path,
                    's3_size': obj['Size'],
                    's3_mtime': self._parse_timestamp(obj['LastModified']),
                    's3_etag': obj.get('ETag', '').strip('"')
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

    def update_files(self):
        """Update file tracking with current state"""
        print("\nUpdating file tracking...")

        local_files = self.scan_local_files()
        s3_files = self.scan_s3_files()

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

        # Update database
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

        # Update database
        with self._get_db() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO files 
                (path, local_size, local_mtime, s3_size, s3_mtime, s3_etag)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                path,
                s3_files[path]['s3_size'],
                datetime.now().timestamp(),
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

    def sync(self, dry_run=False):
        """Perform interactive sync operation"""
        print("\nAnalyzing changes...")

        local_files = self.scan_local_files()
        s3_files = self.scan_s3_files()

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
            self.perform_upload(path, local_files)
            success_count += 1

        # Process downloads
        for path, reason in downloads:
            self.perform_download(path, s3_files)
            success_count += 1

        # Process deletions
        for path, reason in deletes_local:
            self.perform_delete_local(path)
            success_count += 1

        for path, reason in deletes_remote:
            self.perform_delete_remote(path)
            success_count += 1

        # Process resolved conflicts
        for path, action in resolved_actions:
            if action == SyncAction.UPLOAD:
                self.perform_upload(path, local_files)
            elif action == SyncAction.DOWNLOAD:
                self.perform_download(path, s3_files)
            elif action == SyncAction.DELETE_LOCAL:
                self.perform_delete_local(path)
            elif action == SyncAction.DELETE_REMOTE:
                self.perform_delete_remote(path)
            success_count += 1

        print(f"\n✓ Sync complete: {success_count}/{total_operations} operations successful")

    def show_status(self, verbose=False):
        """Show sync status"""
        print(f"S3 Sync Status")
        print(f"Local directory: {self.local_dir}")
        print(f"S3 location: s3://{self.s3_bucket}/{self.s3_prefix}/")

        if self.aws_profile:
            print(f"AWS Profile: {self.aws_profile}")

        # Get database stats
        with self._get_db() as conn:
            cursor = conn.execute('SELECT COUNT(*) as count FROM files')
            tracked_count = cursor.fetchone()['count']
            print(f"Tracked files: {tracked_count}")

        print("\nAnalyzing changes...")

        local_files = self.scan_local_files()
        s3_files = self.scan_s3_files()
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

    # Sync command
    sync_parser = subparsers.add_parser('sync', help='Perform interactive sync')
    sync_parser.add_argument('directory', help='Local directory')
    sync_parser.add_argument('--dry-run', action='store_true', help='Show what would be synced without doing it')

    # Update command
    update_parser = subparsers.add_parser('update', help='Update file tracking with current state')
    update_parser.add_argument('directory', help='Local directory')

    # Reset command
    reset_parser = subparsers.add_parser('reset', help='Reset sync tracking (mark all as synced)')
    reset_parser.add_argument('directory', help='Local directory')

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
        local_files = manager.scan_local_files()
        s3_files = manager.scan_s3_files()

        # Find files that exist in S3 but not locally
        files_to_download = []
        for path in s3_files:
            if path not in local_files:
                files_to_download.append(path)

        if files_to_download:
            print(f"\nFound {len(files_to_download)} files in S3 that don't exist locally.")
            response = input("Download all files? [y/N]: ").strip().lower()

            if response == 'y':
                print(f"\nDownloading {len(files_to_download)} files...")
                success_count = 0

                for i, path in enumerate(files_to_download, 1):
                    if i % 100 == 0:
                        print(f"  Progress: {i}/{len(files_to_download)} files...")

                    local_path = os.path.join(args.directory, path)
                    s3_path = manager._get_s3_url(path)

                    # Create directory if needed
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)

                    cmd = ['aws', 's3', 'cp', s3_path, local_path]
                    if args.profile:
                        cmd.extend(['--profile', args.profile])

                    subprocess.run(cmd, capture_output=True, text=True, check=True)
                    success_count += 1

                print(f"\n✓ Downloaded {success_count} files successfully")
        else:
            print("✓ All S3 files already exist locally")

        # Do initial tracking update after downloads
        manager.update_files()

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
            manager.show_status(args.verbose)

        elif args.command == 'sync':
            manager.sync(args.dry_run)

        elif args.command == 'update':
            manager.update_files()

        elif args.command == 'reset':
            print("Resetting sync tracking...")
            print("This will mark all current files as synced.")
            response = input("Continue? [y/N]: ").strip().lower()

            if response == 'y':
                manager.update_files()
                print("✓ Sync tracking reset. All files marked as in sync.")
            else:
                print("Reset cancelled.")


if __name__ == "__main__":
    main()
