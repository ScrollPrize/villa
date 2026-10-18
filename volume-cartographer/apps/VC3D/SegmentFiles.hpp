#pragma once

#include <QString>

#include <filesystem>
#include <vector>

// Folder-level operations on tifxyz segments, shared by the segment context
// menu and the dialog-free segment operations.
namespace vc3d::segment_files {

// One rotating backup slot of a segment: <backups directory>/<index>.
struct Backup {
    int index{0};                  // 0..9
    std::filesystem::path path;
    QString timestamp;             // meta.json "backup_timestamp", if any
};

// <root>/backups/<segment folder name>, where root is `backupRoot` or, when
// that is empty, the segment's parent folder, matching
// QuadSurface::saveSnapshot().
std::filesystem::path backupsDirectory(const std::filesystem::path& segmentDir,
                                       const std::filesystem::path& backupRoot);

// The numbered slots 0..9 under `backupsDir`, sorted by index. Empty when the
// directory does not exist.
std::vector<Backup> listBackups(const std::filesystem::path& backupsDir);

}  // namespace vc3d::segment_files
