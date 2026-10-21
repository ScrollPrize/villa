#pragma once

#include <QString>

#include <filesystem>
#include <functional>
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

// Restores a backup slot over a segment folder: every regular file of
// `slotDir`, as QuadSurface::saveSnapshot() captured it, replaces the file of
// the same name in `segmentDir`; files the slot lacks are kept. The slot's
// files are first copied into a staging folder beside the segment, so a copy
// failure changes nothing. `beforeReplace` then runs and may refuse by
// returning false with its error set. Finally each staged file is renamed
// over its target, which replaces it atomically. Returns the number of files
// restored, or -1 with `error` set; after a failed rename the files renamed
// before it stay restored and `error` names the rest.
int restoreBackupFiles(const std::filesystem::path& slotDir,
                       const std::filesystem::path& segmentDir,
                       const std::function<bool(QString* error)>& beforeReplace,
                       QString* error);

// Renames the segment folder `source` to `destination` on the same file
// system. An existing destination is refused unless `replaceExisting`; it is
// then renamed aside first, put back if the move fails, and deleted only
// after the move succeeded.
bool moveSegmentDirectory(const std::filesystem::path& source,
                          const std::filesystem::path& destination,
                          bool replaceExisting,
                          QString* error);

}  // namespace vc3d::segment_files
