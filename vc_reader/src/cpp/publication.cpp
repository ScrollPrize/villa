#include "vc_reader/publication.hpp"
#include <algorithm>
#include <fstream>
#include <set>
#include <vector>
#include <unordered_map>
#include <string>
#include <system_error>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
namespace vcr::files {
void replaceFile(
    const std::filesystem::path& source,
    const std::filesystem::path& destination)
{
#if defined(_WIN32)
    if (!::MoveFileExW(source.c_str(), destination.c_str(),
                       MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        const std::error_code error(
            static_cast<int>(::GetLastError()), std::system_category());
        throw std::filesystem::filesystem_error(
            "cannot replace file", source, destination, error);
    }
#else
    std::filesystem::rename(source, destination);
#endif
}

constexpr auto kSaveRollbackDir = ".vc-save-rollback";
constexpr auto kSaveRollbackReady = ".ready";

std::recursive_mutex& directoryMutex(const std::filesystem::path& dir)
{
    // Recursive for callers holding a directory lock across snapshot and publication.
    static std::mutex mapMutex;
    static std::unordered_map<std::string, std::recursive_mutex> locks;
    const std::string key = dir.lexically_normal().string();
    std::lock_guard<std::mutex> guard(mapMutex);
    return locks[key];
}

void recoverDirectory(const std::filesystem::path& destination)
{
    // Loading and publication must be mutually exclusive. In particular, a
    // loader must not mistake an active rollback journal for one left
    // by an interrupted process and restore it while files are being replaced.
    std::lock_guard<std::recursive_mutex> dirLock(
        directoryMutex(destination));

    const auto rollback = destination / kSaveRollbackDir;
    if (!std::filesystem::exists(rollback)) {
        return;
    }

    const auto ready = rollback / kSaveRollbackReady;
    if (!std::filesystem::exists(ready)) {
        // Publication never began, or it committed and only cleanup was
        // interrupted. In either case the destination is already coherent.
        std::filesystem::remove_all(rollback);
        return;
    }

    std::vector<std::filesystem::path> backups;
    std::set<std::filesystem::path> originalNames;
    for (const auto& entry : std::filesystem::directory_iterator(rollback)) {
        if (entry.is_regular_file() && entry.path().filename() != kSaveRollbackReady) {
            backups.push_back(entry.path());
            originalNames.insert(entry.path().filename());
        }
    }

    // Remove files introduced by the interrupted generation before restoring
    // every original. Keep the backups intact until recovery is complete so a
    // second interruption can safely retry the same recovery.
    for (const auto& entry : std::filesystem::directory_iterator(destination)) {
        if (entry.is_regular_file() && !originalNames.contains(entry.path().filename())) {
            std::filesystem::remove(entry.path());
        }
    }

    const auto restoreDir = rollback / ".restore";
    std::filesystem::create_directories(restoreDir);
    for (const auto& backup : backups) {
        const auto staged = restoreDir / backup.filename();
        std::filesystem::copy_file(
            backup, staged, std::filesystem::copy_options::overwrite_existing);
        replaceFile(staged, destination / backup.filename());
    }
    std::filesystem::remove_all(rollback);
}

void createRollbackBackup(const std::filesystem::path& original,
                          const std::filesystem::path& backup)
{
    std::error_code error;
    std::filesystem::create_hard_link(original,backup,error);
    if(error) std::filesystem::copy_file(original,backup,std::filesystem::copy_options::overwrite_existing);
}

void publishDirectory(const std::filesystem::path& source,
                      const std::filesystem::path& destination,bool overwrite)
{
    std::lock_guard<std::recursive_mutex> lock(directoryMutex(destination));
    if (!std::filesystem::exists(destination)) {
        std::filesystem::rename(source,destination);
        return;
    }
    if (!overwrite) throw std::filesystem::filesystem_error("directory already exists",destination,
        std::make_error_code(std::errc::file_exists));

    // If a previous process stopped during publication, restore its complete
    // prior generation before beginning another save.
    recoverDirectory(destination);

    const auto rollback = destination / kSaveRollbackDir;
    std::filesystem::create_directories(rollback);
    for (const auto& entry : std::filesystem::directory_iterator(destination)) {
        if (entry.is_regular_file()) {
            createRollbackBackup(entry.path(), rollback / entry.path().filename());
        }
    }

    // This marker is created only after every original file is durable in the
    // rollback directory. Its presence means publication may have begun.
    const auto ready = rollback / kSaveRollbackReady;
    {
        std::ofstream marker(ready, std::ios::binary);
        marker << "ready\n";
        if (!marker) {
            throw std::runtime_error("failed to prepare save rollback journal: " +
                                     ready.string());
        }
    }

    // Replace files individually, retaining sibling artifacts and open file mappings.
    std::vector<std::filesystem::path> files;
    for (const auto& entry : std::filesystem::directory_iterator(source)) {
        if (entry.is_regular_file()) {
            files.push_back(entry.path());
        }
    }
    // Do not mutate a directory while iterating it: directory_iterator may
    // otherwise skip entries on Windows. Publish metadata last so readers do
    // not observe it before the corresponding data files are replaced.
    std::sort(files.begin(), files.end(), [](const auto& lhs, const auto& rhs) {
        const bool lhsMeta = lhs.filename() == "meta.json";
        const bool rhsMeta = rhs.filename() == "meta.json";
        return lhsMeta != rhsMeta ? !lhsMeta : lhs.filename() < rhs.filename();
    });
    try {
        for (const auto& file : files) {
            replaceFile(file, destination / file.filename());
        }

        // Removing the marker commits the complete new generation. If cleanup
        // is interrupted after this point, recovery keeps the new files.
        std::filesystem::remove(ready);
    } catch (...) {
        const auto error = std::current_exception();
        try {
            recoverDirectory(destination);
        } catch (...) {
            // Leave the journal intact for the next reader to retry recovery.
        }
        std::rethrow_exception(error);
    }

    std::error_code cleanupEc;
    std::filesystem::remove_all(rollback,cleanupEc);
    std::filesystem::remove_all(source,cleanupEc);
}
void snapshot(const std::filesystem::path& path,int maxBackups,std::chrono::seconds throttle,bool force)
{
    if (path.empty()) {
        throw std::runtime_error("snapshot requires a valid path");
    }
    std::lock_guard<std::recursive_mutex> dirLock(directoryMutex(path));
    if (maxBackups <= 0) {
        return;  // backups disabled
    }

    // No on-disk state to back up yet — the first save will populate
    // the directory; subsequent saveOverwrite calls will pick up the
    // rolling history from then on. Treat this as a no-op rather than
    // an error so the saveOverwrite call site doesn't have to special
    // case it.
    if (!std::filesystem::exists(path / "x.tif")) {
        return;
    }

    // Backups live in a "backups/" dir beside the segment.
    std::filesystem::path backupsDir = path.parent_path() / "backups" / path.filename();

    std::error_code ec;
    std::filesystem::create_directories(backupsDir, ec);
    if (ec) {
        throw std::runtime_error("Failed to create backups directory: " + ec.message());
    }

    // Find existing backup directories and determine next backup number.
    // Also track the newest slot's mtime so we can throttle: many operations
    // (autosave, push-pull edits, growth steps) snapshot via saveOverwrite,
    // which at edit cadence would copy the whole segment dir multiple times a
    // second. Skip if a backup was taken within the throttle window.
    std::vector<int> existingBackups;
    std::filesystem::file_time_type newestBackup{};
    bool haveNewest = false;
    if (std::filesystem::exists(backupsDir)) {
        for (const auto& entry : std::filesystem::directory_iterator(backupsDir)) {
            if (entry.is_directory()) {
                try {
                    int backupNum = std::stoi(entry.path().filename().string());
                    if (backupNum >= 0 && backupNum < maxBackups) {
                        existingBackups.push_back(backupNum);
                        std::error_code mtEc;
                        const auto mt = std::filesystem::last_write_time(entry.path(), mtEc);
                        if (!mtEc && (!haveNewest || mt > newestBackup)) {
                            newestBackup = mt;
                            haveNewest = true;
                        }
                    }
                } catch (...) {
                    // Skip non-numeric directories
                }
            }
        }
    }

    // Throttle: at most one snapshot per segment per throttle.
    if (!force && haveNewest) {
        const auto age = std::filesystem::file_time_type::clock::now() - newestBackup;
        if (age < throttle) {
            return;
        }
    }

    std::sort(existingBackups.begin(), existingBackups.end());

    std::filesystem::path snapshot_dest;

    if (existingBackups.size() < static_cast<size_t>(maxBackups)) {
        // We have room for more backups, find the first available slot
        int nextBackup = 0;
        for (int i = 0; i < maxBackups; ++i) {
            if (std::find(existingBackups.begin(), existingBackups.end(), i) == existingBackups.end()) {
                nextBackup = i;
                break;
            }
        }
        snapshot_dest = backupsDir / std::to_string(nextBackup);
    } else {
        // We're at the limit, rotate backups
        // Delete the oldest (0)
        std::filesystem::remove_all(backupsDir / "0", ec);

        // Rename all backups down by 1 (1->0, 2->1, etc.)
        for (int i = 1; i < maxBackups; ++i) {
            std::filesystem::path oldPath = backupsDir / std::to_string(i);
            std::filesystem::path newPath = backupsDir / std::to_string(i - 1);
            if (std::filesystem::exists(oldPath)) {
                std::filesystem::rename(oldPath, newPath, ec);
            }
        }

        // New backup goes in the last slot
        snapshot_dest = backupsDir / std::to_string(maxBackups - 1);
    }

    // Create the snapshot directory
    std::filesystem::create_directories(snapshot_dest, ec);
    if (ec) {
        throw std::runtime_error("Failed to create snapshot directory: " + ec.message());
    }

    // Copy the persisted generation, including channels and auxiliary regular files.
    for (const auto& entry : std::filesystem::directory_iterator(path, ec)) {
        if (ec) break;
        if (!entry.is_regular_file()) {
            continue;
        }
        std::error_code copyEc;
        std::filesystem::copy_file(
            entry.path(),
            snapshot_dest / entry.path().filename(),
            std::filesystem::copy_options::overwrite_existing,
            copyEc);
        if(copyEc) throw std::filesystem::filesystem_error("snapshot copy failed",entry.path(),copyEc);
    }
    if(ec) throw std::filesystem::filesystem_error("snapshot iteration failed",path,ec);
}
}
