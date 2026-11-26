#pragma once
#include <filesystem>
#include <mutex>
#include <chrono>
namespace vcr::files {
// Publication and readers of one local directory share this process-local mutex.
std::recursive_mutex& directoryMutex(const std::filesystem::path& directory);
void replaceFile(const std::filesystem::path& source,const std::filesystem::path& destination);
// Publishes one complete file/directory atomically. With overwrite=false an
// existing destination (including an empty directory/symlink) is never replaced.
// Overwrite exchanges complete directories on Linux/macOS, leaving old contents
// at source for caller cleanup. This is not publishDirectory's merge operation.
// Caller owns payload durability and staging cleanup.
void publishPath(const std::filesystem::path& source,const std::filesystem::path& destination,bool overwrite);
void recoverDirectory(const std::filesystem::path& destination);
// Replaces staged regular files, preserving sibling artifacts and subdirectories.
// A rollback journal restores the previous generation after any failed publication.
// With overwrite, a destination that appears while the source is staged is replaced
// the same way, and where the filesystem refuses the no-replace rename (NFS, exFAT) a
// new destination is created with mkdir and filled file by file, meta.json last.
// Without overwrite, a new destination is published atomically or not at all.
void publishDirectory(const std::filesystem::path& source,const std::filesystem::path& destination,bool overwrite);
void snapshot(const std::filesystem::path& path,int maxBackups,std::chrono::seconds throttle,bool force);
}
