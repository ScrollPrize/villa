#pragma once
#include <filesystem>
#include <mutex>
#include <chrono>
namespace vcr::files {
// Publication and readers of one local directory share this process-local mutex.
std::recursive_mutex& directoryMutex(const std::filesystem::path& directory);
void replaceFile(const std::filesystem::path& source,const std::filesystem::path& destination);
void recoverDirectory(const std::filesystem::path& destination);
// Replaces staged regular files, preserving sibling artifacts and subdirectories.
// A rollback journal restores the previous generation after any failed publication.
void publishDirectory(const std::filesystem::path& source,const std::filesystem::path& destination,bool overwrite);
void snapshot(const std::filesystem::path& path,int maxBackups,std::chrono::seconds throttle,bool force);
}
