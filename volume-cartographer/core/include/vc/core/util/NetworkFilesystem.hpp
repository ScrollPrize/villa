#pragma once

#include <filesystem>
#include <string>

namespace vc {

enum class FilesystemType { Local, NetworkMount, Unknown };

// Detect whether a path resides on a network-mounted filesystem.
// Linux-only (uses statfs + /proc/mounts). Returns Unknown on other platforms.
FilesystemType detectFilesystemType(const std::filesystem::path& path);

// Human-readable label for the filesystem type (e.g. "NFS", "CIFS", "fuse.s3fs").
// Returns empty string if detection fails or platform unsupported.
std::string filesystemTypeLabel(const std::filesystem::path& path);

// Richer mount-point info including parsed mount options (s3fs-specific).
struct NetworkMountInfo {
    FilesystemType type = FilesystemType::Local;
    std::string label;           // "fuse.s3fs", "NFS", etc.
    std::string cacheDir;        // s3fs use_cache path (empty = no s3fs cache)
    int parallelCount = 0;       // s3fs parallel_count (0 = not set)
    int multireqMax = 0;         // s3fs multireq_max (0 = not set)
};

// Detect filesystem type and parse mount options for network mounts.
// Linux-only; returns default (Local) on other platforms.
NetworkMountInfo detectNetworkMount(const std::filesystem::path& path);

}  // namespace vc
