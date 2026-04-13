#include "vc/core/util/NetworkFilesystem.hpp"

#ifdef __linux__
#include <sys/statfs.h>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>

// Filesystem magic numbers (from linux/magic.h)
#define NFS_SUPER_MAGIC     0x6969
#define CIFS_MAGIC_NUMBER   0xFF534D42
#define SMB2_MAGIC_NUMBER   0xFE534D42
#define FUSE_SUPER_MAGIC    0x65735546

namespace {

// Known network FUSE subtypes
bool isFuseNetworkSubtype(const std::string& fstype)
{
    static const char* const kNetworkSubtypes[] = {
        "fuse.s3fs",
        "fuse.sshfs",
        "fuse.rclone",
        "fuse.gcsfuse",
        "fuse.goofys",
        "fuse.juicefs",
        "fuse.s3backer",
        "fuse.s3ql",
    };
    for (const auto* sub : kNetworkSubtypes) {
        if (fstype == sub) return true;
    }
    return false;
}

// Known local FUSE subtypes (not network)
bool isFuseLocalSubtype(const std::string& fstype)
{
    static const char* const kLocalSubtypes[] = {
        "fuse.ntfs-3g",
        "fuse.fuseiso",
        "fuse.gvfsd-fuse",
        "fuse.encfs",
        "fuse.cryfs",
        "fuse.gocryptfs",
        "fuse.fuse-overlayfs",
        "fuse.unionfs-fuse",
    };
    for (const auto* sub : kLocalSubtypes) {
        if (fstype == sub) return true;
    }
    return false;
}

// Unescape octal sequences in mount point (e.g. \040 for space)
std::string unescapeMountPoint(const std::string& mountpoint)
{
    std::string result;
    for (size_t i = 0; i < mountpoint.size(); i++) {
        if (mountpoint[i] == '\\' && i + 3 < mountpoint.size() &&
            mountpoint[i+1] >= '0' && mountpoint[i+1] <= '3') {
            char c = static_cast<char>(
                (mountpoint[i+1] - '0') * 64 +
                (mountpoint[i+2] - '0') * 8 +
                (mountpoint[i+3] - '0'));
            result += c;
            i += 3;
        } else {
            result += mountpoint[i];
        }
    }
    return result;
}

// Result of parsing a /proc/mounts line for the best-matching mount point.
struct MountEntry {
    std::string fstype;   // e.g. "fuse.s3fs", "nfs"
    std::string options;  // e.g. "rw,nosuid,nodev,use_cache=/tmp/s3cache"
};

// Parse /proc/mounts to find the mount entry whose mount point is the
// longest prefix of `path`. Returns fstype and options fields.
MountEntry findMountEntry(const std::filesystem::path& path)
{
    namespace fs = std::filesystem;

    std::error_code ec;
    auto canonical = fs::canonical(path, ec);
    if (ec) canonical = path;
    std::string target = canonical.string();

    std::ifstream mounts("/proc/mounts");
    if (!mounts.is_open()) return {};

    MountEntry best;
    size_t bestLen = 0;

    std::string line;
    while (std::getline(mounts, line)) {
        // Format: device mountpoint fstype options dump pass
        std::istringstream iss(line);
        std::string device, mountpoint, fstype, options;
        if (!(iss >> device >> mountpoint >> fstype >> options)) continue;

        auto unescaped = unescapeMountPoint(mountpoint);

        // Check if this mount point is a prefix of our target path
        if (target.size() >= unescaped.size() &&
            target.compare(0, unescaped.size(), unescaped) == 0 &&
            (unescaped.size() == target.size() ||
             unescaped == "/" ||
             target[unescaped.size()] == '/')) {
            if (unescaped.size() > bestLen) {
                bestLen = unescaped.size();
                best.fstype = fstype;
                best.options = options;
            }
        }
    }
    return best;
}

// Parse a comma-separated options string for a key=value pair.
// Returns the value if found, or empty string if not present.
std::string getMountOption(const std::string& options, const std::string& key)
{
    // Options format: "rw,nosuid,nodev,use_cache=/tmp/s3cache,parallel_count=4"
    size_t pos = 0;
    while (pos < options.size()) {
        size_t comma = options.find(',', pos);
        if (comma == std::string::npos) comma = options.size();

        auto token = options.substr(pos, comma - pos);
        auto eq = token.find('=');
        if (eq != std::string::npos && token.substr(0, eq) == key) {
            return token.substr(eq + 1);
        }

        pos = comma + 1;
    }
    return {};
}

// Parse an integer mount option. Returns 0 if not found or invalid.
int getMountOptionInt(const std::string& options, const std::string& key)
{
    auto val = getMountOption(options, key);
    if (val.empty()) return 0;
    try {
        return std::stoi(val);
    } catch (...) {
        return 0;
    }
}

}  // namespace

namespace vc {

FilesystemType detectFilesystemType(const std::filesystem::path& path)
{
    struct statfs buf;
    std::error_code ec;
    auto canonical = std::filesystem::canonical(path, ec);
    const auto& target = ec ? path : canonical;

    if (statfs(target.c_str(), &buf) != 0) {
        return FilesystemType::Unknown;
    }

    auto magic = static_cast<unsigned long>(buf.f_type);

    // Direct network filesystem detection via magic number
    if (magic == NFS_SUPER_MAGIC || magic == CIFS_MAGIC_NUMBER ||
        magic == SMB2_MAGIC_NUMBER) {
        return FilesystemType::NetworkMount;
    }

    // FUSE: need to check /proc/mounts for the actual subtype
    if (magic == FUSE_SUPER_MAGIC) {
        auto entry = findMountEntry(target);
        if (isFuseNetworkSubtype(entry.fstype)) return FilesystemType::NetworkMount;
        if (isFuseLocalSubtype(entry.fstype))   return FilesystemType::Local;
        // Unknown FUSE subtype — be conservative
        return FilesystemType::Unknown;
    }

    return FilesystemType::Local;
}

std::string filesystemTypeLabel(const std::filesystem::path& path)
{
    struct statfs buf;
    std::error_code ec;
    auto canonical = std::filesystem::canonical(path, ec);
    const auto& target = ec ? path : canonical;

    if (statfs(target.c_str(), &buf) != 0) {
        return {};
    }

    auto magic = static_cast<unsigned long>(buf.f_type);

    if (magic == NFS_SUPER_MAGIC)    return "NFS";
    if (magic == CIFS_MAGIC_NUMBER)  return "CIFS";
    if (magic == SMB2_MAGIC_NUMBER)  return "SMB2";

    if (magic == FUSE_SUPER_MAGIC) {
        auto entry = findMountEntry(target);
        if (!entry.fstype.empty()) return entry.fstype;
        return "FUSE (unknown)";
    }

    return {};
}

NetworkMountInfo detectNetworkMount(const std::filesystem::path& path)
{
    NetworkMountInfo info;

    struct statfs buf;
    std::error_code ec;
    auto canonical = std::filesystem::canonical(path, ec);
    const auto& target = ec ? path : canonical;

    if (statfs(target.c_str(), &buf) != 0) {
        info.type = FilesystemType::Unknown;
        return info;
    }

    auto magic = static_cast<unsigned long>(buf.f_type);

    if (magic == NFS_SUPER_MAGIC) {
        info.type = FilesystemType::NetworkMount;
        info.label = "NFS";
        return info;
    }
    if (magic == CIFS_MAGIC_NUMBER) {
        info.type = FilesystemType::NetworkMount;
        info.label = "CIFS";
        return info;
    }
    if (magic == SMB2_MAGIC_NUMBER) {
        info.type = FilesystemType::NetworkMount;
        info.label = "SMB2";
        return info;
    }

    if (magic == FUSE_SUPER_MAGIC) {
        auto entry = findMountEntry(target);
        if (isFuseNetworkSubtype(entry.fstype)) {
            info.type = FilesystemType::NetworkMount;
            info.label = entry.fstype;

            // Parse s3fs-specific mount options
            if (entry.fstype == "fuse.s3fs") {
                info.cacheDir = getMountOption(entry.options, "use_cache");
                info.parallelCount = getMountOptionInt(entry.options, "parallel_count");
                info.multireqMax = getMountOptionInt(entry.options, "multireq_max");
            }
        } else if (isFuseLocalSubtype(entry.fstype)) {
            info.type = FilesystemType::Local;
            info.label = entry.fstype;
        } else {
            info.type = FilesystemType::Unknown;
            info.label = entry.fstype.empty() ? "FUSE (unknown)" : entry.fstype;
        }
        return info;
    }

    return info;  // Local
}

}  // namespace vc

#else  // non-Linux platforms

namespace vc {

FilesystemType detectFilesystemType(const std::filesystem::path&)
{
    return FilesystemType::Unknown;
}

std::string filesystemTypeLabel(const std::filesystem::path&)
{
    return {};
}

NetworkMountInfo detectNetworkMount(const std::filesystem::path&)
{
    return {};
}

}  // namespace vc

#endif
