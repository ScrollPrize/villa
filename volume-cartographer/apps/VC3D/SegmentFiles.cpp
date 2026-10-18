#include "SegmentFiles.hpp"

#include "utils/Json.hpp"

#include <algorithm>
#include <string>
#include <system_error>

namespace fs = std::filesystem;

namespace vc3d::segment_files {

fs::path backupsDirectory(const fs::path& segmentDir, const fs::path& backupRoot)
{
    const fs::path root = backupRoot.empty() ? segmentDir.parent_path() : backupRoot;
    return root / "backups" / segmentDir.filename();
}

std::vector<Backup> listBackups(const fs::path& backupsDir)
{
    std::vector<Backup> backups;
    std::error_code ec;
    if (!fs::is_directory(backupsDir, ec)) {
        return backups;
    }
    for (const auto& entry : fs::directory_iterator(backupsDir, ec)) {
        if (!entry.is_directory()) {
            continue;
        }
        int index = -1;
        try {
            index = std::stoi(entry.path().filename().string());
        } catch (...) {
            continue;  // Not a numeric directory
        }
        if (index < 0 || index > 9) {
            continue;
        }
        Backup backup;
        backup.index = index;
        backup.path = entry.path();
        const fs::path metaPath = entry.path() / "meta.json";
        if (fs::exists(metaPath, ec)) {
            try {
                const auto meta = utils::Json::parse_file(metaPath);
                if (meta.contains("backup_timestamp")) {
                    backup.timestamp = QString::fromStdString(meta["backup_timestamp"].get_string());
                }
            } catch (...) {
                // Unreadable meta leaves the timestamp empty
            }
        }
        backups.push_back(std::move(backup));
    }
    std::sort(backups.begin(), backups.end(),
              [](const Backup& a, const Backup& b) { return a.index < b.index; });
    return backups;
}

}  // namespace vc3d::segment_files
