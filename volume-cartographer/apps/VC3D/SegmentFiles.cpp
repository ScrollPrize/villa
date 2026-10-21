#include "SegmentFiles.hpp"

#include "utils/Json.hpp"

#include <QCoreApplication>

#include <algorithm>
#include <string>
#include <system_error>
#include <utility>

namespace fs = std::filesystem;

namespace vc3d::segment_files {

namespace {

QString tr(const char* text)
{
    return QCoreApplication::translate("SegmentFiles", text);
}

QString pathText(const fs::path& path)
{
    return QString::fromStdString(path.string());
}

bool fail(QString* error, const QString& message)
{
    if (error) {
        *error = message;
    }
    return false;
}

// A sibling of `target` named ".<name>.<tag>" (with a number when taken), on
// the same file system so renames between them stay atomic.
fs::path freeSibling(const fs::path& target, const std::string& tag)
{
    const fs::path base = target.parent_path() /
                          ("." + target.filename().string() + "." + tag);
    fs::path candidate = base;
    std::error_code ec;
    for (int n = 1; fs::exists(candidate, ec) && n < 1000; ++n) {
        candidate = base.string() + "." + std::to_string(n);
    }
    return candidate;
}

}  // namespace

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

int restoreBackupFiles(const fs::path& slotDir,
                       const fs::path& segmentDir,
                       const std::function<bool(QString* error)>& beforeReplace,
                       QString* error)
{
    std::error_code ec;
    if (!fs::is_directory(slotDir, ec)) {
        fail(error, tr("Backup directory does not exist: %1").arg(pathText(slotDir)));
        return -1;
    }
    if (!fs::is_directory(segmentDir, ec)) {
        fail(error, tr("Segment directory does not exist: %1").arg(pathText(segmentDir)));
        return -1;
    }

    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(slotDir, ec)) {
        if (entry.is_regular_file()) {
            files.push_back(entry.path().filename());
        }
    }
    if (ec) {
        fail(error, tr("Cannot read backup directory %1: %2")
                        .arg(pathText(slotDir), QString::fromStdString(ec.message())));
        return -1;
    }
    if (files.empty()) {
        return 0;
    }

    const fs::path staging = freeSibling(segmentDir, "restoring");
    auto discardStaging = [&staging]() {
        std::error_code ignored;
        fs::remove_all(staging, ignored);
    };
    if (!fs::create_directory(staging, ec) || ec) {
        fail(error, tr("Cannot create staging folder %1: %2")
                        .arg(pathText(staging), QString::fromStdString(ec.message())));
        return -1;
    }
    for (const fs::path& name : files) {
        fs::copy_file(slotDir / name, staging / name, ec);
        if (ec) {
            fail(error, tr("Failed to copy %1: %2")
                            .arg(pathText(name), QString::fromStdString(ec.message())));
            discardStaging();
            return -1;
        }
    }

    if (beforeReplace && !beforeReplace(error)) {
        discardStaging();
        return -1;
    }

    int restored = 0;
    for (const fs::path& name : files) {
        fs::rename(staging / name, segmentDir / name, ec);
        if (ec) {
            fail(error, tr("Restored %1 of %2 files; replacing %3 failed: %4")
                            .arg(restored)
                            .arg(static_cast<int>(files.size()))
                            .arg(pathText(name), QString::fromStdString(ec.message())));
            discardStaging();
            return -1;
        }
        ++restored;
    }
    discardStaging();
    return restored;
}

bool moveSegmentDirectory(const fs::path& source,
                          const fs::path& destination,
                          bool replaceExisting,
                          QString* error)
{
    std::error_code ec;
    if (!fs::is_directory(source, ec)) {
        return fail(error, tr("Segment directory does not exist: %1").arg(pathText(source)));
    }
    fs::path aside;
    if (fs::exists(destination, ec)) {
        if (!replaceExisting) {
            return fail(error, tr("%1 already exists.").arg(pathText(destination)));
        }
        aside = freeSibling(destination, "replaced");
        fs::rename(destination, aside, ec);
        if (ec) {
            return fail(error, tr("Failed to move the existing segment aside: %1")
                                   .arg(QString::fromStdString(ec.message())));
        }
    }

    fs::rename(source, destination, ec);
    if (ec) {
        const QString reason = QString::fromStdString(ec.message());
        if (!aside.empty()) {
            std::error_code restoreEc;
            fs::rename(aside, destination, restoreEc);
            if (restoreEc) {
                return fail(error, tr("Failed to move segment: %1. The replaced segment "
                                      "was left at %2.")
                                       .arg(reason, pathText(aside)));
            }
        }
        return fail(error, tr("Failed to move segment: %1").arg(reason));
    }
    if (!aside.empty()) {
        std::error_code ignored;
        fs::remove_all(aside, ignored);
    }
    return true;
}

}  // namespace vc3d::segment_files
