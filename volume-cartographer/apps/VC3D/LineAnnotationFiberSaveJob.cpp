#include "LineAnnotationFiberSaveJob.hpp"

#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <utility>

namespace fs = std::filesystem;

namespace vc3d::line_annotation {

namespace {

fs::path uniqueRecoveryPath(const fs::path& finalPath, uint64_t sequence, size_t index)
{
    const fs::path base = finalPath.string() + ".recovery." +
                          std::to_string(sequence) + "." +
                          std::to_string(index);
    if (!fs::exists(base)) {
        return base;
    }
    for (int suffix = 1; suffix < 10000; ++suffix) {
        fs::path candidate = base.string() + "." + std::to_string(suffix);
        if (!fs::exists(candidate)) {
            return candidate;
        }
    }
    throw std::runtime_error("Could not choose a recovery path for " +
                             finalPath.string());
}

} // namespace

FiberSaveJobResult runFiberSaveJob(uint64_t sequence,
                                   std::vector<FiberSavePayload> payloads,
                                   std::vector<std::filesystem::path> retirePaths)
{
    FiberSaveJobResult result;
    result.ok = false;
    result.fiberIds.reserve(payloads.size());
    result.generations.reserve(payloads.size());
    for (const auto& payload : payloads) {
        result.fiberIds.push_back(payload.fiberId);
        result.generations.push_back(payload.generation);
    }

    const bool multiFiberSave = payloads.size() > 1;
    std::vector<fs::path> tempPaths;
    tempPaths.reserve(payloads.size());
    // (original path, backup path) for every retirement performed, so a
    // failure can rename each backup straight back into place.
    std::vector<std::pair<fs::path, fs::path>> retiredMoves;
    retiredMoves.reserve(retirePaths.size());
    try {
        for (size_t i = 0; i < payloads.size(); ++i) {
            const auto& payload = payloads[i];
            const fs::path parent = payload.path.parent_path();
            std::error_code ec;
            if (!parent.empty()) {
                fs::create_directories(parent, ec);
                if (ec) {
                    throw std::runtime_error("Failed to create " + parent.string() +
                                             ": " + ec.message());
                }
            }
            const fs::path tempPath = payload.path.string() + ".tmp." +
                                      std::to_string(sequence) + "." +
                                      std::to_string(i);
            {
                std::ofstream out(tempPath);
                if (!out) {
                    throw std::runtime_error("Failed to open " + tempPath.string());
                }
                out << payload.json.dump(2) << '\n';
            }
            tempPaths.push_back(tempPath);
        }

        if (multiFiberSave) {
            for (size_t i = 0; i < payloads.size(); ++i) {
                const auto& payload = payloads[i];
                std::error_code ec;
                if (fs::exists(payload.path, ec)) {
                    const fs::path recoveryPath = uniqueRecoveryPath(payload.path, sequence, i);
                    fs::copy_file(payload.path,
                                  recoveryPath,
                                  fs::copy_options::none,
                                  ec);
                    if (ec) {
                        throw std::runtime_error("Failed to create recovery backup " +
                                                 recoveryPath.string() + ": " +
                                                 ec.message());
                    }
                    result.recoveryFiles.push_back(recoveryPath);
                }
            }
        }

        // Retire originals before the renames: an atomic move into the
        // dot-prefixed sibling directory doubles as the backup, and nothing
        // has been renamed into place yet when a move fails.
        for (size_t i = 0; i < retirePaths.size(); ++i) {
            const fs::path& retirePath = retirePaths[i];
            std::error_code ec;
            if (!fs::exists(retirePath, ec)) {
                continue;
            }
            const fs::path retiredDir = retirePath.parent_path() / ".retired";
            fs::create_directories(retiredDir, ec);
            if (ec) {
                throw std::runtime_error("Failed to create " + retiredDir.string() +
                                         ": " + ec.message());
            }
            const fs::path backupPath = uniqueRecoveryPath(
                retiredDir / retirePath.filename(), sequence, i);
            fs::rename(retirePath, backupPath, ec);
            if (ec) {
                throw std::runtime_error("Failed to retire " + retirePath.string() +
                                         ": " + ec.message());
            }
            retiredMoves.emplace_back(retirePath, backupPath);
        }

        const bool failAfterFirst =
            std::getenv("VC3D_FIBER_SAVE_FAIL_AFTER_FIRST_REPLACE") != nullptr;
        for (size_t i = 0; i < payloads.size(); ++i) {
            std::error_code ec;
            fs::rename(tempPaths[i], payloads[i].path, ec);
            if (ec) {
                throw std::runtime_error("Failed to replace " +
                                         payloads[i].path.string() + ": " +
                                         ec.message());
            }
            if (failAfterFirst && multiFiberSave && i == 0) {
                throw std::runtime_error("Injected failure after first fiber replacement");
            }
        }

        // Backups go only after every rename succeeded. Leftovers in
        // .retired/ are invisible to the loaders and to sync, so removal
        // errors are ignored.
        for (const auto& recoveryPath : result.recoveryFiles) {
            std::error_code ec;
            fs::remove(recoveryPath, ec);
        }
        result.recoveryFiles.clear();
        for (const auto& [retirePath, backupPath] : retiredMoves) {
            (void)retirePath;
            std::error_code ec;
            fs::remove(backupPath, ec);
        }
        result.ok = true;
    } catch (const std::exception& ex) {
        result.error = ex.what();
        for (const auto& tempPath : tempPaths) {
            std::error_code ec;
            fs::remove(tempPath, ec);
        }
        // Restore retirements; a backup that cannot be moved back is
        // surfaced like a kept recovery file.
        for (const auto& [retirePath, backupPath] : retiredMoves) {
            std::error_code ec;
            fs::rename(backupPath, retirePath, ec);
            if (ec) {
                result.recoveryFiles.push_back(backupPath);
            }
        }
    }
    return result;
}

} // namespace vc3d::line_annotation
