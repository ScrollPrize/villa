#include "LineAnnotationFiberSaveJob.hpp"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string_view>
#include <system_error>
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

// Test hook: VC3D_FIBER_SAVE_FAIL_STAGE="<stage>:<index>" makes the job throw
// at that point (stages: write, replace, retire, restore). The legacy
// VC3D_FIBER_SAVE_FAIL_AFTER_FIRST_REPLACE=1 equals "replace:0" on a
// multi-payload job.
struct FailureInjection {
    std::string stage;
    size_t index = 0;
    bool matches(std::string_view atStage, size_t atIndex) const
    {
        return !stage.empty() && stage == atStage && index == atIndex;
    }
};

FailureInjection failureInjectionFromEnv(bool multiFiberSave)
{
    FailureInjection injection;
    if (const char* staged = std::getenv("VC3D_FIBER_SAVE_FAIL_STAGE");
        staged && *staged != '\0') {
        const std::string_view value(staged);
        const auto colon = value.find(':');
        injection.stage = std::string(value.substr(0, colon));
        if (colon != std::string_view::npos) {
            injection.index = static_cast<size_t>(
                std::strtoull(std::string(value.substr(colon + 1)).c_str(), nullptr, 10));
        }
    } else if (std::getenv("VC3D_FIBER_SAVE_FAIL_AFTER_FIRST_REPLACE") != nullptr &&
               multiFiberSave) {
        injection.stage = "replace";
        injection.index = 0;
    }
    return injection;
}

} // namespace

void writeTextFileChecked(const fs::path& path, const std::string& text)
{
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("Failed to open " + path.string());
    }
    out << text;
    out.flush();
    if (!out) {
        throw std::runtime_error("Failed to write " + path.string());
    }
    out.close();
    if (out.fail()) {
        throw std::runtime_error("Failed to close " + path.string());
    }
}

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
    const FailureInjection inject = failureInjectionFromEnv(multiFiberSave);

    // Every artifact this job creates is registered BEFORE the operation
    // that creates it, so a failure inside that operation still cleans it.
    std::vector<fs::path> tempPaths;
    tempPaths.reserve(payloads.size());
    // (original path, backup path) for every retirement performed, so a
    // failure can rename each backup straight back into place.
    std::vector<std::pair<fs::path, fs::path>> retiredMoves;
    retiredMoves.reserve(retirePaths.size());
    // Entries are pushed BEFORE their rename and this counter advanced after
    // it, so no allocation sits between a filesystem mutation and the ledger
    // entry that undoes it.
    size_t retiredCount = 0;
    // Recovery copy per payload index for targets that existed before the
    // job (multi-file saves only; a single-file save is one atomic rename
    // that leaves the old file untouched on failure). On failure the copy
    // is renamed back over the target, so an overwritten peer of an aborted
    // batch is restored, not left with the new content.
    std::vector<fs::path> recoveryByIndex(payloads.size());
    std::vector<bool> targetExisted(payloads.size(), false);
    size_t renamedCount = 0;
    // Undo bookkeeping is sized up front: the undo itself must not allocate
    // between two filesystem steps, and diagnostics are composed only after
    // every undo step ran.
    struct UndoFailure {
        enum Kind { RestoreTarget, RemoveNewTarget, RestoreRetired, NoRecoveryCopy } kind;
        size_t index;
        std::error_code ec;
    };
    std::vector<UndoFailure> undoFailures;
    undoFailures.reserve(payloads.size() + retirePaths.size());
    std::vector<fs::path> kept;
    kept.reserve(payloads.size() + retirePaths.size());
    std::vector<bool> keepRecovery(payloads.size(), false);
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
            tempPaths.push_back(tempPath);
            if (inject.matches("write", i)) {
                writeTextFileChecked(tempPath, "");
                throw std::runtime_error("Injected failure writing payload " +
                                         std::to_string(i));
            }
            writeTextFileChecked(tempPath, payload.json.dump(2) + '\n');
        }

        if (multiFiberSave) {
            for (size_t i = 0; i < payloads.size(); ++i) {
                const auto& payload = payloads[i];
                std::error_code ec;
                if (fs::exists(payload.path, ec)) {
                    const fs::path recoveryPath = uniqueRecoveryPath(payload.path, sequence, i);
                    // Registered first: a copy that fails half-way is still
                    // an artifact to remove.
                    recoveryByIndex[i] = recoveryPath;
                    fs::copy_file(payload.path,
                                  recoveryPath,
                                  fs::copy_options::none,
                                  ec);
                    if (ec) {
                        throw std::runtime_error("Failed to create recovery backup " +
                                                 recoveryPath.string() + ": " +
                                                 ec.message());
                    }
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
            retiredMoves.emplace_back(retirePath, backupPath);
            if (inject.matches("retire", i)) {
                throw std::runtime_error("Injected failure retiring " +
                                         retirePath.string());
            }
            fs::rename(retirePath, backupPath, ec);
            if (ec) {
                throw std::runtime_error("Failed to retire " + retirePath.string() +
                                         ": " + ec.message());
            }
            ++retiredCount;
        }

        for (size_t i = 0; i < payloads.size(); ++i) {
            std::error_code ec;
            if (!multiFiberSave && inject.matches("replace", i)) {
                throw std::runtime_error("Injected failure before replacing payload " +
                                         std::to_string(i));
            }
            targetExisted[i] = fs::exists(payloads[i].path, ec);
            fs::rename(tempPaths[i], payloads[i].path, ec);
            if (ec) {
                throw std::runtime_error("Failed to replace " +
                                         payloads[i].path.string() + ": " +
                                         ec.message());
            }
            renamedCount = i + 1;
            if (!multiFiberSave) {
                // One payload, one rename, nothing after it that can fail:
                // the single-file contract is "the old file is untouched
                // on failure", which holds because no recovery copy is
                // needed. Injected failures for single payloads therefore
                // fire before the rename (see above), never here.
                break;
            }
            if (inject.stage == "restore" && i + 1 == payloads.size()) {
                // The restore stage needs a failure to undo: fail after the
                // last replacement, then make the restore of `index` fail.
                throw std::runtime_error("Injected failure before restoring payload " +
                                         std::to_string(inject.index));
            }
            if (inject.matches("replace", i)) {
                throw std::runtime_error("Injected failure after replacing payload " +
                                         std::to_string(i));
            }
        }

        // Backups go only after every rename succeeded. Leftovers in
        // .retired/ are invisible to the loaders and to sync, so removal
        // errors are ignored.
        for (const auto& recoveryPath : recoveryByIndex) {
            if (recoveryPath.empty()) {
                continue;
            }
            std::error_code ec;
            fs::remove(recoveryPath, ec);
        }
        for (size_t i = 0; i < retiredCount; ++i) {
            std::error_code ec;
            fs::remove(retiredMoves[i].second, ec);
        }
        result.ok = true;
    } catch (const std::exception& ex) {
        // Undo first, allocate last: every step below records only indices
        // and error codes until the filesystem work is done.
        for (const auto& tempPath : tempPaths) {
            std::error_code ec;
            fs::remove(tempPath, ec);
        }
        // Undo every rename that landed: a target that did not exist before
        // is removed, an overwritten one gets its recovery copy renamed
        // back. Nothing here allocates until every filesystem step ran.
        for (size_t i = 0; i < renamedCount; ++i) {
            std::error_code ec;
            if (!targetExisted[i]) {
                fs::remove(payloads[i].path, ec);
                if (ec) {
                    undoFailures.push_back({UndoFailure::RemoveNewTarget, i, ec});
                }
                continue;
            }
            if (recoveryByIndex[i].empty()) {
                undoFailures.push_back({UndoFailure::NoRecoveryCopy, i, ec});
                continue;
            }
            if (inject.matches("restore", i)) {
                ec = std::make_error_code(std::errc::io_error);
            } else {
                fs::rename(recoveryByIndex[i], payloads[i].path, ec);
            }
            if (ec) {
                undoFailures.push_back({UndoFailure::RestoreTarget, i, ec});
                keepRecovery[i] = true;
            }
        }
        // Recovery copies of targets that were never overwritten (or were
        // restored above) are plain leftovers now.
        for (size_t i = 0; i < recoveryByIndex.size(); ++i) {
            if (recoveryByIndex[i].empty() || keepRecovery[i]) {
                continue;
            }
            std::error_code ec;
            fs::remove(recoveryByIndex[i], ec);
        }
        // Restore retirements; a backup that cannot be moved back is
        // surfaced like a kept recovery file.
        for (size_t i = 0; i < retiredCount; ++i) {
            const auto& [retirePath, backupPath] = retiredMoves[i];
            std::error_code ec;
            fs::rename(backupPath, retirePath, ec);
            if (ec) {
                undoFailures.push_back({UndoFailure::RestoreRetired, i, ec});
            }
        }
        // Diagnostics last (these allocate): the caller reports "recovery
        // required" when recoveryFiles is not empty.
        result.error = ex.what();
        for (const auto& failure : undoFailures) {
            switch (failure.kind) {
            case UndoFailure::RestoreTarget:
                kept.push_back(recoveryByIndex[failure.index]);
                break;
            case UndoFailure::RemoveNewTarget:
            case UndoFailure::NoRecoveryCopy:
                kept.push_back(payloads[failure.index].path);
                break;
            case UndoFailure::RestoreRetired:
                kept.push_back(retiredMoves[failure.index].second);
                break;
            }
        }
        for (const auto& failure : undoFailures) {
            switch (failure.kind) {
            case UndoFailure::RestoreTarget:
                result.error += "; could not restore " + payloads[failure.index].path.string() +
                                " from " + recoveryByIndex[failure.index].string() + ": " +
                                failure.ec.message();
                break;
            case UndoFailure::RemoveNewTarget:
                result.error += "; could not remove the new file " +
                                payloads[failure.index].path.string() + ": " +
                                failure.ec.message();
                break;
            case UndoFailure::RestoreRetired:
                result.error += "; could not restore " +
                                retiredMoves[failure.index].first.string() + " from " +
                                retiredMoves[failure.index].second.string() + ": " +
                                failure.ec.message();
                break;
            case UndoFailure::NoRecoveryCopy:
                result.error += "; " + payloads[failure.index].path.string() +
                                " holds the new content (no recovery copy)";
                break;
            }
        }
        result.recoveryFiles = std::move(kept);
    }
    return result;
}

} // namespace vc3d::line_annotation
