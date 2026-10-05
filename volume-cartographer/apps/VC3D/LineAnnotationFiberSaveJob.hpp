#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace vc3d::line_annotation {

struct FiberSavePayload {
    uint64_t fiberId = 0;
    uint64_t generation = 0;
    std::filesystem::path path;
    nlohmann::json json = nlohmann::json::object();
};

struct FiberSaveJobResult {
    bool ok = false;
    std::vector<uint64_t> fiberIds;
    std::vector<uint64_t> generations;
    std::vector<std::filesystem::path> recoveryFiles;
    std::string error;
};

// Writes `text` to `path` through a temporary-free checked stream: open,
// write, flush and close are all verified, so a short write (disk full)
// throws instead of leaving a truncated file that a later rename would
// publish.
void writeTextFileChecked(const std::filesystem::path& path, const std::string& text);

// Writes every payload (checked temp file + atomic rename, recovery copies
// of overwritten targets on multi-file saves) and retires every existing
// retirePath by moving it into a sibling ".retired" directory before the
// renames. On success the recovery copies and retired backups are removed.
// On any failure the job undoes what it did: brand-new targets are removed,
// overwritten targets are restored from their recovery copies, retired
// originals are moved back. `ok` is false, `error` names the cause, and
// `recoveryFiles` lists ONLY the artifacts the undo could not restore or
// remove (normally empty): a non-empty list means the caller must tell the
// user that recovery is required. The batch is therefore all-or-nothing for
// every failure the process observes. ".retired" is dot-prefixed so the
// fiber loaders and vc_sync ignore it (the .s3sync-conflicts rule).
//
// Test hook: VC3D_FIBER_SAVE_FAIL_STAGE="<stage>:<index>" throws at that
// point (stages: write, replace, retire, restore); the legacy
// VC3D_FIBER_SAVE_FAIL_AFTER_FIRST_REPLACE=1 equals "replace:0" on a
// multi-payload job.
FiberSaveJobResult runFiberSaveJob(uint64_t sequence,
                                   std::vector<FiberSavePayload> payloads,
                                   std::vector<std::filesystem::path> retirePaths = {});

} // namespace vc3d::line_annotation
