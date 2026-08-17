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

// Writes every payload (temp file + atomic rename, recovery backups of
// overwritten targets on multi-file saves) and retires every existing
// retirePath by moving it into a sibling ".retired" directory before the
// renames; the retired backups are removed only after every rename
// succeeded, and restored to their original paths on failure. Retirement
// is therefore all-or-nothing across the batch. ".retired" is dot-prefixed
// so the fiber loaders and vc_sync ignore it (the .s3sync-conflicts rule).
FiberSaveJobResult runFiberSaveJob(uint64_t sequence,
                                   std::vector<FiberSavePayload> payloads,
                                   std::vector<std::filesystem::path> retirePaths = {});

} // namespace vc3d::line_annotation
