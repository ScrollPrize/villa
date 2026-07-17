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

FiberSaveJobResult runFiberSaveJob(uint64_t sequence,
                                   std::vector<FiberSavePayload> payloads);

} // namespace vc3d::line_annotation
