#pragma once
#include <nlohmann/json.hpp>

#include <filesystem>
#include <string>

namespace vc::objects::detail {
// A manifest value (kArtifactsMetaKey) is well formed and every file it names
// is held by the store at `storePath` with the stated length.
void checkArtifactReferences(const std::filesystem::path& storePath, const std::string& objectUuid,
                             const nlohmann::json& manifest);
} // namespace vc::objects::detail
