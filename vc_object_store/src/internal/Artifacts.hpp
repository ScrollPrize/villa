#pragma once
#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace vc::objects::detail {
// A manifest value (kArtifactsMetaKey) is well formed and every file it names
// is held by the store at `storePath` with the stated length.
void checkArtifactReferences(const std::filesystem::path& storePath, const std::string& objectUuid,
                             const nlohmann::json& manifest);
// Writes one manifest file to `destination` (atomically), reading its bytes
// in ranges of at most `range` through `read(offset, count)`, and verifies
// its length and SHA-256 before publishing it.
struct ArtifactFileRef { std::string name, sha256; uint64_t length = 0; };
void writeArtifactFile(const ArtifactFileRef&, const std::filesystem::path& destination, uint64_t range,
                       const std::function<std::vector<uint8_t>(uint64_t offset, uint64_t count)>& read);
} // namespace vc::objects::detail
