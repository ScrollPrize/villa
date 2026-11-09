#pragma once
#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>
struct sqlite3;

namespace vc::objects::detail {
// A manifest value (kArtifactsMetaKey) is well formed and every file it names
// is held by the store at `storePath` with the stated length.
void checkArtifactReferences(sqlite3*, const std::string& objectUuid,
                             const nlohmann::json& manifest);
std::optional<uint64_t> resolvedArtifactLength(sqlite3*,const std::string& hash);
std::vector<uint8_t> readResolvedArtifact(sqlite3*,const std::string& hash,uint64_t offset,
                                        const std::optional<uint64_t>& count);
// Writes one manifest file to `destination` (atomically), reading its bytes
// in ranges of at most `range` through `read(offset, count)`, and verifies
// its length and SHA-256 before publishing it.
struct ArtifactFileRef { std::string name, sha256; uint64_t length = 0; };
bool artifactFileMatches(const ArtifactFileRef&, const std::filesystem::path&);
void writeArtifactFile(const ArtifactFileRef&, const std::filesystem::path& destination, uint64_t range,
                       const std::function<std::vector<uint8_t>(uint64_t offset, uint64_t count)>& read);
} // namespace vc::objects::detail
