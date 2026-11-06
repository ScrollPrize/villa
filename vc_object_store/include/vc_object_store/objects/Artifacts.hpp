#pragma once
// Artifacts: files that belong to an object but are not geometry (fitted
// models, exports, logs). Their bytes are immutable and content-addressed
// beside the database ("<store>.artifacts/sha256/<2 hex>/<sha256>"), and
// an object version names them through a manifest held in its metadata under
// kArtifactsMetaKey. A manifest is therefore versioned, historical, in the
// change feed and checked like any other metadata edit:
//
//   const auto hash = store.putArtifact(file);                    // durable first
//   SetMetaOp{object, kArtifactsMetaKey, artifactManifestJson({{"model.pt", hash, size}})}
//
// The commit path refuses a changed manifest that names bytes the store does
// not hold, whichever API wrote it. Artifact bytes are never deleted while the
// store is in use: historical versions keep naming them. Hosted clients use
// HostedStore::putArtifact / fetchArtifact; the layout is the same either way.

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace vc::objects {

struct ObjectRecord;

inline constexpr const char* kArtifactsMetaKey = "artifacts";

struct ArtifactFile {
    // Relative path inside the object's materialized directory: '/'-separated,
    // no empty, "." or ".." components, no backslashes.
    std::string name;
    std::string sha256;  // lowercase hex
    uint64_t length = 0;
    bool operator==(const ArtifactFile&) const = default;
};

// Canonical manifest JSON (sorted by name) for SetMetaOp. Throws
// std::invalid_argument for bad names, hashes or duplicate names.
std::string artifactManifestJson(std::vector<ArtifactFile> files);
// The manifest an object record names (empty without one). Throws
// std::invalid_argument when the stored value is malformed.
std::vector<ArtifactFile> artifactManifest(const ObjectRecord&);
// Where a store keeps artifact bytes.
std::filesystem::path artifactBlobDirectory(const std::filesystem::path& storePath);
std::filesystem::path artifactBlobPath(const std::filesystem::path& storePath, const std::string& sha256);
bool isArtifactHash(const std::string&);

} // namespace vc::objects
