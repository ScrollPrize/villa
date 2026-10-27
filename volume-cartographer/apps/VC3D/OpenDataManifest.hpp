#pragma once
#include <vc_reader/catalog.hpp>

namespace vc3d::opendata {
std::filesystem::path cachedOpenDataManifestPath();
// The cached Open Data catalog (vcr::catalog::cachedManifest of that file),
// nullptr before it has been fetched.
std::shared_ptr<const vcr::catalog::OpenDataManifest> cachedCatalog();
// A catalog id as one cache path component: characters other than
// [A-Za-z0-9._-] become '_' and leading '.'/'_' are dropped ("unnamed" if empty).
[[nodiscard]] std::string safePathComponent(std::string value);
// A short stable name for a cache keyed by `identity` (16 hex digits of FNV-1a).
[[nodiscard]] std::string cacheKey(std::string_view identity);
// Optional subset of a sample's resources to attach. Each axis is independent:
// an absent sub-field means "no filter on that axis". A raw source volume is
// governed by `volumeIds` and optional `rawVolumeIds`; a derived representation (normal grids /
// lasagna / prediction) must pass all three axes. A nullptr selection anywhere
// downstream preserves the attach-everything behavior.
struct OpenDataResourceSelection {
    std::optional<std::vector<std::string>> volumeIds;
    std::optional<std::vector<vcr::catalog::OpenDataRepresentationRef>> representations;
    std::optional<std::vector<vcr::catalog::OpenDataRepresentationKind>> kinds;
    // Independent source-only filter; does not exclude derived representations.
    std::optional<std::vector<std::string>> rawVolumeIds;
    std::optional<std::vector<std::string>> segmentIds;

    // True when this volume and its representations pass the volumeIds axis.
    [[nodiscard]] bool allowsVolume(const std::string& volumeId) const;
    // True when the (volumeIndex, artifactIndex, kind, volumeId) representation passes
    // every provided axis.
    [[nodiscard]] bool allowsRepresentation(std::size_t volumeIndex,
                                            std::size_t artifactIndex,
                                            vcr::catalog::OpenDataRepresentationKind kind,
                                            const std::string& volumeId) const;
};


} // namespace vc3d::opendata
