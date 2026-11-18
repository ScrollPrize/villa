#pragma once
#include <vc_reader/catalog.hpp>

namespace vc3d::opendata {
// Optional subset of a sample's resources to attach. Each axis is independent:
// an absent sub-field means "no filter on that axis". A raw source volume is
// governed by `volumeIds`, optional `rawVolumeIds` and optional `rawVolumeArtifacts`;
// a derived representation (normal grids / lasagna / prediction) must pass all
// three axes. A nullptr selection anywhere downstream preserves the
// attach-everything behavior.

// One source volume artifact of a sample: a raw zarr or a Delta3D re-encoding
// (`vcr::catalog::sourceVolumeArtifacts`).
struct OpenDataSourceArtifactRef {
    std::size_t volumeIndex = 0;
    std::size_t artifactIndex = 0;
};

struct OpenDataResourceSelection {
    std::optional<std::vector<std::string>> volumeIds;
    std::optional<std::vector<vcr::catalog::OpenDataRepresentationRef>> representations;
    std::optional<std::vector<vcr::catalog::OpenDataRepresentationKind>> kinds;
    // Independent source-only filter; does not exclude derived representations.
    std::optional<std::vector<std::string>> rawVolumeIds;
    // Source volumes by artifact, for a volume published in several encodings.
    // Absent: every source artifact of an allowed volume.
    std::optional<std::vector<OpenDataSourceArtifactRef>> rawVolumeArtifacts;
    std::optional<std::vector<std::string>> segmentIds;

    // True when this volume and its representations pass the volumeIds axis.
    [[nodiscard]] bool allowsVolume(const std::string& volumeId) const;
    // True when the source artifact (volumeIndex, artifactIndex) of volumeId passes
    // the volumeIds, rawVolumeIds and rawVolumeArtifacts axes.
    [[nodiscard]] bool allowsSourceArtifact(std::size_t volumeIndex, std::size_t artifactIndex,
                                            const std::string& volumeId) const;
    // True when the (volumeIndex, artifactIndex, kind, volumeId) representation passes
    // every provided axis.
    [[nodiscard]] bool allowsRepresentation(std::size_t volumeIndex,
                                            std::size_t artifactIndex,
                                            vcr::catalog::OpenDataRepresentationKind kind,
                                            const std::string& volumeId) const;
};


} // namespace vc3d::opendata
