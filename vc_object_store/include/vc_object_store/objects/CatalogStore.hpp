#pragma once
// Open Data catalog samples as object stores.
//
// A sample's store holds one placeholder surface record per catalog surface
// (vcr::catalog::OpenDataSample::surfaces): its identity, name, binding to its
// catalog volume and bounds come from the catalog (metadata.json), never from
// the published files. Its geometry is fetched on the first read that needs it
// (attachCatalog) and kept as its first revision.
// Volume data (volumes, normal grids, Lasagna, 3D predictions) never enters it.
#include "vc_object_store/objects/ObjectStoreSession.hpp"
#include <vc_reader/catalog.hpp>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace vc::objects {

// <cacheRoot>/open_data/stores/<sample>.vcobj
std::filesystem::path catalogSampleStorePath(const std::filesystem::path& cacheRoot, const std::string& sampleId);

// The surface record's UUID: a UUIDv5 of the catalog surface id, the same on
// every machine and across catalog revisions that keep the surface.
std::string catalogSurfaceUuid(const vcr::catalog::OpenDataSurface&);

// A placeholder has storageKind kCatalogStorage and no geometry; fetching gives
// it its first revision, whose attributes record the source_url fetched.
inline constexpr const char* kCatalogStorage = "open-data";
struct CatalogFile {
    std::string name;   // file name in the object's artifact directory
    std::string label;
    std::string url;
};
// meta.catalog of a catalog surface: where it is published.
struct CatalogReference {
    std::string sample, segment, artifact, volume;
    std::string url;                 // the published tifxyz
    std::vector<CatalogFile> files;  // published per-object images (ink detections)
};
// nullopt for records that are not catalog surfaces.
std::optional<CatalogReference> catalogReference(const ObjectRecord&);

// What opening or refreshing a sample's store did to its placeholders.
struct CatalogStoreReport {
    std::filesystem::path path;
    int surfaces = 0;  // catalog surfaces of the selected segments
    int added = 0;     // placeholders created
    int updated = 0;   // records whose catalog metadata changed (refresh only)
    int stale = 0;     // fetched surfaces whose published geometry changed; fetched again
    int preserved = 0; // locally edited surfaces left untouched by refresh
};

// The sample's store, created on first use at catalogSampleStorePath(cacheRoot,
// sample.id) and attached to `session` (ObjectStoreSession::attachFetched;
// writable unless options.readOnly), with placeholders for the selected segments'
// surfaces (all when nullopt; an attached store gains newly selected ones).
AttachmentIdentity attachCatalog(ObjectStoreSession&, const vcr::catalog::OpenDataSample&,
                                 const std::filesystem::path& cacheRoot, const OpenOptions& = {},
                                 const std::optional<std::vector<std::string>>& segmentIds = std::nullopt,
                                 CatalogStoreReport* report = nullptr);
// Re-read the published catalog into an attached sample store: placeholder
// metadata follows it, new surfaces are added, and fetched surfaces whose
// published tifxyz changed are fetched again as a new revision. Only when
// asked ("Refresh from Open Data"); locally edited geometry and user stores
// are untouched.
CatalogStoreReport refreshCatalog(ObjectStoreSession&, const AttachmentIdentity&, const vcr::catalog::OpenDataSample&,
                                  const std::optional<std::vector<std::string>>& segmentIds = std::nullopt);
// A published file of a catalog surface (CatalogReference::files) in its
// artifact directory on `attachment` (the sample store, or a store it was
// imported into), downloaded first unless it is there. Blocking.
std::filesystem::path publishedFile(ObjectStoreSession&, const AttachmentIdentity&, const std::string& objectUuid,
                                    const std::string& name);

} // namespace vc::objects
