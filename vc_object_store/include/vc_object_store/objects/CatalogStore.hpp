#pragma once
// Open Data catalog surfaces in object stores.
//
// Any store can hold one placeholder surface record per catalog surface of a
// sample (vcr::catalog::OpenDataSample::surfaces): its identity, name, binding
// to its catalog volume and bounds come from the catalog (metadata.json), never
// from the published files. Its geometry is fetched on the first read that
// needs it (register fetchCatalogSurface for kCatalogMetaKey with
// ObjectStoreSession::setGeometryFetcher) and kept as a revision; local edits
// follow it like any other surface's.
// Volume data (volumes, normal grids, Lasagna, 3D predictions) is not geometry:
// a project references it (objects/Project.hpp).
#include "vc_object_store/objects/ObjectStoreSession.hpp"
#include <vc_reader/catalog.hpp>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace vc::objects {

// The surface record's UUID: a UUIDv5 of the catalog surface id, the same on
// every machine and across catalog revisions that keep the surface.
std::string catalogSurfaceUuid(const vcr::catalog::OpenDataSurface&);

// A placeholder has storageKind kCatalogStorage and no geometry; fetching gives
// it a geometry revision whose attributes record the source_url fetched, and
// records that revision in meta.catalog_geometry_revision.
inline constexpr const char* kCatalogStorage = "open-data";
struct CatalogFile {
    std::string name;   // file name in the object's artifact directory
    std::string label;
    std::string url;
};
// meta.catalog of a catalog surface: where it is published.
inline constexpr const char* kCatalogMetaKey = "catalog";
struct CatalogReference {
    std::string sample, segment, artifact, volume;
    std::string url;                 // the published tifxyz
    std::vector<CatalogFile> files;  // published per-object images (ink detections)
};
// nullopt for records that are not catalog surfaces.
std::optional<CatalogReference> catalogReference(const ObjectRecord&);

// What adding or refreshing a sample did to its placeholders.
struct CatalogStoreReport {
    std::filesystem::path path;
    int surfaces = 0;  // catalog surfaces of the selected segments
    int added = 0;     // placeholders created
    int updated = 0;   // records whose catalog metadata changed (refresh only)
    int stale = 0;     // fetched surfaces whose published geometry changed; fetched again
    int preserved = 0; // locally edited surfaces left untouched by refresh
};

// Add placeholders for the selected segments' surfaces (all when nullopt) to a
// writable attachment; surfaces it already holds are left as they are.
CatalogStoreReport addCatalogSample(ObjectStoreSession&, const AttachmentIdentity&, const vcr::catalog::OpenDataSample&,
                                    const std::optional<std::vector<std::string>>& segmentIds = std::nullopt);
// The geometry fetcher of catalog surfaces: placeholders (and of stale surfaces
// refreshCatalog refetches): downloads the published tifxyz and writes it as
// the object's next geometry revision. No transaction is held during the
// download. Throws on failure; nothing is written then.
void fetchCatalogSurface(ObjectStore&, const std::string& objectUuid, const ObjectStoreSession::FetchWrite& write);
// Restore a catalog surface to the version fetched from the catalog, undoing
// local edits ("Reset to Open Data Version"); its relationships are kept as
// recorded. Throws for other objects and for placeholders never fetched.
void restoreCatalogVersion(ObjectStoreSession&, const AttachmentIdentity&, const std::string& objectUuid);
// Re-read the published catalog into a store holding the sample's surfaces:
// placeholder metadata follows it, new surfaces are added, and fetched surfaces
// whose published tifxyz changed are fetched again as a new revision. Only
// when asked ("Refresh from Open Data"); locally edited surfaces and every
// other object are untouched.
CatalogStoreReport refreshCatalog(ObjectStoreSession&, const AttachmentIdentity&, const vcr::catalog::OpenDataSample&,
                                  const std::optional<std::vector<std::string>>& segmentIds = std::nullopt);
// A published file of a catalog surface (CatalogReference::files) in its
// artifact directory on `attachment`, downloaded first unless it is there.
// Blocking.
std::filesystem::path publishedFile(ObjectStoreSession&, const AttachmentIdentity&, const std::string& objectUuid,
                                    const std::string& name);

} // namespace vc::objects
