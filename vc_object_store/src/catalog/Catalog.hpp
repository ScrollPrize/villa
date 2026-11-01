#pragma once
// Internals of catalog placeholders.
#include "vc_object_store/objects/CatalogStore.hpp"
#include "vc_object_store/objects/ObjectStoreSession.hpp"

namespace vc::objects::detail {

// Placeholders for the selected segments' surfaces (all when nullopt); with
// `refresh`, existing records follow the catalog. Returns the UUIDs of fetched
// surfaces whose published tifxyz changed in `stale`.
CatalogStoreReport updateCatalogRecords(ObjectStoreSession&, const AttachmentIdentity&, const vcr::catalog::OpenDataSample&,
                                        bool refresh, const std::optional<std::vector<std::string>>& segmentIds,
                                        std::vector<std::string>& stale);

// Download `url` to `path` atomically. Throws on failure.
void downloadFile(const std::string& url, const std::filesystem::path& path);

} // namespace vc::objects::detail
