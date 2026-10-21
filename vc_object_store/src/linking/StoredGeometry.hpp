#pragma once
#include "vc_object_store/object_links/ObjectLinks.hpp"
#include "geometry/GeometrySnapshot.hpp"
#include "spatial/PatchGrid.hpp"
#include <map>
#include <memory>

namespace vc::object_links::detail {
// One read snapshot for selection and geometry. Payloads can be read one object
// at a time. Before publishing links, check that every observed input is current.
class StoredGeometry {
public:
    explicit StoredGeometry(objects::ObjectStore& db) : db_(db), read_(db.read()) {}
    std::vector<objects::ObjectRecord> select(const std::string& kind,
        const std::vector<std::string>& uuids = {});
    objects::detail::GeometrySnapshot load(const objects::ObjectRecord& object);
    void validate(const objects::WriteTxn& txn) const;
private:
    objects::ObjectStore& db_;
    objects::ReadTxn read_;
    std::map<std::string, int64_t> generations_;
    std::map<std::string, std::string> revisions_;
};
DetectorReport commitStoredCandidates(objects::ObjectStore& db, const StoredGeometry& inputs,
    objects::WriteTxn& txn, const std::string& detectorName, int detectorVersion,
    const std::vector<LinkCandidate>& candidates, bool replaceStalePending,
    const StaleEdgeScope* staleScope, const objects::util::ProgressCallback& progress);
std::shared_ptr<PatchGrid> patchSurface(const objects::detail::GeometrySnapshot& geometry);
} // namespace vc::object_links::detail
