#pragma once
#include "vc_object_store/object_links/ObjectLinks.hpp"
#include "geometry/GeometrySnapshot.hpp"
#include "spatial/PatchGrid.hpp"
#include "vc_object_store/objects/GeometryCoordinates.hpp"
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
    std::string pointAnnotations(const objects::GeometryRevision& revision, const std::string& array) const
    { return db_.geometryPointAnnotations(read_, revision.uuid, array); }
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
// Record a DetectorIssue, keeping each detector's existing problems text. An
// empty object names no object.
void reportFailure(DetectorReport& report, const std::string& stage, const std::string& object,
    const std::string& reason, const std::string& text);
void reportDiagnostic(DetectorReport& report, const std::string& stage, const std::string& object,
    const std::string& reason, const std::string& text, int64_t count = 1);
// Preflight the complete selection before expensive work or any writes.
void validateCoordinateParams(const CoordinateParams& params, double explicitScale = 1.0);
std::map<std::string, double> coordinateScales(const std::vector<objects::ObjectRecord>& objects,
    const CoordinateParams& params, double explicitScale = 1.0);
nlohmann::json coordinateProvenance(const objects::ObjectRecord& object, const CoordinateParams& params);
std::shared_ptr<PatchGrid> scaledPatchSurface(const std::shared_ptr<PatchGrid>& source, double scale,
    bool scaleParameterization);
std::shared_ptr<PatchGrid> patchSurface(const objects::detail::GeometrySnapshot& geometry);
} // namespace vc::object_links::detail
