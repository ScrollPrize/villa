#pragma once
#include "vc_object_store/object_links/ObjectLinks.hpp"
#include "geometry/GeometrySnapshot.hpp"
#include <vc_object_store/geometry/SurfaceGrid.hpp>
#include "vc_object_store/objects/GeometryCoordinates.hpp"
#include <array>
#include <map>
#include <memory>
#include <optional>
#include <set>

namespace vc::object_links::detail {
// Working-voxel box: min x, y, z, then max x, y, z.
using Box = std::array<double, 6>;

// One read snapshot for selection and geometry. Payloads can be read one object
// at a time. Before publishing links, check that every observed input is current.
class StoredGeometry {
public:
    explicit StoredGeometry(objects::ObjectStore& db) : db_(db), read_(db.read()) {}
    std::vector<objects::ObjectRecord> select(const std::string& kind,
        const std::vector<std::string>& uuids = {});
    // The object's current geometry, or the named revision of it.
    objects::detail::GeometrySnapshot load(const objects::ObjectRecord& object,
        const std::optional<std::string>& revision = std::nullopt);
    std::string pointAnnotations(const objects::GeometryRevision& revision, const std::string& array) const
    { return db_.geometryPointAnnotations(read_, revision.uuid, array); }
    void validate(const objects::WriteTxn& txn) const;
    // Stored (conservative) bounds of an object's current geometry in working
    // voxels: the params' volume, else source XYZ * explicitScale. nullopt
    // when the store records none.
    std::optional<Box> workingBounds(const objects::ObjectRecord& object, const CoordinateParams& params,
                                     double explicitScale = 1.0);
    // The objects of `kind` whose stored bounds overlap `box` (working voxels)
    // grown by `margin`, with objects of that kind at `explicitScale`.
    std::set<std::string> objectsNear(const std::string& kind, const Box& box, double margin,
                                      const CoordinateParams& params, double explicitScale = 1.0);
    const objects::ReadTxn& read() const { return read_; }
private:
    objects::ObjectStore& db_;
    objects::ReadTxn read_;
    std::map<std::string, int64_t> generations_;
    std::map<std::string, std::string> revisions_;
};
// An incremental run's new objects (canonical UUIDs), checked to be among
// `selected`; nullopt for a full run.
std::optional<std::set<std::string>> newAmong(const std::optional<std::vector<std::string>>& newUuids,
    const std::vector<const std::vector<objects::ObjectRecord>*>& selected, const char* detector);
// The UUIDs that failed to load or validate in a report.
std::set<std::string> failedObjects(const DetectorReport& report);
DetectorReport commitStoredCandidates(objects::ObjectStore& db, const StoredGeometry& inputs,
    objects::WriteTxn& txn, const std::string& detectorName, int detectorVersion,
    const std::vector<LinkCandidate>& candidates, bool replaceStalePending,
    const StaleEdgeScope* staleScope, const objects::util::ProgressCallback& progress);
// Record a DetectorIssue with a readable message. An
// empty object names no object.
void reportFailure(DetectorReport& report, const std::string& stage, const std::string& object,
    const std::string& reason, const std::string& text);
void reportDiagnostic(DetectorReport& report, const std::string& stage, const std::string& object,
    const std::string& reason, const std::string& text, int64_t count = 1);
// Preflight the complete selection before expensive work or any writes.
// Collect object frame failures in report; callers must check failed() before
// using the scales or starting discovery. Invalid caller parameters throw.
void validateCoordinateParams(const CoordinateParams& params, double explicitScale = 1.0);
std::map<std::string, double> coordinateScales(const std::vector<objects::ObjectRecord>& objects,
    const CoordinateParams& params, DetectorReport& report, double explicitScale = 1.0);
nlohmann::json coordinateProvenance(const objects::ObjectRecord& object, const CoordinateParams& params);
std::shared_ptr<SurfaceGrid> scaledSurfaceGrid(const std::shared_ptr<SurfaceGrid>& source, double scale,
    bool scaleParameterization);
std::shared_ptr<SurfaceGrid> surfaceGrid(const objects::detail::GeometrySnapshot& geometry);
} // namespace vc::object_links::detail
