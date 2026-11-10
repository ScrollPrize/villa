#pragma once
#include "vc_object_store/object_links/ObjectLinks.hpp"
#include "geometry/GeometrySnapshot.hpp"
#include <vc_object_store/geometry/SurfaceGrid.hpp>
#include "vc_object_store/objects/GeometryCoordinates.hpp"
#include "vc_object_store/objects/StoreReader.hpp"
#include <array>
#include <functional>
#include <atomic>
#include <map>
#include <memory>
#include <optional>
#include <set>

namespace vc::object_links::detail {
// Working-voxel box: min x, y, z, then max x, y, z.
using Box = std::array<double, 6>;

// Reads the arrays a detector needs of an object's geometry: its current
// revision, or the named one (StoredGeometry::load, loadGeometrySnapshot).
using GeometryLoader = std::function<objects::detail::GeometrySnapshot(
    const objects::ObjectRecord&, const std::optional<std::string>& revision)>;
// StoredGeometry::load over any reader (a local transaction or a hosted
// attachment): the object's current geometry, or the named revision of it,
// which must be the object's. Throws when it has none or an unsupported one.
objects::detail::GeometrySnapshot loadGeometrySnapshot(const objects::StoreReader& reader,
    const objects::ObjectRecord& object, const std::optional<std::string>& revision = std::nullopt);

// One read snapshot for selection and geometry. Payloads can be read one object
// at a time. Before publishing links, check that every observed input is current.
class StoredGeometry {
public:
    explicit StoredGeometry(objects::ObjectStore& db) : db_(db), read_(db.read()) {}
    // Release the snapshot once every selection and load is done; what was
    // observed stays for validate() and expectations().
    void close() { read_.reset(); }
    std::vector<objects::ObjectRecord> select(const std::string& kind,
        const std::vector<std::string>& uuids = {});
    // The object's current geometry, or the named revision of it.
    objects::detail::GeometrySnapshot load(const objects::ObjectRecord& object,
        const std::optional<std::string>& revision = std::nullopt);
    nlohmann::json pointAnnotations(const objects::GeometryRevision& revision, const std::string& array) const
    { return db_.geometryPointAnnotations(txn(), revision.uuid, array); }
    void validate(const objects::WriteTxn& txn) const;
    // The numbered version of every object selected so far: what a checked
    // write publishing links over them expects (every endpoint of a published
    // relationship must be expected).
    objects::VersionSelection expectations() const;
    // Stored (conservative) bounds of an object's current geometry in working
    // voxels: the params' volume, else source XYZ * explicitScale. nullopt
    // when the store records none.
    std::optional<Box> workingBounds(const objects::ObjectRecord& object, const CoordinateParams& params,
                                     double explicitScale = 1.0);
    // The objects of `kind` whose stored bounds overlap `box` (working voxels)
    // grown by `margin`, with objects of that kind at `explicitScale`.
    std::set<std::string> objectsNear(const std::string& kind, const Box& box, double margin,
                                      const CoordinateParams& params, double explicitScale = 1.0);
    const objects::ReadTxn& txn() const;
private:
    objects::ObjectStore& db_;
    std::optional<objects::ReadTxn> read_;
    std::map<std::string, int64_t> generations_;
    std::map<std::string, int64_t> versions_;
    std::map<std::string, std::string> revisions_;
};

// What a detector found over a StoredGeometry snapshot, ready to publish.
struct Detection {
    std::string detector;
    int version = 1;
    std::vector<LinkCandidate> candidates;
    StaleEdgeScope scope;
    bool replaceStalePending = true;
    DetectorReport report;   // objectsConsidered and the detect phase's problems
    // False: nothing is to be committed (the detector was cancelled or failed
    // and says so in report.problems); stale edges are then left alone too.
    bool publish = true;
};

// An incremental run's new objects (canonical UUIDs), checked to be among
// `selected`; nullopt for a full run.
std::optional<std::set<std::string>> newAmong(const std::optional<std::vector<std::string>>& newUuids,
    const std::vector<const std::vector<objects::ObjectRecord>*>& selected, const char* detector);
// An incremental run's stale-edge scope: edges of a new object with both
// endpoints selected, leaving out objects that failed to load or link (their
// edges stay). Nothing is removed when no new object survived.
StaleEdgeScope incrementalScope(const std::string& edgeKind, const std::vector<std::string>& selected,
    const std::set<std::string>& fresh, const std::set<std::string>& failed);
DetectorReport commitStoredCandidates(objects::ObjectStore& db, const StoredGeometry& inputs,
    objects::WriteTxn& txn, const std::string& detectorName, int detectorVersion,
    const std::vector<LinkCandidate>& candidates, bool replaceStalePending,
    const StaleEdgeScope* staleScope, const objects::util::ProgressCallback& progress);
// Commit a detection in the caller's write transaction (commitStoredCandidates
// with its scope); the counts join detection.report, which is returned.
DetectorReport commitDetection(objects::ObjectStore& db, const StoredGeometry& inputs, objects::WriteTxn& txn,
    const Detection& detection, const objects::util::ProgressCallback& progress);
// Commit a detection in a write transaction of its own, the way the
// standalone detectors publish: nothing when it is not to be published.
DetectorReport publishDetection(objects::ObjectStore& db, const StoredGeometry& inputs, const Detection& detection,
    const objects::util::ProgressCallback& progress, const std::atomic_bool* cancel);
// Preflight the complete selection before expensive work or any writes:
// each object's source -> working scale, the params' volume frame or the
// explicit scale. Invalid parameters and objects without a resolvable frame
// throw std::invalid_argument.
void validateCoordinateParams(const CoordinateParams& params, double explicitScale = 1.0);
struct CoordinateScales {
    std::map<std::string, double> scales;
    std::string failures; // "uuid: reason; ..." of the objects without a frame
    // Throws std::invalid_argument naming every failure.
    void check(const CoordinateParams& params) const;
};
// Every selected kind is resolved before any is checked, so one error names
// every object of the run that lacks a frame.
CoordinateScales resolveCoordinateScales(const std::vector<objects::ObjectRecord>& objects,
    const CoordinateParams& params, double explicitScale = 1.0);
void appendCoordinateScales(CoordinateScales& into, const CoordinateScales& more);
std::map<std::string, double> coordinateScales(const std::vector<objects::ObjectRecord>& objects,
    const CoordinateParams& params, double explicitScale = 1.0);
nlohmann::json coordinateProvenance(const objects::ObjectRecord& object, const CoordinateParams& params);
std::shared_ptr<SurfaceGrid> scaledSurfaceGrid(const std::shared_ptr<SurfaceGrid>& source, double scale,
    bool scaleParameterization);
std::shared_ptr<SurfaceGrid> surfaceGrid(const objects::detail::GeometrySnapshot& geometry);
} // namespace vc::object_links::detail
