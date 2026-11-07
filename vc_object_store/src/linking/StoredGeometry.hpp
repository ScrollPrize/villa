#pragma once
#include "vc_object_store/object_links/ObjectLinks.hpp"
#include "geometry/GeometrySnapshot.hpp"
#include <vc_object_store/geometry/SurfaceGrid.hpp>
#include <atomic>
#include <map>
#include <memory>
#include <optional>

namespace vc::object_links::detail {
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
    objects::detail::GeometrySnapshot load(const objects::ObjectRecord& object);
    nlohmann::json pointAnnotations(const objects::GeometryRevision& revision, const std::string& array) const
    { return db_.geometryPointAnnotations(txn(), revision.uuid, array); }
    void validate(const objects::WriteTxn& txn) const;
    // The numbered version of every object selected so far: what a checked
    // write publishing links over them expects (every endpoint of a published
    // relationship must be expected).
    objects::VersionSelection expectations() const;
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
std::shared_ptr<SurfaceGrid> surfaceGrid(const objects::detail::GeometrySnapshot& geometry);
} // namespace vc::object_links::detail
