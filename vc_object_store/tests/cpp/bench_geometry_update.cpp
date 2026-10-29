// Timing harness: what one line-annotation save costs inside the store.
//
// Usage: bench_geometry_update <replica.vcobj copy> <fiber uuid>
//
// Run it on a DISPOSABLE COPY of a replica (the last scenario commits). Take
// the copy while no process has the source open (or with SQLite's backup API)
// so it is a consistent snapshot. The copy must be allowed to edit the fiber:
// a plain store always is; a replica cloned from a hub needs a hub_leases row
// for the fiber held by its own replica identity (sync_replica.uuid), as VC3D
// would hold one, or the checked commit is refused by the hub rules. The
// fiber needs at least two controls. Before any timing, a rolled-back probe
// checks the edit against the store (ID ledger, annotations, relationships).
//
// Each scenario runs updateGeometryArrays in its own checked transaction with
// the internal GeometryWork phase profiler, then rolls back (except the last,
// which commits to include the fsync). Scenarios that the fiber cannot host
// (no sampled line) are skipped and say so.
#include "vc_object_store/objects/GeometryEdits.hpp"
#include "vc_object_store/objects/ObjectStore.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include "geometry/GeometryInternal.hpp"
#include <nlohmann/json.hpp>
#include <chrono>
#include <iostream>
#include <functional>
#include <limits>
using namespace vc::objects;
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
static double ms(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }

int main(int argc, char** argv) try {
    if (argc < 3) { std::cerr << "usage: bench_geometry_update <replica copy> <fiber uuid>\n"; return 2; }
    OpenOptions options; options.author = "bench";
    auto db = ObjectStore::open(argv[1], options);
    const std::string uuid = argv[2];
    vc::geometry::Fiber base; std::string head; int64_t freshId = 0; bool lineAnnotated = false;
    {
        auto txn = db.write();
        const auto current = db.currentGeometry(txn, uuid);
        if (!current) throw std::runtime_error("no geometry for " + uuid);
        head = current->uuid;
        base = std::get<vc::geometry::Fiber>(vc::geometry::decode(db.readGeometryData(txn, head)));
        // A never-allocated ID: the allocation ledger survives deletions, so the
        // largest live ID is not enough; VC3D keeps a persistent high-water mark
        // in meta.source.next_control_point_id. Take the larger of the two.
        const auto object = db.objectByUuid(txn, uuid);
        const auto meta = Json::parse(object ? object->meta : "{}");
        freshId = meta.value("source", Json::object()).value("next_control_point_id", int64_t(0));
        for (auto id : base.controlPointIds) {
            if (id == std::numeric_limits<int64_t>::max()) throw std::runtime_error("control point IDs exhausted");
            freshId = std::max(freshId, id + 1);
        }
        const auto lineAttrs = Json::parse(db.geometryPointAnnotations(txn, head, "line_points"));
        for (const auto& a : lineAttrs) if (a.is_object() && !a.empty()) lineAnnotated = true;
    }
    if (base.controlPoints.size() < 2) throw std::runtime_error("the fiber needs at least two control points");
    std::cout << "fiber " << uuid << " controls=" << base.controlPoints.size() << " line=" << base.linePoints.size()
              << " fresh id " << freshId << (lineAnnotated ? " (annotated line)" : "") << "\n";
    // The checked commit validates every relationship endpoint the edge
    // decisions touch, so the fiber's edge endpoints are pinned as well.
    auto pins = [&](Txn& txn) {
        RevisionSelection expected;
        expected[uuid] = db.currentObjectRevision(txn, uuid)->uuid;
        for (const auto& edge : db.edgesOf(txn, uuid, true))  // retired relationships are validated too
            for (const auto& endpoint : {edge.aUuid, edge.bUuid})
                if (const auto revision = db.currentObjectRevision(txn, endpoint)) expected[endpoint] = revision->uuid;
        return expected;
    };
    // The relationship decisions VC3D's save makes: keep every affected edge as
    // recorded, pinned at the endpoints' current geometry revisions.
    auto decide = [&](Txn& txn, int& asked) {
        return [&](const std::vector<EdgeRecord>& affected) {
            std::vector<GeometryEdgeEdit> result;
            for (const auto& edge : affected) {
                GeometryEdgeEdit edit; edit.edgeUuid = edge.uuid; edit.expectedGeneration = edge.generation;
                edit.action = "keep_recorded";
                if (const auto r = db.currentGeometry(txn, edge.aUuid)) edit.aRevision = r->uuid;
                if (const auto r = db.currentGeometry(txn, edge.bUuid)) edit.bRevision = r->uuid;
                result.push_back(std::move(edit)); ++asked;
            }
            return result;
        };
    };
    struct Transform { std::optional<size_t> controlInsertedAt; bool lineAppended = false; };
    auto run = [&](const char* name, const vc::geometry::Fiber& desired, bool commit, Transform transform) {
        RevisionSelection expected;
        { auto probe = db.write(); expected = pins(probe); }
        auto txn = db.write(expected);
        // Structural edits of an annotated sequence need the annotations
        // supplied transformed (as VC3D does): an inserted control copies its
        // predecessor's, an appended line point its predecessor's.
        std::map<std::string, std::string> annotations;
        if (transform.controlInsertedAt) {
            auto attrs = Json::parse(db.geometryPointAnnotations(txn, head, "control_points"));
            attrs.insert(attrs.begin() + *transform.controlInsertedAt, attrs.at(*transform.controlInsertedAt - 1));
            annotations["control_points"] = attrs.dump();
        }
        if (transform.lineAppended && lineAnnotated) {
            auto attrs = Json::parse(db.geometryPointAnnotations(txn, head, "line_points"));
            attrs.push_back(attrs.back());
            annotations["line_points"] = attrs.dump();
        }
        int asked = 0;
        detail::GeometryWork work;
        const auto t0 = Clock::now();
        std::string revision;
        {
            detail::GeometryWorkScope scope(work);
            revision = updateGeometryArrays(db, txn, uuid, vc::geometry::encode(desired), head, annotations, decide(txn, asked));
        }
        const auto t1 = Clock::now();
        double commitMs = 0;
        if (commit) { txn.commit(); commitMs = ms(t1, Clock::now()); }
        std::cout << "\n== " << name << ": update " << ms(t0, t1) << " ms, edges asked " << asked
                  << (commit ? ", commit " + std::to_string(commitMs) + " ms" : " (rolled back)")
                  << (revision == head ? ", no-op" : ", new revision") << "\n";
        uint64_t total = 0;
        for (const auto& [phase, ns] : work.phaseNanoseconds) total += ns;
        for (const auto& [phase, ns] : work.phaseNanoseconds)
            std::cout << "   " << phase << ": " << ns / 1e6 << " ms\n";
        std::cout << "   phases total " << total / 1e6 << " ms; manifest bytes r/w " << work.manifestBytesRead << "/"
                  << work.manifestBytesWritten << " relationshipRows " << work.relationshipRows
                  << " spatialRows " << work.spatialRows << " identityRows " << work.identityRows << "\n";
        uint64_t rb = 0, hb = 0, wb = 0;
        for (const auto& [k, v] : work.readBytes) rb += v;
        for (const auto& [k, v] : work.hashedBytes) hb += v;
        for (const auto& [k, v] : work.writtenBytes) wb += v;
        std::cout << "   bytes read " << rb << " hashed " << hb << " written " << wb << "\n";
    };
    const bool hasLine = !base.linePoints.empty();
    auto skip = [](const char* name) { std::cout << "\n== " << name << ": skipped (the fiber has no sampled line)\n"; };
    // Insert between controls k and k+1 (k+1 < size holds for two or more controls).
    const size_t k = (base.controlPoints.size() - 1) / 2, at = k + 1;
    auto withInsert = [&](vc::geometry::Point point) {
        auto f = base;
        f.controlPoints.insert(f.controlPoints.begin() + at, point);
        f.controlPointIds.insert(f.controlPointIds.begin() + at, freshId);
        return f;
    };
    vc::geometry::Point mid; for (int i = 0; i < 3; ++i) mid[i] = (base.controlPoints[k][i] + base.controlPoints[k + 1][i]) / 2;
    // The allocation ledger is not public and survives deletions and object
    // restoration, so the candidate is validated by the store itself: a
    // rolled-back structural insert, bumping the ID while the ledger rejects it.
    for (int attempt = 0;; ++attempt) {
        if (attempt == 1000 || freshId == std::numeric_limits<int64_t>::max()) throw std::runtime_error("no never-allocated control ID found");
        RevisionSelection expected;
        { auto probe = db.write(); expected = pins(probe); }
        auto txn = db.write(expected);
        auto attrs = Json::parse(db.geometryPointAnnotations(txn, head, "control_points"));
        attrs.insert(attrs.begin() + at, attrs.at(at - 1));
        int asked = 0;
        try {
            updateGeometryArrays(db, txn, uuid, vc::geometry::encode(withInsert(mid)), head, {{"control_points", attrs.dump()}},
                                 decide(txn, asked));
            break;  // accepted; the transaction rolls back here
        } catch (const std::exception& error) {
            // The helper reports the ledger's rejection as a history conflict
            // (std::runtime_error); only that message means "try the next ID".
            if (std::string(error.what()).find("ID already used") == std::string::npos) throw;
            ++freshId;
        }
    }
    std::cout << "probe accepted control ID " << freshId << "\n";

    run("no-op (identical geometry)", base, false, {});
    run("insert one control point (structural, line unchanged)", withInsert(mid), false, {at});
    if (hasLine) {
        auto g = withInsert(mid); for (auto& p : g.linePoints) p[0] += 0.01;
        run("insert one control point + every line point moved", g, false, {at});
        auto h = withInsert(mid); h.linePoints.push_back(h.linePoints.back());
        run("insert one control point + line count +1", h, false, {at, true});
        auto f = base; for (auto& p : f.linePoints) p[0] += 0.01;
        run("line points moved only (scalar edits)", f, false, {});
        auto l = base; l.linePoints.push_back(l.linePoints.back());
        run("line count +1 only (structural line)", l, false, {{}, true});
    } else {
        for (const auto* name : {"insert one control point + every line point moved", "insert one control point + line count +1",
                                 "line points moved only (scalar edits)", "line count +1 only (structural line)"}) skip(name);
    }
    run("insert one control point, committed", withInsert(base.controlPoints[k]), true, {at});
    return 0;
} catch (const std::exception& error) {
    std::cerr << "bench_geometry_update: " << error.what() << "\n";
    return 1;
}
