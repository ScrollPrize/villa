// A selected-graph read over many objects: both ways the incident edges are collected (one sequential pass over the
// edge table when the selection covers much of the store; indexed probes in key-ordered batches when it does not)
// give the same relationships, batched decoding spans batch boundaries (> 500 keys), the overload taking resolved
// revisions equals the reference-taking one, and a duplicated object is refused.
#include "TestSupport.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include "vc_object_store/objects/ObjectStore.hpp"

#include <algorithm>
#include <iostream>
#include <map>
#include <random>
#include <tuple>

using namespace vc::objects;

namespace {
constexpr size_t kTracks = 1300;   // three key-ordered batches of 500 in the probe path
constexpr size_t kFillers = 12000; // enough unselected objects to put 1300 under 10 % of the store

struct Built {
    std::vector<std::string> trackUuids;           // index -> uuid
    std::map<std::string, size_t> indexOf;         // uuid -> index
    std::vector<std::pair<size_t, size_t>> edges;  // (i, j) with i < j
};

// The same tracks and edges in every store (indices and geometry); uuids differ per store.
Built build(ObjectStore& db, size_t fillers)
{
    Built built;
    auto txn = db.write();
    for (size_t i = 0; i < kTracks; ++i) {
        ObjectRecord record; record.kind = "track"; record.track.emplace();
        record.track->datasetRef = "test"; record.track->externalKey = std::to_string(i);
        const auto uuid = db.upsertObject(txn, record);
        vc::geometry::Track track; track.points = {{int32_t(i), 0, 0}, {int32_t(i), 1, 0}, {int32_t(i), 2, 0}};
        db.putGeometry(txn, uuid, vc::geometry::encode(track));
        built.indexOf[uuid] = i; built.trackUuids.push_back(uuid);
    }
    for (size_t i = 0; i < fillers; ++i) {
        ObjectRecord record; record.kind = "track"; record.track.emplace();
        record.track->datasetRef = "filler"; record.track->externalKey = "f" + std::to_string(i);
        db.upsertObject(txn, record);
    }
    std::mt19937 rng(7);
    std::set<std::pair<size_t, size_t>> pairs;
    for (size_t i = 0; i + 1 < kTracks; ++i) pairs.insert({i, i + 1});
    while (pairs.size() < kTracks + 600) {
        size_t i = rng() % kTracks, j = rng() % kTracks;
        if (i == j) continue;
        pairs.insert({std::min(i, j), std::max(i, j)});
    }
    for (const auto& [i, j] : pairs) {
        EdgeRecord link; link.kind = "crossing"; link.aUuid = built.trackUuids[i]; link.bUuid = built.trackUuids[j];
        link.aAnchor.points = {{"points", {1}}}; link.bAnchor.points = {{"points", {1}}};
        link.aAnchor.metadataKeys = link.bAnchor.metadataKeys = {"coordinates"};
        db.addEdge(txn, link);
        built.edges.emplace_back(i, j);
    }
    txn.commit();
    return built;
}

using Row = std::tuple<size_t, size_t, std::string, bool>; // (i, j, kind, applicable), store-independent

std::vector<Row> rows(const Built& built, const std::vector<SelectedRelationship>& selected)
{
    std::vector<Row> result;
    for (const auto& entry : selected) {
        const auto& r = entry.relationship.record;
        // addEdge canonicalises the endpoint order, so the pair is keyed order-independently.
        const auto i = built.indexOf.at(r.aUuid), j = built.indexOf.at(r.bUuid);
        result.emplace_back(std::min(i, j), std::max(i, j), r.kind, entry.applicable);
    }
    std::sort(result.begin(), result.end());
    return result;
}

void requireSame(const std::vector<Row>& got, const std::vector<Row>& expected, const char* message)
{
    if (got == expected) return;
    size_t applicable = 0;
    for (const auto& row : got) applicable += std::get<3>(row);
    std::cerr << message << ": got " << got.size() << " rows (" << applicable << " applicable), expected " << expected.size() << '\n';
    require(false, message);
}

std::vector<ObjectVersion> references(ObjectStore& db, const Txn& txn, const Built& built, size_t count)
{
    std::vector<ObjectVersion> refs;
    for (size_t i = 0; i < count; ++i) refs.push_back(db.currentObjectRevision(txn, built.trackUuids[i])->reference());
    return refs;
}
}

int main()
try {
    TemporaryDirectory directory;
    OpenOptions options; options.author = "selected-graph-read-test";
    auto scanStore = ObjectStore::create(directory.path / "scan.vcobj", options);    // 1300 of 1300 objects: sequential pass
    auto probeStore = ObjectStore::create(directory.path / "probe.vcobj", options);  // 1300 of 13300 objects: indexed probes
    const auto scan = build(scanStore, 0);
    const auto probe = build(probeStore, kFillers);
    require(scan.edges == probe.edges, "the two stores hold the same edges by index");

    std::vector<Row> expected;
    for (const auto& [i, j] : scan.edges) expected.emplace_back(i, j, "crossing", true);
    std::sort(expected.begin(), expected.end());

    for (auto* store : {&scanStore, &probeStore}) {
        const auto& built = store == &scanStore ? scan : probe;
        auto txn = store->read();
        const auto refs = references(*store, txn, built, kTracks);
        const auto byReference = store->relationshipsAt(txn, refs);
        requireSame(rows(built, byReference), expected, "every edge among the selected objects, once, applicable");
        // The order rule: by edge uuid, as before.
        for (size_t k = 1; k < byReference.size(); ++k)
            require(byReference[k - 1].relationship.record.uuid < byReference[k].relationship.record.uuid, "ordered by edge uuid");
        std::vector<ObjectRevision> revisions;
        for (const auto& revision : store->objectsAtVersions(txn, refs)) revisions.push_back(*revision);
        const auto byRevision = store->relationshipsAt(txn, revisions);
        require(byRevision.size() == byReference.size(), "the overload returns the same count");
        for (size_t k = 0; k < byRevision.size(); ++k)
            require(byRevision[k].relationship.reference() == byReference[k].relationship.reference() &&
                    byRevision[k].applicable == byReference[k].applicable &&
                    byRevision[k].relationship.aReference == byReference[k].relationship.aReference &&
                    byRevision[k].relationship.bReference == byReference[k].relationship.bReference,
                    "the overload returns the same relationships in the same order");
        // A selection that stops inside a batch: every edge touching a selected object, once; applicable only when
        // both endpoints are selected (an edge to an unselected endpoint is exposed, not applicable: the store's rule).
        const size_t part = 777;
        const auto partial = store->relationshipsAt(txn, references(*store, txn, built, part));
        std::vector<Row> partialExpected;
        for (const auto& [i, j] : built.edges)
            if (i < part || j < part) partialExpected.emplace_back(i, j, "crossing", i < part && j < part);
        std::sort(partialExpected.begin(), partialExpected.end());
        requireSame(rows(built, partial), partialExpected, "a partial selection sees only its own edges");
        auto twice = revisions; twice.push_back(revisions.front());
        mustFail([&] { store->relationshipsAt(txn, twice); });
    }
    std::cout << "selected graph read passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "FAILED: " << error.what() << '\n';
    return 1;
}
