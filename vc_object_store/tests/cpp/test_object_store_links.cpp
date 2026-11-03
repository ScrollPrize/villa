// Discovery consumes stored revisions, never object source paths.
#include "TestSupport.hpp"
#include "vc_object_store/object_links/ObjectLinks.hpp"
#include "vc_object_store/objects/ObjectStore.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <iostream>
#include <set>
#include <string>
#include <vector>

using namespace vc::objects;
namespace links = vc::object_links;
using Json = nlohmann::json;

namespace {
template<class T>
GeometryArray array(const std::string& name, const std::string& dtype, std::vector<int64_t> shape, const std::vector<T>& values)
{
    GeometryArray result{name, dtype, std::move(shape), std::vector<uint8_t>(values.size() * sizeof(T))};
    std::memcpy(result.bytes.data(), values.data(), result.bytes.size());
    return result;
}

std::string put(ObjectStore& db, ObjectRecord record, const GeometryData& geometry)
{
    record.storageRef = "/missing/source";
    auto txn = db.write();
    const auto uuid = db.upsertObject(txn, record);
    db.putGeometry(txn, uuid, geometry);
    txn.commit();
    return uuid;
}

GeometryData planeGeometry(float offset, const std::string& attributes = R"({"scale":[1,1]})")
{
    std::vector<float> points;
    for (int y = 0; y < 33; ++y)
        for (int x = 0; x < 33; ++x) points.insert(points.end(), {float(x) + offset, float(y), 20});
    return {"surface", 1, "source_xyz", attributes, {array("points", "<f4", {33, 33, 3}, points)}};
}

std::string plane(ObjectStore& db, float offset = 0, const std::string& name = "surface")
{
    ObjectRecord record; record.kind = "surface"; record.name = name;
    return put(db, record, planeGeometry(offset));
}

std::string track(ObjectStore& db, const std::string& key, const std::vector<int32_t>& points, const std::string& family,
                  const std::string& dataset = "dataset-label")
{
    ObjectRecord record; record.kind = "track";
    record.track.emplace(); record.track->datasetRef = dataset; record.track->externalKey = key; record.track->family = family;
    const auto attributes = Json{{"family", family}}.dump();
    return put(db, record, {"track", 1, "source_xyz", attributes, {array("points", "<i4", {int64_t(points.size() / 3), 3}, points)}});
}

GeometryEdgeEdit choice(ObjectStore& db, const Txn& txn, const EdgeRecord& edge, const std::string& action)
{
    GeometryEdgeEdit result; result.edgeUuid = edge.uuid; result.expectedGeneration = edge.generation; result.action = action;
    if (const auto head = db.currentGeometry(txn, edge.aUuid)) result.aRevision = head->uuid;
    if (const auto head = db.currentGeometry(txn, edge.bUuid)) result.bRevision = head->uuid;
    return result;
}

// Replace an object's first array with `bytes`, removing every relationship the edit affects.
void moveRemovingLinks(ObjectStore& db, const std::string& uuid, const std::string& arrayName, const std::vector<uint8_t>& bytes)
{
    auto txn = db.write();
    const auto head = db.currentGeometry(txn, uuid)->uuid;
    db.editGeometry(txn, uuid, {{arrayName, 0, bytes}}, head, [&](const std::vector<EdgeRecord>& affected) {
        std::vector<GeometryEdgeEdit> decisions;
        for (const auto& edge : affected) decisions.push_back(choice(db, txn, edge, "remove"));
        return decisions;
    });
    txn.commit();
}

std::vector<EdgeRecord> edgesOf(ObjectStore& db, const std::string& uuid)
{
    auto txn = db.read();
    return db.edgesOf(txn, uuid);
}

std::string upper(std::string text)
{
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return char(std::toupper(c)); });
    return text;
}

bool hasProblem(const links::DetectorReport& report, const std::string& text)
{
    return std::any_of(report.problems.begin(), report.problems.end(),
                       [&](const std::string& problem) { return problem.find(text) != std::string::npos; });
}

ObjectStore create(const TemporaryDirectory& directory, const std::string& name)
{
    OpenOptions options; options.author = "links-test";
    return ObjectStore::create(directory.path / name, options);
}

void surfaceDiscoveryUsesCurrentRevisionsAndPreservesReview(const TemporaryDirectory& directory)
{
    auto db = create(directory, "overlaps.vcobj");
    const auto a = plane(db), b = plane(db);
    const auto generations = [&] {
        auto txn = db.read();
        return std::pair{db.objectByUuid(txn, a)->generation, db.objectByUuid(txn, b)->generation};
    };
    const auto before = generations();
    require(links::linkSurfaceOverlaps(db).edgesCreated == 1, "overlap found");
    require(generations() == before, "discovery leaves the objects unchanged");
    const auto edge = edgesOf(db, a).at(0).uuid;
    const auto stale = [&] { auto txn = db.read(); return db.edgeGeometryState(txn, edge).stale; };
    require(!stale(), "a new link is current");
    require(links::linkSurfaceOverlaps(db).edgesSkippedExisting == 1, "a rerun keeps the link");
    // A new revision with identical coordinates still needs revalidation.
    {
        auto txn = db.write();
        const auto edited = planeGeometry(0, R"({"scale":[1,1],"edit":true})");
        std::vector<GeometryEdgeEdit> decisions;
        for (const auto& affected : db.affectedGeometryEdges(txn, b, edited)) decisions.push_back(choice(db, txn, affected, "retain"));
        db.putGeometry(txn, b, edited, db.currentGeometry(txn, b)->uuid, decisions);
        txn.commit();
    }
    require(!stale(), "a retained link is current");
    require(links::linkSurfaceOverlaps(db).edgesSkippedExisting == 1 && edgesOf(db, a).at(0).uuid == edge, "same link after a rerun");
    require(!stale(), "still current");
    { auto txn = db.write(); db.setApproval(txn, edge, "approved"); txn.commit(); }
    const auto approved = [&] { auto txn = db.read(); return db.edgeByUuid(txn, edge)->generation; }();
    links::SurfaceOverlapParams strict; strict.minInliers = 10000;
    require(links::linkSurfaceOverlaps(db, strict).staleRemoved == 0, "reviewed links are never stale-removed");
    require([&] { auto txn = db.read(); return db.edgeByUuid(txn, edge)->generation; }() == approved, "reviewed link untouched");
}

void surfaceSelectionMissingGeometryAndStaleScope(const TemporaryDirectory& directory)
{
    auto db = create(directory, "selection.vcobj");
    const auto a = plane(db), b = plane(db), c = plane(db);
    require(links::linkSurfaceOverlaps(db).edgesCreated == 3, "every pair overlaps");
    links::SurfaceOverlapParams params; params.surfaceUuids = {upper(a), a, b}; params.minInliers = 10000;
    require(links::linkSurfaceOverlaps(db, params).staleRemoved == 1, "only links inside the selection go stale");
    require(edgesOf(db, c).size() == 2, "links outside the selection survive");
    params = {}; params.surfaceUuids = {util::newUuid()};
    mustFail([&] { links::linkSurfaceOverlaps(db, params); });
    ObjectRecord missing; missing.kind = "surface"; missing.storageRef = "/missing/source";
    { auto txn = db.write(); params.surfaceUuids = {db.upsertObject(txn, missing)}; txn.commit(); }
    require(hasProblem(links::linkSurfaceOverlaps(db, params), "no stored geometry"), "missing geometry is a problem");
    require(edgesOf(db, c).size() == 2, "a failed run changes nothing");
}

void storedTrackCrossingsSelectionAndAtomicFailure(const TemporaryDirectory& directory)
{
    auto db = create(directory, "tracks.vcobj");
    std::vector<int32_t> horizontal, vertical, moved;
    for (int32_t i = 0; i < 31; ++i) {
        horizontal.insert(horizontal.end(), {i, 15, 20});
        vertical.insert(vertical.end(), {15, i, 20});
        moved.insert(moved.end(), {50, i, 20});
    }
    const auto a = track(db, "11", horizontal, "horizontal");
    const auto b = track(db, "99", vertical, "vertical");
    track(db, "11", vertical, "vertical", "other-dataset");
    links::TrackTrackParams params; params.workers = 2;
    const auto report = links::linkTracksToTracks(db, params);
    require(report.problems.empty() && report.edgesCreated == 1, "one crossing");
    const auto edge = edgesOf(db, a).at(0);
    require(edge.aAnchor.position == std::array<double, 3>{15, 15, 20} && edge.bAnchor.position == std::array<double, 3>{15, 15, 20},
            "anchors at the crossing");
    params = {}; params.trackUuids = {upper(a), a};
    require(links::linkTracksToTracks(db, params).staleRemoved == 0, "a selection without the partner removes nothing");
    std::vector<uint8_t> bytes(moved.size() * sizeof(int32_t));
    std::memcpy(bytes.data(), moved.data(), bytes.size());
    moveRemovingLinks(db, b, "points", bytes);
    params.trackUuids = {a, b};
    require(links::linkTracksToTracks(db, params).staleRemoved == 0, "nothing left to remove");
    ObjectRecord bad; bad.kind = "track"; bad.track.emplace(); bad.track->datasetRef = "missing"; bad.track->externalKey = "0";
    std::string badUuid;
    { auto txn = db.write(); badUuid = db.upsertObject(txn, bad); txn.commit(); }
    const auto journal = [&] { auto txn = db.read(); return db.journalSince(txn, 0).size(); };
    const auto before = journal();
    require(hasProblem(links::linkTracksToTracks(db), "no stored geometry"), "missing geometry is a problem");
    require(journal() == before, "a failed run writes nothing");
    params = {}; params.trackUuids = {badUuid}; params.datasetRef = "other";
    require(!links::linkTracksToTracks(db, params).problems.empty(), "a selection outside the dataset is a problem");
}

void storedCollectionIdsWindingScalingAndRerun(const TemporaryDirectory& directory)
{
    auto db = create(directory, "collections.vcobj");
    const auto surface = plane(db);
    ObjectRecord record; record.kind = "point_collection";
    record.meta = R"({"source":{"point_attributes":[{"wind_a":3},{"wind_a":4}],"winding_is_absolute":true}})";
    const auto collection = put(db, record, {"point_collection", 1, "source_xyz", "{}", {
        array<double>("points", "<f8", {2, 3}, {10, 10, 40, 20, 20, 40}),
        array<int64_t>("point_ids", "<i8", {2}, {90, 7})}});
    links::PointCollectionSurfaceParams params; params.pointScale = 0.5; params.windowPoints = 3; params.windowMinPoints = 3;
    require(links::linkPointCollectionsToSurfaces(db, params).edgesCreated == 2, "both points link");
    for (const auto& edge : edgesOf(db, collection)) {
        const auto& anchor = edge.aUuid == collection ? edge.aAnchor : edge.bAnchor;
        require(anchor.points.at(0).index.at(0) == 7 || anchor.points.at(0).index.at(0) == 90, "anchors keep point IDs");
        require(anchor.position && (*anchor.position)[2] == 40, "anchors keep source XYZ");
        const auto evidence = Json::parse(edge.meta).at("point_collection_surface");
        require(evidence.at("winding_is_absolute") == true, "winding role retained");
        require(evidence.at("winding_annotation") == 3 || evidence.at("winding_annotation") == 4, "winding annotation retained");
    }
    params = {}; params.pointScale = 0.5;
    require(links::linkPointCollectionsToSurfaces(db, params).edgesUpdated == 2, "a changed window updates the links");
    require(links::linkPointCollectionsToSurfaces(db, params).edgesSkippedExisting == 2, "an unchanged rerun skips them");
    const std::vector<double> far{1000, 1000, 40, 2000, 2000, 40};
    std::vector<uint8_t> bytes(far.size() * sizeof(double));
    std::memcpy(bytes.data(), far.data(), bytes.size());
    moveRemovingLinks(db, collection, "points", bytes);
    params.collectionUuids = {collection}; params.surfaceUuids = {surface};
    require(links::linkPointCollectionsToSurfaces(db, params).staleRemoved == 0, "nothing left to remove");
}

void attachedCollectionsAreNotGraphLinks(const TemporaryDirectory& directory)
{
    auto db = create(directory, "attachments.vcobj");
    const auto surface = plane(db), other = plane(db, 1, "other");
    ObjectRecord record; record.kind = "point_collection"; record.meta = R"({"role":"fit_results"})";
    const auto results = put(db, record, {"point_collection", 1, "source_xyz", "{}", {
        array<double>("points", "<f8", {2, 3}, {10, 10, 20, 20, 20, 20}),
        array<int64_t>("point_ids", "<i8", {2}, {1, 2})}});
    const auto link = [&](const std::string& a, const std::string& b, const std::string& kind, const std::string& meta) {
        EdgeRecord edge; edge.kind = kind; edge.aUuid = a; edge.bUuid = b; edge.meta = meta;
        edge.aAnchor.wholeGeometry = edge.bAnchor.wholeGeometry = true;
        auto txn = db.write(); const auto uuid = db.addEdge(txn, edge); txn.commit(); return uuid;
    };
    const auto attachment = link(results, surface, "attachment", R"({"role":"fit_results"})");

    // The linker leaves role-carrying collections alone, and refuses them by name.
    require(links::linkPointCollectionsToSurfaces(db).edgesCreated == 0, "attached data is not linked");
    require(edgesOf(db, results).size() == 1 && edgesOf(db, results)[0].uuid == attachment, "only the attachment");
    links::PointCollectionSurfaceParams named; named.collectionUuids = {results};
    bool refused = false;
    try { links::linkPointCollectionsToSurfaces(db, named); }
    catch (const std::exception& error) { refused = std::string(error.what()).find("role") != std::string::npos; }
    require(refused, "a role-carrying collection is refused by name");

    // Traversal and graph reads skip attachments unless asked for by kind.
    const auto overlap = link(surface, other, "surface-overlap", "{}");
    auto txn = db.read();
    const auto neighbours = db.neighbours(txn, surface);
    require(std::find(neighbours.begin(), neighbours.end(), results) == neighbours.end(), "attachments are not neighbours");
    require(db.neighbours(txn, surface, std::string("attachment")) == std::vector<std::string>{results}, "unless asked for");
    const auto component = db.componentOf(txn, surface);
    require(std::set<std::string>(component.begin(), component.end()) == std::set<std::string>{surface, other}, "component");
    const auto attached = db.componentOf(txn, surface, {"attachment"});
    require(std::find(attached.begin(), attached.end(), results) != attached.end(), "attachment component by kind");
    ObjectSelection connected; connected.connectedTo = results;
    const auto selected = db.selectObjects(txn, connected);
    require(selected.size() == 1 && selected[0].uuid == results, "attachments do not connect objects");
    require(db.graphReadView(txn).edgeUuids == std::vector<std::string>{overlap}, "graph views skip attachments");
    EdgeSelection attachments; attachments.kind = "attachment";
    require(db.graphReadView(txn, {}, attachments).edgeUuids == std::vector<std::string>{attachment}, "unless asked for");
    // Listings still show the attachment.
    const auto listed = [&](const std::vector<EdgeRecord>& edges) {
        return std::any_of(edges.begin(), edges.end(), [&](const EdgeRecord& edge) { return edge.uuid == attachment; });
    };
    require(listed(db.edgesOf(txn, surface)) && listed(db.selectEdges(txn)), "listings include attachments");
}
} // namespace

int main()
try {
    TemporaryDirectory directory;
    surfaceDiscoveryUsesCurrentRevisionsAndPreservesReview(directory);
    surfaceSelectionMissingGeometryAndStaleScope(directory);
    storedTrackCrossingsSelectionAndAtomicFailure(directory);
    storedCollectionIdsWindingScalingAndRerun(directory);
    attachedCollectionsAreNotGraphLinks(directory);
    std::cout << "object links passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "FAILED: " << error.what() << '\n';
    return 1;
}
