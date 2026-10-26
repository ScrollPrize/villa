#include "TestSupport.hpp"
#include "vc_object_store/objects/GeometryEdits.hpp"
#include "vc_object_store/objects/GeometryCoordinates.hpp"
#include "vc_object_store/objects/ObjectStoreSession.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include "vc_object_store/store/GeometryViews.hpp"
#include "vc_object_store/geometry/ArrayBytes.hpp"
#include <tuple>
#include <nlohmann/json.hpp>
#include <iostream>
#include <cstring>
#include <optional>
#include <sqlite3.h>
using namespace vc::objects;
using Json = nlohmann::json;
int main() try {
    TemporaryDirectory directory;
    const auto path = directory.path / "edits.vcobj";
    OpenOptions options; options.author = "generic-edit-test";
    { auto store = ObjectStore::create(path, options); }
    ObjectStoreSession session; const auto attachment = session.attach(path, options);
    const std::string volume = "20260411134726";
    require(geometryCoordinates(Json{{"volume_uuid",volume}})["volume_uuid"] == volume,
            "catalog volume ID is stored verbatim");
    require(geometryCoordinates(Json{{"volume_uuid","MixedCase-ID"}})["volume_uuid"] == "MixedCase-ID",
            "volume identities are not canonicalized as object UUIDs");
    mustFail([&] { geometryCoordinates(Json{{"volume_uuid",""}}); });
    mustFail([&] { geometryCoordinates(Json{{"volume_uuid",20260411134726LL}}); });
    require(!geometryScaleToVolume(Json{{"coordinates",{{"volume_uuid",volume},{"scale_factor",4}}}},
                                  "2d2a9b21-9c82-5343-9443-57838a5fa5a1"),
            "derived UUID does not alias authoritative volume ID");
    require(geometryScaleToVolume(Json{{"coordinates",{{"volume_uuid",volume},{"scale_factor",4}}}},volume) == .25,
            "known coordinates convert exactly");
    require(!geometryScaleToVolume(Json{{"coordinates",{{"volume_uuid",volume},{"scale_factor",4}}}},util::newUuid()), "different volume stays unbound");
    require(!geometryScaleToVolume(Json::object(),volume), "unknown coordinates stay unbound");
    mustFail([&] { geometryCoordinates(Json{{"scale_factor",0}}); });
    mustFail([&] { mergeGeometryCoordinates(Json{{"scale_factor",(uint64_t(1)<<60)+1}},Json{{"scale_factor",double(uint64_t(1)<<60)}}); });
    for (const auto& kind : {"fiber", "point_collection", "track", "surface", "umbilicus"}) {
        const auto uuid = util::newUuid();
        GeometryData first, target;
        if (std::string(kind) == "fiber") {
            first = vc::geometry::encode(vc::geometry::Fiber{{{1,2,3},{4,5,6}},{{1,2,3},{4,5,6}},{10,20}});
            target = vc::geometry::encode(vc::geometry::Fiber{{{4,5,7},{8,9,10},{1,2,3}},{{4,5,7},{1,2,3}},{20,30,10}});
        } else if (std::string(kind) == "point_collection") {
            first = vc::geometry::encode(vc::geometry::PointCollection{{10,20},{{1,2,3},{4,5,6}}});
            target = vc::geometry::encode(vc::geometry::PointCollection{{20,30,10},{{4,5,7},{8,9,10},{1,2,3}}});
        } else if (std::string(kind) == "track") {
            first = vc::geometry::encode(vc::geometry::Track{{{1,2,3},{4,5,6}}});
            target = vc::geometry::encode(vc::geometry::Track{{{1,2,3},{4,5,7},{8,9,10}}});
        } else if (std::string(kind) == "umbilicus") {
            first = vc::geometry::encode(vc::geometry::Umbilicus{{{1,2,3},{4,5,6}}});
            target = vc::geometry::encode(vc::geometry::Umbilicus{{{1,2,3},{4,5,7},{8,9,10}}});
        } else {
            first = vc::geometry::encode(vc::geometry::Surface{1,2,{1,1},{{1,2,3},{4,5,6}}});
            target = vc::geometry::encode(vc::geometry::Surface{1,2,{1,1},{{1,2,3},{4,5,7}}});
        }
        const auto annotated = std::string(kind) == "fiber" || std::string(kind) == "point_collection";
        const auto array = std::string(kind) == "fiber" ? "control_points" : "points";
        std::string initialGeometry, editedGeometry;
        auto pins = session.act(attachment, {{uuid,{}}}, [&](auto& db, auto& txn) {
            ObjectRecord object; object.uuid = uuid; object.kind = kind;
            if (annotated) object.meta = Json{{"source",{{std::string(kind)=="fiber"?"control_points_attributes":"point_attributes",
                Json::array({Json{{"note","first"}},Json{{"note","second"}}})}}}}.dump();
            db.upsertObject(txn,object); initialGeometry = db.putGeometry(txn,uuid,first);
        });
        pins = session.act(attachment,pins,[&](auto& db,auto& txn) {
            editedGeometry = updateGeometryArrays(db,txn,uuid,target,initialGeometry);
            require(updateGeometryArrays(db,txn,uuid,target,editedGeometry) == editedGeometry, "identical update is a no-op");
        });
        session.read(attachment,[&](const auto& db,const auto& txn,const auto&) {
            for (const auto& desired : target.arrays)
                require(db.readGeometryArray(txn,editedGeometry,desired.name) == desired.bytes,"generic edit preserves exact bytes");
            for (const auto& original : first.arrays)
                require(db.readGeometryArray(txn,initialGeometry,original.name) == original.bytes,"historical arrays are immutable");
            if (annotated) require(Json::parse(db.geometryPointAnnotations(txn,editedGeometry,array)) ==
                Json::array({Json{{"note","second"}},Json::object(),Json{{"note","first"}}}),"annotations follow stable IDs through splice");
            // Typed views over the session's own transaction, at a pinned or the current revision.
            GeometrySnapshot inSession(db, txn);
            if (std::string(kind) == "fiber") {
                const auto fiber = inSession.fiber(uuid);
                require(fiber.controlPointIds(0, fiber.controlPointCount()) == std::vector<int64_t>{20, 30, 10},
                        "fiber views read stable control point IDs");
            }
            const auto encoded = [&](const std::optional<std::string>& revision) {
                return std::visit([](const auto& view) { return vc::geometry::encode(view.load()); },
                                  inSession.geometry(uuid, revision));
            };
            for (const auto& [revision, expected] : {std::pair{std::optional(initialGeometry), &first},
                                                     std::pair{std::optional<std::string>{}, &target}}) {
                const auto loaded = encoded(revision);
                require(loaded.arrays.size() == expected->arrays.size(), "pinned view loads the revision's arrays");
                for (size_t i = 0; i < loaded.arrays.size(); ++i)
                    require(loaded.arrays[i].bytes == expected->arrays[i].bytes, "pinned view loads the revision exactly");
            }
        });
        mustFail([&] { session.act(attachment,pins,[&](auto& db,auto& txn) {
            updateGeometryArrays(db,txn,uuid,target,initialGeometry);
        }); });
    }
    {
        // Packed reads decode per item with the public byte helpers.
        const auto a = util::newUuid(), b = util::newUuid();
        std::string ra, rb;
        session.act(attachment, {{a,{}},{b,{}}}, [&](auto& db, auto& txn) {
            for (const auto& [uuid, revision, points] : {std::tuple{a, &ra, std::vector<vc::geometry::Point>{{1,2,3}}},
                                                       std::tuple{b, &rb, std::vector<vc::geometry::Point>{{4,5,6},{7,8,9}}}}) {
                ObjectRecord object; object.uuid = uuid; object.kind = "umbilicus";
                db.upsertObject(txn, object);
                *revision = db.putGeometry(txn, uuid, vc::geometry::encode(vc::geometry::Umbilicus{points}));
            }
        });
        session.read(attachment, [&](const auto& db, const auto& txn, const auto&) {
            const auto packed = db.readGeometryArrays(txn, {ra, rb}, "points");
            require(xyzFromBytes<double>(packedItem(packed, 1)) == std::vector<std::array<double, 3>>{{4,5,6},{7,8,9}} &&
                    xyzFromBytes<double>(packedItem(packed, 0)) == std::vector<std::array<double, 3>>{{1,2,3}},
                    "packed items decode to their points");
            mustFail([&] { (void)packedItem(packed, 2); });
            ObjectSelection umbilici; umbilici.uuids = {a, b};
            const auto heads = db.currentRevisions(txn, umbilici);
            require(heads.size() == 2 && *heads.at(a) == db.currentObjectRevision(txn, a)->uuid &&
                    *heads.at(b) == db.currentObjectRevision(txn, b)->uuid, "current revisions pin a selection");
        });
    }
    // Auxiliary channels and non-finite payload bits belong to the generic
    // store schema even when owned, typed geometry cannot represent them.
    {
        const auto id = util::newUuid();
        auto first = vc::geometry::encode(vc::geometry::Surface{1,2,{1,1},{{1,2,3},{4,5,6}}});
        first.arrays.push_back({"review", "<u2", {1,2}, {1,0,2,0}});
        first.arrays[0].bytes[0] = 0x45; first.arrays[0].bytes[1] = 0x23;
        first.arrays[0].bytes[2] = 0xc1; first.arrays[0].bytes[3] = 0x7f; // NaN payload
        std::string initial, updated;
        auto pins = session.act(attachment, {{id,{}}}, [&](auto& db, auto& txn) {
            ObjectRecord object; object.uuid = id; object.kind = "surface";
            db.upsertObject(txn, object); initial = db.putGeometry(txn, id, first);
        });
        auto target = first; target.arrays.back().bytes = {3,0,4,0};
        pins = session.act(attachment, pins, [&](auto& db, auto& txn) {
            updated = updateGeometryArrays(db, txn, id, target, initial);
            require(db.readGeometryArray(txn, updated, "review") == target.arrays.back().bytes, "auxiliary edits use sparse store updates");
            require(db.readGeometryArray(txn, updated, "points") == first.arrays[0].bytes, "exact non-finite bits survive channel edits");
            require(db.readGeometryArray(txn, initial, "review") == first.arrays.back().bytes, "channel history remains immutable");
        });
        auto malformed = target; malformed.arrays.back().bytes.pop_back();
        mustFail([&] { session.act(attachment, pins, [&](auto& db, auto& txn) {
            updateGeometryArrays(db, txn, id, malformed, updated);
        }); });
    }
    // Surfaces larger than any whole-read budget update in place; a changed run
    // that crosses a read window stays one sparse edit.
    {
        const auto id = util::newUuid();
        const int64_t side = 2400; // 66 MiB of float32 XYZ
        vc::geometry::Surface surface{side, side, {1, 1}, {}};
        surface.points.reserve(size_t(side * side));
        for (int64_t r = 0; r < side; ++r)
            for (int64_t c = 0; c < side; ++c) surface.points.push_back({float(c), float(r), 7});
        auto first = vc::geometry::encode(surface);
        std::string initial, updated;
        auto pins = session.act(attachment, {{id,{}}}, [&](auto& db, auto& txn) {
            ObjectRecord object; object.uuid = id; object.kind = "surface";
            db.upsertObject(txn, object); initial = db.putGeometry(txn, id, first);
        });
        const size_t window = 1024 * 1024 / 12; // the vertex containing the first window boundary
        for (size_t i = window - 3; i < window + 3; ++i) surface.points[i][2] = 9;
        const auto target = vc::geometry::encode(surface);
        pins = session.act(attachment, pins, [&](auto& db, auto& txn) {
            updated = updateGeometryArrays(db, txn, id, target, initial);
        });
        session.read(attachment, [&](const auto& db, const auto& txn, const auto&) {
            require(db.readGeometryArray(txn, updated, "points") == target.arrays[0].bytes, "large surfaces update exactly");
            require(db.readGeometryArray(txn, initial, "points") == first.arrays[0].bytes, "large surface history is immutable");
        });
    }
    // Structural safety must survive the convenience edit API, including a
    // caller catching the error and trying to commit the surrounding action.
    const auto uuid = util::newUuid();
    vc::geometry::Fiber original{{{0,0,0},{1,0,0}},{{0,0,0},{1,0,0}},{10,20}};
    std::string initial;
    auto pins = session.act(attachment, {{uuid,{}}}, [&](auto& db, auto& txn) {
        ObjectRecord object; object.uuid = uuid; object.kind = "fiber";
        object.meta = R"({"source":{"control_points_attributes":[{"segment_to_next":{"direction":[1,0,0]}},{}],"line_points_attributes":[{"label":"sample"},{}]}})";
        db.upsertObject(txn, object); initial = db.putGeometry(txn, uuid, vc::geometry::encode(original));
        require(!Json::parse(db.objectByUuid(txn, uuid)->meta).at("source").contains("control_points_attributes"), "creation consumes initial annotations");
    });
    auto reverse = original;
    std::reverse(reverse.controlPoints.begin(), reverse.controlPoints.end());
    std::reverse(reverse.controlPointIds.begin(), reverse.controlPointIds.end());
    bool rejected = false;
    mustFail([&] { session.act(attachment, pins, [&](auto& db, auto& txn) {
        try { updateGeometryArrays(db, txn, uuid, vc::geometry::encode(reverse), initial); }
        catch (const std::exception&) { rejected = true; }
    }); });
    require(rejected, "directional reorder requires an explicit transform");
    auto partial = original;
    partial.controlPoints[0][0] = 9; // This sparse update precedes the failing resize.
    partial.linePoints.push_back({2,0,0});
    rejected = false;
    mustFail([&] { session.act(attachment, pins, [&](auto& db, auto& txn) {
        try { updateGeometryArrays(db, txn, uuid, vc::geometry::encode(partial), initial); }
        catch (const std::exception&) { rejected = true; }
        require(db.currentGeometry(txn, uuid)->uuid == initial, "failed helper rolls back every array");
    }); });
    require(rejected, "annotated sample resize requires an explicit transform");
    session.read(attachment, [&](const auto& db, const auto& txn, const auto&) {
        require(db.currentGeometry(txn, uuid)->uuid == initial, "caught failures cannot partially commit");
    });
    session.act(attachment, pins, [&](auto& db, auto& txn) {
        const auto head = updateGeometryArrays(db, txn, uuid, vc::geometry::encode(reverse), initial,
            {{"control_points", R"([{"segment_to_next":{"direction":[-1,0,0]}},{}])"}});
        require(Json::parse(db.geometryPointAnnotations(txn, head, "control_points"))[0]["segment_to_next"]["direction"][0] == -1,
            "explicit directional transform is preserved");
        require(Json::parse(db.geometryPointAnnotations(txn, initial, "control_points"))[0]["segment_to_next"]["direction"][0] == 1,
            "initial annotations remain immutable");
    });
    {
        // An edit too large for one sparse delta is published as a full revision
        // that shares unchanged chunks; history and later deltas still work.
        const auto bigPath = directory.path / "large-edits.vcobj";
        OpenOptions small = options; small.maxGeometryDeltaBytes = 64 * 1024;
        std::optional<ObjectStore> opened(ObjectStore::create(bigPath, small));
        auto& store = *opened;
        ObjectStore* active = &store;
        auto deltas = [&] {
            sqlite3* raw = nullptr; sqlite3_open_v2(bigPath.c_str(), &raw, SQLITE_OPEN_READONLY, nullptr);
            sqlite3_stmt* st = nullptr; sqlite3_prepare_v2(raw, "SELECT count(*) FROM geometry_deltas", -1, &st, nullptr);
            sqlite3_step(st); const auto n = sqlite3_column_int64(st, 0); sqlite3_finalize(st); sqlite3_close(raw); return n;
        };
        vc::geometry::Surface surface{300, 300, {1, 1}, {}};
        for (int r = 0; r < 300; ++r) for (int c = 0; c < 300; ++c) surface.points.push_back({float(c), float(r), 5});
        std::string uuid, head;
        { auto txn = store.write(); ObjectRecord record; record.kind = "surface"; uuid = store.upsertObject(txn, record);
          head = store.putGeometry(txn, uuid, vc::geometry::encode(surface)); txn.commit(); }
        auto rectangle = [&](int row, int rows, float z) {
            std::vector<uint8_t> bytes;
            for (int r = row; r < row + rows; ++r) for (int c = 0; c < 300; ++c) {
                const std::array<float, 3> p{float(c), float(r), z + 0.001f * float(r * 300 + c)};
                const auto* b = reinterpret_cast<const uint8_t*>(p.data()); bytes.insert(bytes.end(), b, b + 12);
            }
            auto txn = store.write();
            head = store.editGeometryRectangle(txn, uuid, "points", row, 0, rows, 300, bytes, head);
            txn.commit();
            return head;
        };
        auto vertex = [&](const std::string& revision, int row, int col) {
            auto txn = active->read();
            const auto bytes = active->readGeometryRectangle(txn, revision, "points", row, col, 1, 1);
            std::array<float, 3> p; std::memcpy(p.data(), bytes.data(), 12); return p;
        };
        const auto first = rectangle(10, 1, 6) ;
        require(deltas() == 1, "a small edit is a sparse delta");
        const auto full = rectangle(0, 250, 7);
        require(deltas() == 1, "an edit over the delta limit is a full revision");
        require(vertex(full, 249, 299)[2] == 7 + 0.001f * float(249 * 300 + 299) && vertex(full, 260, 3)[2] == 5,
                "the full revision holds the edit and keeps unchanged chunks");
        require(vertex(first, 10, 7)[2] == 6 + 0.001f * float(10 * 300 + 7) && vertex(first, 200, 7)[2] == 5,
                "history before the full revision still reads");
        const auto after = rectangle(299, 1, 9);
        require(deltas() == 2 && vertex(after, 299, 0)[2] == 9 + 0.001f * float(299 * 300) &&
                vertex(after, 249, 299)[2] == vertex(full, 249, 299)[2], "later edits are deltas on the full revision");
        {
            auto txn = store.read();
            RegionSlice cut; cut.origin = {150, 100, 7.1}; cut.localBounds = {-200, -200, -1, 200, 200, 1};
            RegionOptions surfaces; surfaces.kind = "surface";
            require(store.geometryRegion(txn, cut, surfaces).intersectingObjects == 1, "bounds follow the full revision");
        }
        opened.reset();
        auto reopened = ObjectStore::open(bigPath, small);
        active = &reopened;
        require(vertex(full, 0, 0)[2] == 7.0f && vertex(first, 10, 0)[2] == 6 + 0.001f * float(10 * 300), "revisions survive reopening");
    }
    {
        // A linked seed fiber is reoptimized: its linked control point gets a new
        // ID and the line is resampled. Once the control-point edit keeps the
        // edge as recorded, the line-point edit must not ask about it again.
        const auto seed = util::newUuid(), parent = util::newUuid();
        const vc::geometry::Fiber single{{{5,5,5}},{{5,5,5}},{0}};
        std::string seedHead;
        auto linked = session.act(attachment, {{seed,{}},{parent,{}}}, [&](auto& db, auto& txn) {
            for (const auto& id : {seed, parent}) {
                ObjectRecord object; object.uuid = id; object.kind = "fiber"; db.upsertObject(txn, object);
                const auto head = db.putGeometry(txn, id, vc::geometry::encode(single));
                if (id == seed) seedHead = head;
            }
            EdgeRecord edge; edge.kind = "fiber_branch"; edge.aUuid = seed; edge.bUuid = parent;
            edge.aAnchor.points = {{"control_point_ids", {0}}}; edge.bAnchor.points = {{"control_point_ids", {0}}};
            db.addEdge(txn, edge);
        });
        int asked = 0;
        session.act(attachment, linked, [&](auto& db, auto& txn) {
            const auto head = updateGeometryArrays(db, txn, seed,
                vc::geometry::encode(vc::geometry::Fiber{{{4,5,5},{5,5,6},{6,5,5}},{{4,5,5},{6,5,5}},{1,3,2}}), seedHead, {},
                [&](const std::vector<EdgeRecord>& affected) {
                    std::vector<GeometryEdgeEdit> result;
                    for (const auto& edge : affected) {
                        GeometryEdgeEdit edit; edit.edgeUuid = edge.uuid; edit.expectedGeneration = edge.generation;
                        edit.action = "keep_recorded";
                        edit.aRevision = db.currentGeometry(txn, edge.aUuid)->uuid;
                        edit.bRevision = db.currentGeometry(txn, edge.bUuid)->uuid;
                        result.push_back(std::move(edit)); ++asked;
                    }
                    return result;
                });
            require(db.readGeometryData(txn, head).arrays.size() == 3, "reoptimized seed is published");
        });
        require(asked == 1, "the replaced linked point asks once");
    }
    std::cout << "generic geometry editing and coordinates passed\n";
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
