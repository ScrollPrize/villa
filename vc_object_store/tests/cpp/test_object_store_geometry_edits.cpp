#include "TestSupport.hpp"
#include "vc_object_store/objects/GeometryEdits.hpp"
#include "vc_object_store/objects/GeometryCoordinates.hpp"
#include "vc_object_store/objects/ObjectStoreSession.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include <nlohmann/json.hpp>
#include <iostream>
using namespace vc::objects;
using Json = nlohmann::json;
int main() try {
    TemporaryDirectory directory;
    const auto path = directory.path / "edits.vcobj";
    OpenOptions options; options.author = "generic-edit-test";
    { auto store = ObjectStore::create(path, options); }
    ObjectStoreSession session; const auto attachment = session.attach(path, options);
    const auto volume = util::newUuid();
    require(geometryScaleToVolume(Json{{"coordinates",{{"volume_uuid",volume},{"scale_factor",4}}}},volume) == .25,
            "known coordinates convert exactly");
    require(!geometryScaleToVolume(Json{{"coordinates",{{"volume_uuid",volume},{"scale_factor",4}}}},util::newUuid()), "different volume stays unbound");
    require(!geometryScaleToVolume(Json::object(),volume), "unknown coordinates stay unbound");
    mustFail([&] { geometryCoordinates(Json{{"scale_factor",0}}); });
    mustFail([&] { mergeGeometryCoordinates(Json{{"scale_factor",(uint64_t(1)<<60)+1}},Json{{"scale_factor",double(uint64_t(1)<<60)}}); });
    for (const auto& kind : {"fiber", "point_collection", "track", "patch"}) {
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
        } else {
            first = vc::geometry::encode(vc::geometry::Patch{1,2,{1,1},{{1,2,3},{4,5,6}}});
            target = vc::geometry::encode(vc::geometry::Patch{1,2,{1,1},{{1,2,3},{4,5,7}}});
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
        });
        mustFail([&] { session.act(attachment,pins,[&](auto& db,auto& txn) {
            updateGeometryArrays(db,txn,uuid,target,initialGeometry);
        }); });
    }
    std::cout << "generic geometry editing and coordinates passed\n";
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
