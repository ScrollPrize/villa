#include "TestSupport.hpp"
#include "vc_object_store/objects/FiberJson.hpp"
#include "vc_object_store/objects/ObjectStoreSession.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include <nlohmann/json.hpp>
#include <iostream>
using namespace vc::objects;
using Json = nlohmann::json;
int main() try {
    const auto uuid = util::newUuid();
    Json root{{"type","vc3d_fiber"},{"uuid",uuid},{"version",4},{"name","same name"},
        {"width",5.5},{"tags",{"reviewed"}},
        {"control_points",Json::array({Json{{"position",{1.25,2,3}},{"direction",{1,0,0}},{"tags",{"break"}},{"segment_to_next",{{"goal","global"}}}},
                                      Json{{"position",{5.25,2,3}}}})},
        {"line_points",{{1.25,2,3},{5.25,2,3}}}, {"hv_classification",{{"manual_tag","H"}}}};
    const auto doc = decodeFiberJson(root.dump());
    require(doc.object.uuid == uuid && doc.object.fiber->length == 4, "native summaries");
    require(doc.object.fiber->family == "H" && doc.object.tags == std::vector<std::string>{"reviewed"}, "native metadata");
    const auto source = Json::parse(doc.source);
    require(!source.contains("control_points") && !source.contains("line_points"), "no duplicate geometry in annotations");
    require(Json::parse(encodeFiberJson(uuid, doc.geometry, doc.source)) == root, "lossless shared native codec");
    auto invalid = root; invalid["control_points"][0]["position"] = {1,2,3,4};
    mustFail([&] { decodeFiberJson(invalid.dump()); });
    invalid = source; invalid["control_points_attributes"] = Json::array();
    mustFail([&] { encodeFiberJson(uuid, doc.geometry, invalid.dump()); });
    TemporaryDirectory dir;
    const auto path = dir.path / "receipt.vcobj";
    OpenOptions options; options.author = "receipt-test";
    { auto db = ObjectStore::create(path, options); }
    ObjectStoreSession session; const auto attachment = session.attach(path, options);
    const auto receipt = session.act(attachment, {{uuid,{}}}, [&](auto& db, auto& txn) {
        auto object = doc.object; object.meta = Json{{"source",source}}.dump();
        db.upsertObject(txn, object); db.putGeometry(txn, uuid, doc.geometry);
    });
    require(receipt.size() == 1 && receipt.at(uuid).has_value(), "commit returns exact new pin");
    auto external = ObjectStore::open(path, options);
    { auto txn = external.write(receipt); external.setMeta(txn,uuid,"external","true"); txn.commit();
      require(txn.committedRevisions().at(uuid) != receipt.at(uuid), "external commit produces different pin"); }
    session.read(attachment,[&](const auto& db,const auto& txn,const auto&) {
        require(db.objectRevision(txn,*receipt.at(uuid))->record.meta.find("external") == std::string::npos, "receipt is immutable after external commit");
        const auto historical=Json::parse(fiberRevisionJson(db,txn,*receipt.at(uuid)));
        require(historical["control_points"]==root["control_points"] && historical["line_points"]==root["line_points"], "pinned wire geometry and annotations");
        require(!historical.contains("optimization_mode"), "absent legacy mode stays absent on the wire");
        mustFail([&] {fiberRevisionJson(db,txn,*receipt.at(uuid),1);});
    });
    mustFail([&] { session.act(attachment,receipt,[](auto&,auto&) {}); });
    const auto a = util::newUuid(), b = util::newUuid(), isolated = util::newUuid();
    std::string ordinary, adjacent;
    const auto networkPins = session.act(attachment, {{a,{}},{b,{}},{isolated,{}}}, [&](auto& db, auto& txn) {
        vc::geometry::Fiber geometry{{{1,2,3},{5,2,3}}, {{1,2,3},{5,2,3}}, {90,10}};
        for (const auto& objectUuid : {a,b,isolated}) {
            ObjectRecord record; record.uuid = objectUuid; record.kind = "fiber";
            record.name = "same name"; record.formatVersion = 4;
            record.meta = Json{{"coordinates",{{"volume_uuid",uuid},{"scale_factor",4.0}}}}.dump();
            db.upsertObject(txn, record); db.putGeometry(txn, objectUuid, vc::geometry::encode(geometry));
        }
        EdgeRecord edge; edge.kind = "fiber_branch"; edge.aUuid = a; edge.bUuid = b;
        edge.aAnchor.points = {{"control_point_ids", {10}}}; edge.bAnchor.points = {{"control_point_ids", {90}}};
        edge.aAnchor.position = {5,2,3}; edge.bAnchor.position = {1,2,3};
        edge.aAnchor.direction = {1,0,0}; edge.bAnchor.direction = {0,1,0};
        ordinary = db.addEdge(txn, edge); db.setApproval(txn, ordinary, "approved");
        edge.meta = R"({"adjacent":true})";
        edge.aAnchor.points = {{"control_point_ids", {90}}}; edge.aAnchor.position = {1,2,3};
        edge.bAnchor.points = {{"control_point_ids", {10}}}; edge.bAnchor.position = {5,2,3};
        adjacent = db.addEdge(txn, edge); db.setApproval(txn, adjacent, "pending");
    });
    std::map<std::string,std::string> exported;
    session.read(attachment, [&](const auto& db, const auto& txn, const auto&) {
        exported = fiberNetworkJson(db, txn, networkPins, {a});
        require(exported.size() == 2 && exported.contains(b), "export follows linked component only");
        const auto first = Json::parse(exported.at(a)), second = Json::parse(exported.at(b));
        require(first["name"] == second["name"] && first["filename"] != second["filename"], "UUID wire names preserve duplicate labels");
        const auto& link = first["branches"].at(0);
        require(link["control_point_index"] == 1 && link["branch_control_point_index"] == 0, "stable control IDs resolve to selected indices");
        require(link["branch_file"] == b+".json" && link["edge_uuid"] == ordinary && !link["pending"].get<bool>(), "ordinary branch exported from edge");
        require(second["branches"][0]["branch_file"] == a+".json" && second["branches"][0]["control_point_index"] == 0, "reciprocal orientation exported");
        require(first["adjacent_branches"][0]["edge_uuid"] == adjacent && first["adjacent_branches"][0]["pending"].get<bool>(), "adjacent pending edge preserved separately");
        require(first["coordinates"]["scale_factor"] == 4.0, "coordinate binding exported");
        mustFail([&] { fiberNetworkJson(db,txn,{{a,networkPins.at(a)}}); });
        mustFail([&] { fiberNetworkJson(db,txn,{{a,{}}}); });
        mustFail([&] { fiberNetworkJson(db,txn,networkPins,{a},1); });
    });
    session.act(attachment, networkPins, [&](auto& db, auto& txn) { db.removeEdge(txn,ordinary); });
    session.read(attachment, [&](const auto& db, const auto& txn, const auto&) {
        require(fiberNetworkJson(db,txn,networkPins,{a}) == exported, "pinned topology survives later edge deletion");
        auto current = networkPins;
        for (auto& [objectUuid, pin] : current) pin = db.currentObjectRevision(txn,objectUuid)->uuid;
        const auto now = fiberNetworkJson(db,txn,current,{a});
        require(Json::parse(now.at(a))["branches"].empty(), "current export observes edge deletion");
        require(Json::parse(now.at(a))["adjacent_branches"].size() == 1, "parallel adjacent edge survives");
    });
    std::cout << "fiber codec and checked commit receipt passed\n";
} catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
