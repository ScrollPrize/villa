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
    std::cout << "fiber codec and checked commit receipt passed\n";
} catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
