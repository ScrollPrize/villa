#include "TestSupport.hpp"
#include "vc_object_store/objects/FiberEdit.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include <iostream>

using namespace vc::objects;
int main()
{
    try {
        TemporaryDirectory temp;
        OpenOptions options; options.author = "fiber-edit-test";
        const auto path = temp.path / "objects.vcobj";
        { auto db = ObjectStore::create(path, options); }
        ObjectStoreSession session;
        auto attachment = session.attach(path, options);
        FiberEdit fiber;
        fiber.uuid = util::newUuid(); fiber.name = "fiber";
        fiber.controlPoints = {{{0,0,0},-1},{{2,0,0},-1}};
        fiber.linePoints = {{0,0,0},{1,0,0},{2,0,0}};
        fiber.record.uuid = fiber.uuid; fiber.record.kind = "fiber";
        fiber.record.formatVersion = 4;
        fiber.sourceAnnotations = R"({"version":4})";
        std::vector<FiberEdit> edits{fiber};
        publishFiberEdits(session,attachment,edits,{},{});
        require(edits[0].storePin.has_value(),"publication returns committed pin");
        require(edits[0].controlPoints[0].storeId == 0 && edits[0].controlPoints[1].storeId == 1,"publication assigns stable control IDs");
        const auto original = edits[0];
        const auto pin = *original.storePin;
        session.act(attachment,{{fiber.uuid,pin.objectRevision}},[&](auto& db,auto& txn) { db.setMeta(txn,fiber.uuid,"external","true"); });
        edits[0].controlPoints.push_back({{3,0,0},-1});
        mustFail([&] { publishFiberEdits(session,attachment,edits,{},{}); });
        require(edits[0].storePin == original.storePin && edits[0].controlPoints.back().storeId == -1,"conflict leaves caller payload unchanged");
        session.read(attachment,[&](const auto& db,const auto& txn,const auto&) {
            const auto geometry = std::get<vc::geometry::Fiber>(vc::geometry::decode(db.readGeometryData(txn,*pin.geometryRevision)));
            require(geometry.linePoints == fiber.linePoints && geometry.controlPoints.size() == 2,"historical geometry is unchanged");
        });
        options.readOnly = true; attachment = session.attach(path,options);
        edits = {original}; edits[0].storePin->attachment = attachment;
        mustFail([&] { publishFiberEdits(session,attachment,edits,{},{}); });
        std::cout << "fiber edit publication passed\n";
    } catch(const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
