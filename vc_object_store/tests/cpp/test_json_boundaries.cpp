#include "TestSupport.hpp"
#include "DraftCheckpoints.hpp"
#include "internal/Anchors.hpp"
#include "internal/JsonFields.hpp"
#include "geometry/SequenceMetadata.hpp"
#include "geometry/GeometryDelta.hpp"
#include "internal/Database.hpp"
#include "internal/ExtensionHistory.hpp"
#include "imports/SourceRelationships.hpp"
#include "vc_object_store/objects/FiberJson.hpp"
#include "vc_object_store/objects/GeometryIngest.hpp"
#include "vc_object_store/objects/PackedStore.hpp"
#include "vc_object_store/objects/TransactionJson.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#ifdef VC_TEST_NATIVE_FORMATS
#include "vc_object_store/objects/CatalogStore.hpp"
#endif
#include <fstream>
#include <iostream>
#include <openssl/sha.h>

using namespace vc::objects;
using Json = nlohmann::json;

namespace {
template<class Decode> void rejectsArray(Decode decode)
{
    try { decode(); }
    catch (const std::invalid_argument& error) {
        require(std::string(error.what()).find("array") != std::string::npos, "reject at array schema admission");
        return;
    }
    throw std::runtime_error("malformed array was admitted");
}
// Check each owning decoder, not merely the generic extraction helper. Empty
// and named-object collections must not silently become empty or value lists.
template<class Decode> void arrayBoundary(Json valid, const char* path, Decode decode)
{
    const Json::json_pointer field(path);
    require(valid.at(field).is_array(), "fixture names an array field");
    decode(valid.dump());
    auto named = Json::object();
    if (!valid.at(field).empty()) named["entry"] = valid.at(field).front();
    for (const auto& bad : std::vector<Json>{named, nullptr, "", 1, true}) {
        auto changed = valid; changed[field] = bad;
        rejectsArray([&] { decode(changed.dump()); });
    }
}

void wireBoundaries(const StoreStatus& status)
{
    arrayBoundary(Json::parse(transactionConflictToJson(TransactionConflict(
        TransactionConflict::Reason::Version, {}, "test"))), "/details",
        [](const auto& text) { transactionConflictFromJson(text); });
    ChangePage page; page.next.storeUuid = status.storeUuid; page.next.generation = status.generation;
    page.commits.push_back(FeedCommit{});
    arrayBoundary(Json::parse(changePageToJson(page)), "/commits",
        [](const auto& text) { changePageFromJson(text); });
    arrayBoundary(Json::parse(changePageToJson(page)), "/commits/0/changes",
        [](const auto& text) { changePageFromJson(text); });
    const auto check = [&](const std::string& text, const char* path, auto decode) {
        arrayBoundary(Json::parse(text), path, decode);
    };
    check(objectQueryResultToJson(status, {}, std::nullopt), "/objects",
        [](const auto& text) { objectQueryResultFromJson(text, nullptr); });
    check(edgeQueryResultToJson(status, {}, std::nullopt), "/edges",
        [](const auto& text) { edgeQueryResultFromJson(text, nullptr); });
    check(objectHistoryToJson(status, {}), "/revisions",
        [](const auto& text) { objectHistoryFromJson(text, nullptr); });
    // Null elements mean missing exact versions; null containers do not.
    check(objectsAtVersionsResultToJson(status, {std::nullopt}), "/revisions",
        [](const auto& text) { objectsAtVersionsResultFromJson(text, nullptr); });
    check(relationshipQueryResultToJson(status, {}), "/relationships",
        [](const auto& text) { relationshipQueryResultFromJson(text, nullptr); });
    for (const auto* path : {"/inputs/objects", "/inputs/relationships"})
        check(selectedInputsToJson(status, {}), path,
            [](const auto& text) { selectedInputsFromJson(text); });
    check(regionQueryResultToJson(status, {}), "/result/runs",
        [](const auto& text) { regionQueryResultFromJson(text, nullptr); });
    check(proximityQueryResultToJson(status, {}), "/hits",
        [](const auto& text) { proximityQueryResultFromJson(text, nullptr); });
    GeometryRevision geometry;
    check(geometryRevisionToJson(geometry), "/arrays",
        [](const auto& text) { geometryRevisionFromJson(text); });
}

void commandBoundaries(ObjectStore& store, const StoreStatus& status)
{
    TransactionRequest request;
    request.requestUuid = util::newUuid(); request.storeUuid = status.storeUuid;
    request.generation = status.generation; request.author = "test";
    ObjectRecord record; record.uuid = util::newUuid(); record.kind = "note";
    request.expectedObjects = {{record.uuid, std::nullopt}};
    request.operations = {CreateObjectOp{objectRecordJson(record)}};
    const auto raw = Json::parse(transactionRequestToJson(request));
    for (int route = 0; route != 4; ++route)
        arrayBoundary(raw, "/operations", [&](const auto& text) {
            if (route == 0) transactionRequestFromJson(text, {});
            if (route == 1) transactionJsonPayloadHashes(text);
            if (route == 2) transactionJsonHasPreparedOperations(text);
            if (route == 3) store.submit(text, {});
        });

    // Invalid nested lists must reject during admission, before missing payload
    // discovery or a saved-receipt retry can return an answer.
    const auto data = vc::geometry::encode(vc::geometry::Fiber{{}, {{0,0,0}, {1,0,0}}, {}});
    const std::vector<std::pair<TransactionOperation, const char*>> cases{
        {PutGeometryOp{record.uuid, data, {}}, "/operations/0/data/arrays"},
        {PutGeometryOp{record.uuid, data, {}}, "/operations/0/relationships"},
        {EditGeometryOp{record.uuid, {}, {}}, "/operations/0/edits"},
        {EditGeometryRectangleOp{}, "/operations/0/relationships"},
        {SetSurfaceChannelsOp{}, "/operations/0/channels"},
        {RecordCatalogTransformsOp{}, "/operations/0/registrations"},
        {IngestGeometryOp{}, "/operations/0/items"},
    };
    // Defaults above need canonical UUIDs even though no operation is applied.
    for (auto [operation, path] : cases) {
        std::visit([&](auto& op) {
            if constexpr (requires { op.objectUuid; }) op.objectUuid = record.uuid;
        }, operation);
        request.operations = {operation};
        const auto j = Json::parse(transactionRequestToJson(request));
        for (const auto& bad : std::vector<Json>{Json::object(), nullptr, ""}) {
            auto changed = j; changed[Json::json_pointer(path)] = bad;
            rejectsArray([&] { transactionJsonPayloadHashes(changed.dump()); });
            rejectsArray([&] { transactionJsonHasPreparedOperations(changed.dump()); });
            rejectsArray([&] { transactionRequestFromJson(changed.dump(), {}); });
            rejectsArray([&] { store.submit(changed.dump(), {}); });
        }
    }
}

void persistedAndTopologyBoundaries(const std::filesystem::path& root)
{
    for (const auto& malformed : std::vector<Json>{Json::array(), nullptr, ""})
        mustFail([&] { detail::jsonArrayField(malformed,"optional",true); });
    detail::DraftDirectory checkpoint(root / "drafts");
    auto manifest = Json{{"format",1},{"sequence",1},{"drafts",Json::object()},{"pending",Json::array()}};
    arrayBoundary(manifest, "/pending", [&](const auto& text) {
        { std::ofstream out(root / "drafts" / "checkpoint.json"); out << text; }
        checkpoint.load();
    });
    for (const auto& bad : std::vector<Json>{Json::object(), nullptr, ""}) {
        mustFail([&] { Anchor anchor; detail::dependenciesFromJson(anchor, {{"points",bad}}); });
        mustFail([&] { detail::unpackAnnotation(nullptr, {{"annotation_chunks",bad},{"byte_length",0}}); });
        rejectsArray([&] { detail::loadGeometryManifest(nullptr, "", Json{{"delta",{{"chunks",bad}}}}.dump(), ""); });
        for (const auto* field : {"branches", "adjacent_branches"}) {
            mustFail([&] { detail::validateSourceTopology("fiber", {{field,bad}}); });
            Json source{{"type","vc3d_fiber"},{"version",4},{"uuid",util::newUuid()},{field,bad}};
            mustFail([&] { decodeFiberJson(source.dump()); });
        }
        mustFail([&] { detail::validateSourceTopology("point_collection", {{"windings_linked",bad}}); });
        mustFail([&] { detail::validateSourceTopology("point_collection", {{"point_attributes",Json::array({Json{{"links",bad}}})}}); });
#ifdef VC_TEST_NATIVE_FORMATS
        ObjectRecord catalog; catalog.meta=Json{{"catalog",{{"files",bad}}}}.dump();
        rejectsArray([&] { catalogReference(catalog); });
#endif
    }
    detail::validateSourceTopology("fiber", Json::object());
    detail::validateSourceTopology("point_collection", {{"point_attributes", Json::array({nullptr,Json::object()})}});
    Anchor anchor; detail::dependenciesFromJson(anchor, {{"points",Json::array()}});
    require(anchor.points.empty(), "valid empty anchor points");
}

void packedDirectoryBoundary(const std::filesystem::path& root)
{
    const auto path=root/"packed.vcobj";
    const auto id=util::newUuid();
    {
        PackedStoreBuilder builder(path);
        const auto member=builder.addDatabase(id);
        {
            std::unique_ptr<sqlite3,decltype(&sqlite3_close)> db(
                detail::openConnection(member,SQLITE_OPEN_READWRITE|SQLITE_OPEN_CREATE,{},false),sqlite3_close);
            detail::exec(db.get(),"CREATE TABLE fixture(id INTEGER PRIMARY KEY)");
        }
        builder.finalize(id);
    }
    require(packedDatabases(path).size()==1,"valid producer directory");
    std::ifstream in(path,std::ios::binary);
    std::vector<uint8_t> original(std::istreambuf_iterator<char>(in),{});
    const auto u64=[&](size_t at) {uint64_t v=0;for(size_t i=0;i<8;++i)v|=uint64_t(original[at+i])<<(8*i);return v;};
    const auto offset=u64(16), length=u64(24);
    const auto directory=Json::parse(original.begin()+offset,original.begin()+offset+length);
    arrayBoundary(directory,"/databases",[&](const std::string& text) {
        auto bytes=original; bytes.resize(offset+text.size());
        std::copy(text.begin(),text.end(),bytes.begin()+offset);
        for(size_t i=0;i<8;++i)bytes[24+i]=uint8_t(uint64_t(text.size())>>(8*i));
        SHA256(reinterpret_cast<const unsigned char*>(text.data()),text.size(),bytes.data()+32);
        const auto changed=root/"changed.vcobj";
        {std::ofstream out(changed,std::ios::binary);out.write(reinterpret_cast<const char*>(bytes.data()),bytes.size());}
        packedDatabases(changed);
    });
}
}

int main() try
{
    TemporaryDirectory dir;
    persistedAndTopologyBoundaries(dir.path);
    packedDirectoryBoundary(dir.path);
    for (auto history : {HistoryMode::Off, HistoryMode::On}) {
        OpenOptions options; options.author = "test"; options.history = history;
        auto store = ObjectStore::create(dir.path / (history == HistoryMode::On ? "on.vcobj" : "off.vcobj"), options);
        const auto status = [&] { auto txn = store.read(); return store.status(txn); }();
        wireBoundaries(status);
        commandBoundaries(store, status);
        {
            auto write = store.write();
            const auto a = util::newUuid(), b = util::newUuid();
            ObjectRecord object; object.kind = "fiber"; object.fiber.emplace();
            object.uuid = a; store.upsertObject(write, object);
            object.uuid = b; store.upsertObject(write, object);
            const auto geometry = vc::geometry::encode(vc::geometry::Fiber{{}, {{0,0,0},{1,0,0}}, {}});
            store.putGeometry(write, a, geometry); store.putGeometry(write, b, geometry);
            EdgeRecord edge; edge.kind = "perpendicular"; edge.aUuid = a; edge.bUuid = b;
            edge.aAnchor.wholeGeometry = edge.bAnchor.wholeGeometry = true;
            store.addEdge(write, edge); write.commit();
            auto read = store.read();
            const std::vector<ObjectVersion> refs{store.currentObjectRevision(read,a)->reference(),store.currentObjectRevision(read,b)->reference()};
            const auto relationships = store.relationshipsAt(read,refs);
            require(relationships.size()==1,"relationship fixture");
            const auto encoded = Json::parse(relationshipQueryResultToJson(store.status(read), relationships));
            relationshipQueryResultFromJson(encoded.dump(),nullptr);
            for (const auto* field : {"a_revision","b_revision"}) {
                auto malformed = encoded;
                malformed["relationships"][0]["relationship"][field] = "[\"obsolete\"]";
                mustFail([&] { relationshipQueryResultFromJson(malformed.dump(),nullptr); });
                malformed["relationships"][0]["relationship"][field] = "not-a-uuid";
                mustFail([&] { relationshipQueryResultFromJson(malformed.dump(),nullptr); });
                SelectedInputs inputs; inputs.relationships = relationships;
                auto selected = Json::parse(selectedInputsToJson(store.status(read),inputs));
                selected["inputs"]["relationships"][0]["relationship"][field] = "[\"obsolete\"]";
                mustFail([&] { selectedInputsFromJson(selected.dump()); });
            }
        }
        {
            auto write = store.write(); auto* db=ObjectStoreImplAccess::conn(write);
            for (const auto& bad : std::vector<Json>{Json::object(),nullptr,""}) {
                detail::MetadataSnapshot snapshot(db);
                snapshot.put(0,Json{{"annotation_chunks",bad},{"byte_length",0}});
                const auto root=snapshot.save();
                rejectsArray([&] { detail::metadataPayloads(db,root,[](const auto&){}); });
            }
            ExtensionTable table; table.name="ext_test_json"; table.extension="test.json";
            table.versioned=true; table.keySpace="objects";
            table.columns={{"owner",ExtensionType::Blob},{"value",ExtensionType::Integer}};
            table.primaryKey={"owner"};
            store.createExtensionTable(write,table);
            ObjectRecord owner; owner.kind="note"; owner.uuid=util::newUuid();
            store.upsertObject(write,owner);
            const std::vector<ExtensionRow> rows{{extensionUuid(owner.uuid),int64_t(1)}};
            store.replaceExtensionRows(write,table.name,owner.uuid,rows);
            const auto meta=Json::parse(store.objectByUuid(write,owner.uuid)->meta);
            for (const auto& bad : std::vector<Json>{Json::object(),nullptr,""}) {
                auto broken=meta; broken[detail::extensionManifestKey][table.name]["chunks"]=bad;
                rejectsArray([&] { detail::restoreCurrentExtensionRows(db,"objects",owner.uuid,broken.dump()); });
            }
        }
        const auto snapshot = [&] {
            auto read = store.read(); JsonExportOptions options; options.selection.kind = "note";
            return Json::parse(store.exportJson(read,options));
        }();
        for (const auto* path : {"/objects", "/edges"})
            arrayBoundary(snapshot, path, [&](const auto& text) {
                auto txn = store.write(); store.importJson(txn, text); txn.commit();
            });
    }
    return 0;
}
catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
