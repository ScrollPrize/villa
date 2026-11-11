// Selection records: binary round trip and validation, local capture with the
// recorded transforms a volume selection used, storage as an artifact,
// historical read-back, refusal of unrecorded catalog registrations, hosted
// capture equal to local, and hosted records that stay exact while a writer
// keeps changing the transform that decides membership.
#include "TestSupport.hpp"
#include "vc_object_store/client/HostedStore.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include "vc_object_store/objects/ObjectStore.hpp"
#include "vc_object_store/objects/ObjectCollection.hpp"
#include "vc_object_store/objects/Selections.hpp"
#include "vc_object_store/objects/TransactionJson.hpp"
#include "vc_object_store/server/Server.hpp"

#include <atomic>
#include <fstream>
#include <iostream>
#include <set>
#include <thread>

#include <nlohmann/json.hpp>

using namespace vc::objects;

namespace {
const std::string kSource = "20260101000000", kTarget = "20260202000000";

template<class F> bool rejects(F f)
{
    try { f(); } catch (const std::invalid_argument&) { return true; }
    return false;
}

std::set<std::string> uuidsOf(const std::vector<ObjectVersion>& versions)
{
    std::set<std::string> result;
    for (const auto& v : versions) result.insert(v.objectUuid);
    return result;
}
} // namespace

int main()
try {
    TemporaryDirectory directory;
    const auto path = directory.path / "selections.vcobj";
    OpenOptions options; options.author = "selections-test";
    auto store = std::make_unique<ObjectStore>(ObjectStore::create(path, options));
    const auto initial = [&] { auto txn = store->read(); return store->status(txn); }();
    auto request = [&](VersionSelection expected, std::vector<TransactionOperation> operations) {
        TransactionRequest r;
        r.requestUuid = util::newUuid(); r.storeUuid = initial.storeUuid; r.generation = initial.generation;
        r.author = "alice"; r.expectedObjects = testNumberedGuards(*store, expected); r.operations = std::move(operations);
        return r;
    };

    // Fibers in the source volume along x at y = 10 * row; the registration
    // into the target scales by 2 or 3, which decides which rows fall in the
    // target box y in [15, 45]: rows 1 and 2 at scale 2, row 1 at scale 3.
    std::vector<std::string> rows;
    for (int row = 1; row <= 4; ++row) {
        ObjectRecord record; record.uuid = util::newUuid(); record.kind = "fiber"; record.fiber.emplace();
        record.meta = nlohmann::json{{"coordinates", {{"volume_uuid", kSource}, {"scale_factor", 1.0}}}}.dump();
        vc::geometry::Fiber fiber; fiber.controlPoints = {{0, 10.0 * row, 0}, {10, 10.0 * row, 0}};
        store->submit(request({{record.uuid, std::nullopt}},
            {CreateObjectOp{objectRecordJson(record)}, PutGeometryOp{record.uuid, vc::geometry::encode(fiber), {}}}));
        rows.push_back(record.uuid);
    }
    vcr::catalog::Registration scan{"scan", "v1", kSource, kTarget, vcr::catalog::Affine::scale(2), "test"};
    const auto transformUuid = catalogTransformUuid(scan.id);
    store->submit(request({{transformUuid, std::nullopt}}, {RecordCatalogTransformsOp{{scan}}}));

    ObjectSelection boxed;
    boxed.kind = "fiber"; boxed.volume = kTarget; boxed.bbox = std::array<double, 6>{-1, 15, -1, 100, 45, 1};
    SelectionRecord captured;
    {
        auto txn = store->read();
        captured = recordSelection(*store, txn, boxed);
        const auto status = store->status(txn);
        require(captured.storeUuid == status.storeUuid && captured.generation == status.generation &&
                captured.commitSequence == status.commitSequence, "record names the snapshot");
        require(uuidsOf(captured.objects) == std::set<std::string>{rows[0], rows[1]}, "membership at scale 2");
        for (const auto& object : captured.objects) require(object.version == 1, "fiber versions (one creating transaction)");
        require(captured.transforms.size() == 1 && captured.transforms[0].objectUuid == transformUuid &&
                captured.transforms[0].version == 1, "the recorded transform it used");
        require(objectQueryRequestFromJson(captured.queryJson, nullptr).volume == kTarget, "query kept");

        ObjectSelection plain; plain.kind = "fiber";
        const auto all = recordSelection(*store, txn, plain);
        require(all.objects.size() == 4 && all.transforms.empty(), "non-volume selections need no transforms");
        auto limited = plain; limited.limit = 2;
        require(rejects([&] { recordSelection(*store, txn, limited); }), "limit refused");
        auto explicitTransforms = boxed;
        explicitTransforms.transforms = std::make_shared<VolumeTransforms>(store->volumeTransforms(txn));
        require(rejects([&] { recordSelection(*store, txn, explicitTransforms); }), "explicit transforms refused");
    }
    {
        auto txn = store->write();
        require(rejects([&] { recordSelection(*store, txn, boxed); }), "write transactions refused");
    }

    // Binary form: 24 bytes per object after the header, exact round trip.
    const auto bytes = encodeSelectionRecord(captured);
    {
        uint32_t header = 0;
        for (int i = 0; i < 4; ++i) header |= uint32_t(bytes[8 + i]) << (8 * i);
        require(bytes.size() == 12 + header + 24 * captured.objects.size(), "24 bytes per entry");
        require(decodeSelectionRecord(bytes) == captured, "round trip");
        auto corrupt = bytes; corrupt[0] = 'X';
        require(rejects([&] { decodeSelectionRecord(corrupt); }), "bad magic");
        require(rejects([&] { decodeSelectionRecord(std::span(bytes).first(bytes.size() - 1)); }), "truncated entries");
        require(rejects([&] { decodeSelectionRecord(std::span(bytes).first(20)); }), "truncated header");
        auto badGeneration = bytes; badGeneration[bytes.size() - 8] = 7;
        require(rejects([&] { decodeSelectionRecord(badGeneration); }), "generation index out of range");
        auto zeroVersion = bytes; std::fill(zeroVersion.end() - 4, zeroVersion.end(), 0);
        require(rejects([&] { decodeSelectionRecord(zeroVersion); }), "zero version");
        auto foreign = captured; foreign.objects[0].storeUuid = util::newUuid();
        if(!foreign.transforms.empty())foreign.transforms[0].storeUuid=util::newUuid();
        require(decodeSelectionRecord(encodeSelectionRecord(foreign))==foreign, "mixed-store references round trip");
        // Existing v1 artifacts used a single store plus a generation dictionary.
        auto legacyHeader=nlohmann::json::parse(bytes.begin()+12,bytes.begin()+12+header);
        legacyHeader["generations"]=nlohmann::json::array();
        for(const auto& origin:legacyHeader.at("origins"))legacyHeader["generations"].push_back(origin.at(1));
        legacyHeader.erase("origins");
        for(auto& transform:legacyHeader["transforms"])transform.erase("store");
        const auto text=legacyHeader.dump();
        std::vector<uint8_t> legacy(bytes.begin(),bytes.begin()+8);legacy[6]=1;
        for(int i=0;i<4;++i)legacy.push_back(uint8_t(text.size()>>(8*i)));
        legacy.insert(legacy.end(),text.begin(),text.end());
        legacy.insert(legacy.end(),bytes.begin()+12+header,bytes.end());
        require(decodeSelectionRecord(legacy)==captured,"existing v1 artifacts remain readable");
    }

    // Kept as an artifact, and read back at its versions after edits.
    {
        const auto file = directory.path / "selection.bin";
        { std::ofstream out(file, std::ios::binary); out.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size())); }
        const auto hash = store->putArtifact(file);
        require(decodeSelectionRecord(store->readArtifact(hash)) == captured, "artifact round trip");
        store->submit(request({{rows[0], 1}}, {SetMetaOp{rows[0], "edited", "true"}}));
        auto txn = store->read();
        const auto revisions = store->objectsAtVersions(txn, captured.objects);
        for (std::size_t i = 0; i < revisions.size(); ++i)
            require(revisions[i] && revisions[i]->version == captured.objects[i].version &&
                    !nlohmann::json::parse(revisions[i]->record.meta).contains("edited"), "historical read-back");
    }

    // A client catalog registration the store has not recorded would decide
    // membership without being named: refused.
    {
        auto extra = std::make_shared<vcr::catalog::RegistrationGraph>();
        extra->add({"unrecorded", "v1", "20260303000000", kTarget, vcr::catalog::Affine::scale(1), "client only"});
        OpenOptions client; client.readOnly = true; client.catalogTransforms = extra;
        auto reader = ObjectStore::open(path, client);
        auto txn = reader.read();
        require(rejects([&] { recordSelection(reader, txn, boxed); }), "unrecorded catalog registration refused");
        ObjectSelection plain; plain.kind = "fiber";
        require(recordSelection(reader, txn, plain).objects.size() == 4, "non-volume selections unaffected");
    }

    // Hosted: the same record as the local capture of the same commit.
    SelectionRecord localNow;
    { auto txn = store->read(); localNow = recordSelection(*store, txn, boxed); }
    store.reset();
    server::ServerOptions serverOptions; serverOptions.readWorkers = 2;
    server::Server service(path, serverOptions);
    service.start();
    const auto endpoint = "http://127.0.0.1:" + std::to_string(service.port());
    HostedStore hosted(endpoint);
    require(hosted.recordSelection(boxed, 1) == localNow, "hosted record equals local");
    {
        auto explicitTransforms = boxed;
        explicitTransforms.transforms = std::make_shared<VolumeTransforms>();
        require(rejects([&] { hosted.recordSelection(explicitTransforms); }), "hosted explicit transforms refused");
    }

    // A writer flips the registration between scale 2 and 3 every few
    // milliseconds. Every record either fails with StoreBusy or names the
    // transform version whose scale gives exactly its membership.
    {
        std::atomic<bool> done{false};
        std::string writerError;
        std::thread writer([&] {
            try {
                HostedStore client(endpoint);
                for (int i = 0; !done; ++i) {
                    auto next = scan; next.revision = "v" + std::to_string(i + 2);
                    next.matrix = vcr::catalog::Affine::scale(i % 2 == 0 ? 3 : 2);
                    TransactionRequest r;
                    r.requestUuid = util::newUuid(); r.storeUuid = initial.storeUuid; r.generation = initial.generation;
                    r.author = "flipper"; r.expectedObjects[transformUuid] = client.currentObjectRevision(transformUuid)->reference();
                    r.operations.push_back(RecordCatalogTransformsOp{{next}});
                    client.submit(r);
                    std::this_thread::sleep_for(std::chrono::milliseconds(3));
                }
            } catch (const std::exception& e) {
                writerError = e.what();
            }
        });
        // Stop the writer before any failure leaves this scope.
        struct Join {
            std::atomic<bool>& done; std::thread& thread;
            ~Join() { done = true; if (thread.joinable()) thread.join(); }
        } join{done, writer};
        int recorded = 0, busy = 0;
        std::set<int64_t> versionsSeen;
        const auto stop = std::chrono::steady_clock::now() + std::chrono::milliseconds(800);
        while (std::chrono::steady_clock::now() < stop) {
            try {
                const auto record = hosted.recordSelection(boxed, 1, 1);
                require(record.transforms.size() == 1 && record.transforms[0].objectUuid == transformUuid,
                        "hosted record names its transform");
                // Version 1 and odd versions are scale 2; even versions scale 3.
                const bool scaleTwo = record.transforms[0].version % 2 == 1;
                const auto expected = scaleTwo ? std::set<std::string>{rows[0], rows[1]} : std::set<std::string>{rows[0]};
                require(uuidsOf(record.objects) == expected, "membership matches the recorded transform version");
                versionsSeen.insert(record.transforms[0].version);
                ++recorded;
            } catch (const StoreBusy&) {
                ++busy;
            }
        }
        done = true;
        writer.join();
        require(writerError.empty(), ("transform writer: " + writerError).c_str());
        require(recorded > 0, "some records completed under churn");
        std::cout << "records under transform churn: " << recorded << " exact, " << busy << " busy, "
                  << versionsSeen.size() << " transform versions seen\n";
    }
    // Paged hosted listings retain the owner on every row, not the server's
    // writable attachment identity. Exact references must survive transport.
    {
        const auto base=directory.path/"base.vcobj",overlay=directory.path/"overlay.vcobj";
        OpenOptions slim;slim.author="selection-test";slim.history=HistoryMode::Off;slim.spatialIndex=false;
        {auto baseStore=ObjectStore::create(base,slim);auto w=baseStore.write();
            ObjectRecord r;r.kind="fiber";baseStore.upsertObject(w,r);baseStore.seal(w);w.commit();}
        {auto collection=ObjectCollection::create(overlay,{base},slim);}
        SelectionRecord expected;
        {auto overlayStore=ObjectStore::open(overlay,slim);
            {auto w=overlayStore.write();ObjectRecord r;r.kind="fiber";overlayStore.upsertObject(w,r);w.commit();}
            auto r=overlayStore.read();expected=recordSelection(overlayStore,r,{});}
        server::Server mixed(overlay,serverOptions);mixed.start();
        HostedStore client("http://127.0.0.1:"+std::to_string(mixed.port()));
        const auto recorded=client.recordSelection({},1);
        require(recorded==expected,"hosted selection retains mixed-source ownership");
        for(const auto& revision:client.objectsAtVersions(recorded.objects))
            require(bool(revision),"every hosted selected reference resolves");
    }
    std::cout << "object_store_selections: ok\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "object_store_selections: " << error.what() << "\n";
    return 1;
}
