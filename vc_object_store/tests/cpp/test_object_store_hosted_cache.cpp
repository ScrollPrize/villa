// Delay an old HTTP response until a transform invalidation has completed.
// The late read must neither return nor cache the invalidated mapping.
#include "TestSupport.hpp"
#include "../../src/HostedSession.hpp"
#include "vc_object_store/server/Server.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include "vc_object_store/objects/TransactionJson.hpp"
#include "vc_object_store/objects/ObjectCollection.hpp"
#include "vc_object_store/store/GeometryViews.hpp"
#include <vc_reader/http.hpp>
#include <boost/asio.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <condition_variable>
#include <future>
#include <fstream>
#ifdef VC_REAL_SURFACE
#include "vc_object_store/objects/GeometryIngest.hpp"
#endif
#include <iostream>

using namespace vc::objects;
namespace net = boost::asio;
namespace http = boost::beast::http;
using tcp = net::ip::tcp;

namespace {
class PausingProxy {
public:
    PausingProxy(std::string upstream, std::string target)
        : upstream_(std::move(upstream)), target_(std::move(target)),
          acceptor_(io_, {net::ip::make_address("127.0.0.1"), 0}), port_(acceptor_.local_endpoint().port()),
          thread_([this] { serve(); }) {}
    ~PausingProxy()
    {
        release();
        stopping_ = true;
        // Wake the synchronous accept without cancelling sockets in another thread.
        boost::system::error_code ignored;
        tcp::socket wake(io_);
        wake.connect({net::ip::make_address("127.0.0.1"), port_}, ignored);
        thread_.join();
        for (auto& worker : workers_) worker.join();
    }
    std::string endpoint() const { return "http://127.0.0.1:" + std::to_string(port_); }
    std::atomic<size_t> payloadRequests{0}, payloadBytes{0};
    void wait()
    {
        std::unique_lock lock(mutex_);
        if (!ready_.wait_for(lock, std::chrono::seconds(10), [&] { return paused_; }))
            throw std::runtime_error("transform read did not reach the response gate");
    }
    void release()
    {
        std::lock_guard lock(mutex_);
        released_ = true;
        ready_.notify_all();
    }
private:
    void serve()
    {
        while (!stopping_) {
            tcp::socket socket(io_);
            acceptor_.accept(socket);
            if (stopping_) break;
            workers_.emplace_back([this, socket = std::move(socket)]() mutable {
                try {
                    boost::beast::flat_buffer buffer;
                    http::request<http::string_body> request;
                    http::read(socket, buffer, request);
                    vcr::http::Options options; options.retries = 0; options.requestTimeoutMs = 10000;
                    vcr::http::Session transport(options);
                    const auto reply = transport.request(std::string(request.method_string()).c_str(),
                        upstream_ + std::string(request.target()), request.body());
                    const std::string target(request.target());
                    if (target.find("/arrays/") != std::string::npos || target.starts_with("/v1/artifacts/")) {
                        ++payloadRequests; payloadBytes += reply.body.size();
                    }
                    if (request.target() == target_) {
                        std::unique_lock lock(mutex_);
                        if (!paused_) {
                            paused_ = true;
                            ready_.notify_all();
                            ready_.wait(lock, [&] { return released_; });
                        }
                    }
                    http::response<http::string_body> response{http::status(reply.status), 11};
                    response.body() = reply.body;
                    response.keep_alive(false);
                    response.prepare_payload();
                    http::write(socket, response);
                } catch (...) { /* Closing the socket reports the failure to the client. */ }
            });
        }
    }
    std::string upstream_, target_;
    net::io_context io_;
    tcp::acceptor acceptor_;
    uint16_t port_;
    std::atomic_bool stopping_{false};
    std::mutex mutex_;
    std::condition_variable ready_;
    bool paused_ = false, released_ = false;
    std::vector<std::thread> workers_;
    std::thread thread_;
};

void referencedRecordsCache()
{
    TemporaryDirectory directory;OpenOptions options;options.author="cache-test";options.history=HistoryMode::Off;
    const auto base=directory.path/"base.vcobj",overlay=directory.path/"overlay.vcobj";
    std::vector<ObjectVersion> refs;
    {auto store=ObjectStore::create(base,options);auto w=store.write();ObjectRecord r;r.kind="note";
        const auto id=store.upsertObject(w,r);w.commit();
        {auto read=store.read();refs.push_back(store.currentObjectRevision(read,id)->reference());}
        auto seal=store.write();store.seal(seal);seal.commit();}
    {auto collection=ObjectCollection::create(overlay,{base},options);}
    {auto store=ObjectStore::open(overlay,options);auto w=store.write();ObjectRecord r;r.kind="note";
        const auto id=store.upsertObject(w,r);w.commit();auto read=store.read();refs.push_back(store.currentObjectRevision(read,id)->reference());}
    server::Server server(overlay,{});server.start();
    const auto endpoint="http://127.0.0.1:"+std::to_string(server.port());
    HostedSessionOptions settings;settings.client.retries=0;settings.cacheDirectory=directory.path/"cache";settings.minFreeBytes=0;
    {ObjectStoreSession session;const auto attachment=session.attachHosted(endpoint,settings);
        const auto job=session.prefetchObjects(attachment);
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);
        while(!job->snapshot().done&&std::chrono::steady_clock::now()<deadline)std::this_thread::sleep_for(std::chrono::milliseconds(1));
        require(job->snapshot().complete,"mixed-source prefetch completes");}
    // A fresh attachment must find every prefetched record on disk, with no service.
    detail::HostedAttachment cold(endpoint,settings);
    server.stop();
    const auto cached=cold.objectsAtVersions(refs);
    require(cached.size()==2&&cached[0]&&cached[1]&&cached[0]->reference()==refs[0]&&cached[1]->reference()==refs[1],
            "mixed-source prefetch retains both owner-qualified records across restart");
}

void referencedGraphCache()
{
    TemporaryDirectory directory;
    for(const auto policy:{HistoryMode::On,HistoryMode::Off}) {
        const auto root=directory.path/std::to_string(int(policy));std::filesystem::create_directory(root);
        OpenOptions options;options.author="cache-test";options.history=policy;
        const auto base=root/"base.vcobj",overlay=root/"overlay.vcobj";
        std::vector<ObjectVersion> refs;
        {auto db=ObjectStore::create(base,options);auto w=db.write();ObjectRecord record;record.kind="note";
            const auto a=db.upsertObject(w,record),b=db.upsertObject(w,record);w.commit();
            {auto r=db.read();refs={db.currentObjectRevision(r,a)->reference(),db.currentObjectRevision(r,b)->reference()};}
            auto seal=db.write();db.seal(seal);seal.commit();}
        {auto db=ObjectCollection::create(overlay,{base},options);}
        server::Server server(overlay,{});server.start();
        const auto endpoint="http://127.0.0.1:"+std::to_string(server.port());HostedStore writer(endpoint);
        HostedSessionOptions settings;settings.client.retries=0;settings.cacheDirectory=root/"cache";settings.minFreeBytes=0;
        detail::HostedAttachment cached(endpoint,settings);
        require(cached.relationshipsAt(refs).empty(),"cache initially empty inherited graph");
        const auto request=[&] {
            const auto status=writer.status();TransactionRequest value;value.requestUuid=util::newUuid();
            value.storeUuid=status.storeUuid;value.generation=status.generation;value.author="cache-test";
            for(const auto& ref:refs)expectObject(value,ref);
            return value;
        };
        EdgeRecord edge;edge.uuid=util::newUuid();edge.kind="declared_reference";edge.aUuid=refs[0].objectUuid;edge.bUuid=refs[1].objectUuid;
        auto add=request();add.operations.push_back(AddEdgeOp{edgeRecordJson(edge)});const auto added=writer.submit(add);
        cached.refresh();require(cached.relationshipsAt(refs).size()==1,"refresh invalidates inherited graph disk cache");
        detail::HostedAttachment reopened(endpoint,settings);
        require(reopened.relationshipsAt(refs).size()==1,"reopened cache retains current graph membership");
        auto remove=request();remove.expectedRelationships=testNumberedGuards(writer, {{edge.uuid, added.relationships.at(edge.uuid)}}, true);
        remove.operations.push_back(RemoveEdgeOp{edge.uuid});cached.submit(remove);
        require(cached.relationshipsAt(refs).empty(),"own submit invalidates graph without explicit refresh");
        // An empty response already in flight must not outlive a refresh that
        // observes a new overlay edge, even though endpoint pins are identical.
        PausingProxy proxy(endpoint,"/v1/query/relationships");
        auto raceSettings=settings;raceSettings.cacheDirectory=root/"race-cache";
        detail::HostedAttachment racing(proxy.endpoint(),raceSettings);
        auto pending=std::async(std::launch::async,[&]{return racing.relationshipsAt(refs);});
        try {
            proxy.wait();edge.uuid=util::newUuid();auto next=request();
            next.operations.push_back(AddEdgeOp{edgeRecordJson(edge)});writer.submit(next);
            racing.refresh();proxy.release();
            require(pending.get().size()==1,"late graph response retries after newer observed commit");
        } catch (...) {proxy.release();throw;}
        server.stop();
    }
}

void selectedHeadAndCanonicalCache()
{
    TemporaryDirectory directory;const auto path=directory.path/"store.vcobj";
    const std::string id="abcdefab-abcd-4abc-8abc-abcdefabcdef",upper="ABCDEFAB-ABCD-4ABC-8ABC-ABCDEFABCDEF";
    OpenOptions options;options.author="cache-test";
    ObjectVersion initial;
    {auto store=ObjectStore::create(path,options);auto w=store.write();ObjectRecord record;record.uuid=id;record.kind="track";record.track.emplace();
        store.upsertObject(w,record);store.putGeometry(w,id,vc::geometry::encode(vc::geometry::Track{{{1,2,3},{4,5,6}}}));w.commit();
        auto r=store.read();initial=store.currentObjectRevision(r,id)->reference();}
    server::Server server(path,{});server.start();
    const auto endpoint="http://127.0.0.1:"+std::to_string(server.port());
    HostedStore writer(endpoint);PausingProxy proxy(endpoint,"/v1/objects/"+id);
    HostedSessionOptions settings;settings.client.retries=0;settings.client.requestTimeoutMs=10000;
    detail::HostedAttachment attachment(proxy.endpoint(),settings);
    auto viewRead=std::async(std::launch::async,[&] {
        detail::HostedStoreReader reader(attachment);
        const auto view=GeometrySnapshot(reader).track(id);
        require(view.reference()==initial,"typed current view carries the selected complete identity");
        require(view.points(0,1).front()[0]==1,"typed view must not pair the selected identity with a newer geometry head");
    });
    try {
        proxy.wait();const auto status=writer.status();TransactionRequest request;
        request.requestUuid=util::newUuid();request.storeUuid=status.storeUuid;request.generation=status.generation;request.author="cache-test";
        expectObject(request,initial);
        request.operations={UpdateGeometryArraysOp{id,vc::geometry::encode(vc::geometry::Track{{{9,2,3},{4,5,6}}}),{}}};
        writer.submit(request);attachment.refresh();
    } catch(...) {proxy.release();viewRead.wait();throw;}
    proxy.release();viewRead.get();
    require(attachment.currentObject(upper)->version==2&&attachment.currentObject(id)->version==2,"both UUID spellings select the updated head");
    const auto status=writer.status();TransactionRequest update;
    update.requestUuid=util::newUuid();update.storeUuid=status.storeUuid;update.generation=status.generation;update.author="cache-test";
    update.expectedObjects=testNumberedGuards(writer, {{id,2}});update.operations={SetMetaOp{id,"label","true"}};
    attachment.submit(update);
    require(attachment.currentObject(upper)->version==3&&attachment.currentObject(id)->version==3,"canonical invalidation clears all UUID spellings");
}

void invalidationDuringRead(bool poll)
{
    TemporaryDirectory directory;
    const auto path = directory.path / "store.vcobj";
    OpenOptions options; options.author = "cache-test";
    ObjectRecord registration; registration.uuid = util::newUuid();
    registration = volumeTransformRecord(registration, "source", "target", VolumeAffine::identity(), "registration");
    {
        auto db = ObjectStore::create(path, options);
        auto txn = db.write(); db.upsertObject(txn, registration); txn.commit();
    }
    server::Server server(path, {}); server.start();
    const auto endpoint = "http://127.0.0.1:" + std::to_string(server.port());
    HostedStore writer(endpoint);
    PausingProxy proxy(endpoint, "/v1/volume-transforms");
    HostedSessionOptions hosted; hosted.client.retries = 0; hosted.client.requestTimeoutMs = 10000;
    detail::HostedAttachment attachment(proxy.endpoint(), hosted);
    auto read = std::async(std::launch::async, [&] { return attachment.volumeTransforms(); });
    try {
        proxy.wait();
        auto matrix = VolumeAffine::identity(); matrix.values[3] = 12;
        registration = volumeTransformRecord(registration, "source", "target", matrix, "registration");
        const auto status = writer.status();
        TransactionRequest request;
        request.requestUuid = util::newUuid(); request.storeUuid = status.storeUuid;
        request.generation = status.generation; request.author = "cache-test";
        request.expectedObjects = testNumberedGuards(writer, {{registration.uuid, 1}});
        request.operations = {UpdateObjectOp{objectRecordJson(registration)}};
        if (poll) {
            writer.submit(request);
            attachment.refresh();
        } else attachment.submit(request);
    } catch (...) {
        proxy.release(); read.wait(); throw;
    }
    proxy.release();
    const auto metadata = nlohmann::json{{"coordinates", volumeCoordinates("source", 1)}};
    const auto fetched = read.get().toVolume(metadata, "target");
    require(fetched && fetched->values[3] == 12, "late transform fetch retries after invalidation");
    const auto cached = attachment.volumeTransforms().toVolume(metadata, "target");
    require(cached == fetched, "cache retains the new mapping");
}

void multiPageCommitInvalidatesHeads()
{
    TemporaryDirectory directory;
    const auto path = directory.path / "feed.vcobj";
    OpenOptions options; options.author = "feed-test";
    std::vector<std::string> uuids;
    {
        auto store = ObjectStore::create(path, options);
        auto txn = store.write();
        for (int i = 0; i < 1025; ++i) {
            ObjectRecord object; object.kind = "surface";
            uuids.push_back(store.upsertObject(txn, object));
        }
        txn.commit();
    }
    std::sort(uuids.begin(), uuids.end());
    server::ServerOptions serverOptions; serverOptions.maxFeedChanges = 1024;
    server::Server server(path, serverOptions); server.start();
    const auto endpoint = "http://127.0.0.1:" + std::to_string(server.port());
    HostedStore writer(endpoint);
    HostedSessionOptions hosted; hosted.pollInterval = std::chrono::hours(1);
    detail::HostedAttachment attachment(endpoint, hosted);
    std::vector<StoreSessionEvent> events;
    const AttachmentIdentity identity{attachment.storeUuid(), 1};
    attachment.start(identity, [&](StoreSessionEvent event) { events.push_back(std::move(event)); });
    require(attachment.currentObject(uuids.front())->version == 1, "cache first head before bulk edit");
    require(attachment.currentObject(uuids.back())->version == 1, "cache last head before bulk edit");
    const auto before = writer.status();
    TransactionRequest request;
    request.requestUuid = util::newUuid(); request.storeUuid = before.storeUuid;
    request.generation = before.generation; request.author = "bulk-editor";
    for (const auto& uuid : uuids) {
        request.expectedObjects[uuid] = writer.currentObjectRevision(uuid)->reference();
        request.operations.push_back(SetMetaOp{uuid, "edited", "true"});
    }
    const auto receipt = writer.submit(request);
    attachment.refresh();
    require(attachment.freshness().error.empty(), "bulk feed refresh succeeds");
    require(events.size() == 1 && events.front().attachment == identity, "one complete bulk event");
    const auto& changes = events.front().changes;
    require(changes.size() == uuids.size(), "all continuation pages delivered exactly once");
    for (size_t i = 0; i < uuids.size(); ++i) {
        require(changes[i].recordUuid == uuids[i] && changes[i].seq == *receipt.commitSequence,
                "feed order and sequence preserved");
        require(changes[i].author == "bulk-editor" && changes[i].kind == "surface" &&
                    changes[i].changeKind == "published", "feed attribution and kind preserved");
    }
    require(attachment.currentObject(uuids.front())->version == 2 &&
                attachment.currentObject(uuids.back())->version == 2, "bulk edit invalidates cached heads");
    attachment.refresh();
    require(events.size() == 1, "bulk feed cursor advances past the entire commit");
    attachment.stop();
}

void oldMetadataCacheDoesNotShadowConvertedStore()
{
    TemporaryDirectory directory;
    const auto path = directory.path / "converted.vcobj";
    OpenOptions options; options.author = "cache-upgrade-test";
    StoreStatus status;
    ObjectVersion reference;
    std::string geometry;
    std::vector<uint8_t> expected;
    nlohmann::json oldObject, oldGeometry;
    {
        auto store = ObjectStore::create(path, options);
        std::string uuid;
        {
            auto txn = store.write();
            ObjectRecord record; record.kind = "track";
            uuid = store.upsertObject(txn, record);
            vc::geometry::Track track; track.points = {{1, 2, 3}, {4, 5, 6}};
            geometry = store.putGeometry(txn, uuid, vc::geometry::encode(track));
            txn.commit();
        }
        auto txn = store.read();
        status = store.status(txn);
        const auto current = store.currentObjectRevision(txn, uuid);
        reference = current->reference();
        oldObject = nlohmann::json::parse(objectRevisionToJson(*current));
        auto& record = oldObject["record"];
        record["generation"] = 2; record["approved_generation"] = nullptr;
        for (const auto* name : {"version", "version_generation", "commit_sequence", "approved_version"}) record.erase(name);
        oldGeometry = nlohmann::json::parse(geometryRevisionToJson(*store.currentGeometry(txn, uuid)));
        oldGeometry["object_generation"] = 2; oldGeometry.erase("object_version");
        expected = store.readGeometryArray(txn, geometry, "points");
    }
    HostedSessionOptions hosted;
    hosted.cacheDirectory = directory.path / "cache"; hosted.minFreeBytes = 0;
    const auto cacheRoot = hosted.cacheDirectory / "objects";
    std::filesystem::create_directories(cacheRoot);
    const auto bytes = [](const std::string& text) { return std::vector<uint8_t>(text.begin(), text.end()); };
    const auto cache = [&](const std::string& key, const std::vector<uint8_t>& value) {
        std::ofstream file(cacheRoot / payloadHash(bytes(key)), std::ios::binary);
        file << payloadHash(value);
        file.write(reinterpret_cast<const char*>(value.data()), value.size());
    };
    cache("version/" + status.storeUuid + "/" + status.generation + "/" + reference.objectUuid + "/" +
        std::to_string(reference.version), bytes(oldObject.dump()));
    const auto prefix = status.storeUuid + "/" + status.generation + "/";
    cache(prefix + "geometry/" + geometry, bytes(oldGeometry.dump()));
    cache(prefix + "block/" + geometry + "/points/" + std::to_string(expected.size()) + "/0", expected);
    server::Server server(path, {}); server.start();
    PausingProxy proxy("http://127.0.0.1:" + std::to_string(server.port()), "never");
    detail::HostedAttachment attachment(proxy.endpoint(), hosted);
    const auto found = attachment.objectsAtVersions({reference});
    require(found.front() && found.front()->record.uuid == reference.objectUuid,
            "schema-20 cached records cannot break current numbered reads");
    const auto descriptor = attachment.geometryRevision(geometry);
    require(descriptor && descriptor->objectVersion == reference.version,
            "schema-20 cached descriptors are refetched in the current metadata namespace");
    require(*attachment.geometryArray(geometry, "points") == expected && proxy.payloadRequests == 0,
            "wire metadata upgrade reuses existing raw geometry blocks without download");
}
#ifdef VC_REAL_SURFACE
void persistentObjects()
{
    // Published scroll coordinates from the repository; never modify the input.
    TemporaryDirectory directory;
    const auto path = directory.path / "real.vcobj";
    OpenOptions options; options.author = "persistent-cache-test";
    std::string uuid, geometry;
    std::vector<uint8_t> expected;
    size_t expectedPayloads = 1; // one attached file
    {
        auto store = ObjectStore::create(path, options);
        const auto source = directory.path / "surface";
        std::filesystem::copy(VC_REAL_SURFACE, source, std::filesystem::copy_options::recursive);
        // The repository fixture uses a catalog ID; imported identities are UUIDs.
        std::ifstream metadata(source / "meta.json");
        auto meta = nlohmann::json::parse(metadata); metadata.close();
        meta["uuid"] = util::newUuid();
        std::ofstream(source / "meta.json") << meta.dump();
        const auto report = ingestGeometrySource(store, "surface", source);
        for (const auto& problem : report.problems) std::cerr << problem << "\n";
        require(report.ingested == 1 && report.problems.empty(), "real scroll tifxyz ingests");
        auto txn = store.read();
        uuid = store.selectObjects(txn, {}).at(0).uuid;
        const auto descriptor = store.currentGeometry(txn, uuid);
        geometry = descriptor->uuid;
        for (const auto& array : descriptor->arrays) expectedPayloads += (array.byteLength + (1 << 20) - 1) / (1 << 20);
        expected = store.readGeometryArray(txn, geometry, "points");
    }
    {
        ObjectStoreSession local;
        auto attachment = local.attach(path, options);
        const auto head = local.currentObject(attachment, uuid);
        local.publishArtifacts(attachment, head->reference(),
            {{"source/x.tif", std::filesystem::path(VC_REAL_SURFACE) / "x.tif"}});
    }
    server::Server server(path, {}); server.start();
    const auto endpoint = "http://127.0.0.1:" + std::to_string(server.port());
    HostedSessionOptions hosted;
    hosted.cacheDirectory = directory.path / "cache";
    hosted.diskCacheBytes = 16 << 20;
    hosted.cacheBytes = 128; // real surface is larger than the entire memory cache
    hosted.pollInterval = std::chrono::hours(1);
    const auto finish = [](const auto& job) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
        while (!job->snapshot().done && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        require(job->snapshot().done, "prefetch completes without blocking its owner");
        return job->snapshot();
    };
    StoreStatus captured;
    {
        PausingProxy proxy(endpoint, "never");
        ObjectStoreSession session;
        const auto attachment = session.attachHosted(proxy.endpoint(), hosted);
        auto job = session.prefetchObjects(attachment);
        require(session.prefetchObjects(attachment) == job, "one prefetch per attachment");
        const auto result = finish(job);
        if (!result.errors.empty()) std::cerr << result.errors.front() << '\n';
        require(result.complete && result.completed == 1 && result.total == 1 && result.downloadedBytes > 0,
                "prefetch caches real geometry and attached file");
        require(proxy.payloadRequests > 0, "cold cache downloads payloads");
        captured = *result.captured;
    }
    // New session and a different forwarded port: only immutable store identity
    // determines the namespace. Both normal access paths use the prefetch cache.
    {
        PausingProxy proxy(endpoint, "never");
        ObjectStoreSession session;
        const auto attachment = session.attachHosted(proxy.endpoint(), hosted);
        require(session.readGeometryArray(attachment, geometry, "points") == expected, "restart preserves exact real geometry");
        session.readStore(attachment, [&](const StoreReader& reader, const auto&) {
            require(reader.readGeometryArray(geometry, "points", 13, 111) ==
                std::vector<uint8_t>(expected.begin() + 13, expected.begin() + 124), "unaligned range shares prefetch blocks");
        });
        const auto artifact = session.artifactFile(attachment, uuid, "source/x.tif", directory.path / "materialized");
        require(artifact && std::filesystem::file_size(*artifact) == std::filesystem::file_size(std::filesystem::path(VC_REAL_SURFACE) / "x.tif"),
                "normal artifact access uses prefetched bytes");
        const auto result = finish(session.prefetchObjects(attachment));
        require(result.complete && result.downloadedBytes == 0 && proxy.payloadRequests == 0,
                "disk-warm restart downloads no geometry or artifact payloads");
    }
    const auto key = captured.storeUuid + "/" + captured.generation + "/block/" + geometry + "/points/" +
        std::to_string(expected.size()) + "/0";
    const auto cached = hosted.cacheDirectory / "objects" / payloadHash(std::vector<uint8_t>(key.begin(), key.end()));
    { std::ofstream corrupt(cached, std::ios::binary | std::ios::trunc); corrupt << "interrupted"; }
    {
        PausingProxy proxy(endpoint, "never");
        ObjectStoreSession session;
        const auto attachment = session.attachHosted(proxy.endpoint(), hosted);
        require(session.readGeometryArray(attachment, geometry, "points") == expected && proxy.payloadRequests > 0,
                "truncated block is discarded and refetched");
    }
    {
        auto concurrent = hosted; concurrent.cacheDirectory = directory.path / "concurrent";
        const auto range = std::min<size_t>(1 << 20, expected.size());
        PausingProxy proxy(endpoint, "/v1/geometry/" + geometry + "/arrays/points?offset=0&count=" + std::to_string(range));
        ObjectStoreSession session;
        const auto attachment = session.attachHosted(proxy.endpoint(), concurrent);
        auto job = session.prefetchObjects(attachment);
        proxy.wait();
        auto interactive = std::async(std::launch::async, [&] { return session.readGeometryArray(attachment, geometry, "points"); });
        // A new metadata commit while transfer is paused must not change its
        // captured version or invalidate immutable geometry already downloaded.
        HostedStore writer(endpoint);
        const auto status = writer.status();
        const auto head = writer.currentObjectRevision(uuid);
        TransactionRequest edit;
        edit.requestUuid = util::newUuid(); edit.storeUuid = status.storeUuid;
        edit.generation = status.generation; edit.author = "cache-test";
        edit.expectedObjects = {{uuid, head->reference()}};
        edit.operations = {SetMetaOp{uuid, "during_prefetch", "true"}};
        writer.submit(edit);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        proxy.release();
        require(interactive.get() == expected, "interactive reads share an active prefetch miss");
        const auto result = finish(job);
        require(result.complete && result.captured->commitSequence == status.commitSequence,
                "prefetch remains pinned across a remote edit");
        require(proxy.payloadRequests == expectedPayloads, "concurrent interactive reads add no duplicate payload transfers");
    }
    {
        auto detached = hosted; detached.cacheDirectory = directory.path / "detached";
        const auto range = std::min<size_t>(1 << 20, expected.size());
        PausingProxy proxy(endpoint, "/v1/geometry/" + geometry + "/arrays/points?offset=0&count=" + std::to_string(range));
        ObjectStoreSession session;
        const auto attachment = session.attachHosted(proxy.endpoint(), detached);
        auto job = session.prefetchObjects(attachment);
        proxy.wait();
        const auto started = std::chrono::steady_clock::now();
        session.detach(attachment);
        require(std::chrono::steady_clock::now() - started < std::chrono::seconds(1), "detach does not wait for prefetch transport");
        proxy.release();
        require(finish(job).cancelled, "detachment cancels obsolete work");
    }
    {
        PausingProxy proxy(endpoint, "/v1/query/objects");
        ObjectStoreSession session;
        const auto attachment = session.attachHosted(proxy.endpoint(), hosted);
        auto job = session.prefetchObjects(attachment);
        proxy.wait(); job->cancel(); proxy.release();
        require(finish(job).cancelled, "cancel during discovery keeps completed cache entries");
        require(finish(session.prefetchObjects(attachment)).complete, "cancelled prefetch resumes");
    }
    {
        ObjectStoreSession session;
        auto limited = hosted; limited.cacheDirectory = directory.path / "limited"; limited.diskCacheBytes = 0;
        const auto attachment = session.attachHosted(endpoint, limited);
        require(!finish(session.prefetchObjects(attachment)).complete, "eviction is reported as incomplete coverage");
        require(session.readGeometryArray(attachment, geometry, "points") == expected, "uncacheable interactive read succeeds");
    }
    {
        ObjectStoreSession session;
        auto limited = hosted; limited.cacheDirectory = directory.path / "low-space"; limited.minFreeBytes = INT64_MAX;
        const auto attachment = session.attachHosted(endpoint, limited);
        require(!finish(session.prefetchObjects(attachment)).complete, "minimum free space reports incomplete coverage");
    }
}
#endif

}

int main()
try {
    selectedHeadAndCanonicalCache();
    referencedRecordsCache();
    referencedGraphCache();
#ifdef VC_REAL_SURFACE
    persistentObjects();
#endif
    invalidationDuringRead(false);
    invalidationDuringRead(true);
    multiPageCommitInvalidatesHeads();
    oldMetadataCacheDoesNotShadowConvertedStore();
    std::cout << "object_store_hosted_cache: ok\n";
} catch (const std::exception& error) {
    std::cerr << "object_store_hosted_cache: " << error.what() << '\n';
    return 1;
}
