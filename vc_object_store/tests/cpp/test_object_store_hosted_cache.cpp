// Delay an old HTTP response until a transform invalidation has completed.
// The late read must neither return nor cache the invalidated mapping.
#include "TestSupport.hpp"
#include "../../src/HostedSession.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include <thread>
#include "vc_object_store/server/Server.hpp"
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
    // Transaction payload uploads (PUT /v1/payloads/...): concurrency seen,
    // an optional delay each, and refusal on demand; transactions forwarded.
    std::atomic<int> uploadsInFlight{0}, maxUploadsInFlight{0}, transactions{0};
    std::atomic<int> uploadDelayMs{0};
    std::atomic_bool refuseUploads{false};
    // Answered 404 without forwarding (a host that predates a route).
    std::atomic_bool notFoundGeometryQuery{false};
    std::atomic<int> geometryQueries{0};
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
                    const std::string requested(request.target());
                    if (requested.starts_with("/v1/payloads/")) {
                        const int now = ++uploadsInFlight;
                        for (int seen = maxUploadsInFlight; now > seen && !maxUploadsInFlight.compare_exchange_weak(seen, now);) {}
                        std::this_thread::sleep_for(std::chrono::milliseconds(uploadDelayMs.load()));
                        --uploadsInFlight;
                        if (refuseUploads) {
                            http::response<http::string_body> refused{http::status::internal_server_error, 11};
                            refused.body() = "{\"error\":\"refused by the test\"}";
                            refused.keep_alive(false);
                            refused.prepare_payload();
                            http::write(socket, refused);
                            return;
                        }
                    }
                    if (requested == "/v1/transactions") ++transactions;
                    if (requested == "/v1/query/geometry") {
                        ++geometryQueries;
                        if (notFoundGeometryQuery) {
                            http::response<http::string_body> missing{http::status::not_found, 11};
                            missing.body() = "{\"error\":\"no such route\"}";
                            missing.keep_alive(false);
                            missing.prepare_payload();
                            http::write(socket, missing);
                            return;
                        }
                    }
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
    // The registration heads are read by version (one batched request) after
    // the listing: hold that response while the registration changes.
    PausingProxy proxy(endpoint, "/v1/query/versions");
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
        request.expectedObjects = {{registration.uuid, 1}};
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
        request.expectedObjects[uuid] = 1;
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
        local.publishArtifacts(attachment, uuid, head->version,
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
        edit.expectedObjects = {{uuid, head->version}};
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

// A transaction's payloads upload at most four at a time, all before the
// transaction; a failed upload stops it from being sent.
void boundedPayloadUploads()
{
    TemporaryDirectory directory;
    const auto path = directory.path / "uploads.vcobj";
    OpenOptions options; options.author = "upload-test";
    { (void)ObjectStore::create(path, options); }
    server::Server server(path, {}); server.start();
    const auto endpoint = "http://127.0.0.1:" + std::to_string(server.port());
    PausingProxy proxy(endpoint, "/never-paused");
    proxy.uploadDelayMs = 100;
    HostedStoreOptions clientOptions; clientOptions.retries = 0;
    HostedStore client(proxy.endpoint(), clientOptions);
    const auto submitTracks = [&](int count, int seed) {
        const auto status = client.status();
        TransactionRequest request;
        request.requestUuid = util::newUuid(); request.storeUuid = status.storeUuid;
        request.generation = status.generation; request.author = "upload-test";
        for (int i = 0; i < count; ++i) {
            const auto uuid = util::newUuid();
            ObjectRecord record; record.uuid = uuid; record.kind = "track";
            vc::geometry::Track track; track.points = {{double(seed), double(i), 1}, {double(seed), double(i), 2}};
            request.expectedObjects[uuid] = std::nullopt;
            request.operations.push_back(CreateObjectOp{objectRecordJson(record)});
            request.operations.push_back(PutGeometryOp{uuid, vc::geometry::encode(track), {}});
        }
        return client.submit(request);
    };
    (void)submitTracks(10, 1);
    require(proxy.maxUploadsInFlight >= 2 && proxy.maxUploadsInFlight <= 4,
            "payloads upload concurrently, at most four at a time");
    require(proxy.transactions == 1, "one transaction after its uploads");
    proxy.refuseUploads = true;
    bool failed = false;
    try { (void)submitTracks(6, 2); } catch (const std::exception&) { failed = true; }
    require(failed && proxy.transactions == 1, "a failed upload stops the transaction from being sent");
}

// A prefetch with the disk cache answers the following reads with no
// per-array requests; a host without the batched query (404) falls back to
// per-array reads, which stay correct.
void batchedGeometryPrefetch()
{
    TemporaryDirectory directory;
    const auto path = directory.path / "batch.vcobj";
    OpenOptions options; options.author = "batch-test";
    std::vector<std::string> uuids;
    {
        auto store = ObjectStore::create(path, options);
        auto txn = store.write();
        for (int i = 0; i < 3; ++i) {
            ObjectRecord object; object.kind = "track";
            const auto uuid = store.upsertObject(txn, object);
            vc::geometry::Track track; track.points = {{double(i), 1, 2}, {double(i), 3, 4}, {double(i), 5, 6}};
            store.putGeometry(txn, uuid, vc::geometry::encode(track));
            uuids.push_back(uuid);
        }
        txn.commit();
    }
    server::Server server(path, {}); server.start();
    const auto endpoint = "http://127.0.0.1:" + std::to_string(server.port());
    OpenOptions readOnly; readOnly.readOnly = true;
    auto inspect = ObjectStore::open(path, readOnly);
    for (const bool olderHost : {false, true}) {
        PausingProxy proxy(endpoint, "/never-paused");
        proxy.notFoundGeometryQuery = olderHost;
        ObjectStoreSession session;
        HostedSessionOptions hosted; hosted.pollInterval = std::chrono::hours(1);
        hosted.cacheDirectory = directory.path / (olderHost ? "cache-old" : "cache-new");
        const auto attachment = session.attachHosted(proxy.endpoint(), hosted);
        session.readStore(attachment, [&](const StoreReader& reader, const auto&) {
            std::vector<std::string> revisions;
            for (const auto& uuid : uuids) revisions.push_back(*reader.currentObject(uuid)->geometryRevision);
            reader.prefetchGeometry(revisions, {"points"}, {});
            const auto afterPrefetch = proxy.payloadRequests.load();
            auto txn = inspect.read();
            for (const auto& revision : revisions)
                require(reader.readGeometryArray(revision, "points") == inspect.readGeometryArray(txn, revision, "points"),
                        "reads after a prefetch return the stored bytes");
            require(proxy.geometryQueries >= 1, "the prefetch asked the batched query");
            if (!olderHost)
                require(proxy.payloadRequests == afterPrefetch, "a prefetch leaves no per-array requests to the reads");
            else
                require(proxy.payloadRequests == afterPrefetch + revisions.size(), "an older host: per-array reads");
            reader.prefetchGeometry(revisions, {"points"}, {});
            require(!olderHost || proxy.geometryQueries == 1, "an older host is not asked again");
        });
    }
}

int main()
try {
    boundedPayloadUploads();
    batchedGeometryPrefetch();
#ifdef VC_REAL_SURFACE
    persistentObjects();
#endif
    invalidationDuringRead(false);
    invalidationDuringRead(true);
    multiPageCommitInvalidatesHeads();
    std::cout << "object_store_hosted_cache: ok\n";
} catch (const std::exception& error) {
    std::cerr << "object_store_hosted_cache: " << error.what() << '\n';
    return 1;
}
