// HTTP adapter over the public ObjectStore APIs. Network threads only parse
// and write messages; every storage call runs on a bounded lane: one writer
// for checked transactions, read workers for reads/feed/staging, and a
// control lane so status and receipt checks stay responsive while the others
// are busy. A full lane answers 503 (clients retry with backoff).
#include "vc_object_store/server/Server.hpp"

#include "vc_object_store/Uuid.hpp"
#include "vc_object_store/objects/Artifacts.hpp"
#include "vc_object_store/objects/ObjectStore.hpp"
#include "vc_object_store/objects/StoreReader.hpp"
#include "vc_object_store/objects/TransactionJson.hpp"
#if VC_OBJECT_SERVER_LINKING
#include "vc_object_store/object_links/LinkObjects.hpp"
#endif

#include <boost/asio.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <nlohmann/json.hpp>
#include <vc_reader/native/sigv4.hpp>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <ctime>
#include <deque>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

namespace vc::objects::server {
namespace {
namespace net = boost::asio;
namespace beast = boost::beast;
namespace http = beast::http;
using tcp = net::ip::tcp;
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;

// Fixed threads draining a bounded queue. stop() runs what was accepted.
class Lane {
public:
    Lane(int threads, std::size_t capacity) : capacity_(capacity)
    {
        for (int i = 0; i < std::max(threads, 1); ++i) threads_.emplace_back([this] { loop(); });
    }
    ~Lane() { stop(); }
    bool tryPost(std::function<void()> job)
    {
        {
            std::lock_guard lock(mutex_);
            if (stopping_ || queue_.size() >= capacity_) return false;
            queue_.push_back(std::move(job));
        }
        cv_.notify_one();
        return true;
    }
    std::size_t queued()
    {
        std::lock_guard lock(mutex_);
        return queue_.size();
    }
    void stop()
    {
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
        }
        cv_.notify_all();
        for (auto& t : threads_)
            if (t.joinable()) t.join();
    }
private:
    void loop()
    {
        for (;;) {
            std::function<void()> job;
            {
                std::unique_lock lock(mutex_);
                cv_.wait(lock, [&] { return stopping_ || !queue_.empty(); });
                if (queue_.empty()) return;
                job = std::move(queue_.front());
                queue_.pop_front();
            }
            job();
        }
    }
    std::size_t capacity_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<std::function<void()>> queue_;
    bool stopping_ = false;
    std::vector<std::thread> threads_;
};

struct Reply {
    http::status status = http::status::ok;
    std::string contentType = "application/json", body;
};

Reply jsonReply(std::string body, http::status status = http::status::ok) { return {status, "application/json", std::move(body)}; }
Reply errorReply(http::status status, const std::string& message)
{
    return jsonReply(Json{{"error", message}}.dump(), status);
}

// Uploaded payload files by SHA-256, until they expire. Staging is not part
// of a request's identity: an expired payload is uploaded again. Uploads in
// progress reserve their declared length so the total stays bounded. Files
// get unique names, and a file is removed only when its last reader is done,
// so expiry never deletes bytes a transaction is reading.
class Staging {
public:
    Staging(std::filesystem::path directory, std::size_t limit, std::chrono::seconds lifetime)
        : directory_(std::move(directory)), limit_(limit), lifetime_(lifetime)
    {
        std::filesystem::create_directories(directory_);
        // Staging does not survive a restart: drop payload files a previous
        // run left (names are "<sha256>.<n>" and "<sha256>.<n>.part").
        for (const auto& entry : std::filesystem::directory_iterator(directory_)) {
            const auto name = entry.path().filename().string();
            if (entry.is_regular_file() && name.size() > 65 && name[64] == '.' && isHash(name.substr(0, 64)))
                std::filesystem::remove(entry.path());
        }
    }

    static bool isHash(std::string_view text)
    {
        return text.size() == 64 && std::all_of(text.begin(), text.end(), [](char c) {
            return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
        });
    }

    // One upload being written: removed (and its reservation released)
    // unless finished into an entry.
    class Upload {
    public:
        Upload(Staging& staging, std::string hash, std::size_t length, std::filesystem::path path)
            : staging_(staging), hash_(std::move(hash)), length_(length), path_(std::move(path)) {}
        ~Upload()
        {
            if (file_.is_open()) file_.close();
            if (!finished_) {
                std::error_code ignored;
                std::filesystem::remove(path_, ignored);
            }
            staging_.release(length_);
        }
        // Transfer lane only.
        void write(const char* data, std::size_t size)
        {
            if (!file_.is_open()) {
                file_.open(path_, std::ios::binary | std::ios::trunc);
                if (!file_) throw std::runtime_error("cannot create staging file " + path_.string());
            }
            hasher_.update(data, size);
            file_.write(data, static_cast<std::streamsize>(size));
            if (!file_) throw std::runtime_error("cannot write staging file " + path_.string());
            written_ += size;
        }
        // Transfer lane only: verify the bytes, then hand the file to
        // `deliver` (which may move it) or, without one, stage it as a payload.
        Reply finish(const std::function<void(const std::filesystem::path&)>& deliver = {});
        const std::string& hash() const { return hash_; }
    private:
        Staging& staging_;
        std::string hash_;
        std::size_t length_, written_ = 0;
        std::filesystem::path path_;
        std::ofstream file_;
        PayloadHasher hasher_;
        bool finished_ = false;
    };

    // Room for `length` bytes, or nullptr when staging is full.
    std::unique_ptr<Upload> begin(const std::string& hash, std::size_t length)
    {
        std::lock_guard lock(mutex_);
        purge(Clock::now());
        if (length > limit_ - std::min(limit_, total_ + reserved_)) return nullptr;
        reserved_ += length;
        const auto name = hash + "." + std::to_string(++counter_);
        return std::make_unique<Upload>(*this, hash, length, directory_ / (name + ".part"));
    }

    std::optional<std::vector<uint8_t>> get(const std::string& hash)
    {
        std::shared_ptr<Entry> entry;
        {
            std::lock_guard lock(mutex_);
            const auto found = entries_.find(hash);
            if (found == entries_.end() || found->second->expires <= Clock::now()) return std::nullopt;
            entry = found->second;
        }
        std::ifstream in(entry->path, std::ios::binary);
        std::vector<uint8_t> bytes(entry->size);
        if (!in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size())))
            throw std::runtime_error("cannot read staged payload " + hash);
        return bytes;
    }

    Json report()
    {
        std::lock_guard lock(mutex_);
        return {{"directory", directory_.string()}, {"payloads", entries_.size()}, {"staged_bytes", total_},
                {"reserved_bytes", reserved_}, {"limit_bytes", limit_}};
    }

private:
    struct Entry {
        Entry() = default;
        Entry(const Entry&) = delete;
        Entry& operator=(const Entry&) = delete;
        std::filesystem::path path;
        std::size_t size = 0;
        Clock::time_point expires;
        ~Entry()
        {
            std::error_code ignored;
            std::filesystem::remove(path, ignored);
        }
    };
    void release(std::size_t length)
    {
        std::lock_guard lock(mutex_);
        reserved_ -= length;
    }
    // The upload's file becomes the entry for its hash (or refreshes it).
    void publish(const std::string& hash, const std::filesystem::path& part, std::size_t size)
    {
        auto final = part;
        final.replace_extension();  // drop ".part"
        std::filesystem::rename(part, final);
        auto entry = std::make_shared<Entry>();
        entry->path = final;
        entry->size = size;
        entry->expires = Clock::now() + lifetime_;
        std::lock_guard lock(mutex_);
        auto& slot = entries_[hash];
        if (slot) total_ -= slot->size;
        slot = std::move(entry);  // a replaced file goes once its readers finish
        total_ += size;
    }
    void purge(Clock::time_point now)
    {
        for (auto it = entries_.begin(); it != entries_.end();) {
            if (it->second->expires <= now) {
                total_ -= it->second->size;
                it = entries_.erase(it);
            } else ++it;
        }
    }
    std::filesystem::path directory_;
    std::size_t limit_, total_ = 0, reserved_ = 0;
    uint64_t counter_ = 0;
    std::chrono::seconds lifetime_;
    std::mutex mutex_;
    std::map<std::string, std::shared_ptr<Entry>> entries_;
};

Reply Staging::Upload::finish(const std::function<void(const std::filesystem::path&)>& deliver)
{
    if (!file_.is_open()) write(nullptr, 0);  // an empty payload still gets a file
    file_.close();
    if (!file_) throw std::runtime_error("cannot write staging file " + path_.string());
    if (written_ != length_) return errorReply(http::status::bad_request, "payload shorter than its Content-Length");
    if (hasher_.finish() != hash_) return errorReply(http::status::bad_request, "payload does not match its SHA-256");
    if (deliver) deliver(path_);  // a moved file is no longer there to remove
    else staging_.publish(hash_, path_, written_);
    finished_ = true;
    return jsonReply(Json{{"sha256", hash_}, {"length", written_}}.dump());
}

// Path segments and query parameters of a request target (no percent-escapes:
// every parameter is a UUID, an integer or a plain array name).
struct Target {
    std::vector<std::string> path;
    std::map<std::string, std::string> query;
    std::string param(const std::string& key) const
    {
        const auto found = query.find(key);
        if (found == query.end()) throw std::invalid_argument("missing query parameter " + key);
        return found->second;
    }
    int64_t integer(const std::string& key, std::optional<int64_t> fallback = std::nullopt) const
    {
        if (!query.contains(key) && fallback) return *fallback;
        const auto text = param(key);
        std::size_t used = 0;
        const auto value = std::stoll(text, &used);
        if (used != text.size()) throw std::invalid_argument("query parameter " + key + " is not an integer");
        return value;
    }
};

Target parseTarget(std::string_view target)
{
    Target result;
    const auto q = target.find('?');
    auto path = target.substr(0, q);
    while (!path.empty()) {
        if (path.front() == '/') { path.remove_prefix(1); continue; }
        const auto end = path.find('/');
        result.path.push_back(vcr::percent_decode(std::string(path.substr(0, end))));
        path.remove_prefix(end == std::string_view::npos ? path.size() : end);
    }
    if (q != std::string_view::npos) {
        auto query = target.substr(q + 1);
        while (!query.empty()) {
            const auto end = query.find('&');
            const auto item = query.substr(0, end);
            const auto eq = item.find('=');
            if (eq != std::string_view::npos) result.query.emplace(item.substr(0, eq), item.substr(eq + 1));
            query.remove_prefix(end == std::string_view::npos ? query.size() : end + 1);
        }
    }
    return result;
}

std::string uuidSegment(const std::string& text)
{
    if (!util::isUuid(text)) throw std::invalid_argument("not a UUID: " + text);
    return util::canonicalUuid(text);
}

using Request = http::request<http::string_body>;

std::filesystem::path stagingDirectory(const std::filesystem::path& store, const ServerOptions& options)
{
    return options.stagingDirectory.empty() ? std::filesystem::path(store.string() + ".staging") : options.stagingDirectory;
}

// What the last scheduled WAL checkpoints did (writer lane writes, control reads).
struct WalUpkeep {
    std::mutex mutex;
    std::optional<ObjectStore::WalCheckpoint> last;
    std::string lastAt;
    int64_t checkpoints = 0, truncations = 0, incompleteStreak = 0;
    bool reported = false;  // the current incomplete streak was logged
};

std::string utcNow()
{
    const auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &now);
#else
    gmtime_r(&now, &tm);
#endif
    char text[32];
    std::strftime(text, sizeof text, "%Y-%m-%dT%H:%M:%SZ", &tm);
    return text;
}

// A link_objects request being prepared or committed (submitPrepared).
struct InFlightLink {
    std::string digest;
    std::vector<std::function<void(Reply)>> waiters; // every connection awaiting its answer
};

struct ServerState {
    ServerOptions options;
    std::optional<ObjectStore> store;
    Staging staging;
    struct {
        std::mutex mutex;
        std::map<std::string, std::shared_ptr<InFlightLink>> inFlight; // by request UUID
        uint64_t detections = 0; // link requests whose detection this service ran (a replay runs none)
    } links;
    // Before io: connections still queued in io decrement them as io is destroyed.
    std::atomic<std::size_t> connections{0}, responseBytes{0};
    WalUpkeep wal;
    net::io_context io;
    tcp::acceptor acceptor;
    std::unique_ptr<Lane> writer, readers, control, transfers;
    std::vector<std::thread> network;
    std::thread checkpointer;
    std::mutex runMutex;
    std::condition_variable runCv;
    bool started = false, stopped = false, stopRequested = false;

    // The store opens (and takes service ownership) before staging is
    // touched: a second service refused ownership leaves the first's files.
    static ObjectStore openOwned(const std::filesystem::path& path, const ServerOptions& options)
    {
        OpenOptions open;
        open.author = options.author;
        open.serviceOwner = true;
        open.readerPoolSize = options.readWorkers + 2; // workers, control, writer-side receipt reads
        return ObjectStore::open(path, open);
    }

    static ObjectStore openWithLinking(const std::filesystem::path& path, const ServerOptions& options)
    {
#if VC_OBJECT_SERVER_LINKING
        object_links::enableLinkObjects();
#endif
        return openOwned(path, options);
    }

    ServerState(const std::filesystem::path& path, ServerOptions o)
        : options(std::move(o)), store(openWithLinking(path, options)),
          staging(stagingDirectory(path, options), options.maxStagedBytes, options.stagingLifetime), acceptor(io)
    {
        const tcp::endpoint endpoint(net::ip::make_address(options.listenAddress), options.port);
        acceptor.open(endpoint.protocol());
        acceptor.set_option(net::socket_base::reuse_address(true));
        acceptor.bind(endpoint);
        acceptor.listen();
    }

    // ---- Handlers: decode, call the public store API, encode. -----------

    Reply status()
    {
        auto txn = store->read();
        return jsonReply(storeStatusToJson(store->status(txn)));
    }

    Reply receipt(const std::string& requestUuid)
    {
        auto txn = store->read();
        const auto saved = store->receipt(txn, uuidSegment(requestUuid));
        if (!saved) return errorReply(http::status::not_found, "no receipt for request " + requestUuid);
        return jsonReply(transactionReceiptToJson(*saved));
    }

    // The service's own state; no database access.
    Reply service()
    {
        Json walJson{{"checkpoint_interval_ms", options.checkpointInterval.count()}};
        {
            std::lock_guard lock(wal.mutex);
            walJson["checkpoints"] = wal.checkpoints;
            walJson["truncations"] = wal.truncations;
            walJson["incomplete_streak"] = wal.incompleteStreak;
            if (wal.last)
                walJson["last"] = {{"at", wal.lastAt}, {"log_frames", wal.last->logFrames},
                                   {"checkpointed_frames", wal.last->checkpointedFrames}, {"busy", wal.last->busy},
                                   {"wal_bytes", wal.last->walBytes}};
        }
        return jsonReply(Json{
            {"connections", connections.load()}, {"max_connections", options.maxConnections},
            {"queues", {{"writer", writer->queued()}, {"read", readers->queued()}, {"control", control->queued()},
                        {"transfer", transfers->queued()}}},
            {"response_bytes_in_flight", responseBytes.load()},
            {"link_requests", linkReport()},
            {"staging", staging.report()}, {"wal", walJson}}.dump());
    }

    Json linkReport()
    {
        std::lock_guard lock(links.mutex);
        std::size_t waiting = 0;
        for (const auto& [uuid, entry] : links.inFlight) waiting += entry->waiters.size();
        return {{"in_flight", links.inFlight.size()}, {"waiting_connections", waiting}, {"detections", links.detections}};
    }

    // Writer lane: one scheduled checkpoint between transactions.
    void upkeepWal()
    {
        auto result = store->checkpointWal(ObjectStore::WalCheckpointMode::Passive);
        bool truncated = false;
        if (result.complete() && result.walBytes > options.walTruncateBytes) {
            result = store->checkpointWal(ObjectStore::WalCheckpointMode::Truncate, options.truncateWait);
            truncated = !result.busy;
        }
        std::lock_guard lock(wal.mutex);
        ++wal.checkpoints;
        wal.truncations += truncated;
        wal.last = result;
        wal.lastAt = utcNow();
        if (result.complete()) {
            wal.incompleteStreak = 0;
            wal.reported = false;
        } else if (++wal.incompleteStreak >= 6 && !wal.reported) {
            // Long-lived readers keep the WAL from being copied back: it grows.
            wal.reported = true;
            std::cerr << "vc_object_server: WAL checkpoints incomplete " << wal.incompleteStreak << " times in a row ("
                      << result.checkpointedFrames << "/" << result.logFrames << " frames, " << result.walBytes
                      << " bytes); long-lived readers are holding old snapshots\n";
        }
    }

    Reply submit(const std::string& body)
    {
        const auto receipt = store->submit(body, [this](const std::string& hash) { return staging.get(hash); });
        return jsonReply(transactionReceiptToJson(receipt));
    }

    // A request with a prepared operation (link_objects): its preparation
    // (the detection) runs here, on a read worker; the checked write is then
    // queued to the writer like any transaction, which holds the writer lane
    // only for the commit. A request whose payloads are not all staged takes
    // the ordinary path (a saved receipt or the missing hashes answer it).
    //
    // A link request in flight is remembered by its request UUID until it is
    // answered: a retry of it (a client whose stall timeout elapsed before
    // the detection was done, or a second client sending the same request)
    // waits for the running detection and gets its answer, instead of
    // starting the detection again; a same-UUID request with other contents
    // is refused, as submit refuses it once the receipt is saved.
    void submitPrepared(const std::shared_ptr<std::string>& body, const std::function<void(Reply)>& done)
    {
        auto request = std::make_shared<TransactionRequest>();
        bool parsed = false, unstaged = false;
        const auto parse = guarded([&] {
            TransactionPayloads payloads;
            for (const auto& hash : transactionJsonPayloadHashes(*body)) {
                auto bytes = staging.get(hash);
                if (!bytes) { unstaged = true; return Reply{}; }
                payloads.emplace(hash, std::move(*bytes));
            }
            *request = transactionRequestFromJson(*body, payloads);
            parsed = true;
            return Reply{};
        });
        if (unstaged) {
            if (!writer->tryPost([this, body, done] { done(guarded([&] { return submit(*body); })); }))
                done(errorReply(http::status::service_unavailable, "server busy; retry"));
            return;
        }
        if (!parsed) { done(parse); return; }
        const auto uuid = request->requestUuid;
        const auto digest = transactionRequestDigest(*request);
        {
            std::lock_guard lock(links.mutex);
            if (const auto running = links.inFlight.find(uuid); running != links.inFlight.end()) {
                if (running->second->digest != digest)
                    done(errorReply(http::status::bad_request, "request " + uuid + " is in flight with different contents"));
                else running->second->waiters.push_back(done);
                return;
            }
            auto entry = std::make_shared<InFlightLink>();
            entry->digest = digest;
            entry->waiters.push_back(done);
            links.inFlight.emplace(uuid, std::move(entry));
        }
        // Answers everyone waiting on this request and forgets it.
        auto finish = [this, uuid](const Reply& reply) {
            std::vector<std::function<void(Reply)>> waiters;
            {
                std::lock_guard lock(links.mutex);
                const auto running = links.inFlight.find(uuid);
                waiters.swap(running->second->waiters);
                links.inFlight.erase(running);
            }
            for (const auto& waiter : waiters) waiter(reply);
        };
        auto prepared = std::make_shared<PreparedOperations>();
        bool ready = false;
        const auto reply = guarded([&] {
            PrepareOptions prepare;
            prepare.localPaths = false;
            prepare.scratchDirectory = options.linkScratchDirectory;
            *prepared = store->prepare(*request, prepare);
            ready = true;
            return Reply{};
        });
        if (ready && !prepared->empty()) { std::lock_guard lock(links.mutex); ++links.detections; }
        if (!ready) { finish(reply); return; }
        const auto commit = [this, request, prepared, finish] {
            finish(guarded([&] { return jsonReply(transactionReceiptToJson(store->submit(*request, *prepared))); }));
        };
        if (!writer->tryPost(commit)) finish(errorReply(http::status::service_unavailable, "server busy; retry"));
    }

    Reply changes(const Target& target)
    {
        FeedCursor cursor{uuidSegment(target.param("store")), uuidSegment(target.param("generation")),
                          target.integer("sequence"), target.integer("offset", 0)};
        const auto max = std::clamp<int64_t>(target.integer("max", options.maxFeedChanges), 1, options.maxFeedChanges);
        auto txn = store->read();
        return jsonReply(changePageToJson(store->changesSince(txn, cursor, max)));
    }

    Reply currentObject(const std::string& uuid)
    {
        auto txn = store->read();
        const auto revision = store->currentObjectRevision(txn, uuidSegment(uuid));
        if (!revision) return errorReply(http::status::not_found, "no object " + uuid);
        return jsonReply(currentObjectResultToJson(store->status(txn), *revision));
    }

    Reply objectVersion(const std::string& uuid, const std::string& version, const Target& target)
    {
        std::size_t used = 0;
        const auto number = std::stoll(version, &used);
        if (used != version.size()) throw std::invalid_argument("version is not an integer");
        const ObjectVersion reference{uuidSegment(target.param("store")), uuidSegment(target.param("generation")),
                                      uuidSegment(uuid), number};
        auto txn = store->read();
        const auto revision = store->objectAtVersion(txn, reference);
        if (!revision) return errorReply(http::status::not_found, "version not recorded in this store and generation");
        return jsonReply(objectRevisionToJson(*revision));
    }

    Reply geometry(const std::string& revisionUuid)
    {
        auto txn = store->read();
        const auto revision = store->geometryRevision(txn, uuidSegment(revisionUuid));
        if (!revision) return errorReply(http::status::not_found, "no geometry revision " + revisionUuid);
        return jsonReply(geometryRevisionToJson(*revision));
    }

    Reply geometryObject(const std::string& revisionUuid)
    {
        auto txn=store->read();
        const auto object=store->objectAtGeometry(txn,uuidSegment(revisionUuid));
        return object?jsonReply(objectRevisionToJson(*object)):errorReply(http::status::not_found,"geometry metadata unavailable");
    }

    Reply geometryAnnotations(const std::string& revisionUuid, const std::string& array)
    {
        auto txn = store->read();
        const auto revision = uuidSegment(revisionUuid);
        if (!store->geometryRevision(txn, revision)) return errorReply(http::status::not_found, "no geometry revision " + revisionUuid);
        auto text = store->geometryPointAnnotations(txn, revision, array).dump();
        if (int64_t(text.size()) > options.maxResponseBytes)
            return errorReply(http::status::payload_too_large, "point annotations exceed the response limit");
        return jsonReply(std::move(text));
    }

    Reply nextPointId(const std::string& objectUuid,const std::string& array)
    {
        auto txn=store->read();
        return jsonReply(Json(store->nextGeometryPointId(txn,uuidSegment(objectUuid),array)).dump());
    }

    Reply geometryArray(const std::string& revisionUuid, const std::string& array, const Target& target)
    {
        const auto offset = target.integer("offset", 0);
        std::optional<int64_t> count;
        if (target.query.contains("count")) count = target.integer("count");
        auto txn = store->read();
        const auto revision = store->geometryRevision(txn, uuidSegment(revisionUuid));
        if (!revision) return errorReply(http::status::not_found, "no geometry revision " + revisionUuid);
        const auto info = std::find_if(revision->arrays.begin(), revision->arrays.end(),
                                       [&](const GeometryArrayInfo& a) { return a.name == array; });
        const auto size = count ? *count : info == revision->arrays.end() ? 0 : info->byteLength - offset;
        if (size > int64_t(options.maxResponseBytes))
            return jsonReply(Json{{"error", "geometry read exceeds the response limit of " +
                                            std::to_string(options.maxResponseBytes) + " bytes; read ranges"},
                                  {"max_response_bytes", options.maxResponseBytes}}.dump(),
                             http::status::payload_too_large);
        const auto bytes = store->readGeometryArray(txn, revision->uuid, array, offset, count);
        return {http::status::ok, "application/octet-stream", std::string(bytes.begin(), bytes.end())};
    }

    // Many geometry revisions in one reply (hosted decodes of many fibers):
    // request {"revisions": [uuid], "arrays": [name], "annotations": [name]};
    // reply = 8-byte little-endian header length, the JSON header, then the
    // arrays' bytes in header order. Revisions are answered in request order
    // until the response limit; "answered" says how many (at least one, unless
    // that one alone exceeds the limit: then its arrays are left out and the
    // client reads them by range).
    Reply queryGeometry(const std::string& body)
    {
        const auto request = Json::parse(body);
        const auto revisions = request.at("revisions").get<std::vector<std::string>>();
        const auto unique = [](std::vector<std::string> names) {
            std::sort(names.begin(), names.end());
            names.erase(std::unique(names.begin(), names.end()), names.end());
            return names;
        };
        const auto arrays = unique(request.value("arrays", std::vector<std::string>{}));
        const auto annotations = unique(request.value("annotations", std::vector<std::string>{}));
        if (revisions.size() > size_t(options.maxQueryResults))
            throw std::invalid_argument("geometry query: at most " + std::to_string(options.maxQueryResults) + " revisions");
        auto txn = store->read();
        Json header = Json::object();
        header["revisions"] = Json::array();
        std::string data;
        size_t answered = 0;
        // Bounded: each entry's arrays are sized from its descriptor before any
        // array is read, annotations are estimated (per point) before they are
        // loaded, and the entry's actual encoded size is checked before it is
        // added. An entry that alone would exceed the limit is answered without
        // data (or, when even its descriptor is too large, as {"uuid",
        // "oversized": true}); the client reads those individually.
        const size_t limit = size_t(options.maxResponseBytes);
        const size_t headerOverhead = 64;
        size_t used = 8 + headerOverhead;
        // The smallest answer (one compact entry) must fit, or nothing can.
        if (!revisions.empty()) {
            const auto compact = Json{{"uuid", revisions.front()}, {"descriptor", nullptr}, {"oversized", true},
                                      {"arrays", Json::object()}, {"annotations", Json::object()}}.dump().size() + 2;
            if (used + compact > limit)
                return errorReply(http::status::payload_too_large, "geometry query: the response limit is smaller than one entry");
        }
        const auto annotationEstimate = [](const GeometryArrayInfo& info) {
            return size_t(info.shape.empty() ? 0 : std::max<int64_t>(info.shape.front(), 0)) * 512 + 64;
        };
        for (const auto& requested : revisions) {
            Json entry = Json::object();
            entry["uuid"] = requested;
            const auto revision = store->geometryRevision(txn, uuidSegment(requested));
            std::string arrayBytes;
            if (!revision) {
                entry["descriptor"] = nullptr;
            } else {
                entry["descriptor"] = Json::parse(geometryRevisionToJson(*revision));
                const auto infoOf = [&](const std::string& name) -> const GeometryArrayInfo* {
                    const auto found = std::find_if(revision->arrays.begin(), revision->arrays.end(),
                                                    [&](const GeometryArrayInfo& a) { return a.name == name; });
                    return found == revision->arrays.end() ? nullptr : &*found;
                };
                const size_t base = entry.dump().size() + 64;
                size_t planned = base;
                for (const auto& name : arrays)
                    if (const auto* info = infoOf(name)) planned += size_t(info->byteLength) + name.size() + 64;
                for (const auto& name : annotations)
                    if (const auto* info = infoOf(name)) planned += annotationEstimate(*info);
                if (answered > 0 && used + planned > limit) break;
                Json arrayIndex = Json::object();
                Json annotationValues = Json::object();
                if (used + planned <= limit) {
                    // Annotations only for arrays of at most kBatchAnnotationPoints
                    // points (fibers' control points), read with a byte budget
                    // checked against each value's packed size before it is
                    // loaded; otherwise left for the client's own read.
                    constexpr int64_t kBatchAnnotationPoints = 4096;
                    for (const auto& name : annotations) {
                        const auto* info = infoOf(name);
                        if (!info || info->shape.empty() || info->shape.front() > kBatchAnnotationPoints) continue;
                        // Read only up to what the response can still carry
                        // (one budget shared by this entry's annotation arrays).
                        const size_t spent = annotationValues.dump().size();
                        const size_t room = limit > used + base + spent ? limit - used - base - spent : 0;
                        auto value = store->geometryPointAnnotationsWithin(txn, revision->uuid, name, room);
                        if (!value) continue;
                        annotationValues[name] = std::move(*value);
                    }
                    for (const auto& name : arrays) {
                        if (!infoOf(name)) continue;
                        const auto bytes = store->readGeometryArray(txn, revision->uuid, name, 0, std::nullopt);
                        arrayIndex[name] = {{"offset", data.size() + arrayBytes.size()}, {"length", bytes.size()}};
                        arrayBytes.append(bytes.begin(), bytes.end());
                    }
                }
                entry["arrays"] = std::move(arrayIndex);
                entry["annotations"] = std::move(annotationValues);
            }
            if (!entry.contains("arrays")) entry["arrays"] = Json::object();
            if (!entry.contains("annotations")) entry["annotations"] = Json::object();
            size_t actual = entry.dump().size() + 2 + arrayBytes.size();
            if (used + actual > limit) {
                if (answered > 0) break;
                // Alone too large after all (an estimate fell short): no data.
                arrayBytes.clear();
                entry["arrays"] = Json::object();
                entry["annotations"] = Json::object();
                actual = entry.dump().size() + 2;
                if (used + actual > limit) {
                    entry = Json{{"uuid", requested}, {"descriptor", nullptr}, {"oversized", true},
                                 {"arrays", Json::object()}, {"annotations", Json::object()}};
                    actual = entry.dump().size() + 2;
                }
            }
            used += actual;
            data += arrayBytes;
            header["revisions"].push_back(std::move(entry));
            ++answered;
        }
        header["answered"] = answered;
        const auto headerText = header.dump();
        std::string reply(8, '\0');
        uint64_t length = headerText.size();
        for (int i = 0; i < 8; ++i) reply[size_t(i)] = char((length >> (8 * i)) & 0xff);
        reply += headerText;
        reply += data;
        return {http::status::ok, "application/octet-stream", std::move(reply)};
    }

    Reply artifactInfo(const std::string& hash)
    {
        if (!isArtifactHash(hash)) return errorReply(http::status::bad_request, "not a lowercase SHA-256: " + hash);
        const auto length = store->artifactLength(hash);
        if (!length) return errorReply(http::status::not_found, "this store does not hold artifact " + hash);
        return jsonReply(Json{{"sha256", hash}, {"length", *length}}.dump());
    }

    Reply artifactBytes(const std::string& hash, const Target& target)
    {
        if (!isArtifactHash(hash)) return errorReply(http::status::bad_request, "not a lowercase SHA-256: " + hash);
        const auto length = store->artifactLength(hash);
        if (!length) return errorReply(http::status::not_found, "this store does not hold artifact " + hash);
        const auto offset = target.integer("offset", 0);
        std::optional<uint64_t> count;
        if (target.query.contains("count")) count = uint64_t(std::max<int64_t>(target.integer("count"), 0));
        if (offset < 0) return errorReply(http::status::bad_request, "negative offset");
        const auto size = count ? *count : *length - std::min<uint64_t>(uint64_t(offset), *length);
        if (size > options.maxResponseBytes)
            return jsonReply(Json{{"error", "artifact read exceeds the response limit of " +
                                            std::to_string(options.maxResponseBytes) + " bytes; read ranges"},
                                  {"max_response_bytes", options.maxResponseBytes}}.dump(),
                             http::status::payload_too_large);
        const auto bytes = store->readArtifact(hash, uint64_t(offset), count);
        return {http::status::ok, "application/octet-stream", std::string(bytes.begin(), bytes.end())};
    }

    // Query pages are bounded: an absent or larger limit is refused, not cut.
    void requirePage(const std::optional<int64_t>& limit) const
    {
        if (!limit || *limit > options.maxQueryResults)
            throw std::invalid_argument("query limit is required and at most " + std::to_string(options.maxQueryResults) +
                                        "; page larger selections");
    }

    Reply queryObjects(const std::string& body)
    {
        std::optional<std::vector<std::string>> fields;
        const auto selection = objectQueryRequestFromJson(body, &fields);
        requirePage(selection.limit);
        auto txn = store->read();
        return jsonReply(objectQueryResultToJson(store->status(txn), store->selectObjects(txn, selection, fields), fields));
    }

    Reply queryEdges(const std::string& body)
    {
        std::optional<std::vector<std::string>> fields;
        const auto selection = edgeQueryRequestFromJson(body, &fields);
        requirePage(selection.limit);
        auto txn = store->read();
        return jsonReply(edgeQueryResultToJson(store->status(txn), store->selectEdges(txn, selection, fields), fields));
    }

    Reply queryCount(const std::string& body)
    {
        const auto query = kindQueryRequestFromJson(body);
        if (!query.kind) throw std::invalid_argument("an object count needs a kind");
        auto txn = store->read();
        const auto count = query.unboundVolume ? store->unboundObjectCount(txn, *query.kind, *query.unboundVolume)
                                               : store->objectCount(txn, *query.kind, query.includeRetired);
        return jsonReply(objectCountResultToJson(store->status(txn), count));
    }

    Reply queryTags(const std::string& body)
    {
        const auto query = kindQueryRequestFromJson(body);
        if (query.includeRetired || query.unboundVolume) throw std::invalid_argument("tags take only a kind");
        auto txn = store->read();
        return jsonReply(objectTagsResultToJson(store->status(txn), store->objectTags(txn, query.kind)));
    }

    Reply queryGraph(const std::string& body)
    {
        const auto query = graphQueryRequestFromJson(body);
        auto txn = store->read();
        const auto uuids = query.kind == GraphQuery::Kind::Component
            ? store->componentOf(txn, query.objectUuid, query.edgeKinds, nullptr, query.includeRetired)
            : store->neighbours(txn, query.objectUuid,
                                query.edgeKinds.empty() ? std::nullopt : std::optional(query.edgeKinds.front()),
                                query.neighbourKind, query.includeRetired);
        return jsonReply(uuidListResultToJson(store->status(txn), uuids));
    }

    Reply objectHistory(const std::string& uuid, const Target& target)
    {
        auto txn = store->read();
        const auto history = store->objectHistory(txn, uuidSegment(uuid), target.integer("if_retained", 0) != 0);
        if (history.empty()) return errorReply(http::status::not_found, "no object " + uuid);
        return jsonReply(objectHistoryToJson(store->status(txn), history));
    }

    Reply queryRelationships(const std::string& body)
    {
        const auto versions = objectVersionsRequestFromJson(body);
        auto txn = store->read();
        return jsonReply(relationshipQueryResultToJson(store->status(txn), store->relationshipsAt(txn, versions)));
    }

    Reply querySelectedInputs(const std::string& body)
    {
        const auto [versions, position] = selectedInputsRequestFromJson(body);
        auto txn = store->read();
        return jsonReply(selectedInputsToJson(store->status(txn), LocalStoreReader(*store, txn).selectedInputs(versions, position)));
    }

    Reply queryVersions(const std::string& body)
    {
        const auto versions = objectVersionsRequestFromJson(body);
        if (int64_t(versions.size()) > options.maxQueryResults)
            throw std::invalid_argument("at most " + std::to_string(options.maxQueryResults) +
                                        " versions per request; send larger batches in parts");
        auto txn = store->read();
        return jsonReply(objectsAtVersionsResultToJson(store->status(txn), store->objectsAtVersions(txn, versions)));
    }

    Reply queryRegion(const std::string& body)
    {
        auto [slice, region] = regionQueryRequestFromJson(body);
        region.outputByteBudget = std::min(region.outputByteBudget, options.maxRegionOutputBytes);
        region.processingByteBudget = std::min(region.processingByteBudget, options.maxRegionProcessingBytes);
        region.candidateWorkBudget = std::min(region.candidateWorkBudget, options.maxRegionCandidateWork);
        region.sqlStepBudget = std::min(region.sqlStepBudget, options.maxRegionSqlSteps);
        auto txn = store->read();
        return jsonReply(regionQueryResultToJson(store->status(txn), store->geometryRegion(txn, slice, region)));
    }

    Reply queryNearestSurfaces(const std::string& body)
    {
        const auto query = proximityQueryRequestFromJson(body);
        auto txn = store->read();
        return jsonReply(proximityQueryResultToJson(store->status(txn), store->nearestSurfaces(txn, query)));
    }

    // ---- Routing ------------------------------------------------------------

    struct Route {
        Lane* lane = nullptr;
        std::function<Reply()> work;
        bool busy = false;
        // Set instead of `work` for a request that answers from another lane
        // (submitPrepared): runs on `lane` and calls done when it has a reply.
        std::function<void(std::function<void(Reply)>)> continued;
    };

    Route route(const Request& request)
    {
        const auto raw = request.target();
        auto target = std::make_shared<Target>(parseTarget(std::string_view(raw.data(), raw.size())));
        const auto& p = target->path;
        const auto method = request.method();
        const auto body = std::make_shared<std::string>(request.body());
        auto is = [&](std::initializer_list<const char*> parts) {
            if (p.size() != parts.size()) return false;
            std::size_t i = 0;
            for (const char* part : parts) {
                if (*part != '*' && p[i] != part) return false;
                ++i;
            }
            return true;
        };
        if (method == http::verb::get && is({"v1", "status"})) return {control.get(), [this] { return status(); }};
        if (method == http::verb::get && is({"v1", "service"})) return {control.get(), [this] { return service(); }};
        if (method == http::verb::get && is({"v1", "receipts", "*"}))
            return {control.get(), [this, target] { return receipt(target->path[2]); }};
        if (method == http::verb::post && is({"v1", "transactions"})) {
            if (transactionJsonHasPreparedOperations(*body)) {
                Route route;
                route.lane = readers.get();
                route.continued = [this, body](std::function<void(Reply)> done) { submitPrepared(body, done); };
                return route;
            }
            return {writer.get(), [this, body] { return submit(*body); }};
        }
        if (method == http::verb::get && is({"v1", "changes"}))
            return {readers.get(), [this, target] { return changes(*target); }};
        if (method == http::verb::get && is({"v1", "volume-transforms"}))
            return {readers.get(), [this] {
                auto txn = store->read();
                std::vector<std::optional<ObjectRevision>> selected;
                for (const auto* kind : {"volume_transform", "volume_transform_choice"})
                    for (const auto& object : store->objectsByKind(txn, kind))
                        selected.push_back(store->currentObjectRevision(txn, object.uuid));
                return jsonReply(objectsAtVersionsResultToJson(store->status(txn), selected));
            }};
        if (method == http::verb::get && is({"v1", "objects", "*"}))
            return {readers.get(), [this, target] { return currentObject(target->path[2]); }};
        if (method == http::verb::get && is({"v1", "objects", "*", "versions", "*"}))
            return {readers.get(), [this, target] { return objectVersion(target->path[2], target->path[4], *target); }};
        if (method == http::verb::get && is({"v1", "objects", "*", "history"}))
            return {readers.get(), [this, target] { return objectHistory(target->path[2], *target); }};
        if (method == http::verb::post && is({"v1", "query", "objects"}))
            return {readers.get(), [this, body] { return queryObjects(*body); }};
        if (method == http::verb::post && is({"v1", "query", "edges"}))
            return {readers.get(), [this, body] { return queryEdges(*body); }};
        if (method == http::verb::post && is({"v1", "query", "relationships"}))
            return {readers.get(), [this, body] { return queryRelationships(*body); }};
        if (method == http::verb::post && is({"v1", "query", "selected-inputs"}))
            return {readers.get(), [this, body] { return querySelectedInputs(*body); }};
        if (method == http::verb::post && is({"v1", "query", "count"}))
            return {readers.get(), [this, body] { return queryCount(*body); }};
        if (method == http::verb::post && is({"v1", "query", "graph"}))
            return {readers.get(), [this, body] { return queryGraph(*body); }};
        if (method == http::verb::post && is({"v1", "query", "tags"}))
            return {readers.get(), [this, body] { return queryTags(*body); }};
        if (method == http::verb::post && is({"v1", "query", "versions"}))
            return {readers.get(), [this, body] { return queryVersions(*body); }};
        if (method == http::verb::post && is({"v1", "query", "versions-at"}))
            return {readers.get(), [this, body] {
                const auto anchors = objectVersionsRequestFromJson(*body);
                if (anchors.size() != 1) throw std::invalid_argument("version snapshot needs one anchor");
                auto txn = store->read();
                return jsonReply(objectVersionsRequestToJson(store->versionsAt(txn, anchors.front())));
            }};
        if (method == http::verb::post && is({"v1", "query", "storage-path"}))
            return {readers.get(), [this, body] {
                const auto request = Json::parse(*body);
                std::optional<std::filesystem::path> root;
                if (!request.at("dataset_root").is_null()) root = request.at("dataset_root").get<std::string>();
                auto txn = store->read();
                const auto path = store->resolveStoragePath(txn, request.at("uuid").get<std::string>(), root);
                return jsonReply((path ? Json(path->string()) : Json(nullptr)).dump());
            }};
        if (method == http::verb::post && is({"v1", "query", "region"}))
            return {readers.get(), [this, body] { return queryRegion(*body); }};
        if (method == http::verb::post && is({"v1", "query", "nearest_surfaces"}))
            return {readers.get(), [this, body] { return queryNearestSurfaces(*body); }};
        if (method == http::verb::post && is({"v1", "query", "geometry"})) {
            if (responseBytes.load() > options.maxResponseBytesInFlight) return {nullptr, nullptr, true};
            return {readers.get(), [this, body] { return queryGeometry(*body); }};
        }
        if (method == http::verb::get && is({"v1", "geometry", "*"}))
            return {readers.get(), [this, target] { return geometry(target->path[2]); }};
        if (method == http::verb::get && is({"v1", "geometry", "*", "object"}))
            return {readers.get(), [this, target] { return geometryObject(target->path[2]); }};
        if (method == http::verb::get && is({"v1", "artifacts", "*", "info"}))
            return {transfers.get(), [this, target] { return artifactInfo(target->path[2]); }};
        if (method == http::verb::get && is({"v1", "artifacts", "*"})) {
            if (responseBytes.load() > options.maxResponseBytesInFlight) return {nullptr, nullptr, true};
            return {transfers.get(), [this, target] { return artifactBytes(target->path[2], *target); }};
        }
        if (method == http::verb::get && is({"v1", "geometry", "*", "annotations", "*"}))
            return {readers.get(), [this, target] { return geometryAnnotations(target->path[2], target->path[4]); }};
        if (method == http::verb::get && is({"v1", "objects", "*", "next-point-id", "*"}))
            return {readers.get(), [this, target] { return nextPointId(target->path[2], target->path[4]); }};
        if (method == http::verb::get && is({"v1", "geometry", "*", "arrays", "*"})) {
            // Bounded download memory: unsent replies stay below the limit
            // (plus the replies being produced, each maxResponseBytes at most).
            if (responseBytes.load() > options.maxResponseBytesInFlight) return {nullptr, nullptr, true};
            return {readers.get(), [this, target] { return geometryArray(target->path[2], target->path[4], *target); }};
        }
        return {};
    }

    static Reply guarded(const std::function<Reply()>& work)
    {
        try {
            return work();
        } catch (const TransactionConflict& conflict) {
            return jsonReply(transactionConflictToJson(conflict), http::status::conflict);
        } catch (const StoreBusy& e) {
            return jsonReply(Json{{"error", e.what()}, {"store_busy", true}}.dump(), http::status::conflict);
        } catch (const MissingPayloads& missing) {
            return jsonReply(Json{{"error", missing.what()}, {"missing", missing.hashes()}}.dump(),
                             http::status::unprocessable_entity);
        } catch (const PreparationFailed& failed) {
            // Not the service's fault and not transient: no retry.
            return jsonReply(Json{{"error", failed.what()}, {"preparation_failed", true}}.dump(),
                             http::status::unprocessable_entity);
        } catch (const std::invalid_argument& e) {
            return errorReply(http::status::bad_request, e.what());
        } catch (const std::out_of_range& e) {
            return errorReply(http::status::bad_request, e.what());
        } catch (const nlohmann::json::exception& e) {
            return errorReply(http::status::bad_request, e.what());
        } catch (const std::exception& e) {
            std::cerr << "vc_object_server: internal error: " << e.what() << "\n";
            return errorReply(http::status::internal_server_error, e.what());
        }
    }

    // Runs `request` on its lane and calls `done` there with the reply.
    void dispatch(const Request& request, std::function<void(Reply)> done)
    {
        Route r;
        try {
            r = route(request);
        } catch (const std::exception& e) {
            done(errorReply(http::status::bad_request, e.what()));
            return;
        }
        if (r.busy) {
            done(errorReply(http::status::service_unavailable, "server busy; retry"));
            return;
        }
        if (!r.lane) {
            done(errorReply(http::status::not_found, "unknown endpoint"));
            return;
        }
        std::function<void()> job;
        if (r.continued) job = [continued = std::move(r.continued), done] { continued(done); };
        else job = [work = std::move(r.work), done] { done(guarded(work)); };
        if (!r.lane->tryPost(job)) done(errorReply(http::status::service_unavailable, "server busy; retry"));
    }

    // Streamed uploads: "PUT /v1/payloads/<sha256>" (staged transaction
    // payloads) and "PUT /v1/artifacts/<sha256>" (artifact bytes, kept).
    struct UploadTarget {
        std::string hash;
        bool artifact = false;
    };
    static std::optional<UploadTarget> uploadTarget(const http::request_header<>& header)
    {
        if (header.method() != http::verb::put) return std::nullopt;
        const auto raw = header.target();
        const auto target = parseTarget(std::string_view(raw.data(), raw.size()));
        if (target.path.size() != 3 || target.path[0] != "v1") return std::nullopt;
        if (target.path[1] == "payloads") return UploadTarget{target.path[2], false};
        if (target.path[1] == "artifacts") return UploadTarget{target.path[2], true};
        return std::nullopt;
    }

    // Transfer lane: a verified artifact upload becomes durable store bytes.
    void keepArtifact(const std::filesystem::path& file) { (void)store->putArtifact(file, true); }

    void scheduleCheckpoints()
    {
        if (options.checkpointInterval.count() <= 0) return;
        checkpointer = std::thread([this] {
            std::unique_lock lock(runMutex);
            while (!runCv.wait_for(lock, options.checkpointInterval, [this] { return stopped; })) {
                // A full writer queue skips this round; transactions come first.
                writer->tryPost([this] {
                    try {
                        upkeepWal();
                    } catch (const std::exception& e) {
                        std::cerr << "vc_object_server: WAL checkpoint failed: " << e.what() << "\n";
                    }
                });
            }
        });
    }

    void accept();
};

class Connection : public std::enable_shared_from_this<Connection> {
public:
    Connection(tcp::socket socket, ServerState& server) : stream_(std::move(socket)), server_(server) { ++server_.connections; }
    ~Connection() { --server_.connections; }
    void start() { readHeader(); }
private:
    void readHeader()
    {
        header_.emplace();
        header_->header_limit(64 * 1024);
        // Body limits depend on the request; they are set once the header names it.
        header_->body_limit(std::numeric_limits<std::uint64_t>::max());
        stream_.expires_after(server_.options.requestDeadline);
        http::async_read_header(stream_, buffer_, *header_,
            [self = shared_from_this()](beast::error_code ec, std::size_t) { self->onHeader(ec); });
    }
    void onHeader(beast::error_code ec)
    {
        if (ec) return close();
        keepAlive_ = header_->get().keep_alive();
        version_ = header_->get().version();
        if (const auto upload = ServerState::uploadTarget(header_->get())) return startUpload(*upload);
        // Beast checked Content-Length while parsing the unrestricted header.
        // Lowering body_limit now only bounds chunked bodies.
        if (const auto length = header_->content_length(); length && *length > server_.options.maxCommandBytes)
            return refuse(http::status::payload_too_large, "request body exceeds the server limit");
        command_.emplace(std::move(*header_));
        command_->body_limit(server_.options.maxCommandBytes);
        http::async_read(stream_, buffer_, *command_,
            [self = shared_from_this()](beast::error_code ec, std::size_t) { self->onCommand(ec); });
    }
    void onCommand(beast::error_code ec)
    {
        if (ec == http::error::body_limit) return refuse(http::status::payload_too_large, "request body exceeds the server limit");
        if (ec) return close();
        auto request = command_->release();
        command_.reset();
        // Wait for the reply before reading on: one request at a time per connection.
        stream_.expires_never();
        server_.dispatch(request, [self = shared_from_this()](Reply reply) {
            net::post(self->stream_.get_executor(), [self, reply = std::move(reply)]() mutable { self->write(std::move(reply)); });
        });
    }

    // ---- Payload uploads: streamed to a staging file in bounded chunks. ----

    void startUpload(const ServerState::UploadTarget& target)
    {
        const auto& hash = target.hash;
        artifact_ = target.artifact;
        if (!Staging::isHash(hash)) return refuse(http::status::bad_request, "payload name is not a lowercase SHA-256");
        const auto length = header_->content_length();
        if (!length) return refuse(http::status::length_required, "payload uploads need a Content-Length");
        if (*length > server_.options.maxPayloadBytes)
            return refuse(http::status::payload_too_large, "payload exceeds the server limit of " +
                          std::to_string(server_.options.maxPayloadBytes) + " bytes");
        upload_ = server_.staging.begin(hash, static_cast<std::size_t>(*length));
        if (!upload_) return refuse(http::status::insufficient_storage, "payload staging is full; retry later");
        body_.emplace(std::move(*header_));
        body_->body_limit(*length);
        chunk_.resize(std::max<std::size_t>(server_.options.uploadChunkBytes, 4096));
        if (body_->is_done()) {  // empty payload: nothing to read
            body_->get().body().size = chunk_.size();
            return onChunk({});
        }
        readChunk();
    }
    void readChunk()
    {
        auto& body = body_->get().body();
        body.data = chunk_.data();
        body.size = chunk_.size();
        stream_.expires_after(server_.options.requestDeadline);
        http::async_read_some(stream_, buffer_, *body_,
            [self = shared_from_this()](beast::error_code ec, std::size_t) { self->onChunk(ec); });
    }
    void onChunk(beast::error_code ec)
    {
        if (ec == http::error::need_buffer) ec = {};
        if (ec) {
            upload_.reset();  // removes the partial file and its reservation
            return close();
        }
        const auto used = chunk_.size() - body_->get().body().size;
        const bool done = body_->is_done();
        if (!used && !done) return readChunk();
        // File I/O and hashing run on the transfer lane; the next chunk is
        // read only after this one is written, so a slow disk slows the sender.
        stream_.expires_never();
        auto job = [self = shared_from_this(), used, done] {
            std::optional<Reply> reply;
            try {
                self->upload_->write(self->chunk_.data(), used);
                if (done && self->artifact_)
                    reply = self->upload_->finish([&](const std::filesystem::path& file) { self->server_.keepArtifact(file); });
                else if (done)
                    reply = self->upload_->finish();
            } catch (const std::exception& e) {
                std::cerr << "vc_object_server: staging failed: " << e.what() << "\n";
                reply = errorReply(http::status::internal_server_error, e.what());
            }
            net::post(self->stream_.get_executor(), [self, done, reply = std::move(reply)]() mutable {
                if (!reply) return self->readChunk();
                self->upload_.reset();
                self->body_.reset();
                // A failure before the end leaves unread body bytes: close.
                if (!done) self->keepAlive_ = false;
                self->write(std::move(*reply));
            });
        };
        // The lane holds one job per connection at most, so this only fails
        // while the server stops.
        if (!server_.transfers->tryPost(std::move(job))) {
            upload_.reset();
            return refuse(http::status::service_unavailable, "server busy; retry");
        }
    }

    // Answers without reading the rest of the request, then closes.
    void refuse(http::status status, const std::string& message)
    {
        keepAlive_ = false;
        write(errorReply(status, message));
    }

    void write(Reply reply)
    {
        auto response = std::make_shared<http::response<http::string_body>>(reply.status, version_);
        response->set(http::field::server, "vc_object_server");
        response->set(http::field::content_type, reply.contentType);
        if (reply.status == http::status::service_unavailable || reply.status == http::status::insufficient_storage)
            response->set(http::field::retry_after, "1");
        response->keep_alive(keepAlive_);
        response->body() = std::move(reply.body);
        response->prepare_payload();
        const auto bytes = response->body().size();
        server_.responseBytes += bytes;
        stream_.expires_after(server_.options.requestDeadline);
        http::async_write(stream_, *response, [self = shared_from_this(), response, bytes](beast::error_code ec, std::size_t) {
            self->server_.responseBytes -= bytes;
            if (ec || !self->keepAlive_) return self->close();
            self->readHeader();
        });
    }
    void close()
    {
        beast::error_code ignored;
        stream_.socket().shutdown(tcp::socket::shutdown_both, ignored);
    }

    beast::tcp_stream stream_;
    beast::flat_buffer buffer_;
    std::optional<http::request_parser<http::empty_body>> header_;
    std::optional<http::request_parser<http::string_body>> command_;
    std::optional<http::request_parser<http::buffer_body>> body_;
    std::unique_ptr<Staging::Upload> upload_;
    bool artifact_ = false;
    std::vector<char> chunk_;
    ServerState& server_;
    bool keepAlive_ = true;
    unsigned version_ = 11;
};
void ServerState::accept()
{
    acceptor.async_accept(net::make_strand(io), [this](beast::error_code ec, tcp::socket socket) {
        if (ec) return; // acceptor closed
        if (connections.load() >= options.maxConnections) {
            beast::error_code ignored;
            socket.close(ignored);
        } else {
            std::make_shared<Connection>(std::move(socket), *this)->start();
        }
        accept();
    });
}
} // namespace

struct Server::Impl : ServerState {
    using ServerState::ServerState;
};

Server::Server(const std::filesystem::path& store, ServerOptions options)
    : impl_(std::make_unique<Impl>(store, std::move(options)))
{
}

Server::~Server() { stop(); }

uint16_t Server::port() const { return impl_->acceptor.local_endpoint().port(); }

void Server::start()
{
    auto& s = *impl_;
    std::lock_guard lock(s.runMutex);
    if (s.started) return;
    s.started = true;
    s.writer = std::make_unique<Lane>(1, s.options.writeQueue);
    s.readers = std::make_unique<Lane>(s.options.readWorkers, s.options.readQueue);
    s.control = std::make_unique<Lane>(1, s.options.controlQueue);
    s.transfers = std::make_unique<Lane>(s.options.transferWorkers, s.options.maxConnections + 1);
    s.accept();
    s.scheduleCheckpoints();
    for (int i = 0; i < std::max(s.options.networkThreads, 1); ++i) s.network.emplace_back([&s] { s.io.run(); });
}

void Server::stop()
{
    auto& s = *impl_;
    {
        std::lock_guard lock(s.runMutex);
        if (!s.started || s.stopped) {
            s.stopped = true;
            s.runCv.notify_all();
            return;
        }
        s.stopped = true;
    }
    s.runCv.notify_all();
    net::post(s.acceptor.get_executor(), [&s] {
        beast::error_code ignored;
        s.acceptor.close(ignored);
    });
    if (s.checkpointer.joinable()) s.checkpointer.join();
    // Finish accepted storage work; its replies go to sockets still served.
    s.transfers->stop();
    s.writer->stop();
    s.readers->stop();
    s.control->stop();
    s.io.stop();
    for (auto& t : s.network)
        if (t.joinable()) t.join();
    s.store.reset();
}

void Server::run()
{
    start();
    auto& s = *impl_;
    net::signal_set signals(s.io, SIGINT, SIGTERM);
    signals.async_wait([&s](beast::error_code ec, int) {
        if (ec) return;
        std::lock_guard lock(s.runMutex);
        s.stopRequested = true;
        s.runCv.notify_all();
    });
    {
        std::unique_lock lock(s.runMutex);
        s.runCv.wait(lock, [&] { return s.stopped || s.stopRequested; });
    }
    signals.cancel();
    stop();
}

} // namespace vc::objects::server
