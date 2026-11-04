// HTTP adapter over the public ObjectStore APIs. Network threads only parse
// and write messages; every storage call runs on a bounded lane: one writer
// for checked transactions, read workers for reads/feed/staging, and a
// control lane so status and receipt checks stay responsive while the others
// are busy. A full lane answers 503 (clients retry with backoff).
#include "vc_object_store/server/Server.hpp"

#include "vc_object_store/Uuid.hpp"
#include "vc_object_store/objects/ObjectStore.hpp"
#include "vc_object_store/objects/TransactionJson.hpp"

#include <boost/asio.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <iostream>
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

// Uploaded payload bytes by SHA-256, until they expire. Staging is not
// part of a request's identity: an expired payload is uploaded again.
class Staging {
public:
    Staging(std::size_t limit, std::chrono::seconds lifetime) : limit_(limit), lifetime_(lifetime) {}
    bool put(const std::string& hash, std::vector<uint8_t> bytes)
    {
        std::lock_guard lock(mutex_);
        const auto now = Clock::now();
        if (auto found = entries_.find(hash); found != entries_.end()) {
            found->second.expires = now + lifetime_;
            return true;
        }
        purge(now);
        if (bytes.size() > limit_ - total_) return false;
        total_ += bytes.size();
        entries_.emplace(hash, Entry{std::move(bytes), now + lifetime_});
        return true;
    }
    std::optional<std::vector<uint8_t>> get(const std::string& hash)
    {
        std::lock_guard lock(mutex_);
        const auto found = entries_.find(hash);
        if (found == entries_.end() || found->second.expires <= Clock::now()) return std::nullopt;
        return found->second.bytes;
    }
private:
    struct Entry {
        std::vector<uint8_t> bytes;
        Clock::time_point expires;
    };
    void purge(Clock::time_point now)
    {
        for (auto it = entries_.begin(); it != entries_.end();) {
            if (it->second.expires <= now) {
                total_ -= it->second.bytes.size();
                it = entries_.erase(it);
            } else ++it;
        }
    }
    std::size_t limit_, total_ = 0;
    std::chrono::seconds lifetime_;
    std::mutex mutex_;
    std::map<std::string, Entry> entries_;
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
        result.path.emplace_back(path.substr(0, end));
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

struct ServerState {
    ServerOptions options;
    std::optional<ObjectStore> store;
    Staging staging;
    // Before io: connections still queued in io decrement it as io is destroyed.
    std::atomic<std::size_t> connections{0};
    net::io_context io;
    tcp::acceptor acceptor;
    std::unique_ptr<Lane> writer, readers, control;
    std::vector<std::thread> network;
    std::mutex runMutex;
    std::condition_variable runCv;
    bool started = false, stopped = false, stopRequested = false;

    ServerState(const std::filesystem::path& path, ServerOptions o)
        : options(std::move(o)), staging(options.maxStagedBytes, options.stagingLifetime), acceptor(io)
    {
        OpenOptions open;
        open.author = options.author;
        open.serviceOwner = true;
        open.readerPoolSize = options.readWorkers + 2; // workers, control, writer-side receipt reads
        store.emplace(ObjectStore::open(path, open));
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

    Reply stage(const std::string& hash, const std::string& body)
    {
        std::vector<uint8_t> bytes(body.begin(), body.end());
        if (payloadHash(bytes) != hash) return errorReply(http::status::bad_request, "payload does not match its SHA-256");
        if (!staging.put(hash, std::move(bytes)))
            return errorReply(http::status::insufficient_storage, "payload staging is full; retry later");
        return jsonReply(Json{{"sha256", hash}, {"length", body.size()}}.dump());
    }

    Reply submit(const std::string& body)
    {
        const auto receipt = store->submit(body, [this](const std::string& hash) { return staging.get(hash); });
        return jsonReply(transactionReceiptToJson(receipt));
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
        return jsonReply(Json{{"status", Json::parse(storeStatusToJson(store->status(txn)))},
                              {"revision", Json::parse(objectRevisionToJson(*revision))}}.dump());
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

    Reply geometryArray(const std::string& revisionUuid, const std::string& array, const Target& target)
    {
        const auto offset = target.integer("offset", 0);
        std::optional<int64_t> count;
        if (target.query.contains("count")) count = target.integer("count");
        if (count && *count > int64_t(options.maxPayloadBytes))
            return errorReply(http::status::payload_too_large, "geometry read exceeds the response limit; read ranges");
        auto txn = store->read();
        const auto revision = store->geometryRevision(txn, uuidSegment(revisionUuid));
        if (!revision) return errorReply(http::status::not_found, "no geometry revision " + revisionUuid);
        const auto bytes = store->readGeometryArray(txn, revision->uuid, array, offset, count);
        return {http::status::ok, "application/octet-stream", std::string(bytes.begin(), bytes.end())};
    }

    // ---- Routing ------------------------------------------------------------

    struct Route {
        Lane* lane = nullptr;
        std::function<Reply()> work;
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
        if (method == http::verb::get && is({"v1", "receipts", "*"}))
            return {control.get(), [this, target] { return receipt(target->path[2]); }};
        if (method == http::verb::put && is({"v1", "payloads", "*"}))
            return {readers.get(), [this, target, body] { return stage(target->path[2], *body); }};
        if (method == http::verb::post && is({"v1", "transactions"}))
            return {writer.get(), [this, body] { return submit(*body); }};
        if (method == http::verb::get && is({"v1", "changes"}))
            return {readers.get(), [this, target] { return changes(*target); }};
        if (method == http::verb::get && is({"v1", "objects", "*"}))
            return {readers.get(), [this, target] { return currentObject(target->path[2]); }};
        if (method == http::verb::get && is({"v1", "objects", "*", "versions", "*"}))
            return {readers.get(), [this, target] { return objectVersion(target->path[2], target->path[4], *target); }};
        if (method == http::verb::get && is({"v1", "geometry", "*"}))
            return {readers.get(), [this, target] { return geometry(target->path[2]); }};
        if (method == http::verb::get && is({"v1", "geometry", "*", "arrays", "*"}))
            return {readers.get(), [this, target] { return geometryArray(target->path[2], target->path[4], *target); }};
        return {};
    }

    static Reply guarded(const std::function<Reply()>& work)
    {
        try {
            return work();
        } catch (const TransactionConflict& conflict) {
            return jsonReply(transactionConflictToJson(conflict), http::status::conflict);
        } catch (const MissingPayloads& missing) {
            return jsonReply(Json{{"error", missing.what()}, {"missing", missing.hashes()}}.dump(),
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
        if (!r.lane) {
            done(errorReply(http::status::not_found, "unknown endpoint"));
            return;
        }
        auto job = [work = std::move(r.work), done]() mutable { done(guarded(work)); };
        if (!r.lane->tryPost(job)) done(errorReply(http::status::service_unavailable, "server busy; retry"));
    }

    std::size_t bodyLimit(const Request& header) const
    {
        const auto raw = header.target();
        const std::string_view target(raw.data(), raw.size());
        return header.method() == http::verb::put && target.starts_with("/v1/payloads/") ? options.maxPayloadBytes
                                                                                         : options.maxCommandBytes;
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
        parser_.emplace();
        parser_->header_limit(64 * 1024);
        stream_.expires_after(server_.options.requestDeadline);
        http::async_read_header(stream_, buffer_, *parser_,
            [self = shared_from_this()](beast::error_code ec, std::size_t) { self->onHeader(ec); });
    }
    void onHeader(beast::error_code ec)
    {
        if (ec) return close();
        parser_->body_limit(server_.bodyLimit(parser_->get()));
        http::async_read(stream_, buffer_, *parser_,
            [self = shared_from_this()](beast::error_code ec, std::size_t) { self->onBody(ec); });
    }
    void onBody(beast::error_code ec)
    {
        if (ec == http::error::body_limit) {
            keepAlive_ = false;
            return write(errorReply(http::status::payload_too_large, "request body exceeds the server limit"));
        }
        if (ec) return close();
        auto request = parser_->release();
        keepAlive_ = request.keep_alive();
        version_ = request.version();
        // Wait for the reply before reading on: one request at a time per connection.
        stream_.expires_never();
        server_.dispatch(request, [self = shared_from_this()](Reply reply) {
            net::post(self->stream_.get_executor(), [self, reply = std::move(reply)]() mutable { self->write(std::move(reply)); });
        });
    }
    void write(Reply reply)
    {
        auto response = std::make_shared<http::response<http::string_body>>(reply.status, version_);
        response->set(http::field::server, "vc_object_server");
        response->set(http::field::content_type, reply.contentType);
        if (reply.status == http::status::service_unavailable) response->set(http::field::retry_after, "1");
        response->keep_alive(keepAlive_);
        response->body() = std::move(reply.body);
        response->prepare_payload();
        stream_.expires_after(server_.options.requestDeadline);
        http::async_write(stream_, *response, [self = shared_from_this(), response](beast::error_code ec, std::size_t) {
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
    std::optional<http::request_parser<http::string_body>> parser_;
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
    s.accept();
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
    // Finish accepted storage work; its replies go to sockets still served.
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
