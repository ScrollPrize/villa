#pragma once
// vc_object_server: hosts one object store over HTTP for clients that have no
// filesystem access to it. Handlers only adapt the wire to the public
// ObjectStore APIs (submit, receipt, status, changesSince, historical reads);
// storage semantics live in the shared library.
//
// The store is opened as its service owner: startup fails while a standalone
// writer has it open, and standalone writable opens fail while it is hosted.
// There is no authentication; listen on loopback (SSH forwarding) or a
// trusted private network only. Request authors are attribution.

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace vc::objects::server {

struct ServerOptions {
    std::string listenAddress = "127.0.0.1";
    uint16_t port = 0;                     // 0: any free port (see Server::port)
    int networkThreads = 2;                // socket I/O only; never storage work
    int readWorkers = 4;                   // reads, feed pages, payload staging
    std::size_t writeQueue = 64;           // checked transactions waiting for the one writer
    std::size_t readQueue = 256;
    std::size_t controlQueue = 256;        // status/receipt checks, kept off busy lanes
    std::size_t maxConnections = 64;
    std::size_t maxCommandBytes = 16u << 20;  // one transaction/command body
    std::size_t maxPayloadBytes = 512u << 20; // one staged payload
    std::size_t maxStagedBytes = std::size_t(4) << 30;
    std::chrono::seconds stagingLifetime{3600};
    std::chrono::seconds requestDeadline{300}; // per request read/write on a connection
    int64_t maxFeedChanges = 4096;
    std::string author = "vc_object_server";  // open author; requests carry their own
};

class Server {
public:
    Server(const std::filesystem::path& store, ServerOptions);
    ~Server();
    Server(const Server&) = delete;
    Server& operator=(const Server&) = delete;
    // The bound port (useful with port 0).
    uint16_t port() const;
    // Serve on background threads until stop() or destruction.
    void start();
    // Stop accepting, finish queued work, close connections and the store.
    void stop();
    // start() then block until stop() is called from another thread or a signal handler.
    void run();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace vc::objects::server
