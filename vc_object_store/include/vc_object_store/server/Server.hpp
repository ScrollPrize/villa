#pragma once
// vc_object_server: hosts one object store over HTTP for clients that have no
// filesystem access to it. Handlers only adapt the wire to the public
// ObjectStore APIs (submit, receipt, status, changesSince, historical reads,
// object/edge selections, object history, batched reads at versions,
// numbered relationship selections,
// region and surface proximity queries, artifact bytes); /v1/service reports the service's
// own state (queues, staging, connections, WAL checkpoints);
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
    int readWorkers = 4;                   // reads, feed pages, queries
    int transferWorkers = 2;               // staging file writes and hashing
    std::size_t writeQueue = 64;           // checked transactions waiting for the one writer
    std::size_t readQueue = 256;
    std::size_t controlQueue = 256;        // status/receipt checks, kept off busy lanes
    std::size_t maxConnections = 64;
    std::size_t maxCommandBytes = 16u << 20;  // one transaction/command body
    // Payload uploads stream to files under stagingDirectory in chunks of
    // uploadChunkBytes; a connection reads its next chunk only after the
    // previous one is on disk (backpressure). Requires Content-Length.
    std::size_t maxPayloadBytes = std::size_t(4) << 30;  // one staged payload
    std::size_t maxStagedBytes = std::size_t(16) << 30;  // staged plus reserved for uploads in progress
    std::size_t uploadChunkBytes = 1u << 20;
    std::chrono::seconds stagingLifetime{3600};
    // Empty: "<store>.staging" beside the database. Staging is not durable:
    // payload files there are removed at startup.
    std::filesystem::path stagingDirectory;
    // One geometry array response; larger reads are refused (read ranges).
    std::size_t maxResponseBytes = 64u << 20;
    // Array reads are refused (503) while unsent response bytes exceed this.
    std::size_t maxResponseBytesInFlight = 512u << 20;
    // Per request read/write on a connection. A link_objects request answers
    // only when its detection is done, so a service that takes them sets this
    // above the longest expected link (vc_object_server --request-deadline).
    std::chrono::seconds requestDeadline{300};
    // Where link_objects keeps the fiber strip cache between requests
    // (vc_object_server --strip-cache); empty keeps strips in memory. Client
    // requests may not name a cache directory of their own.
    std::filesystem::path linkScratchDirectory;
    // WAL upkeep on the writer lane between transactions: a passive
    // checkpoint every interval (zero disables), then a truncating one when
    // everything was copied but the WAL file is larger than walTruncateBytes.
    // Readers that keep a checkpoint from completing are reported in
    // /v1/service and logged.
    std::chrono::milliseconds checkpointInterval{10000};
    uint64_t walTruncateBytes = 256u << 20;
    std::chrono::milliseconds truncateWait{200};
    int64_t maxFeedChanges = 4096;
    // Object/edge query pages must name a limit no larger than this, and a
    // batched read at versions may name at most this many references; larger
    // listings and batches are split by the client (see HostedStore).
    int64_t maxQueryResults = 10000;
    // Region queries are clamped to these budgets; a clamped query reports
    // the budget it stopped at (RegionResult::stop), never a silent cut.
    uint64_t maxRegionOutputBytes = 32u << 20;
    uint64_t maxRegionProcessingBytes = 512u << 20;
    uint64_t maxRegionCandidateWork = 10000000;
    uint64_t maxRegionSqlSteps = 100000000;
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
