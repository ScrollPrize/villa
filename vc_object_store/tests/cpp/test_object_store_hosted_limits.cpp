// Hosted service limits and concurrency: streamed staging uploads and their
// bounds (hash, length, size, staging capacity, disconnects), ranged geometry
// downloads, full lanes answering 503 while status stays responsive, status
// latency during simultaneous transfers, 2-4 concurrent writers, a client
// that disconnects after sending a transaction (its outcome is recovered by
// receipt), and scheduled WAL checkpoints.
#include "TestSupport.hpp"
#include "vc_object_store/client/HostedStore.hpp"
#include "vc_object_store/objects/Artifacts.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include "vc_object_store/objects/ObjectStore.hpp"
#include "vc_object_store/objects/TransactionJson.hpp"
#include "vc_object_store/server/Server.hpp"

#include <nlohmann/json.hpp>
#include <vc_reader/http.hpp>

#include <algorithm>
#include <arpa/inet.h>
#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <netinet/in.h>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

using namespace vc::objects;
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;

namespace {
ObjectRecord trackRecord(const std::string& uuid)
{
    ObjectRecord record; record.uuid = uuid; record.kind = "track"; record.track.emplace();
    record.track->datasetRef = "limits"; record.track->externalKey = uuid;
    return record;
}

// Sends `text` on a fresh connection; optionally waits for the reply, then
// closes. Raw sockets let the test hang up mid-request.
std::string rawExchange(uint16_t port, const std::string& text, bool awaitReply)
{
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) throw std::runtime_error("socket failed");
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(port);
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (::connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof address) != 0) {
        ::close(fd);
        throw std::runtime_error("connect failed");
    }
    for (std::size_t sent = 0; sent < text.size();) {
        const auto n = ::send(fd, text.data() + sent, text.size() - sent, 0);
        if (n <= 0) break;
        sent += std::size_t(n);
    }
    std::string reply;
    if (awaitReply) {
        char buffer[4096];
        for (ssize_t n; (n = ::recv(fd, buffer, sizeof buffer, 0)) > 0;) {
            reply.append(buffer, std::size_t(n));
            if (reply.find("\r\n\r\n") != std::string::npos) break;
        }
    }
    ::close(fd);
    return reply;
}

template<class F> void eventually(F&& condition, const char* what)
{
    for (const auto deadline = Clock::now() + std::chrono::seconds(10); Clock::now() < deadline;) {
        if (condition()) return;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    throw std::runtime_error(std::string("timed out waiting: ") + what);
}

std::string percentiles(std::vector<double> ms)
{
    std::sort(ms.begin(), ms.end());
    auto at = [&](double q) { return ms[std::min(ms.size() - 1, std::size_t(q * double(ms.size())))]; };
    return "n=" + std::to_string(ms.size()) + " median=" + std::to_string(at(0.5)) + "ms p95=" + std::to_string(at(0.95)) +
           "ms max=" + std::to_string(ms.back()) + "ms";
}

std::size_t stagingFiles(const std::filesystem::path& directory, const std::string& suffix)
{
    std::size_t count = 0;
    for (const auto& entry : std::filesystem::directory_iterator(directory))
        if (entry.path().filename().string().ends_with(suffix)) ++count;
    return count;
}
} // namespace

int main()
try {
    TemporaryDirectory directory;
    OpenOptions options; options.author = "limits-test";
    const auto path = directory.path / "limits.vcobj";
    { (void)ObjectStore::create(path, options); }
    const auto stagingDir = directory.path / "staging";

    // Startup drops payload files a previous run left; other files stay.
    std::filesystem::create_directories(stagingDir);
    const std::string staleName = std::string(64, 'a') + ".7";
    { std::ofstream(stagingDir / staleName) << "stale"; std::ofstream(stagingDir / "notes.txt") << "keep"; }

    server::ServerOptions o;
    o.readWorkers = 2;
    o.transferWorkers = 1;
    o.stagingDirectory = stagingDir;
    o.uploadChunkBytes = 64 << 10;
    o.maxPayloadBytes = 24u << 20;
    o.maxStagedBytes = 160u << 20;
    o.maxResponseBytes = 1u << 20;
    o.maxCommandBytes = 64u << 10;
    o.checkpointInterval = std::chrono::milliseconds(50);
    o.walTruncateBytes = 0;  // truncate whenever a checkpoint copied everything
    auto service = std::make_unique<server::Server>(path, o);
    require(!std::filesystem::exists(stagingDir / staleName) && std::filesystem::exists(stagingDir / "notes.txt"),
            "startup clears stale staging files only");
    service->start();
    const auto port = service->port();
    // A second service is refused before it touches the first one's staging.
    {
        std::ofstream(stagingDir / (std::string(64, '9') + ".1")) << "in use";
        mustFail([&] { server::Server second(path, o); });
        require(std::filesystem::exists(stagingDir / (std::string(64, '9') + ".1")), "refused service leaves staging alone");
    }
    const auto endpoint = "http://127.0.0.1:" + std::to_string(port);
    // Content-Length must be refused from the header, before allocating or
    // waiting for the body. Chunked commands still enforce the same bound.
    {
        vcr::http::Options once; once.retries = 0;
        vcr::http::Session http(once);
        auto body = kindQueryRequestToJson({"track", false, std::nullopt});
        body.resize(o.maxCommandBytes, ' ');
        require(http.request("POST", endpoint + "/v1/query/count", body).status == 200,
                "command at the size limit accepted");
        const auto oversized = rawExchange(port,
            "POST /v1/query/count HTTP/1.1\r\nHost: x\r\nContent-Length: " +
            std::to_string(o.maxCommandBytes + 1) + "\r\n\r\n", true);
        require(oversized.starts_with("HTTP/1.1 413"), "oversized command refused before receiving its body");
        const auto chunked = rawExchange(port,
            "POST /v1/query/count HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n10001\r\n", true);
        require(chunked.starts_with("HTTP/1.1 413"), "oversized chunked command refused");
    }
    HostedStore hosted(endpoint);
    const auto initial = hosted.status();
    auto request = [&](VersionSelection expected, std::vector<TransactionOperation> operations, std::string author) {
        TransactionRequest r;
        r.requestUuid = util::newUuid(); r.storeUuid = initial.storeUuid; r.generation = initial.generation;
        r.author = std::move(author); r.expectedObjects = std::move(expected); r.operations = std::move(operations);
        return r;
    };

    // A large payload streams through 64 KiB chunks and reads back exactly,
    // in ranges no larger than the server's 1 MiB response limit.
    const auto big = util::newUuid();
    vc::geometry::Track track;
    for (int i = 0; i < 400000; ++i) track.points.push_back({i, i / 2, i % 97});
    const auto encoded = vc::geometry::encode(track);
    const auto created = hosted.submit(request({{big, std::nullopt}},
        {CreateObjectOp{objectRecordJson(trackRecord(big))}, PutGeometryOp{big, encoded, {}}}, "uploader"));
    require(created.objects.at(big) == 1, "large payload committed");
    const auto head = hosted.currentObjectRevision(big);
    require(head && head->geometryRevision, "large payload head");
    const auto descriptor = hosted.geometryRevision(*head->geometryRevision);
    require(descriptor.has_value(), "geometry descriptor");
    OpenOptions readOnly; readOnly.readOnly = true;
    auto inspect = ObjectStore::open(path, readOnly);
    for (const auto& array : descriptor->arrays) {
        auto txn = inspect.read();
        const auto expected = inspect.readGeometryArray(txn, descriptor->uuid, array.name);
        require(hosted.readGeometryArray(descriptor->uuid, array.name) == expected, "ranged whole-array read");
        if (array.byteLength > 3) {
            const int64_t count = array.byteLength - 3;
            require(hosted.readGeometryArray(descriptor->uuid, array.name, 3, count) ==
                        std::vector<uint8_t>(expected.begin() + 3, expected.end()), "ranged partial read");
        }
    }
    {
        HostedStoreOptions small; small.rangeBytes = 100000;
        HostedStore client(endpoint, small);
        const auto& array = descriptor->arrays.front();
        auto txn = inspect.read();
        require(client.readGeometryArray(descriptor->uuid, array.name) ==
                    inspect.readGeometryArray(txn, descriptor->uuid, array.name), "client-side ranges");
    }
    // The batched geometry query under the 1 MiB response limit: a revision
    // whose arrays alone exceed it is answered without its data (read by
    // ranges instead); after an answered small revision an oversized one waits
    // for the next request; every answered small revision carries its bytes.
    {
        const auto small = util::newUuid();
        vc::geometry::Track few; few.points = {{1, 2, 3}, {4, 5, 6}};
        hosted.submit(request({{small, std::nullopt}},
            {CreateObjectOp{objectRecordJson(trackRecord(small))}, PutGeometryOp{small, vc::geometry::encode(few), {}}}, "uploader"));
        const auto smallRevision = *hosted.currentObjectRevision(small)->geometryRevision;
        std::vector<std::string> names;
        for (const auto& array : descriptor->arrays) names.push_back(array.name);
        const auto bigFirst = hosted.geometryBatch({descriptor->uuid, smallRevision}, names, {});
        require(bigFirst && bigFirst->size() == 2 && (*bigFirst)[0].descriptor && (*bigFirst)[0].arrays.empty() &&
                    !(*bigFirst)[1].arrays.empty(),
                "an oversized revision is answered without its arrays; the small one after it still fits");
        const auto smallFirst = hosted.geometryBatch({smallRevision, descriptor->uuid}, names, {});
        require(smallFirst && smallFirst->size() == 1 && !(*smallFirst)[0].arrays.empty(),
                "a small revision keeps its arrays and the oversized one waits");
        auto txn = inspect.read();
        for (const auto& [name, bytes] : (*smallFirst)[0].arrays)
            require(bytes == inspect.readGeometryArray(txn, smallRevision, name), "batched bytes under the limit");
    }
    // A host whose response limit cannot fit even one compact entry refuses
    // the batched query (413), which the client treats as unsupported.
    {
        const auto tinyPath = directory.path / "tiny.vcobj";
        { (void)ObjectStore::create(tinyPath, options); }
        auto tinyOptions = o;
        tinyOptions.maxResponseBytes = 64;
        tinyOptions.stagingDirectory = directory.path / "tiny-staging";
        server::Server tiny(tinyPath, tinyOptions);
        tiny.start();
        HostedStore client("http://127.0.0.1:" + std::to_string(tiny.port()));
        require(!client.geometryBatch({util::newUuid()}, {"points"}, {}), "a too-small response limit refuses the batch");
    }
    {
        HostedStoreOptions larger; larger.rangeBytes = 2 * o.maxResponseBytes;
        HostedStore client(endpoint, larger);
        const auto& array = descriptor->arrays.front();
        auto txn = inspect.read();
        const auto expected = inspect.readGeometryArray(txn, descriptor->uuid, array.name);
        require(array.byteLength > larger.rangeBytes, "fixture exceeds the client range size");
        require(client.readGeometryArray(descriptor->uuid, array.name, 0, array.byteLength) == expected,
                "explicit count negotiates smaller server ranges");
        require(client.readGeometryArray(descriptor->uuid, array.name, 3, array.byteLength - 7) ==
                    std::vector<uint8_t>(expected.begin() + 3, expected.end() - 4),
                "range negotiation preserves the offset and final partial range");
    }

    // A 3 MiB artifact streams in and comes back through 1 MiB ranges.
    {
        std::string bytes(3u << 20, '\0');
        for (std::size_t i = 0; i < bytes.size(); ++i) bytes[i] = char(i * 13 % 241);
        const auto file = directory.path / "big.artifact";
        std::ofstream(file, std::ios::binary) << bytes;
        const auto hash = hosted.putArtifact(file);
        require(inspect.artifactLength(hash) == bytes.size(), "streamed artifact kept by the store");
        const auto back = hosted.readArtifact(hash);
        require(std::string(back.begin(), back.end()) == bytes, "ranged artifact read");
        hosted.fetchArtifact({"big.bin", hash, bytes.size()}, directory.path / "big.fetched");
        std::ifstream in(directory.path / "big.fetched", std::ios::binary);
        require(std::string(std::istreambuf_iterator<char>(in), {}) == bytes, "fetched artifact");
        require(stagingFiles(stagingDir, ".part") == 0, "artifact upload left no staging file");
    }

    // Upload refusals: wrong hash, oversized, no length; none leave files.
    vcr::http::Options once; once.retries = 0;
    vcr::http::Session http(once);
    {
        const std::string bytes = "payload";
        const auto wrong = http.request("PUT", endpoint + "/v1/payloads/" + std::string(64, 'b'), bytes);
        require(wrong.status == 400, "hash mismatch refused");
        const auto badName = http.request("PUT", endpoint + "/v1/payloads/XYZ", bytes);
        require(badName.status == 400, "malformed payload name refused");
        const std::string oversized((24u << 20) + 1, 'x');
        const auto tooBig = http.request("PUT", endpoint + "/v1/payloads/" + std::string(64, 'c'), oversized);
        require(tooBig.status == 413, "oversized payload refused");
        const auto noLength = rawExchange(port,
            "PUT /v1/payloads/" + std::string(64, 'd') + " HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n", true);
        require(noLength.starts_with("HTTP/1.1 411"), "upload without Content-Length refused");
        const std::vector<uint8_t> empty;
        const auto emptyHash = payloadHash(empty);
        require(http.request("PUT", endpoint + "/v1/payloads/" + emptyHash, "").status == 200, "empty payload staged");
    }

    // A client that hangs up mid-upload leaves no partial file or reservation.
    rawExchange(port, "PUT /v1/payloads/" + std::string(64, 'e') + " HTTP/1.1\r\nHost: x\r\nContent-Length: 1000000\r\n\r\n" +
                          std::string(300000, 'z'), false);
    eventually([&] {
        const auto report = Json::parse(hosted.serviceReport());
        return report.at("staging").at("reserved_bytes").get<int64_t>() == 0 && stagingFiles(stagingDir, ".part") == 0;
    }, "abandoned upload cleaned up");

    // Staging capacity: a payload that does not fit is refused (507), not queued.
    {
        const std::string fill(20u << 20, 'f');
        const std::vector<uint8_t> fillBytes(fill.begin(), fill.end());
        int refused = 0;
        for (int i = 0; i < 12 && !refused; ++i) {
            auto bytes = fillBytes; bytes[0] = uint8_t(i);
            const std::string body(bytes.begin(), bytes.end());
            try {
                const auto r = http.request("PUT", endpoint + "/v1/payloads/" + payloadHash(bytes), body);
                require(r.status == 200, "staging fill");
            } catch (const std::runtime_error& e) {
                require(std::string(e.what()).find("507") != std::string::npos, "staging full answers 507");
                ++refused;
            }
        }
        require(refused == 1, "staging capacity enforced");
    }
    service.reset();

    // Full lanes answer 503 at once, while status (control lane) still answers.
    {
        auto saturated = o;
        saturated.readQueue = 0;
        saturated.checkpointInterval = std::chrono::milliseconds(0);
        server::Server busy(path, saturated);
        busy.start();
        const auto url = "http://127.0.0.1:" + std::to_string(busy.port());
        try {
            (void)http.request("GET", url + "/v1/objects/" + big);
            throw std::runtime_error("expected a busy answer");
        } catch (const std::runtime_error& e) {
            require(std::string(e.what()).find("503") != std::string::npos, "full read lane answers 503");
        }
        require(http.request("GET", url + "/v1/status").status == 200, "status answers while reads are refused");
        const auto raw = rawExchange(busy.port(), "GET /v1/changes?x=1 HTTP/1.1\r\nHost: x\r\n\r\n", true);
        require(raw.find("Retry-After: 1") != std::string::npos, "busy answers carry Retry-After");
    }

    service = std::make_unique<server::Server>(path, o);
    service->start();
    const auto endpoint2 = "http://127.0.0.1:" + std::to_string(service->port());
    const auto port2 = service->port();
    HostedStore client(endpoint2);

    // Status latency while four clients upload 16 MiB payloads at once
    // through one transfer worker.
    {
        std::atomic<bool> done{false};
        std::atomic<int64_t> uploaded{0};
        std::vector<std::thread> uploaders;
        std::vector<std::string> errors(4);
        for (int t = 0; t < 4; ++t)
            uploaders.emplace_back([&, t] {
                try {
                    vcr::http::Session session;
                    std::vector<uint8_t> bytes(16u << 20, uint8_t(t + 1));
                    const auto hash = payloadHash(bytes);
                    const std::string body(bytes.begin(), bytes.end());
                    while (!done) {
                        const auto r = session.request("PUT", endpoint2 + "/v1/payloads/" + hash, body);
                        if (r.status != 200) throw std::runtime_error("upload HTTP " + std::to_string(r.status));
                        uploaded += int64_t(body.size());
                    }
                } catch (const std::exception& e) {
                    errors[t] = e.what();
                }
            });
        std::vector<double> latencies;
        const auto until = Clock::now() + std::chrono::seconds(2);
        while (Clock::now() < until) {
            const auto start = Clock::now();
            (void)client.status();
            latencies.push_back(std::chrono::duration<double, std::milli>(Clock::now() - start).count());
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        done = true;
        for (auto& t : uploaders) t.join();
        for (const auto& e : errors)
            if (!e.empty()) throw std::runtime_error("uploader: " + e);
        std::cout << "status during 4 uploads (" << (uploaded.load() >> 20) << " MiB staged): " << percentiles(latencies) << "\n";
        require(*std::max_element(latencies.begin(), latencies.end()) < 1000, "status stays responsive during uploads");
    }

    // Two to four concurrent writers on their own objects all commit, with
    // their own attribution.
    for (int writers : {2, 4}) {
        constexpr int edits = 25;
        std::vector<std::string> objects;
        std::vector<TransactionOperation> creates;
        VersionSelection absent;
        for (int w = 0; w < writers; ++w) {
            objects.push_back(util::newUuid());
            creates.push_back(CreateObjectOp{objectRecordJson(trackRecord(objects.back()))});
            absent[objects.back()] = std::nullopt;
        }
        const auto before = *client.submit(request(absent, creates, "setup")).commitSequence;
        std::vector<std::vector<double>> latencies(writers);
        std::vector<std::string> errors(writers);
        std::vector<std::thread> threads;
        for (int w = 0; w < writers; ++w)
            threads.emplace_back([&, w] {
                try {
                    HostedStore own(endpoint2);
                    for (int64_t v = 1; v <= edits; ++v) {
                        const auto start = Clock::now();
                        const auto r = own.submit(request({{objects[w], v}},
                            {SetMetaOp{objects[w], "n", std::to_string(v)}}, "writer-" + std::to_string(w)));
                        latencies[w].push_back(std::chrono::duration<double, std::milli>(Clock::now() - start).count());
                        if (r.objects.at(objects[w]) != v + 1) throw std::runtime_error("unexpected version");
                    }
                } catch (const std::exception& e) {
                    errors[w] = e.what();
                }
            });
        for (auto& t : threads) t.join();
        std::vector<double> all;
        for (int w = 0; w < writers; ++w) {
            if (!errors[w].empty()) throw std::runtime_error("writer: " + errors[w]);
            all.insert(all.end(), latencies[w].begin(), latencies[w].end());
        }
        std::cout << writers << " concurrent writers, submit latency: " << percentiles(all) << "\n";
        require(client.status().commitSequence == before + writers * edits, "every concurrent edit committed once");
        const auto history = client.objectHistory(objects.back());
        require(history.back().record.changedBy == "writer-" + std::to_string(writers - 1), "concurrent attribution");
    }

    // A client that sends a transaction and hangs up before the reply: the
    // write still runs, and its outcome is recovered by receipt; the
    // identical retry replays it.
    {
        const auto lost = util::newUuid();
        const auto r = request({{lost, std::nullopt}}, {CreateObjectOp{objectRecordJson(trackRecord(lost))}}, "hangs-up");
        const auto body = transactionRequestToJson(r);
        rawExchange(port2, "POST /v1/transactions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: " +
                               std::to_string(body.size()) + "\r\n\r\n" + body, false);
        std::optional<TransactionReceipt> recovered;
        eventually([&] { return (recovered = client.receipt(r.requestUuid)).has_value(); }, "receipt of a disconnected write");
        require(recovered->objects.at(lost) == 1, "disconnected write committed");
        const auto again = client.submit(r);
        require(again.replayed && again.commitSequence == recovered->commitSequence, "retry after disconnect replays");
    }

    // Scheduled checkpoints copy the WAL back and truncate it.
    eventually([&] {
        const auto wal = Json::parse(client.serviceReport()).at("wal");
        return wal.at("checkpoints").get<int64_t>() > 0 && wal.at("truncations").get<int64_t>() > 0 &&
               wal.contains("last") && wal.at("last").at("wal_bytes").get<int64_t>() == 0;
    }, "scheduled WAL checkpoint and truncation");
    {
        const auto report = Json::parse(client.serviceReport());
        require(report.at("queues").at("writer").get<int64_t>() >= 0 && report.at("connections").get<int64_t>() >= 1,
                "service report shape");
    }

    service.reset();
    std::cout << "object_store_hosted_limits: ok\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "object_store_hosted_limits: " << error.what() << "\n";
    return 1;
}
