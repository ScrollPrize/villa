// Benchmark for large-store selections (not a ctest).
//
//   bench_object_store_selection create STORE COUNT
//       Synthetic store: COUNT objects, half fibers (8 control points) and
//       half tracks (16 points), random walks in a 10000^3 volume, fixed seed.
//   bench_object_store_selection list STORE [ITERATIONS]
//       Times full listings: local single snapshot, local UUID pages, and
//       hosted pages through an in-process vc_object_server.
//   bench_object_store_selection versions STORE [COUNT] [ITERATIONS]
//       Times reads of COUNT fibers at their (head) versions: batched versus
//       one reference at a time, local and hosted.
//   bench_object_store_selection record STORE [ITERATIONS]
//       Times selection records of every fiber (local snapshot, hosted pages),
//       their encoding and size.
//   bench_object_store_selection geometry STORE [COUNT] [ITERATIONS]
//       Times local geometry reads of COUNT fibers: descriptors and control
//       points one revision at a time, and one packed batch.
#include "vc_object_store/Uuid.hpp"
#include "vc_object_store/client/HostedStore.hpp"
#include "vc_object_store/objects/TransactionJson.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include "vc_object_store/objects/ObjectStore.hpp"
#include "vc_object_store/objects/Selections.hpp"
#include "vc_object_store/server/Server.hpp"

#include <algorithm>
#include <atomic>
#include <thread>
#include <chrono>
#include <cstdio>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <sys/resource.h>

using namespace vc::objects;
using Clock = std::chrono::steady_clock;

namespace {
double seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }

long maxRssMiB()
{
    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
#ifdef __APPLE__
    return usage.ru_maxrss >> 20;
#else
    return usage.ru_maxrss >> 10;
#endif
}

// Deterministic UUID text from the generator (version 4 layout).
std::string uuidFrom(std::mt19937_64& rng)
{
    uint64_t hi = rng(), lo = rng();
    hi = (hi & ~0xF000ull) | 0x4000ull;
    lo = (lo & ~(3ull << 62)) | (2ull << 62);
    char text[37];
    std::snprintf(text, sizeof text, "%08llx-%04llx-%04llx-%04llx-%012llx", (unsigned long long)(hi >> 32),
                  (unsigned long long)((hi >> 16) & 0xFFFF), (unsigned long long)(hi & 0xFFFF),
                  (unsigned long long)(lo >> 48), (unsigned long long)(lo & 0xFFFFFFFFFFFFull));
    return text;
}

int create(const std::string& path, int64_t count)
{
    OpenOptions options;
    options.author = "selection-bench";
    auto store = ObjectStore::create(path, options);
    std::mt19937_64 rng(20261104);
    std::uniform_real_distribution<double> position(0, 10000), step(-20, 20);
    const auto start = Clock::now();
    constexpr int64_t batch = 10000;
    for (int64_t done = 0; done < count;) {
        auto txn = store.write();
        for (int64_t end = std::min(count, done + batch); done < end; ++done) {
            ObjectRecord record;
            record.uuid = uuidFrom(rng);
            const bool fiber = done % 2 == 0;
            std::vector<vc::geometry::Point> points(fiber ? 8 : 16);
            vc::geometry::Point p{position(rng), position(rng), position(rng)};
            for (auto& point : points) {
                point = p;
                p = {p[0] + step(rng), p[1] + step(rng), p[2] + step(rng)};
            }
            if (fiber) {
                vc::geometry::Fiber geometry;
                geometry.controlPoints = points;
                record.kind = "fiber";
                record.name = "fiber " + std::to_string(done);
                record.fiber.emplace();
                record.fiber->family = done % 4 == 0 ? "H" : "V";
                record.fiber->controlPointCount = int64_t(points.size());
                record.fiber->length = geometry.length();
                store.upsertObject(txn, record);
                store.putGeometry(txn, record.uuid, vc::geometry::encode(geometry));
            } else {
                vc::geometry::Track geometry;
                for (const auto& point : points)
                    geometry.points.push_back({int32_t(point[0]), int32_t(point[1]), int32_t(point[2])});
                record.kind = "track";
                record.track.emplace();
                record.track->datasetRef = "bench";
                record.track->externalKey = std::to_string(done);
                record.track->pointCount = int64_t(points.size());
                record.track->length = geometry.length();
                store.upsertObject(txn, record);
                store.putGeometry(txn, record.uuid, vc::geometry::encode(geometry));
            }
        }
        txn.commit();
        std::cerr << "\r" << done << " objects, " << seconds(start) << " s" << std::flush;
    }
    store.checkpoint();
    std::cerr << "\n";
    std::cout << "created " << count << " objects in " << seconds(start) << " s (Release, one run); max RSS "
              << maxRssMiB() << " MiB\n";
    return 0;
}

void report(const std::string& what, std::vector<double> times, std::size_t rows)
{
    std::sort(times.begin(), times.end());
    std::cout << what << ": rows=" << rows << " n=" << times.size() << " min=" << times.front()
              << " s median=" << times[times.size() / 2] << " s max=" << times.back() << " s\n";
}

int list(const std::string& path, int iterations)
{
    const std::vector<std::pair<std::string, std::optional<std::vector<std::string>>>> projections{
        {"uuids", std::vector<std::string>{}}, {"name+fiber+track", std::vector<std::string>{"name", "fiber", "track"}}};
    OpenOptions readOnly;
    readOnly.readOnly = true;
    {
        auto store = ObjectStore::open(path, readOnly);
        for (const auto& kind : {"fiber", "track"})
            for (const auto& [label, fields] : projections) {
                std::vector<double> whole, paged;
                std::size_t rows = 0;
                for (int i = 0; i < iterations; ++i) {
                    ObjectSelection selection;
                    selection.kind = kind;
                    auto start = Clock::now();
                    {
                        auto txn = store.read();
                        rows = store.selectObjects(txn, selection, fields).size();
                    }
                    whole.push_back(seconds(start));
                    start = Clock::now();
                    std::size_t pagedRows = 0;
                    selection.limit = 10000;
                    for (;;) {
                        auto txn = store.read();
                        const auto page = store.selectObjects(txn, selection, std::vector<std::string>{});
                        pagedRows += page.size();
                        if (int64_t(page.size()) < *selection.limit) break;
                        selection.afterUuid = page.back().uuid;
                    }
                    paged.push_back(seconds(start));
                    if (pagedRows != rows) throw std::runtime_error("paged listing differs");
                }
                report(std::string("local ") + kind + " " + label + " one snapshot", whole, rows);
                if (label == "uuids") report(std::string("local ") + kind + " uuids 10k pages", paged, rows);
            }
    }
    server::ServerOptions options;
    server::Server service(path, options);
    service.start();
    HostedStore hosted("http://127.0.0.1:" + std::to_string(service.port()));
    for (const auto& kind : {"fiber", "track"})
        for (const auto& [label, fields] : projections) {
            std::vector<double> times;
            std::size_t rows = 0;
            for (int i = 0; i < iterations; ++i) {
                ObjectSelection selection;
                selection.kind = kind;
                const auto start = Clock::now();
                rows = hosted.selectAllObjects(selection, fields, 10000).size();
                times.push_back(seconds(start));
            }
            report(std::string("hosted ") + kind + " " + label + " 10k pages", times, rows);
        }
    // The same full fiber listing while a writer adds 150 fibers and makes 25
    // edits, spread over the listing (the expected churn of a big run).
    {
        const auto initial = hosted.status();
        std::atomic<bool> done{false};
        std::atomic<int> commits{0};
        std::string writerError;
        std::thread writer([&] {
            try {
                HostedStore client("http://127.0.0.1:" + std::to_string(service.port()));
                std::mt19937_64 rng(7);
                std::vector<std::string> created;
                for (int i = 0; i < 175 && !done; ++i) {
                    TransactionRequest r;
                    r.requestUuid = vc::objects::util::newUuid();
                    r.storeUuid = initial.storeUuid;
                    r.generation = initial.generation;
                    r.author = "churn";
                    if (i % 7 == 6 && !created.empty()) {
                        const auto& target = created[rng() % created.size()];
                        r.expectedObjects[target] = client.currentVersion(target);
                        r.operations.push_back(SetMetaOp{target, "edit", std::to_string(i)});
                    } else {
                        ObjectRecord record;
                        record.uuid = vc::objects::util::newUuid();
                        record.kind = "fiber";
                        record.fiber.emplace();
                        vc::geometry::Fiber geometry;
                        geometry.controlPoints = {{double(i), 0, 0}, {double(i), 10, 0}};
                        record.fiber->controlPointCount = 2;
                        record.fiber->length = geometry.length();
                        r.expectedObjects[record.uuid] = std::nullopt;
                        r.operations.push_back(CreateObjectOp{objectRecordJson(record)});
                        r.operations.push_back(PutGeometryOp{record.uuid, vc::geometry::encode(geometry), {}});
                        created.push_back(record.uuid);
                    }
                    client.submit(r);
                    ++commits;
                    std::this_thread::sleep_for(std::chrono::milliseconds(40));
                }
            } catch (const std::exception& e) {
                writerError = e.what();
            }
        });
        ObjectSelection fibers;
        fibers.kind = "fiber";
        StoreStatus asOf;
        const auto start = Clock::now();
        const auto rows = hosted.selectAllObjects(fibers, std::vector<std::string>{}, 10000, 3, &asOf).size();
        const auto elapsed = seconds(start);
        const int during = commits.load();
        done = true;
        writer.join();
        if (!writerError.empty()) throw std::runtime_error("churn writer: " + writerError);
        std::cout << "hosted fiber uuids 10k pages with churn: rows=" << rows << " " << elapsed << " s, " << during
                  << " commits landed during the listing, as of commit " << asOf.commitSequence << " (from "
                  << initial.commitSequence << ")\n";
    }
    std::cout << "max RSS (client and server in one process) " << maxRssMiB() << " MiB\n";
    return 0;
}
int versions(const std::string& path, int64_t count, int iterations)
{
    std::vector<ObjectVersion> references;
    {
        OpenOptions readOnly;
        readOnly.readOnly = true;
        auto store = ObjectStore::open(path, readOnly);
        auto txn = store.read();
        const auto status = store.status(txn);
        ObjectSelection fibers;
        fibers.kind = "fiber";
        fibers.limit = count;
        for (const auto& record : store.selectObjects(txn, fibers, std::vector<std::string>{}))
            references.push_back({status.storeUuid, record.versionGeneration, record.uuid, *record.version});
        std::vector<double> batched, single;
        for (int i = 0; i < iterations; ++i) {
            auto start = Clock::now();
            const auto all = store.objectsAtVersions(txn, references);
            batched.push_back(seconds(start));
            start = Clock::now();
            std::size_t found = 0;
            for (const auto& reference : references) found += store.objectAtVersion(txn, reference).has_value();
            single.push_back(seconds(start));
            if (found != all.size() || std::count(all.begin(), all.end(), std::nullopt))
                throw std::runtime_error("versions missing");
        }
        report("local objectsAtVersions", batched, references.size());
        report("local objectAtVersion one at a time", single, references.size());
    }
    server::Server service(path, server::ServerOptions{});
    service.start();
    HostedStore hosted("http://127.0.0.1:" + std::to_string(service.port()));
    const std::vector<ObjectVersion> few(references.begin(), references.begin() + std::min<std::size_t>(references.size(), 1000));
    std::vector<double> batched, single;
    for (int i = 0; i < iterations; ++i) {
        auto start = Clock::now();
        if (hosted.objectsAtVersions(references, 1000).size() != references.size())
            throw std::runtime_error("hosted versions missing");
        batched.push_back(seconds(start));
        start = Clock::now();
        for (const auto& reference : few)
            if (!hosted.objectAtVersion(reference)) throw std::runtime_error("hosted version missing");
        single.push_back(seconds(start));
    }
    report("hosted objectsAtVersions 1000 per request", batched, references.size());
    report("hosted objectAtVersion one request each", single, few.size());
    return 0;
}
int record(const std::string& path, int iterations)
{
    ObjectSelection fibers;
    fibers.kind = "fiber";
    std::vector<double> local, hostedTimes, encode, decode;
    SelectionRecord captured;
    std::size_t bytes = 0;
    {
        OpenOptions readOnly;
        readOnly.readOnly = true;
        auto store = ObjectStore::open(path, readOnly);
        for (int i = 0; i < iterations; ++i) {
            auto start = Clock::now();
            {
                auto txn = store.read();
                captured = recordSelection(store, txn, fibers);
            }
            local.push_back(seconds(start));
            start = Clock::now();
            const auto encoded = encodeSelectionRecord(captured);
            encode.push_back(seconds(start));
            bytes = encoded.size();
            start = Clock::now();
            if (!(decodeSelectionRecord(encoded) == captured)) throw std::runtime_error("round trip differs");
            decode.push_back(seconds(start));
        }
    }
    server::Server service(path, server::ServerOptions{});
    service.start();
    HostedStore hosted("http://127.0.0.1:" + std::to_string(service.port()));
    for (int i = 0; i < iterations; ++i) {
        const auto start = Clock::now();
        const auto remote = hosted.recordSelection(fibers, 10000);
        hostedTimes.push_back(seconds(start));
        if (remote.objects != captured.objects) throw std::runtime_error("hosted record differs");
    }
    report("local recordSelection fibers", local, captured.objects.size());
    report("hosted recordSelection fibers 10k pages", hostedTimes, captured.objects.size());
    report("encodeSelectionRecord", encode, captured.objects.size());
    report("decodeSelectionRecord", decode, captured.objects.size());
    std::cout << "encoded size " << bytes << " bytes (" << double(bytes) / double(captured.objects.size())
              << " per object)\n";
    return 0;
}
int geometry(const std::string& path, std::int64_t count, int iterations)
{
    OpenOptions readOnly;
    readOnly.readOnly = true;
    auto store = ObjectStore::open(path, readOnly);
    std::vector<std::string> revisions;
    {
        auto txn = store.read();
        ObjectSelection fibers;
        fibers.kind = "fiber";
        fibers.limit = count;
        for (const auto& record : store.selectObjects(txn, fibers, std::vector<std::string>{}))
            revisions.push_back(store.currentGeometry(txn, record.uuid)->uuid);
    }
    std::vector<double> descriptors, arrays, packed;
    std::size_t bytes = 0;
    for (int i = 0; i < iterations; ++i) {
        auto txn = store.read();
        auto start = Clock::now();
        for (const auto& revision : revisions)
            if (!store.geometryRevision(txn, revision)) throw std::runtime_error("missing geometry");
        descriptors.push_back(seconds(start));
        start = Clock::now();
        bytes = 0;
        for (const auto& revision : revisions) bytes += store.readGeometryArray(txn, revision, "control_points").size();
        arrays.push_back(seconds(start));
        start = Clock::now();
        if (store.readGeometryArrays(txn, revisions, "control_points").data.size() != bytes)
            throw std::runtime_error("packed size differs");
        packed.push_back(seconds(start));
    }
    report("geometryRevision one at a time", descriptors, revisions.size());
    report("readGeometryArray control_points one at a time", arrays, revisions.size());
    report("readGeometryArrays control_points packed", packed, revisions.size());
    return 0;
}
} // namespace

int main(int argc, char** argv)
try {
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "create" && argc == 4) return create(argv[2], std::stoll(argv[3]));
    if (mode == "list" && (argc == 3 || argc == 4)) return list(argv[2], argc == 4 ? std::stoi(argv[3]) : 3);
    if (mode == "record" && (argc == 3 || argc == 4)) return record(argv[2], argc == 4 ? std::stoi(argv[3]) : 3);
    if (mode == "versions" && argc >= 3 && argc <= 5)
        return versions(argv[2], argc >= 4 ? std::stoll(argv[3]) : 10000, argc == 5 ? std::stoi(argv[4]) : 5);
    if (mode == "geometry" && argc >= 3 && argc <= 5)
        return geometry(argv[2], argc >= 4 ? std::stoll(argv[3]) : 10000, argc == 5 ? std::stoi(argv[4]) : 5);
    std::cerr << "usage: bench_object_store_selection create STORE COUNT | list STORE [ITERATIONS] |"
                 " versions STORE [COUNT] [ITERATIONS] | record STORE [ITERATIONS] | geometry STORE [COUNT] [ITERATIONS]\n";
    return 2;
} catch (const std::exception& error) {
    std::cerr << "bench_object_store_selection: " << error.what() << "\n";
    return 1;
}
