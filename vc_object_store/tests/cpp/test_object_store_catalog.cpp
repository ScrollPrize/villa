// Open Data samples as catalog stores: placeholders from the catalog, geometry
// fetched by reads that reach it (once, through the session), failures not
// cached, imports shadowing catalog surfaces, published files and refresh.
#include "TestSupport.hpp"
#include "vc_object_store/Tiff.hpp"
#include "vc_object_store/objects/CatalogStore.hpp"
#include "vc_object_store/objects/GeometryRegion.hpp"
#include "vc_object_store/store/GeometryViews.hpp"
#include <nlohmann/json.hpp>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <thread>

using namespace vc::objects;
namespace fs = std::filesystem;

namespace {
// Serves files under a root over HTTP/1.0, counting requests per path.
class FileServer {
public:
    explicit FileServer(fs::path root) : root_(std::move(root))
    {
        socket_ = ::socket(AF_INET, SOCK_STREAM, 0);
        int on = 1;
        ::setsockopt(socket_, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        require(::bind(socket_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0, "bind");
        socklen_t length = sizeof(address);
        ::getsockname(socket_, reinterpret_cast<sockaddr*>(&address), &length);
        port_ = ntohs(address.sin_port);
        require(::listen(socket_, 16) == 0, "listen");
        thread_ = std::thread([this] { serve(); });
    }
    ~FileServer()
    {
        stopping_ = true;
        ::shutdown(socket_, SHUT_RDWR);
        ::close(socket_);
        thread_.join();
    }
    std::string url(const std::string& path) const { return "http://127.0.0.1:" + std::to_string(port_) + "/" + path; }
    int requests(const std::string& path)
    {
        std::lock_guard lock(mutex_);
        return counts_[path];
    }

private:
    void serve()
    {
        while (!stopping_) {
            const int client = ::accept(socket_, nullptr, nullptr);
            if (client < 0) continue;
            std::string request;
            char buffer[4096];
            while (request.find("\r\n\r\n") == std::string::npos) {
                const auto n = ::recv(client, buffer, sizeof(buffer), 0);
                if (n <= 0) break;
                request.append(buffer, size_t(n));
            }
            std::istringstream line(request);
            std::string method, path;
            line >> method >> path;
            if (!path.empty() && path.front() == '/') path.erase(path.begin());
            {
                std::lock_guard lock(mutex_);
                ++counts_[path];
            }
            std::ifstream file(root_ / path, std::ios::binary);
            std::string body((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            const bool found = file.good() || (fs::is_regular_file(root_ / path));
            const std::string head = std::string(found ? "HTTP/1.0 200 OK" : "HTTP/1.0 404 Not Found") +
                                     "\r\nContent-Length: " + std::to_string(found ? body.size() : 0) + "\r\n\r\n";
            ::send(client, head.data(), head.size(), MSG_NOSIGNAL);
            if (found) ::send(client, body.data(), body.size(), MSG_NOSIGNAL);
            ::close(client);
        }
    }
    fs::path root_;
    int socket_ = -1;
    uint16_t port_ = 0;
    std::atomic_bool stopping_{false};
    std::mutex mutex_;
    std::map<std::string, int> counts_;
    std::thread thread_;
};

void writeSurface(const fs::path& directory, float z)
{
    fs::create_directories(directory);
    const int rows = 6, cols = 8;
    cv::Mat_<float> x(rows, cols), y(rows, cols), zs(rows, cols, z);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) { x(r, c) = float(10 + c); y(r, c) = float(10 + r); zs(r, c) = z + float(c) - 4; }
    writeTiff(directory / "x.tif", x);
    writeTiff(directory / "y.tif", y);
    writeTiff(directory / "z.tif", zs);
    std::ofstream(directory / "meta.json") << R"({"format":"tifxyz","type":"seg","uuid":"z_dbg_gen","scale":[1,1],
        "bbox":[[0,0,0],[1,1,1]],"name":"not the record"})";
}

std::string sampleJson(const FileServer& server, const std::string& tifxyz, const std::string& missing)
{
    const auto origin = [&](const std::string& path) {
        return nlohmann::json::array({{{"path", path}, {"access_roots", {{{"type", "https"}, {"url", server.url("")},
                                                                         {"usage", "public-read"}}}}}});
    };
    nlohmann::json segment = {
        {"id", "seg-a"}, {"long_id", "sample-seg-a"}, {"suffix", "w001"}, {"original_volume_id", "vol"},
        {"creation", {{"date", "2026-03-11T00:00:00Z"}, {"process", "surface_mesh_generation"},
                      {"derived_from", {{"type", "volume"}, {"id", "vol"}}}}},
        {"properties", {{"original_volume_downscale", 1},
                        {"volume_coverage", {{"vol", {{"bbox_transformed", {{10, 10, 26}, {17, 15, 33}}},
                                                       {"overlap_ratio", 1.0}, {"computed_at", "x"}}}}}}},
        {"data", {{{"type", "tifxyz"}, {"origins", origin(tifxyz)}},
                  {{"type", "ink-detection"}, {"origins", origin("ink.jpg")}}}}};
    nlohmann::json offline = {
        {"id", "seg-b"}, {"original_volume_id", "vol"},
        {"properties", {{"volume_coverage", {{"vol", {{"bbox_transformed", {{900, 900, 900}, {901, 901, 901}}},
                                                       {"overlap_ratio", 1.0}}}}}}},
        {"data", {{{"type", "tifxyz"}, {"origins", origin(missing)}}}}};
    return nlohmann::json{{"samples", {{"sample", {{"sample", {{"id", "sample"}}},
                                                   {"volumes", {{"vol", {{"id", "vol"}}}}},
                                                   {"segments", {{"seg-a", segment}, {"seg-b", offline}}}}}}}}
        .dump();
}

template <class Predicate> void waitFor(Predicate&& done, const char* what)
{
    for (int i = 0; i < 300 && !done(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(20));
    require(done(), what);
}

RegionResult regionAt(ObjectStoreSession& session, const AttachmentIdentity& attachment, double z)
{
    RegionSlice slice;
    slice.origin = {12, 12, z};
    slice.localBounds = {-20, -20, -1, 20, 20, 1};
    RegionOptions options;
    options.kind = "surface";
    options.volume = "vol";
    RegionResult result;
    session.read(attachment, [&](const auto& db, const auto& txn, const auto&) { result = db.geometryRegion(txn, slice, options); });
    return result;
}
} // namespace

int main() try
{
    TemporaryDirectory directory;
    const auto served = directory.path / "served";
    writeSurface(served / "seg-a.tifxyz", 30);
    std::ofstream(served / "ink.jpg") << "jpeg bytes";
    FileServer server(served);
    auto sample = vcr::catalog::parseOpenDataManifest(sampleJson(server, "seg-a.tifxyz/", "missing.tifxyz/"))
                      .samples.front();
    const auto surfaces = sample.surfaces(sample.segments.front());
    require(surfaces.size() == 1, "one catalog surface");
    const auto uuid = catalogSurfaceUuid(surfaces.front());
    // RFC 4122 version 5, as Python uuid.uuid5 computes it.
    require(util::nameUuid("3f0c2b8e-6a8d-4d5f-9a3e-7c1b2d4e5f60", "sample-seg-a-derived-native-L0-0123456789abcdef") ==
                "d25d990e-3c53-59d3-9c1c-ad1de1b2956e",
            "namespaced UUIDv5");

    ObjectStoreSession session;
    const auto cache = directory.path / "cache";
    CatalogStoreReport report;
    const auto catalog = attachCatalog(session, sample, cache, {}, std::nullopt, &report);
    require(report.added == 2 && report.surfaces == 2, "placeholders for every catalog surface");
    require(session.path(catalog) == catalogSampleStorePath(cache, "sample"), "store path");
    require(session.writable(catalog), "the sample store accepts local edits without another store");
    require(attachCatalog(session, sample, cache) == catalog, "an attached sample store is reused");

    // Placeholder metadata comes from the catalog, never the published meta.json.
    std::optional<ObjectRecord> record;
    session.read(catalog, [&](const auto& db, const auto& txn, const auto&) { record = db.objectByUuid(txn, uuid); });
    require(record && record->name == "w001", "name from the catalog");
    const auto meta = nlohmann::json::parse(record->meta);
    require(meta.at("coordinates").at("volume_uuid") == "vol" && meta.at("provenance").at("tool") == "surface_mesh_generation",
            "binding and provenance");
    require(!meta.contains("format") && !meta.contains("uuid") && !meta.contains("bbox"), "no meta.json keys");
    require(catalogReference(*record)->url == server.url("seg-a.tifxyz") && catalogReference(*record)->files.size() == 1,
            "catalog reference");
    require(record->surface && record->surface->bbox == std::array<double, 6>{10, 10, 26, 17, 15, 33}, "bounds from coverage");

    // Spatial selection works before any geometry is fetched.
    ObjectSelection near;
    near.kind = "surface";
    near.bbox = std::array<double, 6>{0, 0, 25, 20, 20, 35};
    near.volume = "vol";
    require(session.selectObjects({catalog}, near).size() == 1, "placeholder found by its bounds");
    const auto local = [&](const std::string& object) {
        bool found = false;
        session.read(catalog, [&](const auto& db, const auto& txn, const auto&) { found = db.currentGeometry(txn, object).has_value(); });
        return found;
    };
    require(!local(uuid), "no geometry before a read reaches it");
    require(server.requests("seg-a.tifxyz/x.tif") == 0, "nothing fetched at open");

    // A region reaching it reports it and queues one fetch; the geometry arrives.
    auto first = regionAt(session, catalog, 30);
    require(first.runs.empty() && first.remote == std::vector<std::string>{uuid}, "region reports the placeholder");
    (void)regionAt(session, catalog, 30);
    waitFor([&] { return local(uuid); }, "fetched");
    auto events = session.takeEvents();
    require(server.requests("seg-a.tifxyz/x.tif") == 1, "concurrent reads share one fetch");
    const auto fetched = regionAt(session, catalog, 30);
    require(!fetched.runs.empty() && fetched.remote.empty(), "region has the geometry once local");
    bool sawCommit = false;
    for (auto& event : session.takeEvents()) events.push_back(std::move(event));
    for (const auto& event : events)
        sawCommit |= event.kind == StoreSessionEvent::Kind::Committed && event.attachment == catalog;
    require(sawCommit, "the fetched geometry is a commit event");
    session.read(catalog, [&](const auto& db, const auto& txn, const auto&) {
        const auto geometry = db.currentGeometry(txn, uuid);
        require(geometry && nlohmann::json::parse(geometry->attributes).at("source_url") == server.url("seg-a.tifxyz"),
                "revision records its source");
        require(db.objectByUuid(txn, uuid)->name == "w001", "record not replaced by meta.json");
        require(db.objectByUuid(txn, uuid)->surface->gridWidth == 8, "grid filled at fetch");
    });
    regionAt(session, catalog, 900);  // reaches the offline placeholder
    // A failed fetch reports unavailable, caches nothing, and is retried.
    const auto offline = catalogSurfaceUuid(sample.surfaces(sample.segments[1]).front());
    mustFail([&] { session.ensureGeometry(catalog, offline); });
    require(!local(offline), "failure not cached");
    writeSurface(served / "missing.tifxyz", 900);
    session.ensureGeometry(catalog, offline);
    require(local(offline), "retried on the next read");

    // Published files land in the artifact directory.
    const auto ink = publishedFile(session, catalog, uuid, "ink-detection.jpg");
    require(fs::is_regular_file(ink) && ink.parent_path() == session.artifactDirectory(catalog, uuid, false),
            "published image fetched");

    // Importing makes the same object in the user's store, which then hides the catalog's.
    OpenOptions user;
    user.author = "catalog-test";
    { auto created = ObjectStore::create(directory.path / "user.vcobj", user); }
    const auto mine = session.attach(directory.path / "user.vcobj", user);
    session.importObject(catalog, mine, uuid);
    const auto both = session.selectObjects({mine, catalog}, near);
    require(both.size() == 1 && both.front().attachment == mine, "imported copy shadows the catalog surface");
    const auto regions = session.geometryRegion({mine, catalog}, [] {
        RegionSlice slice;
        slice.origin = {12, 12, 30};
        slice.localBounds = {-20, -20, -1, 20, 20, 1};
        return slice;
    }(), [] {
        RegionOptions options;
        options.kind = "surface";
        options.volume = "vol";
        return options;
    }());
    require(!regions[0].result.runs.empty() && regions[1].result.runs.empty(), "region shadowing");
    std::string firstRevision;
    session.read(mine, [&](const auto& db, const auto& txn, const auto&) { firstRevision = db.currentObjectRevision(txn, uuid)->uuid; });
    session.act(mine, {{uuid, firstRevision}}, [&](auto& db, auto& txn) { db.setMeta(txn, uuid, "review", "\"edited\""); });
    session.restoreImported(mine, uuid);
    session.read(mine, [&](const auto& db, const auto& txn, const auto&) {
        require(!nlohmann::json::parse(db.objectByUuid(txn, uuid)->meta).contains("review"), "reset to the imported version");
    });

    // Refresh follows the catalog only when asked; a changed tifxyz is fetched again.
    writeSurface(served / "seg-a-v2.tifxyz", 31);
    const auto revised = vcr::catalog::parseOpenDataManifest(sampleJson(server, "seg-a-v2.tifxyz/", "missing.tifxyz/"))
                             .samples.front();
    const auto refreshed = refreshCatalog(session, catalog, revised);
    require(refreshed.updated == 1 && refreshed.stale == 1, "refresh updates and marks stale");
    waitFor([&] { return server.requests("seg-a-v2.tifxyz/x.tif") == 1; }, "stale surface fetched again");
    waitFor([&] {
        bool fresh = false;
        session.read(catalog, [&](const auto& db, const auto& txn, const auto&) {
            fresh = nlohmann::json::parse(db.currentGeometry(txn, uuid)->attributes).at("source_url") ==
                    server.url("seg-a-v2.tifxyz");
        });
        return fresh;
    }, "new revision");
    session.read(mine, [&](const auto& db, const auto& txn, const auto&) {
        require(nlohmann::json::parse(db.currentGeometry(txn, uuid)->attributes).at("source_url") == server.url("seg-a.tifxyz"),
                "user store untouched by refresh");
    });
    // Editing the fetched sample itself survives refresh and reopening.
    session.editCurrent(catalog, {uuid}, [&](auto& db, auto& txn, const auto& id) {
        auto geometry = db.currentGeometry(txn, id);
        const auto view = GeometrySnapshot(db, txn).viewStored(id);
        db.editGeometry(txn, id, view.surfaceEdits(0, 0, 1, 1, {{42, 43, 44}}), geometry->uuid);
    });
    std::string editedGeometry;
    session.read(catalog, [&](const auto& db, const auto& txn, const auto&) {
        editedGeometry = db.currentGeometry(txn, uuid)->uuid;
    });
    const auto kept = refreshCatalog(session, catalog, sample);
    require(kept.preserved == 1 && kept.stale == 0, "refresh preserves locally edited geometry");
    session.detach(catalog);
    const auto reopened = attachCatalog(session, sample, cache);
    session.read(reopened, [&](const auto& db, const auto& txn, const auto&) {
        require(db.currentGeometry(txn, uuid)->uuid == editedGeometry, "edits survive reopening the sample");
        require(GeometrySnapshot(db, txn).surface(uuid).vertex(0, 0) == vc::geometry::SurfacePoint{42, 43, 44},
                "saved coordinates survive refresh and reopening");
    });
    session.detach(reopened);
    OpenOptions readOnly;
    readOnly.readOnly = true;
    const auto viewed = attachCatalog(session, sample, cache, readOnly);
    require(!session.writable(viewed), "explicit read-only sample attachments remain read-only");
    mustFail([&] { session.editRecords(viewed, {uuid}, [](auto& r) { r.name = "forbidden"; }); });
    std::cout << "ok\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "FAILED: " << error.what() << '\n';
    return 1;
}
