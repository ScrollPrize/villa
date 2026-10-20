// Persistence benchmark on real tifxyz segments. Never writes to the input.
#include "vc_object_store/objects/ObjectStore.hpp"
#include "spatial/PatchGrid.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <set>
#include <stdexcept>

namespace {
using namespace vc::objects;
using Clock = std::chrono::steady_clock;
using Json = nlohmann::json;
namespace fs = std::filesystem;
double ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
Json stats(std::vector<double> v) {
    if (v.empty()) return nullptr;
    std::sort(v.begin(), v.end());
    auto q = [&](double p) { return v[static_cast<size_t>(p * (v.size() - 1))]; };
    return {{"n", v.size()}, {"mean", std::accumulate(v.begin(), v.end(), 0.0) / v.size()},
        {"p50", q(.50)}, {"p95", q(.95)}, {"p99", q(.99)}, {"max", v.back()}};
}
uintmax_t fileBytes(const fs::path& p) { return fs::exists(p) ? fs::file_size(p) : 0; }
bool valid(const cv::Vec3f& p) {
    return std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2])
        && p[0] >= 0 && p[1] >= 0 && p[2] >= 0;
}
struct Update { int row, col; cv::Vec3f world; };
struct Cleanup {
    fs::path path;
    ~Cleanup() { std::error_code ec; fs::remove_all(path, ec); }
};
}

int main(int argc, char** argv) try {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        std::cout << "bench_object_edits SEGMENT NEW_WORK_DIR RADIUS DB_EDITS FILE_SAVES [per-point]\n"
                     "Uses 64 deterministic brush centers, alternating +/- XYZ translations.\n"
                     "Reports milliseconds; verifies full current/base/middle geometry bytes.\n";
        return 0;
    }
    if (argc < 6 || argc > 7) throw std::runtime_error("Use --help");
    const fs::path source = fs::canonical(argv[1]), work = fs::absolute(argv[2]);
    const double radius = std::stod(argv[3]);
    const int count = std::stoi(argv[4]), saves = std::stoi(argv[5]);
    const bool perPoint = argc == 7 && std::string(argv[6]) == "per-point";
    if (!std::isfinite(radius) || radius < 1 || radius > 512 || count < 2 || saves < 2)
        throw std::runtime_error("Invalid radius or repetition count");
    const int extent = static_cast<int>(std::ceil(radius));
    if (!fs::create_directory(work)) throw std::runtime_error("Work directory must not exist");
    Cleanup cleanup{work};
    fs::copy(source, work / "surface", fs::copy_options::recursive);
    auto surface = vc::object_links::readPatchGrid(work / "surface");
    auto original = surface.rawPoints().clone();
    if (!original.isContinuous()) throw std::runtime_error("Expected contiguous points");
    const int rows = original.rows, cols = original.cols;
    std::vector<cv::Point> candidates;
    for (int r = extent; r < rows - extent; ++r)
        for (int c = extent; c < cols - extent; ++c)
            if (valid(original(r, c))) candidates.emplace_back(c, r);
    if (candidates.empty()) throw std::runtime_error("No valid interior brush center");
    std::vector<std::vector<cv::Point>> footprints;
    std::vector<double> pointsCounts, tileCounts;
    for (size_t i = 0; i < 64; ++i) {
        const auto center = candidates[(i * 15485863ULL + candidates.size() / 2) % candidates.size()];
        std::vector<cv::Point> points;
        std::set<std::pair<int, int>> tiles;
        for (int r = center.y - extent; r <= center.y + extent; ++r)
            for (int c = center.x - extent; c <= center.x + extent; ++c)
                if ((r-center.y)*(r-center.y)+(c-center.x)*(c-center.x) <= radius*radius
                    && valid(original(r,c))) {
                    points.emplace_back(c, r);
                    tiles.emplace(r / 128, c / 128);
                }
        pointsCounts.push_back(points.size()); tileCounts.push_back(tiles.size());
        footprints.push_back(std::move(points));
    }
    auto updates = [&](const cv::Mat_<cv::Vec3f>& current, int iteration) {
        std::vector<Update> result;
        const auto& points = footprints[iteration % footprints.size()];
        result.reserve(points.size());
        const float sign = (iteration / footprints.size()) % 2 ? -1.f : 1.f;
        for (const auto& p : points)
            result.push_back({p.y, p.x, current(p.y,p.x) + cv::Vec3f(.125f, -.0625f, .25f) * sign});
        return result;
    };
    auto pack = [&](const std::vector<Update>& changes) {
        std::vector<GeometryArrayEdit> edits;
        for (const auto& u : changes) {
            const int64_t offset = (int64_t(u.row) * cols + u.col) * 3;
            if (perPoint || edits.empty() ||
                edits.back().elementOffset + int64_t(edits.back().bytes.size()/4) != offset)
                edits.push_back({"points", offset, {}});
            auto& bytes = edits.back().bytes;
            const auto* ptr = reinterpret_cast<const uint8_t*>(u.world.val);
            bytes.insert(bytes.end(), ptr, ptr + 12);
        }
        return edits;
    };
    auto apply = [](cv::Mat_<cv::Vec3f>& current, const std::vector<Update>& changes) {
        for (const auto& u : changes) current(u.row, u.col) = u.world;
    };
    OpenOptions options; options.author = "edit-benchmark";
    auto db = ObjectStore::create(work / "objects.vcobj", options);
    GeometryData data; data.kind = "patch";
    data.attributes = Json({{"scale", {surface.scale()[0], surface.scale()[1]}}}).dump();
    const auto* begin = reinterpret_cast<const uint8_t*>(original.data);
    data.arrays.push_back({"points", "<f4", {rows, cols, 3},
                          std::vector<uint8_t>(begin, begin + original.total()*12)});
    std::string uid, revision;
    {
        auto w = db.write(); ObjectRecord object; object.kind = "patch";
        uid = db.upsertObject(w, object);
        revision = db.putGeometry(w, uid, data); w.commit();
    }
    const auto first = revision;
    db.checkpoint();
    const auto initialBytes = fileBytes(db.path());
    cv::Mat_<cv::Vec3f> current = original.clone(), middle, previous;
    std::string middleRevision, previousRevision;
    std::vector<double> queueTimes, packTimes, editTimes, commitTimes, totalTimes;
    uintmax_t maxWal = 0;
    for (int i = 0; i < count; ++i) {
        auto start = Clock::now();
        auto changes = updates(current, i);
        queueTimes.push_back(ms(start));
        start = Clock::now();
        auto edits = pack(changes);
        packTimes.push_back(ms(start));
        start = Clock::now();
        auto w = db.write();
        revision = db.editGeometry(w, uid, edits, revision);
        editTimes.push_back(ms(start));
        auto commitStart = Clock::now();
        w.commit();
        commitTimes.push_back(ms(commitStart)); totalTimes.push_back(ms(start));
        apply(current, changes);
        maxWal = std::max(maxWal, fileBytes(db.path().string() + "-wal"));
        if (i == count/2 - 1) { middle = current.clone(); middleRevision = revision; }
        if (i == count - 2) { previous = current.clone(); previousRevision = revision; }
    }
    auto verify = [&](const std::string& rev, const cv::Mat_<cv::Vec3f>& expected) {
        auto start = Clock::now();
        auto r = db.read(); const auto bytes = db.readGeometryArray(r, rev, "points");
        const auto elapsed = ms(start);
        if (bytes.size() != expected.total()*12 ||
            std::memcmp(bytes.data(), expected.data, bytes.size()) != 0)
            throw std::runtime_error("Geometry verification failed");
        return elapsed;
    };
    const auto latestRead = verify(revision, current);
    const auto previousRead = verify(previousRevision, previous);
    const auto middleRead = verify(middleRevision, middle);
    const auto baseRead = verify(first, original);
    db.checkpoint();
    const auto growth = fileBytes(db.path()) - initialBytes;

    // Compare durable database edits with rewriting all three XYZ TIFFs.
    std::vector<double> fileTimes;
    auto* filePoints = surface.rawPointsPtr();
    for (int i = 0; i < saves; ++i) {
        const auto changes = updates(*filePoints, i);
        const auto start = Clock::now();
        apply(*filePoints, changes); vc::object_links::writePatchGrid(work / "surface", surface);
        fileTimes.push_back(ms(start));
    }
    auto reopened = vc::object_links::readPatchGrid(work / "surface");
    auto reread = reopened.rawPoints();
    if (reread.size() != filePoints->size() ||
        std::memcmp(reread.data, filePoints->data, filePoints->total()*12) != 0)
        throw std::runtime_error("File save verification failed");
    const int window = std::min(count / 2, 100);
    Json output = {
        {"source", source.string()}, {"shape", {rows, cols, 3}}, {"radius", radius},
        {"packing", perPoint ? "per-point" : "contiguous-runs"}, {"points", stats(pointsCounts)},
        {"tiles", stats(tileCounts)}, {"queue_ms", stats(queueTimes)}, {"pack_ms", stats(packTimes)},
        {"edit_ms", stats(editTimes)}, {"commit_ms", stats(commitTimes)}, {"db_ms", stats(totalTimes)},
        {"first_window_ms", stats({totalTimes.begin(), totalTimes.begin()+window})},
        {"last_window_ms", stats({totalTimes.end()-window, totalTimes.end()})},
        {"file_ms", stats(fileTimes)}, {"file_first_ms", fileTimes.front()},
        {"file_warm_ms", stats({fileTimes.begin()+1, fileTimes.end()})},
        {"initial_bytes", initialBytes}, {"growth_bytes", growth}, {"max_wal_bytes", maxWal},
        {"latest_read_ms", latestRead}, {"previous_read_ms", previousRead},
        {"middle_read_ms", middleRead}, {"base_read_ms", baseRead},
        {"verified", true}};
    std::cout << output.dump() << '\n';
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
