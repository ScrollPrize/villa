// Native, warm-cache comparison of DB region loading and VC3D's surface index.
// See docs/object_storage.md for the deliberately limited comparison contract.
#include "vc/core/objects/ObjectStore.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/SurfacePatchIndex.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

using namespace vc::objects;
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
using Key = std::pair<int, int>;
using Blocks = std::map<Key, std::vector<uint8_t>>;

struct TempDir {
    std::filesystem::path path;
    TempDir() {
        for (int i = 0; i < 100; ++i) {
            path = std::filesystem::temp_directory_path() /
                ("vc_surface_perf_" + std::to_string(Clock::now().time_since_epoch().count()) + "_" + std::to_string(i));
            if (std::filesystem::create_directory(path)) return;
        }
        throw std::runtime_error("cannot create benchmark directory");
    }
    ~TempDir() { std::error_code ec; std::filesystem::remove_all(path, ec); }
};

template<class F> double elapsed(F&& f) {
    auto start = Clock::now(); f();
    return std::chrono::duration<double, std::micro>(Clock::now() - start).count();
}

Json stats(std::vector<double> samples) {
    auto raw = samples;
    std::sort(samples.begin(), samples.end());
    return {{"samples_us", raw}, {"min_us", samples.front()},
        {"median_us", samples[samples.size()/2]}, {"max_us", samples.back()},
        {"mean_us", std::accumulate(samples.begin(), samples.end(), 0.)/samples.size()}};
}

Json run(int rows, int cols, size_t cacheBytes) {
    TempDir tmp;
    cv::Mat_<cv::Vec3f> points(rows, cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            points(r,c) = cv::Vec3f(float(c), float(r), 1.f + float(c)/16.f);
    auto surface = std::make_shared<QuadSurface>(points, cv::Vec2f(1,1));
    SurfacePatchIndex index;
    const auto buildUs = elapsed([&] { index.rebuild({surface}, 0, false); });
    surface->id = "surface";
    surface->save(tmp.path / "surface");
    auto lazy = std::make_shared<QuadSurface>(tmp.path / "surface");
    SurfacePatchIndex mapped;
    const auto mappedBuildUs = elapsed([&] { mapped.rebuild({lazy}); });
    if (lazy->isLoaded()) throw std::runtime_error("mapped baseline unexpectedly materialized surface");

    OpenOptions opts; opts.author = "benchmark";
    opts.geometryCacheBytes = cacheBytes;
    auto db = ObjectStore::create(tmp.path / "objects.vcobj", opts);
    GeometryData data; data.kind = "patch"; data.attributes = R"({"scale":[1,1]})";
    data.arrays.push_back({"points", "<f4", {rows,cols,3}, std::vector<uint8_t>(points.total()*12)});
    std::memcpy(data.arrays[0].bytes.data(), points.data, data.arrays[0].bytes.size());
    const auto publishUs = elapsed([&] {
        auto w = db.write(); ObjectRecord record; record.kind = "patch";
        const auto id = db.upsertObject(w, record); db.putGeometry(w, id, data); w.commit();
    });
    Json out = {{"rows",rows}, {"cols",cols}, {"vc3d_build_us",buildUs},
        {"vc3d_mapped_build_us",mappedBuildUs}, {"db_publish_us",publishUs},
        {"vc3d_index_tiles",index.patchCount()}, {"queries",Json::array()}};
    // Non-integral bounds avoid ambiguous grid-edge inclusion in this planar fixture.
    for (const std::string name : {"small", "boundary", "viewport", "miss"}) {
        const float x = name == "boundary" ? 127.5f : 400.25f;
        const float y = name == "boundary" ? 31.5f : float(std::min(400,rows/2)) + .25f;
        const float width = name == "viewport" ? 255.f : 1.f;
        SurfacePatchIndex::TriangleQuery query;
        query.bounds.low = {x, y, -1.f};
        query.bounds.high = {x+width, std::min(y+ (name == "viewport" ? 15.f : 1.f), float(rows)-1.25f), 100000.f};
        if (name == "miss") { query.bounds.low[0] = float(cols)+10; query.bounds.high[0] = float(cols)+20; }
        ObjectSelection selection;
        selection.bbox = std::array<double,6>{query.bounds.low[0],query.bounds.low[1],query.bounds.low[2],
            query.bounds.high[0],query.bounds.high[1],query.bounds.high[2]};
        auto vcKeys = [&] {
            std::set<Key> keys;
            auto q = query;
            // Match the conservative query padding used by VC3D's plane
            // intersection path: its R-tree boxes are quantized (~1.54 units).
            q.bounds.low -= cv::Vec3f(4,4,4);
            q.bounds.high += cv::Vec3f(4,4,4);
            // For this planar fixture only, XYZ box minima recover source grid
            // positions by rounding to the nearest 8-cell anchor. Normalize
            // those to DB 32-cell tiles and recheck their exact planar bounds.
            q.patchFilter = [&](const SurfacePatchIndex::PatchBounds& b) {
                int r = int(std::lround(b.low[1]/8))*8/32*32;
                int c = int(std::lround(b.low[0]/8))*8/32*32;
                if (c+32 >= query.bounds.low[0] && c <= query.bounds.high[0] &&
                    r+32 >= query.bounds.low[1] && r <= query.bounds.high[1]) keys.emplace(r,c);
                return false;
            };
            index.forEachTriangle(q, [](const auto&) {}); return keys;
        };
        auto vcLoad = [&] {
            Blocks blocks;
            for (const auto& [r,c] : vcKeys()) {
                int nr = std::min(33,rows-r), nc = std::min(33,cols-c);
                auto& bytes = blocks[{r,c}]; bytes.resize(size_t(nr)*nc*12);
                for (int j=0; j<nr; ++j)
                    std::memcpy(bytes.data()+size_t(j)*nc*12, &points(r+j,c), size_t(nc)*12);
            }
            return blocks;
        };
        auto dbLoad = [&] {
            auto txn = db.read(); Blocks blocks;
            for (const auto& region : db.spatialRegions(txn, selection)) {
                blocks[{int(region.start.at(0)),int(region.start.at(1))}] = db.readGeometryRectangle(
                    txn, region.revisionUuid.value(), "points",region.start[0],region.start[1],region.shape[0],region.shape[1]);
            }
            return blocks;
        };
        auto dbBatchLoad = [&](const Txn& txn) {
            const auto regions = db.spatialRegions(txn, selection);
            Blocks blocks;
            if (regions.empty()) return blocks;
            std::vector<std::array<int64_t,4>> rectangles;
            for (const auto& r : regions) rectangles.push_back({r.start[0],r.start[1],r.shape[0],r.shape[1]});
            auto values = db.readGeometryRectangles(txn, regions.front().revisionUuid.value(), "points", rectangles);
            for (size_t i=0;i<regions.size();++i)
                blocks[{int(regions[i].start[0]),int(regions[i].start[1])}] = std::move(values[i]);
            return blocks;
        };
        const auto expected = vcLoad();
        const auto actual = dbLoad();
        if (actual != expected) throw std::runtime_error("DB/native region bytes differ: " + name);
        auto snapshot = db.read();
        if (dbBatchLoad(snapshot) != expected) throw std::runtime_error("batched region bytes differ");
        const auto knownRegions = db.spatialRegions(snapshot, selection);
        std::vector<std::array<int64_t,4>> knownRectangles;
        for (const auto& r : knownRegions) knownRectangles.push_back({r.start[0],r.start[1],r.shape[0],r.shape[1]});
        std::set<Key> regionOracle, actualKeys;
        for (int r=0;r<rows-1;r+=32) for(int c=0;c<cols-1;c+=32)
            if (c+32 >= query.bounds.low[0] && c <= query.bounds.high[0] &&
                r+32 >= query.bounds.low[1] && r <= query.bounds.high[1]) regionOracle.emplace(r,c);
        for (const auto& [key,bytes] : actual) actualKeys.insert(key);
        if (actualKeys != regionOracle) throw std::runtime_error("region oracle differs");
        using Tri = std::tuple<int,int,int>;
        auto triangles = [&](SurfacePatchIndex& source) {
            std::set<Tri> result;
            source.forEachTriangle(query,[&](const auto& t) { result.emplace(t.i,t.j,t.triangleIndex); });
            return result;
        };
        const auto tri = triangles(index);
        if (triangles(mapped) != tri) throw std::runtime_error("resident/mapped triangles differ");
        // Independent planar cell AABB oracle, outside timings.
        std::set<Tri> oracle;
        for (int r=0; r<rows-1; ++r) for (int c=0; c<cols-1; ++c)
            if (c+1 >= query.bounds.low[0] && c <= query.bounds.high[0] &&
                r+1 >= query.bounds.low[1] && r <= query.bounds.high[1])
                for (int t=0;t<2;++t) oracle.emplace(c,r,t);
        // The generic unpadded TriangleQuery can miss tiles at quantized box
        // boundaries. Record this explicitly instead of treating less work as
        // an equivalent-result speed comparison. Plane queries add padding.
        if (!std::includes(oracle.begin(),oracle.end(),tri.begin(),tri.end()))
            throw std::runtime_error("native triangles outside oracle");
        size_t sink = 0;
        std::vector<std::pair<std::string,std::function<void()>>> ops{
            {"vc3d_index_normalized", [&] {sink += vcKeys().size();}},
            {"db_index", [&] {auto txn=db.read(); sink += db.spatialRegions(txn,selection).size();}},
            {"vc3d_index_and_resident_copy", [&] {sink += vcLoad().size();}},
            {"db_index_and_rectangles", [&] {sink += dbLoad().size();}},
            {"db_index_and_batch", [&] {auto txn=db.read(); sink += dbBatchLoad(txn).size();}},
            {"db_index_and_batch_pinned_snapshot", [&] {sink += dbBatchLoad(snapshot).size();}},
            {"db_known_rectangles_pinned_snapshot", [&] {
                if (!knownRegions.empty()) sink += db.readGeometryRectangles(snapshot,
                    knownRegions.front().revisionUuid.value(),"points",knownRectangles).size();}},
            {"vc3d_triangles_resident", [&] {index.forEachTriangle(query,[&](const auto&) {++sink;});}},
            {"vc3d_triangles_mapped", [&] {mapped.forEachTriangle(query,[&](const auto&) {++sink;});}}
        };
        for (auto& op : ops) for (int i=0;i<10;++i) op.second();
        std::vector<std::vector<double>> samples(ops.size());
        for (size_t round=0;round<9;++round) for (size_t k=0;k<ops.size();++k) {
            size_t i=(k+round)%ops.size();
            samples[i].push_back(elapsed([&] {for(int n=0;n<20;++n) ops[i].second();})/20.);
        }
        Json results;
        for(size_t i=0;i<ops.size();++i) results[ops[i].first]=stats(samples[i]);
        size_t bytes=0; for(const auto& [key,block] : expected) bytes+=block.size();
        out["queries"].push_back({{"name",name},{"regions",expected.size()},{"bytes",bytes},
            {"triangles",tri.size()},{"oracle_triangles",oracle.size()},
            {"native_triangle_missing",oracle.size()-tri.size()},{"timings",results},{"sink",sink}});
    }
    if (lazy->isLoaded()) throw std::runtime_error("queries materialized mapped surface");
    return out;
}

int main(int argc, char** argv) {
    try {
        if (argc < 2 || argc > 3) throw std::runtime_error("usage: benchmark_object_store_surface OUTPUT.json [CACHE_BYTES]");
        const size_t cacheBytes = argc == 3 ? std::stoull(argv[2]) : OpenOptions{}.geometryCacheBytes;
        if constexpr (std::endian::native != std::endian::little)
            throw std::runtime_error("fixture byte serialization requires little endian");
        Json report = {{"warmup_calls",10},{"batches",9},{"calls_per_batch",20},
            {"units","microseconds per call; batch averages"},
            {"sampling_stride",1}, {"db_storage_tile_vertices",128},
            {"geometry_cache_bytes_per_connection",cacheBytes},
            {"fixture","float32 XYZ=(column,row,1+column/16); scale=(1,1)"},
            {"candidate_adapter","VC3D query padded by 4; quantized 8-cell anchors rounded and normalized to exact 32-cell regions"},
            {"cache_policy","warm repeated ROI; resident/mapped VC3D; DB connection cache configurable; new transaction unless pinned_snapshot"},
            {"shapes",Json::array()}};
        for (auto [rows,cols] : {Key{1025,1025},Key{2049,2049},Key{65,65537}}) {
            report["shapes"].push_back(run(rows,cols,cacheBytes));
            std::ofstream file(argv[1]); file << report.dump(2) << '\n';
            if (!file) throw std::runtime_error("cannot write output report");
            std::cerr << "validated " << rows << 'x' << cols << '\n';
        }
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
