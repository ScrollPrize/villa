// helirec-projections-zarr: the reduced projection copies of a mosaic (python -m helirec.download_projections) as one
// multiscale Zarr v3 store.
//
//     helirec-projections-zarr OUT.zarr SCALE_DIR [SCALE_DIR ...] [--tiles T1-03 ...] [--threads N] [--level L] [--no_verify]
//
// Each SCALE_DIR holds a plan.json and the files it lists, at one scale (reduce 1, 2 or 3). The store:
//
//     OUT.zarr/<tile>/<scan>/        a group per detector scan of a tile (darks, flats, the projections)
//         0, 1, 2, 3                 (frame, row, col) at 1, 1/2, 1/4, 1/8 of the detector; every frame at every level
//         angle_deg, epoch, height_mm, original_frame_index      one value per frame (NaN / -1 where no file gave one)
//
// A level is written from the SCALE_DIR of its reduce; a level no SCALE_DIR gives is an array without chunks (it
// reads as zeros) and the group's "helirec" attribute lists which levels are stored and, per detector file, which of
// the scan's frames it holds ("files": frame `first` + i of the scan is frame `source_first` + i of `file`). Value (j, k) of level l is the
// wavelet low band at the stored pixel (j, k) 2^l (helirec/raw_reader.hpp), so the levels share an origin: the
// "multiscales" attribute declares scales and no translation.
//
// Arrays: blosc (bitshuffle, lz4), chunks of one frame by 128 rows by all columns, a shard per 100 frames, values
// as stored in the files (uint16 projections, float32 reduced darks and flats). Writing is vc_reader's
// (zarr::Writer); each file is one shard, written once and, unless --no_verify, read back and compared. Running it
// again skips the files already stored.
#include <hdf5.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "vc_reader/array.hpp"
#include "vc_reader/zarr_write.hpp"

namespace fs = std::filesystem;
using vcr::Index;
using vcr::zarr::Json;

namespace {

constexpr int kLevels = 4, kChunkRows = 128, kShardFrames = 100;

struct Job {
    fs::path file;
    std::string dataset, dtype;  // numpy typestr of the plan's dtype
    std::string source;  // the detector's file under the scan's directory
    std::int64_t first = 0, count = 0, source_first = 0;  // frames [first, first + count) are its frames from source_first on
    Json metadata;
};
struct Scan {
    std::string tile, name, role;
    std::int64_t rows = 0, cols = 0, base = std::numeric_limits<std::int64_t>::max(), end = 0;  // stored size; frames [base, end)
    std::map<int, std::vector<Job>> levels;  // reduce -> files
    std::map<int, std::string> source;       // reduce -> SCALE_DIR
    std::int64_t frames() const { return end - base; }
};

std::mutex h5_mutex;  // libhdf5's serial build is not thread safe

std::string typestr(const std::string& dtype) {
    if (dtype == "uint16") return "<u2";
    if (dtype == "float32") return "<f4";
    throw std::runtime_error("unsupported dtype " + dtype);
}

// the whole dataset, whose shape must be `shape`, as `dtype`; false when the file is not marked complete
bool read_file(const Job& job, const Index& shape, std::vector<std::uint8_t>& out) {
    std::lock_guard lock(h5_mutex);
    const hid_t file = H5Fopen(job.file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) return false;
    bool ok = false;
    std::string error;
    hbool_t complete = 0;
    if (H5Aexists(file, "complete") > 0) {
        const hid_t a = H5Aopen(file, "complete", H5P_DEFAULT);
        H5Aread(a, H5T_NATIVE_HBOOL, &complete);
        H5Aclose(a);
    }
    if (complete) {
        const hid_t data = H5Dopen2(file, job.dataset.c_str(), H5P_DEFAULT);
        if (data < 0) {
            error = "no dataset " + job.dataset;
        } else {
            const hid_t space = H5Dget_space(data);
            hsize_t dims[3] = {0, 0, 0};
            const hid_t type = H5Dget_type(data);
            const hid_t want = job.dtype == "<u2" ? H5T_NATIVE_UINT16 : H5T_NATIVE_FLOAT;
            if (H5Sget_simple_extent_ndims(space) != 3 || H5Sget_simple_extent_dims(space, dims, nullptr) != 3 ||
                std::int64_t(dims[0]) != shape[0] || std::int64_t(dims[1]) != shape[1] || std::int64_t(dims[2]) != shape[2])
                error = "unexpected shape";
            else if (H5Tequal(type, want) <= 0)
                error = "values are not " + job.dtype;  // no conversion: the store holds the file's values
            else
                ok = H5Dread(data, want, H5S_ALL, H5S_ALL, H5P_DEFAULT, out.data()) >= 0;
            if (!ok && error.empty()) error = "cannot read " + job.dataset;
            H5Tclose(type);
            H5Sclose(space);
            H5Dclose(data);
        }
    }
    H5Fclose(file);
    if (!error.empty()) throw std::runtime_error(job.file.string() + ": " + error);
    return ok;
}

vcr::zarr::CreateOptions array_options(const Index& shape, const std::string& dtype, int level) {
    vcr::zarr::CreateOptions o;
    o.version = 3;
    o.shape = shape;
    o.dtype = dtype;
    o.separator = "/";
    o.compressor = "blosc";
    o.cname = "lz4";
    o.shuffle = 2;
    o.level = level;
    const std::int64_t rows = std::min<std::int64_t>(kChunkRows, shape[1]);
    o.chunks = {1, rows, shape[2]};
    o.shardShape = {kShardFrames, (shape[1] + rows - 1) / rows * rows, shape[2]};
    return o;
}

// an existing array must be the one this run would create
vcr::zarr::Writer open_or_create(const fs::path& path, const vcr::zarr::CreateOptions& o) {
    if (!fs::exists(path / "zarr.json")) return vcr::zarr::Writer::create(path, o);
    auto w = vcr::zarr::Writer::open(path);
    if (w.meta().shape != o.shape || w.meta().chunks != o.chunks || w.meta().dtype.substr(1) != o.dtype.substr(1))
        throw std::runtime_error(path.string() + " exists with another shape, chunking or dtype");
    return w;
}

void read_plan(const fs::path& dir, const std::set<std::string>& only, std::map<std::string, Scan>& scans, std::map<std::string, Json>& tiles) {
    std::ifstream in(dir / "plan.json");
    if (!in) throw std::runtime_error("no plan.json in " + dir.string());
    const Json plan = Json::parse(in);
    const int reduce = plan.at("settings").at("reduce").get<int>();
    if (reduce < 1 || reduce >= kLevels) throw std::runtime_error(dir.string() + ": reduce must be 1, 2 or 3");
    for (const auto& t : plan.at("tiles")) {
        Json kept = t;
        for (const char* key : {"source_master", "output_shape", "slices"}) kept.erase(key);  // of one download, not of the tile
        tiles.emplace(t.at("label").get<std::string>(), std::move(kept));
    }
    for (const auto& j : plan.at("jobs")) {
        const auto tile = j.at("tile").get<std::string>();
        if (!only.empty() && !only.count(tile)) continue;
        const fs::path output = j.at("output").get<std::string>();
        const std::string name = output.parent_path().filename().string();
        Scan& s = scans[tile + "/" + name];
        const auto rows = j.at("rows").get<std::int64_t>(), cols = j.at("cols").get<std::int64_t>();
        const auto role = j.at("role").get<std::string>();
        if (s.tile.empty()) {
            s.tile = tile, s.name = name, s.role = role, s.rows = rows, s.cols = cols;
        } else if (s.rows != rows || s.cols != cols || s.role != role) {
            throw std::runtime_error(dir.string() + ": " + tile + "/" + name + " differs between plans");
        }
        if (rows % (1 << (kLevels - 1)) || cols % (1 << (kLevels - 1))) throw std::runtime_error("detector size is not a multiple of 8");
        if (s.source.count(reduce) && s.source[reduce] != dir.string()) throw std::runtime_error("two folders give reduce " + std::to_string(reduce));
        s.source[reduce] = dir.string();
        Job job;
        job.file = dir / output;
        job.dataset = j.at("dataset").get<std::string>();
        job.first = j.at("first").get<std::int64_t>();
        job.count = j.at("count").get<std::int64_t>();
        job.source = fs::path(j.value("relative_file", output.string())).filename().string();
        job.source_first = j.value("source_first", std::int64_t{0});
        // reduced darks and flats are float32 (calibration.at_resolution); the projections keep the source's dtype
        job.dtype = role == "projection" ? typestr(j.at("dtype").get<std::string>()) : "<f4";
        job.metadata = j.value("metadata", Json::object());
        s.base = std::min(s.base, job.first);
        s.end = std::max(s.end, job.first + job.count);
        s.levels[reduce].push_back(std::move(job));
    }
}

// one value per frame of the scan, from the plans' records of the files (the same at every scale)
void write_frame_arrays(const fs::path& group, const Scan& s) {
    const std::size_t n = std::size_t(s.frames());
    std::map<std::string, std::vector<double>> values;
    std::vector<std::int64_t> original(n, -1);
    for (const auto& [reduce, jobs] : s.levels)
        for (const auto& job : jobs) {
            for (std::int64_t i = 0; i < job.count; ++i) original[std::size_t(job.first - s.base + i)] = job.first + i;
            for (const auto& [key, list] : job.metadata.items()) {
                if (!list.is_array() || std::int64_t(list.size()) != job.count) continue;
                auto& v = values.try_emplace(key, n, std::numeric_limits<double>::quiet_NaN()).first->second;
                for (std::int64_t i = 0; i < job.count; ++i)
                    if (list[std::size_t(i)].is_number()) v[std::size_t(job.first - s.base + i)] = list[std::size_t(i)].get<double>();
            }
        }
    auto write = [&](const std::string& name, const std::string& dtype, const void* data, const Json& fill) {
        vcr::zarr::CreateOptions o;
        o.version = 3, o.shape = {std::int64_t(n)}, o.chunks = {std::int64_t(n)}, o.dtype = dtype, o.separator = "/";
        o.compressor = "blosc", o.cname = "lz4", o.shuffle = 1, o.fillValue = fill, o.overwrite = true;
        auto w = vcr::zarr::Writer::create(group / name, o);
        w.writeChunk({0}, static_cast<const std::uint8_t*>(data), n * 8);
    };
    write("original_frame_index", "<i8", original.data(), -1);
    for (const auto& [key, v] : values) write(key, "<f8", v.data(), "NaN");
}

}  // namespace

int main(int argc, char** argv) try {
    std::vector<fs::path> positional;
    std::set<std::string> only;
    unsigned threads = std::max(1u, std::thread::hardware_concurrency());
    int level = 1;
    bool verify = true;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--tiles") {
            while (i + 1 < argc && argv[i + 1][0] != '-') only.insert(argv[++i]);
        } else if (a == "--threads" && i + 1 < argc) {
            threads = unsigned(std::max(1, std::atoi(argv[++i])));
        } else if (a == "--level" && i + 1 < argc) {
            level = std::atoi(argv[++i]);
        } else if (a == "--no_verify" || a == "--no-verify") {
            verify = false;
        } else if (a.rfind("--", 0) == 0) {
            positional.clear();
            break;
        } else {
            positional.emplace_back(a);
        }
    }
    if (positional.size() < 2) {
        std::fprintf(stderr, "usage: helirec-projections-zarr OUT.zarr SCALE_DIR [SCALE_DIR ...] [--tiles T ...] [--threads N] [--level L] [--no_verify]\n"
                             "(see the head of tools/helirec_projections_zarr.cpp and README.md)\n");
        return 2;
    }
    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
    const fs::path root = positional[0];
    std::map<std::string, Scan> scans;
    std::map<std::string, Json> tiles;
    for (std::size_t i = 1; i < positional.size(); ++i) read_plan(positional[i], only, scans, tiles);
    if (scans.empty()) throw std::runtime_error("the plans list no files for these tiles");

    // groups, arrays and attributes first: a reader can open the store while it fills
    vcr::zarr::createGroup(root, Json{{"helirec_projections", {{"format_version", 1}}}}, 3);
    struct Work {
        const Scan* scan;
        const Job* job;
        vcr::zarr::Writer* writer;
        Index shape;
    };
    std::vector<std::unique_ptr<vcr::zarr::Writer>> writers;
    std::vector<Work> work;
    std::set<std::string> tile_groups;
    for (const auto& [key, s] : scans) {
        if (tile_groups.insert(s.tile).second) vcr::zarr::createGroup(root / s.tile, Json{{"helirec_tile", tiles.at(s.tile)}}, 3);
        const fs::path group = root / s.tile / s.name;
        Json stored = Json::array();
        for (int l = 0; l < kLevels; ++l) {
            const Index shape{s.frames(), s.rows >> l, s.cols >> l};
            const auto found = s.levels.find(l);
            // an absent level is declared as its files would be: the detector's uint16, or a reduced dark's float32
            const std::string dtype = found != s.levels.end() ? found->second.front().dtype : (s.role == "projection" || l == 0 ? "<u2" : "<f4");
            auto writer = std::make_unique<vcr::zarr::Writer>(open_or_create(group / std::to_string(l), array_options(shape, dtype, level)));
            Json entry{{"path", std::to_string(l)}, {"reduce", l}, {"stored", found != s.levels.end()}};
            if (found != s.levels.end()) {
                entry["source"] = s.source.at(l);
                for (const auto& job : found->second) {
                    if (job.dtype != dtype) throw std::runtime_error(key + ": files of one level differ in dtype");
                    work.push_back({&s, &job, writer.get(), {job.count, shape[1], shape[2]}});
                }
            }
            stored.push_back(std::move(entry));
            writers.push_back(std::move(writer));
        }
        // which frames of the detector's files the store's frames are (the same at every scale)
        std::map<std::int64_t, Json> sources;
        for (const auto& [reduce, jobs] : s.levels)
            for (const auto& job : jobs)
                sources.emplace(job.first, Json{{"file", job.source}, {"source_first", job.source_first}, {"first", job.first}, {"count", job.count}});
        Json files = Json::array();
        for (auto& [first, entry] : sources) files.push_back(std::move(entry));
        vcr::zarr::MultiscalesOptions m;
        m.name = s.tile + "/" + s.name, m.levels = kLevels - 1, m.factors = {1, 2, 2}, m.axes = {"frame", "row", "col"};
        m.reduction = vcr::zarr::Reduction::Nearest;  // no translation: level values sit at stored pixels (j, k) 2^l
        Json multiscales = vcr::zarr::omeMultiscales(m);
        multiscales["metadata"] = {{"downsampling_method", "JPEG 2000 9/7 wavelet low band"}};
        vcr::zarr::createGroup(group,
                               Json{{"multiscales", Json::array({multiscales})},
                                    {"helirec", {{"role", s.role}, {"tile", s.tile}, {"scan", s.name}, {"frame_first", s.base}, {"frames", s.frames()}, {"levels", stored}, {"files", files}}}},
                               3);
        write_frame_arrays(group, s);
    }

    std::atomic<std::size_t> next{0}, finished{0}, written{0}, skipped{0}, incomplete{0}, bytes{0};
    std::mutex report;
    std::string failure;
    const auto t0 = std::chrono::steady_clock::now();
    auto run = [&] {
        std::vector<std::uint8_t> data, back;
        for (std::size_t i; (i = next++) < work.size();) {
            const Work& w = work[i];
            try {
                const Index start{w.job->first - w.scan->base, 0, 0};
                const auto& chunks = w.writer->meta().chunks;
                const Index first_chunk{start[0], 0, 0}, last_chunk{start[0] + w.shape[0] - 1, (w.shape[1] - 1) / chunks[1], 0};
                // a file is one rewrite of its shard, so its first and last chunks are stored together
                if (w.writer->chunkExists(first_chunk) && w.writer->chunkExists(last_chunk)) {
                    ++skipped;
                } else {
                    const std::size_t size = std::size_t(w.shape[0] * w.shape[1] * w.shape[2]) * w.writer->meta().itemsize();
                    data.resize(size);
                    if (!read_file(*w.job, w.shape, data)) {
                        ++incomplete;
                        ++finished;
                        std::lock_guard lock(report);
                        std::fprintf(stderr, "not complete, left out: %s\n", w.job->file.c_str());
                        continue;
                    }
                    w.writer->writeRegion(start, w.shape, data.data());
                    if (verify) {
                        back.resize(size);
                        w.writer->reader().read_many({start}, {w.shape}, {back.data()});
                        if (std::memcmp(data.data(), back.data(), size) != 0) throw std::runtime_error("stored values differ from the file");
                    }
                    bytes += size;
                    ++written;
                }
            } catch (const std::exception& e) {
                std::lock_guard lock(report);
                if (failure.empty()) failure = w.job->file.string() + ": " + e.what();
                next = work.size();
                return;
            }
            const std::size_t done = ++finished;  // each count is seen by one thread
            if (done % 100 == 0 || done == work.size()) {
                std::lock_guard lock(report);
                const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                std::fprintf(stderr, "%zu/%zu files (%zu already stored), %.1f GiB at %.2f GiB/s\n", done, work.size(), skipped.load(),
                             double(bytes) / double(1ull << 30), double(bytes) / double(1ull << 30) / std::max(seconds, 1e-9));
            }
        }
    };
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < std::min<std::size_t>(threads, work.size()); ++t) pool.emplace_back(run);
    for (auto& t : pool) t.join();
    if (!failure.empty()) throw std::runtime_error(failure);
    std::printf("%s: %zu files written%s, %zu already stored, %zu not complete\n", root.c_str(), written.load(), verify ? " and verified" : "", skipped.load(),
                incomplete.load());
    return incomplete ? 1 : 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "helirec-projections-zarr: %s\n", e.what());
    return 1;
}
