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
// as stored in the files (uint16 projections, float32 reduced darks and flats). The store's layout and writing
// are the library's (helirec/projection_store.hpp, on vc_reader's zarr::Writer); each file is one shard, written once and, unless --no_verify, read back and compared. Running it
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

#include "helirec/projection_store.hpp"
#include "vc_reader/zarr_write.hpp"

namespace fs = std::filesystem;
using vcr::Index;
using vcr::zarr::Json;

namespace {

constexpr int kLevels = 4;

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

// what the library declares a scan group from (helirec/projection_store.hpp)
std::string spec_of(const Scan& s, const Json& tile) {
    const std::size_t n = std::size_t(s.frames());
    std::map<std::int64_t, Json> sources;
    Json values = Json::object();
    for (const auto& [reduce, jobs] : s.levels)
        for (const auto& job : jobs) {
            sources.emplace(job.first, Json{{"file", job.source}, {"source_first", job.source_first}, {"first", job.first}, {"count", job.count}});
            for (const auto& [key, list] : job.metadata.items()) {
                if (!list.is_array() || std::int64_t(list.size()) != job.count) continue;
                if (!values.contains(key)) values[key] = Json::array_t(n);  // n nulls
                for (std::int64_t i = 0; i < job.count; ++i) values[key][std::size_t(job.first - s.base + i)] = list[std::size_t(i)];
            }
        }
    Json files = Json::array();
    for (auto& [first, entry] : sources) files.push_back(std::move(entry));
    return Json{{"tile", s.tile}, {"scan", s.name}, {"role", s.role}, {"rows", s.rows}, {"cols", s.cols}, {"frame_first", s.base}, {"frames", s.frames()},
                {"tile_attributes", tile}, {"files", files}, {"frame_values", values}}.dump();
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
    struct Work {
        const Scan* scan;
        const Job* job;
        helirec::ProjectionStoreWriter* writer;
        Index shape;
    };
    std::vector<std::unique_ptr<helirec::ProjectionStoreWriter>> writers;
    std::vector<Work> work;
    for (const auto& [key, s] : scans) {
        const std::string spec = spec_of(s, tiles.at(s.tile));
        for (const auto& [l, jobs] : s.levels) {
            const std::string dtype = jobs.front().dtype == "<u2" ? "uint16" : "float32";
            helirec::declare_projection_store_scan(root.string(), spec, l, s.source.at(l), dtype, level);
            writers.push_back(std::make_unique<helirec::ProjectionStoreWriter>((root / s.tile / s.name).string(), l));
            for (const auto& job : jobs) {
                if (job.dtype != jobs.front().dtype) throw std::runtime_error(key + ": files of one level differ in dtype");
                work.push_back({&s, &job, writers.back().get(), {job.count, s.rows >> l, s.cols >> l}});
            }
        }
    }

    std::atomic<std::size_t> next{0}, finished{0}, written{0}, skipped{0}, incomplete{0}, bytes{0};
    std::mutex report;
    std::string failure;
    const auto t0 = std::chrono::steady_clock::now();
    auto run = [&] {
        std::vector<std::uint8_t> data;
        for (std::size_t i; (i = next++) < work.size();) {
            const Work& w = work[i];
            try {
                if (w.writer->holds(w.job->first, w.job->count)) {
                    ++skipped;
                } else {
                    const std::size_t size = std::size_t(w.shape[0] * w.shape[1] * w.shape[2]) * (w.job->dtype == "<u2" ? 2 : 4);
                    data.resize(size);
                    if (!read_file(*w.job, w.shape, data)) {
                        ++incomplete;
                        ++finished;
                        std::lock_guard lock(report);
                        std::fprintf(stderr, "not complete, left out: %s\n", w.job->file.c_str());
                        continue;
                    }
                    w.writer->write(w.job->first, w.job->count, data.data(), verify);
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
