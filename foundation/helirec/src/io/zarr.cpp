#include "helirec/zarr.hpp"

#include <zlib.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <thread>
#include <unordered_map>

#include "helirec/delta3d.hpp"
#include "json.hpp"
#include "remote.hpp"

namespace helirec {

namespace fs = std::filesystem;

namespace {

// ------------------------------------------------------------------------------------------ files
bool read_file(const std::string& path, std::vector<std::uint8_t>& out) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    out.resize(n > 0 ? static_cast<std::size_t>(n) : 0);
    const bool ok = out.empty() || std::fread(out.data(), 1, out.size(), f) == out.size();
    std::fclose(f);
    if (!ok) throw Error("zarr: cannot read " + path);
    return true;
}

void write_file(const std::string& path, const void* data, std::size_t n) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {   // (its directory may be new)
        fs::create_directories(fs::path(path).parent_path());
        f = std::fopen(path.c_str(), "wb");
    }
    if (!f) throw Error("zarr: cannot write " + path);
    const bool ok = std::fwrite(data, 1, n, f) == n;
    if (std::fclose(f) != 0 || !ok) throw Error("zarr: cannot write " + path);
}

void write_text(const std::string& path, const std::string& text) { write_file(path, text.data(), text.size()); }

// ------------------------------------------------------------------------------------------ JSON text
std::string json_string(const std::string& s) {
    std::string o = "\"";
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += static_cast<char>(c); }
        else if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; }
        else o += static_cast<char>(c);
    }
    return o + "\"";
}

std::string json_number(double v) {
    char b[40];
    std::snprintf(b, sizeof b, "%.17g", v);
    std::string s = b;
    if (s.find_first_of(".eEn") == std::string::npos) s += ".0";
    return s;
}

// ------------------------------------------------------------------------------------------ codecs
bool all_zero(const std::uint8_t* p, std::size_t n) { return n == 0 || (p[0] == 0 && std::memcmp(p, p + 1, n - 1) == 0); }

std::vector<std::uint8_t> encode(const ZarrArrayMeta& m, const std::uint8_t* data) {
    const std::size_t n = m.chunk_bytes();
    if (m.compression == "vc-delta3d")
        return delta3d_compress_u8(data, static_cast<int>(m.chunks[0]), static_cast<int>(m.chunks[1]),
                                   static_cast<int>(m.chunks[2]), m.delta3d_preset.c_str());
    if (m.compression == "zlib") {
        uLongf size = compressBound(static_cast<uLong>(n));
        std::vector<std::uint8_t> out(size);
        if (compress2(out.data(), &size, data, static_cast<uLong>(n), m.zlib_level) != Z_OK) throw Error("zarr: zlib failed");
        out.resize(size);
        return out;
    }
    return {};   // raw: stored as it is
}

void decode(const ZarrArrayMeta& m, const std::vector<std::uint8_t>& raw, std::uint8_t* out, const std::string& path) {
    const std::size_t n = m.chunk_bytes();
    if (m.compression == "vc-delta3d") {
        delta3d_decompress(raw.data(), raw.size(), out, n);
    } else if (m.compression == "zlib") {
        uLongf size = static_cast<uLongf>(n);
        if (uncompress(out, &size, raw.data(), static_cast<uLong>(raw.size())) != Z_OK || size != n)
            throw Error("zarr: bad zlib chunk " + path);
    } else {
        if (raw.size() != n) throw Error("zarr: chunk " + path + " has " + std::to_string(raw.size()) + " bytes, not " + std::to_string(n));
        std::memcpy(out, raw.data(), n);
    }
}

void check_meta(const ZarrArrayMeta& m, const std::string& path) {
    if (m.dtype != "|u1" && m.dtype != "<f4") throw Error("zarr: " + path + ": dtype " + m.dtype + " (uint8 and float32 arrays only)");
    if (m.compression != "none" && m.compression != "zlib" && m.compression != "vc-delta3d")
        throw Error("zarr: " + path + ": compression must be 'none', 'zlib' or 'vc-delta3d'");
    if (m.compression == "vc-delta3d") {
        if (m.dtype != "|u1") throw Error("vc-delta3d output is uint8 only");
        if (!delta3d_available()) throw Error("helirec was built without vc-delta3d (HELIREC_DELTA3D=OFF)");
    }
    for (int a = 0; a < 3; ++a)
        if (m.shape[a] < 1 || m.chunks[a] < 1) throw Error("zarr: " + path + ": empty shape or chunks");
    if (m.separator != '/' && m.separator != '.') throw Error("zarr: " + path + ": dimension separator must be '/' or '.'");
}

std::string zarray_text(const ZarrArrayMeta& m) {
    std::ostringstream o;
    o << "{\n  \"zarr_format\": 2,\n  \"shape\": [" << m.shape[0] << ", " << m.shape[1] << ", " << m.shape[2] << "],\n"
      << "  \"chunks\": [" << m.chunks[0] << ", " << m.chunks[1] << ", " << m.chunks[2] << "],\n"
      << "  \"dtype\": " << json_string(m.dtype) << ",\n  \"compressor\": ";
    if (m.compression == "zlib") o << "{\"id\": \"zlib\", \"level\": " << m.zlib_level << "}";
    else if (m.compression == "vc-delta3d") o << "{\"id\": \"vc-delta3d\", \"quant\": 1}";
    else o << "null";
    o << ",\n  \"fill_value\": 0,\n  \"order\": \"C\",\n  \"filters\": null,\n  \"dimension_separator\": \"" << m.separator << "\"\n}\n";
    return o.str();
}

// ------------------------------------------------------------------------------------------ workers
// A fixed set of threads running tasks; tasks belong to groups that can be waited for (an exception of a task is
// rethrown by the wait of its group).
class Pool {
  public:
    struct Group {
        std::mutex m;
        std::condition_variable cv;
        std::size_t left = 0;
        std::exception_ptr error;
    };
    explicit Pool(int n) {
        for (int i = 0; i < std::max(1, n); ++i) threads_.emplace_back([this] { run(); });
    }
    ~Pool() {
        {
            std::lock_guard<std::mutex> lock(m_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& t : threads_) t.join();
    }
    void submit(const std::shared_ptr<Group>& g, std::function<void()> f) {
        {
            std::lock_guard<std::mutex> lock(g->m);
            ++g->left;
        }
        {
            std::lock_guard<std::mutex> lock(m_);
            tasks_.emplace_back(g, std::move(f));
        }
        cv_.notify_one();
    }
    static void wait(Group& g) {
        std::unique_lock<std::mutex> lock(g.m);
        g.cv.wait(lock, [&] { return g.left == 0; });
        if (g.error) {
            const auto e = g.error;
            g.error = nullptr;
            std::rethrow_exception(e);
        }
    }

  private:
    void run() {
        for (;;) {
            std::pair<std::shared_ptr<Group>, std::function<void()>> task;
            {
                std::unique_lock<std::mutex> lock(m_);
                cv_.wait(lock, [&] { return stop_ || !tasks_.empty(); });
                if (tasks_.empty()) return;
                task = std::move(tasks_.front());
                tasks_.pop_front();
            }
            std::exception_ptr error;
            try {
                task.second();
            } catch (...) {
                error = std::current_exception();
            }
            task.second = nullptr;
            {
                std::lock_guard<std::mutex> lock(task.first->m);
                if (error && !task.first->error) task.first->error = error;
                --task.first->left;
            }
            task.first->cv.notify_all();
        }
    }
    std::mutex m_;
    std::condition_variable cv_;
    std::deque<std::pair<std::shared_ptr<Group>, std::function<void()>>> tasks_;
    std::vector<std::thread> threads_;
    bool stop_ = false;
};

}  // namespace

// ============================================================================================ ZarrArray
ZarrArray ZarrArray::create(const std::string& path, const ZarrArrayMeta& meta) {
    if (remote::is_remote(path)) throw Error("zarr: " + path + ": remote arrays are read only");
    check_meta(meta, path);
    fs::create_directories(path);
    write_text(path + "/.zarray", zarray_text(meta));
    return ZarrArray(path, meta);
}

ZarrArray::ZarrArray(std::string path, ZarrArrayMeta meta, std::string cache_dir)
    : path_(std::move(path)), cache_dir_(std::move(cache_dir)), meta_(std::move(meta)), remote_(remote::is_remote(path_)) {}

namespace {

// an object of a Zarr node, local or remote (false: absent); `metadata` objects are fetched again when cached
bool read_object(const std::string& location, const std::string& name, const std::string& cache_dir, bool metadata,
                 std::vector<std::uint8_t>& out) {
    const std::string at = location + "/" + name;
    return remote::is_remote(location) ? remote::fetch(at, out, cache_dir, metadata) : read_file(at, out);
}

std::string strip_slashes(std::string location) {
    while (location.size() > 1 && location.back() == '/') location.pop_back();
    return location;
}

ZarrIndex index_of(const Json& doc, const std::string& key, const std::string& where) {
    const Json* v = doc.find(key);
    if (!v || !v->is(Json::Array) || v->array.size() != 3) throw Error("zarr: " + where + ": \"" + key + "\" is not (z, y, x)");
    ZarrIndex out{};
    for (int a = 0; a < 3; ++a) {
        if (!v->array[a].is(Json::Number)) throw Error("zarr: " + where + ": \"" + key + "\" is not (z, y, x)");
        out[a] = static_cast<std::int64_t>(v->array[a].number);
    }
    return out;
}

ZarrArrayMeta parse_zarray(const std::vector<std::uint8_t>& raw, const std::string& where) {
    const Json doc = Json::parse(std::string(raw.begin(), raw.end()));
    ZarrArrayMeta m;
    m.shape = index_of(doc, "shape", where);
    m.chunks = index_of(doc, "chunks", where);
    if (const Json* v = doc.find("dtype"); v && v->is(Json::String)) m.dtype = v->string;
    if (m.dtype == "<u1" || m.dtype == ">u1") m.dtype = "|u1";
    m.separator = '.';
    if (const Json* v = doc.find("dimension_separator"); v && v->is(Json::String) && !v->string.empty()) m.separator = v->string[0];
    if (const Json* c = doc.find("compressor"); c && c->is(Json::Object)) {
        const Json* id = c->find("id");
        m.compression = id && id->is(Json::String) ? id->string : "?";
        if (const Json* level = c->find("level"); level && level->is(Json::Number)) m.zlib_level = static_cast<int>(level->number);
    }
    if (const Json* f = doc.find("filters"); f && !f->is(Json::Null) && !(f->is(Json::Array) && f->array.empty()))
        throw Error("zarr: " + where + ": filters are not supported");
    if (const Json* o = doc.find("order"); o && o->is(Json::String) && o->string != "C") throw Error("zarr: " + where + ": C order arrays only");
    if (const Json* f = doc.find("fill_value"); f && f->is(Json::Number) && f->number != 0)
        throw Error("zarr: " + where + ": the fill value must be 0");
    check_meta(m, where);
    return m;
}

}  // namespace

ZarrArray ZarrArray::open(const std::string& location, const std::string& cache_dir) {
    const std::string at = strip_slashes(location);
    std::vector<std::uint8_t> raw;
    if (!read_object(at, ".zarray", cache_dir, true, raw)) throw Error("zarr: no Zarr v2 array at " + at);
    return ZarrArray(at, parse_zarray(raw, at), cache_dir);
}

std::vector<OmeZarrLevel> ome_zarr_levels(const std::string& location, const std::string& cache_dir) {
    const std::string at = strip_slashes(location);
    std::vector<std::uint8_t> raw;
    if (read_object(at, ".zarray", cache_dir, true, raw)) return {OmeZarrLevel{0, at, {1, 1, 1}, {0, 0, 0}}};   // a plain array
    if (!read_object(at, ".zattrs", cache_dir, true, raw)) throw Error("zarr: no Zarr v2 array or OME-Zarr group at " + at);
    const Json doc = Json::parse(std::string(raw.begin(), raw.end()));
    const Json* ms = doc.find("multiscales");
    const Json* sets = ms && ms->is(Json::Array) && !ms->array.empty() ? ms->array[0].find("datasets") : nullptr;
    if (!sets || !sets->is(Json::Array) || sets->array.empty()) throw Error("zarr: " + at + " has no OME-Zarr multiscales");
    // Physical position of voxel i: scale * i + translation, in each dataset's coordinateTransformations. A level that
    // declares no translation is taken as block-reduced from the first (pyramids of n x n x n means seldom declare
    // the half-voxel shifts): its voxels are centred on their blocks, offset (scale - 1) / 2. A level that declares
    // no scale either halves the one before it.
    std::vector<OmeZarrLevel> levels;
    std::array<double, 3> s0{1, 1, 1}, t0{0, 0, 0};
    for (const Json& d : sets->array) {
        const Json* path = d.find("path");
        if (!path || !path->is(Json::String)) throw Error("zarr: " + at + ": a multiscales dataset has no path");
        const double halved = std::ldexp(1.0, static_cast<int>(levels.size()));
        std::array<double, 3> sc{halved * s0[0], halved * s0[1], halved * s0[2]}, tr{0, 0, 0};
        bool translated = false;
        if (const Json* cts = d.find("coordinateTransformations"); cts && cts->is(Json::Array))
            for (const Json& ct : cts->array) {
                const Json* type = ct.find("type");
                const bool scale = type && type->is(Json::String) && type->string == "scale";
                const Json* v = ct.find(scale ? "scale" : "translation");
                if (!v || !v->is(Json::Array)) continue;
                if (v->array.size() != 3) throw Error("zarr: " + at + ": only (z, y, x) images are supported");
                for (int a = 0; a < 3; ++a) (scale ? sc : tr)[a] = v->array[a].number;
                translated = translated || !scale;
            }
        OmeZarrLevel l;
        l.level = static_cast<int>(levels.size());
        l.location = at + "/" + path->string;
        if (levels.empty()) { s0 = sc; t0 = tr; }
        for (int a = 0; a < 3; ++a) {
            if (!(s0[a] > 0)) throw Error("zarr: " + at + ": bad scale");
            l.scale[a] = sc[a] / s0[a];
            l.offset[a] = translated ? (tr[a] - t0[a]) / s0[a] : (l.scale[a] - 1) / 2;
        }
        levels.push_back(std::move(l));
    }
    return levels;
}

std::string ZarrArray::chunk_path(std::int64_t iz, std::int64_t iy, std::int64_t ix) const {
    return path_ + "/" + std::to_string(iz) + meta_.separator + std::to_string(iy) + meta_.separator + std::to_string(ix);
}

bool ZarrArray::write_chunk(std::int64_t iz, std::int64_t iy, std::int64_t ix, const void* data, bool skip_empty) const {
    if (remote_) throw Error("zarr: " + path_ + ": remote arrays are read only");
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    const std::string p = chunk_path(iz, iy, ix);
    if (skip_empty && all_zero(bytes, meta_.chunk_bytes())) {
        std::error_code ignored;
        fs::remove(p, ignored);
        return false;
    }
    if (meta_.compression == "none") {
        write_file(p, bytes, meta_.chunk_bytes());
    } else {
        const auto enc = encode(meta_, bytes);
        write_file(p, enc.data(), enc.size());
    }
    return true;
}

bool ZarrArray::read_chunk(std::int64_t iz, std::int64_t iy, std::int64_t ix, void* out) const {
    std::vector<std::uint8_t> raw;
    const std::string p = chunk_path(iz, iy, ix);
    if (!(remote_ ? remote::fetch(p, raw, cache_dir_) : read_file(p, raw))) return false;
    decode(meta_, raw, static_cast<std::uint8_t*>(out), p);
    return true;
}

namespace {

// the chunks of `a` that the region [start, start + shape) reads, as (iz, iy, ix)
std::vector<ZarrIndex> region_chunks(const ZarrArrayMeta& m, const ZarrIndex& start, const ZarrIndex& shape) {
    std::vector<ZarrIndex> out;
    for (int a = 0; a < 3; ++a)
        if (start[a] < 0 || shape[a] < 0 || start[a] + shape[a] > m.shape[a]) throw Error("zarr: region outside the array");
    if (!shape[0] || !shape[1] || !shape[2]) return out;
    for (std::int64_t iz = start[0] / m.chunks[0]; iz <= (start[0] + shape[0] - 1) / m.chunks[0]; ++iz)
        for (std::int64_t iy = start[1] / m.chunks[1]; iy <= (start[1] + shape[1] - 1) / m.chunks[1]; ++iy)
            for (std::int64_t ix = start[2] / m.chunks[2]; ix <= (start[2] + shape[2] - 1) / m.chunks[2]; ++ix)
                out.push_back({iz, iy, ix});
    return out;
}

// the part of a decoded chunk inside the region, into the region's array
void place_chunk(const ZarrArrayMeta& m, const ZarrIndex& c, const std::uint8_t* chunk, const ZarrIndex& start,
                 const ZarrIndex& shape, std::uint8_t* out) {
    const std::size_t item = m.itemsize();
    std::int64_t a[3], b[3];
    for (int d = 0; d < 3; ++d) {
        a[d] = std::max(start[d], c[d] * m.chunks[d]);
        b[d] = std::min(start[d] + shape[d], (c[d] + 1) * m.chunks[d]);
    }
    for (std::int64_t z = a[0]; z < b[0]; ++z)
        for (std::int64_t y = a[1]; y < b[1]; ++y)
            std::memcpy(out + (((z - start[0]) * shape[1] + (y - start[1])) * shape[2] + (a[2] - start[2])) * item,
                        chunk + (((z - c[0] * m.chunks[0]) * m.chunks[1] + (y - c[1] * m.chunks[1])) * m.chunks[2] +
                                 (a[2] - c[2] * m.chunks[2])) * item,
                        static_cast<std::size_t>(b[2] - a[2]) * item);
}

}  // namespace

void ZarrArray::read(const ZarrIndex& start, const ZarrIndex& shape, void* out) const {
    const auto chunks = region_chunks(meta_, start, shape);
    auto* o = static_cast<std::uint8_t*>(out);
    const std::size_t total = static_cast<std::size_t>(shape[0]) * shape[1] * shape[2] * meta_.itemsize();
    std::string error;
#pragma omp parallel
    {
        std::vector<std::uint8_t> chunk(meta_.chunk_bytes());
#pragma omp for schedule(static)
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>((total + (1u << 22) - 1) >> 22); ++i)   // (first touch)
            std::memset(o + (static_cast<std::size_t>(i) << 22), 0, std::min<std::size_t>(std::size_t(1) << 22, total - (static_cast<std::size_t>(i) << 22)));
#pragma omp for schedule(dynamic, 1)
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(chunks.size()); ++i) {
            try {
                const auto& c = chunks[i];
                if (read_chunk(c[0], c[1], c[2], chunk.data())) place_chunk(meta_, c, chunk.data(), start, shape, o);
            } catch (const std::exception& e) {
#pragma omp critical(helirec_zarr_read)
                if (error.empty()) error = e.what();
            }
        }
    }
    if (!error.empty()) throw Error(error);
}

// ============================================================================================ OmeZarrWriter
struct OmeZarrWriter::Impl {
    std::string path;
    OmeZarrOptions opt;
    ZarrIndex shape, chunks;
    std::vector<ZarrIndex> level_shapes;
    std::vector<ZarrArray> arrays;
    std::size_t item = 1, chunk_bytes = 0;
    std::unique_ptr<Pool> pool;
    std::atomic<std::int64_t> written{0}, skipped{0};

    std::mutex m;   // the producer's side: block / write / close
    bool closed = false;
    // a z-chunk of level 0 being filled: only the chunks its blocks touch are held
    struct Slab {
        std::vector<char> filled;
        std::int64_t left = 0;
        std::unordered_map<std::int64_t, std::uint8_t*> chunks;   // iy * grid x + ix -> chunk_bytes, 0 where nothing landed
    };
    std::map<std::int64_t, Slab> open;
    std::vector<char> stored;               // per z-chunk: its slices are all in and its chunks went to the workers
    std::shared_ptr<Pool::Group> storing;   // the slab whose chunks are being stored
    // chunk buffers, used in turn (all 0 when free)
    std::mutex bm;
    std::vector<std::uint8_t*> free_buffers;

    ~Impl() {
        pool.reset();   // (running tasks hold buffers)
        for (auto& [iz, s] : open)
            for (auto& [k, b] : s.chunks) delete[] b;
        for (auto* b : free_buffers) delete[] b;
    }
    std::uint8_t* take() {
        {
            std::lock_guard<std::mutex> lock(bm);
            if (!free_buffers.empty()) {
                auto* b = free_buffers.back();
                free_buffers.pop_back();
                return b;
            }
        }
        return new std::uint8_t[chunk_bytes]();
    }
    void give(std::uint8_t* b, bool dirty) {
        if (dirty) std::memset(b, 0, chunk_bytes);
        std::lock_guard<std::mutex> lock(bm);
        free_buffers.push_back(b);
    }
    void count(bool stored) { (stored ? written : skipped).fetch_add(1, std::memory_order_relaxed); }

    Slab& slab(std::int64_t iz) {
        auto it = open.find(iz);
        if (it == open.end()) {
            if (stored[static_cast<std::size_t>(iz)])
                throw Error("OmeZarrWriter: slices of z-chunk " + std::to_string(iz) + " written twice");
            it = open.emplace(iz, Slab{}).first;
            const std::int64_t nz = std::min(chunks[0], shape[0] - iz * chunks[0]);
            it->second.filled.assign(static_cast<std::size_t>(nz), 0);
            it->second.left = nz;
        }
        return it->second;
    }
    // every slice of z-chunk iz is in (or the writer closes): its chunks go to the workers
    void store(std::int64_t iz) {
        // one slab is stored while the next fills: no more memory than two slabs' chunks
        if (storing) Pool::wait(*storing);
        Slab s = std::move(open.at(iz));
        open.erase(iz);
        stored[static_cast<std::size_t>(iz)] = 1;
        const ZarrIndex grid = arrays[0].meta().grid();
        if (!opt.skip_empty)   // every chunk is stored, the ones nothing landed in as zeros
            for (std::int64_t k = 0; k < grid[1] * grid[2]; ++k)
                if (!s.chunks.count(k)) s.chunks[k] = take();
        skipped.fetch_add(grid[1] * grid[2] - static_cast<std::int64_t>(s.chunks.size()), std::memory_order_relaxed);
        storing = std::make_shared<Pool::Group>();
        for (const auto& [key, buffer] : s.chunks) {
            const std::int64_t k = key;
            std::uint8_t* b = buffer;
            pool->submit(storing, [this, iz, k, b, gx = grid[2]] {
                bool stored = true;
                try {
                    stored = arrays[0].write_chunk(iz, k / gx, k % gx, b, opt.skip_empty);
                } catch (...) {
                    give(b, true);
                    throw;
                }
                count(stored);
                give(b, stored);
            });
        }
    }
    void pyramid_level(int l, const std::function<void()>& step);
};

OmeZarrWriter::OmeZarrWriter(const std::string& path, const OmeZarrOptions& options) : impl_(std::make_unique<Impl>()) {
    auto& I = *impl_;
    I.path = path;
    I.opt = options;
    I.shape = options.shape;
    for (int a = 0; a < 3; ++a) {
        if (options.shape[a] < 1) throw Error("OmeZarrWriter: empty shape");
        if (options.chunks[a] < 1) throw Error("OmeZarrWriter: chunks must be >= 1");
        I.chunks[a] = std::min(options.chunks[a], options.shape[a]);
    }
    ZarrArrayMeta meta;
    meta.shape = I.shape;
    meta.chunks = I.chunks;
    meta.dtype = options.dtype;
    meta.compression = options.compression;
    meta.zlib_level = options.zlib_level;
    meta.delta3d_preset = options.delta3d_preset;
    check_meta(meta, path);
    if (fs::exists(path)) {
        if (!options.overwrite) throw Error(path + " exists");
        fs::remove_all(path);
    }
    I.item = meta.itemsize();
    I.chunk_bytes = meta.chunk_bytes();
    fs::create_directories(path);
    write_text(path + "/.zgroup", "{\n  \"zarr_format\": 2\n}\n");
    const int levels = std::max(1, options.levels);
    I.level_shapes.push_back(I.shape);
    for (int l = 1; l < levels; ++l) {
        const auto& p = I.level_shapes.back();
        I.level_shapes.push_back({(p[0] + 1) / 2, (p[1] + 1) / 2, (p[2] + 1) / 2});
    }
    for (int l = 0; l < levels; ++l) {
        meta.shape = I.level_shapes[l];
        I.arrays.push_back(ZarrArray::create(path + "/" + std::to_string(l), meta));
    }
    // OME-NGFF 0.4 multiscales, then the caller's attributes
    std::string name = options.name;
    if (name.empty()) {
        std::string p = path;
        while (p.size() > 1 && p.back() == '/') p.pop_back();
        name = fs::path(p).filename().string();
    }
    std::ostringstream o;
    o << "{\n  \"multiscales\": [\n    {\n      \"version\": \"0.4\",\n      \"name\": " << json_string(name) << ",\n      \"axes\": [";
    for (int a = 0; a < 3; ++a)
        o << (a ? ", " : "") << "{\"name\": \"" << "zyx"[a] << "\", \"type\": \"space\", \"unit\": \"micrometer\"}";
    o << "],\n      \"datasets\": [";
    for (int l = 0; l < levels; ++l) {
        const double f = std::ldexp(1.0, l), s = options.voxel_size_um * f;
        o << (l ? "," : "") << "\n        {\"path\": \"" << l << "\", \"coordinateTransformations\": [{\"type\": \"scale\", \"scale\": ["
          << json_number(s) << ", " << json_number(s) << ", " << json_number(s) << "]}, {\"type\": \"translation\", \"translation\": [";
        for (int a = 0; a < 3; ++a)
            o << (a ? ", " : "") << json_number((static_cast<double>(options.origin[a]) + (f - 1) / 2) * options.voxel_size_um);
        o << "]}]}";
    }
    o << "\n      ],\n      \"type\": \"mean\"\n    }\n  ]";
    std::string extra = options.attrs_json;
    const std::size_t a = extra.find_first_not_of(" \t\r\n"), b = extra.find_last_not_of(" \t\r\n");
    if (a != std::string::npos) {
        if (extra[a] != '{' || extra[b] != '}') throw Error("OmeZarrWriter: attrs_json must be a JSON object");
        extra = extra.substr(a + 1, b - a - 1);
        if (extra.find_first_not_of(" \t\r\n") != std::string::npos) o << ",\n  " << extra;
    }
    o << "\n}\n";
    write_text(path + "/.zattrs", o.str());
    I.stored.assign(static_cast<std::size_t>(I.arrays[0].meta().grid()[0]), 0);
    I.pool = std::make_unique<Pool>(options.threads);
}

OmeZarrWriter::~OmeZarrWriter() = default;

const std::string& OmeZarrWriter::path() const { return impl_->path; }
const std::string& OmeZarrWriter::dtype() const { return impl_->opt.dtype; }
const ZarrIndex& OmeZarrWriter::shape() const { return impl_->shape; }
const ZarrIndex& OmeZarrWriter::chunks() const { return impl_->chunks; }
int OmeZarrWriter::levels() const { return static_cast<int>(impl_->level_shapes.size()); }
const std::vector<ZarrIndex>& OmeZarrWriter::level_shapes() const { return impl_->level_shapes; }
std::int64_t OmeZarrWriter::n_written() const { return impl_->written.load(); }
std::int64_t OmeZarrWriter::n_skipped() const { return impl_->skipped.load(); }

void OmeZarrWriter::block(std::int64_t z, int x0, int y0, int w, int h, const void* rows) {
    auto& I = *impl_;
    std::lock_guard<std::mutex> lock(I.m);
    if (I.closed) throw Error("OmeZarrWriter: closed");
    const std::int64_t cz = I.chunks[0], cy = I.chunks[1], cx = I.chunks[2];
    if (z < 0 || z >= I.shape[0]) throw Error("OmeZarrWriter: slice " + std::to_string(z) + " outside the volume");
    const bool empty = w <= 0 || h <= 0;
    if (!empty && (x0 < 0 || y0 < 0 || x0 + static_cast<std::int64_t>(w) > I.shape[2] || y0 + static_cast<std::int64_t>(h) > I.shape[1]))
        throw Error("OmeZarrWriter: block outside the slice");
    const std::int64_t iz = z / cz, oz = z % cz;
    Impl::Slab& s = I.slab(iz);
    if (s.filled[oz]) throw Error("OmeZarrWriter: slice " + std::to_string(z) + " written twice");
    if (!empty) {
        const std::int64_t gx = I.arrays[0].meta().grid()[2];
        const std::int64_t iy0 = y0 / cy, iy1 = (y0 + h - 1) / cy, ix0 = x0 / cx, ix1 = (x0 + w - 1) / cx;
        for (std::int64_t iy = iy0; iy <= iy1; ++iy)
            for (std::int64_t ix = ix0; ix <= ix1; ++ix) {
                auto& b = s.chunks[iy * gx + ix];
                if (!b) b = I.take();
            }
        const auto* src = static_cast<const std::uint8_t*>(rows);
        const std::size_t item = I.item;
        // (one row of the slice at a time across its chunks; rows of different chunk rows on several threads)
#pragma omp parallel for schedule(static) if (static_cast<std::int64_t>(w) * h > (1 << 20))
        for (std::int64_t y = y0; y < y0 + h; ++y) {
            const std::int64_t iy = y / cy;
            for (std::int64_t ix = ix0; ix <= ix1; ++ix) {
                const std::int64_t xa = std::max<std::int64_t>(x0, ix * cx), xb = std::min<std::int64_t>(x0 + w, (ix + 1) * cx);
                std::memcpy(s.chunks.find(iy * gx + ix)->second + ((oz * cy + (y - iy * cy)) * cx + (xa - ix * cx)) * item,
                            src + ((y - y0) * w + (xa - x0)) * item, static_cast<std::size_t>(xb - xa) * item);
            }
        }
    }
    s.filled[oz] = 1;
    if (--s.left == 0) I.store(iz);
}

void OmeZarrWriter::write(std::int64_t z0, std::int64_t n, const void* data) {
    const auto& I = *impl_;
    if (z0 < 0 || n < 0 || z0 + n > I.shape[0]) throw Error("OmeZarrWriter: slices outside the volume");
    const std::size_t plane = static_cast<std::size_t>(I.shape[1]) * I.shape[2] * I.item;
    for (std::int64_t k = 0; k < n; ++k)
        block(z0 + k, 0, 0, static_cast<int>(I.shape[2]), static_cast<int>(I.shape[1]), static_cast<const std::uint8_t*>(data) + k * plane);
}

// Level l from level l - 1: each voxel the mean of its 2 x 2 x 2 voxels below (the last one of an odd axis repeated),
// integers rounded half to even. One task per chunk row of each z-chunk; chunks whose sources are not stored are not.
void OmeZarrWriter::Impl::pyramid_level(int l, const std::function<void()>& step) {
    const ZarrArray& src = arrays[l - 1];
    const ZarrArray& dst = arrays[l];
    const ZarrIndex S = level_shapes[l - 1], D = level_shapes[l], grid = dst.meta().grid(), c = chunks;
    const bool u8 = opt.dtype == "|u1";
    for (std::int64_t iz = 0; iz < grid[0]; ++iz) {
        auto group = std::make_shared<Pool::Group>();
        for (std::int64_t iy = 0; iy < grid[1]; ++iy)
            pool->submit(group, [&, iz, iy] {
                std::vector<std::uint8_t> below, chunk(chunk_bytes), tmp(chunk_bytes);
                for (std::int64_t ix = 0; ix < grid[2]; ++ix) {
                    // the chunk's voxels [a, b) of level l, from [2a, min(2b, S)) below
                    const std::int64_t a[3] = {iz * c[0], iy * c[1], ix * c[2]};
                    std::int64_t n[3], sn[3];
                    const std::int64_t idx[3] = {iz, iy, ix};
                    for (int d = 0; d < 3; ++d) {
                        n[d] = std::min(c[d], D[d] - idx[d] * c[d]);
                        sn[d] = std::min(2 * n[d], S[d] - 2 * a[d]);
                    }
                    const ZarrIndex start{2 * a[0], 2 * a[1], 2 * a[2]}, shape{sn[0], sn[1], sn[2]};
                    below.assign(static_cast<std::size_t>(sn[0]) * sn[1] * sn[2] * item, 0);
                    bool any = false;
                    for (const auto& sc : region_chunks(src.meta(), start, shape))
                        if (src.read_chunk(sc[0], sc[1], sc[2], tmp.data())) {
                            place_chunk(src.meta(), sc, tmp.data(), start, shape, below.data());
                            any = true;
                        }
                    if (!any && opt.skip_empty) {
                        std::error_code ignored;
                        fs::remove(dst.chunk_path(iz, iy, ix), ignored);
                        count(false);
                        continue;
                    }
                    std::fill(chunk.begin(), chunk.end(), std::uint8_t(0));
                    const auto* bf = reinterpret_cast<const float*>(below.data());
                    auto* cf = reinterpret_cast<float*>(chunk.data());
                    for (std::int64_t z = 0; z < n[0]; ++z)
                        for (std::int64_t y = 0; y < n[1]; ++y)
                            for (std::int64_t x = 0; x < n[2]; ++x) {
                                double sum = 0;
                                for (int k = 0; k < 8; ++k) {
                                    const std::int64_t zz = std::min(2 * z + (k >> 2), sn[0] - 1), yy = std::min(2 * y + ((k >> 1) & 1), sn[1] - 1),
                                                       xx = std::min(2 * x + (k & 1), sn[2] - 1);
                                    const std::size_t i = static_cast<std::size_t>((zz * sn[1] + yy) * sn[2] + xx);
                                    sum += u8 ? static_cast<double>(below[i]) : static_cast<double>(bf[i]);
                                }
                                const std::size_t o = static_cast<std::size_t>((z * c[1] + y) * c[2] + x);
                                if (u8) chunk[o] = static_cast<std::uint8_t>(std::nearbyint(sum / 8));
                                else cf[o] = static_cast<float>(sum / 8);
                            }
                    count(dst.write_chunk(iz, iy, ix, chunk.data(), opt.skip_empty));
                }
            });
        Pool::wait(*group);
        if (step) step();
    }
}

void OmeZarrWriter::close(const std::function<std::function<void()>(int, std::int64_t)>& progress) {
    auto& I = *impl_;
    std::lock_guard<std::mutex> lock(I.m);
    if (I.closed) return;
    while (!I.open.empty()) I.store(I.open.begin()->first);   // (slices never written are 0)
    if (I.storing) Pool::wait(*I.storing);
    I.storing.reset();
    I.closed = true;
    for (int l = 1; l < levels(); ++l)
        I.pyramid_level(l, progress ? progress(l, I.arrays[l].meta().grid()[0]) : std::function<void()>());
    {
        std::lock_guard<std::mutex> buffers(I.bm);
        for (auto* b : I.free_buffers) delete[] b;
        I.free_buffers.clear();
    }
}

void OmeZarrWriter::read(int level, std::int64_t z0, std::int64_t z1, void* out) const {
    const auto& I = *impl_;
    if (level < 0 || level >= levels()) throw Error("OmeZarrWriter: no level " + std::to_string(level));
    const auto& s = I.level_shapes[level];
    I.arrays[level].read({z0, 0, 0}, {z1 - z0, s[1], s[2]}, out);
}

}  // namespace helirec
