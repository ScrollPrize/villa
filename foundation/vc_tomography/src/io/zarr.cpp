// OmeZarrWriter (vc_tomography/zarr.hpp): the streaming assembly of level 0 from the blocks of slices is ours; the
// arrays, their chunk codecs, the pyramid levels and the OME metadata are vc_reader's (vcr::zarr::Writer,
// createPyramidLevels / writePyramidLevel, omeMultiscales).
#include "vc_tomography/zarr.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <filesystem>
#include <future>
#include <map>
#include <mutex>
#include <unordered_map>

#include "vc_reader/array.hpp"
#include "vc_reader/native/thread_pool.hpp"
#include "vc_reader/zarr_write.hpp"

namespace vc_tomography {

namespace fs = std::filesystem;
namespace vz = vcr::zarr;

struct OmeZarrWriter::Impl {
    std::string path;
    OmeZarrOptions opt;
    ZarrIndex shape, chunks, grid;           // level 0's; grid: chunks along each axis
    std::vector<ZarrIndex> level_shapes;
    std::vector<vz::Writer> arrays;          // per level
    vz::PyramidOptions pyramid;
    std::size_t item = 1, chunk_bytes = 0;
    std::unique_ptr<vcr::ThreadPool> pool;
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
    std::vector<char> stored;   // per z-chunk: its slices are all in and its chunks went to the workers
    std::future<void> storing;  // the slab whose chunks are being stored
    // chunk buffers, used in turn (all 0 when free)
    std::mutex bm;
    std::vector<std::uint8_t*> free_buffers;

    ~Impl() {
        if (storing.valid()) storing.wait();   // (its tasks hold buffers)
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

    Slab& slab(std::int64_t iz) {
        auto it = open.find(iz);
        if (it == open.end()) {
            if (stored[static_cast<std::size_t>(iz)]) throw Error("OmeZarrWriter: slices of z-chunk " + std::to_string(iz) + " written twice");
            it = open.emplace(iz, Slab{}).first;
            const std::int64_t nz = std::min(chunks[0], shape[0] - iz * chunks[0]);
            it->second.filled.assign(static_cast<std::size_t>(nz), 0);
            it->second.left = nz;
        }
        return it->second;
    }
    void finish_storing() {
        if (storing.valid()) storing.get();   // (rethrows a worker's error)
    }
    // every slice of z-chunk iz is in (or the writer closes): its chunks go to the workers, while the next slab fills
    // (no more memory than two slabs' chunks)
    void store(std::int64_t iz) {
        finish_storing();
        Slab s = std::move(open.at(iz));
        open.erase(iz);
        stored[static_cast<std::size_t>(iz)] = 1;
        if (!opt.skip_empty)   // every chunk is stored, the ones nothing landed in as zeros
            for (std::int64_t k = 0; k < grid[1] * grid[2]; ++k)
                if (!s.chunks.count(k)) s.chunks[k] = take();
        skipped.fetch_add(grid[1] * grid[2] - static_cast<std::int64_t>(s.chunks.size()), std::memory_order_relaxed);
        auto list = std::make_shared<std::vector<std::pair<std::int64_t, std::uint8_t*>>>(s.chunks.begin(), s.chunks.end());
        storing = std::async(std::launch::async, [this, iz, list] {
            std::exception_ptr error;
            std::mutex em;
            pool->parallel_for(list->size(), [&](std::size_t i) {
                const auto [k, b] = (*list)[i];
                bool kept = true;
                try {
                    const vcr::Index c{iz, k / grid[2], k % grid[2]};
                    if (opt.skip_empty) kept = arrays[0].writeChunkSkipEmpty(c, b, chunk_bytes);
                    else arrays[0].writeChunk(c, b, chunk_bytes);
                } catch (...) {
                    std::lock_guard<std::mutex> lock(em);
                    if (!error) error = std::current_exception();
                }
                (kept ? written : skipped).fetch_add(1, std::memory_order_relaxed);
                give(b, kept);
            });
            if (error) std::rethrow_exception(error);
        });
    }
};

namespace {

std::string compressor_id(const std::string& compression) {
    if (compression == "none" || compression == "zlib" || compression == "vc-delta3d") return compression;
    throw Error("OmeZarrWriter: compression must be none, zlib or vc-delta3d, not '" + compression + "'");
}

}  // namespace

OmeZarrWriter::OmeZarrWriter(const std::string& path, const OmeZarrOptions& options) : impl_(std::make_unique<Impl>()) {
    auto& I = *impl_;
    I.path = path;
    I.opt = options;
    I.shape = options.shape;
    for (int a = 0; a < 3; ++a) {
        if (options.shape[a] < 1) throw Error("OmeZarrWriter: empty shape");
        if (options.chunks[a] < 1) throw Error("OmeZarrWriter: chunks must be >= 1");
        I.chunks[a] = std::min(options.chunks[a], options.shape[a]);
        I.grid[a] = (I.shape[a] + I.chunks[a] - 1) / I.chunks[a];
    }
    if (options.dtype != "|u1" && options.dtype != "<f4") throw Error("OmeZarrWriter: dtype must be |u1 or <f4, not " + options.dtype);
    if (options.compression == "vc-delta3d" && options.dtype != "|u1") throw Error("OmeZarrWriter: vc-delta3d is for uint8 volumes");
    vz::CreateOptions level0;
    level0.shape = {I.shape[0], I.shape[1], I.shape[2]};
    level0.chunks = {I.chunks[0], I.chunks[1], I.chunks[2]};
    level0.dtype = options.dtype;
    level0.compressor = compressor_id(options.compression);
    level0.level = options.compression == "zlib" ? options.zlib_level : -1;
    level0.separator = "/";
    level0.delta3dPredictor = options.compression == "vc-delta3d" ? options.delta3d_preset : std::string();
    if (fs::exists(path)) {
        if (!options.overwrite) throw Error(path + " exists");
        fs::remove_all(path);
    }
    // (the attributes are written last: a reader finds the group only once its levels exist)
    I.arrays.push_back(vz::Writer::create(fs::path(path) / "0", level0));
    const int levels = std::max(1, options.levels);
    I.pyramid.levels = levels - 1;
    I.pyramid.threads = static_cast<std::size_t>(std::max(1, options.threads));
    I.pyramid.delta3dPredictor = level0.delta3dPredictor;
    if (levels > 1) vz::createPyramidLevels(path, level0, I.pyramid);
    for (int l = 0; l < levels; ++l) {
        if (l) I.arrays.push_back(vz::Writer::open(fs::path(path) / std::to_string(l)));
        const vcr::Index& s = I.arrays.back().meta().shape;
        I.level_shapes.push_back({s[0], s[1], s[2]});
    }
    // OME-NGFF 0.4 multiscales (voxels in micrometers, placed at the origin), then the caller's attributes
    std::string name = options.name;
    if (name.empty()) {
        std::string p = path;
        while (p.size() > 1 && p.back() == '/') p.pop_back();
        name = fs::path(p).filename().string();
    }
    vz::MultiscalesOptions ms;
    ms.name = name;
    ms.levels = levels - 1;
    ms.scale = {options.voxel_size_um, options.voxel_size_um, options.voxel_size_um};
    ms.unit = "micrometer";
    vz::Json multiscales = vz::omeMultiscales(ms);
    for (vz::Json& d : multiscales["datasets"])
        for (vz::Json& t : d["coordinateTransformations"])
            if (t.value("type", "") == "translation")
                for (int a = 0; a < 3; ++a) t["translation"][a] = t["translation"][a].get<double>() + static_cast<double>(options.origin[a]) * options.voxel_size_um;
    vz::Json attrs{{"multiscales", vz::Json::array({multiscales})}};
    if (options.attrs_json.find_first_not_of(" \t\r\n") != std::string::npos) {
        vz::Json extra;
        try {
            extra = vz::Json::parse(options.attrs_json);
        } catch (const std::exception& e) {
            throw Error(std::string("OmeZarrWriter: attrs_json: ") + e.what());
        }
        if (!extra.is_object()) throw Error("OmeZarrWriter: attrs_json must be a JSON object");
        for (const auto& [k, v] : extra.items()) attrs[k] = v;
    }
    vz::createGroup(path, attrs);
    I.item = I.arrays[0].meta().itemsize();
    I.chunk_bytes = I.arrays[0].meta().chunkBytes();
    I.stored.assign(static_cast<std::size_t>(I.grid[0]), 0);
    I.pool = std::make_unique<vcr::ThreadPool>(static_cast<std::size_t>(std::max(1, options.threads) - 1));   // (+ the caller)
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
        const std::int64_t gx = I.grid[2];
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

// Level l from level l - 1 (vcr::zarr::writePyramidLevel): each voxel the mean of the voxels of its 2 x 2 x 2 block
// that lie in the level below, integers rounded half to even; chunks that come out all 0 are not stored.
void OmeZarrWriter::close(const std::function<std::function<void()>(int, std::int64_t)>& progress) {
    auto& I = *impl_;
    std::lock_guard<std::mutex> lock(I.m);
    if (I.closed) return;
    while (!I.open.empty()) I.store(I.open.begin()->first);   // (slices never written are 0)
    I.finish_storing();
    I.closed = true;
    {
        std::lock_guard<std::mutex> buffers(I.bm);
        for (auto* b : I.free_buffers) delete[] b;
        I.free_buffers.clear();
    }
    for (int l = 1; l < levels(); ++l) {
        const vcr::Index& c = I.arrays[l].meta().chunks;
        const ZarrIndex& s = I.level_shapes[l];
        const std::int64_t gz = (s[0] + c[0] - 1) / c[0], per_slab = ((s[1] + c[1] - 1) / c[1]) * ((s[2] + c[2] - 1) / c[2]);
        const std::function<void()> step = progress ? progress(l, gz) : std::function<void()>();
        std::mutex pm;
        std::int64_t slabs = 0;
        // (a meter: one step per z-chunk's worth of chunks done, in whatever order the workers finish them)
        vz::writePyramidLevel(I.path, l, I.pyramid, std::nullopt, [&](std::size_t done, std::size_t) {
            if (!step) return;
            std::lock_guard<std::mutex> p(pm);
            while (slabs < gz && static_cast<std::int64_t>(done) >= (slabs + 1) * per_slab) ++slabs, step();
        });
        const std::int64_t kept = static_cast<std::int64_t>(I.arrays[l].storedChunks().size());
        I.written += kept;
        I.skipped += gz * per_slab - kept;
    }
}

void OmeZarrWriter::read(int level, std::int64_t z0, std::int64_t z1, void* out) const {
    auto& I = *impl_;
    if (level < 0 || level >= levels()) throw Error("OmeZarrWriter: no level " + std::to_string(level));
    const auto& s = I.level_shapes[level];
    if (z0 < 0 || z1 < z0 || z1 > s[0]) throw Error("OmeZarrWriter: slices outside the level");
    if (z1 == z0) return;
    I.arrays[static_cast<std::size_t>(level)].reader().read_many({vcr::Index{z0, 0, 0}}, {vcr::Index{z1 - z0, s[1], s[2]}},
                                                                 {static_cast<std::uint8_t*>(out)});
}

}  // namespace vc_tomography
