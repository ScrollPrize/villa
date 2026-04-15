// vc_zarr_prefetch — pre-populate the local H.265 disk cache that vc3d uses
// when streaming a remote zarr volume.
//
// Behaviour matches the live vc3d path for any source format:
//   - canonical zarr v3 sharded with 128^3 inner H.265 chunks → byte-passthrough
//     of every inner chunk into the local sharded zarr (no decode/re-encode).
//   - anything else (zarr v2, raw, non-128^3 chunks, alternate codecs) → drive
//     BlockPipeline end-to-end so chunks are downloaded, decoded, rechunked
//     to canonical 128^3, H.265-encoded, and written to the local zarr —
//     exactly what vc3d does on a cache miss.
//
// --qp / --air-clamp / --bit-shift only affect the transcode path. Canonical
// passthrough keeps the source's encoding bit-for-bit.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include <boost/program_options.hpp>

#include "vc/core/cache/BlockPipeline.hpp"
#include "vc/core/cache/HttpMetadataFetcher.hpp"
#include "vc/core/cache/VolumeSource.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VcDataset.hpp"
#include "vc/core/util/RemoteUrl.hpp"

#include <utils/video_codec.hpp>
#include <utils/zarr.hpp>

namespace po = boost::program_options;
namespace fs = std::filesystem;

namespace {

// Slash-trimmed concat: "https://host/foo" + "0/zarr.json".
std::string urlJoin(const std::string& base, const std::string& tail) {
    if (base.empty()) return tail;
    if (base.back() == '/') return base + tail;
    return base + "/" + tail;
}

// Per-level remote metadata. v3 → from zarr.json; v2 → from .zarray.
// Returns empty optional if neither exists at the level.
std::optional<utils::ZarrMetadata> fetchLevelMetadata(
    const std::string& baseUrl, int level, const vc::cache::HttpAuth& auth)
{
    const std::string lvl = std::to_string(level);
    auto v3 = vc::cache::httpGetString(urlJoin(baseUrl, lvl + "/zarr.json"), auth);
    if (!v3.empty()) return utils::detail::parse_zarr_json(v3);
    auto v2 = vc::cache::httpGetString(urlJoin(baseUrl, lvl + "/.zarray"), auth);
    if (!v2.empty()) return utils::detail::parse_zarray(v2);
    return std::nullopt;
}

// Build the per-level LevelMeta vector HttpSource expects, from VcDataset.
std::vector<vc::cache::FileSystemSource::LevelMeta> levelMetasFromVolume(
    const Volume& vol, int numLevels)
{
    std::vector<vc::cache::FileSystemSource::LevelMeta> out;
    out.reserve(numLevels);
    for (int lvl = 0; lvl < numLevels; ++lvl) {
        auto* ds = vol.zarrDataset(lvl);
        if (!ds) throw std::runtime_error("missing zarrDataset for level " + std::to_string(lvl));
        const auto& shape = ds->shape();
        const auto& chunks = ds->defaultChunkShape();
        vc::cache::FileSystemSource::LevelMeta lm;
        lm.shape = {int(shape[0]), int(shape[1]), int(shape[2])};
        lm.chunkShape = {int(chunks[0]), int(chunks[1]), int(chunks[2])};
        out.push_back(lm);
    }
    return out;
}

// Round-up integer division.
int divUp(int a, int b) { return (a + b - 1) / b; }

// Open or create the local canonical disk zarr at <volumePath>/<level>/.
// Mirrors Volume::createTieredCache() so vc3d sees an identical layout.
std::shared_ptr<utils::ZarrArray> openOrCreateDiskLevel(
    const fs::path& volumePath, int level, std::array<int, 3> sourceShape)
{
    auto lvlPath = volumePath / std::to_string(level);
    if (fs::exists(lvlPath / "zarr.json")) {
        return std::make_shared<utils::ZarrArray>(utils::ZarrArray::open(lvlPath));
    }
    auto pad128 = [](int v) -> std::size_t {
        return std::size_t((v + 127) / 128 * 128);
    };
    utils::ZarrMetadata meta;
    meta.version = utils::ZarrVersion::v3;
    meta.node_type = "array";
    meta.shape = {pad128(sourceShape[0]), pad128(sourceShape[1]), pad128(sourceShape[2])};
    meta.chunks = {1024, 1024, 1024};
    meta.dtype = utils::ZarrDtype::uint8;
    meta.fill_value = 0;
    meta.chunk_key_encoding = "default";
    utils::ShardConfig sc;
    sc.sub_chunks = {128, 128, 128};
    meta.shard_config = std::move(sc);
    return std::make_shared<utils::ZarrArray>(
        utils::ZarrArray::create(lvlPath, std::move(meta)));
}

// Canonical → canonical passthrough: source and local layout are identical
// (zarr v3 sharded, 1024^3 shards, 128^3 inner H.265 chunks, identical
// codec chain). The shard FILE is therefore byte-identical between source
// and local. Download each whole shard with one HTTP request and dump the
// bytes verbatim to disk. No decode, no re-encode, no per-chunk parsing.
//
// Worker pool of `jobs` threads pops shard keys from a shared queue; writes
// go atomically via temp+rename so a killed run leaves no half-written shard.
void prefetchCanonicalLevel(
    vc::cache::HttpSource& src,
    const std::filesystem::path& lvlPath,
    int level,
    int jobs)
{
    namespace fs = std::filesystem;
    const auto shardsPerAxis = src.shardsPerAxis(level);
    std::fprintf(stderr,
        "[prefetch] level %d  shardsPerAxis=%dx%dx%d  jobs=%d\n",
        level, shardsPerAxis[0], shardsPerAxis[1], shardsPerAxis[2], jobs);
    std::fflush(stderr);
    if (shardsPerAxis[0] <= 0 || shardsPerAxis[1] <= 0 || shardsPerAxis[2] <= 0) {
        std::fprintf(stderr, "[prefetch] level %d: invalid shardsPerAxis — bailing\n", level);
        return;
    }
    const std::uint64_t totalShards =
        std::uint64_t(shardsPerAxis[0])
        * std::uint64_t(shardsPerAxis[1])
        * std::uint64_t(shardsPerAxis[2]);

    // Single shared queue of shard coordinates.
    struct Shard { int sz, sy, sx; };
    std::vector<Shard> queue;
    queue.reserve(totalShards);
    for (int sz = 0; sz < shardsPerAxis[0]; ++sz)
    for (int sy = 0; sy < shardsPerAxis[1]; ++sy)
    for (int sx = 0; sx < shardsPerAxis[2]; ++sx)
        queue.push_back({sz, sy, sx});
    std::fprintf(stderr, "[prefetch] level %d  queue built (%zu shards)\n",
                 level, queue.size());
    std::fflush(stderr);

    std::atomic<std::size_t> nextIdx{0};
    std::atomic<std::uint64_t> done{0}, written{0}, missing{0}, skipped{0};
    std::atomic<std::uint64_t> bytesWritten{0};
    std::atomic<bool> magicChecked{false};
    std::mutex stderrMutex;
    const auto t0 = std::chrono::steady_clock::now();

    auto worker = [&]() {
        while (true) {
            std::size_t i = nextIdx.fetch_add(1, std::memory_order_relaxed);
            if (i >= queue.size()) return;
            auto [sz, sy, sx] = queue[i];

            // Local on-disk path matches zarr v3 default key encoding.
            fs::path shardPath = lvlPath / "c"
                / std::to_string(sz) / std::to_string(sy) / std::to_string(sx);
            if (fs::exists(shardPath)) {
                skipped.fetch_add(1, std::memory_order_relaxed);
                done.fetch_add(1, std::memory_order_relaxed);
                continue;
            }

            std::vector<std::uint8_t> bytes;
            try { bytes = src.fetchWholeShard(level, sz, sy, sx); }
            catch (const std::exception& e) {
                std::lock_guard lk(stderrMutex);
                std::fprintf(stderr,
                    "[prefetch] level %d shard (%d,%d,%d) fetch failed: %s\n",
                    level, sz, sy, sx, e.what());
                done.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            if (bytes.empty()) {
                missing.fetch_add(1, std::memory_order_relaxed);
                done.fetch_add(1, std::memory_order_relaxed);
                continue;
            }

            // Spot-check the first non-sentinel inner chunk's VC3D magic
            // on the very first downloaded shard. Sharded shards have a
            // 16-byte-per-entry index at offset 0 (index_location=start);
            // sentinel entries (~0,~0) mean the inner chunk is absent.
            if (!magicChecked.exchange(true, std::memory_order_acq_rel)) {
                constexpr std::size_t kEntries = 512;
                constexpr std::size_t kIndexBytes = kEntries * 16;
                bool sampled = false;
                if (bytes.size() >= kIndexBytes) {
                    auto rd64 = [&](std::size_t o) {
                        std::uint64_t v;
                        std::memcpy(&v, bytes.data() + o, 8);
                        return v;
                    };
                    for (std::size_t e = 0; e < kEntries; ++e) {
                        std::uint64_t off = rd64(e * 16);
                        std::uint64_t n = rd64(e * 16 + 8);
                        if (off == ~std::uint64_t(0) || n < 4) continue;
                        if (off + n > bytes.size()) break;
                        if (bytes[off + 0] != 'V' || bytes[off + 1] != 'C'
                            || bytes[off + 2] != '3' || bytes[off + 3] != 'D') {
                            throw std::runtime_error(
                                "level " + std::to_string(level) +
                                " shard inner chunk lacks VC3D magic — "
                                "source is not H.265");
                        }
                        sampled = true;
                        break;
                    }
                }
                if (!sampled) {
                    // Allow it through; this shard may be entirely sparse.
                    magicChecked.store(false, std::memory_order_release);
                }
            }

            fs::create_directories(shardPath.parent_path());
            fs::path tmpPath = shardPath;
            tmpPath += ".part";
            {
                std::ofstream out(tmpPath, std::ios::binary | std::ios::trunc);
                out.write(reinterpret_cast<const char*>(bytes.data()),
                          static_cast<std::streamsize>(bytes.size()));
                if (!out) {
                    std::lock_guard lk(stderrMutex);
                    std::fprintf(stderr,
                        "[prefetch] level %d shard (%d,%d,%d) write failed\n",
                        level, sz, sy, sx);
                    done.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
            }
            std::error_code ec;
            fs::rename(tmpPath, shardPath, ec);
            if (ec) {
                std::lock_guard lk(stderrMutex);
                std::fprintf(stderr,
                    "[prefetch] level %d shard (%d,%d,%d) rename failed: %s\n",
                    level, sz, sy, sx, ec.message().c_str());
                done.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            written.fetch_add(1, std::memory_order_relaxed);
            bytesWritten.fetch_add(bytes.size(), std::memory_order_relaxed);
            done.fetch_add(1, std::memory_order_relaxed);
        }
    };

    std::vector<std::jthread> pool;
    pool.reserve(jobs);
    for (int t = 0; t < jobs; ++t) pool.emplace_back(worker);

    auto fmtEta = [](std::uint64_t s) -> std::string {
        if (s == 0) return "0s";
        unsigned h = unsigned(s / 3600);
        unsigned m = unsigned((s % 3600) / 60);
        unsigned ss = unsigned(s % 60);
        char buf[32];
        if (h) std::snprintf(buf, sizeof(buf), "%uh%02um", h, m);
        else if (m) std::snprintf(buf, sizeof(buf), "%um%02us", m, ss);
        else std::snprintf(buf, sizeof(buf), "%us", ss);
        return buf;
    };
    auto report = [&](bool final) {
        auto d = done.load(), w = written.load();
        auto m = missing.load(), sk = skipped.load();
        auto bw = bytesWritten.load();
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - t0).count();
        double mbs = elapsed > 0.0 ? (double(bw) / (1024.0 * 1024.0)) / elapsed : 0.0;
        std::uint64_t remaining = totalShards - std::min<std::uint64_t>(d, totalShards);
        double etaSec = (w > 0 && d > sk)
            ? double(remaining) * (elapsed / double(d - sk)) : 0.0;
        std::lock_guard lk(stderrMutex);
        std::fprintf(stderr,
            "\r[prefetch] L%d passthrough  shards %lu/%lu  written=%lu cached=%lu empty=%lu  "
            "%.0fMB @ %.1fMB/s  ETA %s%s",
            level,
            static_cast<unsigned long>(d),
            static_cast<unsigned long>(totalShards),
            static_cast<unsigned long>(w),
            static_cast<unsigned long>(sk),
            static_cast<unsigned long>(m),
            double(bw) / (1024.0 * 1024.0),
            mbs,
            fmtEta(static_cast<std::uint64_t>(etaSec)).c_str(),
            final ? "\n" : "");
        std::fflush(stderr);
    };
    while (done.load(std::memory_order_relaxed) < totalShards) {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        report(false);
    }
    for (auto& t : pool) t.join();
    report(true);
}

// Non-canonical: drive the live BlockPipeline so chunks are transcoded into
// the canonical disk format. Enqueues every canonical 128^3 chunk for the
// level and waits for the three pools to drain.
void prefetchTranscodeLevel(
    vc::cache::BlockPipeline& pipe, int level, const std::array<int, 3>& sourceShape)
{
    const int nz = divUp(sourceShape[0], 128);
    const int ny = divUp(sourceShape[1], 128);
    const int nx = divUp(sourceShape[2], 128);

    std::vector<vc::cache::ChunkKey> keys;
    keys.reserve(std::size_t(nz) * std::size_t(ny) * std::size_t(nx));
    for (int iz = 0; iz < nz; ++iz)
    for (int iy = 0; iy < ny; ++iy)
    for (int ix = 0; ix < nx; ++ix)
        keys.push_back({level, iz, iy, ix});

    const std::uint64_t total = keys.size();
    std::fprintf(stderr,
        "[prefetch] level %d  transcode  enqueueing %lu chunks\n",
        level, static_cast<unsigned long>(total));

    pipe.fetchInteractive(keys, level);

    const auto t0 = std::chrono::steady_clock::now();
    const auto initial = pipe.stats();
    const std::uint64_t initialWrites = initial.diskWrites;
    const std::uint64_t initialBytes = initial.diskBytes;
    auto lastReport = t0;
    auto fmtEta = [](std::uint64_t s) -> std::string {
        if (s == 0) return "0s";
        unsigned h = unsigned(s / 3600);
        unsigned m = unsigned((s % 3600) / 60);
        unsigned ss = unsigned(s % 60);
        char buf[32];
        if (h) std::snprintf(buf, sizeof(buf), "%uh%02um", h, m);
        else if (m) std::snprintf(buf, sizeof(buf), "%um%02us", m, ss);
        else std::snprintf(buf, sizeof(buf), "%us", ss);
        return buf;
    };
    while (true) {
        auto s = pipe.stats();
        if (s.ioPending == 0) break;
        auto now = std::chrono::steady_clock::now();
        if (now - lastReport > std::chrono::seconds(2)) {
            const std::uint64_t doneCount = s.diskWrites - initialWrites;
            const std::uint64_t doneBytes = s.diskBytes - initialBytes;
            const double elapsed = std::chrono::duration<double>(now - t0).count();
            const double mbs = elapsed > 0.0
                ? (double(doneBytes) / (1024.0 * 1024.0)) / elapsed : 0.0;
            const std::uint64_t remaining = total > doneCount ? total - doneCount : 0;
            const double etaSec = doneCount > 0
                ? double(remaining) * (elapsed / double(doneCount)) : 0.0;
            std::fprintf(stderr,
                "\r[prefetch] L%d transcode  chunks %lu/%lu  pending dl=%zu enc=%zu load=%zu  "
                "%.0fMB @ %.1fMB/s  ETA %s",
                level,
                static_cast<unsigned long>(doneCount),
                static_cast<unsigned long>(total),
                s.downloadPending, s.encodePending, s.loadPending,
                double(doneBytes) / (1024.0 * 1024.0),
                mbs,
                fmtEta(static_cast<std::uint64_t>(etaSec)).c_str());
            std::fflush(stderr);
            lastReport = now;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    auto s = pipe.stats();
    std::fprintf(stderr,
        "\r[prefetch] L%d transcode  done  diskWrites=%lu  iceFetches=%lu\n",
        level,
        static_cast<unsigned long>(s.diskWrites - initialWrites),
        static_cast<unsigned long>(s.iceFetches));
}

}  // namespace

int main(int argc, char** argv) {
    std::string url;
    std::string cacheDirStr;
    std::string levelsArg = "0";
    int qp = 36, airClamp = 0, bitShift = 0;
    int jobs = 0;

    po::options_description visible("Options");
    visible.add_options()
        ("help,h", "show help")
        ("cache-dir", po::value(&cacheDirStr),
            "local cache root (default: $HOME/.VC3D/remote_cache)")
        ("levels", po::value(&levelsArg)->default_value("0"),
            "comma-separated pyramid levels to prefetch in the given order, "
            "e.g. \"5,4,3,2,1,0\" prefetches coarsest first")
        ("qp", po::value(&qp)->default_value(36),
            "H.265 QP for transcoded chunks (0..51, lower=better)")
        ("air-clamp", po::value(&airClamp)->default_value(0),
            "voxels <= threshold snap to threshold pre-encode")
        ("bit-shift", po::value(&bitShift)->default_value(0),
            "right-shift voxels by N before encode (0..7)")
        ("jobs", po::value(&jobs)->default_value(0),
            "concurrent IO jobs (0 = auto)");

    po::options_description hidden;
    hidden.add_options()("url", po::value(&url));

    po::positional_options_description pos;
    pos.add("url", 1);

    po::options_description all;
    all.add(visible).add(hidden);

    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv)
                      .options(all).positional(pos).run(), vm);
        po::notify(vm);
    } catch (const std::exception& e) {
        std::cerr << "argument error: " << e.what() << "\n";
        return 2;
    }

    if (vm.count("help") || url.empty()) {
        std::cout <<
            "Usage: vc_zarr_prefetch <s3-or-https-url> "
            "[--cache-dir PATH] [--levels 5,4,3,2,1,0] [--qp 36] "
            "[--air-clamp 0] [--bit-shift 0] [--jobs N]\n\n"
            << visible << "\n";
        return url.empty() ? 2 : 0;
    }
    if (qp < 0 || qp > 51) { std::cerr << "qp out of range [0..51]\n"; return 2; }
    if (bitShift < 0 || bitShift > 7) { std::cerr << "bit-shift out of range [0..7]\n"; return 2; }
    if (airClamp < 0 || airClamp > 255) { std::cerr << "air-clamp out of range [0..255]\n"; return 2; }

    // Parse comma-separated level list (preserves order; rejects duplicates).
    std::vector<int> levelOrder;
    {
        std::string tok;
        std::unordered_set<int> seen;
        for (std::size_t i = 0; i <= levelsArg.size(); ++i) {
            char c = i < levelsArg.size() ? levelsArg[i] : ',';
            if (c == ',') {
                if (tok.empty()) continue;
                int v;
                try { v = std::stoi(tok); }
                catch (...) {
                    std::cerr << "bad level token: '" << tok << "'\n"; return 2;
                }
                if (v < 0) { std::cerr << "level must be >= 0\n"; return 2; }
                if (seen.insert(v).second) levelOrder.push_back(v);
                tok.clear();
            } else if (c != ' ' && c != '\t') {
                tok += c;
            }
        }
        if (levelOrder.empty()) {
            std::cerr << "--levels must list at least one level\n"; return 2;
        }
    }

    fs::path cacheDir = cacheDirStr.empty() ? fs::path{} : fs::path(cacheDirStr);

    auto vol = Volume::NewFromUrl(url, cacheDir);
    if (!vol) { std::cerr << "failed to open remote volume\n"; return 1; }
    if (jobs > 0) vol->setIOThreads(jobs);
    utils::VideoCodecParams enc;
    enc.qp = qp; enc.air_clamp = airClamp; enc.shift_n = bitShift;
    vol->setEncodeParams(enc);

    const int availableLevels = static_cast<int>(vol->numScales());
    // Drop any requested levels the source doesn't have; warn instead of fail.
    {
        std::vector<int> kept;
        kept.reserve(levelOrder.size());
        for (int v : levelOrder) {
            if (v >= availableLevels) {
                std::fprintf(stderr,
                    "[prefetch] level %d not present (source has %d) — skipping\n",
                    v, availableLevels);
            } else {
                kept.push_back(v);
            }
        }
        levelOrder = std::move(kept);
        if (levelOrder.empty()) {
            std::cerr << "no requested levels are present in the source\n"; return 1;
        }
    }

    // Resolve s3:// → https:// once for raw metadata fetches.
    auto resolved = vc::resolveRemoteUrl(url);
    const std::string baseUrl = resolved.httpsUrl;
    vc::cache::HttpAuth auth = vol->remoteAuth();

    // Per-level info indexed by the actual pyramid level index. Only entries
    // for levels we'll touch get populated.
    const std::size_t nLevels = std::size_t(availableLevels);
    std::vector<bool> levelCanonical(nLevels, false);
    std::vector<std::array<int, 3>> levelShapes(nLevels, {0, 0, 0});
    std::vector<std::array<int, 3>> levelShardShapes(nLevels, {0, 0, 0});
    for (int lvl : levelOrder) {
        auto meta = fetchLevelMetadata(baseUrl, lvl, auth);
        if (!meta) {
            std::cerr << "no metadata at level " << lvl << " — stopping\n";
            return 1;
        }
        levelCanonical[lvl] = utils::is_canonical_vc3d(*meta);
        levelShapes[lvl] = {
            int(meta->shape[0]), int(meta->shape[1]), int(meta->shape[2])};
        if (meta->chunks.size() >= 3) {
            levelShardShapes[lvl] = {
                int(meta->chunks[0]), int(meta->chunks[1]), int(meta->chunks[2])};
        }
        std::fprintf(stderr,
            "[prefetch] level %d  shape=%dx%dx%d  shard=%dx%dx%d  %s\n",
            lvl, levelShapes[lvl][0], levelShapes[lvl][1], levelShapes[lvl][2],
            levelShardShapes[lvl][0], levelShardShapes[lvl][1], levelShardShapes[lvl][2],
            levelCanonical[lvl] ? "canonical → passthrough" : "non-canonical → transcode");
    }

    // Build a standalone HttpSource for the passthrough path. Mirrors the
    // construction inside Volume::createTieredCache.
    auto httpLevels = levelMetasFromVolume(*vol, availableLevels);
    vc::cache::HttpSource passthroughSource(
        vol->remoteUrl(), vol->remoteDelimiter(), httpLevels, auth);
    const auto& volSc = vol->remoteShardConfig();
    std::fprintf(stderr,
        "[prefetch] volume.remoteShardConfig: enabled=%d shardShape=%dx%dx%d\n",
        int(volSc.enabled), volSc.shardShape[0], volSc.shardShape[1], volSc.shardShape[2]);
    if (volSc.enabled) {
        passthroughSource.setShardConfig(volSc);
    } else {
        // The Volume's metadata fetcher didn't propagate a shard config
        // for this volume — fall back to the shard shape we read from
        // the first canonical level's zarr.json.
        for (int lvl : levelOrder) {
            if (!levelCanonical[lvl]) continue;
            vc::cache::ShardConfig sc;
            sc.enabled = true;
            sc.shardShape = levelShardShapes[lvl];
            passthroughSource.setShardConfig(sc);
            std::fprintf(stderr,
                "[prefetch] applied fallback shardConfig %dx%dx%d from canonical level %d\n",
                sc.shardShape[0], sc.shardShape[1], sc.shardShape[2], lvl);
            break;
        }
    }

    // Lazily realise the BlockPipeline only when a transcode level needs it.
    vc::cache::BlockPipeline* pipe = nullptr;
    auto pipeFor = [&]() -> vc::cache::BlockPipeline& {
        if (!pipe) pipe = vol->tieredCache();
        if (!pipe) throw std::runtime_error("failed to create BlockPipeline");
        return *pipe;
    };

    const int passthroughJobs = jobs > 0 ? jobs : 16;
    for (int lvl : levelOrder) {
        if (levelCanonical[lvl]) {
            // openOrCreateDiskLevel writes zarr.json so vc3d can read the
            // cache later; we then ignore the returned ZarrArray and write
            // shards directly to the level path.
            (void)openOrCreateDiskLevel(vol->path(), lvl, levelShapes[lvl]);
            auto lvlPath = vol->path() / std::to_string(lvl);
            prefetchCanonicalLevel(passthroughSource, lvlPath, lvl, passthroughJobs);
        } else {
            prefetchTranscodeLevel(pipeFor(), lvl, levelShapes[lvl]);
        }
    }

    std::fprintf(stderr, "[prefetch] all levels complete\n");
    return 0;
}
