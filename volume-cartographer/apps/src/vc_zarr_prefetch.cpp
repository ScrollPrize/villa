// vc_zarr_prefetch — pre-populate the local c3d disk cache that vc3d uses
// when streaming a remote zarr volume.
//
// Behaviour matches the live vc3d path for any source format:
//   - canonical zarr v3 sharded with 256^3 inner c3d chunks / 4096^3 shards
//     → byte-passthrough of each whole shard into the local sharded zarr
//     (no decode/re-encode).
//   - anything else (zarr v2, raw, other chunk sizes, alternate codecs) →
//     drive BlockPipeline end-to-end so chunks are downloaded, decoded,
//     rechunked to canonical 256^3, c3d-encoded, and written to the local
//     zarr — exactly what vc3d does on a cache miss.
//
// --target-ratio only affects the transcode path. Canonical passthrough
// keeps the source's encoding bit-for-bit.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
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

#include <utils/http_fetch.hpp>

#include <boost/program_options.hpp>

#include "vc/core/cache/BlockPipeline.hpp"
#include "vc/core/cache/HttpMetadataFetcher.hpp"
#include "vc/core/cache/VolumeSource.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VcDataset.hpp"
#include "vc/core/util/RemoteUrl.hpp"

#include <utils/c3d_codec.hpp>
#include <utils/zarr.hpp>

namespace po = boost::program_options;
namespace fs = std::filesystem;

namespace {

std::atomic<bool> g_shutdown{false};

void handleSigint(int) {
    g_shutdown.store(true, std::memory_order_release);
    utils::HttpClient::abortAll();
}

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

// Open or create the local canonical c3d disk zarr at <volumePath>/<level>/.
// Mirrors Volume::createTieredCache() so vc3d sees an identical layout:
// 4096³ shards with 256³ inner C3DC chunks, sub_codec "c3d". If a
// zarr.json already exists, open it untouched — the caller is responsible
// for verifying its layout matches before writing shards.
std::shared_ptr<utils::ZarrArray> openOrCreateDiskLevel(
    const fs::path& volumePath, int level, std::array<int, 3> sourceShape)
{
    auto lvlPath = volumePath / std::to_string(level);
    if (fs::exists(lvlPath / "zarr.json")) {
        return std::make_shared<utils::ZarrArray>(utils::ZarrArray::open(lvlPath));
    }
    auto pad256 = [](int v) -> std::size_t {
        return std::size_t((v + 255) / 256 * 256);
    };
    utils::ZarrMetadata meta;
    meta.version = utils::ZarrVersion::v3;
    meta.node_type = "array";
    meta.shape = {pad256(sourceShape[0]), pad256(sourceShape[1]), pad256(sourceShape[2])};
    meta.chunks = {4096, 4096, 4096};
    meta.dtype = utils::ZarrDtype::uint8;
    meta.fill_value = 0;
    meta.chunk_key_encoding = "default";
    utils::ShardConfig sc;
    sc.sub_chunks = {256, 256, 256};
    utils::ZarrCodecConfig cc;
    cc.name = "c3d";
    sc.sub_codecs.push_back(cc);
    meta.shard_config = std::move(sc);
    return std::make_shared<utils::ZarrArray>(
        utils::ZarrArray::create(lvlPath, std::move(meta)));
}

// Canonical → canonical passthrough: source and local layout are identical
// (zarr v3 sharded, matching shard size + inner-chunk codec). The shard FILE
// is therefore byte-identical between source and local. Download each whole
// shard with one HTTP request and dump the bytes verbatim to disk. No decode,
// no re-encode, no per-chunk parsing.
//
// Every passthrough-eligible level is flattened into a single shard queue
// (coarsest level first), so the worker pool stays saturated across level
// boundaries instead of stalling whenever a shallow level has fewer shards
// than there are workers. Writes go atomically via temp+rename so a killed
// run leaves no half-written shard.
struct PassthroughLevel {
    int level;
    std::filesystem::path lvlPath;
};

void prefetchCanonicalLevels(
    vc::cache::HttpSource& src,
    const std::vector<PassthroughLevel>& levels,
    int jobs)
{
    namespace fs = std::filesystem;
    if (levels.empty()) return;

    struct Shard {
        int level;
        int sz, sy, sx;
        const std::filesystem::path* lvlPath;
    };
    std::vector<Shard> queue;
    std::vector<std::uint64_t> perLevelTotal;
    perLevelTotal.reserve(levels.size());

    for (const auto& lvl : levels) {
        const auto shardsPerAxis = src.shardsPerAxis(lvl.level);
        std::fprintf(stderr,
            "[prefetch] level %d  shardsPerAxis=%dx%dx%d  codec=c3d\n",
            lvl.level, shardsPerAxis[0], shardsPerAxis[1], shardsPerAxis[2]);
        if (shardsPerAxis[0] <= 0 || shardsPerAxis[1] <= 0 || shardsPerAxis[2] <= 0) {
            std::fprintf(stderr,
                "[prefetch] level %d: invalid shardsPerAxis — skipping\n", lvl.level);
            perLevelTotal.push_back(0);
            continue;
        }
        // Sweep leftover .part files so interrupted runs don't leak.
        std::error_code ec;
        fs::path cDir = lvl.lvlPath / "c";
        if (fs::exists(cDir, ec)) {
            std::uint64_t removed = 0;
            for (auto it = fs::recursive_directory_iterator(cDir, ec);
                 !ec && it != fs::recursive_directory_iterator(); it.increment(ec)) {
                if (it->is_regular_file(ec) && it->path().extension() == ".part") {
                    fs::remove(it->path(), ec);
                    if (!ec) ++removed;
                }
            }
            if (removed) {
                std::fprintf(stderr,
                    "[prefetch] level %d  cleaned %lu stale .part files\n",
                    lvl.level, static_cast<unsigned long>(removed));
            }
        }
        std::uint64_t count = 0;
        for (int sz = 0; sz < shardsPerAxis[0]; ++sz)
        for (int sy = 0; sy < shardsPerAxis[1]; ++sy)
        for (int sx = 0; sx < shardsPerAxis[2]; ++sx) {
            queue.push_back({lvl.level, sz, sy, sx, &lvl.lvlPath});
            ++count;
        }
        perLevelTotal.push_back(count);
    }

    const std::uint64_t totalShards = queue.size();
    std::fprintf(stderr,
        "[prefetch] passthrough queue built (%lu shards across %zu levels, jobs=%d)\n",
        static_cast<unsigned long>(totalShards), levels.size(), jobs);
    std::fflush(stderr);
    if (totalShards == 0) return;

    std::atomic<std::size_t> nextIdx{0};
    std::atomic<std::uint64_t> done{0}, written{0}, missing{0}, skipped{0};
    std::atomic<std::uint64_t> bytesWritten{0};
    // Per-level flag: spot-check the first non-sentinel inner chunk's C3DC
    // magic once per level. Sharded shards have a 16-byte-per-entry index
    // at offset 0 (index_location=start); sentinel entries (~0,~0) mean
    // the inner chunk is absent.
    std::vector<std::atomic<bool>> magicChecked(levels.size());
    for (auto& f : magicChecked) f.store(false, std::memory_order_relaxed);
    std::unordered_map<int, std::size_t> levelIndex;
    for (std::size_t i = 0; i < levels.size(); ++i) levelIndex[levels[i].level] = i;
    std::mutex stderrMutex;
    const auto t0 = std::chrono::steady_clock::now();

    auto worker = [&]() {
        while (true) {
            if (g_shutdown.load(std::memory_order_acquire)) return;
            std::size_t i = nextIdx.fetch_add(1, std::memory_order_relaxed);
            if (i >= queue.size()) return;
            const Shard& s = queue[i];

            fs::path shardPath = *s.lvlPath / "c"
                / std::to_string(s.sz) / std::to_string(s.sy) / std::to_string(s.sx);
            if (fs::exists(shardPath)) {
                skipped.fetch_add(1, std::memory_order_relaxed);
                done.fetch_add(1, std::memory_order_relaxed);
                continue;
            }

            std::vector<std::uint8_t> bytes;
            try { bytes = src.fetchWholeShard(s.level, s.sz, s.sy, s.sx); }
            catch (const std::exception& e) {
                std::lock_guard lk(stderrMutex);
                std::fprintf(stderr,
                    "[prefetch] level %d shard (%d,%d,%d) fetch failed: %s\n",
                    s.level, s.sz, s.sy, s.sx, e.what());
                done.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            if (bytes.empty()) {
                missing.fetch_add(1, std::memory_order_relaxed);
                done.fetch_add(1, std::memory_order_relaxed);
                continue;
            }

            const std::size_t li = levelIndex[s.level];
            if (!magicChecked[li].exchange(true, std::memory_order_acq_rel)) {
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
                        if (bytes[off + 0] != 'C' || bytes[off + 1] != '3'
                            || bytes[off + 2] != 'D' || bytes[off + 3] != 'C') {
                            throw std::runtime_error(
                                "level " + std::to_string(s.level) +
                                " shard inner chunk lacks C3DC magic");
                        }
                        sampled = true;
                        break;
                    }
                }
                if (!sampled) {
                    magicChecked[li].store(false, std::memory_order_release);
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
                        s.level, s.sz, s.sy, s.sx);
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
                    s.level, s.sz, s.sy, s.sx, ec.message().c_str());
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
        // Current level is the one containing queue[nextIdx - inflight], but
        // with 16 workers it's enough to show the frontier index's level.
        std::size_t ni = std::min<std::size_t>(
            nextIdx.load(std::memory_order_relaxed), queue.size() ? queue.size() - 1 : 0);
        int curLevel = queue.empty() ? -1 : queue[ni].level;
        std::lock_guard lk(stderrMutex);
        std::fprintf(stderr,
            "\r[prefetch] passthrough  shards %lu/%lu  written=%lu cached=%lu empty=%lu  "
            "%.0fMB @ %.1fMB/s  cur=L%d  ETA %s%s",
            static_cast<unsigned long>(d),
            static_cast<unsigned long>(totalShards),
            static_cast<unsigned long>(w),
            static_cast<unsigned long>(sk),
            static_cast<unsigned long>(m),
            double(bw) / (1024.0 * 1024.0),
            mbs,
            curLevel,
            fmtEta(static_cast<std::uint64_t>(etaSec)).c_str(),
            final ? "\n" : "");
        std::fflush(stderr);
    };
    while (done.load(std::memory_order_relaxed) < totalShards) {
        if (g_shutdown.load(std::memory_order_acquire)) break;
        std::this_thread::sleep_for(std::chrono::seconds(2));
        report(false);
    }
    for (auto& t : pool) t.join();
    report(true);
    if (g_shutdown.load(std::memory_order_acquire)) {
        std::fprintf(stderr, "[prefetch] passthrough aborted by user\n");
    }
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
        if (g_shutdown.load(std::memory_order_acquire)) break;
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
    std::signal(SIGINT, handleSigint);
    std::signal(SIGTERM, handleSigint);
    std::string url;
    std::string cacheDirStr;
    std::string levelsArg = "0";
    float targetRatio = 50.0f;
    int jobs = 0;

    po::options_description visible("Options");
    visible.add_options()
        ("help,h", "show help")
        ("cache-dir", po::value(&cacheDirStr),
            "local cache root (default: $HOME/.VC3D/remote_cache)")
        ("levels", po::value(&levelsArg)->default_value("0"),
            "comma-separated pyramid levels to prefetch in the given order, "
            "e.g. \"5,4,3,2,1,0\" prefetches coarsest first")
        ("target-ratio", po::value(&targetRatio)->default_value(50.0f),
            "c3d target compression ratio (transcoded chunks only; "
            "50 ≈ 40 dB PSNR on scroll CT)")
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
            "[--cache-dir PATH] [--levels 5,4,3,2,1,0] "
            "[--target-ratio 50] [--jobs N]\n\n"
            << visible << "\n";
        return url.empty() ? 2 : 0;
    }
    if (targetRatio < 1.0f) { std::cerr << "target-ratio must be >= 1.0\n"; return 2; }

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
    utils::C3dCodecParams enc;
    enc.target_ratio = targetRatio;
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
    std::vector<bool> levelCanonicalC3d(nLevels, false);
    std::vector<std::array<int, 3>> levelShapes(nLevels, {0, 0, 0});
    std::vector<std::array<int, 3>> levelShardShapes(nLevels, {0, 0, 0});
    for (int lvl : levelOrder) {
        auto meta = fetchLevelMetadata(baseUrl, lvl, auth);
        if (!meta) {
            std::cerr << "no metadata at level " << lvl << " — stopping\n";
            return 1;
        }
        bool canonical = utils::is_canonical_c3d(*meta);
        levelCanonicalC3d[lvl] = canonical;
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
            canonical ? "canonical c3d → passthrough" : "non-canonical → transcode");
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
            if (!levelCanonicalC3d[lvl]) continue;
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

    // Collect all passthrough levels into one queue so the worker pool
    // stays saturated across level boundaries. Transcode levels still run
    // sequentially via the BlockPipeline (they have their own internal
    // parallelism).
    std::vector<PassthroughLevel> passthroughLevels;
    for (int lvl : levelOrder) {
        if (!levelCanonicalC3d[lvl]) continue;
        (void)openOrCreateDiskLevel(vol->path(), lvl, levelShapes[lvl]);
        passthroughLevels.push_back({lvl, vol->path() / std::to_string(lvl)});
    }
    if (!passthroughLevels.empty() && !g_shutdown.load(std::memory_order_acquire)) {
        prefetchCanonicalLevels(passthroughSource, passthroughLevels, passthroughJobs);
    }

    for (int lvl : levelOrder) {
        if (g_shutdown.load(std::memory_order_acquire)) break;
        if (!levelCanonicalC3d[lvl]) {
            prefetchTranscodeLevel(pipeFor(), lvl, levelShapes[lvl]);
        }
    }

    if (g_shutdown.load(std::memory_order_acquire)) {
        std::fprintf(stderr, "[prefetch] aborted by user\n");
        return 130;
    }
    std::fprintf(stderr, "[prefetch] all levels complete\n");
    return 0;
}
