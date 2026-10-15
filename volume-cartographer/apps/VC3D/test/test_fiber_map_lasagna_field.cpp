// The Lasagna sheet-normal field adapter against a tiny dataset written on
// disk: real channels (compact nx/ny, grad_mag), a nonunit coordinate scale,
// a hole, the identity, and the geometry unit run on it.
#include "FiberMapBentRays.hpp"
#include "FiberMapLasagnaField.hpp"
#include "FiberNetworkLayout.hpp"
#include "LineAnnotationCoordinateScale.hpp"

#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/ChannelSampler.hpp"
#include "vc/lasagna/LasagnaNormalSampler.hpp"
#include "utils/zarr.hpp"

#include <QtTest/QtTest>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <atomic>
#include <chrono>
#include <sys/stat.h>
#include <csignal>
#include <fcntl.h>
#include <unistd.h>

namespace fs = std::filesystem;
using namespace vc3d::fiber_map::bent;

namespace
{

constexpr std::size_t kN = 32;         // array edge (one chunk)
constexpr double kScaledown = 2.0;     // group scaledown -> factor 4
constexpr double kWorkingToBase = 0.5; // annotation L0 vs lasagna base L1
// Working voxels per array index: factor * source_to_base / workingToBase.
constexpr double kSpacing = 4.0 * 1.0 / kWorkingToBase;
// The shelf slab (array z indices) where the axis is +z, and the hole slab
// where grad_mag is zero.
constexpr std::size_t kShelfLo = 14;
constexpr std::size_t kShelfHi = 17;
constexpr std::size_t kHoleLo = 26;
constexpr std::size_t kHoleHi = 28;


// A v2 uint8 array of the fixture's payload, one chunk unless `zChunk`
// splits it, raw unless a compressor is named.
void createU8Zarr(const fs::path& path, const std::vector<uint8_t>& payload, std::size_t zChunk = kN,
                  const std::string& compressor = "")
{
    utils::ZarrMetadata meta;
    meta.version = utils::ZarrVersion::v2;
    meta.shape = {kN, kN, kN};
    meta.chunks = {zChunk, kN, kN};
    meta.dtype = utils::ZarrDtype::uint8;
    meta.compressor_id = compressor;
    meta.fill_value = 0.0;
    fs::remove_all(path);
    auto array = utils::ZarrArray::create(path, meta, vc::buildZarrCodecRegistry(1));
    const std::size_t chunkBytes = zChunk * kN * kN;
    for (std::size_t c = 0; c * zChunk < kN; ++c) {
        std::vector<std::byte> bytes(chunkBytes);
        for (std::size_t i = 0; i < chunkBytes; ++i) {
            bytes[i] = static_cast<std::byte>(payload[c * chunkBytes + i]);
        }
        array.write_chunk(std::vector<std::size_t>{c, 0, 0}, bytes);
    }
}

// Compact normal component: raw = 128 + 127 * component.
uint8_t encode(double component)
{
    return static_cast<uint8_t>(std::lround(128.0 + 127.0 * component));
}

// A fixture dataset in a temporary directory removed with the fixture.
struct Dataset {
    std::shared_ptr<QTemporaryDir> owner;
    fs::path dir;
    fs::path manifest;
};

// `allShelf`: the axis is +z everywhere (a shelf the size of the volume,
// the hole slab kept), for the layout test.
Dataset writeDataset(const std::string& extra = "", bool allShelf = false)
{
    Dataset out;
    out.owner = std::make_shared<QTemporaryDir>();
    if (!out.owner->isValid()) {
        throw std::runtime_error("no temporary directory");
    }
    out.dir = fs::path(out.owner->path().toStdString());
    std::vector<uint8_t> nx(kN * kN * kN);
    std::vector<uint8_t> ny(kN * kN * kN);
    std::vector<uint8_t> gradMag(kN * kN * kN);
    for (std::size_t z = 0; z < kN; ++z) {
        for (std::size_t y = 0; y < kN; ++y) {
            for (std::size_t x = 0; x < kN; ++x) {
                const std::size_t i = (z * kN + y) * kN + x;
                const bool shelf = allShelf || (z >= kShelfLo && z <= kShelfHi);
                // +x everywhere, +z on the shelf (nz = sqrt(1 - nx^2 - ny^2)).
                nx[i] = encode(shelf ? 0.0 : 1.0);
                ny[i] = encode(0.0);
                const bool hole = z >= kHoleLo && z <= kHoleHi;
                gradMag[i] = hole ? 0 : 255;
            }
        }
    }
    createU8Zarr(out.dir / "nx.zarr", nx);
    createU8Zarr(out.dir / "ny.zarr", ny);
    createU8Zarr(out.dir / "grad_mag.zarr", gradMag);
    out.manifest = out.dir / "dataset.lasagna.json";
    std::ofstream file(out.manifest);
    file << R"({
  "version": 2,
  "source_to_base": 1.0,
  "grad_mag_encode_scale": 255.0,
  "grad_mag_factor": 1.0,
  "base_shape_zyx": [128, 128, 128],
  "groups": {
    "grad_mag": {"zarr": "grad_mag.zarr", "scaledown": 2, "channels": ["grad_mag"]},
    "nx": {"zarr": "nx.zarr", "scaledown": 2, "channels": ["nx"]},
    "ny": {"zarr": "ny.zarr", "scaledown": 2, "channels": ["ny"]}
  })" << extra << "\n}\n";
    return out;
}

std::shared_ptr<vc::lasagna::LasagnaDataset> openAt(const fs::path& manifest, double workingToBase)
{
    vc::lasagna::LasagnaDatasetOpenOptions options;
    options.workingToBaseScale = workingToBase;
    return std::make_shared<vc::lasagna::LasagnaDataset>(vc::lasagna::LasagnaDataset::open(manifest, options));
}

std::unique_ptr<LasagnaSheetNormalField> fieldFor(const fs::path& manifest, double workingToBase)
{
    auto dataset = openAt(manifest, workingToBase);
    auto sampler = std::make_shared<vc::lasagna::LasagnaNormalSampler>(*dataset);
    return std::make_unique<LasagnaSheetNormalField>(sampler, lasagnaFieldIdentity(*dataset, workingToBase), 2);
}

// Working-frame point at array indices (ix, iy, iz).
cv::Vec3d at(double ix, double iy, double iz)
{
    return cv::Vec3d(ix * kSpacing, iy * kSpacing, iz * kSpacing);
}

// The umbilicus is the z axis of the working frame.
UmbilicusFrame frame()
{
    UmbilicusFrame f;
    f.radialUnit = [](const cv::Vec3d& p) {
        const double r = std::hypot(p[0], p[1]);
        return r > 0.0 ? cv::Vec3d(p[0] / r, p[1] / r, 0.0) : cv::Vec3d(1.0, 0.0, 0.0);
    };
    f.radius = [](const cv::Vec3d& p) { return std::hypot(p[0], p[1]); };
    f.theta = [](const cv::Vec3d& p) { return std::atan2(p[1], p[0]); };
    // Exact along a segment: the horizontal offset is linear in the
    // parameter, its squared length a quadratic.
    f.minRadiusAlong = [](const cv::Vec3d& a, const cv::Vec3d& b) {
        const cv::Vec3d o0(a[0], a[1], 0.0);
        const cv::Vec3d d(b[0] - a[0], b[1] - a[1], 0.0);
        const double dd = d.dot(d);
        const double s = dd > 0.0 ? std::clamp(-o0.dot(d) / dd, 0.0, 1.0) : 0.0;
        const cv::Vec3d o = o0 + d * s;
        return std::min({std::sqrt(o.dot(o)), std::hypot(a[0], a[1]), std::hypot(b[0], b[1])});
    };
    return f;
}

} // namespace

class TestFiberMapLasagnaField : public QObject {
    Q_OBJECT

private slots:
    // A failing read does not end the batch while another reader is still
    // at work. Two chunks are FIFOs (a reader blocks in its open until a
    // writer comes; a writer end opened without blocking succeeds only
    // while a reader waits there, and is what lets it through), a third is
    // a directory (its read fails). Two workers: the reader of the first
    // FIFO is let through (it goes on to the directory and fails there)
    // while the reader of the second still waits; the call must still be
    // running 400 ms later; then the second is let through and the call
    // ends with the failure. Every path is bounded: both FIFOs are
    // released again and again until the call ends, and a call that does
    // not end within ten seconds fails the test (its thread is detached,
    // its state shared, nothing dangles). Which of the two pool tasks is
    // joined first is a race the test cannot pin: without the drain it
    // fails in the runs where the failed task's future comes first.
    void failedReadWaitsForTheBlockedReader()
    {
        const Dataset ds = writeDataset();
        {
            std::vector<uint8_t> nx(kN * kN * kN, 128);
            fs::remove_all(ds.dir / "nx.zarr");
            createU8Zarr(ds.dir / "nx.zarr", nx, 8);
        }
        struct State {
            std::shared_ptr<vc::lasagna::LasagnaDataset> dataset;
            vc::lasagna::LasagnaChannelBinding binding;
            std::vector<vc::lasagna::LasagnaChannelChunkKey> keys;
            vc::lasagna::LasagnaChannelChunkCache cache{1};
            std::atomic<bool> finished{false};
            std::atomic<bool> threw{false};
        };
        auto state = std::make_shared<State>();
        state->dataset = openAt(ds.manifest, kWorkingToBase);
        const auto& manifest = state->dataset->manifest();
        const vc::lasagna::LasagnaChannelGroup* group = nullptr;
        for (const auto& g : manifest.groups) {
            if (g.name == "nx") {
                group = &g;
            }
        }
        QVERIFY(group != nullptr);
        state->binding.group = group;
        state->binding.arrayId = 9;
        state->binding.channelIndex = 0;
        state->binding.array =
            std::make_shared<utils::ZarrArray>(vc::lasagna::openLasagnaChannelArray(manifest, *group));
        const auto& meta = state->binding.array->metadata();
        for (int i = 0; i < 3; ++i) {
            state->binding.shapeZYX[static_cast<std::size_t>(i)] = meta.shape[static_cast<std::size_t>(i)];
            state->binding.chunksZYX[static_cast<std::size_t>(i)] = meta.chunks[static_cast<std::size_t>(i)];
        }
        const fs::path first = ds.dir / "nx.zarr" / "0.0.0";
        const fs::path second = ds.dir / "nx.zarr" / "1.0.0";
        const fs::path failing = ds.dir / "nx.zarr" / "2.0.0";
        for (const fs::path& chunk : {first, second, failing}) {
            QVERIFY(fs::exists(chunk));
            fs::remove(chunk);
        }
        QCOMPARE(::mkfifo(first.c_str(), 0600), 0);
        QCOMPARE(::mkfifo(second.c_str(), 0600), 0);
        fs::create_directory(failing);
        state->keys = {{state->binding.arrayId, 0, 0, 0, 0}, {state->binding.arrayId, 0, 1, 0, 0},
                       {state->binding.arrayId, 0, 2, 0, 0}};
        // A reader that closes early turns a write into EPIPE, not a signal.
        struct PipeSignal {
            void (*previous)(int);
            PipeSignal() : previous(::signal(SIGPIPE, SIG_IGN)) {}
            ~PipeSignal() { ::signal(SIGPIPE, previous); }
        } pipeSignal;
        std::thread caller([state]() {
            vc::lasagna::LasagnaChannelChunkCache::ResolvedChunkMap resolved;
            try {
                (void)state->cache.prefetchResolved(state->binding, *state->binding.array, state->keys, 2, resolved);
            } catch (const std::exception&) {
                state->threw = true;
            }
            state->finished = true;
        });
        const std::vector<char> bytes(8 * kN * kN, 0);
        // Let a reader through: a non-blocking writer end (only while a
        // reader waits), a whole chunk of bytes, closed. False when no
        // reader waits there.
        const auto release = [&](const fs::path& fifo) {
            const int fd = ::open(fifo.c_str(), O_WRONLY | O_NONBLOCK);
            if (fd < 0) {
                return false;
            }
            std::size_t written = 0;
            while (written < bytes.size()) {
                const ssize_t n = ::write(fd, bytes.data() + written, bytes.size() - written);
                if (n <= 0) {
                    break;
                }
                written += static_cast<std::size_t>(n);
            }
            ::close(fd);
            return true;
        };
        // Both readers get a head start to reach their opens.
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        const bool firstWaited = release(first);
        bool finishedEarly = false;
        if (firstWaited) {
            std::this_thread::sleep_for(std::chrono::milliseconds(400));
            finishedEarly = state->finished.load();
        }
        const bool secondWaited = release(second);
        // Bounded cleanup: whatever happened above, every reader that comes
        // to either FIFO is let through until the call ends.
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (!state->finished.load() && std::chrono::steady_clock::now() < deadline) {
            (void)release(first);
            (void)release(second);
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        if (!state->finished.load()) {
            caller.detach();
            QFAIL("the prefetch did not end within ten seconds of both readers being let through");
        }
        caller.join();
        QVERIFY(state->threw.load());
        if (!(firstWaited && secondWaited)) {
            QSKIP("the read pool ran the two FIFO reads in sequence: the race under test did not arise");
        }
        QVERIFY(!finishedEarly);
    }

    // The frame a stored fiber's geometry is carried into (specialist S4):
    // a fiber stamped with a 32768^3 coordinate base shown in an untagged
    // 65536^3 level-0 volume is doubled into the frame; a fiber annotated in
    // this volume's base, or one whose base and level tags agree, moves
    // nothing; and the sheet field's working-to-base scale then carries the
    // frame point back onto the normals' own grid: the stored point samples
    // its own position.
    void storedFiberGeometryIsCarriedIntoTheFrame()
    {
        using vc3d::line_annotation::fiberMapFrameScale;
        const std::optional<std::array<std::size_t, 3>> base{{32768, 32768, 32768}};
        const std::array<int, 3> untaggedL0{65536, 65536, 65536};
        QCOMPARE(fiberMapFrameScale(base, untaggedL0, 1.0, true).value, 2.0);
        QCOMPARE(fiberMapFrameScale(base, std::array<int, 3>{32768, 32768, 32768}, 1.0, true).value, 1.0);
        QCOMPARE(fiberMapFrameScale(std::optional<std::array<std::size_t, 3>>{{65536, 65536, 65536}},
                                    std::array<int, 3>{32768, 32768, 32768}, 2.0, true).value,
                 1.0);
        QCOMPARE(fiberMapFrameScale(std::nullopt, untaggedL0, 1.0, true).value, 1.0);
        // The production conversion of a fiber's geometry: every control
        // and line point carried into the frame.
        std::vector<cv::Vec3d> controls = {cv::Vec3d(16000.0, 16000.0, 10000.0), cv::Vec3d(16100.0, 16000.0, 10000.0)};
        std::vector<cv::Vec3d> line = {cv::Vec3d(16000.0, 16000.0, 10000.0), cv::Vec3d(16050.0, 16000.0, 10000.0),
                                       cv::Vec3d(16100.0, 16000.0, 10000.0)};
        const std::vector<cv::Vec3d> storedLine = line;
        vc3d::line_annotation::scaleFiberMapGeometry(controls, line, fiberMapFrameScale(base, untaggedL0, 1.0, true), true);
        QCOMPARE(controls[0], cv::Vec3d(32000.0, 32000.0, 20000.0));
        QCOMPARE(controls[1], cv::Vec3d(32200.0, 32000.0, 20000.0));
        QCOMPARE(line[1], cv::Vec3d(32100.0, 32000.0, 20000.0));
        QCOMPARE(line[2], cv::Vec3d(32200.0, 32000.0, 20000.0));
        // The frame's extent is the untagged volume's; the normals' base is
        // the fiber's: the sheet field, opened at the scale the worker
        // resolves, queries the framed point at the stored one - the
        // sampler's working-to-base scale takes the frame coordinate back
        // onto the normals' grid.
        const double fieldScale = sheetFieldWorkingToBaseScale({65536.0, 65536.0, 65536.0}, base);
        QCOMPARE(fieldScale, 0.5);
        for (std::size_t i = 0; i < line.size(); ++i) {
            QCOMPARE(line[i] * fieldScale, storedLine[i]);
        }
        // And nothing moves for a fiber annotated in this volume's base.
        std::vector<cv::Vec3d> same = storedLine;
        std::vector<cv::Vec3d> sameControls = {storedLine[0]};
        vc3d::line_annotation::scaleFiberMapGeometry(sameControls, same, fiberMapFrameScale(base, std::array<int, 3>{32768, 32768, 32768}, 1.0, true), true);
        QCOMPARE(same, storedLine);
    }

    // The batch sampler's chunk prefetch resolves every chunk it loaded
    // even when the cache cannot keep them (a capacity of one byte evicts
    // each chunk as soon as it is cached): a chunk that loaded is never a
    // missing sample, so a batch under cache pressure reads what a single
    // sample reads.
    void batchPrefetchKeepsItsChunksUnderEviction()
    {
        const Dataset ds = writeDataset();
        // The nx array in four z slabs of eight: four chunks to resolve.
        {
            std::vector<uint8_t> nx(kN * kN * kN, 128);
            fs::remove_all(ds.dir / "nx.zarr");
            createU8Zarr(ds.dir / "nx.zarr", nx, 8);
        }
        const auto dataset = openAt(ds.manifest, kWorkingToBase);
        const auto& manifest = dataset->manifest();
        const vc::lasagna::LasagnaChannelGroup* group = nullptr;
        for (const auto& g : manifest.groups) {
            if (g.name == "nx") {
                group = &g;
            }
        }
        QVERIFY(group != nullptr);
        vc::lasagna::LasagnaChannelBinding binding;
        binding.group = group;
        binding.arrayId = 7;
        binding.channelIndex = 0;
        binding.array = std::make_shared<utils::ZarrArray>(vc::lasagna::openLasagnaChannelArray(manifest, *group));
        const auto& meta = binding.array->metadata();
        for (int i = 0; i < 3; ++i) {
            binding.shapeZYX[static_cast<std::size_t>(i)] = meta.shape[static_cast<std::size_t>(i)];
            binding.chunksZYX[static_cast<std::size_t>(i)] = meta.chunks[static_cast<std::size_t>(i)];
        }
        std::vector<vc::lasagna::LasagnaChannelChunkKey> keys;
        for (uint32_t zChunk = 0; zChunk * binding.chunksZYX[0] < binding.shapeZYX[0]; ++zChunk) {
            keys.push_back({binding.arrayId, 0, zChunk, 0, 0});
        }
        QVERIFY(keys.size() >= 2);
        const vc::lasagna::LasagnaChannelChunkCache cache(1);
        vc::lasagna::LasagnaChannelChunkCache::ResolvedChunkMap resolved;
        const auto report = cache.prefetchResolved(binding, *binding.array, keys, 2, resolved);
        QCOMPARE(report.requestedChunks, keys.size());
        QCOMPARE(resolved.size(), keys.size());
        for (const auto& key : keys) {
            const auto found = resolved.find(key);
            QVERIFY(found != resolved.end());
            QVERIFY(found->second != nullptr);
            QVERIFY(!found->second->values.empty());
        }
        // A second prefetch over the same keys resolves them all again (the
        // cache kept none).
        vc::lasagna::LasagnaChannelChunkCache::ResolvedChunkMap again;
        (void)cache.prefetchResolved(binding, *binding.array, keys, 1, again);
        QCOMPARE(again.size(), keys.size());
        // A read that fails (a chunk file cut short) is reported as an
        // exception once every worker has finished; the batch's storage is
        // not left to running workers. Repeated so a late worker gets its
        // chance to misbehave.
        {
            const fs::path chunk = ds.dir / "nx.zarr" / "2.0.0";
            QVERIFY(fs::exists(chunk));
            fs::remove(chunk);
            fs::create_directory(chunk);
            for (int repeat = 0; repeat < 8; ++repeat) {
                vc::lasagna::LasagnaChannelChunkCache::ResolvedChunkMap partial;
                bool threw = false;
                try {
                    (void)cache.prefetchResolved(binding, *binding.array, keys, 3, partial);
                } catch (const std::exception&) {
                    threw = true;
                }
                QVERIFY(threw);
            }
        }
    }

    void axesDecodeAtTheWorkingScale()
    {
        const Dataset ds = writeDataset();
        const auto field = fieldFor(ds.manifest, kWorkingToBase);
        // Index (15, 15, 5): +x; (15, 15, 15): on the shelf, +z; (15, 15,
        // 27): in the hole, no value. Working coordinates are 8 per index.
        std::vector<cv::Vec3d> points = {at(15, 15, 5), at(15, 15, 15.5), at(15, 15, 27)};
        std::vector<std::optional<cv::Vec3d>> axes;
        field->axes(points, axes);
        QCOMPARE(axes.size(), std::size_t{3});
        QVERIFY(axes[0].has_value());
        QVERIFY(std::abs((*axes[0])[0]) > 0.99);
        QVERIFY(axes[1].has_value());
        QVERIFY((*axes[1])[2] > 0.99);
        QVERIFY(!axes[2].has_value());
        // The scalar path agrees with the batch path.
        const auto single = field->axis(points[1]);
        QVERIFY(single.has_value());
        QVERIFY(std::abs((*single)[2] - (*axes[1])[2]) < 1e-9);
        // The same dataset opened at scale 1 reads a different place for the
        // same working point: the scale is part of the frame.
        const auto unscaled = fieldFor(ds.manifest, 1.0);
        const auto other = unscaled->axis(at(15, 15, 15.5));
        QVERIFY(other.has_value());
        // At scale 1 the spacing halves: working z 124 is index 31, off
        // the shelf.
        QVERIFY(std::abs((*other)[0]) > 0.99);
    }

    void identityFollowsTheContent()
    {
        const Dataset ds = writeDataset();
        const std::string a = fieldFor(ds.manifest, kWorkingToBase)->identity();
        const std::string b = fieldFor(ds.manifest, kWorkingToBase)->identity();
        QCOMPARE(a, b);
        QVERIFY(a.rfind("lasagna|", 0) == 0);
        QVERIFY(a.find("|grad_mag:") != std::string::npos);
        QVERIFY(a.find("|nx:") != std::string::npos);
        QVERIFY(a.find("|ny:") != std::string::npos);
        // The scale is part of it.
        QVERIFY(fieldFor(ds.manifest, 1.0)->identity() != a);
        // A manifest edit at the SAME location is a different field, and the
        // text restored is the field again.
        {
            std::ifstream in(ds.manifest);
            std::stringstream text;
            text << in.rdbuf();
            in.close();
            const std::string original = text.str();
            std::string edited = original;
            const std::size_t brace = edited.find('{');
            QVERIFY(brace != std::string::npos);
            edited.insert(brace + 1, "\n  \"note\": \"edited\",");
            {
                std::ofstream out(ds.manifest, std::ios::trunc);
                out << edited;
            }
            QVERIFY(fieldFor(ds.manifest, kWorkingToBase)->identity() != a);
            {
                std::ofstream out(ds.manifest, std::ios::trunc);
                out << original;
            }
            QCOMPARE(fieldFor(ds.manifest, kWorkingToBase)->identity(), a);
        }
        // So is an array whose metadata differs, at the SAME location: the
        // same bytes rechunked, then the same bytes under a v2 compressor
        // (which the library's v3 serializer would not show).
        const std::vector<uint8_t> ny(kN * kN * kN, encode(0.0));
        createU8Zarr(ds.dir / "ny.zarr", ny, kN / 2);
        const std::string rechunked = fieldFor(ds.manifest, kWorkingToBase)->identity();
        QVERIFY(rechunked != a);
        createU8Zarr(ds.dir / "ny.zarr", ny, kN, "zlib");
        const std::string compressed = fieldFor(ds.manifest, kWorkingToBase)->identity();
        QVERIFY(compressed != a);
        QVERIFY(compressed != rechunked);
        // The sampler still reads the rewritten channel.
        QVERIFY(fieldFor(ds.manifest, kWorkingToBase)->axis(at(16.0, 16.0, 8.0)).has_value());
        // Restored, the identity is the original one.
        createU8Zarr(ds.dir / "ny.zarr", ny);
        QCOMPARE(fieldFor(ds.manifest, kWorkingToBase)->identity(), a);
    }

    void profileAndCurtainOnRealChannels()
    {
        const Dataset ds = writeDataset();
        const auto field = fieldFor(ds.manifest, kWorkingToBase);
        // A fiber at index y 8 running diagonally from (x 10, z 6) to
        // (x 28, z 24): well conditioned (+x against e_r, which has a
        // positive x component there) except on the shelf, where the axis
        // is +z, across the fiber's direction.
        std::vector<cv::Vec3d> h;
        for (int i = 0; i <= 180; ++i) {
            h.push_back(at(10.0 + 18.0 * i / 180.0, 8.0, 6.0 + 18.0 * i / 180.0));
        }
        BentRayParams params;
        params.stepVx = 4.0;
        params.maxLengthVx = 100.0;
        params.spacingVx = 8.0;
        const ConditioningProfile profile = conditioningProfile(*field, h, frame());
        const auto runs = illConditionedRuns(illConditionedSamples(profile, params.conditioningGate));
        QCOMPARE(runs.size(), std::size_t{1});
        // The run spans the shelf indices (sampling interpolates across
        // one index at each edge).
        QVERIFY(h[runs[0].first][2] / kSpacing > kShelfLo - 1.5);
        QVERIFY(h[runs[0].second][2] / kSpacing < kShelfHi + 1.5);
        const FiberOrientation orientation = orientFiber(h, profile, frame(), params);
        QCOMPARE(orientation.runs.size(), std::size_t{1});
        QCOMPARE(orientation.runs[0].status, RunStatus::Oriented);
        const BentCurtain curtain = traceCurtain(*field, h, orientation, frame(), params, 10.0);
        QCOMPARE(curtain.stretches.size(), std::size_t{1});
        QVERIFY(curtain.stretches[0].rays.size() >= 4);
        // Outward rays climb through the slab (they turn radial again above
        // it, where the field does): a radial crosser 8 vx above a shelf
        // sample, inside the slab, is read on side +1 at that arclength.
        const BentStretch& stretch = curtain.stretches[0];
        const std::size_t i = stretch.firstSample + (stretch.lastSample - stretch.firstSample) / 2;
        const cv::Vec3d er = frame().radialUnit(h[i]);
        std::vector<cv::Vec3d> crosser;
        for (int k = 0; k <= 4; ++k) {
            crosser.push_back(h[i] + cv::Vec3d(0.0, 0.0, 8.0) + er * (-40.0 + 20.0 * k));
        }
        const auto hits = intersectCurtain(curtain, crosser);
        QCOMPARE(hits.size(), std::size_t{1});
        QCOMPARE(hits[0].side, 1);
        QVERIFY(std::abs(hits[0].s - 8.0) < 4.5);
    }

    // The whole layout on the real channels: the shelf fixture of the
    // layout tests, scaled into this dataset (a shelf everywhere: every
    // fiber sample is ill conditioned, every ray vertical). H0 one pitch
    // above Vb's radial run, Va a thickness behind H0 as the witness, Vb
    // plain-linked to H1 above. Cached and fresh builds agree bit for bit
    // in their semantic fields; the bent reading places H0 one winding out
    // of H1; a forced repair (H0 plain-linked to H1, which witnesses
    // nothing) drops the bent reading and marks it with its 3D hit; the
    // coverage records the set-aside straight reading.
    // The worker's scale: the annotation frame's extent against the
    // manifest's base shape, dyadic. An annotation at L0 over a base at L1
    // gives 0.5; the same grid 1; no base shape 1; an unknown extent with a
    // base shape cannot be resolved.
    void workingToBaseScaleIsResolvedFromTheFrame()
    {
        using vc3d::fiber_map::bent::sheetFieldWorkingToBaseScale;
        const std::optional<std::array<std::size_t, 3>> base{{128, 128, 128}};
        QCOMPARE(sheetFieldWorkingToBaseScale({256.0, 256.0, 256.0}, base), 0.5);
        QCOMPARE(sheetFieldWorkingToBaseScale({128.0, 128.0, 128.0}, base), 1.0);
        QCOMPARE(sheetFieldWorkingToBaseScale({64.0, 64.0, 64.0}, base), 2.0);
        QCOMPARE(sheetFieldWorkingToBaseScale({0.0, 0.0, 0.0}, std::nullopt), 1.0);
        bool threw = false;
        try {
            (void)sheetFieldWorkingToBaseScale({0.0, 0.0, 0.0}, base);
        } catch (const std::exception&) {
            threw = true;
        }
        QVERIFY(threw);
        // The dataset opened at that scale reads its channels in the
        // annotation frame: the shelf slab (array z 14..17 at scaledown 2 of a
        // 128 base, 4 base voxels per index) sits at annotation z 56..68 when
        // the annotation is the base grid, 112..136 when it is twice it.
        const Dataset ds = writeDataset();
        const double atBase = sheetFieldWorkingToBaseScale({128.0, 128.0, 128.0}, base);
        const auto fieldBase = fieldFor(ds.manifest, atBase);
        QVERIFY(fieldBase->axis(cv::Vec3d(64.0, 64.0, 62.0)).has_value());
        QVERIFY(std::abs((*fieldBase->axis(cv::Vec3d(64.0, 64.0, 62.0)))[2] - 1.0) < 1e-6);
        QVERIFY(std::abs((*fieldBase->axis(cv::Vec3d(64.0, 64.0, 124.0)))[2]) < 1e-6);
        const double atL0 = sheetFieldWorkingToBaseScale({256.0, 256.0, 256.0}, base);
        const auto fieldL0 = fieldFor(ds.manifest, atL0);
        QVERIFY(std::abs((*fieldL0->axis(cv::Vec3d(128.0, 128.0, 124.0)))[2] - 1.0) < 1e-6);
        QVERIFY(std::abs((*fieldL0->axis(cv::Vec3d(128.0, 128.0, 62.0)))[2]) < 1e-6);
    }

    void layoutOnTheDataset()
    {
        using vc3d::fiber_map::CrossingEvent;
        using vc3d::fiber_map::GlobalLayoutCache;
        using vc3d::fiber_map::GlobalLayoutParams;
        using vc3d::fiber_map::GlobalResult;
        using vc3d::fiber_map::InputFiber;
        using vc3d::fiber_map::InputLink;
        using vc3d::fiber_map::buildGlobalLayout;
        using vc3d::fiber_map::digestGlobalResult;
        using vc3d::fiber_map::winding::CrossingKind;
        using vc3d::fiber_map::winding::CrossingStatus;
        const Dataset ds = writeDataset("", true);
        const auto field = fieldFor(ds.manifest, kWorkingToBase);
        // The working frame spans 32 * kSpacing; the umbilicus runs up its
        // middle. Lengths in working voxels.
        const double center = 16.0 * kSpacing;
        const double radius = 8.0 * kSpacing;
        const double zRun = 8.0 * kSpacing;
        const double zShelf = 11.0 * kSpacing;
        const double zTop = 22.0 * kSpacing;
        const double thetaA = 0.5 * M_PI;
        const double thetaB = 0.8 * M_PI;
        std::vector<cv::Vec3f> umbilicus;
        for (int z = 0; z <= 32; ++z) {
            umbilicus.emplace_back(static_cast<float>(center), static_cast<float>(center),
                                   static_cast<float>(z * kSpacing));
        }
        const auto arc = [&](double r, double z, double t0, double t1, int count) {
            std::vector<cv::Vec3d> points;
            for (int i = 0; i < count; ++i) {
                const double t = t0 + (t1 - t0) * i / (count - 1);
                points.emplace_back(center + r * std::cos(t), center + r * std::sin(t), z);
            }
            return points;
        };
        const auto radial = [&](double theta, double r0, double r1, double z, int count) {
            std::vector<cv::Vec3d> points;
            for (int i = 0; i < count; ++i) {
                const double r = r0 + (r1 - r0) * i / (count - 1);
                points.emplace_back(center + r * std::cos(theta), center + r * std::sin(theta), z);
            }
            return points;
        };
        const auto fiber = [](uint64_t id, const char* label, char tag, std::vector<cv::Vec3d> line,
                              std::vector<int> controls) {
            InputFiber f;
            f.id = id;
            f.fileName = std::string(label) + ".json";
            f.label = QString::fromLatin1(label);
            f.hvTag = tag;
            for (const int c : controls) {
                f.controlPoints.push_back(line[static_cast<std::size_t>(c)]);
            }
            f.linePoints = std::move(line);
            f.tracedSegments.assign(f.controlPoints.size() - 1, true);
            return f;
        };
        const auto link = [](InputFiber& a, int ca, InputFiber& b, int cb) {
            a.links.push_back(InputLink{ca, b.id, cb});
            b.links.push_back(InputLink{cb, a.id, ca});
        };
        const int arcCount = 201;
        const int ia = static_cast<int>(std::llround((thetaA - 0.2 * M_PI) / M_PI * (arcCount - 1)));
        const int ib = static_cast<int>(std::llround((thetaB - 0.2 * M_PI) / M_PI * (arcCount - 1)));
        InputFiber h0 = fiber(1, "d-h0", 'H', arc(radius, zShelf, 0.2 * M_PI, 1.2 * M_PI, arcCount), {0, ia, ib, arcCount - 1});
        InputFiber va = fiber(2, "d-va", 'V', radial(thetaA, radius - 8.0, radius + 8.0, zShelf + 1.0, 9), {0, 4, 8});
        std::vector<cv::Vec3d> vbLine = radial(thetaB, radius - 24.0, radius + 24.0, zRun, 25);
        for (int i = 1; i <= 28; ++i) {
            vbLine.emplace_back(center + (radius + 24.0) * std::cos(thetaB),
                                center + (radius + 24.0) * std::sin(thetaB), zRun + i * 4.0);
        }
        const int vbLast = static_cast<int>(vbLine.size()) - 1;
        InputFiber vb = fiber(3, "d-vb", 'V', std::move(vbLine), {0, 12, 24, vbLast});
        InputFiber h1 = fiber(4, "d-h1", 'H', arc(radius + 20.0, zTop, 0.2 * M_PI, 1.2 * M_PI, arcCount), {0, ib, arcCount - 1});
        link(h0, 1, va, 1);
        link(h1, 1, vb, 3);
        std::vector<InputFiber> fibers{h0, va, vb, h1};
        GlobalLayoutParams params;
        params.smoothVx = 0.0;
        params.resampleStepVx = 2.0;
        params.minPadXVx = 10.0;
        params.minPadYVx = 10.0;
        params.solver.chiralityOverride = 1;
        params.solver.minUmbilicusRadiusVx = 4.0;
        params.solver.tieBandVx = 2.0;
        params.solver.zMergeVx = 4.0;
        params.solver.neighborhoodZVx = 8.0;
        params.solver.neighborhoodArcVx = 8.0;
        params.bentRays.stepVx = 2.0;
        params.bentRays.maxLengthVx = 40.0;
        params.bentRays.spacingVx = 2.0;
        GlobalLayoutCache cache;
        const GlobalResult cold = buildGlobalLayout(fibers, umbilicus, params, &cache, field.get(), std::string());
        const GlobalResult warm = buildGlobalLayout(fibers, umbilicus, params, &cache, field.get(), std::string());
        const GlobalResult fresh = buildGlobalLayout(fibers, umbilicus, params, nullptr, field.get(), std::string());
        QVERIFY(digestGlobalResult(cold) == digestGlobalResult(warm));
        QVERIFY(digestGlobalResult(cold) == digestGlobalResult(fresh));
        QCOMPARE(cache.lastStats().pairsRecomputed, 0);
        QCOMPARE(cold.effectiveField, field->identity());
        QVERIFY(cold.bentCrossingCount >= 1);
        QCOMPARE(cold.droppedCrossingCount, 0);
        QVERIFY(cold.setAsideCount >= 1);
        const CrossingEvent* reading = nullptr;
        for (const CrossingEvent& event : cold.crossingEvents) {
            if (event.bent && !event.withheld && !event.tangential && event.hFiberId == 1 &&
                event.vFiberId == 3) {
                reading = &event;
            }
        }
        QVERIFY(reading != nullptr);
        QCOMPARE(reading->kind, CrossingKind::Outside);
        QCOMPARE(reading->anchor, 2);
        QVERIFY(std::abs(reading->hitVx[2] - zRun) < 1e-6);
        QVERIFY(std::abs(std::hypot(reading->hitVx[0] - center, reading->hitVx[1] - center) - radius) < 1.0);
        double w0 = 0.0;
        double w1 = 0.0;
        for (const auto& placed : cold.fibers) {
            if (placed.fiber.id == 1) {
                w0 = placed.meta.windingLo;
            }
            if (placed.fiber.id == 4) {
                w1 = placed.meta.windingLo;
            }
        }
        QVERIFY(std::abs((w0 - w1) - 1.0) < 1e-9);
        bool covered = false;
        for (const auto& record : cold.pairCoverage) {
            if (record.hFiberId == 1 && record.vFiberId == 3) {
                covered = true;
                QCOMPARE(record.reason, std::string("replaced"));
            }
        }
        QVERIFY(covered);
        // The forced repair.
        std::vector<InputFiber> conflicted = fibers;
        link(conflicted[0], 2, conflicted[3], 1);
        const GlobalResult repaired = buildGlobalLayout(conflicted, umbilicus, params, &cache, field.get(), std::string());
        QCOMPARE(repaired.droppedCrossingCount, 1);
        QCOMPARE(repaired.suspectCrossings.size(), std::size_t{1});
        QVERIFY(repaired.suspectCrossings.front().bent);
        QVERIFY(std::abs(repaired.suspectCrossings.front().hitVx[2] - zRun) < 1e-6);
        QCOMPARE(repaired.crossingEvents[repaired.suspectCrossings.front().eventIndex].status,
                 CrossingStatus::Dropped);
        // The link edit recomputed nothing.
        QCOMPARE(cache.lastStats().pairsRecomputed, 0);
    }
};

QTEST_APPLESS_MAIN(TestFiberMapLasagnaField)
#include "test_fiber_map_lasagna_field.moc"
