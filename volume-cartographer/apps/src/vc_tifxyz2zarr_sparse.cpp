#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/Zarr.hpp"
#include "vc/core/types/VcDataset.hpp"

#include <boost/program_options.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <numeric>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <optional>
#include <cstring>

#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
namespace po = boost::program_options;

namespace {

constexpr size_t kPyramidLevels = 5;
constexpr size_t kLocalSpoolFlushThreshold = 8192;
constexpr float kAxisPlaneEps = 1e-5f;
constexpr float kSegmentEps2 = 1e-12f;
constexpr uint8_t kSurfaceValue = 255;
const char* kChunkSuffix = "vc_tifxyz2zarr_sparse";

using Shape3 = std::array<size_t, 3>;

struct ChunkIndex {
    uint32_t z;
    uint32_t y;
    uint32_t x;

    bool operator==(const ChunkIndex& other) const noexcept {
        return z == other.z && y == other.y && x == other.x;
    }
};

struct ChunkIndexHash {
    size_t operator()(const ChunkIndex& c) const noexcept {
        size_t h = 1469598103934665603ull;
        h ^= static_cast<size_t>(c.z);
        h *= 1099511628211ull;
        h ^= static_cast<size_t>(c.y);
        h *= 1099511628211ull;
        h ^= static_cast<size_t>(c.x);
        h *= 1099511628211ull;
        return h;
    }
};

static bool chunkIndexLess(const ChunkIndex& a, const ChunkIndex& b) {
    if (a.z != b.z) return a.z < b.z;
    if (a.y != b.y) return a.y < b.y;
    return a.x < b.x;
}

struct VoxelCoord {
    uint32_t z;
    uint32_t y;
    uint32_t x;
};
using VoxelCoord32 = VoxelCoord;
static_assert(sizeof(VoxelCoord32) == 12, "VoxelCoord32 must be a compact POD");

enum class SpoolCoordType : uint8_t {
    U8,
    U16,
    U32
};

static size_t coordTypeBytes(SpoolCoordType type) {
    switch (type) {
    case SpoolCoordType::U8: return 1;
    case SpoolCoordType::U16: return 2;
    case SpoolCoordType::U32: return 4;
    }
    return 4;
}

static const char* coordTypeName(SpoolCoordType type) {
    switch (type) {
    case SpoolCoordType::U8: return "uint8";
    case SpoolCoordType::U16: return "uint16";
    case SpoolCoordType::U32: return "uint32";
    }
    return "uint32";
}

static SpoolCoordType chooseSpoolCoordType(const Shape3& shape) {
    if (shape[0] == 0 || shape[1] == 0 || shape[2] == 0) {
        throw std::runtime_error("invalid volume shape with zero dimension");
    }

    const auto maxIndex = std::max(shape[0] - 1,
                                  std::max(shape[1] - 1, shape[2] - 1));
    if (maxIndex <= std::numeric_limits<uint8_t>::max()) return SpoolCoordType::U8;
    if (maxIndex <= std::numeric_limits<uint16_t>::max()) return SpoolCoordType::U16;
    return SpoolCoordType::U32;
}

template <typename T>
static void appendBinary(std::vector<uint8_t>& out, const T value) {
    const auto old = out.size();
    out.resize(old + sizeof(T));
    std::memcpy(out.data() + old, &value, sizeof(T));
}

template <typename T>
static T readBinary(const uint8_t* p) {
    T v;
    std::memcpy(&v, p, sizeof(T));
    return v;
}

static_assert(sizeof(VoxelCoord) == 12, "VoxelCoord must be a compact POD");

class SpoolManager {
public:
    SpoolManager(fs::path spoolDir,
                 const Shape3& chunkShape,
                 const Shape3& volumeShape,
                 SpoolCoordType coordType,
                 size_t inMemoryMaxBytes)
        : m_spoolDir(std::move(spoolDir)),
          m_chunkShape(chunkShape),
          m_volumeShape(volumeShape),
          m_coordType(coordType),
          m_recordBytes(3 * coordTypeBytes(coordType)),
          m_inMemoryMaxBytes(inMemoryMaxBytes) {
        fs::create_directories(m_spoolDir);
    }

    fs::path spoolPathFor(const ChunkIndex& chunk) const {
        return m_spoolDir / (std::to_string(chunk.z) + "_" +
                            std::to_string(chunk.y) + "_" +
                            std::to_string(chunk.x) + ".bin");
    }

    void appendChunkCoords(const ChunkIndex& chunk,
                          const std::vector<VoxelCoord32>& coords) {
        if (coords.empty()) return;

        if (m_inMemoryMaxBytes == 0) {
            appendChunkCoordsPacked(chunk, packCoords(coords));
            std::lock_guard<std::mutex> lk(m_touchedMutex);
            m_touchedChunks.emplace(chunk);
            return;
        }

        std::vector<uint8_t> packed;
            packed.reserve(coords.size() * m_recordBytes);
            for (const auto& c : coords) {
                packCoord(packed, c);
            }

        std::vector<std::pair<ChunkIndex, std::vector<uint8_t>>> toSpill;

        {
            std::lock_guard<std::mutex> lk(m_memoryMutex);
            auto it = m_memorySpool.find(chunk);
            auto& memVec = (it == m_memorySpool.end())
                               ? m_memorySpool.emplace(chunk, std::vector<uint8_t> {}).first->second
                               : it->second;
            if (it == m_memorySpool.end()) {
                m_memOrder.push_back(chunk);
            }

            memVec.insert(memVec.end(), packed.begin(), packed.end());
            m_inMemoryBytes += packed.size();

            while (m_inMemoryBytes > m_inMemoryMaxBytes && !m_memOrder.empty()) {
                const auto victim = m_memOrder.front();
                const auto victimIt = m_memorySpool.find(victim);
                if (victimIt == m_memorySpool.end()) {
                    m_memOrder.pop_front();
                    continue;
                }

            toSpill.emplace_back(victim, std::move(victimIt->second));
            m_inMemoryBytes -= toSpill.back().second.size();
            m_memorySpool.erase(victimIt);
            m_memOrder.pop_front();
        }
        }

        for (auto& item : toSpill) {
            if (!item.second.empty()) {
                appendChunkCoordsPacked(item.first, item.second);
            }
        }

        {
            std::lock_guard<std::mutex> lk(m_touchedMutex);
            m_touchedChunks.emplace(chunk);
        }
    }

    bool readChunkCoords(const ChunkIndex& chunk, std::vector<VoxelCoord32>& out) const {
        const auto file = spoolPathFor(chunk);
        const size_t startSize = out.size();

        size_t fileCount = 0;
        size_t memCount = 0;

        if (fs::exists(file)) {
            std::ifstream in(file, std::ios::binary);
            if (!in) return false;

            in.seekg(0, std::ios::end);
            const auto fileBytes = static_cast<size_t>(in.tellg());
            if ((fileBytes % m_recordBytes) != 0) {
                throw std::runtime_error("invalid spool file size for chunk: " + file.string());
            }
            if (fileBytes > 0) {
                const size_t count = fileBytes / m_recordBytes;
                fileCount = count;
                in.seekg(0, std::ios::beg);
                std::vector<uint8_t> fileBuf(fileBytes);
                in.read(reinterpret_cast<char*>(fileBuf.data()),
                        static_cast<std::streamsize>(fileBytes));
                if (!in) {
                    throw std::runtime_error("failed reading spool file: " + file.string());
                }
                out.reserve(startSize + fileCount + memCount);
                unpackCoords(fileBuf, out);
            }
        }

        {
            std::lock_guard<std::mutex> lk(m_memoryMutex);
            const auto memIt = m_memorySpool.find(chunk);
            if (memIt != m_memorySpool.end() && !memIt->second.empty()) {
                memCount = memIt->second.size() / m_recordBytes;
                if (memCount > 0) {
                    out.reserve(startSize + fileCount + memCount);
                }
                unpackCoords(memIt->second.data(), memIt->second.size(), out);
            }
        }

        return !out.empty();
    }

    size_t inMemoryBudgetBytes() const { return m_inMemoryMaxBytes; }
    size_t recordBytes() const { return m_recordBytes; }
    SpoolCoordType coordType() const { return m_coordType; }

    const Shape3& chunkShape() const { return m_chunkShape; }
    const Shape3& volumeShape() const { return m_volumeShape; }

    std::vector<ChunkIndex> touchedChunks() const {
        std::lock_guard<std::mutex> lk(m_touchedMutex);
        std::vector<ChunkIndex> touched;
        touched.reserve(m_touchedChunks.size());
        for (const auto& c : m_touchedChunks) touched.push_back(c);
        return touched;
    }

private:
    void packCoord(std::vector<uint8_t>& out, const VoxelCoord32& c) const {
        const auto maxZ = m_volumeShape[0] - 1;
        const auto maxY = m_volumeShape[1] - 1;
        const auto maxX = m_volumeShape[2] - 1;
        if (c.z > maxZ || c.y > maxY || c.x > maxX) {
            throw std::runtime_error("voxel coordinate outside shape while writing spool");
        }

        switch (m_coordType) {
        case SpoolCoordType::U8:
            appendBinary<uint8_t>(out, static_cast<uint8_t>(c.z));
            appendBinary<uint8_t>(out, static_cast<uint8_t>(c.y));
            appendBinary<uint8_t>(out, static_cast<uint8_t>(c.x));
            break;
        case SpoolCoordType::U16:
            appendBinary<uint16_t>(out, static_cast<uint16_t>(c.z));
            appendBinary<uint16_t>(out, static_cast<uint16_t>(c.y));
            appendBinary<uint16_t>(out, static_cast<uint16_t>(c.x));
            break;
        case SpoolCoordType::U32:
            appendBinary<uint32_t>(out, static_cast<uint32_t>(c.z));
            appendBinary<uint32_t>(out, static_cast<uint32_t>(c.y));
            appendBinary<uint32_t>(out, static_cast<uint32_t>(c.x));
            break;
        }
    }

    void unpackCoords(const uint8_t* buf, size_t bytes, std::vector<VoxelCoord32>& out) const {
        if (bytes == 0) return;
        if ((bytes % m_recordBytes) != 0) {
            throw std::runtime_error("invalid spool byte count for unpacking");
        }

        const size_t count = bytes / m_recordBytes;
        const size_t start = out.size();
        out.resize(start + count);
        const uint8_t* p = buf;

        switch (m_coordType) {
        case SpoolCoordType::U8:
            for (size_t i = 0; i < count; ++i) {
                out[start + i] = {readBinary<uint8_t>(p),
                                  readBinary<uint8_t>(p + 1),
                                  readBinary<uint8_t>(p + 2)};
                p += 3;
            }
            break;
        case SpoolCoordType::U16: {
            for (size_t i = 0; i < count; ++i) {
                out[start + i] = {readBinary<uint16_t>(p),
                                  readBinary<uint16_t>(p + 2),
                                  readBinary<uint16_t>(p + 4)};
                p += 6;
            }
            break;
        }
        case SpoolCoordType::U32: {
            for (size_t i = 0; i < count; ++i) {
                out[start + i] = {readBinary<uint32_t>(p),
                                  readBinary<uint32_t>(p + 4),
                                  readBinary<uint32_t>(p + 8)};
                p += 12;
            }
            break;
        }
        }
    }

    void unpackCoords(const std::vector<uint8_t>& buf, std::vector<VoxelCoord32>& out) const {
        unpackCoords(buf.data(), buf.size(), out);
    }

    std::vector<uint8_t> packCoords(const std::vector<VoxelCoord32>& coords) const {
        std::vector<uint8_t> packed;
        packed.reserve(coords.size() * m_recordBytes);
        for (const auto& c : coords) {
            packCoord(packed, c);
        }
        return packed;
    }

    void appendChunkCoordsPacked(const ChunkIndex& chunk,
                                const std::vector<uint8_t>& packed) {
        auto mtx = lockForChunk(chunk);
        {
            std::lock_guard<std::mutex> lk(*mtx);
            std::ofstream out(spoolPathFor(chunk),
                             std::ios::binary | std::ios::app);
            if (!out) {
                throw std::runtime_error("failed opening spool file: " +
                                         spoolPathFor(chunk).string());
            }
            if (!packed.empty()) {
                out.write(reinterpret_cast<const char*>(packed.data()),
                          static_cast<std::streamsize>(packed.size()));
            }
            if (!out) {
                throw std::runtime_error("failed writing to spool file: " +
                                         spoolPathFor(chunk).string());
            }
        }
    }

    std::shared_ptr<std::mutex> lockForChunk(const ChunkIndex& chunk) {
        std::lock_guard<std::mutex> lk(m_lockMapMutex);
        auto it = m_lockMap.find(chunk);
        if (it != m_lockMap.end()) return it->second;
        auto ptr = std::make_shared<std::mutex>();
        m_lockMap.emplace(chunk, ptr);
        return ptr;
    }

    fs::path m_spoolDir;
    Shape3 m_chunkShape;
    Shape3 m_volumeShape;
    const SpoolCoordType m_coordType;
    const size_t m_recordBytes;

    const size_t m_inMemoryMaxBytes;

    mutable std::mutex m_memoryMutex;
    size_t m_inMemoryBytes{0};
    std::unordered_map<ChunkIndex,
                       std::vector<uint8_t>,
                       ChunkIndexHash> m_memorySpool;
    std::list<ChunkIndex> m_memOrder;

    mutable std::mutex m_touchedMutex;
    mutable std::mutex m_lockMapMutex;
    std::unordered_set<ChunkIndex, ChunkIndexHash> m_touchedChunks;
    std::unordered_map<ChunkIndex,
                       std::shared_ptr<std::mutex>,
                       ChunkIndexHash> m_lockMap;
};

class ThreadSpoolBuffer {
public:
    ThreadSpoolBuffer(SpoolManager& manager,
                      const Shape3& chunkShape,
                      const Shape3& volumeShape)
        : m_manager(manager), m_chunkShape(chunkShape), m_volumeShape(volumeShape) {}

    void emit(size_t z, size_t y, size_t x) {
        if (z >= m_volumeShape[0] || y >= m_volumeShape[1] || x >= m_volumeShape[2]) {
            return;
        }

        const auto cz = static_cast<uint32_t>(z / m_chunkShape[0]);
        const auto cy = static_cast<uint32_t>(y / m_chunkShape[1]);
        const auto cx = static_cast<uint32_t>(x / m_chunkShape[2]);
        const ChunkIndex chunk{cz, cy, cx};

        auto& vec = m_buffers[chunk];
        vec.push_back(VoxelCoord32{static_cast<uint32_t>(z),
                                static_cast<uint32_t>(y),
                                static_cast<uint32_t>(x)});

        if (vec.size() >= kLocalSpoolFlushThreshold) {
            flushChunk(chunk);
        }
    }

    void flushAll() {
        for (auto& kv : m_buffers) {
            flushChunk(kv.first);
        }
    }

private:
    void flushChunk(const ChunkIndex& chunk) {
        auto it = m_buffers.find(chunk);
        if (it == m_buffers.end() || it->second.empty()) return;

        m_manager.appendChunkCoords(chunk, it->second);
        it->second.clear();
    }

    SpoolManager& m_manager;
    Shape3 m_chunkShape;
    Shape3 m_volumeShape;

    std::unordered_map<ChunkIndex,
                       std::vector<VoxelCoord32>,
                       ChunkIndexHash> m_buffers;
};

static std::vector<size_t> toVector(const Shape3& s) {
    return {s[0], s[1], s[2]};
}

static bool hasTifxyzInDir(const fs::path& path) {
    return fs::is_directory(path)
        && fs::is_regular_file(path / "x.tif")
        && fs::is_regular_file(path / "y.tif")
        && fs::is_regular_file(path / "z.tif");
}

static double parseVoxelSizeFromReference(const fs::path& ref) {
    std::vector<fs::path> candidates = {
        ref / "meta.json",
        ref / "0" / "meta.json"
    };
    if (ref.has_parent_path()) {
        candidates.emplace_back(ref.parent_path() / "meta.json");
    }

    for (const auto& metaPath : candidates) {
        if (!std::filesystem::is_regular_file(metaPath)) {
            continue;
        }

        try {
            std::ifstream in(metaPath);
            if (!in) {
                continue;
            }
            const auto meta = nlohmann::json::parse(in);
            const auto it = meta.find("voxelsize");
            if (it != meta.end() && it->is_number()) {
                const double vox = it->get<double>();
                if (std::isfinite(vox) && vox > 0.0) {
                    return vox;
                }
            }
        } catch (...) {
            // intentionally ignored; fallback below
        }
    }

    return 0.0;
}

static std::vector<fs::path> discoverTifxyzMeshes(const fs::path& inputRoot) {
    std::vector<fs::path> meshes;
    std::unordered_set<std::string> seen;

    if (!fs::is_directory(inputRoot)) {
        throw std::runtime_error("input is not a directory: " + inputRoot.string());
    }

    if (hasTifxyzInDir(inputRoot)) {
        const auto key = fs::weakly_canonical(inputRoot).string();
        if (seen.insert(key).second) meshes.push_back(inputRoot);
    }

    fs::recursive_directory_iterator it(inputRoot), end;
    for (; it != end; ++it) {
        if (!it->is_directory()) continue;
        if (hasTifxyzInDir(it->path())) {
            const auto key = fs::weakly_canonical(it->path()).string();
            if (seen.insert(key).second) {
                meshes.push_back(it->path());
            }
        }
    }

    std::sort(meshes.begin(), meshes.end());
    return meshes;
}

static Shape3 parseShapeFromReference(const fs::path& ref) {
    fs::path ds = ref;
    if (fs::exists(ref / "0") && fs::is_directory(ref / "0") &&
        fs::exists(ref / "0" / ".zarray")) {
        ds = ref / "0";
    } else if (fs::exists(ref / ".zarray")) {
        ds = ref;
    } else {
        throw std::runtime_error("reference path is not a zarr array or level-0 array: " + ref.string());
    }

    vc::VcDataset dset(ds);
    const auto& s = dset.shape();
    if (s.size() < 3) {
        throw std::runtime_error("reference zarr must have 3 dimensions (Z,Y,X)");
    }
    return {s[0], s[1], s[2]};
}

static nlohmann::json toJsonArray(const std::vector<std::string>& values) {
    nlohmann::json result = nlohmann::json::array();
    for (const auto& value : values) {
        result.push_back(value);
    }
    return result;
}

static bool chunkHasAnyNonZero(const std::vector<uint8_t>& data) {
    return std::any_of(data.begin(), data.end(), [](uint8_t v) { return v != 0; });
}

static std::vector<ChunkIndex> buildTouchedParents(const std::vector<ChunkIndex>& sourceTouched,
                                                  const Shape3& sourceShape,
                                                  const Shape3& sourceChunk,
                                                  const Shape3& targetShape,
                                                  const Shape3& targetChunk) {
    if (sourceTouched.empty()) return {};

    const size_t targetChunksZ = (targetShape[0] + targetChunk[0] - 1) / targetChunk[0];
    const size_t targetChunksY = (targetShape[1] + targetChunk[1] - 1) / targetChunk[1];
    const size_t targetChunksX = (targetShape[2] + targetChunk[2] - 1) / targetChunk[2];

    std::unordered_set<ChunkIndex, ChunkIndexHash> touched;

    for (const auto& src : sourceTouched) {
        const size_t srcZ0 = static_cast<size_t>(src.z) * sourceChunk[0];
        const size_t srcY0 = static_cast<size_t>(src.y) * sourceChunk[1];
        const size_t srcX0 = static_cast<size_t>(src.x) * sourceChunk[2];

        if (srcZ0 >= sourceShape[0] || srcY0 >= sourceShape[1] || srcX0 >= sourceShape[2]) {
            continue;
        }

        const size_t srcZ1 = std::min(sourceShape[0] - 1, srcZ0 + sourceChunk[0] - 1);
        const size_t srcY1 = std::min(sourceShape[1] - 1, srcY0 + sourceChunk[1] - 1);
        const size_t srcX1 = std::min(sourceShape[2] - 1, srcX0 + sourceChunk[2] - 1);

        const size_t tgtZMin = (srcZ0 / 2) / targetChunk[0];
        const size_t tgtZMax = (srcZ1 / 2) / targetChunk[0];
        const size_t tgtYMin = (srcY0 / 2) / targetChunk[1];
        const size_t tgtYMax = (srcY1 / 2) / targetChunk[1];
        const size_t tgtXMin = (srcX0 / 2) / targetChunk[2];
        const size_t tgtXMax = (srcX1 / 2) / targetChunk[2];

        for (size_t z = tgtZMin; z <= tgtZMax; ++z) {
            if (z >= targetChunksZ) break;
            for (size_t y = tgtYMin; y <= tgtYMax; ++y) {
                if (y >= targetChunksY) break;
                for (size_t x = tgtXMin; x <= tgtXMax; ++x) {
                    if (x >= targetChunksX) break;
                    touched.insert({static_cast<uint32_t>(z), static_cast<uint32_t>(y), static_cast<uint32_t>(x)});
                }
            }
        }
    }

    std::vector<ChunkIndex> result;
    result.reserve(touched.size());
    for (const auto& c : touched) {
        result.push_back(c);
    }
    std::sort(result.begin(), result.end(), chunkIndexLess);
    return result;
}

static void downsampleNearestBinary(const uint8_t* src,
                                   size_t srcY, size_t srcX,
                                   uint8_t* dst,
                                   size_t dstZ, size_t dstY, size_t dstX,
                                   size_t srcActualZ, size_t srcActualY, size_t srcActualX) {
    if (srcActualZ == 0 || srcActualY == 0 || srcActualX == 0) return;
    const size_t srcStrideY = srcX;
    const size_t srcStrideZ = srcY * srcX;

    for (size_t zz = 0; zz < dstZ; ++zz) {
        if (2 * zz >= srcActualZ) break;
        for (size_t yy = 0; yy < dstY; ++yy) {
            if (2 * yy >= srcActualY) break;
            for (size_t xx = 0; xx < dstX; ++xx) {
                if (2 * xx >= srcActualX) break;
                dst[zz * dstY * dstX + yy * dstX + xx] =
                    src[(2 * zz) * srcStrideZ + (2 * yy) * srcStrideY + (2 * xx)];
            }
        }
    }
}

struct SegmentPt {
    float x;
    float y;
};

template<typename EmitVoxel>
static void rasterizeLineToShape(int z, float x0, float y0,
                                float x1, float y1,
                                const Shape3& shape,
                                EmitVoxel&& emit) {
    const float dx = x1 - x0;
    const float dy = y1 - y0;
    const float maxDelta = std::max(std::fabs(dx), std::fabs(dy));
    const int steps = static_cast<int>(std::floor(maxDelta + 0.5f));

    const float sx = (steps > 0) ? (dx / std::max(1.0f, static_cast<float>(steps))) : 0.0f;
    const float sy = (steps > 0) ? (dy / std::max(1.0f, static_cast<float>(steps))) : 0.0f;

    if (steps <= 0) {
        const int iy = static_cast<int>(std::llround(y0));
        const int ix = static_cast<int>(std::llround(x0));
        if (iy >= 0 && iy < static_cast<int>(shape[1]) &&
            ix >= 0 && ix < static_cast<int>(shape[2])) {
            emit(size_t(z), size_t(iy), size_t(ix));
        }
        return;
    }

    float fx = x0;
    float fy = y0;
    for (int i = 0; i <= steps; ++i) {
        const int iy = static_cast<int>(std::llround(fy));
        const int ix = static_cast<int>(std::llround(fx));
        if (iy >= 0 && iy < static_cast<int>(shape[1]) &&
            ix >= 0 && ix < static_cast<int>(shape[2])) {
            emit(size_t(z), size_t(iy), size_t(ix));
        }
        fx += sx;
        fy += sy;
    }
}

static bool addSegmentPoint(std::array<SegmentPt, 4>& pts, int& n, float x, float y) {
    for (int i = 0; i < n; ++i) {
        const float dx = pts[i].x - x;
        const float dy = pts[i].y - y;
        if (dx * dx + dy * dy < kSegmentEps2) {
            return false;
        }
    }
    if (n < static_cast<int>(pts.size())) {
        pts[n++] = {x, y};
        return true;
    }
    return false;
}

template<typename EmitVoxel>
static void rasterizeTriangleToSpool(const cv::Vec3f& a,
                                    const cv::Vec3f& b,
                                    const cv::Vec3f& c,
                                    const Shape3& shape,
                                    EmitVoxel&& emit) {
    if (!std::isfinite(a[2]) || !std::isfinite(b[2]) || !std::isfinite(c[2])) return;

    float zMin = std::min({a[2], b[2], c[2]});
    float zMax = std::max({a[2], b[2], c[2]});

    int zStart = static_cast<int>(std::floor(zMin + 1e-6f));
    int zEnd = static_cast<int>(std::floor(zMax + 1e-6f));

    if (zEnd < 0 || zStart >= static_cast<int>(shape[0])) return;
    zStart = std::max(0, zStart);
    zEnd = std::min(zEnd, static_cast<int>(shape[0] - 1));

    for (int zi = zStart; zi <= zEnd; ++zi) {
        const float zPlane = static_cast<float>(zi);
        std::array<SegmentPt, 4> pts{};
        int n = 0;

        auto visitEdge = [&](const cv::Vec3f& p0, const cv::Vec3f& p1) {
            if (!std::isfinite(p0[2]) || !std::isfinite(p1[2])) return;
            float dz = p1[2] - p0[2];
            if (std::fabs(dz) <= kAxisPlaneEps) {
                if (std::fabs(p0[2] - zPlane) <= kAxisPlaneEps) {
                    addSegmentPoint(pts, n, p0[0], p0[1]);
                    addSegmentPoint(pts, n, p1[0], p1[1]);
                }
                return;
            }

            float t = (zPlane - p0[2]) / dz;
            if (t < -kAxisPlaneEps || t > 1.0f + kAxisPlaneEps) return;
            t = std::max(0.0f, std::min(1.0f, t));
            const float x = p0[0] + t * (p1[0] - p0[0]);
            const float y = p0[1] + t * (p1[1] - p0[1]);
            addSegmentPoint(pts, n, x, y);
        };

        visitEdge(a, b);
        visitEdge(b, c);
        visitEdge(c, a);

        if (n < 2) continue;
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                rasterizeLineToShape(zi, pts[i].x, pts[i].y,
                                    pts[j].x, pts[j].y, shape, emit);
            }
        }
    }
}

template<typename EmitVoxel>
static void rasterizeTifxyzMeshToSpool(const fs::path& meshPath,
                                       const Shape3& shape,
                                       EmitVoxel&& emit) {
    auto surf = load_quad_from_tifxyz(meshPath.string());
    const auto& pts = surf->rawPoints();
    if (pts.empty()) return;

    for (auto [r, c, p00, p01, p10, p11] : surf->validQuads()) {
        (void)r; (void)c;
        rasterizeTriangleToSpool(p00, p01, p10, shape, emit);
        rasterizeTriangleToSpool(p10, p01, p11, shape, emit);
    }
}

static void rasterizePhase(const std::vector<fs::path>& meshes,
                          size_t start,
                          size_t stride,
                          const Shape3& shape,
                          SpoolManager& spool,
                          std::atomic<bool>& hadError,
                          std::atomic<size_t>& rasterizedCount,
                          bool verbose,
                          std::mutex& outMutex,
                          size_t meshCount) {
    ThreadSpoolBuffer buffer(spool, spool.chunkShape(), spool.volumeShape());

    for (size_t i = start; i < meshes.size(); i += stride) {
        if (hadError.load()) return;

        try {
            rasterizeTifxyzMeshToSpool(meshes[i], shape,
                [&](size_t z, size_t y, size_t x) {
                    buffer.emit(z, y, x);
                });
        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lk(outMutex);
            std::cerr << "mesh rasterization failed for " << meshes[i]
                      << ": " << e.what() << "\n";
            hadError.store(true);
            return;
        }

        const size_t done = rasterizedCount.fetch_add(1) + 1;
        if (verbose && (done == 1 || (done % 32) == 0 || done == meshCount)) {
            std::lock_guard<std::mutex> lk(outMutex);
            std::cerr << "\r[rasterize] " << done << "/" << meshCount << " meshes\n";
            if (done == meshCount) {
                std::cerr << "\r[rasterize] complete\n";
            }
        }
    }

    buffer.flushAll();
}

static void materializeChunk(const ChunkIndex& chunk,
                            const Shape3& shape,
                            const Shape3& chunkShape,
                            const fs::path& level0Dir,
                            SpoolManager& spool) {
    const std::vector<size_t> chunkShapeVec = {chunkShape[0], chunkShape[1], chunkShape[2]};
    const size_t chunkElements = chunkShapeVec[0] * chunkShapeVec[1] * chunkShapeVec[2];
    std::vector<VoxelCoord32> coords;
    if (!spool.readChunkCoords(chunk, coords) || coords.empty()) {
        return;
    }

    std::vector<uint8_t> chunkBuf(chunkElements, 0);
    const size_t chunkBaseZ = static_cast<size_t>(chunk.z) * chunkShape[0];
    const size_t chunkBaseY = static_cast<size_t>(chunk.y) * chunkShape[1];
    const size_t chunkBaseX = static_cast<size_t>(chunk.x) * chunkShape[2];

    for (const auto& p : coords) {
        if (p.z >= shape[0] || p.y >= shape[1] || p.x >= shape[2]) {
            continue;
        }
        const size_t lz = static_cast<size_t>(p.z) - chunkBaseZ;
        const size_t ly = static_cast<size_t>(p.y) - chunkBaseY;
        const size_t lx = static_cast<size_t>(p.x) - chunkBaseX;

        if (lz < chunkShape[0] && ly < chunkShape[1] && lx < chunkShape[2]) {
                    chunkBuf[lz * chunkShape[1] * chunkShape[2] + ly * chunkShape[2] + lx] = kSurfaceValue;
        }
    }

    if (!chunkHasAnyNonZero(chunkBuf)) return;

    vc::VcDataset ds(level0Dir);
    ds.writeChunk(chunk.z, chunk.y, chunk.x,
                 chunkBuf.data(), chunkBuf.size() * sizeof(uint8_t));
}

static void materializePhase(const std::vector<ChunkIndex>& touched,
                            const Shape3& shape,
                            const Shape3& chunkShape,
                            const fs::path& level0Dir,
                            SpoolManager& spool,
                            size_t numThreads,
                            std::atomic<bool>& hadError) {
    if (touched.empty()) return;

    std::atomic<size_t> nextIndex{0};
    std::vector<std::thread> threads;
    threads.reserve(numThreads);

    for (size_t tid = 0; tid < numThreads; ++tid) {
        threads.emplace_back([&, tid]() {
            (void)tid;
            while (!hadError.load()) {
                const size_t idx = nextIndex.fetch_add(1);
                if (idx >= touched.size()) break;

                try {
                    materializeChunk(touched[idx], shape, chunkShape, level0Dir, spool);
                } catch (const std::exception& e) {
                    std::cerr << "materialization failed for chunk ["
                              << touched[idx].z << ", "
                              << touched[idx].y << ", "
                              << touched[idx].x << "]: " << e.what() << "\n";
                    hadError.store(true);
                }
            }
        });
    }
    for (auto& t : threads) t.join();
}

static std::vector<ChunkIndex> buildIsotropicPyramidLevel(const fs::path& outDir,
                                                         int level,
                                                         const std::vector<ChunkIndex>& sourceTouched,
                                                         size_t numThreads,
                                                         std::atomic<bool>& hadError) {
    const fs::path srcPath = outDir / std::to_string(level - 1);
    const fs::path dstPath = outDir / std::to_string(level);

    vc::VcDataset src(srcPath);
    vc::VcDataset dst(dstPath);

    const auto& srcShape = src.shape();
    const auto& srcChunk = src.defaultChunkShape();
    const auto& dstShape = dst.shape();
    const auto& dstChunk = dst.defaultChunkShape();
    const Shape3 srcShape3 = {srcShape[0], srcShape[1], srcShape[2]};
    const Shape3 srcChunk3 = {srcChunk[0], srcChunk[1], srcChunk[2]};
    const Shape3 dstShape3 = {dstShape[0], dstShape[1], dstShape[2]};
    const Shape3 dstChunk3 = {dstChunk[0], dstChunk[1], dstChunk[2]};

    const auto touchedChunks = buildTouchedParents(sourceTouched, srcShape3, srcChunk3,
                                                  dstShape3, dstChunk3);
    if (touchedChunks.empty()) return {};
    const size_t totalChunks = touchedChunks.size();

    std::vector<ChunkIndex> outputTouched;
    outputTouched.reserve(totalChunks);
    std::mutex outTouchedMutex;

    const size_t dstChunkElements = dstChunk3[0] * dstChunk3[1] * dstChunk3[2];

    const size_t dstChunkY = dstChunk3[1];
    const size_t dstChunkX = dstChunk3[2];
    const size_t dstChunkZ = dstChunk3[0];

    std::atomic<size_t> nextChunk{0};

    std::vector<std::thread> threads;
    threads.reserve(numThreads);

    for (size_t tid = 0; tid < numThreads; ++tid) {
        threads.emplace_back([&, tid]() {
            (void)tid;
            vc::VcDataset srcLocal(srcPath);
            vc::VcDataset dstLocal(dstPath);
            std::vector<uint8_t> dstBuf(dstChunkElements, 0);

            while (!hadError.load()) {
                const size_t idx = nextChunk.fetch_add(1);
                if (idx >= totalChunks) break;

                const auto chunk = touchedChunks[idx];
                const size_t cz = chunk.z;
                const size_t cy = chunk.y;
                const size_t cx = chunk.x;

                const size_t srcZ0 = cz * dstChunkZ * 2;
                const size_t srcY0 = cy * dstChunkY * 2;
                const size_t srcX0 = cx * dstChunkX * 2;

                const size_t srcActualZ = srcZ0 < srcShape3[0]
                    ? std::min(srcShape3[0] - srcZ0, dstChunkZ * 2)
                    : 0;
                const size_t srcActualY = srcY0 < srcShape3[1]
                    ? std::min(srcShape3[1] - srcY0, dstChunkY * 2)
                    : 0;
                const size_t srcActualX = srcX0 < srcShape3[2]
                    ? std::min(srcShape3[2] - srcX0, dstChunkX * 2)
                    : 0;

                std::vector<uint8_t> srcBuf(srcActualZ * srcActualY * srcActualX, 0);
                if (!srcBuf.empty()) {
                    srcLocal.readRegion({srcZ0, srcY0, srcX0},
                                       {srcActualZ, srcActualY, srcActualX},
                                       srcBuf.data());
                }

                std::fill(dstBuf.begin(), dstBuf.end(), 0);
                if (!srcBuf.empty()) {
                    downsampleNearestBinary(srcBuf.data(), srcActualY, srcActualX,
                                            dstBuf.data(), dstChunkZ, dstChunkY, dstChunkX,
                                            srcActualZ, srcActualY, srcActualX);
                }

                if (chunkHasAnyNonZero(dstBuf)) {
                    const bool ok = dstLocal.writeChunk(cz, cy, cx,
                                                       dstBuf.data(),
                                                       dstBuf.size() * sizeof(uint8_t));
                    if (!ok) {
                        hadError.store(true);
                        return;
                    }
                    std::lock_guard<std::mutex> lk(outTouchedMutex);
                    outputTouched.push_back(chunk);
                }
            }
        });
    }

    for (auto& t : threads) t.join();
    return outputTouched;
}

static void writeSparseAttrs(const fs::path& outDir,
                            const Shape3& level0Shape,
                            const Shape3& level0Chunk,
                            const fs::path* sourceRef,
                            int sourceGroup,
                            size_t maxLevel) {
    nlohmann::json attrs;
    attrs["note_axes_order"] = "ZYX (slice, row, col)";
    attrs["num_slices"] = level0Shape[0];
    attrs["chunk_size"] = {static_cast<long long>(level0Chunk[0]),
                            static_cast<long long>(level0Chunk[1]),
                            static_cast<long long>(level0Chunk[2])};
    attrs["isotropic_chunks"] = true;
    attrs["pyramid"] = true;
    attrs["pyramid_levels"] = static_cast<int>(maxLevel);
    if (sourceRef) {
        attrs["source_zarr"] = sourceRef->string();
    }
    if (sourceGroup >= 0) {
        attrs["source_group"] = sourceGroup;
    }

    nlohmann::json ms;
    ms["version"] = "0.4";
    ms["name"] = "vc_tifxyz2zarr_sparse";
    ms["axes"] = nlohmann::json::array({
        nlohmann::json{{"name", "z"}, {"type", "space"}},
        nlohmann::json{{"name", "y"}, {"type", "space"}},
        nlohmann::json{{"name", "x"}, {"type", "space"}},
    });

    ms["datasets"] = nlohmann::json::array();
    for (int level = 0; level <= static_cast<int>(maxLevel); ++level) {
        const double scale = std::pow(2.0, static_cast<double>(level));
        ms["datasets"].push_back({
            {"path", std::to_string(level)},
            {"coordinateTransformations", nlohmann::json::array({
                {
                    {"type", "scale"},
                    {"scale", nlohmann::json::array({scale, scale, scale})},
                },
                {
                    {"type", "translation"},
                    {"translation", nlohmann::json::array({0.0, 0.0, 0.0})},
                },
            })}
        });
    }
    ms["metadata"] = {{"downsampling_method", "nearest"},
                       {"chunk_isotropic", true},
                       {"spool_binary", true}};

    attrs["multiscales"] = nlohmann::json::array({ms});

    vc::writeZarrAttributes(outDir, attrs);
}

static void writeVolumeMetadata(const fs::path& outDir,
                               const Shape3& level0Shape,
                               const std::optional<fs::path>& sourceRef,
                               int sourceGroup,
                               double voxelSize,
                               const std::vector<std::string>& sourceSegments,
                               const std::vector<std::string>& sourceMeshes) {
    const std::string uuid = outDir.filename().string();
    nlohmann::json meta;
    meta["type"] = "vol";
    meta["uuid"] = uuid;
    meta["name"] = uuid;
    meta["width"] = static_cast<long long>(level0Shape[2]);
    meta["height"] = static_cast<long long>(level0Shape[1]);
    meta["slices"] = static_cast<long long>(level0Shape[0]);
    meta["voxelsize"] = voxelSize;
    meta["min"] = 0.0;
    meta["max"] = 255.0;
    meta["format"] = "zarr";

    if (sourceRef) {
        meta["source_zarr"] = sourceRef->string();
    }
    if (sourceGroup >= 0) {
        meta["source_group"] = sourceGroup;
    }
    if (!sourceSegments.empty()) {
        meta["source_segments"] = toJsonArray(sourceSegments);
    }
    if (!sourceMeshes.empty()) {
        meta["source_meshes"] = toJsonArray(sourceMeshes);
    }
    if (!sourceSegments.empty() || !sourceMeshes.empty()) {
        meta["source_mesh_count"] = static_cast<long long>(
            !sourceSegments.empty() ? sourceSegments.size() : sourceMeshes.size());
    }
    meta["rasterizer"] = "vc_tifxyz2zarr_sparse";

    const fs::path metaPath = outDir / "meta.json";
    std::ofstream metaOut(metaPath);
    if (!metaOut) {
        throw std::runtime_error("failed creating meta.json: " + metaPath.string());
    }
    metaOut << meta.dump(2) << "\n";
    if (!metaOut) {
        throw std::runtime_error("failed writing meta.json: " + metaPath.string());
    }
}

} // namespace

int main(int argc, char* argv[])
{
    if (argc == 2 && (std::string(argv[1]) == "-h" ||
                      std::string(argv[1]) == "--help")) {
        std::cout << "vc_tifxyz2zarr_sparse: rasterize tifxyz quads into sparse OME-Zarr\n"
                  << "\n"
                  << "Usage:\n"
                  << "  " << argv[0]
                  << " <input_tifxyz_dir> <output_omezarr> [options]\n"
                  << "\n"
                  << "Options:\n"
                  << "  Surface voxels are written as 255 in uint8 output.\n"
                  << "  --shape-z N               full output Z\n"
                  << "  --shape-y N               full output Y\n"
                  << "  --shape-x N               full output X\n"
                  << "  --reference-zarr PATH      infer shape from existing zarr (uses /0 if present)\n"
                  << "  --chunk-size N             isotropic chunk edge (default 128)\n"
                  << "  --source-segment SEGMENT    optional segment ID provenance (repeatable)\n"
                  << "  --source-mesh PATH         optional mesh path provenance (repeatable)\n"
                  << "  --spool-memory-mb N       in-memory spool budget in MiB (default 4096, 0 = immediate disk append)\n"
                  << "  --threads N                raster + materialize threads (default hw)\n"
                  << "  --spool-dir PATH           spool directory (default <output>/.spool)\n"
                  << "  --keep-spool               keep binary spool files\n"
                  << "  --overwrite                overwrite output directory if exists\n"
                  << "  --source-group IDX          optional source group for metadata\n"
                  << "\n"
                  << "Note:\n"
                  << "  --chunk-size-z/x/y are intentionally not supported; use --chunk-size.\n";
        return EXIT_SUCCESS;
    }

    try {
        po::options_description desc("Options");
        desc.add_options()
            ("help,h", "Show help")
            ("input", po::value<std::string>(), "Input folder containing tifxyz meshes")
            ("output", po::value<std::string>(), "Output OME-Zarr root")
            ("shape-z", po::value<size_t>(), "Output Z size")
            ("shape-y", po::value<size_t>(), "Output Y size")
            ("shape-x", po::value<size_t>(), "Output X size")
            ("reference-zarr", po::value<std::string>(), "Reference OME-Zarr for shape inference")
            ("chunk-size", po::value<size_t>()->default_value(128), "Isotropic chunk size")
            ("source-segment", po::value<std::vector<std::string>>()->multitoken(),
             "Segment ID used for provenance (repeatable)")
            ("source-mesh", po::value<std::vector<std::string>>()->multitoken(),
             "Mesh path used for provenance (repeatable)")
            ("spool-memory-mb", po::value<size_t>()->default_value(4096),
             "In-memory spool budget in MiB; 0 disables RAM buffering")
            ("threads,t", po::value<int>()->default_value(0), "Worker threads (raster/materialize/pyramid)")
            ("spool-dir", po::value<std::string>(), "Spool directory")
            ("source-group", po::value<int>()->default_value(-1), "Optional source group index in metadata")
            ("keep-spool", po::bool_switch()->default_value(false), "Keep binary spool directory")
            ("overwrite", po::bool_switch()->default_value(false), "Overwrite existing output path")
            ("verbose,v", po::bool_switch()->default_value(false), "Print progress")
            ("chunk-size-z", po::value<size_t>(), "DEPRECATED: ignored; use --chunk-size")
            ("chunk-size-y", po::value<size_t>(), "DEPRECATED: ignored; use --chunk-size")
            ("chunk-size-x", po::value<size_t>(), "DEPRECATED: ignored; use --chunk-size");

        po::positional_options_description pos;
        pos.add("input", 1);
        pos.add("output", 1);

        po::variables_map vm;
        auto parsed = po::command_line_parser(argc, argv)
                          .options(desc)
                          .positional(pos)
                          .run();
        po::store(parsed, vm);
        po::notify(vm);

        if (vm.count("help") || !vm.count("input") || !vm.count("output")) {
            std::cerr << desc << "\n";
            return EXIT_FAILURE;
        }

        const bool hasChunkAxisFlags = vm.count("chunk-size-z") ||
                                      vm.count("chunk-size-y") ||
                                      vm.count("chunk-size-x");
        if (hasChunkAxisFlags) {
            std::cerr << "strict isotropy is enforced; --chunk-size-z/x/y are unsupported.\n"
                      << "Use --chunk-size instead.\n";
            return EXIT_FAILURE;
        }

        const fs::path inputRoot = vm["input"].as<std::string>();
        const fs::path outRoot = vm["output"].as<std::string>();
        const bool overwrite = vm["overwrite"].as<bool>();
        const bool keepSpool = vm["keep-spool"].as<bool>();
        const bool verbose = vm["verbose"].as<bool>();
        const int sourceGroup = vm["source-group"].as<int>();
        const size_t spoolMemMB = vm["spool-memory-mb"].as<size_t>();
        const auto sourceSegments = vm.count("source-segment")
                                       ? vm["source-segment"].as<std::vector<std::string>>()
                                       : std::vector<std::string>{};
        const auto sourceMeshes = vm.count("source-mesh")
                                     ? vm["source-mesh"].as<std::vector<std::string>>()
                                     : std::vector<std::string>{};

        const size_t chunkSize = vm["chunk-size"].as<size_t>();
        if (chunkSize == 0) throw std::runtime_error("--chunk-size must be > 0");
        if (spoolMemMB > (std::numeric_limits<size_t>::max() / (1024ull * 1024ull))) {
            throw std::runtime_error("spool-memory-mb too large");
        }
        const size_t inMemoryBudgetBytes = spoolMemMB * 1024ull * 1024ull;
        double voxelSize = 0.0;

        const bool hasShape = vm.count("shape-z") || vm.count("shape-y") || vm.count("shape-x");
        if (hasShape && !(vm.count("shape-z") && vm.count("shape-y") && vm.count("shape-x"))) {
            throw std::runtime_error("if using --shape-z/y/x, all three dimensions must be provided");
        }

        Shape3 shape;
        std::optional<fs::path> sourceRef;
        if (vm.count("shape-z")) {
            shape = {vm["shape-z"].as<size_t>(),
                     vm["shape-y"].as<size_t>(),
                     vm["shape-x"].as<size_t>()};
            if (shape[0] == 0 || shape[1] == 0 || shape[2] == 0) {
                throw std::runtime_error("all dimensions must be > 0");
            }
        } else if (vm.count("reference-zarr")) {
            const fs::path ref = vm["reference-zarr"].as<std::string>();
            sourceRef = fs::weakly_canonical(ref);
            shape = parseShapeFromReference(*sourceRef);
            voxelSize = parseVoxelSizeFromReference(*sourceRef);
        } else {
            throw std::runtime_error("provide either --shape-z/y/x or --reference-zarr");
        }

        int threadCount = vm["threads"].as<int>();
        if (threadCount <= 0) {
            threadCount = static_cast<int>(std::max<size_t>(1, std::thread::hardware_concurrency()));
        }
        const size_t numThreads = static_cast<size_t>(threadCount);

        const auto meshes = discoverTifxyzMeshes(inputRoot);
        if (meshes.empty()) {
            throw std::runtime_error("no tifxyz meshes found: " + inputRoot.string());
        }

        if (fs::exists(outRoot)) {
            if (!overwrite) {
                throw std::runtime_error("output exists; use --overwrite to replace: " + outRoot.string());
            }
            fs::remove_all(outRoot);
        }

        fs::path spoolDir = vm.count("spool-dir")
                              ? vm["spool-dir"].as<std::string>()
                              : outRoot.string() + "/." + kChunkSuffix;

        Shape3 levelShape[1 + kPyramidLevels];
        Shape3 levelChunks[1 + kPyramidLevels];
        levelShape[0] = shape;
        levelChunks[0] = {std::min(shape[0], chunkSize),
                          std::min(shape[1], chunkSize),
                          std::min(shape[2], chunkSize)};

        for (size_t l = 1; l <= kPyramidLevels; ++l) {
            const size_t z = (levelShape[l - 1][0] + 1) / 2;
            const size_t y = (levelShape[l - 1][1] + 1) / 2;
            const size_t x = (levelShape[l - 1][2] + 1) / 2;
            levelShape[l] = {z, y, x};
            levelChunks[l] = {std::min(z, chunkSize),
                              std::min(y, chunkSize),
                              std::min(x, chunkSize)};
        }

        fs::create_directories(outRoot);
        vc::createZarrDataset(outRoot, "0", toVector(levelShape[0]),
                              toVector(levelChunks[0]), vc::VcDtype::uint8, "blosc");
        for (size_t l = 1; l <= kPyramidLevels; ++l) {
            vc::createZarrDataset(outRoot, std::to_string(l), toVector(levelShape[l]),
                                  toVector(levelChunks[l]), vc::VcDtype::uint8, "blosc");
        }

        const SpoolCoordType spoolCoordType = chooseSpoolCoordType(shape);
        SpoolManager spool(spoolDir, levelChunks[0], shape, spoolCoordType, inMemoryBudgetBytes);
        if (verbose) {
            std::cerr << "[spool] in-memory budget: "
                      << spoolMemMB << " MiB (" << spool.inMemoryBudgetBytes() << " bytes)\n";
            std::cerr << "[spool] coordinate packing: "
                      << coordTypeName(spoolCoordType)
                      << " (" << spool.recordBytes() << " bytes/voxel)\n";
        }

        std::atomic<bool> hadError{false};
        std::atomic<size_t> rasterizedCount{0};
        std::mutex ioMutex;

        std::vector<std::thread> rasterThreads;
        rasterThreads.reserve(numThreads);
        for (size_t tid = 0; tid < numThreads; ++tid) {
            rasterThreads.emplace_back([&, tid]() {
                rasterizePhase(meshes, tid, numThreads, shape,
                              spool, hadError, rasterizedCount,
                              verbose, ioMutex, meshes.size());
            });
        }
        for (auto& t : rasterThreads) t.join();

        if (hadError.load()) {
            throw std::runtime_error("rasterization phase failed");
        }

        auto touched = spool.touchedChunks();
        if (verbose) {
            std::lock_guard<std::mutex> lk(ioMutex);
            std::cerr << "[rasterize] touched chunks: " << touched.size() << "\n";
        }

        materializePhase(touched, shape, levelChunks[0], outRoot / "0",
                        spool, numThreads, hadError);
        if (hadError.load()) {
            throw std::runtime_error("materialization phase failed");
        }

        auto activeTouched = touched;
        for (size_t level = 1; level <= kPyramidLevels; ++level) {
            if (activeTouched.empty()) {
                if (verbose) {
                    std::lock_guard<std::mutex> lk(ioMutex);
                    std::cerr << "[pyramid] level " << level << " skipped (no active chunks)\n";
                }
                break;
            }
            if (verbose) {
                std::cerr << "[pyramid] building level " << level << "\n";
            }
            activeTouched = buildIsotropicPyramidLevel(outRoot,
                                                     static_cast<int>(level),
                                                     activeTouched,
                                                     numThreads,
                                                     hadError);
            if (verbose) {
                std::lock_guard<std::mutex> lk(ioMutex);
                std::cerr << "[pyramid] level " << level << " touched chunks: "
                          << activeTouched.size() << "\n";
            }
            if (hadError.load()) {
                throw std::runtime_error("pyramid level " + std::to_string(level) + " failed");
            }
        }

        writeSparseAttrs(outRoot, levelShape[0], levelChunks[0],
                        sourceRef ? &(*sourceRef) : nullptr,
                        sourceGroup, kPyramidLevels);
        writeVolumeMetadata(outRoot, levelShape[0], sourceRef, sourceGroup,
                           voxelSize, sourceSegments, sourceMeshes);

        if (!keepSpool) {
            if (fs::exists(spoolDir)) {
                fs::remove_all(spoolDir);
            }
        }

        std::cout << "vc_tifxyz2zarr_sparse completed\n";
        std::cout << "touched chunks: " << touched.size() << "\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
