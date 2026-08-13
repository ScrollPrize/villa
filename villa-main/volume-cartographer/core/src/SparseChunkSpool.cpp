#include "vc/core/util/SparseChunkSpool.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <fstream>
#include <list>
#include <mutex>
#include <stdexcept>
#include <unordered_set>

namespace vc::core::util {

namespace {

constexpr size_t kRecordBytes = 10;
constexpr size_t kLocalFlushThreshold = 4096;

static_assert(sizeof(SparseChunkRecordU8x7) == kRecordBytes,
              "SparseChunkRecordU8x7 must remain tightly packed");

std::vector<uint8_t> packRecords(const std::vector<SparseChunkRecordU8x7>& records)
{
    std::vector<uint8_t> packed(records.size() * kRecordBytes);
    if (!records.empty()) {
        std::memcpy(packed.data(), records.data(), packed.size());
    }
    return packed;
}

void unpackRecords(const uint8_t* data,
                   size_t bytes,
                   std::vector<SparseChunkRecordU8x7>& out)
{
    if (bytes == 0) {
        return;
    }
    if ((bytes % kRecordBytes) != 0) {
        throw std::runtime_error("invalid sparse chunk spool record payload");
    }

    const size_t count = bytes / kRecordBytes;
    const auto oldSize = out.size();
    out.resize(oldSize + count);
    std::memcpy(out.data() + oldSize, data, bytes);
}

} // namespace

size_t SparseChunkIndexHash::operator()(const SparseChunkIndex& c) const noexcept
{
    size_t h = 1469598103934665603ull;
    h ^= static_cast<size_t>(c.z);
    h *= 1099511628211ull;
    h ^= static_cast<size_t>(c.y);
    h *= 1099511628211ull;
    h ^= static_cast<size_t>(c.x);
    h *= 1099511628211ull;
    return h;
}

bool sparseChunkIndexLess(const SparseChunkIndex& a, const SparseChunkIndex& b)
{
    if (a.z != b.z) return a.z < b.z;
    if (a.y != b.y) return a.y < b.y;
    return a.x < b.x;
}

struct SparseChunkSpool::Impl {
    mutable std::mutex memoryMutex;
    size_t inMemoryBytes = 0;
    std::unordered_map<SparseChunkIndex, std::vector<uint8_t>, SparseChunkIndexHash> memorySpool;
    std::list<SparseChunkIndex> memOrder;

    mutable std::mutex touchedMutex;
    std::unordered_set<SparseChunkIndex, SparseChunkIndexHash> touchedChunks;

    mutable std::mutex lockMapMutex;
    std::unordered_map<SparseChunkIndex, std::shared_ptr<std::mutex>, SparseChunkIndexHash> lockMap;

    std::atomic<size_t> spillFiles{0};
    std::atomic<uint64_t> appendedRecords{0};

    std::shared_ptr<std::mutex> lockForChunk(const SparseChunkIndex& chunk)
    {
        std::lock_guard<std::mutex> lock(lockMapMutex);
        auto it = lockMap.find(chunk);
        if (it != lockMap.end()) {
            return it->second;
        }
        auto ptr = std::make_shared<std::mutex>();
        lockMap.emplace(chunk, ptr);
        return ptr;
    }
};

SparseChunkSpool::SparseChunkSpool(std::filesystem::path spoolDir,
                                   const Shape3& chunkShape,
                                   const Shape3& volumeShape,
                                   size_t inMemoryMaxBytes)
    : spoolDir_(std::move(spoolDir)),
      chunkShape_(chunkShape),
      volumeShape_(volumeShape),
      inMemoryMaxBytes_(inMemoryMaxBytes),
      impl_(std::make_unique<Impl>())
{
    if (chunkShape_[0] == 0 || chunkShape_[1] == 0 || chunkShape_[2] == 0) {
        throw std::runtime_error("sparse chunk spool requires non-zero chunk dimensions");
    }
    if (chunkShape_[0] > 255 || chunkShape_[1] > 255 || chunkShape_[2] > 255) {
        throw std::runtime_error("sparse chunk spool currently requires chunk dimensions <= 255");
    }
    std::filesystem::create_directories(spoolDir_);
}

SparseChunkSpool::~SparseChunkSpool() = default;

std::filesystem::path SparseChunkSpool::spoolPathFor(const SparseChunkIndex& chunk) const
{
    return spoolDir_ / (std::to_string(chunk.z) + "_" +
                        std::to_string(chunk.y) + "_" +
                        std::to_string(chunk.x) + ".bin");
}

void SparseChunkSpool::appendChunkRecords(const SparseChunkIndex& chunk,
                                          const std::vector<SparseChunkRecordU8x7>& records)
{
    if (records.empty()) {
        return;
    }

    auto packed = packRecords(records);
    impl_->appendedRecords.fetch_add(records.size(), std::memory_order_relaxed);

    if (inMemoryMaxBytes_ == 0) {
        auto chunkLock = impl_->lockForChunk(chunk);
        std::lock_guard<std::mutex> lock(*chunkLock);
        const auto path = spoolPathFor(chunk);
        const bool existed = std::filesystem::exists(path);
        std::ofstream out(path, std::ios::binary | std::ios::app);
        if (!out) {
            throw std::runtime_error("failed opening sparse chunk spool file: " + path.string());
        }
        out.write(reinterpret_cast<const char*>(packed.data()),
                  static_cast<std::streamsize>(packed.size()));
        if (!out) {
            throw std::runtime_error("failed writing sparse chunk spool file: " + path.string());
        }
        if (!existed) {
            impl_->spillFiles.fetch_add(1, std::memory_order_relaxed);
        }
        std::lock_guard<std::mutex> touchedLock(impl_->touchedMutex);
        impl_->touchedChunks.emplace(chunk);
        return;
    }

    std::vector<std::pair<SparseChunkIndex, std::vector<uint8_t>>> toSpill;
    {
        std::lock_guard<std::mutex> lock(impl_->memoryMutex);
        auto it = impl_->memorySpool.find(chunk);
        auto& memVec = (it == impl_->memorySpool.end())
            ? impl_->memorySpool.emplace(chunk, std::vector<uint8_t>{}).first->second
            : it->second;
        if (it == impl_->memorySpool.end()) {
            impl_->memOrder.push_back(chunk);
        }

        memVec.insert(memVec.end(), packed.begin(), packed.end());
        impl_->inMemoryBytes += packed.size();

        while (impl_->inMemoryBytes > inMemoryMaxBytes_ && !impl_->memOrder.empty()) {
            const auto victim = impl_->memOrder.front();
            const auto victimIt = impl_->memorySpool.find(victim);
            impl_->memOrder.pop_front();
            if (victimIt == impl_->memorySpool.end()) {
                continue;
            }

            impl_->inMemoryBytes -= victimIt->second.size();
            toSpill.emplace_back(victim, std::move(victimIt->second));
            impl_->memorySpool.erase(victimIt);
        }
    }

    for (auto& [victim, bytes] : toSpill) {
        if (bytes.empty()) {
            continue;
        }
        auto chunkLock = impl_->lockForChunk(victim);
        std::lock_guard<std::mutex> lock(*chunkLock);
        const auto path = spoolPathFor(victim);
        const bool existed = std::filesystem::exists(path);
        std::ofstream out(path, std::ios::binary | std::ios::app);
        if (!out) {
            throw std::runtime_error("failed opening sparse chunk spool file: " + path.string());
        }
        out.write(reinterpret_cast<const char*>(bytes.data()),
                  static_cast<std::streamsize>(bytes.size()));
        if (!out) {
            throw std::runtime_error("failed writing sparse chunk spool file: " + path.string());
        }
        if (!existed) {
            impl_->spillFiles.fetch_add(1, std::memory_order_relaxed);
        }
    }

    std::lock_guard<std::mutex> touchedLock(impl_->touchedMutex);
    impl_->touchedChunks.emplace(chunk);
}

bool SparseChunkSpool::readChunkRecords(const SparseChunkIndex& chunk,
                                        std::vector<SparseChunkRecordU8x7>& out) const
{
    const auto startSize = out.size();
    const auto path = spoolPathFor(chunk);

    if (std::filesystem::exists(path)) {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            throw std::runtime_error("failed opening sparse chunk spool file for reading: " + path.string());
        }
        in.seekg(0, std::ios::end);
        const auto bytes = static_cast<size_t>(in.tellg());
        in.seekg(0, std::ios::beg);
        std::vector<uint8_t> fileBytes(bytes);
        if (bytes > 0) {
            in.read(reinterpret_cast<char*>(fileBytes.data()), static_cast<std::streamsize>(bytes));
            if (!in) {
                throw std::runtime_error("failed reading sparse chunk spool file: " + path.string());
            }
            unpackRecords(fileBytes.data(), fileBytes.size(), out);
        }
    }

    {
        std::lock_guard<std::mutex> lock(impl_->memoryMutex);
        const auto it = impl_->memorySpool.find(chunk);
        if (it != impl_->memorySpool.end() && !it->second.empty()) {
            unpackRecords(it->second.data(), it->second.size(), out);
        }
    }

    return out.size() != startSize;
}

std::vector<SparseChunkIndex> SparseChunkSpool::touchedChunks() const
{
    std::lock_guard<std::mutex> lock(impl_->touchedMutex);
    std::vector<SparseChunkIndex> touched;
    touched.reserve(impl_->touchedChunks.size());
    for (const auto& chunk : impl_->touchedChunks) {
        touched.push_back(chunk);
    }
    std::sort(touched.begin(), touched.end(), sparseChunkIndexLess);
    return touched;
}

SparseChunkSpoolStats SparseChunkSpool::stats() const
{
    SparseChunkSpoolStats stats;
    {
        std::lock_guard<std::mutex> lock(impl_->touchedMutex);
        stats.touchedChunks = impl_->touchedChunks.size();
    }
    {
        std::lock_guard<std::mutex> lock(impl_->memoryMutex);
        stats.inMemoryBytes = impl_->inMemoryBytes;
    }
    stats.spillFiles = impl_->spillFiles.load(std::memory_order_relaxed);
    stats.appendedRecords = impl_->appendedRecords.load(std::memory_order_relaxed);
    stats.inMemoryBudgetBytes = inMemoryMaxBytes_;
    return stats;
}

SparseChunkSpoolBuffer::SparseChunkSpoolBuffer(SparseChunkSpool& spool)
    : spool_(&spool)
{
}

void SparseChunkSpoolBuffer::emit(size_t z,
                                  size_t y,
                                  size_t x,
                                  const std::array<uint8_t, 7>& values)
{
    if (z >= spool_->volumeShape()[0] || y >= spool_->volumeShape()[1] || x >= spool_->volumeShape()[2]) {
        return;
    }

    const SparseChunkIndex chunk{
        static_cast<uint32_t>(z / spool_->chunkShape()[0]),
        static_cast<uint32_t>(y / spool_->chunkShape()[1]),
        static_cast<uint32_t>(x / spool_->chunkShape()[2]),
    };

    auto& vec = buffers_[chunk];
    vec.push_back(SparseChunkRecordU8x7{
        static_cast<uint8_t>(z % spool_->chunkShape()[0]),
        static_cast<uint8_t>(y % spool_->chunkShape()[1]),
        static_cast<uint8_t>(x % spool_->chunkShape()[2]),
        values,
    });

    if (vec.size() >= kLocalFlushThreshold) {
        flushChunk(chunk);
    }
}

void SparseChunkSpoolBuffer::flushAll()
{
    for (auto& [chunk, records] : buffers_) {
        if (!records.empty()) {
            spool_->appendChunkRecords(chunk, records);
            records.clear();
        }
    }
}

void SparseChunkSpoolBuffer::flushChunk(const SparseChunkIndex& chunk)
{
    auto it = buffers_.find(chunk);
    if (it == buffers_.end() || it->second.empty()) {
        return;
    }
    spool_->appendChunkRecords(chunk, it->second);
    it->second.clear();
}

} // namespace vc::core::util
