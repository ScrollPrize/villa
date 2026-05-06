#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include <opencv2/core.hpp>
#include <vc/core/types/Array3D.hpp>
#include <vc/core/types/Volume.hpp>

namespace vc::core::util {

enum class NormalGridSliceDirection { XY, XZ, YZ };

struct NormalGridBatchPlan {
    size_t bytesPerSlice = 0;
    size_t chunkSizeTarget = 1;
    size_t estimatedBatchBytes = 0;
};

struct NormalGridSampledSlice {
    size_t sliceIndex = 0;
    size_t localSliceIndex = 0;
};

struct NormalGridSampledChunkPlan {
    size_t sourceChunkIndex = 0;
    size_t sourceSliceBegin = 0;
    size_t sourceSliceCount = 0;
    std::vector<NormalGridSampledSlice> sampledSlices;
};

inline size_t normalGridSliceAxis(NormalGridSliceDirection direction)
{
    switch (direction) {
    case NormalGridSliceDirection::XY: return 0;
    case NormalGridSliceDirection::XZ: return 1;
    case NormalGridSliceDirection::YZ: return 2;
    }
    return 0;
}

inline cv::Size normalGridSliceSize(
    const std::vector<size_t>& shape,
    NormalGridSliceDirection direction)
{
    if (shape.size() != 3) {
        throw std::runtime_error("normalGridSliceSize expects 3D ZYX shape");
    }

    switch (direction) {
    case NormalGridSliceDirection::XY:
        return cv::Size(static_cast<int>(shape[2]), static_cast<int>(shape[1]));
    case NormalGridSliceDirection::XZ:
        return cv::Size(static_cast<int>(shape[2]), static_cast<int>(shape[0]));
    case NormalGridSliceDirection::YZ:
        return cv::Size(static_cast<int>(shape[1]), static_cast<int>(shape[0]));
    }
    return cv::Size();
}

inline std::vector<NormalGridSampledChunkPlan> planNormalGridSampledChunks(
    const std::vector<size_t>& shape,
    const std::vector<size_t>& sourceChunkShape,
    NormalGridSliceDirection direction,
    int sparseVolume)
{
    if (shape.size() != 3 || sourceChunkShape.size() != 3) {
        throw std::runtime_error("planNormalGridSampledChunks expects 3D ZYX shape/chunkShape");
    }

    const size_t sliceAxis = normalGridSliceAxis(direction);
    const size_t numSlices = shape[sliceAxis];
    const size_t chunkDepth = sourceChunkShape[sliceAxis];
    if (chunkDepth == 0) {
        throw std::runtime_error("source chunk depth must be > 0");
    }

    const size_t sampleStep = static_cast<size_t>(std::max(1, sparseVolume));
    const size_t numSourceChunks = (numSlices + chunkDepth - 1) / chunkDepth;

    std::vector<NormalGridSampledChunkPlan> plans;
    plans.reserve(numSourceChunks);

    for (size_t sourceChunkIndex = 0; sourceChunkIndex < numSourceChunks; ++sourceChunkIndex) {
        const size_t sourceSliceBegin = sourceChunkIndex * chunkDepth;
        const size_t sourceSliceEnd = std::min(numSlices, sourceSliceBegin + chunkDepth);

        NormalGridSampledChunkPlan plan;
        plan.sourceChunkIndex = sourceChunkIndex;
        plan.sourceSliceBegin = sourceSliceBegin;
        plan.sourceSliceCount = sourceSliceEnd - sourceSliceBegin;

        for (size_t sliceIndex = sourceSliceBegin; sliceIndex < sourceSliceEnd; ++sliceIndex) {
            if ((sliceIndex % sampleStep) != 0) {
                continue;
            }
            plan.sampledSlices.push_back(
                NormalGridSampledSlice{sliceIndex, sliceIndex - sourceSliceBegin});
        }

        if (!plan.sampledSlices.empty()) {
            plans.push_back(std::move(plan));
        }
    }

    return plans;
}

inline NormalGridBatchPlan planNormalGridBatch(
    const std::vector<size_t>& shape,
    NormalGridSliceDirection direction,
    int numThreads,
    int sparseVolume,
    size_t chunkBudgetMiB,
    size_t bytesPerVoxel = 1)
{
    if (shape.size() != 3) {
        throw std::runtime_error("planNormalGridBatch expects 3D ZYX shape");
    }
    if (bytesPerVoxel == 0) {
        throw std::runtime_error("bytesPerVoxel must be > 0");
    }

    const size_t budgetBytes = chunkBudgetMiB * 1024ull * 1024ull;
    const size_t threadSlices = static_cast<size_t>(std::max(1, numThreads)) *
                                static_cast<size_t>(std::max(1, sparseVolume));

    size_t bytesPerSlice = 0;
    switch (direction) {
    case NormalGridSliceDirection::XY:
        bytesPerSlice = shape[1] * shape[2] * bytesPerVoxel;
        break;
    case NormalGridSliceDirection::XZ:
        bytesPerSlice = shape[0] * shape[2] * bytesPerVoxel;
        break;
    case NormalGridSliceDirection::YZ:
        bytesPerSlice = shape[0] * shape[1] * bytesPerVoxel;
        break;
    }

    size_t chunkSizeTarget = threadSlices;
    if (bytesPerSlice > 0 && budgetBytes > 0) {
        chunkSizeTarget = std::max<size_t>(1, std::min(threadSlices, budgetBytes / bytesPerSlice));
    } else if (budgetBytes == 0) {
        chunkSizeTarget = 1;
    }

    return NormalGridBatchPlan{
        bytesPerSlice,
        chunkSizeTarget,
        bytesPerSlice * chunkSizeTarget,
    };
}

struct BinarySliceTarget {
    cv::Mat* binarySlice = nullptr;     // pre-allocated full-extent CV_8U slice; written by the helper
    int localSliceIndex = 0;            // 0..sourceChunkShape[sliceAxis]-1
    bool anyNonZero = false;            // out: set true if any voxel of this slice is non-zero
};

namespace detail {

template <NormalGridSliceDirection Dir>
inline void distributeChunk(
    const std::byte* chunkBytes,
    const std::array<int, 3>& chunkShape,           // ZYX
    int validD, int validH, int validW,
    int rowOffset, int colOffset,
    std::span<const int> sliceForLocal,             // size = sourceSliceAxisLen, -1 for none
    std::span<BinarySliceTarget> targets)
{
    const int H = chunkShape[1];
    const int W = chunkShape[2];
    const auto* chunk = reinterpret_cast<const uint8_t*>(chunkBytes);

    for (int lz = 0; lz < validD; ++lz) {
        for (int ly = 0; ly < validH; ++ly) {
            const uint8_t* row = chunk + (lz * H + ly) * W;
            for (int lx = 0; lx < validW; ++lx) {
                int local;
                int dr;
                int dc;
                if constexpr (Dir == NormalGridSliceDirection::XY) {
                    local = lz; dr = rowOffset + ly; dc = colOffset + lx;
                } else if constexpr (Dir == NormalGridSliceDirection::XZ) {
                    local = ly; dr = rowOffset + lz; dc = colOffset + lx;
                } else { // YZ
                    local = lx; dr = rowOffset + lz; dc = colOffset + ly;
                }
                const int slot = sliceForLocal[local];
                if (slot < 0) continue;
                const uint8_t v = row[lx];
                targets[slot].binarySlice->ptr<uint8_t>(dr)[dc] =
                    v ? static_cast<uint8_t>(255) : static_cast<uint8_t>(0);
                if (v) targets[slot].anyNonZero = true;
            }
        }
    }
}

}  // namespace detail

// Walk the chunk grid of one chunk-aligned slab along the slice axis,
// decode each chunk via volume.readChunk (uncached; nullopt = missing or
// AllFill), and distribute voxels into the caller-provided binary slice
// batch. Each target's cv::Mat is pre-allocated zero CV_8U at the slice
// extent given by normalGridSliceSize(volumeShape, direction); the helper
// writes 255 where voxel != 0, 0 otherwise, and toggles anyNonZero.
//
// Sequential implementation; no global slab buffer.
inline void fillBinarySliceBatchFromVolume(
    Volume& volume,
    int level,
    NormalGridSliceDirection direction,
    int sliceAxisChunkIndex,
    const std::array<int, 3>& volumeShape,
    const std::array<int, 3>& sourceChunkShape,
    std::span<BinarySliceTarget> targets)
{
    if (targets.empty()) {
        return;
    }
    if (volume.dtype() != vc::render::ChunkDtype::UInt8) {
        throw std::runtime_error(
            "fillBinarySliceBatchFromVolume currently supports uint8 volumes only");
    }

    const int sliceAxis = static_cast<int>(normalGridSliceAxis(direction));
    const int sourceSliceAxisLen = sourceChunkShape[sliceAxis];
    if (sourceSliceAxisLen <= 0) {
        throw std::runtime_error("source chunk slice-axis length must be > 0");
    }

    int axisA = 0;
    int axisB = 0;
    switch (direction) {
    case NormalGridSliceDirection::XY: axisA = 1; axisB = 2; break;
    case NormalGridSliceDirection::XZ: axisA = 0; axisB = 2; break;
    case NormalGridSliceDirection::YZ: axisA = 0; axisB = 1; break;
    }
    const int chunkSizeA = sourceChunkShape[axisA];
    const int chunkSizeB = sourceChunkShape[axisB];
    const int volA = volumeShape[axisA];
    const int volB = volumeShape[axisB];
    const int Ca = (volA + chunkSizeA - 1) / chunkSizeA;
    const int Cb = (volB + chunkSizeB - 1) / chunkSizeB;

    // local-slice -> target slot
    std::vector<int> sliceForLocal(static_cast<size_t>(sourceSliceAxisLen), -1);
    for (size_t i = 0; i < targets.size(); ++i) {
        const int local = targets[i].localSliceIndex;
        if (local < 0 || local >= sourceSliceAxisLen) {
            throw std::runtime_error(
                "fillBinarySliceBatchFromVolume: localSliceIndex out of range");
        }
        sliceForLocal[static_cast<size_t>(local)] = static_cast<int>(i);
    }

    const int D = sourceChunkShape[0];
    const int H = sourceChunkShape[1];
    const int W = sourceChunkShape[2];
    const int volZ = volumeShape[0];
    const int volY = volumeShape[1];
    const int volX = volumeShape[2];

    for (int ca = 0; ca < Ca; ++ca) {
        for (int cb = 0; cb < Cb; ++cb) {
            std::array<size_t, 3> chunkKey{};
            chunkKey[static_cast<size_t>(sliceAxis)] = static_cast<size_t>(sliceAxisChunkIndex);
            chunkKey[static_cast<size_t>(axisA)] = static_cast<size_t>(ca);
            chunkKey[static_cast<size_t>(axisB)] = static_cast<size_t>(cb);

            auto bytes = volume.readChunk(level, chunkKey);
            if (!bytes) {
                continue;  // missing or AllFill
            }

            const int rowOffset = ca * chunkSizeA;
            const int colOffset = cb * chunkSizeB;

            int validD = D;
            int validH = H;
            int validW = W;

            switch (direction) {
            case NormalGridSliceDirection::XY:
                validD = std::min(D, std::max(0, volZ - sliceAxisChunkIndex * D));
                validH = std::min(H, std::max(0, volY - rowOffset));
                validW = std::min(W, std::max(0, volX - colOffset));
                detail::distributeChunk<NormalGridSliceDirection::XY>(
                    bytes->data(), sourceChunkShape, validD, validH, validW,
                    rowOffset, colOffset,
                    std::span<const int>(sliceForLocal), targets);
                break;
            case NormalGridSliceDirection::XZ:
                validD = std::min(D, std::max(0, volZ - rowOffset));
                validH = std::min(H, std::max(0, volY - sliceAxisChunkIndex * H));
                validW = std::min(W, std::max(0, volX - colOffset));
                detail::distributeChunk<NormalGridSliceDirection::XZ>(
                    bytes->data(), sourceChunkShape, validD, validH, validW,
                    rowOffset, colOffset,
                    std::span<const int>(sliceForLocal), targets);
                break;
            case NormalGridSliceDirection::YZ:
                validD = std::min(D, std::max(0, volZ - rowOffset));
                validH = std::min(H, std::max(0, volY - colOffset));
                validW = std::min(W, std::max(0, volX - sliceAxisChunkIndex * W));
                detail::distributeChunk<NormalGridSliceDirection::YZ>(
                    bytes->data(), sourceChunkShape, validD, validH, validW,
                    rowOffset, colOffset,
                    std::span<const int>(sliceForLocal), targets);
                break;
            }
        }
    }
}

}  // namespace vc::core::util
