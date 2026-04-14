#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include <opencv2/core.hpp>
#include <xtensor/containers/xtensor.hpp>

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

inline void copyBinarySliceRegionFromChunk(
    const uint8_t* chunkData,
    const std::array<int, 3>& chunkShape,
    NormalGridSliceDirection direction,
    size_t localSliceIndex,
    size_t validRows,
    size_t validCols,
    int dstRowOffset,
    int dstColOffset,
    cv::Mat& binarySlice,
    bool& anyNonZero)
{
    const int strideZ = chunkShape[1] * chunkShape[2];
    const int strideY = chunkShape[2];

    switch (direction) {
    case NormalGridSliceDirection::XY:
        for (size_t row = 0; row < validRows; ++row) {
            const auto* src = chunkData +
                static_cast<size_t>(localSliceIndex) * strideZ +
                row * strideY;
            auto* dst = binarySlice.ptr<uint8_t>(dstRowOffset + static_cast<int>(row)) + dstColOffset;
            for (size_t col = 0; col < validCols; ++col) {
                const bool isSet = src[col] > 0;
                dst[col] = isSet ? static_cast<uint8_t>(255) : static_cast<uint8_t>(0);
                anyNonZero = anyNonZero || isSet;
            }
        }
        break;
    case NormalGridSliceDirection::XZ:
        for (size_t row = 0; row < validRows; ++row) {
            auto* dst = binarySlice.ptr<uint8_t>(dstRowOffset + static_cast<int>(row)) + dstColOffset;
            for (size_t col = 0; col < validCols; ++col) {
                const size_t srcIndex =
                    row * strideZ +
                    static_cast<size_t>(localSliceIndex) * strideY +
                    col;
                const bool isSet = chunkData[srcIndex] > 0;
                dst[col] = isSet ? static_cast<uint8_t>(255) : static_cast<uint8_t>(0);
                anyNonZero = anyNonZero || isSet;
            }
        }
        break;
    case NormalGridSliceDirection::YZ:
        for (size_t row = 0; row < validRows; ++row) {
            auto* dst = binarySlice.ptr<uint8_t>(dstRowOffset + static_cast<int>(row)) + dstColOffset;
            for (size_t col = 0; col < validCols; ++col) {
                const size_t srcIndex =
                    row * strideZ +
                    col * strideY +
                    static_cast<size_t>(localSliceIndex);
                const bool isSet = chunkData[srcIndex] > 0;
                dst[col] = isSet ? static_cast<uint8_t>(255) : static_cast<uint8_t>(0);
                anyNonZero = anyNonZero || isSet;
            }
        }
        break;
    }
}

inline bool extractBinarySliceFromChunk(
    const xt::xtensor<uint8_t, 3, xt::layout_type::column_major>& chunkData,
    NormalGridSliceDirection direction,
    size_t iChunk,
    cv::Mat& binarySlice)
{
    bool anyNonZero = false;

    switch (direction) {
    case NormalGridSliceDirection::XY:
        binarySlice.create(static_cast<int>(chunkData.shape()[1]),
                           static_cast<int>(chunkData.shape()[2]),
                           CV_8U);
        for (int row = 0; row < binarySlice.rows; ++row) {
            uint8_t* dst = binarySlice.ptr<uint8_t>(row);
            for (int col = 0; col < binarySlice.cols; ++col) {
                const bool isSet = chunkData(iChunk, static_cast<size_t>(row), static_cast<size_t>(col)) > 0;
                dst[col] = isSet ? static_cast<uint8_t>(255) : static_cast<uint8_t>(0);
                anyNonZero = anyNonZero || isSet;
            }
        }
        break;
    case NormalGridSliceDirection::XZ:
        binarySlice.create(static_cast<int>(chunkData.shape()[0]),
                           static_cast<int>(chunkData.shape()[2]),
                           CV_8U);
        for (int row = 0; row < binarySlice.rows; ++row) {
            uint8_t* dst = binarySlice.ptr<uint8_t>(row);
            for (int col = 0; col < binarySlice.cols; ++col) {
                const bool isSet = chunkData(static_cast<size_t>(row), iChunk, static_cast<size_t>(col)) > 0;
                dst[col] = isSet ? static_cast<uint8_t>(255) : static_cast<uint8_t>(0);
                anyNonZero = anyNonZero || isSet;
            }
        }
        break;
    case NormalGridSliceDirection::YZ:
        binarySlice.create(static_cast<int>(chunkData.shape()[0]),
                           static_cast<int>(chunkData.shape()[1]),
                           CV_8U);
        for (int row = 0; row < binarySlice.rows; ++row) {
            uint8_t* dst = binarySlice.ptr<uint8_t>(row);
            for (int col = 0; col < binarySlice.cols; ++col) {
                const bool isSet = chunkData(static_cast<size_t>(row), static_cast<size_t>(col), iChunk) > 0;
                dst[col] = isSet ? static_cast<uint8_t>(255) : static_cast<uint8_t>(0);
                anyNonZero = anyNonZero || isSet;
            }
        }
        break;
    }

    return anyNonZero;
}

} // namespace vc::core::util
