#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include <opencv2/core.hpp>
#include <vc/core/types/Array3D.hpp>

namespace vc::core::util {

enum class NormalGridSliceDirection { XY, XZ, YZ };

struct NormalGridBatchPlan {
    size_t bytesPerSlice = 0;
    size_t chunkSizeTarget = 1;
    size_t estimatedBatchBytes = 0;
};

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

inline bool extractBinarySliceFromChunk(
    const Array3D<uint8_t>& chunkData,
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
