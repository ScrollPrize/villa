#pragma once

#include "vc/core/render/IChunkedArray.hpp"

#include <opencv2/core.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>

namespace vc::util {

enum class SeedScanStatus {
    Data,
    ZeroVoxel,
    MissingChunk,
    OutsideVolume
};

enum class SeedScanPolicy {
    Warn,
    Require
};

struct SeedScanSample {
    SeedScanStatus status = SeedScanStatus::Data;
    double value = 0.0;
    std::array<int, 3> chunk{};
};

struct SeedScanDecision {
    bool abort = false;
    std::string message;
};

inline SeedScanSample sampleSeedInScan(vc::render::IChunkedArray& scan,
                                       const cv::Vec3d& seed_xyz)
{
    SeedScanSample sample;

    const auto shape = scan.shape(0);
    const auto chunk_shape = scan.chunkShape(0);

    std::array<long long, 3> voxel_zyx{};
    for (int i = 0; i < 3; ++i) {
        if (shape[i] <= 0 || chunk_shape[i] <= 0) {
            throw std::runtime_error("scan volume reports an invalid level 0 shape");
        }
        const double coordinate = seed_xyz[2 - i];
        if (!std::isfinite(coordinate)) {
            sample.status = SeedScanStatus::OutsideVolume;
            return sample;
        }
        voxel_zyx[i] = std::llround(coordinate);
        if (voxel_zyx[i] < 0 || voxel_zyx[i] >= shape[i]) {
            sample.status = SeedScanStatus::OutsideVolume;
            return sample;
        }
    }

    std::array<long long, 3> local_zyx{};
    for (int i = 0; i < 3; ++i) {
        sample.chunk[i] = static_cast<int>(voxel_zyx[i] / chunk_shape[i]);
        local_zyx[i] = voxel_zyx[i] % chunk_shape[i];
    }

    const auto result = scan.getChunkBlocking(0, sample.chunk[0], sample.chunk[1], sample.chunk[2]);
    if (result.status == vc::render::ChunkStatus::Error) {
        throw std::runtime_error(result.error.empty()
            ? std::string("failed to read the scan chunk holding the seed")
            : result.error);
    }

    const size_t element_size = result.dtype == vc::render::ChunkDtype::UInt16 ? 2 : 1;
    const size_t index =
        (static_cast<size_t>(local_zyx[0]) * static_cast<size_t>(chunk_shape[1]) +
         static_cast<size_t>(local_zyx[1])) * static_cast<size_t>(chunk_shape[2]) +
        static_cast<size_t>(local_zyx[2]);

    if (result.status != vc::render::ChunkStatus::Data || !result.bytes ||
        result.bytes->size() < (index + 1) * element_size) {
        sample.status = SeedScanStatus::MissingChunk;
        sample.value = scan.fillValue();
        return sample;
    }

    if (element_size == 2) {
        sample.value = static_cast<double>(
            reinterpret_cast<const uint16_t*>(result.bytes->data())[index]);
    } else {
        sample.value = static_cast<double>(
            reinterpret_cast<const uint8_t*>(result.bytes->data())[index]);
    }
    sample.status = sample.value == 0.0 ? SeedScanStatus::ZeroVoxel : SeedScanStatus::Data;
    return sample;
}

inline SeedScanDecision evaluateSeedScan(const SeedScanSample& sample,
                                         const cv::Vec3d& seed_xyz,
                                         SeedScanPolicy policy = SeedScanPolicy::Warn)
{
    SeedScanDecision decision;
    if (sample.status == SeedScanStatus::Data) {
        return decision;
    }

    std::ostringstream reason;
    reason << "seed " << seed_xyz[0] << " " << seed_xyz[1] << " " << seed_xyz[2] << " ";
    switch (sample.status) {
    case SeedScanStatus::ZeroVoxel:
        reason << "lands on scan voxel 0";
        break;
    case SeedScanStatus::MissingChunk:
        reason << "lands in scan chunk " << sample.chunk[0] << "/" << sample.chunk[1]
               << "/" << sample.chunk[2] << ", which the scan does not hold";
        break;
    case SeedScanStatus::OutsideVolume:
        reason << "is outside the scan volume";
        break;
    case SeedScanStatus::Data:
        break;
    }
    reason << ": the scan has no data there, so a surface grown from it follows the"
              " prediction through empty air and the area it reports means nothing";

    decision.message = reason.str();
    decision.abort = policy == SeedScanPolicy::Require;
    return decision;
}

}  // namespace vc::util
