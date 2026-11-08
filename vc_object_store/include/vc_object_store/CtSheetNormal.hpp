#pragma once

#include <atomic>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <opencv2/core/types.hpp>

namespace vc {

// The normal source itself could not be read (a missing or truncated pack
// brick, an unreadable chunk): a failure of the run, unlike a query outside
// the volume or without data, which is an invalid sample.
struct SheetNormalSourceError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct CtSheetNormal {
    cv::Vec3d normal{0, 0, 0};
    bool valid = false;
    double gap = 0;
    std::string reason;
};

// Bounded cached adapter over vc_reader's native structure-tensor estimator.
// XYZ queries are scaled into the selected uint8 CT level. Insufficient
// 109-cubed context or an ambiguous tensor returns an invalid sample.
class CtSheetNormalSampler {
public:
    static constexpr const char* method = "ct-anisotropic-1-24-4-grid3-r48-v4";
    CtSheetNormalSampler(const std::filesystem::path& ctVolume, int level = 0,
                         double workingToCtScale = 1.0);
    ~CtSheetNormalSampler();
    CtSheetNormalSampler(const CtSheetNormalSampler&) = delete;
    CtSheetNormalSampler& operator=(const CtSheetNormalSampler&) = delete;

    CtSheetNormal sampleNormal(const cv::Vec3d& workingXYZ,
                               const std::atomic_bool* cancel = nullptr) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Precomputed lasagna normals: paired uint8 nx/ny volumes at the same level,
// decoded as the spiral fitter does: (v - 128) / 127, nz = sqrt(1 - nx^2 - ny^2),
// and nx = ny = 0 marks no data. XYZ queries are scaled into the selected level
// and read at the nearest voxel; outside the volume returns an invalid sample.
class LasagnaSheetNormalSampler {
public:
    static constexpr const char* method = "lasagna-nxny-u8-nearest-v1";
    LasagnaSheetNormalSampler(const std::filesystem::path& normalX,
                              const std::filesystem::path& normalY, int level = 0,
                              double workingToNormalScale = 1.0);
    ~LasagnaSheetNormalSampler();
    LasagnaSheetNormalSampler(const LasagnaSheetNormalSampler&) = delete;
    LasagnaSheetNormalSampler& operator=(const LasagnaSheetNormalSampler&) = delete;

    CtSheetNormal sampleNormal(const cv::Vec3d& workingXYZ,
                               const std::atomic_bool* cancel = nullptr) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// The same lasagna normals read from a resident-pool pack instead
// (spiral-fitting's pack_resident_pools.py: meta.json, table.npy and the
// channel_0.u8 / channel_1.u8 brick pools of nx and ny), sampled and decoded
// exactly as LasagnaSheetNormalSampler does; an absent brick (one the pack
// dropped as empty or CT-masked) reads as no data. Bricks are read on demand.
class PooledLasagnaSheetNormalSampler {
public:
    static constexpr const char* method = LasagnaSheetNormalSampler::method;
    explicit PooledLasagnaSheetNormalSampler(const std::filesystem::path& pack,
                                             double workingToNormalScale = 1.0);
    ~PooledLasagnaSheetNormalSampler();
    PooledLasagnaSheetNormalSampler(const PooledLasagnaSheetNormalSampler&) = delete;
    PooledLasagnaSheetNormalSampler& operator=(const PooledLasagnaSheetNormalSampler&) = delete;

    CtSheetNormal sampleNormal(const cv::Vec3d& workingXYZ,
                               const std::atomic_bool* cancel = nullptr) const;
    // The pack's meta.json text (its dataset identity, CT mask and shape).
    const std::string& metadata() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace vc
