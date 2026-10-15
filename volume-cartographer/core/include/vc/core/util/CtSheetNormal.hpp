#pragma once

#include <atomic>
#include <filesystem>
#include <memory>
#include <string>
#include <opencv2/core/types.hpp>

class Volume;

namespace vc {

struct CtSheetNormal {
    cv::Vec3d normal{0, 0, 0};
    bool valid = false;
    double gap = 0;
    std::string reason;
};

// CT-only sheet normal: Gaussian derivatives at sigma 2/1, isotropic sigma-24
// seed, then three anisotropic 24/4 refinements with a median-gradient floor.
// Tensor quadrature uses a global 3-voxel lattice with radius-48 support; the
// derivatives themselves use the original CT resolution. Units: selected CT
// voxels. Query positions are not rounded or interpolated.
// Requires complete quadrature support plus a 6-voxel derivative halo and
// uint8 CT. Cached tile prefetch may be clipped at volume edges, but every
// gradient used by a valid query has its complete input halo.
// A 1024-entry indexed LRU caches 48-voxel CT gradient tiles, and a separate 256-entry exact-point
// cache avoids duplicate normal queries. Sampling is serialized. Float32
// storage/filtering, float64 tensor integration/eigendecomposition.
// Missing tiles share one bounding CT read. Retained buffers grow lazily to
// about 102 MiB (excluding the volume's chunk cache).
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
} // namespace vc
