#pragma once

#include <opencv2/core.hpp>

#include <array>
#include <map>
#include <memory>
#include <mutex>
#include <optional>

class Volume;

// CT sheet normals at world points, from the anisotropic structure tensor in
// vc_reader (sheet_normals, recipe ct-anisotropic-1-24-4-grid3-r48-v4).
// Queries are rounded to the 3-voxel lattice that estimator works on and
// cached, so walking neighboring lines reuses them. Thread-safe.
class SheetNormalField
{
public:
    explicit SheetNormalField(std::shared_ptr<Volume> volume);

    // Unit normal (unsigned) at the lattice point nearest `world`, or nothing
    // where the tensor is flat or ambiguous or the volume has no support.
    std::optional<cv::Vec3f> normalAt(const cv::Vec3f& world);

private:
    std::shared_ptr<Volume> _volume;
    std::mutex _mutex;
    std::map<std::array<int, 3>, std::optional<cv::Vec3f>> _cache;
};
