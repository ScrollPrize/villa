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
// Legacy queries are rounded to a 3-voxel lattice and cached. Substep and
// streamline queries evaluate at their actual position. Thread-safe.
class SheetNormalField
{
public:
    explicit SheetNormalField(std::shared_ptr<Volume> volume);

    // Unit normal (unsigned) at the lattice point nearest `world`, or nothing
    // where the tensor is flat or ambiguous or the volume has no support.
    // continuous=true evaluates tensor weights at the exact world point,
    // without rounding the query. The patch and gradient
    // sampling lattice stay fixed within each 3-voxel cache cell.
    std::optional<cv::Vec3f> normalAt(const cv::Vec3f& world, bool continuous = false);

private:
    std::shared_ptr<Volume> _volume;
    std::mutex _mutex;
    std::map<std::array<int, 3>, std::optional<cv::Vec3f>> _cache;
};
