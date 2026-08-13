#include "alpha_wrap.hpp"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polygon_mesh_processing/bbox.h>
#include <CGAL/Side_of_triangle_mesh.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/alpha_wrap_3.h>

using AlphaWrapKernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using AlphaWrapPoint = AlphaWrapKernel::Point_3;
using AlphaWrapMesh = CGAL::Surface_mesh<AlphaWrapPoint>;

namespace {

std::size_t countNonZero(const uint8_t* data, std::size_t n)
{
    std::size_t c = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (data[i] != 0) {
            ++c;
        }
    }
    return c;
}

std::size_t volumeElements(const Shape3& shape)
{
    return shape[0] * shape[1] * shape[2];
}

std::size_t linearIndex(const Shape3& shape, std::size_t z, std::size_t y, std::size_t x)
{
    return (z * shape[1] + y) * shape[2] + x;
}

Shape3 relativeOrigin(const Box3& inner, const Box3& outer)
{
    return {inner.origin[0] - outer.origin[0],
            inner.origin[1] - outer.origin[1],
            inner.origin[2] - outer.origin[2]};
}

AlphaWrapPoint voxelCenterPoint(std::size_t z, std::size_t y, std::size_t x)
{
    return AlphaWrapPoint(static_cast<double>(z) + 0.5,
                          static_cast<double>(y) + 0.5,
                          static_cast<double>(x) + 0.5);
}

bool bboxContains(const CGAL::Bbox_3& bbox, const AlphaWrapPoint& p)
{
    return p.x() >= bbox.xmin() && p.x() <= bbox.xmax() &&
           p.y() >= bbox.ymin() && p.y() <= bbox.ymax() &&
           p.z() >= bbox.zmin() && p.z() <= bbox.zmax();
}

}  // namespace

std::vector<uint8_t> classifyOuterAlphaWrap(const std::vector<uint8_t>& halo,
                                            const Box3& haloBox,
                                            const Box3& coreBox,
                                            double alpha,
                                            double offset)
{
    std::vector<AlphaWrapPoint> points;
    points.reserve(countNonZero(halo.data(), halo.size()));
    for (std::size_t z = 0; z < haloBox.shape[0]; ++z) {
        for (std::size_t y = 0; y < haloBox.shape[1]; ++y) {
            const std::size_t base = linearIndex(haloBox.shape, z, y, 0);
            for (std::size_t x = 0; x < haloBox.shape[2]; ++x) {
                if (halo[base + x] == 0) {
                    continue;
                }
                points.push_back(voxelCenterPoint(haloBox.origin[0] + z,
                                                  haloBox.origin[1] + y,
                                                  haloBox.origin[2] + x));
            }
        }
    }

    const Shape3 coreRel = relativeOrigin(coreBox, haloBox);
    std::vector<uint8_t> ignore(volumeElements(coreBox.shape), 0);
    if (points.size() < 4) {
        return ignore;
    }

    AlphaWrapMesh wrap;
    CGAL::alpha_wrap_3(points, alpha, offset, wrap);
    if (num_faces(wrap) == 0) {
        return ignore;
    }

    const CGAL::Bbox_3 bbox = CGAL::Polygon_mesh_processing::bbox(wrap);
    CGAL::Side_of_triangle_mesh<AlphaWrapMesh, AlphaWrapKernel> sideOfMesh(wrap);

    for (std::size_t z = 0; z < coreBox.shape[0]; ++z) {
        for (std::size_t y = 0; y < coreBox.shape[1]; ++y) {
            for (std::size_t x = 0; x < coreBox.shape[2]; ++x) {
                const std::size_t haloIdx = linearIndex(haloBox.shape,
                                                        coreRel[0] + z,
                                                        coreRel[1] + y,
                                                        coreRel[2] + x);
                if (halo[haloIdx] != 0) {
                    continue;
                }

                const AlphaWrapPoint p = voxelCenterPoint(coreBox.origin[0] + z,
                                                          coreBox.origin[1] + y,
                                                          coreBox.origin[2] + x);
                if (!bboxContains(bbox, p) || sideOfMesh(p) == CGAL::ON_UNBOUNDED_SIDE) {
                    ignore[linearIndex(coreBox.shape, z, y, x)] = 255;
                }
            }
        }
    }

    return ignore;
}
