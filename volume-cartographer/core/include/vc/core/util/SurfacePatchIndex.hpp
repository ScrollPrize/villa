#pragma once

#include <array>
#include <functional>
#include <memory>
#include <optional>
#include <unordered_set>
#include <vector>

#include <opencv2/core.hpp>

class QuadSurface;
class PlaneSurface;
struct Rect3D;

class SurfacePatchIndex {
public:
    struct LookupResult {
        QuadSurface* surface = nullptr;
        cv::Vec3f ptr = {0, 0, 0};
        float distance = -1.0f;
    };

    struct TriangleCandidate {
        QuadSurface* surface = nullptr;
        int i = 0;
        int j = 0;
        int triangleIndex = 0; // 0 = (p00,p10,p01), 1 = (p10,p11,p01)
        std::array<cv::Vec3f, 3> world{};
        std::array<cv::Vec3f, 3> surfaceParams{}; // ptr-space coordinates for vertices
    };

    struct TriangleSegment {
        QuadSurface* surface = nullptr;
        std::array<cv::Vec3f, 2> world{};
        std::array<cv::Vec3f, 2> surfaceParams{};
    };

    SurfacePatchIndex();
    ~SurfacePatchIndex();

    SurfacePatchIndex(SurfacePatchIndex&&) noexcept;
    SurfacePatchIndex& operator=(SurfacePatchIndex&&) noexcept;

    SurfacePatchIndex(const SurfacePatchIndex&) = delete;
    SurfacePatchIndex& operator=(const SurfacePatchIndex&) = delete;

    void rebuild(const std::vector<QuadSurface*>& surfaces, float bboxPadding = 0.0f);
    void clear();
    bool empty() const;

    std::optional<LookupResult> locate(const cv::Vec3f& worldPoint,
                                       float tolerance,
                                       QuadSurface* targetSurface = nullptr) const;

    void queryTriangles(const Rect3D& bounds,
                        QuadSurface* targetSurface,
                        std::vector<TriangleCandidate>& outCandidates) const;

    void queryTriangles(const Rect3D& bounds,
                        const std::unordered_set<QuadSurface*>& targetSurfaces,
                        std::vector<TriangleCandidate>& outCandidates) const;

    void forEachTriangle(const Rect3D& bounds,
                         QuadSurface* targetSurface,
                         const std::function<void(const TriangleCandidate&)>& visitor) const;

    void forEachTriangle(const Rect3D& bounds,
                         const std::unordered_set<QuadSurface*>& targetSurfaces,
                         const std::function<void(const TriangleCandidate&)>& visitor) const;

    static std::optional<TriangleSegment> clipTriangleToPlane(const TriangleCandidate& tri,
                                                              const PlaneSurface& plane,
                                                              float epsilon = 1e-4f);

    bool updateSurface(QuadSurface* surface);
    bool updateSurfaceRegion(QuadSurface* surface,
                             int rowStart,
                             int rowEnd,
                             int colStart,
                             int colEnd);
    bool removeSurface(QuadSurface* surface);
    bool setSamplingStride(int stride);
    int samplingStride() const;

private:
    void forEachTriangleImpl(const Rect3D& bounds,
                             QuadSurface* targetSurface,
                             const std::unordered_set<QuadSurface*>* filterSurfaces,
                             const std::function<void(const TriangleCandidate&)>& visitor) const;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};
