#pragma once

#include <array>
#include <cstdint>
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
    using SurfacePtr = std::shared_ptr<QuadSurface>;

    struct LookupResult {
        SurfacePtr surface;
        cv::Vec3f ptr = {0, 0, 0};
        float distance = -1.0f;
    };

    struct TriangleCandidate {
        SurfacePtr surface;
        int i = 0;
        int j = 0;
        int triangleIndex = 0; // 0 = (p00,p10,p01), 1 = (p10,p11,p01)
        std::array<cv::Vec3f, 3> world{};
        std::array<cv::Vec3f, 3> surfaceParams{}; // ptr-space coordinates for vertices
    };

    struct TriangleSegment {
        SurfacePtr surface;
        std::array<cv::Vec3f, 2> world{};
        std::array<cv::Vec3f, 2> surfaceParams{};
    };

    SurfacePatchIndex();
    ~SurfacePatchIndex();

    SurfacePatchIndex(SurfacePatchIndex&&) noexcept;
    SurfacePatchIndex& operator=(SurfacePatchIndex&&) noexcept;

    SurfacePatchIndex(const SurfacePatchIndex&) = delete;
    SurfacePatchIndex& operator=(const SurfacePatchIndex&) = delete;

    void rebuild(const std::vector<SurfacePtr>& surfaces, float bboxPadding = 0.0f);
    void clear();
    bool empty() const;

    std::optional<LookupResult> locate(const cv::Vec3f& worldPoint,
                                       float tolerance,
                                       const SurfacePtr& targetSurface = nullptr) const;

    void queryTriangles(const Rect3D& bounds,
                        const SurfacePtr& targetSurface,
                        std::vector<TriangleCandidate>& outCandidates) const;

    void queryTriangles(const Rect3D& bounds,
                        const std::unordered_set<SurfacePtr>& targetSurfaces,
                        std::vector<TriangleCandidate>& outCandidates) const;

    void forEachTriangle(const Rect3D& bounds,
                         const SurfacePtr& targetSurface,
                         const std::function<void(const TriangleCandidate&)>& visitor) const;

    void forEachTriangle(const Rect3D& bounds,
                         const std::unordered_set<SurfacePtr>& targetSurfaces,
                         const std::function<void(const TriangleCandidate&)>& visitor) const;

    static std::optional<TriangleSegment> clipTriangleToPlane(const TriangleCandidate& tri,
                                                              const PlaneSurface& plane,
                                                              float epsilon = 1e-4f);

    bool updateSurface(const SurfacePtr& surface);
    bool updateSurfaceRegion(const SurfacePtr& surface,
                             int rowStart,
                             int rowEnd,
                             int colStart,
                             int colEnd);
    bool removeSurface(const SurfacePtr& surface);
    bool setSamplingStride(int stride);
    int samplingStride() const;

    // Pending update tracking for incremental R-tree updates
    // Queue the 4 cells surrounding a vertex for update
    void queueCellUpdateForVertex(const SurfacePtr& surface, int vertexRow, int vertexCol);
    // Queue a range of cells for update
    void queueCellRangeUpdate(const SurfacePtr& surface,
                              int rowStart,
                              int rowEnd,
                              int colStart,
                              int colEnd);
    // Apply all pending cell updates to R-tree (nullptr = all surfaces)
    bool flushPendingUpdates(const SurfacePtr& surface = nullptr);
    // Check if surface has pending cell updates
    bool hasPendingUpdates(const SurfacePtr& surface = nullptr) const;

    // Generation tracking for undo/redo detection
    void incrementGeneration(const SurfacePtr& surface);
    uint64_t generation(const SurfacePtr& surface) const;
    void setGeneration(const SurfacePtr& surface, uint64_t gen);

private:
    void forEachTriangleImpl(const Rect3D& bounds,
                             const SurfacePtr& targetSurface,
                             const std::unordered_set<SurfacePtr>* filterSurfaces,
                             const std::function<void(const TriangleCandidate&)>& visitor) const;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};
