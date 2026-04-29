#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <opencv2/core.hpp>

#include "vc/core/util/QuadSurface.hpp"

class QuadSurface;
class PlaneSurface;

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

    struct PatchBounds {
        cv::Vec3f low = {0, 0, 0};
        cv::Vec3f high = {0, 0, 0};
    };

    struct TriangleQuery {
        Rect3D bounds;
        SurfacePtr targetSurface;
        const std::unordered_set<SurfacePtr>* targetSurfaces = nullptr;
        std::function<bool(const PatchBounds&)> patchFilter;
    };

    SurfacePatchIndex();
    ~SurfacePatchIndex();

    SurfacePatchIndex(SurfacePatchIndex&&) noexcept;
    SurfacePatchIndex& operator=(SurfacePatchIndex&&) noexcept;

    SurfacePatchIndex(const SurfacePatchIndex&) = delete;
    SurfacePatchIndex& operator=(const SurfacePatchIndex&) = delete;

    void rebuild(const std::vector<SurfacePtr>& surfaces, float bboxPadding = 0.0f);
    static std::string cacheKeyForSurfaces(const std::vector<SurfacePtr>& surfaces,
                                           int samplingStride,
                                           float bboxPadding);
    bool loadCache(const std::filesystem::path& cachePath,
                   const std::vector<SurfacePtr>& surfaces,
                   const std::string& expectedKey);
    bool saveCache(const std::filesystem::path& cachePath,
                   const std::string& cacheKey) const;
    void clear();
    bool empty() const;
    size_t patchCount() const;
    size_t surfaceCount() const;
    bool containsSurface(const SurfacePtr& surface) const;

    std::optional<LookupResult> locate(const cv::Vec3f& worldPoint,
                                       float tolerance,
                                       const SurfacePtr& targetSurface = nullptr) const;
    std::vector<LookupResult> locateAll(const cv::Vec3f& worldPoint,
                                        float tolerance,
                                        const SurfacePtr& targetSurface = nullptr) const;
    void locateSurfaceHits(const cv::Vec3f& worldPoint,
                           float tolerance,
                           const std::unordered_set<const QuadSurface*>& excludedSurfaces,
                           std::vector<const QuadSurface*>& outSurfaces,
                           const SurfacePtr& targetSurface = nullptr) const;

    void forEachTriangle(const TriangleQuery& query,
                         const std::function<void(const TriangleCandidate&)>& visitor) const;

    static std::optional<TriangleSegment> clipTriangleToPlane(const TriangleCandidate& tri,
                                                              const PlaneSurface& plane,
                                                              float epsilon = 1e-4f);

    std::unordered_map<SurfacePtr, std::vector<TriangleSegment>>
    computePlaneIntersections(
        const PlaneSurface& plane,
        const cv::Rect& planeRoi,
        const std::unordered_set<SurfacePtr>& targets,
        float clipTolerance = 1e-4f) const;

    bool updateSurface(const SurfacePtr& surface);
    bool updateSurfaceRegion(const SurfacePtr& surface,
                             int rowStart,
                             int rowEnd,
                             int colStart,
                             int colEnd);
    bool removeSurface(const SurfacePtr& surface);
    bool setSamplingStride(int stride);
    int samplingStride() const;
    void setReadOnly(bool readOnly);

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
    uint64_t generation(const SurfacePtr& surface) const;

private:
    struct NoPatchFilter {
        template <typename Box>
        bool operator()(const Box&) const { return true; }
    };

    template <typename Visitor, typename PatchFilter = NoPatchFilter>
    void forEachTriangleImpl(const Rect3D& bounds,
                             const SurfacePtr& targetSurface,
                             const std::unordered_set<SurfacePtr>* filterSurfaces,
                             Visitor&& visitor,
                             PatchFilter&& patchFilter = NoPatchFilter{}) const;

    struct Impl;
    mutable std::shared_mutex mutex_;
    std::unique_ptr<Impl> impl_;
};
