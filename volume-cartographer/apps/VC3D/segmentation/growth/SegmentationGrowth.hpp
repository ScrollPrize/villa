#pragma once

#include <QString>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "utils/Json.hpp"

#include <opencv2/core.hpp>

#include "vc/ui/VCCollection.hpp"

class QuadSurface;
class SurfacePatchIndex;
class Volume;

enum class SegmentationGrowthMethod {
    Tracer = 0,
    Corrections = 1,
    PatchTracer = 3,
    ManualAdd = 4,
    Extend = 5,
};

inline QString segmentationGrowthMethodToString(SegmentationGrowthMethod method)
{
    switch (method) {
    case SegmentationGrowthMethod::Tracer:
        return QStringLiteral("Tracer");
    case SegmentationGrowthMethod::Corrections:
        return QStringLiteral("Corrections");
    case SegmentationGrowthMethod::PatchTracer:
        return QStringLiteral("Patch Tracer");
    case SegmentationGrowthMethod::ManualAdd:
        return QStringLiteral("Manual Add");
    case SegmentationGrowthMethod::Extend:
        return QStringLiteral("Extend");
    }
    return QStringLiteral("Unknown");
}

inline SegmentationGrowthMethod segmentationGrowthMethodFromInt(int value)
{
    if (value == static_cast<int>(SegmentationGrowthMethod::Corrections)) {
        return SegmentationGrowthMethod::Corrections;
    }
    if (value == static_cast<int>(SegmentationGrowthMethod::PatchTracer)) {
        return SegmentationGrowthMethod::PatchTracer;
    }
    if (value == static_cast<int>(SegmentationGrowthMethod::ManualAdd)) {
        return SegmentationGrowthMethod::ManualAdd;
    }
    if (value == static_cast<int>(SegmentationGrowthMethod::Extend)) {
        return SegmentationGrowthMethod::Extend;
    }
    return SegmentationGrowthMethod::Tracer;
}

enum class SegmentationGrowthDirection {
    All = 0,
    Up,
    Down,
    Left,
    Right,
    Fill,
};

inline QString segmentationGrowthDirectionToString(SegmentationGrowthDirection direction)
{
    switch (direction) {
    case SegmentationGrowthDirection::All:
        return QStringLiteral("All");
    case SegmentationGrowthDirection::Up:
        return QStringLiteral("Up");
    case SegmentationGrowthDirection::Down:
        return QStringLiteral("Down");
    case SegmentationGrowthDirection::Left:
        return QStringLiteral("Left");
    case SegmentationGrowthDirection::Right:
        return QStringLiteral("Right");
    case SegmentationGrowthDirection::Fill:
        return QStringLiteral("Fill");
    }
    return QStringLiteral("All");
}

inline SegmentationGrowthDirection segmentationGrowthDirectionFromInt(int value)
{
    switch (value) {
    case static_cast<int>(SegmentationGrowthDirection::Up):
        return SegmentationGrowthDirection::Up;
    case static_cast<int>(SegmentationGrowthDirection::Down):
        return SegmentationGrowthDirection::Down;
    case static_cast<int>(SegmentationGrowthDirection::Left):
        return SegmentationGrowthDirection::Left;
    case static_cast<int>(SegmentationGrowthDirection::Right):
        return SegmentationGrowthDirection::Right;
    case static_cast<int>(SegmentationGrowthDirection::Fill):
        return SegmentationGrowthDirection::Fill;
    default:
        return SegmentationGrowthDirection::All;
    }
}

enum class SegmentationDirectionFieldOrientation {
    Normal = 0,
    Horizontal = 1,
    Vertical = 2,
};

inline QString segmentationDirectionFieldOrientationKey(SegmentationDirectionFieldOrientation orientation)
{
    switch (orientation) {
    case SegmentationDirectionFieldOrientation::Horizontal:
        return QStringLiteral("horizontal");
    case SegmentationDirectionFieldOrientation::Vertical:
        return QStringLiteral("vertical");
    case SegmentationDirectionFieldOrientation::Normal:
    default:
        return QStringLiteral("normal");
    }
}

inline SegmentationDirectionFieldOrientation segmentationDirectionFieldOrientationFromInt(int value)
{
    switch (value) {
    case static_cast<int>(SegmentationDirectionFieldOrientation::Horizontal):
        return SegmentationDirectionFieldOrientation::Horizontal;
    case static_cast<int>(SegmentationDirectionFieldOrientation::Vertical):
        return SegmentationDirectionFieldOrientation::Vertical;
    default:
        return SegmentationDirectionFieldOrientation::Normal;
    }
}

struct SegmentationDirectionFieldConfig {
    QString path;
    SegmentationDirectionFieldOrientation orientation{SegmentationDirectionFieldOrientation::Normal};
    int scale{0};
    double weight{1.0};

    [[nodiscard]] bool isValid() const { return !path.isEmpty(); }
};

struct SegmentationCorrectionsPayload {
    struct Collection {
        uint64_t id{0};
        std::string name;
        std::vector<ColPoint> points;
        CollectionMetadata metadata;
        cv::Vec3f color{0.0f, 0.0f, 0.0f};
        std::optional<cv::Vec2f> anchor2d;  // 2D grid anchor for persistent corrections
    };

    std::vector<Collection> collections;

    [[nodiscard]] bool empty() const { return collections.empty(); }
};

struct SegmentationGrowthRequest {
    SegmentationGrowthMethod method{SegmentationGrowthMethod::Tracer};
    SegmentationGrowthDirection direction{SegmentationGrowthDirection::All};
    int steps{0};
    std::vector<SegmentationGrowthDirection> allowedDirections;
    SegmentationCorrectionsPayload corrections;
    std::optional<std::pair<int, int>> correctionsZRange;
    std::vector<SegmentationDirectionFieldConfig> directionFields;
    utils::Json customParams;
    cv::Mat allowedGrowthMask;
    bool manualAddPreview{false};
    bool inpaintOnly{false};
};

struct TracerGrowthContext {
    QuadSurface* resumeSurface{nullptr};
    class Volume* volume{nullptr};
    int level{0};
    QString cacheRoot;
    double voxelSize{1.0};
    QString normalGridPath;
    QString normal3dZarrPath;
    cv::Mat allowedGrowthMask;
    // For corrections annotation saving
    std::filesystem::path volpkgRoot;
    std::vector<std::string> volumeIds;
    std::string growthVolumeId;
    SurfacePatchIndex* surfacePatchIndex{nullptr};
    std::vector<std::shared_ptr<QuadSurface>> patchSurfaces;
};

struct TracerGrowthResult {
    QuadSurface* surface{nullptr};
    QString error;
    QString statusMessage;
    std::vector<std::filesystem::path> temporarySurfacePaths;
    QString patchTracerCacheSourcePath;
    std::vector<std::shared_ptr<QuadSurface>> patchTracerCachedSurfaces;
    std::shared_ptr<SurfacePatchIndex> patchTracerCachedIndex;
};

TracerGrowthResult runTracerGrowth(const SegmentationGrowthRequest& request,
                                   const TracerGrowthContext& context);
TracerGrowthResult runPatchTracerGrowth(const SegmentationGrowthRequest& request,
                                        const TracerGrowthContext& context);

void updateSegmentationSurfaceMetadata(QuadSurface* surface,
                                       double voxelSize);
