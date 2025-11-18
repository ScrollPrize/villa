#pragma once

#include <QString>

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json_fwd.hpp>

#include <opencv2/core.hpp>

#include "vc/ui/VCCollection.hpp"

class QuadSurface;
class Volume;
template <typename T> class ChunkCache;

enum class SegmentationGrowthMethod {
    Tracer = 0,
    Corrections = 1,
};

inline QString segmentationGrowthMethodToString(SegmentationGrowthMethod method)
{
    switch (method) {
    case SegmentationGrowthMethod::Tracer:
        return QStringLiteral("Tracer");
    case SegmentationGrowthMethod::Corrections:
        return QStringLiteral("Corrections");
    }
    return QStringLiteral("Unknown");
}

inline SegmentationGrowthMethod segmentationGrowthMethodFromInt(int value)
{
    if (value == static_cast<int>(SegmentationGrowthMethod::Corrections)) {
        return SegmentationGrowthMethod::Corrections;
    }
    return SegmentationGrowthMethod::Tracer;
}

enum class SegmentationGrowthDirection {
    All = 0,
    Up,
    Down,
    Left,
    Right,
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
    std::optional<nlohmann::json> customParams;
    bool inpaintOnly{false};
};

struct TracerGrowthContext {
    QuadSurface* resumeSurface{nullptr};
    class Volume* volume{nullptr};
    class ChunkCache<uint8_t>* cache{nullptr};
    QString cacheRoot;
    double voxelSize{1.0};
    QString normalGridPath;
};

struct TracerGrowthResult {
    QuadSurface* surface{nullptr};
    QString error;
    QString statusMessage;
};

TracerGrowthResult runTracerGrowth(const SegmentationGrowthRequest& request,
                                   const TracerGrowthContext& context);

void updateSegmentationSurfaceMetadata(QuadSurface* surface,
                                       double voxelSize);
