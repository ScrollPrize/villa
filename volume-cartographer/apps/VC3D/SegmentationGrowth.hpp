#pragma once

#include <QString>

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

#include "vc/ui/VCCollection.hpp"

class QuadSurface;
class Volume;
class ChunkCache;

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
};

struct TracerGrowthContext {
    QuadSurface* resumeSurface{nullptr};
    class Volume* volume{nullptr};
    class ChunkCache* cache{nullptr};
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
