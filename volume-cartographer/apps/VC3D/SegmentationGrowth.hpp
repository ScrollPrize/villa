#pragma once

#include <QString>

class QuadSurface;
class Volume;
class ChunkCache;

enum class SegmentationGrowthMethod {
    Tracer = 0,
    Interpolate = 1,
};

inline QString segmentationGrowthMethodToString(SegmentationGrowthMethod method)
{
    switch (method) {
    case SegmentationGrowthMethod::Tracer:
        return QStringLiteral("Tracer");
    case SegmentationGrowthMethod::Interpolate:
        return QStringLiteral("Interpolate");
    }
    return QStringLiteral("Unknown");
}

inline SegmentationGrowthMethod segmentationGrowthMethodFromInt(int value)
{
    if (value == static_cast<int>(SegmentationGrowthMethod::Interpolate)) {
        return SegmentationGrowthMethod::Interpolate;
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

struct SegmentationGrowthRequest {
    SegmentationGrowthMethod method{SegmentationGrowthMethod::Tracer};
    SegmentationGrowthDirection direction{SegmentationGrowthDirection::All};
    int steps{0};
};

bool growSurfaceByInterpolation(QuadSurface* surface,
                                SegmentationGrowthDirection direction,
                                int steps,
                                QString* error = nullptr);

struct TracerGrowthContext {
    QuadSurface* resumeSurface{nullptr};
    class Volume* volume{nullptr};
    class ChunkCache* cache{nullptr};
    QString cacheRoot;
    double voxelSize{1.0};
};

struct TracerGrowthResult {
    QuadSurface* surface{nullptr};
    QString error;
    QString statusMessage;
};

TracerGrowthResult runTracerGrowth(const SegmentationGrowthRequest& request,
                                   const TracerGrowthContext& context);
