#pragma once

#include <QString>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include <opencv2/core/mat.hpp>

#include "vc/ui/VCCollection.hpp"

class QuadSurface;
class Volume;

enum class SegmentationGrowthMethod {
    Tracer = 0,
    Corrections = 1,
    ManualAdd = 4,
};

inline QString segmentationGrowthMethodToString(SegmentationGrowthMethod method)
{
    switch (method) {
    case SegmentationGrowthMethod::Tracer:
        return QStringLiteral("Tracer");
    case SegmentationGrowthMethod::Corrections:
        return QStringLiteral("Corrections");
    case SegmentationGrowthMethod::ManualAdd:
        return QStringLiteral("Manual Add");
    }
    return QStringLiteral("Unknown");
}

inline SegmentationGrowthMethod segmentationGrowthMethodFromInt(int value)
{
    if (value == static_cast<int>(SegmentationGrowthMethod::Corrections)) {
        return SegmentationGrowthMethod::Corrections;
    }
    if (value == static_cast<int>(SegmentationGrowthMethod::ManualAdd)) {
        return SegmentationGrowthMethod::ManualAdd;
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
    nlohmann::json customParams;
    cv::Mat allowedGrowthMask;
    bool manualAddPreview{false};
    bool inpaintOnly{false};
};

struct TracerGrowthContext {
    QuadSurface* resumeSurface{nullptr};
    class Volume* volume{nullptr};
    int level{0};
    double voxelSize{1.0};
    QString normalGridPath;
    cv::Mat allowedGrowthMask;
    std::string growthVolumeId;
};

struct TracerGrowthResult {
    QuadSurface* surface{nullptr};
    QString error;
    QString statusMessage;
    // UUIDs of the surfaces a displacement copy created (front, back).
    std::vector<std::string> createdSurfaces;
};

// Give a surface without a "generations" channel one (every vertex generation
// 1), which tracer growth resumes from; false for a surface without points.
bool ensureGenerationsChannel(QuadSurface* surface);

TracerGrowthResult runTracerGrowth(const SegmentationGrowthRequest& request,
                                   const TracerGrowthContext& context);

