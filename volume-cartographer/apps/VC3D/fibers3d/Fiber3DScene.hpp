#pragma once

#include "FiberOverlayColors.hpp"
#include <QColor>
#include <opencv2/core/types.hpp>
#include <QString>
#include <QStringList>
#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace vc3d::fibers3d
{

// Display snapshots only. Coordinates are doubles in the active CT volume's
// frame. Rendering never changes source geometry or creates annotations.
using Point = std::array<double, 3>;
struct Bounds {
    Point lower{}, upper{};
};
enum class Source { Annotation, Store };
enum class SpanStyle { Solid, Gap, InferredGap, Damaged };
enum class MarkerKind { ControlPoint, Break, KollesisTermination };
struct Marker {
    Point point{};
    MarkerKind kind{};
};
struct Run {
    std::vector<Point> points;
    // The index in the ORIGINAL fiber, including when a region clips its runs.
    int64_t firstPoint = 0;
    // One style per segment; an empty vector means all segments are solid.
    std::vector<SpanStyle> styles;
};
struct Fiber {
    std::string objectUuid;
    // Store UUID + object UUID; live Line Annotation has its own scene key.
    QString key, label, fileName;
    uint64_t id = 0;
    Source source = Source::Annotation;
    char family = '?';
    // Complete source-fiber length, not the length left after display clipping.
    // NaN means the source has no verified physical scale.
    double lengthMm = std::numeric_limits<double>::quiet_NaN();
    QStringList tags;
    std::vector<Run> runs;
    std::vector<Marker> markers;
    // Same per-control display overrides used by Line Annotation.
    std::vector<std::optional<cv::Vec3d>> displayNormals, controlDirections;
};
struct Scene {
    std::vector<Fiber> fibers;
    std::optional<Bounds> clipBounds;
    double voxelSizeMm = std::numeric_limits<double>::quiet_NaN();
    bool complete = true;
    QString message;
};

enum class ColorMode { Custom, Family, Direction, Length };
enum class DirectionMode { XYZ, AngleX, AngleY, AngleZ };
enum class Palette { Viridis, Magma, Cividis, Turbo };
struct DisplaySettings {
    ColorMode colorMode = ColorMode::Family;
    DirectionMode directionMode = DirectionMode::XYZ;
    Palette palette = Palette::Viridis;
    QColor custom{220, 220, 225};
    QColor horizontal = vc3d::horizontalFiberColor();
    QColor vertical = vc3d::verticalFiberColor();
    float lineWidth = 1.0f;
    float unselectedOpacity = 0.2f;
    // A pinned Line Annotation fiber stays selected for CT/picking, but keeps
    // its palette and width; the position marker supplies the emphasis.
    bool highlightSelection = true;
    bool showGaps = false;
    bool showAnnotations = false;
    bool annotationsSelectedOnly = true;
    bool perspective = false;
    bool showBounds = false;
    bool showCursor = true;
    bool nearbyOnly = false;
    double nearbyDistanceVoxels = 64.0;
    bool fixedLengthRange = false;
    double lengthMinMm = 0.0, lengthMaxMm = 10.0;
};

}  // namespace vc3d::fibers3d
