#pragma once

#include <vector>
#include <QString>
#include <QColor>
#include <QPainterPath>
#include <opencv2/core.hpp>



/**
 * @brief Structure representing a drawn path
 * 
 * This structure is used to pass path data between widgets and the volume viewer.
 * Any widget can create paths and emit them for rendering.
 */
struct PathData {
    std::vector<cv::Vec3f> points;  ///< 3D points making up the path
    QColor color;                   ///< Color of the path
    float lineWidth = 3.0f;         ///< Width of the line when rendered
    QString id;                     ///< Unique identifier for the path
    QString ownerWidget;            ///< Widget that created this path
    
    // Optional metadata
    enum class PathType {
        FREEHAND,    ///< Continuous freehand drawing
        POLYLINE,    ///< Connected line segments
        SPLINE       ///< Smooth spline curve
    };
    PathType type = PathType::FREEHAND;
    
    enum class BrushShape {
        CIRCLE,      ///< Circular brush
        SQUARE       ///< Square brush
    };
    BrushShape brushShape = BrushShape::CIRCLE;
    
    float opacity = 1.0f;        ///< Opacity/transparency (0.0-1.0)
    bool isEraser = false;       ///< Flag for eraser mode
    int pathId = 0;             ///< Persistent ID for grouped paths
    
    // Constructor
    PathData() = default;
    PathData(const std::vector<cv::Vec3f>& pts, const QColor& col, const QString& owner = "")
        : points(pts), color(col), ownerWidget(owner) {}
    
    /**
     * @brief Create a densified version of this path using Qt's interpolation
     * @param samplingInterval Distance between sampled points in pixels
     * @return PathData with interpolated points along the same visual path
     */
    PathData densify(float samplingInterval = 0.5f) const;

private:
    /**
     * @brief Interpolate Z coordinate based on position along the path
     * @param percent Position along path (0.0 to 1.0)
     * @param totalLength Total length of the path
     * @param path The QPainterPath for reference
     * @return Interpolated Z coordinate
     */
    float interpolateZ(float percent, float totalLength, const QPainterPath& path) const;
};


