#pragma once
#include "Scene3DCanvas.hpp"
#include "VolumePreview.hpp"
#include "fibers3d/Fiber3DSurfaceRenderer.hpp"
#include "fibers3d/Fiber3DVolumeRenderer.hpp"
#include <QTimer>
#include <unordered_map>
#include <cmath>
#include <algorithm>

namespace vc3d::scene3d
{
inline QVector3D relative(const Point& p, const Point& origin)
{
    return {float(p[0]-origin[0]), float(p[1]-origin[1]), float(p[2]-origin[2])};
}
struct Scene3DCanvas::State {
    bool surfaceVisible = true, surfaceTextured = true;
    std::shared_ptr<const surface::Mesh> surfaceMesh;
    std::unique_ptr<fibers3d::Fiber3DSurfaceRenderer> surfaceRenderer;
    float surfaceOpacity = .5f;
    bool surfaceUploading = false;
    slices::Images slices;
    std::shared_ptr<const VolumeData> ctVolume;
    bool volumeUploading = false;
    float volumeDensity = 8.f;
    std::unique_ptr<fibers3d::Fiber3DVolumeRenderer> volumeRenderer;
    std::array<CompositeParams, 3> sliceTransfer;
    std::unique_ptr<slices::Tree> sliceTree;
    std::vector<GLuint> sliceTextures;
    std::unordered_map<const slices::Polygon*, std::pair<GLint, GLsizei>> sliceRanges;
    GLuint sliceBuffer = 0;
    Point sliceOrigin{};
    bool slicesDirty = false;
    float sliceOpacity = .7f;
    std::unique_ptr<QOpenGLShaderProgram> ctProgram;
    GLuint ctVao = 0;
    float windowLow = 0, windowHigh = 255;
    QString ctError;
    QTimer refine;
    bool interactive = false;
    Point origin{}, target{};
    QQuaternion rotation = QQuaternion::fromEulerAngles(-25.0f, 30.0f, 0.0f);
    double radius = 100.0, scale = 100.0;
    bool fitted = false, glReady = false, dragging = false;
    Qt::MouseButton pressed = Qt::NoButton;
    QPointF pressPosition, previousPosition;
    QMatrix4x4 matrix(int width, int height) const
    {
        const float aspect = float(std::max(1, width)) / std::max(1, height);
        QMatrix4x4 projection, view;
        // Preserve apparent scale when switching between perspective and ortho.
        const double eye = scale / std::tan(0.39269908169872415481);
        double extent = radius + distance(target, origin);
        if (surfaceVisible && surfaceMesh && !surfaceMesh->indices.empty())
            extent = std::max(extent, distance(target, surfaceMesh->bounds.lower) + distance(surfaceMesh->bounds.lower, surfaceMesh->bounds.upper));
        if (ctVolume)
            extent = std::max(extent, distance(target, ctVolume->bounds.lower) + distance(ctVolume->bounds.lower, ctVolume->bounds.upper));
        for (const auto& image : slices) {
            for (const auto& p : slices::corners(*image))
                extent = std::max(extent, distance(target, p));
        }
        const double farPlane = eye + 2.0 * extent + scale;
        if (perspective)
            projection.perspective(45.0f, aspect, float(std::max(0.001, scale * 0.001)), float(farPlane));
        else
            projection.ortho(float(-scale * aspect), float(scale * aspect), float(-scale), float(scale), float(-farPlane), float(farPlane));
        if (perspective)
            view.translate(0.0f, 0.0f, float(-eye));
        view.rotate(rotation);
        view.translate(-relative(target, origin));
        return projection * view;
    }
    bool perspective = false;

};
}
