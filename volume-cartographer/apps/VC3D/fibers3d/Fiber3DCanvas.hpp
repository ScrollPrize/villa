#pragma once

#include "Fiber3DScene.hpp"
#include "Fiber3DSlices.hpp"
#include "vc/core/util/Compositing.hpp"
#include "scene3d/Scene3DCanvas.hpp"
#include <QOpenGLWidget>
#include <memory>
#include <optional>

namespace vc3d::fibers3d
{
namespace ribbon
{
struct Image;
}
namespace surface
{
struct Mesh;
}

// Reusable display-only view. Loading, selection ownership and CT navigation
// belong to the controller; orbiting never reads the store again.
class Fiber3DCanvas : public scene3d::Scene3DCanvas
{
    Q_OBJECT
public:
    explicit Fiber3DCanvas(QWidget* parent = nullptr);
    ~Fiber3DCanvas() override;
    void setScene(std::shared_ptr<const Scene> scene);
    std::shared_ptr<const Scene> scene() const;
    bool isPreparing() const;
    bool isFiltering() const;
    size_t visibleFiberCount() const;
    void setDisplaySettings(const DisplaySettings& settings);
    void setSelectedKey(const QString& key);
    void setCTRibbon(std::shared_ptr<const ribbon::Image> image);
    std::shared_ptr<const ribbon::Image> ctRibbon() const;
    void setSurfaceMesh(std::shared_ptr<const surface::Mesh>) override;
    void setSurfaceFilter(bool enabled, double distanceVoxels);
    void fitAll() override;
    void fitSurface() override;
    void fitSelected();
    // Navigation overlay at the CT focus, drawn above fibers for readability.
    void setCursorPosition(std::optional<Point> position);
    void setLinePosition(std::optional<Point> position);
    void clear();

signals:
    void visibilityChanged();
    void fiberPositionPicked(QString key, double linePosition);
    void fiberPicked(QString key, qint64 linePointIndex, double x, double y, double z);
    void fiberContextRequested(QString key, qint64 linePointIndex, double x, double y, double z, QPoint globalPos);

protected:
    void initializeGL() override;
    void paintGL() override;

private:
    struct Impl;
    std::unique_ptr<Impl> d;
    void cleanupGL();
    void drawCTRibbon();
    void startGeometryBuild();
    void updateSelectedGeometry();
    void updateProximity();
    void startProximityBuild();
    void visibilityUpdated();
    void fit(bool selectedOnly);
    void pick(const QPointF& position, bool context, const QPoint& globalPosition);
};

}  // namespace vc3d::fibers3d
