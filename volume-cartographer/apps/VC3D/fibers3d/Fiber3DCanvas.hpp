#pragma once

#include "Fiber3DScene.hpp"
#include "Fiber3DSlices.hpp"
#include "vc/core/util/Compositing.hpp"
#include <QOpenGLFunctions_3_3_Core>
#include <QOpenGLWidget>
#include <memory>
#include <optional>

namespace vc3d::fibers3d
{
namespace ribbon { struct Image; }

// Reusable display-only view. Loading, selection ownership and CT navigation
// belong to the controller; orbiting never reads the AFV again.
class Fiber3DCanvas : public QOpenGLWidget, protected QOpenGLFunctions_3_3_Core
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
    void setCTWindow(float low, float high);
    void setCTSlices(slices::Images);
    const slices::Images& ctSlices() const;
    void setCTSliceOpacity(float opacity);
    void setCTSliceTransfer(const std::array<CompositeParams, 3>&);
    void fitAll();
    void fitSelected();
    void focusPoint(Point point);
    // Navigation overlay at the CT focus, drawn above fibers for readability.
    // Unlike focusPoint(), updating it never moves the camera.
    void setCursorPosition(std::optional<Point> position);
    void clear();

signals:
    void visibilityChanged();
    void fiberPicked(QString key, qint64 linePointIndex, double x, double y, double z);
    void fiberContextRequested(QString key, qint64 linePointIndex, double x, double y, double z, QPoint globalPos);
    void emptyContextRequested(QPoint globalPos);
    void clearSelectionRequested();

protected:
    void initializeGL() override;
    void paintGL() override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;
    void contextMenuEvent(QContextMenuEvent* event) override;
    void keyPressEvent(QKeyEvent* event) override;

private:
    struct Impl;
    std::unique_ptr<Impl> d;
    void cleanupGL();
    void drawCTRibbon();
    void drawCTSlices();
    GLuint uploadCTTexture(const ct::Tile& tile);
    void startGeometryBuild();
    void updateSelectedGeometry();
    void updateProximity();
    void startProximityBuild();
    void visibilityUpdated();
    void fit(bool selectedOnly);
    void pick(const QPointF& position, bool context, const QPoint& globalPosition);
};

}  // namespace vc3d::fibers3d
