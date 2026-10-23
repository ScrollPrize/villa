#pragma once

#include "Fiber3DScene.hpp"
#include <QOpenGLFunctions_3_3_Core>
#include <QOpenGLWidget>
#include <memory>
#include <optional>

namespace vc3d::fibers3d
{

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
    void setDisplaySettings(const DisplaySettings& settings);
    void setSelectedKey(const QString& key);
    void fitAll();
    void fitSelected();
    void focusPoint(Point point);
    // Navigation overlay at the CT focus, drawn above fibers for readability.
    // Unlike focusPoint(), updating it never moves the camera.
    void setCursorPosition(std::optional<Point> position);
    void clear();

signals:
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
    void startGeometryBuild();
    void updateSelectedGeometry();
    void fit(bool selectedOnly);
    void pick(const QPointF& position, bool context, const QPoint& globalPosition);
};

}  // namespace vc3d::fibers3d
