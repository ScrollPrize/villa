#pragma once
#include "fibers3d/Fiber3DSlices.hpp"
#include "fibers3d/Fiber3DSurfaceMesh.hpp"
#include "vc/core/util/Compositing.hpp"
#include <QOpenGLWidget>
#include <QOpenGLFunctions_3_3_Core>
#include <memory>

namespace vc3d::scene3d
{
struct VolumeData;
namespace slices = fibers3d::slices;
namespace surface = fibers3d::surface;
namespace ct = fibers3d::ct;

// Display-only OpenGL viewport in native CT coordinates. Works without fibers.
// Controllers own loading and sampling; camera motion reuses resident data.
class Scene3DCanvas : public QOpenGLWidget, protected QOpenGLFunctions_3_3_Core
{
    Q_OBJECT
public:
    explicit Scene3DCanvas(QWidget* parent = nullptr);
    ~Scene3DCanvas() override;
    void clearContent();
    void setCTSlices(slices::Images);
    const slices::Images& ctSlices() const;
    void setCTVolume(std::shared_ptr<const VolumeData>);
    std::shared_ptr<const VolumeData> ctVolume() const;
    bool isCTVolumeReady() const;
    void setCTVolumeDensity(float);
    void setCTWindow(float low, float high);
    void setCTSliceOpacity(float opacity);
    void setCTSliceTransfer(const std::array<CompositeParams, 3>&);
    virtual void setSurfaceMesh(std::shared_ptr<const surface::Mesh>);
    std::shared_ptr<const surface::Mesh> surfaceMesh() const;
    bool isSurfaceMeshReady() const;
    void setSurfaceTextured(bool enabled);
    void setSurfaceOpacity(float opacity);
    void setSurfaceVisible(bool visible);
    bool isSurfaceVisible() const;
    void setPerspective(bool perspective);
    float pixelsPerVoxel() const;
    virtual void fitAll();
    virtual void fitSurface();
    void focusPoint(Point point);

signals:
    void surfaceDisplayChanged();
    void viewScaleChanged();
    void positionClicked(QPointF position, bool context, QPoint globalPosition);
    void emptyContextRequested(QPoint globalPosition);
    void clearSelectionRequested();

protected:
    struct State;
    std::unique_ptr<State> v;
    void initializeGL() override;
    void paintGL() override;
    void resizeGL(int, int) override;
    void mousePressEvent(QMouseEvent*) override;
    void mouseMoveEvent(QMouseEvent*) override;
    void mouseReleaseEvent(QMouseEvent*) override;
    void wheelEvent(QWheelEvent*) override;
    void contextMenuEvent(QContextMenuEvent*) override;
    void keyPressEvent(QKeyEvent*) override;
    void beginFrame();
    void drawSurface();
    void drawCTSlices();
    GLuint uploadCTTexture(const ct::Tile& tile);
    void fitBounds(const Bounds&);

private:
    void cleanupGL();
};
}
