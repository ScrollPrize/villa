#pragma once
#include <QWidget>
#include "Fiber3DScene.hpp"
#include "scene3d/Scene3DGeometry.hpp"
#include <memory>

class CState;
class ViewerManager;
class QToolButton;
namespace vc3d::scene3d { class Scene3DCanvas; }
namespace vc3d::fibers3d
{
class Fiber3DSliceControls : public QWidget
{
public:
    Fiber3DSliceControls(CState*, ViewerManager*, scene3d::Scene3DCanvas*, QToolButton*, QWidget* parent, QString settingsGroup = QStringLiteral("fibers3d"));
    ~Fiber3DSliceControls() override;
    // Optional local XY-plane Z navigation for the general 3D viewer.
    void enableXYPositionControl();
    void resume();
    void cancel();
    // Annotation slices follow the live point instead of the main CT focus.
    void setCenter(std::optional<Point> center);
    // The standalone mesh viewer shows only XY, covering the visible mesh bounds.
    // Clearing the mesh restores the user's axis choices.
    void setMeshBounds(std::optional<scene3d::Bounds> bounds);

private:
    struct Impl;
    std::unique_ptr<Impl> d;
};
}  // namespace vc3d::fibers3d
