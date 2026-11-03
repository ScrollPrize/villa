#pragma once
#include <QWidget>
#include <memory>
#include <functional>

class CState;
namespace vc3d::scene3d { class Scene3DCanvas; }
namespace vc3d::fibers3d
{
class Fiber3DSurfaceControls : public QWidget
{
public:
    Fiber3DSurfaceControls(CState*, scene3d::Scene3DCanvas*, QWidget* parent,
                           QString settingsGroup = QStringLiteral("fibers3d"),
                           std::function<void(bool, double)> filterChanged = {});
    ~Fiber3DSurfaceControls() override;
    void resume();
    void suspend();
    void setMemoryBudget(size_t bytes);
    // Standalone viewer: derive the display surface from the active CT volume.
    void setVolumeSource(bool);

private:
    struct Impl;
    std::unique_ptr<Impl> d;
};
}  // namespace vc3d::fibers3d
