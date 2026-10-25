#pragma once
#include <QWidget>
#include <memory>

class CState;
class ViewerManager;
class QToolButton;
namespace vc3d::fibers3d
{
class Fiber3DCanvas;
class Fiber3DSliceControls : public QWidget
{
public:
    Fiber3DSliceControls(CState*, ViewerManager*, Fiber3DCanvas*, QToolButton*, QWidget* parent);
    ~Fiber3DSliceControls() override;
    void resume();
    void cancel();

private:
    struct Impl;
    std::unique_ptr<Impl> d;
};
}  // namespace vc3d::fibers3d
