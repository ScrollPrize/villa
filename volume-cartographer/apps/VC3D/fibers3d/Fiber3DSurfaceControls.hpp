#pragma once
#include <QWidget>
#include <memory>

class CState;
namespace vc3d::fibers3d
{
class Fiber3DCanvas;
class Fiber3DSurfaceControls : public QWidget
{
public:
    Fiber3DSurfaceControls(CState*, Fiber3DCanvas*, QWidget* parent);
    ~Fiber3DSurfaceControls() override;
    void resume();
    void suspend();
    void setMemoryBudget(size_t bytes);
private:
    struct Impl;
    std::unique_ptr<Impl> d;
};
}
