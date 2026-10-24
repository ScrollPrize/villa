#pragma once
#include "Fiber3DScene.hpp"
#include <QWidget>
#include <memory>

class CState;
class ViewerManager;
class QToolButton;
namespace vc3d::fibers3d
{
class Fiber3DCanvas;

// Owns one selected-fiber CT request. Workers only see immutable snapshots and
// keep their volume alive; late results cannot restore a previous selection.
class Fiber3DRibbonControls : public QWidget
{
public:
    Fiber3DRibbonControls(CState*, ViewerManager*, Fiber3DCanvas*, QToolButton*, QWidget* parent);
    ~Fiber3DRibbonControls() override;
    void setSelection(std::shared_ptr<const Scene>, const QString& key);
    void cancel();

private:
    struct Impl;
    std::unique_ptr<Impl> d;
};
}  // namespace vc3d::fibers3d
