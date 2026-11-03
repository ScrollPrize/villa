#pragma once
#include "Fiber3DScene.hpp"
#include "Fiber3DRibbon.hpp"
#include <QWidget>
#include <memory>

class CState;
class LineAnnotationController;
class FiberCollectionController;
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
    Fiber3DRibbonControls(CState*, ViewerManager*, LineAnnotationController*, FiberCollectionController*, Fiber3DCanvas*, QToolButton*, QWidget* parent);
    ~Fiber3DRibbonControls() override;
    using FramesFactory = std::function<ribbon::Frames(const std::atomic_bool*)>;
    void setSelection(std::shared_ptr<const Scene>, const QString& key, FramesFactory canonicalSurface = {});
    void cancel();
    void resume();

private:
    struct Impl;
    std::unique_ptr<Impl> d;
};
}  // namespace vc3d::fibers3d
