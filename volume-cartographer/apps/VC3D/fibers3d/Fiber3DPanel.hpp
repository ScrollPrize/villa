#pragma once

#include <QWidget>
#include <cstdint>
#include <memory>

class CState;
class ViewerManager;
class LineAnnotationController;
class FiberCollectionController;

namespace vc3d::fibers3d
{

// Main-workspace integration; the canvas and read-only data adapters are reusable
// without this panel (e.g. in the Line Annotation workspace).
class Fiber3DPanel : public QWidget
{
    Q_OBJECT
public:
    Fiber3DPanel(CState* state, ViewerManager* viewers, LineAnnotationController* annotations, FiberCollectionController* automated, QWidget* parent = nullptr);
    ~Fiber3DPanel() override;
    void selectNativeFiber(uint64_t id);
signals:
    void nativeFiberSelected(uint64_t id);
    void detachRequested();

protected:
    void showEvent(QShowEvent*) override;
    void hideEvent(QHideEvent*) override;

private:
    struct Impl;
    std::unique_ptr<Impl> d;
};

}  // namespace vc3d::fibers3d
