#pragma once

#include <QWidget>
#include <cstdint>
#include <memory>

class CState;
class ViewerManager;
class LineAnnotationController;
class FiberCollectionController;

namespace vc3d::line_annotation
{
struct GeneratedViews;
}

namespace vc3d::fibers3d
{

// Shared dock content for Main and Line Annotation. Annotation mode displays
// the open live line, independently of the global collection selection.
class Fiber3DPanel : public QWidget
{
    Q_OBJECT
public:
    Fiber3DPanel(CState* state, ViewerManager* viewers, LineAnnotationController* annotations, FiberCollectionController* automated, QWidget* parent = nullptr, bool lineAnnotation = false);
    ~Fiber3DPanel() override;
    void selectNativeFiber(uint64_t id);
    // Live display-only snapshot of the open line, independent of global AFV selection.
    void setLineAnnotationViews(const vc3d::line_annotation::GeneratedViews&, const QString& name, const QString& family);
    void setLineAnnotationPosition(double position);
    void clearLineAnnotation();
signals:
    void linePositionPicked(double position);
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
