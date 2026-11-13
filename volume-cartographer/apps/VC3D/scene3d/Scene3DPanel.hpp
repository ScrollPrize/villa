#pragma once
#include <QWidget>

class CState;
class ViewerManager;
class QComboBox;
namespace vc3d::fibers3d {
class Fiber3DSliceControls;
class Fiber3DSurfaceControls;
}
namespace vc3d::scene3d {
class Scene3DCanvas;

// General CT/surface inspection, independent of fiber controllers or selection.
class Scene3DPanel : public QWidget
{
    Q_OBJECT
public:
    Scene3DPanel(CState*, ViewerManager*, QWidget* parent = nullptr);

signals:
    void attachVolumeRequested();
    void importSurfaceRequested();
    void detachRequested();

protected:
    void showEvent(QShowEvent*) override;
    void hideEvent(QHideEvent*) override;

private:
    Scene3DCanvas* canvas_;
    fibers3d::Fiber3DSliceControls* slices_;
    fibers3d::Fiber3DSurfaceControls* surface_;
    QComboBox* displayMode_;
    bool savedCT_ = true;
};
}
