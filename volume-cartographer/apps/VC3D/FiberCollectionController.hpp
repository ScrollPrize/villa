#pragma once

#include "FiberCollectionGenerator.hpp"
#include "overlays/ViewerOverlayControllerBase.hpp"
#include "vc/core/types/FiberCollection.hpp"
#include <QPointF>
#include <QPointer>
#include <QStringList>
#include <atomic>
#include <memory>
#include <limits>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <vector>

class CState;
class QDockWidget;
class QLabel;
class QListWidget;
class QSpinBox;
class QDoubleSpinBox;
class QCheckBox;
class QSlider;
class QLineEdit;
class QPushButton;
class QProcess;
class QProgressBar;
class QTemporaryDir;
class QMainWindow;
class LineAnnotationController;

// Opening a volume keeps its catalog outside the native annotation machinery.
// Explicit inspection materializes only the selected fiber as an editable copy.
class FiberCollectionController : public ViewerOverlayControllerBase
{
    Q_OBJECT
public:
    FiberCollectionController(CState* state, ViewerManager* manager, QMainWindow* window, LineAnnotationController* annotations);
    ~FiberCollectionController() override;
    QDockWidget* dock() const { return dock_; }
    void setLineAnnotationActive(bool active);
    void openCollection(const QString& path, bool persist = true, const std::string& expectedUuid = {});
    // Selects the trace under a plain left click that no native tool, collection
    // point or fiber annotation used. Returns true if a trace was selected.
    bool handleVolumeClick(Qt::MouseButton button, Qt::KeyboardModifiers modifiers);
    void detachViewer(VolumeViewerBase* viewer) override;
protected:
    bool isOverlayEnabledFor(VolumeViewerBase* viewer) const override;
    bool needsOverlayRebuild(VolumeViewerBase* viewer) const override;
    void collectPrimitives(VolumeViewerBase* viewer, OverlayBuilder& builder) override;
    bool eventFilter(QObject* watched, QEvent* event) override;

private:
    struct View;
    using Slice = vc::fibers::ViewRegion;
    CState* state_;
    ViewerManager* manager_;
    LineAnnotationController* annotations_;
    QDockWidget* dock_{};
    QLabel *status_{}, *selection_{}, *annotationStatus_{};
    QListWidget* list_{};
    QDoubleSpinBox* minLength_{};
    QSpinBox* maxDisplayed_{};
    QDoubleSpinBox* distance_{};
    QDoubleSpinBox* lineWidth_{};
    QCheckBox* visible_{};
    QCheckBox* rainbow_{};
    QCheckBox* horizontal_{};
    QCheckBox* vertical_{};
    QSlider* along_{};
    QWidget* selectionSection_{};
    QWidget* selectionSeparator_{};
    QLineEdit* idInput_{};
    QPushButton* next_{};
    QPushButton* openAnnotation_{};
    QPushButton* detach_{};
    // A background run of the Python generator, creating a new volume.
    QPushButton* create_{};
    QProgressBar* generatorProgress_{};
    QPushButton* generatorCancel_{};
    QLabel* generatorStatus_{};
    QPointer<QProcess> generator_;
    QString generatorOutput_, generatorError_, generatorSpace_, generatorPython_;
    QStringList generatorLog_;
    // Receives the generator's previews; removed with them when the run ends.
    std::unique_ptr<QTemporaryDir> generatorPreview_;
    // The blocks of the zone being chosen or generated and their states,
    // drawn on the CT views.
    std::vector<vc3d::fibergen::Block> generationBlocks_;
    std::vector<QString> generationStates_;
    uint64_t generationRevision_{0};
    QString path_, attachment_;
    std::string uuid_, coordinateSpace_;
    double sourceResolution_{};
    int64_t totalFibers_{};
    bool annotationBusy_{false};
    bool lineAnnotationActive_{false};
    int64_t pendingAnnotation_{};
    double nativeToViewer_{1.0};
    double nativeVoxelMm_{1.0};
    int64_t selected_{0}, lastRow_{0};
    double lastLength_{std::numeric_limits<double>::infinity()};
    uint64_t revision_{0}, selectionRevision_{0}, navigationRevision_{0}, annotationRevision_{0};
    bool enabled_{false}, catalogBusy_{false}, catalogPending_{false}, navigationBusy_{false};
    struct Click {
        QPointer<QObject> viewport;
        QPointF position;
    };
    std::optional<Click> press_, click_;
    std::vector<QMetaObject::Connection> annotationRenderConnections_;
    std::optional<int> pendingIndex_;
    std::unordered_map<VolumeViewerBase*, std::unique_ptr<View>> views_;
    std::unordered_map<VolumeViewerBase*, std::tuple<uint64_t, QRectF, std::optional<Slice>>> drawnGeneration_;
    void clear();
    void invalidate();
    void showInLineAnnotation(int64_t id);
    void startPendingAnnotation();
    void finishAnnotationRequest();
    void watchAnnotationRender(class LineAnnotationDialog* dialog, int64_t id, uint64_t token);
    void projectChanged();
    void volumeChanged();
    void listPage(bool reset);
    vc::fibers::FamilyFilter familyFilter() const;
    void selectFiber(int64_t id, bool focusView = true, bool openAnnotation = false);
    bool highlightListFiber(int64_t id);
    void revealListFiber(const vc::fibers::Summary& summary);
    void navigate(int index);
    void request(VolumeViewerBase* viewer, const Slice& slice);
    std::optional<Slice> slice(VolumeViewerBase* viewer) const;
    QString attachmentPath() const;
    void createCollection();
    void readGeneratorOutput();
    void finishGenerator(int exitCode, bool crashed);
    void showGeneratorStatus(const QString& text);
    void showGenerationBlocks(std::vector<vc3d::fibergen::Block> blocks);
    void drawGenerationBlocks(VolumeViewerBase* viewer, OverlayBuilder& builder);
};
