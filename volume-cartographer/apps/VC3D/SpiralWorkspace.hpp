#pragma once

#include <QMainWindow>
#include <QFutureWatcher>
#include <QHash>
#include <QImage>
#include <QJsonObject>
#include <QSet>
#include <QStringList>
#include <opencv2/core/types.hpp>
#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <vector>

class AxisAlignedSliceController;
class CState;
class ConsoleOutputWidget;
class QDialog;
class QKeyEvent;
class QuadSurface;
class SpiralPanel;
class SpiralServiceManager;
class ViewerManager;
class ViewerSplitGrid;
class VolumePkg;
class PolylineIndex;
class SpiralOverlayController;
class SegmentationOverlayController;

class SpiralWorkspace : public QMainWindow
{
    Q_OBJECT
public:
    explicit SpiralWorkspace(CState* mainState, QWidget* parent = nullptr);
    ~SpiralWorkspace() override;

    ViewerManager* viewerManager() const { return _viewerManager.get(); }

    // Cross-panel entry points for "Add to current spiral fit".
    bool hasActiveSpiralSession() const;
    void addPatchToCurrentFit(const QString& tifxyzDirectory,
                              const std::shared_ptr<QuadSurface>& surface = {});
    void addFiberToCurrentFit(const QString& fiberJsonPath);

signals:
    void spiralSessionActiveChanged(bool active);

protected:
    void keyPressEvent(QKeyEvent* event) override;

private:
    struct PreviewComponent {
        int firstColumn = 0;
        int endColumn = 0;
        int winding = 0;
    };
    struct PreviewLoadResult {
        std::shared_ptr<QuadSurface> surface;
        QString surfaceId;
        std::vector<PreviewComponent> components;
        QString error;
    };
    struct GeometryLoadResult {
        std::shared_ptr<PolylineIndex> index;
        QString error;
    };
    struct InputSurfaceEntry {
        QString category;
        QString id;
        QString sourceId;
        std::shared_ptr<QuadSurface> surface;
    };
    struct InputSurfaceLoadResult {
        std::vector<InputSurfaceEntry> surfaces;
        QStringList warnings;
    };

    void refreshVolumes();
    void selectVolume(const QString& id);
    QString mapServicePath(const QString& servicePath) const;
    void loadPreview(const QString& manifestPath, qint64 generation);
    void installPreview(const PreviewLoadResult& result, qint64 generation);
    void applyPreviewWindingRange(bool preserveFocus);
    void loadRunDiff(const std::shared_ptr<QuadSurface>& previous,
                     const std::vector<PreviewComponent>& previousComponents,
                     const std::shared_ptr<QuadSurface>& current,
                     const std::vector<PreviewComponent>& currentComponents,
                     qint64 generation);
    static QImage buildRunDiffImage(
        const std::shared_ptr<QuadSurface>& previous,
        const std::vector<PreviewComponent>& previousComponents,
        const std::shared_ptr<QuadSurface>& current,
        const std::vector<PreviewComponent>& currentComponents);
    void updateRunDiffOverlay();
    std::shared_ptr<QuadSurface> makeDisplayedPreview(QString& registrationId) const;
    void installPreviewAliasWhenIndexed(const std::shared_ptr<QuadSurface>& preview,
                                        const QString& registrationId,
                                        qint64 generation, quint64 revision,
                                        bool preserveFocus, int attempt);
    void loadGeometrySnapshot(const QString& manifestPath, quint64 generation);
    void loadInputSurfaces(const QJsonObject& paths, quint64 generation);
    void installInputSurfaces(const InputSurfaceLoadResult& result, quint64 generation);
    void registerPendingPatchSurface(const QString& inputId,
                                     const std::shared_ptr<QuadSurface>& surface);
    void setSurfaceCategoryVisible(const QString& category, bool visible);
    void updatePendingPatchIds(const QJsonObject& status);
    void updateSurfaceIntersections();
    void ensureInitialFocus();
    void initializePreviewFocus();
    void mirrorFocusToMainWorkspace(const cv::Vec3f& position);

    CState* _mainState = nullptr;
    CState* _state = nullptr;
    std::unique_ptr<ViewerManager> _viewerManager;
    std::unique_ptr<AxisAlignedSliceController> _slices;
    std::unique_ptr<SpiralOverlayController> _overlay;
    std::unique_ptr<SegmentationOverlayController> _surfaceOverlapOverlay;
    SpiralServiceManager* _service = nullptr;
    SpiralPanel* _panel = nullptr;
    ConsoleOutputWidget* _pythonOutput = nullptr;
    QDialog* _pythonOutputDialog = nullptr;
    ViewerSplitGrid* _grid = nullptr;
    qint64 _requestedPreviewGeneration = -1;
    QString _geometryManifestPath;
    QHash<QString, QStringList> _surfaceCategoryIds;
    QHash<QString, QString> _surfaceSourceIds;
    QHash<QString, bool> _surfaceCategoryVisible;
    QSet<QString> _pendingPatchIds;
    std::map<std::string, std::size_t> _surfaceOverlayColorAssignments;
    std::map<std::string, cv::Vec3b> _surfaceOverlayColors;
    std::size_t _nextSurfaceOverlayColorIndex = 0;
    quint64 _inputSurfaceGeneration = 0;
    std::shared_ptr<QuadSurface> _previewSource;
    QString _previewSourceId;
    std::vector<PreviewComponent> _previewComponents;
    std::shared_ptr<QuadSurface> _currentPreview;
    QString _currentPreviewRegistrationId;
    QImage _previewRunDiffImage;
    quint64 _previewDisplayRevision = 0;
    int _minimumDisplayedWinding = 10;
    int _maximumDisplayedWinding = -1;
    bool _outputVisible = true;
    bool _showSurfaceIntersections = true;
    bool _pendingPatchesOnly = false;
    bool _haveRunDiffBaseline = false;
    // True while the focus is the automatic volume-center default (no user
    // interaction and no preview yet); the first preview may then retarget it.
    bool _focusIsAutoDefault = false;
    bool _shuttingDown = false;
};
