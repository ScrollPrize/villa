#pragma once

#include <QMainWindow>
#include <QFutureWatcher>
#include <QHash>
#include <QImage>
#include <QJsonObject>
#include <QColor>
#include <QPointer>
#include <QSet>
#include <QStringList>
#include <opencv2/core/types.hpp>
#include <cstddef>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <functional>

class AxisAlignedSliceController;
class CState;
class ConsoleOutputWidget;
class QDialog;
class QKeyEvent;
class QProgressDialog;
class QuadSurface;
class LasagnaServiceManager;
class SpiralPanel;
class SpiralServiceManager;
class ViewerManager;
class ViewerSplitGrid;
class VolumePkg;
class PolylineIndex;
class SpiralOverlayController;
class SpiralBrushController;
class SegmentationOverlayController;
class VolumeViewerBase;

class SpiralWorkspace : public QMainWindow
{
    Q_OBJECT
public:
    explicit SpiralWorkspace(CState* mainState, QWidget* parent = nullptr);
    ~SpiralWorkspace() override;

    ViewerManager* viewerManager() const { return _viewerManager.get(); }
    void setFiberViewDistance(double distance);

    // Cross-panel entry points for "Add to current spiral fit".
    bool hasActiveSpiralSession() const;
    void addPatchToCurrentFit(const QString& tifxyzDirectory,
                              const std::shared_ptr<QuadSurface>& surface = {});
    void addFiberToCurrentFit(const QString& fiberJsonPath);
    void requestSessionExit(std::function<void()> continuation);
    bool hasPendingBrushWork() const;

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
        struct LossMap {
            QString name;
            QString imagePath;
            double weight = 0.0;
            double p50 = 0.0;
            double p95 = 0.0;
            double maximum = 0.0;
            double displayMaximum = 0.0;
            qint64 sampleCount = 0;
            qint64 eligibleSampleCount = 0;
            qint64 projectedSampleCount = 0;
            qint64 offSurfaceSampleCount = 0;
            qint64 omittedSampleCount = 0;
            qint64 supportedPixels = 0;
        };
        std::vector<LossMap> lossMaps;
        bool connected = false;
    };
    struct PreviewDisplaySelection {
        int firstColumn = 0;
        int endColumn = 0;
        std::vector<PreviewComponent> diffComponents;
        std::vector<std::pair<int, int>> surfaceComponents;
        QString registrationId;
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
    void loadRunDiff();
    static QImage buildRunDiffImage(
        const std::shared_ptr<QuadSurface>& previous,
        const std::vector<PreviewComponent>& previousComponents,
        const std::shared_ptr<QuadSurface>& current,
        const std::vector<PreviewComponent>& currentComponents);
    void updateRunDiffOverlay();
    void updateLossMapOverlay();
    std::optional<PreviewDisplaySelection> displayedPreviewSelection() const;
    void installPreviewAliasWhenIndexed(const std::shared_ptr<QuadSurface>& preview,
                                        const QString& registrationId,
                                        qint64 generation, quint64 revision,
                                        bool preserveFocus, int attempt,
                                        std::vector<PreviewComponent> diffComponents = {});
    void loadGeometrySnapshot(const QString& manifestPath, quint64 generation);
    void loadInputSurfaces(const QJsonObject& paths, quint64 generation);
    void installInputSurfaces(const InputSurfaceLoadResult& result, quint64 generation);
    void registerPendingPatchSurface(const QString& inputId,
                                     const std::shared_ptr<QuadSurface>& surface,
                                     const std::optional<QColor>& color = std::nullopt);
    void finalizeBrushPaint();
    void maybeCommitForPendingExit();
    QString provisionalBrushRoot() const;
    void discardBrushWork();
    void setSurfaceCategoryVisible(const QString& category, bool visible);
    void updatePendingPatchIds(const QJsonObject& status);
    void updateSurfaceIntersections();
    void ensureInitialFocus();
    void initializePreviewFocus();
    void mirrorFocusToMainWorkspace(const cv::Vec3f& position);
    void updateLasagnaFlattenAvailability();
    QString lasagnaDataDirectorySettingsKey() const;
    QString resolveLasagnaDataDirectory();
    void startLasagnaFlatten();
    void updateLasagnaFlattenProgress(const QJsonObject& job);
    void failLasagnaFlatten(const QString& error, bool cancelled = false);
    void releaseLasagnaFlattenService();
    void closeLasagnaFlattenProgress();
    void handleLasagnaResults(const QString& outputDir,
                              const QStringList& segmentNames);

    CState* _mainState = nullptr;
    CState* _state = nullptr;
    std::unique_ptr<ViewerManager> _viewerManager;
    std::unique_ptr<AxisAlignedSliceController> _slices;
    std::unique_ptr<SpiralOverlayController> _overlay;
    std::unique_ptr<SpiralBrushController> _brush;
    std::unique_ptr<SegmentationOverlayController> _surfaceOverlapOverlay;
    SpiralServiceManager* _service = nullptr;
    SpiralPanel* _panel = nullptr;
    ConsoleOutputWidget* _pythonOutput = nullptr;
    QDialog* _pythonOutputDialog = nullptr;
    ViewerSplitGrid* _grid = nullptr;
    VolumeViewerBase* _flattenedViewer = nullptr;
    qint64 _requestedPreviewGeneration = -1;
    QString _geometryManifestPath;
    QJsonObject _sessionPaths;
    QHash<QString, QStringList> _surfaceCategoryIds;
    QHash<QString, QString> _surfaceSourceIds;
    QHash<QString, bool> _surfaceCategoryVisible;
    QSet<QString> _pendingPatchIds;
    QSet<QString> _visibleUncommittedPointCollectionIds;
    std::map<std::string, std::size_t> _surfaceOverlayColorAssignments;
    std::map<std::string, cv::Vec3b> _surfaceOverlayColors;
    std::size_t _nextSurfaceOverlayColorIndex = 0;
    quint64 _inputSurfaceGeneration = 0;
    std::shared_ptr<QuadSurface> _previewSource;
    QString _previewSourceId;
    std::vector<PreviewComponent> _previewComponents;
    bool _previewConnected = false;
    std::shared_ptr<QuadSurface> _runDiffPreviousSource;
    std::vector<PreviewComponent> _runDiffPreviousComponents;
    std::shared_ptr<QuadSurface> _currentPreview;
    std::vector<PreviewComponent> _currentPreviewComponents;
    QString _currentPreviewRegistrationId;
    QImage _previewRunDiffImage;
    QHash<QString, PreviewLoadResult::LossMap> _previewLossMaps;
    QString _selectedLossMap;
    QString _loadedLossMap;
    QImage _loadedLossMapImage;
    qreal _lossMapOpacity = 0.8;
    quint64 _previewDisplayRevision = 0;
    quint64 _runDiffRequestRevision = 0;
    int _minimumDisplayedWinding = 10;
    int _maximumDisplayedWinding = 130;
    bool _outputVisible = true;
    bool _showSurfaceIntersections = true;
    bool _showSurfaceOverlap = true;
    bool _pendingPatchesOnly = false;
    bool _haveRunDiffBaseline = false;
    bool _runDiffVisible = false;
    bool _flattenedPreviewActive = false;
    LasagnaServiceManager* _transientLasagnaManager = nullptr;
    LasagnaServiceManager* _activeLasagnaManager = nullptr;
    bool _lasagnaFlattenRunning = false;
    bool _lasagnaFlattenCancelRequested = false;
    QString _pendingLasagnaOutputDir;
    QString _pendingLasagnaOutputName;
    QString _pendingLasagnaJobId;
    std::shared_ptr<QuadSurface> _pendingLasagnaSource;
    QPointer<QProgressDialog> _lasagnaFlattenProgress;
    // True while the focus is the automatic volume-center default (no user
    // interaction and no preview yet); the first preview may then retarget it.
    bool _focusIsAutoDefault = false;
    bool _shuttingDown = false;
    struct PendingBrushPatch {
        QString path;
        QColor color;
        std::shared_ptr<QuadSurface> surface;
    };
    QHash<QString, PendingBrushPatch> _pendingBrushPatches;
    QHash<QString, QString> _brushProvisionalPaths;
    QSet<QString> _unverifiedBrushIds;
    QHash<QString, QString> _pendingPointCollectionPaths;
    QHash<QString, QString> _pointCollectionProvisionalPaths;
    QSet<QString> _uncommittedPointCollectionIds;
    std::function<void()> _pendingExitAction;
    bool _commitAfterBrushUploads = false;
};
