#pragma once

#include <QMainWindow>
#include <QFutureWatcher>
#include <QHash>
#include <QJsonObject>
#include <QStringList>
#include <opencv2/core/types.hpp>
#include <memory>
#include <string>
#include <vector>

class AxisAlignedSliceController;
class CState;
class QKeyEvent;
class QuadSurface;
class SpiralPanel;
class SpiralServiceManager;
class ViewerManager;
class ViewerSplitGrid;
class VolumePkg;
class PolylineIndex;
class SpiralOverlayController;

class SpiralWorkspace : public QMainWindow
{
    Q_OBJECT
public:
    explicit SpiralWorkspace(CState* mainState, QWidget* parent = nullptr);
    ~SpiralWorkspace() override;

    ViewerManager* viewerManager() const { return _viewerManager.get(); }

protected:
    void keyPressEvent(QKeyEvent* event) override;

private:
    struct PreviewLoadResult {
        std::shared_ptr<QuadSurface> surface;
        QString surfaceId;
        QString error;
    };
    struct GeometryLoadResult {
        std::shared_ptr<PolylineIndex> index;
        QString error;
    };
    struct InputSurfaceEntry {
        QString category;
        QString id;
        std::shared_ptr<QuadSurface> surface;
    };
    struct InputSurfaceLoadResult {
        std::vector<InputSurfaceEntry> surfaces;
        QStringList warnings;
    };

    void refreshVolumes();
    void selectVolume(const QString& id);
    void loadPreview(const QString& manifestPath, qint64 generation);
    void installPreview(const PreviewLoadResult& result, qint64 generation);
    void installPreviewAliasWhenIndexed(const PreviewLoadResult& result, qint64 generation, int attempt);
    void loadGeometrySnapshot(const QString& manifestPath, quint64 generation);
    void loadInputSurfaces(const QJsonObject& paths, quint64 generation);
    void installInputSurfaces(const InputSurfaceLoadResult& result, quint64 generation);
    void setSurfaceCategoryVisible(const QString& category, bool visible);
    void updateSurfaceIntersections();
    void initializePreviewFocus();
    void setFocusAt(const cv::Vec3f& position, const cv::Vec3f& normal,
                    const std::string& surfaceId);
    void finishFocusChange();
    bool centerFocusOnCursor();

    CState* _mainState = nullptr;
    CState* _state = nullptr;
    std::unique_ptr<ViewerManager> _viewerManager;
    std::unique_ptr<AxisAlignedSliceController> _slices;
    std::unique_ptr<SpiralOverlayController> _overlay;
    SpiralServiceManager* _service = nullptr;
    SpiralPanel* _panel = nullptr;
    ViewerSplitGrid* _grid = nullptr;
    std::vector<std::pair<QString, std::shared_ptr<QuadSurface>>> _retiredPreviews;
    qint64 _requestedPreviewGeneration = -1;
    QString _geometryManifestPath;
    QHash<QString, QStringList> _surfaceCategoryIds;
    QHash<QString, bool> _surfaceCategoryVisible;
    quint64 _inputSurfaceGeneration = 0;
    std::shared_ptr<QuadSurface> _currentPreview;
    bool _outputVisible = true;
    bool _shuttingDown = false;
};
