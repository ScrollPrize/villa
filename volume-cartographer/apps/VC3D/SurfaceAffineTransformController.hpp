#pragma once

#include <QObject>
#include <QString>

#include <opencv2/core.hpp>

#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

class AxisAlignedSliceController;
class CState;
class QuadSurface;
class SegmentationModule;
class SurfacePanelController;
class ViewerControlsPanel;
class ViewerManager;
class ViewerTransformsPanel;
class QWidget;

class SurfaceAffineTransformController : public QObject
{
    Q_OBJECT

public:
    struct Deps {
        CState* state{nullptr};
        ViewerControlsPanel* viewerControlsPanel{nullptr};
        ViewerManager* viewerManager{nullptr};
        SegmentationModule* segmentationModule{nullptr};
        SurfacePanelController* surfacePanel{nullptr};
        AxisAlignedSliceController* axisAlignedSliceController{nullptr};
        QWidget* dialogParent{nullptr};
        std::function<void(const QString&, int)> showStatus;
    };

    explicit SurfaceAffineTransformController(const Deps& deps, QObject* parent = nullptr);

    void refresh();
    void clearPreview(bool restoreDisplayedSurface = true);

private:
    enum class RemoteTransformFetchState { Unknown, Pending, Available, Missing };

    void connectPanelSignals();
    void showStatus(const QString& text, int timeoutMs);
    ViewerTransformsPanel* transformsPanel() const;
    void ensureCurrentRemoteTransformJsonAsync();
    bool applyTransformPreview(bool allowRemoteFetch = true);
    std::shared_ptr<QuadSurface> currentTransformSourceSurface() const;
    QString currentTransformSourceDescription() const;
    bool setCustomTransformSource(const QString& source, QString* errorMessage = nullptr);
    std::filesystem::path localCurrentTransformJsonPath() const;
    std::string currentRemoteTransformJsonUrl() const;
    std::optional<cv::Matx44d> currentTransformMatrix(bool allowRemoteFetch = true);

    void onPreviewTransformToggled(bool enabled);
    void onSaveTransformedRequested();
    void onLoadAffineRequested();

    CState* _state{nullptr};
    ViewerControlsPanel* _viewerControlsPanel{nullptr};
    ViewerManager* _viewerManager{nullptr};
    SegmentationModule* _segmentationModule{nullptr};
    SurfacePanelController* _surfacePanel{nullptr};
    AxisAlignedSliceController* _axisAlignedSliceController{nullptr};
    QWidget* _dialogParent{nullptr};
    std::function<void(const QString&, int)> _showStatus;

    std::unordered_map<std::string, RemoteTransformFetchState> _remoteTransformFetchStates;
    std::unordered_map<std::string, cv::Matx44d> _remoteTransformMatrices;
    std::shared_ptr<QuadSurface> _transformPreviewSourceSurface;
    std::shared_ptr<QuadSurface> _transformPreviewSurface;
    QString _customTransformSource;
    std::filesystem::path _customTransformLocalPath;
    std::optional<cv::Matx44d> _customTransformMatrix;
};
