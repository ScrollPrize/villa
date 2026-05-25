#pragma once

#include <QObject>
#include <QPointF>
#include <QPointer>
#include <QString>

#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>

#include "vc/lasagna/LineOptimizer.hpp"

class CChunkedVolumeViewer;
class CState;
class LineAnnotationDialog;
class Surface;
class ViewerManager;
class QWidget;

class LineAnnotationController : public QObject
{
    Q_OBJECT

public:
    struct OptimizationTaskResult {
        bool ok = false;
        std::filesystem::path manifestPath;
        cv::Vec3d seedPoint{0.0, 0.0, 0.0};
        vc::lasagna::LineOptimizationResult result;
        std::string error;
    };

    using DatasetPicker =
        std::function<std::optional<std::string>(QWidget*, const std::filesystem::path&)>;
    using OptimizationTaskFactory =
        std::function<OptimizationTaskResult(std::filesystem::path, cv::Vec3d)>;

    LineAnnotationController(CState* state,
                             ViewerManager* viewerManager,
                             QWidget* parentWidget,
                             QObject* parent = nullptr);

    bool canLaunchFromViewer(const CChunkedVolumeViewer* viewer) const;
    void launchFromViewer(CChunkedVolumeViewer* viewer, const QPointF& scenePoint);

    void setDatasetPickerForTesting(DatasetPicker picker);
    void setOptimizationTaskFactoryForTesting(OptimizationTaskFactory factory);

private slots:
    void onSurfaceChanged(std::string name, std::shared_ptr<Surface> surf, bool isEditUpdate = false);

private:
    enum class SourceKind {
        Plane,
        Segmentation,
    };

    struct LineAnnotationSession;

    struct PaneRecord {
        int id = 0;
        SourceKind sourceKind = SourceKind::Plane;
        std::string surfaceName;
        QPointer<LineAnnotationDialog> dialog;
        std::shared_ptr<LineAnnotationSession> session;
    };

    std::string nextSurfaceName();
    void cleanupSurfaceName(const std::string& surfaceName);
    void handleLineSeed(const std::string& surfaceName, cv::Vec3f volumePoint);
    bool ensureDatasetForSession(LineAnnotationSession& session);
    void startOptimization(LineAnnotationSession& session, cv::Vec3d seedPoint);
    void finishOptimization(const std::string& surfaceName);
    bool materializeGeneratedViews(LineAnnotationSession& session);
    [[nodiscard]] PaneRecord* paneForSurface(const std::string& surfaceName);
    [[nodiscard]] const PaneRecord* paneForSurface(const std::string& surfaceName) const;
    [[nodiscard]] std::optional<std::string> pickDataset(QWidget* parent,
                                                          const std::filesystem::path& startDir) const;
    [[nodiscard]] OptimizationTaskResult runOptimizationTask(std::filesystem::path manifestPath,
                                                             cv::Vec3d seedPoint) const;
    void showError(const QString& message) const;

    CState* _state = nullptr;
    ViewerManager* _viewerManager = nullptr;
    QPointer<QWidget> _parentWidget;
    int _nextPaneId = 1;
    std::vector<PaneRecord> _panes;
    DatasetPicker _datasetPicker;
    OptimizationTaskFactory _optimizationTaskFactory;
};
