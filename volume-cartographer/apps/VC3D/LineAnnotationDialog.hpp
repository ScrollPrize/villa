#pragma once

#include <QDialog>
#include <QMetaObject>
#include <QPointer>

#include <memory>
#include <map>
#include <limits>
#include <string>
#include <vector>
#include <utility>

#include "volume_viewers/CChunkedVolumeViewer.hpp"

#include <opencv2/core/mat.hpp>

class CState;
class QComboBox;
class QMdiArea;
class QMdiSubWindow;
class QPushButton;
class QVBoxLayout;
class ViewerManager;

class LineAnnotationDialog : public QDialog
{
    Q_OBJECT

public:
    enum class InitialDirectionMode {
        Sideways,
        ZInOut,
    };

    struct Pane {
        std::string surfaceName;
        QPointer<CChunkedVolumeViewer> viewer;
        QPointer<QMdiSubWindow> subWindow;
    };

    struct GeneratedOverlay {
        std::vector<cv::Vec3f> linePoints;
        cv::Vec3f seedPoint{std::numeric_limits<float>::quiet_NaN(),
                            std::numeric_limits<float>::quiet_NaN(),
                            std::numeric_limits<float>::quiet_NaN()};
        cv::Vec3f pointMarker{std::numeric_limits<float>::quiet_NaN(),
                              std::numeric_limits<float>::quiet_NaN(),
                              std::numeric_limits<float>::quiet_NaN()};
        int seedLineIndex = -1;
        bool useSurfaceCenterLine = false;
    };

    explicit LineAnnotationDialog(ViewerManager* viewerManager, QWidget* parent = nullptr);

    CChunkedVolumeViewer* addPane(const std::string& surfaceName,
                                  const QString& title,
                                  const CChunkedVolumeViewer::CameraState& camera);
    bool setGeneratedRows(
        const std::vector<std::vector<std::pair<std::string, QString>>>& rows,
        const CChunkedVolumeViewer::CameraState& camera,
        const std::map<std::string, GeneratedOverlay>& overlays = {});
    const std::vector<Pane>& panes() const { return _panes; }
    InitialDirectionMode initialDirectionMode() const;

signals:
    void paneClosed(const std::string& surfaceName);
    void lineSeedRequested(const std::string& surfaceName, cv::Vec3f volumePoint, QPointF scenePoint);
    void showAsMeshRequested();

protected:
    void keyPressEvent(QKeyEvent* event) override;

private:
    void bindPaneInteractions(const std::string& surfaceName,
                              CChunkedVolumeViewer* viewer,
                              bool seedPlacementEnabled);
    void setGeneratedOverlay(const std::string& surfaceName,
                             CChunkedVolumeViewer* viewer,
                             const GeneratedOverlay& overlay);
    void applyGeneratedOverlay(const std::string& surfaceName,
                               CChunkedVolumeViewer* viewer,
                               const GeneratedOverlay& overlay);

    ViewerManager* _viewerManager = nullptr;
    QVBoxLayout* _layout = nullptr;
    QComboBox* _initialDirectionCombo = nullptr;
    QPushButton* _showAsMeshButton = nullptr;
    QMdiArea* _mdiArea = nullptr;
    std::vector<Pane> _panes;
    bool _suppressPaneClosed = false;
};
