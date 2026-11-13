#pragma once

#include "InkPreviewClient.hpp"
#include "ViewerOverlayControllerBase.hpp"

#include "vc/core/util/RenderCanvas.hpp"

#include <QImage>
#include <QPixmap>
#include <QRect>
#include <QRectF>
#include <QString>
#include <QUrl>

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>

class CState;
class QTimer;
struct InkPreviewSnapshot;
class QuadSurface;
class Volume;
class VolumeViewerBase;

// The live ink preview of the Ink Detection dock (docs/ink_preview.md): an ink
// model's output for the part of the segment on screen, from a model server the
// user started. Off by default. When on, it watches the segmentation viewer and,
// once the view has settled, asks the server for the visible part of the segment's
// render canvas (vc::RenderCanvas: what vc_render_tifxyz --scale 1 -g LEVEL writes).
// When the server asks for layers, it samples them from a copy of the surface in
// memory and the open volume with vc::sampleRenderLayers on a worker thread, posts
// them, and draws the reply placed by its canvas rectangle, so the image follows
// pans and zooms while the next one is computed. One request is in flight at a time.
class InkPreviewOverlayController : public ViewerOverlayControllerBase
{
    Q_OBJECT

public:
    static constexpr int kAutoLevel = -1;
    static constexpr qint64 kViewBudgetPx = 4 * 1024 * 1024;
    static constexpr int kSettleMs = 150;

    // A level of the current volume as the preview samples it.
    struct LevelInfo {
        int level = 0;
        float voxelScale = 1.0f;  // level voxels per level-0 voxel: the render scale of its canvas
        double voxelUm = 0.0;     // 0 when the volume's voxel size is unknown
    };

    explicit InkPreviewOverlayController(CState* state, QObject* parent = nullptr);
    ~InkPreviewOverlayController() override;

    [[nodiscard]] bool enabled() const { return _enabled; }
    [[nodiscard]] QUrl serverUrl() const;
    [[nodiscard]] bool reversed() const { return _reversed; }
    // kAutoLevel or a pyramid level of the volume.
    [[nodiscard]] int levelChoice() const { return _levelChoice; }
    [[nodiscard]] int opacity() const { return _opacity; }
    [[nodiscard]] const QString& statusText() const { return _status; }
    [[nodiscard]] bool statusIsError() const { return _statusIsError; }
    [[nodiscard]] const std::optional<InkPreviewClient::Health>& health() const { return _health; }
    // The current volume's levels, once read; empty before.
    [[nodiscard]] const std::vector<LevelInfo>& levels() const { return _levels; }
    // The level requests use: the choice, or for Auto the level whose voxel size is
    // nearest the model's training voxel size (level 0 when either is unknown).
    [[nodiscard]] std::optional<int> resolvedLevel() const;

#ifdef VC_TESTING
    // Stand-ins for the server's health and the volume's levels, so the geometry can
    // be tested without either.
    void setHealthForTesting(const InkPreviewClient::Health& health);
    void setLevelsForTesting(const std::string& volumeId, const std::vector<LevelInfo>& levels);
    // Runs what the settle timer runs.
    void settleForTesting();
    [[nodiscard]] bool settlePendingForTesting() const;
    [[nodiscard]] int requestsStartedForTesting() const { return _requestsStarted; }
    // The key, canvas and visible canvas rectangle the last request decision used.
    [[nodiscard]] QString keyForTesting(VolumeViewerBase* viewer) const;
    [[nodiscard]] std::optional<vc::RenderCanvas> canvasForTesting(VolumeViewerBase* viewer) const;
    [[nodiscard]] std::optional<QRect> visibleRectForTesting(VolumeViewerBase* viewer) const;
    [[nodiscard]] QRect requestedRectForTesting() const { return _lastRequested; }
    // Installs `pixels` (one byte per pixel) as the reply for canvas rectangle `rect`.
    void showForTesting(VolumeViewerBase* viewer, const QRect& rect, const cv::Mat_<uint8_t>& pixels);
    [[nodiscard]] InkPreviewClient* clientForTesting() const { return _client; }
#endif

public slots:
    void setEnabled(bool enabled);
    void setServerUrl(const QUrl& url);
    void setReversed(bool reversed);
    void setLevelChoice(int level);
    void setOpacity(int opacity);
    void setColormapId(const std::string& id);
    // Opacity 0 and back, as the Space key does for the published detections.
    void toggleVisibility();

signals:
    void enabledChanged(bool enabled);
    void statusChanged(const QString& text, bool error);
    void levelsChanged();
    void opacityChanged(int opacity);

protected:
    bool isOverlayEnabledFor(VolumeViewerBase* viewer) const override;
    void collectPrimitives(VolumeViewerBase* viewer, OverlayBuilder& builder) override;
    void detachViewer(VolumeViewerBase* viewer) override;

private:
    // What decides the voxels under the canvas, and the canvas.
    struct Frame {
        std::string surfaceId;
        std::string volumeId;
        int level = 0;
        vc::RenderCanvas canvas;
        QString key;
    };
    struct Shown {
        Frame frame;
        bool reversed = false;
        QRect rect;  // canvas pixels
        cv::Mat_<uint8_t> scalar;
        QImage image;
        QPixmap pixmap;
    };
    struct InFlight {
        quint64 id = 0;
        Frame frame;
        bool reversed = false;
        QRect rect;
        quint64 edit = 0;
        int uploads = 0;
        std::shared_ptr<std::atomic<bool>> cancel;
    };

    bool isSegmentationView(VolumeViewerBase* viewer) const;
    std::optional<Frame> frameFor(VolumeViewerBase* viewer, const QuadSurface& surface) const;
    std::optional<QRect> visibleCanvasRect(VolumeViewerBase* viewer, const QuadSurface& surface,
                                           const vc::RenderCanvas& canvas) const;
    QPointF canvasToScene(VolumeViewerBase* viewer, const QuadSurface& surface, const vc::RenderCanvas& canvas,
                          double x, double y) const;
    std::vector<float> layerOffsets() const;
    qint64 viewBudget() const;

    void ensureHealth();
    void ensureLevels(const std::shared_ptr<Volume>& volume);
    void startRequest();
    void send(const InkPreviewClient::Upload* upload);
    void onReply(quint64 id, const InkPreviewClient::Reply& reply);
    void sampleNeed(const QRect& need);
    void finishFlight(bool rerun);
    void cancelFlight();
    void dropShown();
    void install(const InkPreviewClient::Reply& reply);
    void setStatus(const QString& text, bool error = false);
    QString connectedText() const;

    CState* _state{nullptr};
    InkPreviewClient* _client{nullptr};
    QTimer* _settle{nullptr};
    VolumeViewerBase* _target{nullptr};

    bool _enabled{false};
    bool _reversed{false};
    int _levelChoice{kAutoLevel};
    int _opacity{70};
    int _opacityBeforeToggle{70};
    std::string _colormapId;
    QString _status;
    bool _statusIsError{false};

    std::optional<InkPreviewClient::Health> _health;
    bool _healthPending{false};
    std::string _levelsVolumeId;   // the volume `_levels` describes
    std::string _levelsPendingId;  // the volume whose levels are being read
    std::vector<LevelInfo> _levels;

    // Changes on every change of the segmentation surface, edits included.
    quint64 _edit{0};
    QString _session;  // per-run text in every key: edit counts restart with VC3D
    // What the last overlay rebuild saw: a change restarts the settle timer.
    QString _seenKey;
    QRectF _seenView;
    bool _seenReversed{false};

    std::optional<Shown> _shown;
    std::optional<InFlight> _flight;
    quint64 _nextFlightId{0};
    bool _rerun{false};
    std::shared_ptr<InkPreviewSnapshot> _snapshot;
    int _requestsStarted{0};
    QRect _lastRequested;
};
