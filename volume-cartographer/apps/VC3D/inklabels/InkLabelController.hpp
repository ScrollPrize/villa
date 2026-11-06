#pragma once

#include "overlays/ViewerOverlayControllerBase.hpp"
#include "inklabels/InkLabelIO.hpp"
#include "inklabels/InkLabelLayer.hpp"

#include <QImage>
#include <QPixmap>
#include <QPointF>
#include <QString>

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

class CState;
class SurfaceCatalog;
class ViewerManager;
class VolumeViewerBase;
class QuadSurface;

// Paints ink / no-ink / unlabelled labels over the flattened segmentation
// viewer, in the segment's render pixels, and keeps a store surface's labels
// in VC3D's export folder for it (vc3d::surfaceExportDirectory) in the
// trainer's flat label format.
//
// Left drag paints the current tool; Alt+drag paints the other class (ink and
// no-ink swap). While labelling is on, the segmentation viewer also takes
// B / N / U (tool), [ / ] (radius), L (show labels), Esc (cancel stroke) and
// the standard Undo / Redo keys.
class InkLabelController : public ViewerOverlayControllerBase
{
    Q_OBJECT

public:
    enum class Tool { Ink, NoInk, Erase };

    explicit InkLabelController(CState* state, QObject* parent = nullptr);
    ~InkLabelController() override;

    // The catalog tells which surfaces are in a store, and their names.
    void setSurfaceCatalog(SurfaceCatalog* catalog);
    // Draw on the manager's viewers and take input from its segmentation viewer.
    void bindViewerManager(ViewerManager* manager);

    [[nodiscard]] bool labellingEnabled() const { return _enabled; }
    [[nodiscard]] Tool tool() const { return _tool; }
    [[nodiscard]] float brushRadius() const { return _radius; }
    [[nodiscard]] int opacity() const { return _opacity; }
    [[nodiscard]] bool labelsVisible() const { return _visible; }
    [[nodiscard]] bool hasSurface() const { return !_surfaceId.empty() && !_layer.empty(); }
    [[nodiscard]] const vc3d::inklabels::InkLabelLayer& layer() const { return _layer; }
    [[nodiscard]] const std::string& surfaceName() const { return _surfaceName; }
    // Where Save writes (empty without a store-backed surface).
    [[nodiscard]] const std::filesystem::path& labelDirectory() const { return _labelDirectory; }
    // The label files saved for the surface did not load (another grid size after an
    // edit, or unreadable): VC3D neither paints nor saves over them.
    [[nodiscard]] bool savedLabelsBlocked() const { return _savedLabelsBlocked; }
    [[nodiscard]] vc3d::inklabels::TrainingLabelFiles defaultFiles() const;

public slots:
    void setLabellingEnabled(bool enabled);
    void setTool(InkLabelController::Tool tool);
    void setBrushRadius(float radius);
    void setOpacity(int opacity);
    void setLabelsVisible(bool visible);
    void undo();
    void redo();
    // Save to the surface's export folder.
    bool save();
    bool saveTo(const std::filesystem::path& directory, const std::string& prefix);
    bool load(const vc3d::inklabels::TrainingLabelFiles& files);

signals:
    void stateChanged();
    void statusMessageRequested(const QString& text, int timeoutMs);

protected:
    bool isOverlayEnabledFor(VolumeViewerBase* viewer) const override;
    void collectPrimitives(VolumeViewerBase* viewer, OverlayBuilder& builder) override;
    bool eventFilter(QObject* watched, QEvent* event) override;

private:
    // Scene position of label pixel (0, 0)'s corner and scene units per label pixel.
    struct Mapping {
        QPointF origin;
        qreal sx{1.0};
        qreal sy{1.0};
    };
    struct Tile {
        QImage image;
        QPixmap pixmap;
    };

    void bindInputViewer(VolumeViewerBase* viewer);
    void onSurfaceChanged();
    void loadDefaultFiles();
    void handlePress(Qt::MouseButton button, Qt::KeyboardModifiers modifiers, const QPointF& scenePos);
    void handleMove(Qt::MouseButtons buttons, const QPointF& scenePos);
    void handleRelease(Qt::MouseButton button, const QPointF& scenePos);
    bool handleKey(int key, Qt::KeyboardModifiers modifiers, bool apply);
    [[nodiscard]] QuadSurface* activeSurface() const;
    [[nodiscard]] std::optional<Mapping> mapping(VolumeViewerBase* viewer) const;
    [[nodiscard]] std::optional<cv::Point2f> sceneToLabel(const QPointF& scenePos) const;
    void changed(const cv::Rect& labelRect);
    void rebuildPyramid();
    void updatePyramid(const cv::Rect& labelRect);
    const Tile& tile(int level, int tx, int ty);
    void refreshInputViewer();
    bool autosave();

    CState* _state{nullptr};
    SurfaceCatalog* _catalog{nullptr};
    ViewerManager* _viewerManager{nullptr};
    VolumeViewerBase* _inputViewer{nullptr};
    QObject* _inputView{nullptr};
    QObject* _inputViewport{nullptr};

    bool _enabled{false};
    Tool _tool{Tool::Ink};
    float _radius{6.0f};
    int _opacity{55};
    bool _visible{true};
    std::optional<QPointF> _hoverScene;

    std::string _surfaceId;
    std::string _surfaceName;
    std::filesystem::path _labelDirectory;
    bool _savedLabelsBlocked{false};
    cv::Size _gridSize;
    vc3d::inklabels::InkLabelLayer _layer;
    // Max-pooled copies for zoomed-out display (index k holds level k + 1).
    std::vector<cv::Mat_<uint8_t>> _pyramid;
    std::unordered_map<uint64_t, Tile> _tiles;
};
