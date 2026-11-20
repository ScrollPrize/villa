#include "inklabels/InkLabelController.hpp"

#include "CState.hpp"
#include "SurfaceCatalog.hpp"
#include "ToolOutput.hpp"
#include "ViewerManager.hpp"
#include "volume_viewers/CChunkedVolumeViewer.hpp"
#include "volume_viewers/CVolumeViewerView.hpp"
#include "volume_viewers/VolumeViewerBase.hpp"

#include "vc/core/util/QuadSurface.hpp"

#include <QEvent>
#include <QKeyEvent>
#include <QKeySequence>

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <exception>
#include <utility>

namespace fs = std::filesystem;
using namespace vc3d::inklabels;

namespace {

constexpr qreal kInkLabelZ = 19.0;  // just above the ink detection overlay
constexpr qreal kCursorZ = 96.0;
constexpr int kTilePx = 256;
constexpr int kMaxLevel = 6;

uint64_t tileKey(int level, int tx, int ty)
{
    return (static_cast<uint64_t>(level) << 56) | (static_cast<uint64_t>(static_cast<uint32_t>(ty)) << 28) |
           static_cast<uint64_t>(static_cast<uint32_t>(tx));
}

// Label colours, unpremultiplied ARGB: unlabelled is transparent.
constexpr QRgb kColours[3] = {
    qRgba(0, 0, 0, 0),
    qRgba(40, 150, 255, 255),  // no-ink
    qRgba(255, 90, 20, 255),   // ink
};

// A store surface's labels are a local output of VC3D, kept where its tools keep theirs.
fs::path labelDirectoryFor(const std::string& surfaceUuid)
{
    return fs::path(vc3d::surfaceExportDirectory(QString::fromStdString(surfaceUuid)).toStdString());
}

} // namespace

InkLabelController::InkLabelController(CState* state, QObject* parent)
    : ViewerOverlayControllerBase("ink_label_overlay", parent)
    , _state(state)
{
    if (_state) {
        connect(_state, &CState::surfaceChanged, this,
                [this](const std::string& name, std::shared_ptr<Surface>, bool isEditUpdate) {
                    if (name == "segmentation" && !isEditUpdate) {
                        onSurfaceChanged();
                    }
                });
    }
}

InkLabelController::~InkLabelController()
{
    // The owner may be mid-destruction: save without notifying anyone.
    blockSignals(true);
    autosave();
}

void InkLabelController::setSurfaceCatalog(SurfaceCatalog* catalog)
{
    _catalog = catalog;
    if (_catalog) {
        connect(_catalog, &SurfaceCatalog::rowsChanged, this, [this]() {
            if (_labelDirectory.empty()) {
                onSurfaceChanged();
            }
        });
    }
    onSurfaceChanged();
}

void InkLabelController::bindViewerManager(ViewerManager* manager)
{
    _viewerManager = manager;
    bindToViewerManager(manager);
    if (!manager) {
        return;
    }
    connect(manager, &ViewerManager::baseViewerCreated, this, &InkLabelController::bindInputViewer);
    manager->forEachBaseViewer([this](VolumeViewerBase* viewer) { bindInputViewer(viewer); });
    onSurfaceChanged();
}

void InkLabelController::bindInputViewer(VolumeViewerBase* viewer)
{
    if (!viewer || _inputViewer || viewer->surfName() != "segmentation") {
        return;
    }
    auto* chunked = qobject_cast<CChunkedVolumeViewer*>(viewer->asQObject());
    if (!chunked) {
        return;
    }
    _inputViewer = viewer;
    connect(chunked, &CChunkedVolumeViewer::sendMousePressVolume, this,
            [this](const cv::Vec3f&, const cv::Vec3f&, Qt::MouseButton button, Qt::KeyboardModifiers modifiers,
                   const QPointF& scenePos) { handlePress(button, modifiers, scenePos); });
    connect(chunked, &CChunkedVolumeViewer::sendMouseMoveVolume, this,
            [this](const cv::Vec3f&, Qt::MouseButtons buttons, Qt::KeyboardModifiers, const QPointF& scenePos) {
                handleMove(buttons, scenePos);
            });
    connect(chunked, &CChunkedVolumeViewer::sendMouseReleaseVolume, this,
            [this](const cv::Vec3f&, Qt::MouseButton button, Qt::KeyboardModifiers, const QPointF& scenePos) {
                handleRelease(button, scenePos);
            });
    if (auto* view = viewer->graphicsView()) {
        _inputView = view;
        _inputViewport = view->viewport();
        view->installEventFilter(this);
        view->viewport()->installEventFilter(this);
    }
    connect(chunked, &QObject::destroyed, this, [this]() {
        _inputViewer = nullptr;
        _inputView = nullptr;
        _inputViewport = nullptr;
    });
}

QuadSurface* InkLabelController::activeSurface() const
{
    if (!_state) {
        return nullptr;
    }
    return dynamic_cast<QuadSurface*>(_state->surface("segmentation").get());
}

TrainingLabelFiles InkLabelController::defaultFiles() const
{
    return trainingLabelFiles(_labelDirectory, _surfaceName.empty() ? _surfaceId : _surfaceName);
}

bool InkLabelController::autosave()
{
    if (_surfaceId.empty() || _layer.empty() || !_layer.modified() || _labelDirectory.empty() ||
        _savedLabelsBlocked) {
        return true;
    }
    return save();
}

void InkLabelController::onSurfaceChanged()
{
    QuadSurface* surface = activeSurface();
    const std::string id = surface ? surface->id : std::string();
    const auto* points = surface ? surface->rawPointsPtr() : nullptr;
    const cv::Size gridSize = points ? cv::Size(points->cols, points->rows) : cv::Size();
    if (id == _surfaceId && gridSize == _gridSize && !_layer.empty()) {
        // The catalog may list the surface after it opened: find its folder then.
        if (_labelDirectory.empty() && !_layer.modified() && _layer.undoDepth() == 0) {
            if (const auto* row = _catalog ? _catalog->row(id) : nullptr) {
                _surfaceName = row->record.name;
                _labelDirectory = labelDirectoryFor(id);
                loadDefaultFiles();
            }
        }
        return;
    }
    // Never drop labels: keep them with the surface they were painted on.
    autosave();
    if (_layer.strokeActive()) {
        _layer.cancelStroke();
    }

    _surfaceId = id;
    _gridSize = gridSize;
    _surfaceName.clear();
    _labelDirectory.clear();
    _savedLabelsBlocked = false;
    _tiles.clear();
    _pyramid.clear();
    if (!surface || !points || points->empty()) {
        _layer = InkLabelLayer();
        emit stateChanged();
        refreshAll();
        return;
    }
    if (const auto* row = _catalog ? _catalog->row(id) : nullptr) {
        _surfaceName = row->record.name;
        _labelDirectory = labelDirectoryFor(id);
    }
    _layer = InkLabelLayer(canvasSize(points->rows, points->cols, surface->scale()));
    loadDefaultFiles();
}

void InkLabelController::loadDefaultFiles()
{
    const auto files = defaultFiles();
    if (!_labelDirectory.empty() && fs::exists(files.inklabels) && fs::exists(files.supervisionMask) &&
        load(files)) {
        return;
    }
    // Files that are there but do not load (a grid resized by an edit, or unreadable)
    // are kept as they are: no painting on this grid, no saving over them.
    _savedLabelsBlocked = !_labelDirectory.empty() && (fs::exists(files.inklabels) || fs::exists(files.supervisionMask));
    rebuildPyramid();
    emit stateChanged();
    refreshAll();
}

void InkLabelController::setLabellingEnabled(bool enabled)
{
    if (_enabled == enabled) {
        return;
    }
    if (!enabled && _layer.strokeActive()) {
        _layer.endStroke();
    }
    _enabled = enabled;
    if (!enabled) {
        _hoverScene.reset();
    }
    emit stateChanged();
    refreshAll();
}

void InkLabelController::setTool(Tool tool)
{
    if (_tool == tool) {
        return;
    }
    _tool = tool;
    emit stateChanged();
    refreshInputViewer();
}

void InkLabelController::setBrushRadius(float radius)
{
    radius = std::clamp(radius, 0.5f, 400.0f);
    if (std::abs(radius - _radius) < 1e-4f) {
        return;
    }
    _radius = radius;
    emit stateChanged();
    refreshInputViewer();
}

void InkLabelController::setOpacity(int opacity)
{
    opacity = std::clamp(opacity, 0, 100);
    if (opacity == _opacity) {
        return;
    }
    _opacity = opacity;
    emit stateChanged();
    refreshAll();
}

void InkLabelController::setLabelsVisible(bool visible)
{
    if (visible == _visible) {
        return;
    }
    _visible = visible;
    emit stateChanged();
    refreshAll();
}

void InkLabelController::undo()
{
    const cv::Rect r = _layer.undo();
    if (!r.empty()) {
        changed(r);
    }
}

void InkLabelController::redo()
{
    const cv::Rect r = _layer.redo();
    if (!r.empty()) {
        changed(r);
    }
}

bool InkLabelController::save()
{
    if (_labelDirectory.empty()) {
        emit statusMessageRequested(tr("Ink labels: this surface is not in a store; use Save to Folder."), 5000);
        return false;
    }
    if (_savedLabelsBlocked) {
        emit statusMessageRequested(
            tr("Ink labels: the labels saved for this surface do not match its grid; not saving over them."), 8000);
        return false;
    }
    try {
        fs::create_directories(_labelDirectory);
    } catch (const std::exception& error) {
        emit statusMessageRequested(tr("Ink labels: %1").arg(QString::fromUtf8(error.what())), 5000);
        return false;
    }
    return saveTo(_labelDirectory, _surfaceName.empty() ? _surfaceId : _surfaceName);
}

bool InkLabelController::saveTo(const fs::path& directory, const std::string& prefix)
{
    if (_layer.empty()) {
        return false;
    }
    if (_layer.strokeActive()) {
        _layer.endStroke();
    }
    const auto files = trainingLabelFiles(directory, prefix);
    try {
        writeTrainingLabels(files, _layer.labels());
    } catch (const std::exception& error) {
        emit statusMessageRequested(tr("Saving ink labels failed: %1").arg(QString::fromUtf8(error.what())), 8000);
        return false;
    }
    if (directory == _labelDirectory) {
        _layer.markSaved();
    }
    emit statusMessageRequested(tr("Saved ink labels to %1").arg(QString::fromStdString(files.inklabels.string())), 4000);
    emit stateChanged();
    return true;
}

bool InkLabelController::load(const TrainingLabelFiles& files)
{
    if (_surfaceId.empty()) {
        return false;
    }
    try {
        auto loaded = readTrainingLabels(files);
        if (loaded.labels.size() != _layer.size()) {
            emit statusMessageRequested(
                tr("Ink labels are %1 x %2 px; this surface renders at %3 x %4 px.")
                    .arg(loaded.labels.cols).arg(loaded.labels.rows)
                    .arg(_layer.size().width).arg(_layer.size().height),
                8000);
            return false;
        }
        _layer.reset(loaded.labels);
        if (loaded.inkOutsideSupervision > 0) {
            emit statusMessageRequested(
                tr("%1 ink px lay outside the supervision mask and loaded as unlabelled.")
                    .arg(static_cast<qulonglong>(loaded.inkOutsideSupervision)),
                8000);
        }
    } catch (const std::exception& error) {
        emit statusMessageRequested(tr("Loading ink labels failed: %1").arg(QString::fromUtf8(error.what())), 8000);
        return false;
    }
    _tiles.clear();
    rebuildPyramid();
    emit stateChanged();
    refreshAll();
    return true;
}

bool InkLabelController::isOverlayEnabledFor(VolumeViewerBase* viewer) const
{
    return viewer && viewer == _inputViewer && hasSurface() && (_visible || _enabled);
}

std::optional<InkLabelController::Mapping> InkLabelController::mapping(VolumeViewerBase* viewer) const
{
    auto* surface = viewer ? dynamic_cast<QuadSurface*>(viewer->currentSurface()) : nullptr;
    if (!surface || surface->id != _surfaceId || _layer.empty() || _savedLabelsBlocked) {
        return std::nullopt;
    }
    const auto* points = surface->rawPointsPtr();
    const cv::Vec2f scale = surface->scale();
    if (!points || points->empty() || std::abs(scale[0]) < 1e-6f || std::abs(scale[1]) < 1e-6f) {
        return std::nullopt;
    }
    // Labels belong to the grid they were painted on; a resized grid (growth,
    // crop) would misplace them, so neither draw nor paint until it matches.
    if (canvasSize(points->rows, points->cols, scale) != _layer.size()) {
        return std::nullopt;
    }
    // The same placement as the ink detection overlay: the image spans the grid.
    const cv::Vec3f centre = surface->center();
    auto gridToScene = [&](double row, double col) {
        return viewer->surfaceCoordsToScene(static_cast<float>(col / scale[0] - centre[0]),
                                            static_cast<float>(row / scale[1] - centre[1]));
    };
    const QPointF g00 = gridToScene(0, 0);
    const QPointF g01 = gridToScene(0, 1);
    const QPointF g10 = gridToScene(1, 0);
    const qreal gridX = std::hypot(g01.x() - g00.x(), g01.y() - g00.y());
    const qreal gridY = std::hypot(g10.x() - g00.x(), g10.y() - g00.y());
    if (gridX < 1e-9 || gridY < 1e-9) {
        return std::nullopt;
    }
    Mapping m;
    m.origin = g00;
    m.sx = gridX * points->cols / static_cast<qreal>(_layer.size().width);
    m.sy = gridY * points->rows / static_cast<qreal>(_layer.size().height);
    return m;
}

std::optional<cv::Point2f> InkLabelController::sceneToLabel(const QPointF& scenePos) const
{
    const auto m = mapping(_inputViewer);
    if (!m) {
        return std::nullopt;
    }
    return cv::Point2f(static_cast<float>((scenePos.x() - m->origin.x()) / m->sx),
                       static_cast<float>((scenePos.y() - m->origin.y()) / m->sy));
}

void InkLabelController::handlePress(Qt::MouseButton button, Qt::KeyboardModifiers modifiers,
                                     const QPointF& scenePos)
{
    if (!_enabled || button != Qt::LeftButton || !hasSurface() ||
        modifiers.testFlag(Qt::ControlModifier) || modifiers.testFlag(Qt::ShiftModifier)) {
        return;
    }
    const auto px = sceneToLabel(scenePos);
    if (!px) {
        return;
    }
    Label value = _tool == Tool::Ink ? Label::Ink : _tool == Tool::NoInk ? Label::NoInk : Label::Unlabelled;
    if (modifiers.testFlag(Qt::AltModifier) && value != Label::Unlabelled) {
        value = value == Label::Ink ? Label::NoInk : Label::Ink;
    }
    _layer.beginStroke(value, _radius);
    changed(_layer.strokeTo(*px));
}

void InkLabelController::handleMove(Qt::MouseButtons buttons, const QPointF& scenePos)
{
    if (!_enabled) {
        return;
    }
    _hoverScene = scenePos;
    if (_layer.strokeActive() && buttons.testFlag(Qt::LeftButton)) {
        if (const auto px = sceneToLabel(scenePos)) {
            const cv::Rect r = _layer.strokeTo(*px);
            if (!r.empty()) {
                changed(r);
                return;
            }
        }
    }
    refreshInputViewer();
}

void InkLabelController::handleRelease(Qt::MouseButton button, const QPointF& scenePos)
{
    if (button != Qt::LeftButton || !_layer.strokeActive()) {
        return;
    }
    if (const auto px = sceneToLabel(scenePos)) {
        changed(_layer.strokeTo(*px));
    }
    _layer.endStroke();
    emit stateChanged();
}

bool InkLabelController::eventFilter(QObject* watched, QEvent* event)
{
    if (_enabled && (watched == _inputView || watched == _inputViewport) &&
        (event->type() == QEvent::ShortcutOverride || event->type() == QEvent::KeyPress)) {
        auto* key = static_cast<QKeyEvent*>(event);
        const bool apply = event->type() == QEvent::KeyPress;
        if (key->matches(QKeySequence::Undo) || key->matches(QKeySequence::Redo)) {
            if (apply) {
                key->matches(QKeySequence::Undo) ? undo() : redo();
            }
            event->accept();
            return true;
        }
        if (handleKey(key->key(), key->modifiers(), apply && !key->isAutoRepeat())) {
            event->accept();
            return true;
        }
    }
    if (_enabled && watched == _inputViewport && event->type() == QEvent::Leave) {
        _hoverScene.reset();
        refreshInputViewer();
    }
    return ViewerOverlayControllerBase::eventFilter(watched, event);
}

bool InkLabelController::handleKey(int key, Qt::KeyboardModifiers modifiers, bool apply)
{
    if (modifiers.testAnyFlags(Qt::ControlModifier | Qt::AltModifier | Qt::MetaModifier)) {
        return false;
    }
    switch (key) {
    case Qt::Key_B: if (apply) setTool(Tool::Ink); return true;
    case Qt::Key_N: if (apply) setTool(Tool::NoInk); return true;
    case Qt::Key_U: if (apply) setTool(Tool::Erase); return true;
    case Qt::Key_L: if (apply) setLabelsVisible(!_visible); return true;
    case Qt::Key_BracketLeft: if (apply) setBrushRadius(std::max(0.5f, _radius / 1.25f)); return true;
    case Qt::Key_BracketRight: if (apply) setBrushRadius(_radius * 1.25f); return true;
    case Qt::Key_Escape:
        if (!_layer.strokeActive()) {
            return false;
        }
        if (apply) changed(_layer.cancelStroke());
        return true;
    default:
        return false;
    }
}

void InkLabelController::changed(const cv::Rect& labelRect)
{
    if (!labelRect.empty()) {
        updatePyramid(labelRect);
        // Drop cached tiles the change touches, at every level.
        for (int level = 0; level <= static_cast<int>(_pyramid.size()); ++level) {
            const int span = kTilePx << level;
            for (int ty = labelRect.y / span; ty <= (labelRect.y + labelRect.height - 1) / span; ++ty) {
                for (int tx = labelRect.x / span; tx <= (labelRect.x + labelRect.width - 1) / span; ++tx) {
                    _tiles.erase(tileKey(level, tx, ty));
                }
            }
        }
    }
    if (!_layer.strokeActive()) {
        emit stateChanged();
    }
    refreshInputViewer();
}

void InkLabelController::rebuildPyramid()
{
    _pyramid.clear();
    if (_layer.empty()) {
        return;
    }
    updatePyramid(cv::Rect(0, 0, _layer.size().width, _layer.size().height));
}

void InkLabelController::updatePyramid(const cv::Rect& labelRect)
{
    // Max pooling keeps a one-pixel stroke visible when zoomed out; ink wins over no-ink.
    const cv::Mat_<uint8_t>* source = &_layer.labels();
    cv::Rect rect = labelRect;
    for (int level = 1; level <= kMaxLevel; ++level) {
        const cv::Size size((source->cols + 1) / 2, (source->rows + 1) / 2);
        if (static_cast<int>(_pyramid.size()) < level) {
            if (source->cols <= kTilePx && source->rows <= kTilePx) {
                break;
            }
            _pyramid.emplace_back(size, uint8_t(0));
            rect = cv::Rect(0, 0, source->cols, source->rows);
        }
        cv::Mat_<uint8_t>& target = _pyramid[level - 1];
        const int x0 = rect.x / 2, y0 = rect.y / 2;
        const int x1 = std::min(size.width, (rect.x + rect.width + 1) / 2);
        const int y1 = std::min(size.height, (rect.y + rect.height + 1) / 2);
        for (int y = y0; y < y1; ++y) {
            const uint8_t* a = source->ptr<uint8_t>(2 * y);
            const uint8_t* b = 2 * y + 1 < source->rows ? source->ptr<uint8_t>(2 * y + 1) : a;
            uint8_t* out = target.ptr<uint8_t>(y);
            for (int x = x0; x < x1; ++x) {
                const int c0 = 2 * x;
                const int c1 = std::min(source->cols - 1, c0 + 1);
                out[x] = std::max({a[c0], a[c1], b[c0], b[c1]});
            }
        }
        rect = cv::Rect(x0, y0, x1 - x0, y1 - y0);
        source = &target;
        if (rect.empty()) {
            break;
        }
    }
}

const InkLabelController::Tile& InkLabelController::tile(int level, int tx, int ty)
{
    const uint64_t key = tileKey(level, tx, ty);
    if (auto it = _tiles.find(key); it != _tiles.end()) {
        return it->second;
    }
    if (_tiles.size() > 256) {
        _tiles.clear();
    }
    const cv::Mat_<uint8_t>& source = level == 0 ? _layer.labels() : _pyramid[level - 1];
    const cv::Rect r = cv::Rect(tx * kTilePx, ty * kTilePx, kTilePx, kTilePx) & cv::Rect(0, 0, source.cols, source.rows);
    Tile result;
    result.image = QImage(std::max(1, r.width), std::max(1, r.height), QImage::Format_ARGB32);
    for (int y = 0; y < r.height; ++y) {
        const uint8_t* in = source.ptr<uint8_t>(r.y + y) + r.x;
        auto* out = reinterpret_cast<QRgb*>(result.image.scanLine(y));
        for (int x = 0; x < r.width; ++x) {
            out[x] = kColours[std::min<uint8_t>(in[x], 2)];
        }
    }
    result.pixmap = QPixmap::fromImage(result.image);
    return _tiles.emplace(key, std::move(result)).first->second;
}

void InkLabelController::collectPrimitives(VolumeViewerBase* viewer, OverlayBuilder& builder)
{
    const auto m = mapping(viewer);
    auto* view = viewer ? viewer->graphicsView() : nullptr;
    if (!m || !view) {
        return;
    }
    if (_visible) {
        const QRectF visible = view->mapToScene(view->viewport()->rect()).boundingRect();
        const qreal screenPerPx = m->sx * std::max(1e-9, std::abs(view->transform().m11()));
        int level = 0;
        while (level < static_cast<int>(_pyramid.size()) && screenPerPx * (1 << (level + 1)) <= 1.0) {
            ++level;
        }
        const qreal span = static_cast<qreal>(1 << level);
        const cv::Mat_<uint8_t>& source = level == 0 ? _layer.labels() : _pyramid[level - 1];
        const int x0 = std::max(0, static_cast<int>(std::floor((visible.left() - m->origin.x()) / (m->sx * span))));
        const int y0 = std::max(0, static_cast<int>(std::floor((visible.top() - m->origin.y()) / (m->sy * span))));
        const int x1 = std::min(source.cols, static_cast<int>(std::ceil((visible.right() - m->origin.x()) / (m->sx * span))));
        const int y1 = std::min(source.rows, static_cast<int>(std::ceil((visible.bottom() - m->origin.y()) / (m->sy * span))));
        const qreal opacity = static_cast<qreal>(_opacity) / 100.0;
        for (int ty = y0 / kTilePx; x1 > x0 && y1 > y0 && ty <= (y1 - 1) / kTilePx; ++ty) {
            for (int tx = x0 / kTilePx; tx <= (x1 - 1) / kTilePx; ++tx) {
                const Tile& t = tile(level, tx, ty);
                const QPointF offset(m->origin.x() + tx * kTilePx * span * m->sx,
                                     m->origin.y() + ty * kTilePx * span * m->sy);
                builder.addImage(t.image, t.pixmap, offset, m->sx * span, m->sy * span, opacity, kInkLabelZ);
            }
        }
    }
    if (_enabled && _hoverScene && viewer == _inputViewer) {
        OverlayStyle style;
        style.penColor = _tool == Tool::Ink ? QColor(255, 90, 20) : _tool == Tool::NoInk ? QColor(40, 150, 255)
                                                                                         : QColor(240, 240, 240);
        style.penWidth = 1.5;
        style.z = kCursorZ;
        builder.addCircle(*_hoverScene, _radius * m->sx, false, style);
    }
}

void InkLabelController::refreshInputViewer()
{
    if (_inputViewer) {
        refreshViewer(_inputViewer);
    }
}

#ifdef VC_TESTING
#include "InkLabelControllerTest.inc"
#endif
