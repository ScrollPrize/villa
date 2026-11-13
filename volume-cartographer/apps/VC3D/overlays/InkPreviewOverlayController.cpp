#include "InkPreviewOverlayController.hpp"

#include "../CState.hpp"
#include "../OpenedVolumes.hpp"
#include "../volume_viewers/VolumeViewerBase.hpp"
#include "ScalarOverlayImage.hpp"

#include "vc/core/types/Volume.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/RenderLayers.hpp"

#include <vc_reader/affine.hpp>
#include <vc_reader/array.hpp>
#include <vc_reader/zarr.hpp>

#include <QCryptographicHash>
#include <QFutureWatcher>
#include <QRandomGenerator>
#include <QTimer>
#include <QtConcurrent/QtConcurrentRun>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace {

// The published ink detections draw at 18 and the ink labels above them; the preview
// takes the published detections' place.
constexpr qreal kInkPreviewZ = 18.0;
// The client's key is at most this long (ink preview api 1).
constexpr int kMaxKeyLength = 200;

// Indices of `level` per working level-0 voxel, as vc_render_tifxyz maps them for
// --group-idx `level`: the identity at level 0, otherwise from the levels' placements.
vc::RenderLevelMap levelMap(const OpenedVolumes::Level& opened, int level)
{
    return level == 0 ? vc::RenderLevelMap{} : vc::renderLevelMap(opened.working, opened.array->placement);
}

bool sampleableType(const vcr::ElementType& type)
{
    return type.kind == 'u' && (type.itemsize == 1 || type.itemsize == 2);
}

struct LevelsRead {
    std::vector<InkPreviewOverlayController::LevelInfo> levels;
    QString error;
};

struct Sampled {
    QByteArray voxels;
    bool wide = false;
    QString error;
};

} // namespace

// A copy of the segmentation surface's grid taken on the GUI thread, which workers
// sample while the user goes on editing the original. The copy has the stored
// surface's origin (an origin moved in memory does not move the canvas); its layer
// surface is built once per level of one volume, on a worker. A level's layer surface
// holds that volume's placement of the level, so another volume takes another
// snapshot (which shares the copy of the grid).
struct InkPreviewSnapshot {
    std::string surfaceId;
    quint64 edit = 0;
    std::string volumeId;
    std::shared_ptr<const QuadSurface> surface;
    std::mutex mutex;
    std::map<int, std::shared_ptr<const vc::RenderSurface>> layers;
};

namespace {

// The layers of canvas rectangle `pixels` at `offsets`, as a request body. Runs on a
// worker: it reads the volume blocking, at one level, with linear interpolation, and
// stops when `cancel` is set.
Sampled sampleLayers(const std::shared_ptr<OpenedVolumes>& opened,
                     const std::shared_ptr<Volume>& volume,
                     int level,
                     const std::shared_ptr<InkPreviewSnapshot>& snapshot,
                     const vc::RenderCanvas& canvas,
                     const cv::Rect& pixels,
                     const std::vector<float>& offsets,
                     const std::shared_ptr<std::atomic<bool>>& cancel)
{
    Sampled out;
    try {
        const OpenedVolumes::Level opened_level = opened->level(*volume, level);
        const vcr::zarr::OpenedArray& array = *opened_level.array;
        if (array.meta.shape.size() != 3 || !sampleableType(array.meta.type)) {
            throw std::runtime_error("the live preview samples 3-D uint8 or uint16 volumes");
        }
        std::shared_ptr<const vc::RenderSurface> layers;
        {
            std::lock_guard lock(snapshot->mutex);
            auto& slot = snapshot->layers[level];
            if (!slot) {
                slot = std::make_shared<const vc::RenderSurface>(vc::renderSurface(
                    *snapshot->surface, vcr::grid::Interpolation::Bicubic, 1.0f, false, vcr::catalog::Affine{},
                    levelMap(opened_level, level)));
            }
            layers = slot;
        }
        // The level alone: no coarser level, no GPU reader, and its placement is in `layers`.
        vcr::grid::Level plain;
        plain.array = array.array.get();
        plain.type = array.meta.type;
        const cv::Mat stack = vc::sampleRenderLayers(*snapshot->surface, *layers, plain, canvas, pixels, offsets,
                                                     false, cancel.get());
        out.wide = stack.elemSize() == 2;
        out.voxels = QByteArray(reinterpret_cast<const char*>(stack.data),
                                static_cast<qsizetype>(stack.total() * stack.elemSize()));
        if constexpr (std::endian::native == std::endian::big) {
            if (out.wide) {
                auto* bytes = out.voxels.data();
                for (qsizetype i = 0; i + 1 < out.voxels.size(); i += 2) {
                    std::swap(bytes[i], bytes[i + 1]);
                }
            }
        }
    } catch (const std::exception& error) {
        out.error = QString::fromUtf8(error.what());
    }
    return out;
}

LevelsRead readLevels(const std::shared_ptr<OpenedVolumes>& opened, const std::shared_ptr<Volume>& volume)
{
    LevelsRead out;
    try {
        const auto& pyramid = opened->pyramid(*volume);
        if (pyramid.dtype != "|u1" && pyramid.dtype != "<u2") {
            throw std::runtime_error("the live preview samples uint8 or uint16 volumes, not " + pyramid.dtype);
        }
        const double voxel = volume->voxelSize();
        for (const int level : pyramid.levels) {
            const auto map = levelMap(opened->level(*volume, level), level);
            const float scale = map.voxelScale();
            if (!(std::isfinite(scale) && scale > 0.0f)) {
                continue;
            }
            out.levels.push_back({level, scale, voxel > 0.0 ? voxel / double(scale) : 0.0});
        }
        if (out.levels.empty()) {
            throw std::runtime_error("the volume has no level to sample");
        }
    } catch (const std::exception& error) {
        out.levels.clear();
        out.error = QString::fromUtf8(error.what());
    }
    return out;
}

template <class T, class Work, class Done>
void runInBackground(QObject* owner, Work work, Done done)
{
    auto* watcher = new QFutureWatcher<T>(owner);
    QObject::connect(watcher, &QFutureWatcher<T>::finished, owner, [watcher, done]() {
        T result = watcher->result();
        watcher->deleteLater();
        done(std::move(result));
    });
    watcher->setFuture(QtConcurrent::run(std::move(work)));
}

} // namespace

InkPreviewOverlayController::InkPreviewOverlayController(CState* state, QObject* parent)
    : ViewerOverlayControllerBase("ink_preview_overlay", parent)
    , _state(state)
    , _client(new InkPreviewClient(this))
    , _settle(new QTimer(this))
    , _session(QString::number(QRandomGenerator::global()->generate64(), 36))
{
    _settle->setSingleShot(true);
    _settle->setInterval(kSettleMs);
    connect(_settle, &QTimer::timeout, this, &InkPreviewOverlayController::startRequest);
    if (_state) {
        // Every change of the segmentation surface, an edit or another surface, changes
        // the key: the server never serves layers of an older geometry.
        connect(_state, &CState::surfaceChanged, this, [this](const std::string& name, std::shared_ptr<Surface>, bool) {
            if (name.empty() || name == "segmentation") {
                ++_edit;
                if (_enabled && _target) {
                    refreshViewer(_target);
                }
            }
        });
    }
}

InkPreviewOverlayController::~InkPreviewOverlayController()
{
    cancelFlight();
}

QUrl InkPreviewOverlayController::serverUrl() const
{
    return _client->serverUrl();
}

std::optional<int> InkPreviewOverlayController::resolvedLevel() const
{
    if (_levels.empty()) {
        return std::nullopt;
    }
    if (_levelChoice != kAutoLevel) {
        const bool present = std::any_of(_levels.begin(), _levels.end(),
                                         [&](const LevelInfo& info) { return info.level == _levelChoice; });
        return present ? std::optional<int>(_levelChoice) : std::nullopt;
    }
    const LevelInfo* best = nullptr;
    if (_health && _health->voxelUm) {
        for (const auto& info : _levels) {
            if (info.voxelUm > 0.0 &&
                (!best || std::abs(info.voxelUm - *_health->voxelUm) < std::abs(best->voxelUm - *_health->voxelUm))) {
                best = &info;
            }
        }
    }
    if (!best) {
        const auto zero = std::find_if(_levels.begin(), _levels.end(), [](const LevelInfo& info) { return info.level == 0; });
        best = zero != _levels.end() ? &*zero : &_levels.front();
    }
    return best->level;
}

std::vector<float> InkPreviewOverlayController::layerOffsets() const
{
    // Layer i of the model's D sits at i - D/2: the layers a centered render with an odd
    // layer count gives villa's flat inference after its center crop.
    const int depth = _health ? _health->depth : 0;
    if (depth < 1) {
        return {};
    }
    auto offsets = vc::renderLayerOffsets(depth % 2 ? depth : depth + 1, 1.0);
    offsets.resize(std::size_t(depth));
    return offsets;
}

qint64 InkPreviewOverlayController::viewBudget() const
{
    return _health ? std::min(kViewBudgetPx, _health->maxViewPx) : kViewBudgetPx;
}

bool InkPreviewOverlayController::isSegmentationView(VolumeViewerBase* viewer) const
{
    return viewer && viewer->surfName() == "segmentation" && dynamic_cast<QuadSurface*>(viewer->currentSurface());
}

bool InkPreviewOverlayController::isOverlayEnabledFor(VolumeViewerBase* viewer) const
{
    // Enabled before the first reply: collectPrimitives is what notices the view.
    return _enabled && _opacity > 0 && isSegmentationView(viewer);
}

std::optional<InkPreviewOverlayController::Frame> InkPreviewOverlayController::frameFor(
    VolumeViewerBase* viewer, const QuadSurface& surface) const
{
    const auto volume = viewer ? viewer->currentVolume() : nullptr;
    const auto level = resolvedLevel();
    if (!_health || !volume || volume->id() != _levelsVolumeId || !level) {
        return std::nullopt;
    }
    const auto info = std::find_if(_levels.begin(), _levels.end(), [&](const LevelInfo& l) { return l.level == *level; });
    const cv::Vec2f scale = surface.scale();
    const auto offsets = layerOffsets();
    if (info == _levels.end() || offsets.empty() ||
        !(std::isfinite(scale[0]) && std::isfinite(scale[1]) && scale[0] > 0.0f && scale[1] > 0.0f)) {
        return std::nullopt;
    }

    Frame frame;
    frame.surfaceId = surface.id;
    frame.volumeId = volume->id();
    frame.level = *level;
    // The canvas of vc_render_tifxyz --scale 1 --group-idx level: its render scale is the
    // level's voxel scale, as the tool computes it.
    frame.canvas = vc::RenderCanvas::make(surface.gridSize(), scale, double(info->voxelScale));
    QString key = QStringLiteral("vc3d:%1:s=%2:e=%3:v=%4:l=%5:o=%6..%7")
                      .arg(_session, QString::fromStdString(frame.surfaceId))
                      .arg(_edit)
                      .arg(QString::fromStdString(frame.volumeId))
                      .arg(frame.level)
                      .arg(double(offsets.front()))
                      .arg(double(offsets.back()));
    if (key.size() > kMaxKeyLength) {
        key = QStringLiteral("vc3d:sha1:") +
              QString::fromLatin1(QCryptographicHash::hash(key.toUtf8(), QCryptographicHash::Sha1).toHex());
    }
    frame.key = key;
    return frame;
}

std::optional<QRect> InkPreviewOverlayController::visibleCanvasRect(
    VolumeViewerBase* viewer, const QuadSurface& surface, const vc::RenderCanvas& canvas) const
{
    const QRectF scene = visibleSceneRect(viewer);
    if (scene.isEmpty()) {
        return std::nullopt;
    }
    double x0 = INFINITY, y0 = INFINITY, x1 = -INFINITY, y1 = -INFINITY;
    for (const QPointF& corner : {scene.topLeft(), scene.topRight(), scene.bottomLeft(), scene.bottomRight()}) {
        const cv::Vec2f s = viewer->sceneToSurfaceCoords(corner);
        const cv::Vec2d c = canvas.gridToCanvas(surface.surfaceToGrid({double(s[0]), double(s[1])}));
        if (!std::isfinite(c[0]) || !std::isfinite(c[1])) {
            return std::nullopt;
        }
        x0 = std::min(x0, c[0]);
        y0 = std::min(y0, c[1]);
        x1 = std::max(x1, c[0]);
        y1 = std::max(y1, c[1]);
    }
    // Every canvas pixel the view touches, inside the canvas. The camera's float
    // arithmetic moves an edge that lies on a pixel boundary by far less than kEdge;
    // that does not add a pixel.
    constexpr double kEdge = 1e-3;
    const auto clampTo = [](double v, int size) { return int(std::clamp(v, 0.0, double(size))); };
    const int left = clampTo(std::floor(x0 + kEdge), canvas.size.width);
    const int top = clampTo(std::floor(y0 + kEdge), canvas.size.height);
    const int right = clampTo(std::ceil(x1 - kEdge), canvas.size.width);
    const int bottom = clampTo(std::ceil(y1 - kEdge), canvas.size.height);
    if (right <= left || bottom <= top) {
        return std::nullopt;
    }
    return QRect(left, top, right - left, bottom - top);
}

QPointF InkPreviewOverlayController::canvasToScene(VolumeViewerBase* viewer, const QuadSurface& surface,
                                                   const vc::RenderCanvas& canvas, double x, double y) const
{
    const cv::Vec2d s = surface.gridToSurface(canvas.canvasToGrid({x, y}));
    return viewer->surfaceCoordsToScene(float(s[0]), float(s[1]));
}

void InkPreviewOverlayController::collectPrimitives(VolumeViewerBase* viewer, OverlayBuilder& builder)
{
    if (!isOverlayEnabledFor(viewer)) {
        return;
    }
    auto* surface = dynamic_cast<QuadSurface*>(viewer->currentSurface());
    _target = viewer;
    const auto frame = frameFor(viewer, *surface);

    // A change of what is on screen restarts the settle timer; the request starts when it fires.
    const QRectF scene = visibleSceneRect(viewer);
    const cv::Vec2f a = viewer->sceneToSurfaceCoords(scene.topLeft());
    const cv::Vec2f b = viewer->sceneToSurfaceCoords(scene.bottomRight());
    const QRectF view(QPointF(a[0], a[1]), QPointF(b[0], b[1]));
    const QString key = frame ? frame->key : QString::fromStdString(surface->id) + QString::number(_edit);
    if (key != _seenKey || view != _seenView || _reversed != _seenReversed) {
        _seenKey = key;
        _seenView = view;
        _seenReversed = _reversed;
        _settle->start();
    }

    // The last reply stays where its canvas rectangle is, while the next one is computed;
    // after an edit too, until the reply for the edited surface replaces it.
    if (!frame || !_shown || _shown->reversed != _reversed || _shown->frame.surfaceId != frame->surfaceId ||
        _shown->frame.volumeId != frame->volumeId || _shown->frame.level != frame->level ||
        !(_shown->frame.canvas == frame->canvas)) {
        return;
    }
    const QRect& r = _shown->rect;
    const QPointF origin = canvasToScene(viewer, *surface, frame->canvas, r.x(), r.y());
    const QPointF right = canvasToScene(viewer, *surface, frame->canvas, r.x() + r.width(), r.y());
    const QPointF down = canvasToScene(viewer, *surface, frame->canvas, r.x(), r.y() + r.height());
    const qreal scaleX = (right.x() - origin.x()) / r.width();
    const qreal scaleY = (down.y() - origin.y()) / r.height();
    if (!std::isfinite(scaleX) || !std::isfinite(scaleY) || scaleX == 0.0 || scaleY == 0.0) {
        return;
    }
    builder.addImage(_shown->image, _shown->pixmap, origin, scaleX, scaleY, _opacity / 100.0, kInkPreviewZ);
}

void InkPreviewOverlayController::detachViewer(VolumeViewerBase* viewer)
{
    if (viewer && viewer == _target) {
        _target = nullptr;
        cancelFlight();
    }
    ViewerOverlayControllerBase::detachViewer(viewer);
}

void InkPreviewOverlayController::setEnabled(bool enabled)
{
    if (_enabled == enabled) {
        return;
    }
    _enabled = enabled;
    if (enabled) {
        ensureHealth();
    } else {
        _settle->stop();
        cancelFlight();
        _client->cancel();
        _healthPending = false;
        _health.reset();
        _depthRecheckKey.clear();
        _shown.reset();
        _seenKey.clear();
        setStatus({});
    }
    emit enabledChanged(enabled);
    refreshAll();
}

void InkPreviewOverlayController::setServerUrl(const QUrl& url)
{
    if (url == _client->serverUrl()) {
        return;
    }
    cancelFlight();
    _client->setServerUrl(url);
    _healthPending = false;
    _health.reset();
    _depthRecheckKey.clear();
    // Another server may run another model.
    dropShown();
    if (_enabled) {
        ensureHealth();
    }
}

void InkPreviewOverlayController::setReversed(bool reversed)
{
    if (_reversed == reversed) {
        return;
    }
    _reversed = reversed;
    cancelFlight();
    dropShown();
    if (_enabled) {
        _settle->start();
    }
}

void InkPreviewOverlayController::setLevelChoice(int level)
{
    level = level < 0 ? kAutoLevel : level;
    if (_levelChoice == level) {
        return;
    }
    _levelChoice = level;
    cancelFlight();
    dropShown();
    emit levelsChanged();
    if (_enabled) {
        _settle->start();
    }
}

void InkPreviewOverlayController::setOpacity(int opacity)
{
    const int clamped = std::clamp(opacity, 0, 100);
    if (_opacity == clamped) {
        return;
    }
    _opacity = clamped;
    if (_opacity > 0) {
        _opacityBeforeToggle = _opacity;
    }
    emit opacityChanged(_opacity);
    refreshAll();
}

void InkPreviewOverlayController::setColormapId(const std::string& id)
{
    if (_colormapId == id) {
        return;
    }
    _colormapId = id;
    if (_shown) {
        _shown->image = vc3d::scalarOverlayImage(_shown->scalar, _colormapId);
        _shown->pixmap = QPixmap::fromImage(_shown->image);
    }
    refreshAll();
}

void InkPreviewOverlayController::toggleVisibility()
{
    if (!_enabled) {
        return;
    }
    if (_opacity > 0) {
        _opacityBeforeToggle = _opacity;
        setOpacity(0);
    } else {
        setOpacity(_opacityBeforeToggle > 0 ? _opacityBeforeToggle : 70);
    }
}

void InkPreviewOverlayController::setStatus(const QString& text, bool error)
{
    if (_status == text && _statusIsError == error) {
        return;
    }
    _status = text;
    _statusIsError = error;
    emit statusChanged(_status, _statusIsError);
}

QString InkPreviewOverlayController::connectedText() const
{
    if (!_health) {
        return {};
    }
    const QString voxel = _health->voxelUm ? tr("trained at %1 µm").arg(*_health->voxelUm)
                                           : tr("training voxel size unknown");
    return tr("Connected: %1 (%2), %3 layers, %4.")
        .arg(_health->model, _health->backend)
        .arg(_health->depth)
        .arg(voxel);
}

void InkPreviewOverlayController::ensureHealth()
{
    if (_healthPending || _health) {
        return;
    }
    _healthPending = true;
    setStatus(tr("Connecting to %1...").arg(_client->serverUrl().toString()));
    _client->checkHealth([this](const std::optional<InkPreviewClient::Health>& health, const QString& error) {
        _healthPending = false;
        if (!health) {
            setStatus(error, true);
            return;
        }
        _health = *health;
        setStatus(connectedText());
        // Auto may pick another level now that the model's voxel size is known.
        emit levelsChanged();
        if (_enabled) {
            _settle->start();
        }
    });
}

void InkPreviewOverlayController::ensureLevels(const std::shared_ptr<Volume>& volume)
{
    const std::string id = volume->id();
    if (_levelsPendingId == id) {
        return;
    }
    const auto opened = _state ? _state->openedVolumes() : nullptr;
    if (!opened) {
        setStatus(tr("No volume levels to sample."), true);
        return;
    }
    _levelsPendingId = id;
    setStatus(tr("Reading the volume's levels..."));
    runInBackground<LevelsRead>(this, [opened, volume]() { return readLevels(opened, volume); },
        [this, id](LevelsRead read) {
            if (_levelsPendingId != id) {
                return;
            }
            _levelsPendingId.clear();
            if (!read.error.isEmpty()) {
                setStatus(tr("Cannot read the volume's levels: %1").arg(read.error), true);
                return;
            }
            _levels = std::move(read.levels);
            _levelsVolumeId = id;
            emit levelsChanged();
            setStatus(connectedText());
            if (_enabled) {
                _settle->start();
            }
        });
}

void InkPreviewOverlayController::startRequest()
{
    if (!_enabled || _opacity <= 0) {
        return;
    }
    if (!_health) {
        ensureHealth();
        return;
    }
    VolumeViewerBase* viewer = _target;
    if (!isSegmentationView(viewer)) {
        return;
    }
    auto* surface = dynamic_cast<QuadSurface*>(viewer->currentSurface());
    if (viewer->isCompositeEnabled() && viewer->compositeRenderSettings().params.volumetric) {
        cancelFlight();
        setStatus(tr("The live preview is not drawn in the volumetric view."));
        return;
    }
    const auto volume = viewer->currentVolume();
    if (!volume) {
        setStatus(tr("No volume is open."), true);
        return;
    }
    if (volume->id() != _levelsVolumeId) {
        ensureLevels(volume);
        return;
    }
    // Missing chunks must not read as fill: a preview of fill is not the model's output on the volume.
    if (volume->isRemote() && volume->sourceOpenOptions().offlineFill) {
        cancelFlight();
        setStatus(tr("This remote volume reads chunks missing offline as fill (remote offline \"fill\" in the "
                     "settings); the live preview needs every chunk."), true);
        return;
    }
    const auto frame = frameFor(viewer, *surface);
    if (!frame) {
        setStatus(_levelChoice != kAutoLevel && !resolvedLevel()
                      ? tr("Level %1 is not in this volume.").arg(_levelChoice)
                      : tr("The surface has no grid to sample."),
                  true);
        return;
    }
    const auto visible = visibleCanvasRect(viewer, *surface, frame->canvas);
    if (!visible) {
        return;
    }
    const qint64 budget = viewBudget();
    if (qint64(visible->width()) * visible->height() > budget) {
        cancelFlight();
        setStatus(tr("Zoom in for the live preview: the view holds %1 Mpx of level %2, over the budget of %3 Mpx.")
                      .arg(double(visible->width()) * visible->height() / 1.0e6, 0, 'f', 1)
                      .arg(frame->level)
                      .arg(double(budget) / 1.0e6, 0, 'f', 1));
        return;
    }
    if (_flight) {
        // A request that still covers part of the view finishes first: cancelling on every
        // pan would starve a moving view of results.
        if (_flight->frame.key == frame->key && _flight->reversed == _reversed && _flight->rect.intersects(*visible)) {
            _rerun = true;
            return;
        }
        cancelFlight();
    }
    if (_shown && _shown->frame.key == frame->key && _shown->reversed == _reversed && _shown->rect.contains(*visible)) {
        return;
    }

    // A margin around the view, so a small pan stays inside the image.
    const int mx = std::max(64, visible->width() / 8);
    const int my = std::max(64, visible->height() / 8);
    const QRect canvasRect(0, 0, frame->canvas.size.width, frame->canvas.size.height);
    QRect rect = visible->adjusted(-mx, -my, mx, my) & canvasRect;
    if (qint64(rect.width()) * rect.height() > budget) {
        rect = *visible;
    }

    InFlight flight;
    flight.id = ++_nextFlightId;
    flight.frame = *frame;
    flight.reversed = _reversed;
    flight.rect = rect;
    flight.edit = _edit;
    flight.patch = _health->patch;
    flight.cancel = std::make_shared<std::atomic<bool>>(false);
    _flight = std::move(flight);
    _rerun = false;
    ++_requestsStarted;
    _lastRequested = rect;
    setStatus(tr("Computing %1 x %2 px of level %3...").arg(rect.width()).arg(rect.height()).arg(frame->level));
    send(nullptr);
}

void InkPreviewOverlayController::send(const InkPreviewClient::Upload* upload)
{
    InkPreviewClient::View view;
    view.key = _flight->frame.key;
    view.canvas = QSize(_flight->frame.canvas.size.width, _flight->frame.canvas.size.height);
    view.rect = _flight->rect;
    view.reversed = _flight->reversed;
    view.patch = _flight->patch;
    const quint64 id = _flight->id;
    _client->requestPreview(view, upload, [this, id](const InkPreviewClient::Reply& reply) { onReply(id, reply); });
}

void InkPreviewOverlayController::onReply(quint64 id, const InkPreviewClient::Reply& reply)
{
    if (!_flight || _flight->id != id) {
        return;
    }
    switch (reply.kind) {
    case InkPreviewClient::Reply::Kind::Image:
        install(reply);
        finishFlight(true);
        return;
    case InkPreviewClient::Reply::Kind::Need:
        if (reply.depth != int(layerOffsets().size())) {
            if (_depthRecheckKey != _flight->frame.key) {
                // The server may have been restarted with another model: read its health
                // again, once per key.
                _depthRecheckKey = _flight->frame.key;
                _health.reset();
                setStatus(tr("The server now reads %1 layers; reconnecting.").arg(reply.depth), true);
                finishFlight(false);
                _settle->start();
                return;
            }
            // Its health still names another depth: stop until the view, the key or the
            // server address changes, instead of asking again and again.
            setStatus(tr("The server reports %1 layers in GET /health but asks for %2; the live preview stops until "
                         "the view, the segment or the server address changes.")
                          .arg(layerOffsets().size())
                          .arg(reply.depth),
                      true);
            finishFlight(false);
            return;
        }
        if (_flight->uploads >= 2) {
            setStatus(tr("The server asked again for layers it was sent; its cache may be too small for the view."), true);
            finishFlight(false);
            return;
        }
        sampleNeed(reply.need);
        return;
    case InkPreviewClient::Reply::Kind::Failed:
        // Nothing new is drawn; the status says why. A server that did not answer is
        // checked again before the next request.
        if (reply.httpStatus == 0) {
            _health.reset();
        }
        setStatus(reply.error, true);
        finishFlight(false);
        return;
    }
}

void InkPreviewOverlayController::sampleNeed(const QRect& need)
{
    VolumeViewerBase* viewer = _target;
    auto* surface = isSegmentationView(viewer) ? dynamic_cast<QuadSurface*>(viewer->currentSurface()) : nullptr;
    const auto volume = surface ? viewer->currentVolume() : nullptr;
    const auto opened = _state ? _state->openedVolumes() : nullptr;
    if (!surface || !volume || !opened || surface->id != _flight->frame.surfaceId ||
        volume->id() != _flight->frame.volumeId || _edit != _flight->edit) {
        // The voxels under the view changed while the server answered: ask again for the new ones.
        cancelFlight();
        _settle->start();
        return;
    }

    if (_snapshot && _snapshot->surfaceId == surface->id && _snapshot->edit == _edit &&
        _snapshot->volumeId != volume->id()) {
        // The same grid on another volume: its levels are placed differently.
        auto snapshot = std::make_shared<InkPreviewSnapshot>();
        snapshot->surfaceId = _snapshot->surfaceId;
        snapshot->edit = _snapshot->edit;
        snapshot->volumeId = volume->id();
        snapshot->surface = _snapshot->surface;
        _snapshot = std::move(snapshot);
    }
    if (!_snapshot || _snapshot->surfaceId != surface->id || _snapshot->edit != _edit) {
        const cv::Mat_<cv::Vec3f>* points = static_cast<const QuadSurface&>(*surface).rawPointsPtr();
        if (!points || points->empty()) {
            setStatus(tr("The surface has no points to sample."), true);
            finishFlight(false);
            return;
        }
        auto copy = std::make_shared<QuadSurface>(new cv::Mat_<cv::Vec3f>(points->clone()), surface->scale());
        copy->setComponents(surface->components());
        auto snapshot = std::make_shared<InkPreviewSnapshot>();
        snapshot->surfaceId = surface->id;
        snapshot->edit = _edit;
        snapshot->volumeId = volume->id();
        snapshot->surface = std::move(copy);
        _snapshot = std::move(snapshot);
    }

    const quint64 id = _flight->id;
    const int level = _flight->frame.level;
    const vc::RenderCanvas canvas = _flight->frame.canvas;
    const cv::Rect pixels(need.x(), need.y(), need.width(), need.height());
    const auto offsets = layerOffsets();
    const auto cancel = _flight->cancel;
    const auto snapshot = _snapshot;
    setStatus(tr("Sampling %1 x %2 px, %3 layers of level %4...")
                  .arg(need.width())
                  .arg(need.height())
                  .arg(offsets.size())
                  .arg(level));
    runInBackground<Sampled>(this,
        [opened, volume, level, snapshot, canvas, pixels, offsets, cancel]() {
            return sampleLayers(opened, volume, level, snapshot, canvas, pixels, offsets, cancel);
        },
        [this, id, need](Sampled sampled) {
            if (!_flight || _flight->id != id) {
                return;
            }
            if (!sampled.error.isEmpty()) {
                // Nothing new is drawn: no fill and no other level stands in for chunks that could not be read.
                setStatus(tr("Could not sample the layers: %1").arg(sampled.error), true);
                finishFlight(false);
                return;
            }
            InkPreviewClient::Upload upload;
            upload.rect = need;
            upload.wide = sampled.wide;
            upload.voxels = std::move(sampled.voxels);
            ++_flight->uploads;
            setStatus(tr("Computing %1 x %2 px of level %3...")
                          .arg(_flight->rect.width())
                          .arg(_flight->rect.height())
                          .arg(_flight->frame.level));
            send(&upload);
        });
}

void InkPreviewOverlayController::install(const InkPreviewClient::Reply& reply)
{
    const QRect rect = _flight->rect;
    Shown shown;
    shown.frame = _flight->frame;
    shown.reversed = _flight->reversed;
    shown.rect = rect;
    shown.scalar = cv::Mat_<uint8_t>(rect.height(), rect.width());
    std::memcpy(shown.scalar.data, reply.pixels.constData(), std::size_t(rect.width()) * std::size_t(rect.height()));
    // A fixed mapping: byte value v is probability v / 255 in every view.
    shown.image = vc3d::scalarOverlayImage(shown.scalar, _colormapId);
    shown.pixmap = QPixmap::fromImage(shown.image);
    _shown = std::move(shown);

    const QString layers = _flight->reversed ? tr("layers reversed (control view)") : tr("forward layers");
    setStatus(tr("Model output, %1: %2 (%3), %4 x %5 px of level %6; %7 patches computed, %8 cached.")
                  .arg(layers, _health ? _health->model : QString(), reply.backend)
                  .arg(rect.width())
                  .arg(rect.height())
                  .arg(_flight->frame.level)
                  .arg(reply.computed)
                  .arg(reply.cached));
    if (_target) {
        refreshViewer(_target);
    }
}

void InkPreviewOverlayController::finishFlight(bool rerun)
{
    const bool again = rerun && _rerun;
    _rerun = false;
    _flight.reset();
    if (again && _enabled) {
        _settle->start();
    }
}

void InkPreviewOverlayController::cancelFlight()
{
    if (!_flight) {
        return;
    }
    _flight->cancel->store(true);
    _flight.reset();
    _rerun = false;
    _client->cancel();
    // The client dropped a health check too, if one was waiting.
    _healthPending = false;
}

void InkPreviewOverlayController::dropShown()
{
    if (!_shown) {
        return;
    }
    _shown.reset();
    if (_target) {
        refreshViewer(_target);
    }
}

#ifdef VC_TESTING
void InkPreviewOverlayController::setHealthForTesting(const InkPreviewClient::Health& health)
{
    _healthPending = false;
    _health = health;
}

void InkPreviewOverlayController::setLevelsForTesting(const std::string& volumeId, const std::vector<LevelInfo>& levels)
{
    _levels = levels;
    _levelsVolumeId = volumeId;
    _levelsPendingId.clear();
}

void InkPreviewOverlayController::settleForTesting()
{
    _settle->stop();
    startRequest();
}

bool InkPreviewOverlayController::settlePendingForTesting() const
{
    return _settle->isActive();
}

QString InkPreviewOverlayController::keyForTesting(VolumeViewerBase* viewer) const
{
    auto* surface = viewer ? dynamic_cast<QuadSurface*>(viewer->currentSurface()) : nullptr;
    const auto frame = surface ? frameFor(viewer, *surface) : std::nullopt;
    return frame ? frame->key : QString();
}

std::optional<vc::RenderCanvas> InkPreviewOverlayController::canvasForTesting(VolumeViewerBase* viewer) const
{
    auto* surface = viewer ? dynamic_cast<QuadSurface*>(viewer->currentSurface()) : nullptr;
    const auto frame = surface ? frameFor(viewer, *surface) : std::nullopt;
    return frame ? std::optional<vc::RenderCanvas>(frame->canvas) : std::nullopt;
}

std::optional<QRect> InkPreviewOverlayController::visibleRectForTesting(VolumeViewerBase* viewer) const
{
    auto* surface = viewer ? dynamic_cast<QuadSurface*>(viewer->currentSurface()) : nullptr;
    const auto frame = surface ? frameFor(viewer, *surface) : std::nullopt;
    return frame ? visibleCanvasRect(viewer, *surface, frame->canvas) : std::nullopt;
}

void InkPreviewOverlayController::showForTesting(VolumeViewerBase* viewer, const QRect& rect,
                                                 const cv::Mat_<uint8_t>& pixels)
{
    auto* surface = viewer ? dynamic_cast<QuadSurface*>(viewer->currentSurface()) : nullptr;
    const auto frame = surface ? frameFor(viewer, *surface) : std::nullopt;
    if (!frame) {
        return;
    }
    Shown shown;
    shown.frame = *frame;
    shown.reversed = _reversed;
    shown.rect = rect;
    shown.scalar = pixels.clone();
    shown.image = vc3d::scalarOverlayImage(shown.scalar, _colormapId);
    shown.pixmap = QPixmap::fromImage(shown.image);
    _shown = std::move(shown);
}
#endif
