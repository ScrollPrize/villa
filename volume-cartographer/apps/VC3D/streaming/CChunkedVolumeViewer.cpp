#include "streaming/CChunkedVolumeViewer.hpp"

#include "CState.hpp"
#include "VCSettings.hpp"
#include "ViewerManager.hpp"
#include "vc/core/render/Colormaps.hpp"
#include "vc/core/render/PostProcess.hpp"
#include "vc/core/render/ZarrChunkFetcher.hpp"
#include "render/ChunkCache.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/Geometry.hpp"
#include "vc/core/util/Surface.hpp"

#include <QApplication>
#include <QCursor>
#include <QElapsedTimer>
#include <QGraphicsEllipseItem>
#include <QGraphicsItem>
#include <QGraphicsPathItem>
#include <QGraphicsScene>
#include <QLabel>
#include <QPainter>
#include <QPainterPath>
#include <QPointer>
#include <QSettings>
#include <QTimer>
#include <QTransform>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <sstream>
#include <unordered_map>

#include <opencv2/imgproc.hpp>

namespace {

constexpr float kMinScale = 0.002f;
constexpr float kMaxScale = 128.0f;
constexpr int kInteractionSettleMs = 140;
constexpr int kResizeSettleMs = 140;
constexpr int kChunkReadyActiveDelayMs = 500;
constexpr float kResolutionLodZoomBias = 0.5f;
constexpr float kSegmentationResolutionLodZoomBias = 1.0f;
constexpr int kSurfaceResolutionLevelBias = 1;
constexpr int kInitialSegmentationSurfaceLevel = 5;
constexpr double kSlowMotionPxPerSec = 180.0;
constexpr double kFastMotionPxPerSec = 1800.0;
constexpr qint64 kInteractivePreviewMinIntervalMs = 50;
constexpr float kPanSmoothingAlpha = 0.65f;
constexpr std::array<QRgb, 12> kIntersectionPalette = {
    qRgb(255, 120, 120), qRgb(120, 200, 255), qRgb(120, 255, 140),
    qRgb(255, 220, 100), qRgb(220, 140, 255), qRgb(255, 160, 200),
    qRgb(140, 255, 220), qRgb(200, 255, 140), qRgb(255, 180, 120),
    qRgb(180, 200, 255), qRgb(255, 140, 180), qRgb(160, 255, 180),
};
constexpr int kIntersectionZ = 100;
constexpr int kHighlightedIntersectionZ = 110;
constexpr int kActiveIntersectionZ = 120;
constexpr float kActiveIntersectionOpacityScale = 1.2f;
constexpr float kActiveIntersectionWidthScale = 1.3f;
constexpr float kActiveIntersectionMinWidthDelta = 0.75f;

struct IntersectionStyle {
    QRgb color = 0;
    int z = kIntersectionZ;
    int widthQ = 0;

    bool operator==(const IntersectionStyle& other) const
    {
        return color == other.color && z == other.z && widthQ == other.widthQ;
    }
};

struct IntersectionStyleHash {
    size_t operator()(const IntersectionStyle& style) const
    {
        size_t h = std::hash<QRgb>{}(style.color);
        h ^= std::hash<int>{}(style.z) + 0x9e3779b9u + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(style.widthQ) + 0x9e3779b9u + (h << 6) + (h >> 2);
        return h;
    }
};

bool isSupportedStreamingCompositeMethod(const std::string& method)
{
    return method == "mean" || method == "max" || method == "min" || method == "alpha";
}

int dominantAxis(const cv::Vec3f& v, float axisEps = 1e-4f)
{
    int axis = 0;
    float best = std::abs(v[0]);
    for (int i = 1; i < 3; ++i) {
        const float a = std::abs(v[i]);
        if (a > best) {
            best = a;
            axis = i;
        }
    }
    if (best < 1.0f - axisEps)
        return -1;
    for (int i = 0; i < 3; ++i) {
        if (i != axis && std::abs(v[i]) > axisEps)
            return -1;
    }
    return axis;
}

std::vector<vc::render::ChunkCache::LevelInfo>
makeLevelInfo(const vc::render::OpenedChunkedZarr& opened)
{
    std::vector<vc::render::ChunkCache::LevelInfo> levels;
    levels.reserve(opened.fetchers.size());
    for (std::size_t i = 0; i < opened.fetchers.size(); ++i) {
        vc::render::ChunkCache::LevelInfo level;
        level.shape = opened.shapes[i];
        level.chunkShape = opened.chunkShapes[i];
        level.transform = opened.transforms[i];
        levels.push_back(level);
    }
    return levels;
}

std::string stableHexHash(const std::string& value)
{
    std::uint64_t hash = 1469598103934665603ULL;
    for (unsigned char c : value) {
        hash ^= static_cast<std::uint64_t>(c);
        hash *= 1099511628211ULL;
    }
    std::ostringstream out;
    out << std::hex << hash;
    return out.str();
}

std::string normalizedVolumeCacheIdentity(const std::shared_ptr<Volume>& volume)
{
    if (!volume)
        return {};
    if (volume->isRemote()) {
        return "remote|" + volume->remoteUrl() +
               "|base=" + std::to_string(volume->baseScaleLevel()) +
               "|id=" + volume->id();
    }

    std::error_code ec;
    auto path = std::filesystem::weakly_canonical(volume->path(), ec);
    if (ec)
        path = std::filesystem::absolute(volume->path(), ec);
    if (ec)
        path = volume->path();
    return "local|" + path.string() + "|id=" + volume->id();
}

uint32_t alphaBlendArgb(uint32_t base, uint32_t overlay, float alpha)
{
    const float a = std::clamp(alpha, 0.0f, 1.0f);
    const auto mix = [a](uint32_t b, uint32_t o) -> uint32_t {
        return static_cast<uint32_t>(
            std::clamp(std::lround((1.0f - a) * float(b) + a * float(o)), 0L, 255L));
    };
    const uint32_t br = (base >> 16) & 0xFFu;
    const uint32_t bg = (base >> 8) & 0xFFu;
    const uint32_t bb = base & 0xFFu;
    const uint32_t or_ = (overlay >> 16) & 0xFFu;
    const uint32_t og = (overlay >> 8) & 0xFFu;
    const uint32_t ob = overlay & 0xFFu;
    return 0xFF000000u | (mix(br, or_) << 16) | (mix(bg, og) << 8) | mix(bb, ob);
}

QColor activeSegmentationColorForView(const std::string& surfName)
{
    if (surfName == "seg yz" || surfName == "yz plane")
        return QColor(Qt::yellow);
    if (surfName == "seg xz" || surfName == "xz plane")
        return QColor(Qt::red);
    return QColor(255, 140, 0);
}

float activeSegmentationIntersectionWidth(float baseWidth)
{
    return std::max(baseWidth * kActiveIntersectionWidthScale,
                    baseWidth + kActiveIntersectionMinWidthDelta);
}

QString formatVec3(const cv::Vec3f& v)
{
    return QString("(%1, %2, %3)")
        .arg(v[0], 0, 'f', 1)
        .arg(v[1], 0, 'f', 1)
        .arg(v[2], 0, 'f', 1);
}

QString planeCoordinateText(PlaneSurface& plane)
{
    return QString("plane pos %1").arg(formatVec3(plane.origin()));
}

std::size_t streamingCacheCapacityBytes(const CState* state)
{
    constexpr std::size_t kFallbackCapacity = 2ULL * 1024ULL * 1024ULL * 1024ULL;
    if (!state || state->cacheSizeBytes() == 0)
        return kFallbackCapacity;
    return state->cacheSizeBytes();
}

float scaleForSurfaceRenderStartLevel(int renderLevel, int numLevels)
{
    const int maxLevel = std::max(0, numLevels - 1);
    const int clampedRenderLevel = std::clamp(renderLevel, 0, maxLevel);
    int dsLevel = clampedRenderLevel + kSurfaceResolutionLevelBias;
    if (dsLevel > maxLevel)
        dsLevel = maxLevel;

    const float dsScale = static_cast<float>(std::uint64_t{1} << dsLevel);
    return std::clamp(0.75f / (dsScale * kResolutionLodZoomBias), kMinScale, kMaxScale);
}

float scaleForCoarsestPlaneRenderLevel(int numLevels)
{
    const int coarsestLevel = std::max(0, numLevels - 1);
    const float dsScale = static_cast<float>(std::uint64_t{1} << coarsestLevel);
    return std::clamp(0.75f / (dsScale * kResolutionLodZoomBias), kMinScale, kMaxScale);
}

std::unique_ptr<vc::render::ChunkCache> makeChunkCacheForVolume(const std::shared_ptr<Volume>& volume,
                                                                std::size_t decodedByteCapacity)
{
    if (!volume)
        return nullptr;

    const bool isRemote = volume->isRemote();
    vc::render::OpenedChunkedZarr opened = isRemote
        ? vc::render::openHttpZarrPyramid(
              volume->remoteUrl(), volume->remoteAuth(), volume->baseScaleLevel())
        : vc::render::openLocalZarrPyramid(volume->path());

    if (opened.fetchers.empty())
        return nullptr;

    vc::render::ChunkCache::Options options;
    options.decodedByteCapacity = decodedByteCapacity > 0
        ? decodedByteCapacity
        : streamingCacheCapacityBytes(nullptr);
    options.maxConcurrentReads = 16;
    if (isRemote) {
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        const QString defaultCache = vc3d::defaultCacheBase() + "/remote_cache";
        const auto cacheRoot = std::filesystem::path(
            settings.value(vc3d::settings::viewer::REMOTE_CACHE_DIR, defaultCache)
                .toString()
                .toStdString());
        options.persistentCachePath = cacheRoot / stableHexHash(normalizedVolumeCacheIdentity(volume));
    }

    return std::make_unique<vc::render::ChunkCache>(
        makeLevelInfo(opened), opened.fetchers, opened.fillValue, opened.dtype, options);
}

} // namespace

CChunkedVolumeViewer::CChunkedVolumeViewer(CState* state, ViewerManager* manager, QWidget* parent)
    : QWidget(parent)
    , _state(state)
    , _viewerManager(manager)
{
    _view = new CVolumeViewerView(this);
    _view->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    _view->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    _view->setTransformationAnchor(QGraphicsView::NoAnchor);
    _view->setAlignment(Qt::AlignLeft | Qt::AlignTop);
    _view->setRenderHint(QPainter::Antialiasing, false);
    _view->setScrollPanDisabled(true);
    _view->setViewportUpdateMode(QGraphicsView::FullViewportUpdate);

    connect(_view, &CVolumeViewerView::sendScrolled, this, &CChunkedVolumeViewer::onScrolled);
    connect(_view, &CVolumeViewerView::sendVolumeClicked, this, &CChunkedVolumeViewer::onVolumeClicked);
    connect(_view, &CVolumeViewerView::sendZoom, this, &CChunkedVolumeViewer::onZoom);
    connect(_view, &CVolumeViewerView::sendResized, this, &CChunkedVolumeViewer::onResized);
    connect(_view, &CVolumeViewerView::sendCursorMove, this, &CChunkedVolumeViewer::onCursorMove);
    connect(_view, &CVolumeViewerView::sendPanRelease, this, &CChunkedVolumeViewer::onPanRelease);
    connect(_view, &CVolumeViewerView::sendPanStart, this, &CChunkedVolumeViewer::onPanStart);
    connect(_view, &CVolumeViewerView::sendMousePress, this, &CChunkedVolumeViewer::onMousePress);
    connect(_view, &CVolumeViewerView::sendMouseMove, this, &CChunkedVolumeViewer::onMouseMove);
    connect(_view, &CVolumeViewerView::sendMouseRelease, this, &CChunkedVolumeViewer::onMouseRelease);
    connect(_view, &CVolumeViewerView::sendKeyPress, this, &CChunkedVolumeViewer::onKeyPress);
    connect(_view, &CVolumeViewerView::sendKeyRelease, this, &CChunkedVolumeViewer::onKeyRelease);

    _scene = new QGraphicsScene(this);
    _scene->setItemIndexMethod(QGraphicsScene::NoIndex);
    _view->setScene(_scene);
    _view->setDirectFramebuffer(&_framebuffer);

    _renderTimer = new QTimer(this);
    _renderTimer->setSingleShot(true);
    _renderTimer->setInterval(16);
    connect(_renderTimer, &QTimer::timeout, this, [this]() {
        if (!_renderPending)
            return;
        if (_interactivePreview) {
            if (_settleRenderTimer)
                _settleRenderTimer->start();
            return;
        }
        _renderPending = false;
        submitRender();
        updateStatusLabel();
    });

    _settleRenderTimer = new QTimer(this);
    _settleRenderTimer->setSingleShot(true);
    _settleRenderTimer->setInterval(kInteractionSettleMs);
    connect(_settleRenderTimer, &QTimer::timeout, this, [this]() {
        if (!_isPanning) {
            _interactivePreview = false;
            scheduleRender();
        }
    });

    _resizeRenderTimer = new QTimer(this);
    _resizeRenderTimer->setSingleShot(true);
    _resizeRenderTimer->setInterval(kResizeSettleMs);
    connect(_resizeRenderTimer, &QTimer::timeout, this, [this]() {
        scheduleRender();
        emit overlaysUpdated();
    });

    reloadPerfSettings();

    auto* layout = new QVBoxLayout;
    layout->addWidget(_view);
    setLayout(layout);

    _lbl = new QLabel(this);
    _lbl->setStyleSheet("QLabel { color : #00FF00; background-color: rgba(0,0,0,128); padding: 2px 4px; }");
    _lbl->setMinimumWidth(520);
    _lbl->move(10, 5);
}

CChunkedVolumeViewer::~CChunkedVolumeViewer()
{
    if (_chunkCbId != 0 && _chunkArray) {
        _chunkArray->removeChunkReadyListener(_chunkCbId);
        _chunkCbId = 0;
    }
    if (_overlayChunkCbId != 0 && _overlayChunkArray) {
        _overlayChunkArray->removeChunkReadyListener(_overlayChunkCbId);
        _overlayChunkCbId = 0;
    }
    clearIntersectionItems();
}

void CChunkedVolumeViewer::reloadPerfSettings()
{
    QSettings s(vc3d::settingsFilePath(), QSettings::IniFormat);
    using namespace vc3d::settings;
    _panSensitivity = std::max(0.01f, s.value(viewer::PAN_SENSITIVITY, viewer::PAN_SENSITIVITY_DEFAULT).toFloat());
    _zoomSensitivity = std::max(0.01f, s.value(viewer::ZOOM_SENSITIVITY, viewer::ZOOM_SENSITIVITY_DEFAULT).toFloat());
    _zScrollSensitivity = std::max(0.01f, s.value(viewer::ZSCROLL_SENSITIVITY, viewer::ZSCROLL_SENSITIVITY_DEFAULT).toFloat());
    const int interpIdx = s.value(perf::INTERPOLATION_METHOD, 1).toInt();
    _samplingMethod = static_cast<vc::Sampling>(std::clamp(interpIdx, 0, 1));
}

void CChunkedVolumeViewer::setSurface(const std::string& name)
{
    _surfName = name;
    if (_state)
        onSurfaceChanged(name, _state->surface(name));
}

Surface* CChunkedVolumeViewer::currentSurface() const
{
    if (!_state) {
        auto shared = _surfWeak.lock();
        return shared ? shared.get() : nullptr;
    }
    return _state->surfaceRaw(_surfName);
}

void CChunkedVolumeViewer::rebuildChunkArray()
{
    if (_chunkCbId != 0 && _chunkArray) {
        _chunkArray->removeChunkReadyListener(_chunkCbId);
        _chunkCbId = 0;
    }
    _chunkArray.reset();
    if (!_volume)
        return;

    try {
        _chunkArray = makeChunkCacheForVolume(_volume, streamingCacheCapacityBytes(_state));
    } catch (const std::exception& e) {
        if (_lbl)
            _lbl->setText(QString("Streaming unavailable: %1").arg(e.what()));
        return;
    }

    if (!_chunkArray)
        return;

    QPointer<CChunkedVolumeViewer> guard(this);
    std::weak_ptr<Volume> volumeWeak = _volume;
    _chunkCbId = _chunkArray->addChunkReadyListener([guard, volumeWeak]() {
        QMetaObject::invokeMethod(qApp, [guard, volumeWeak]() {
            if (!guard)
                return;
            auto volume = volumeWeak.lock();
            if (!volume || guard->_volume != volume)
                return;
            if (guard->_interactivePreview) {
                if (guard->_settleRenderTimer)
                    guard->_settleRenderTimer->start(kChunkReadyActiveDelayMs);
                return;
            }
            guard->scheduleRender();
        }, Qt::QueuedConnection);
    });
}

void CChunkedVolumeViewer::OnVolumeChanged(std::shared_ptr<Volume> vol)
{
    invalidateIntersect();
    if (_surfWeak.lock() == _defaultSurface) {
        _surfWeak.reset();
        _defaultSurface.reset();
    }
    _genCacheDirty = true;
    _genCoords.release();
    _genNormals.release();
    _zOffWorldDir = {0, 0, 0};
    _stableFramebufferValid = false;
    if (_cursorCrosshair)
        _cursorCrosshair->hide();
    if (_focusMarker)
        _focusMarker->hide();

    _volume = std::move(vol);
    rebuildChunkArray();
    ensureDefaultSurface();
    if (_volume && isAxisAlignedView()) {
        const int n = _chunkArray ? _chunkArray->numLevels()
                                  : static_cast<int>(_volume->numScales());
        _scale = scaleForCoarsestPlaneRenderLevel(n);
    }
    recalcPyramidLevel();
    if (_volume) {
        const double vs = _volume->voxelSize() / static_cast<double>(_dsScale);
        _view->setVoxelSize(vs, vs);
    }
    updateContentBounds();
    resizeFramebuffer();
    scheduleRender();
    renderIntersections();
    updateStatusLabel();
}

void CChunkedVolumeViewer::onSurfaceChanged(const std::string& name,
                                            const std::shared_ptr<Surface>& surf,
                                            bool isEditUpdate)
{
    const bool isCurrentSurface = (_surfName == name);
    const bool isIntersectionTarget =
        _intersectTgts.count(name) != 0 ||
        (_intersectTgts.count("visible_segmentation") != 0 &&
         (name == "segmentation" || _highlightedSurfaceIds.count(name) != 0));

    if (!isCurrentSurface) {
        if (isIntersectionTarget) {
            invalidateIntersect(name);
            renderIntersections();
            emit overlaysUpdated();
        }
        return;
    }

    _surfWeak = surf;
    _genCacheDirty = true;
    _zOffWorldDir = {0, 0, 0};
    _stableFramebufferValid = false;
    invalidateIntersect(name);
    if (!surf) {
        clearIntersectionItems();
        _scene->clear();
        _overlayGroups.clear();
        _cursorCrosshair = nullptr;
        _focusMarker = nullptr;
        return;
    }
    updateContentBounds();
    if (!isEditUpdate && _resetViewOnSurfaceChange && _surfName == "segmentation" &&
        dynamic_cast<QuadSurface*>(surf.get())) {
        _surfacePtrX = 0.0f;
        _surfacePtrY = 0.0f;
        _zOff = 0.0f;
        const int n = _chunkArray ? _chunkArray->numLevels()
                                  : (_volume ? static_cast<int>(_volume->numScales()) : 1);
        _scale = scaleForSurfaceRenderStartLevel(kInitialSegmentationSurfaceLevel, n);
        recalcPyramidLevel();
    }
    updateFocusMarker();
    scheduleRender();
    renderIntersections();
    emit overlaysUpdated();
}

void CChunkedVolumeViewer::onSurfaceWillBeDeleted(const std::string&, const std::shared_ptr<Surface>& surf)
{
    auto current = _surfWeak.lock();
    if (current && current == surf)
        _surfWeak.reset();
}

void CChunkedVolumeViewer::onVolumeClosing()
{
    if (_chunkCbId != 0 && _chunkArray) {
        _chunkArray->removeChunkReadyListener(_chunkCbId);
        _chunkCbId = 0;
    }
    _chunkArray.reset();
    _volume.reset();
    invalidateIntersect();
    onSurfaceChanged(_surfName, nullptr);
}

void CChunkedVolumeViewer::onPOIChanged(const std::string& name, POI* poi)
{
    if (name != "focus" || !poi)
        return;

    auto surf = _surfWeak.lock();
    if (auto* plane = dynamic_cast<PlaneSurface*>(surf.get())) {
        plane->setOrigin(poi->p);
        if (cv::norm(poi->n) > 0.5f)
            plane->setNormal(poi->n);
        updateContentBounds();
        _genCacheDirty = true;
    }

    updateFocusMarker(poi);
    updateStatusLabel();
    emit overlaysUpdated();
    scheduleRender();
    renderIntersections();
}

void CChunkedVolumeViewer::ensureDefaultSurface()
{
    if (_surfWeak.lock() || !_volume || !isAxisAlignedView())
        return;
    const auto shape = _volume->shape();
    cv::Vec3f center(static_cast<float>(shape[0]) * 0.5f,
                     static_cast<float>(shape[1]) * 0.5f,
                     static_cast<float>(shape[2]) * 0.5f);
    cv::Vec3f normal;
    if (_surfName == "xy plane") normal = {0, 0, 1};
    else if (_surfName == "xz plane" || _surfName == "seg xz") normal = {0, 1, 0};
    else normal = {1, 0, 0};
    _defaultSurface = std::make_shared<PlaneSurface>(center, normal);
    _surfWeak = _defaultSurface;
}

bool CChunkedVolumeViewer::isAxisAlignedView() const
{
    return _surfName == "xy plane" || _surfName == "xz plane" ||
           _surfName == "yz plane" || _surfName == "seg xz" ||
           _surfName == "seg yz";
}

void CChunkedVolumeViewer::updateContentBounds()
{
    auto surf = _surfWeak.lock();
    if (!_volume || !surf)
        return;
    auto* plane = dynamic_cast<PlaneSurface*>(surf.get());
    if (!plane)
        return;

    const auto [w, h, d] = _volume->shape();
    const float corners[][3] = {
        {0, 0, 0}, {float(w), 0, 0}, {0, float(h), 0}, {float(w), float(h), 0},
        {0, 0, float(d)}, {float(w), 0, float(d)}, {0, float(h), float(d)}, {float(w), float(h), float(d)}
    };
    _contentMinU = _contentMinV = std::numeric_limits<float>::max();
    _contentMaxU = _contentMaxV = std::numeric_limits<float>::lowest();
    for (const auto& c : corners) {
        const cv::Vec3f proj = plane->project({c[0], c[1], c[2]}, 1.0, 1.0);
        _contentMinU = std::min(_contentMinU, proj[0]);
        _contentMinV = std::min(_contentMinV, proj[1]);
        _contentMaxU = std::max(_contentMaxU, proj[0]);
        _contentMaxV = std::max(_contentMaxV, proj[1]);
    }
}

void CChunkedVolumeViewer::recalcPyramidLevel()
{
    const int n = _chunkArray ? _chunkArray->numLevels() : (_volume ? static_cast<int>(_volume->numScales()) : 1);
    const float lodZoomBias = _surfName == "segmentation"
        ? kSegmentationResolutionLodZoomBias
        : kResolutionLodZoomBias;
    const float lodScale = std::max(_scale * lodZoomBias, 1e-6f);
    _dsScaleIdx = std::clamp(
        static_cast<int>(std::floor(std::max(0.0f, std::log2(1.0f / lodScale)))),
        0, std::max(0, n - 1));
    _dsScale = static_cast<float>(std::uint64_t{1} << _dsScaleIdx);
}

void CChunkedVolumeViewer::resizeFramebuffer()
{
    const QSize vpSize = _view->viewport()->size();
    const int w = std::max(1, vpSize.width());
    const int h = std::max(1, vpSize.height());
    if (_framebuffer.isNull() || _framebuffer.width() != w || _framebuffer.height() != h) {
        _framebuffer = QImage(w, h, QImage::Format_RGB32);
        _framebuffer.fill(QColor(64, 64, 64));
        _stableFramebufferValid = false;
    }
    _scene->setSceneRect(0, 0, w, h);
}

void CChunkedVolumeViewer::scheduleRender()
{
    syncCameraTransform();
    _renderPending = true;
    if (_interactivePreview) {
        if (_settleRenderTimer)
            _settleRenderTimer->start();
        return;
    }
    if (_renderTimer && !_renderTimer->isActive())
        _renderTimer->start();
}

void CChunkedVolumeViewer::syncCameraTransform()
{
    _camSurfX = _surfacePtrX;
    _camSurfY = _surfacePtrY;
    _camScale = _scale;
    updateFocusMarker();
}

bool CChunkedVolumeViewer::renderInteractiveAxisAlignedSlicePreview()
{
    if (!_chunkArray || !_volume || _framebuffer.isNull())
        return false;
    if (_overlayVolume || _compositeSettings.enabled || _compositeSettings.planeEnabled)
        return false;

    auto surf = _surfWeak.lock();
    auto* plane = dynamic_cast<PlaneSurface*>(surf.get());
    if (!plane)
        return false;

    const cv::Vec3f vx = plane->basisX();
    const cv::Vec3f vy = plane->basisY();
    const cv::Vec3f n = plane->normal({0, 0, 0});
    const int uAxis = dominantAxis(vx);
    const int vAxis = dominantAxis(vy);
    const int fixedAxis = dominantAxis(n);
    if (uAxis < 0 || vAxis < 0 || fixedAxis < 0 ||
        uAxis == vAxis || uAxis == fixedAxis || vAxis == fixedAxis)
        return false;

    const int fbW = _framebuffer.width();
    const int fbH = _framebuffer.height();
    if (fbW <= 0 || fbH <= 0)
        return false;

    const int level = renderStartLevel();
    if (level < 0 || level >= _chunkArray->numLevels())
        return false;

    const auto shapeZyx = _chunkArray->shape(level);
    const auto chunkZyx = _chunkArray->chunkShape(level);
    std::array<int, 3> shapeXyz{shapeZyx[2], shapeZyx[1], shapeZyx[0]};
    std::array<int, 3> chunkXyz{chunkZyx[2], chunkZyx[1], chunkZyx[0]};
    if (chunkXyz[0] <= 0 || chunkXyz[1] <= 0 || chunkXyz[2] <= 0)
        return false;

    const float halfW = static_cast<float>(fbW) * 0.5f / _scale;
    const float halfH = static_cast<float>(fbH) * 0.5f / _scale;
    const cv::Vec3f origin0 = vx * (_surfacePtrX - halfW)
                            + vy * (_surfacePtrY - halfH)
                            + plane->origin()
                            + n * _zOff;
    const cv::Vec3f vxStep0 = vx / _scale;
    const cv::Vec3f vyStep0 = vy / _scale;
    const auto transform = _chunkArray->levelTransform(level);
    auto toLevel = [&transform](const cv::Vec3f& p) {
        return cv::Vec3f(
            float(double(p[0]) * transform.scaleFromLevel0[0] + transform.offsetFromLevel0[0]),
            float(double(p[1]) * transform.scaleFromLevel0[1] + transform.offsetFromLevel0[1]),
            float(double(p[2]) * transform.scaleFromLevel0[2] + transform.offsetFromLevel0[2]));
    };
    auto stepToLevel = [&transform](const cv::Vec3f& p) {
        return cv::Vec3f(
            float(double(p[0]) * transform.scaleFromLevel0[0]),
            float(double(p[1]) * transform.scaleFromLevel0[1]),
            float(double(p[2]) * transform.scaleFromLevel0[2]));
    };

    const cv::Vec3f origin = toLevel(origin0);
    const cv::Vec3f uStep = stepToLevel(vxStep0);
    const cv::Vec3f vStep = stepToLevel(vyStep0);
    if (std::abs(uStep[fixedAxis]) > 1e-5f || std::abs(vStep[fixedAxis]) > 1e-5f)
        return false;

    if (origin[fixedAxis] < 0.0f || origin[fixedAxis] >= float(shapeXyz[fixedAxis]))
        return false;

    const int fixed = std::clamp(int(std::lround(origin[fixedAxis])), 0, shapeXyz[fixedAxis] - 1);
    const float u0 = origin[uAxis];
    const float u1 = origin[uAxis] + uStep[uAxis] * float(std::max(0, fbW - 1));
    const float v0 = origin[vAxis];
    const float v1 = origin[vAxis] + vStep[vAxis] * float(std::max(0, fbH - 1));

    int uBegin = std::clamp(int(std::floor(std::min(u0, u1))) - 1, 0, shapeXyz[uAxis]);
    int uEnd = std::clamp(int(std::ceil(std::max(u0, u1))) + 2, 0, shapeXyz[uAxis]);
    int vBegin = std::clamp(int(std::floor(std::min(v0, v1))) - 1, 0, shapeXyz[vAxis]);
    int vEnd = std::clamp(int(std::ceil(std::max(v0, v1))) + 2, 0, shapeXyz[vAxis]);
    if (uEnd <= uBegin || vEnd <= vBegin) {
        _framebuffer.fill(QColor(64, 64, 64));
        syncCameraTransform();
        _view->viewport()->update();
        return true;
    }

    const int srcW = uEnd - uBegin;
    const int srcH = vEnd - vBegin;
    constexpr int kMaxPreviewSourcePixels = 4096 * 4096;
    if (srcW <= 0 || srcH <= 0 || srcW * srcH > kMaxPreviewSourcePixels)
        return false;

    const uint8_t fillByte = static_cast<uint8_t>(
        std::clamp(std::lround(_chunkArray->fillValue()), 0L, 255L));
    cv::Mat_<uint8_t> src(srcH, srcW, fillByte);
    cv::Mat_<uint8_t> srcCoverage(srcH, srcW, uint8_t(0));

    const int uChunkBegin = uBegin / chunkXyz[uAxis];
    const int uChunkEnd = (uEnd - 1) / chunkXyz[uAxis];
    const int vChunkBegin = vBegin / chunkXyz[vAxis];
    const int vChunkEnd = (vEnd - 1) / chunkXyz[vAxis];
    const int fixedChunk = fixed / chunkXyz[fixedAxis];
    int coveredChunks = 0;

    for (int vc = vChunkBegin; vc <= vChunkEnd; ++vc) {
        for (int uc = uChunkBegin; uc <= uChunkEnd; ++uc) {
            std::array<int, 3> chunkXyzCoord{};
            chunkXyzCoord[uAxis] = uc;
            chunkXyzCoord[vAxis] = vc;
            chunkXyzCoord[fixedAxis] = fixedChunk;

            vc::render::ChunkResult chunk = _chunkArray->tryGetChunk(
                level,
                chunkXyzCoord[2],
                chunkXyzCoord[1],
                chunkXyzCoord[0]);
            if (chunk.status == vc::render::ChunkStatus::MissQueued ||
                chunk.status == vc::render::ChunkStatus::Error)
                continue;

            const int cu0 = chunkXyzCoord[uAxis] * chunkXyz[uAxis];
            const int cv0 = chunkXyzCoord[vAxis] * chunkXyz[vAxis];
            const int uA = std::max(uBegin, cu0);
            const int uB = std::min(uEnd, cu0 + chunkXyz[uAxis]);
            const int vA = std::max(vBegin, cv0);
            const int vB = std::min(vEnd, cv0 + chunkXyz[vAxis]);
            if (uA >= uB || vA >= vB)
                continue;

            ++coveredChunks;
            if (chunk.status == vc::render::ChunkStatus::AllFill) {
                src(cv::Range(vA - vBegin, vB - vBegin),
                    cv::Range(uA - uBegin, uB - uBegin)).setTo(fillByte);
                srcCoverage(cv::Range(vA - vBegin, vB - vBegin),
                            cv::Range(uA - uBegin, uB - uBegin)).setTo(uint8_t(1));
                continue;
            }
            if (chunk.status != vc::render::ChunkStatus::Data || !chunk.bytes)
                continue;

            const auto& bytes = *chunk.bytes;
            for (int vv = vA; vv < vB; ++vv) {
                uint8_t* dst = src.ptr<uint8_t>(vv - vBegin);
                uint8_t* cov = srcCoverage.ptr<uint8_t>(vv - vBegin);
                for (int uu = uA; uu < uB; ++uu) {
                    std::array<int, 3> xyz{};
                    xyz[uAxis] = uu;
                    xyz[vAxis] = vv;
                    xyz[fixedAxis] = fixed;
                    const int lx = xyz[0] - chunkXyzCoord[0] * chunkXyz[0];
                    const int ly = xyz[1] - chunkXyzCoord[1] * chunkXyz[1];
                    const int lz = xyz[2] - chunkXyzCoord[2] * chunkXyz[2];
                    const std::size_t offset = (std::size_t(lz) * std::size_t(chunkZyx[1])
                                              + std::size_t(ly)) * std::size_t(chunkZyx[2])
                                              + std::size_t(lx);
                    if (offset >= bytes.size())
                        continue;
                    dst[uu - uBegin] = std::to_integer<uint8_t>(bytes[offset]);
                    cov[uu - uBegin] = 1;
                }
            }
        }
    }

    if (coveredChunks == 0)
        return false;

    cv::Mat_<uint8_t> displayValues;
    cv::Mat_<uint8_t> displayCoverage;
    const cv::Matx23f dstToSrc(
        uStep[uAxis], 0.0f, u0 - float(uBegin),
        0.0f, vStep[vAxis], v0 - float(vBegin));
    cv::warpAffine(src, displayValues, dstToSrc, cv::Size(fbW, fbH),
                   cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                   cv::BORDER_CONSTANT, cv::Scalar(fillByte));
    cv::warpAffine(srcCoverage, displayCoverage, dstToSrc, cv::Size(fbW, fbH),
                   cv::INTER_NEAREST | cv::WARP_INVERSE_MAP,
                   cv::BORDER_CONSTANT, cv::Scalar(0));

    std::array<uint32_t, 256> lut{};
    vc::buildWindowLevelColormapLut(lut, _windowLow, _windowHigh, _baseColormapId);
    QImage preview(fbW, fbH, QImage::Format_RGB32);
    auto* bits = reinterpret_cast<uint32_t*>(preview.bits());
    const int stride = preview.bytesPerLine() / 4;
    for (int y = 0; y < fbH; ++y) {
        auto* row = bits + size_t(y) * size_t(stride);
        const auto* values = displayValues.ptr<uint8_t>(y);
        const auto* coverage = displayCoverage.ptr<uint8_t>(y);
        for (int x = 0; x < fbW; ++x)
            row[x] = coverage[x] ? lut[values[x]] : 0xFF404040u;
    }

    _framebuffer = std::move(preview);
    syncCameraTransform();
    if (_interactivePreview)
        updateIntersectionPreviewTransform();
    else
        renderIntersections();
    emit overlaysUpdated();
    _view->viewport()->update();
    return true;
}

void CChunkedVolumeViewer::updateInteractivePreviewFromStableFrame(float newSurfX,
                                                                   float newSurfY,
                                                                   float newScale)
{
    resizeFramebuffer();
    if (renderInteractiveAxisAlignedSlicePreview())
        return;

    if (!_stableFramebufferValid || _stableFramebuffer.isNull() ||
        _stableFramebuffer.size() != _framebuffer.size() ||
        _stableScale <= 0.0f || newScale <= 0.0f) {
        syncCameraTransform();
        if (_interactivePreview)
            updateIntersectionPreviewTransform();
        else
            renderIntersections();
        _view->viewport()->update();
        return;
    }

    const int w = _framebuffer.width();
    const int h = _framebuffer.height();
    const float cx = float(w) * 0.5f;
    const float cy = float(h) * 0.5f;
    const float r = newScale / _stableScale;
    const float tx = (_stableSurfX - newSurfX) * newScale + cx - cx * r;
    const float ty = (_stableSurfY - newSurfY) * newScale + cy - cy * r;

    QImage preview(w, h, QImage::Format_RGB32);
    preview.fill(QColor(64, 64, 64));
    QPainter painter(&preview);
    painter.setRenderHint(QPainter::SmoothPixmapTransform, std::abs(r - 1.0f) > 0.02f);
    painter.translate(tx, ty);
    painter.scale(r, r);
    painter.drawImage(0, 0, _stableFramebuffer);
    painter.end();

    _framebuffer = std::move(preview);
    syncCameraTransform();
    if (_interactivePreview)
        updateIntersectionPreviewTransform();
    else
        renderIntersections();
    emit overlaysUpdated();
    _view->viewport()->update();
}

bool CChunkedVolumeViewer::shouldRefreshInteractivePreview()
{
    if (!_interactionClock.isValid())
        _interactionClock.start();
    const qint64 now = _interactionClock.elapsed();
    if (_lastInteractivePreviewMs < 0 ||
        now - _lastInteractivePreviewMs >= kInteractivePreviewMinIntervalMs) {
        _lastInteractivePreviewMs = now;
        return true;
    }
    return false;
}

int CChunkedVolumeViewer::renderStartLevel(bool preferSurfaceResolution) const
{
    if (!_chunkArray)
        return 0;

    // `_dsScaleIdx` intentionally waits for about 2x more zoom before moving
    // to a finer level. Plane views can bias coarser during active motion;
    // surface-resolution views keep their target level to avoid panning blur.
    int level = _dsScaleIdx;
    if (preferSurfaceResolution && _chunkArray && level < _chunkArray->numLevels() - 1)
        level -= kSurfaceResolutionLevelBias;
    if (_interactivePreview && !preferSurfaceResolution && _chunkArray->numLevels() > 1) {
        const double speed = std::max(0.0, _interactionSpeedPxPerSec);
        const double t = std::clamp((speed - kSlowMotionPxPerSec) /
                                    (kFastMotionPxPerSec - kSlowMotionPxPerSec),
                                    0.0, 1.0);
        const int maxBias = std::max(0, _chunkArray->numLevels() - 1 - level);
        const int bias = int(std::ceil(t * double(maxBias)));
        level += bias;
    }
    return std::clamp(level, 0, _chunkArray->numLevels() - 1);
}

void CChunkedVolumeViewer::markInteractiveMotion(double motionPx)
{
    if (!_interactionClock.isValid())
        _interactionClock.start();

    const qint64 now = _interactionClock.elapsed();
    if (_lastInteractionMs >= 0) {
        const qint64 dtMs = std::max<qint64>(1, now - _lastInteractionMs);
        const double instantaneous = std::max(0.0, motionPx) * 1000.0 / double(dtMs);
        _interactionSpeedPxPerSec = 0.65 * _interactionSpeedPxPerSec + 0.35 * instantaneous;
    } else {
        _interactionSpeedPxPerSec = std::max(0.0, motionPx) * 60.0;
    }
    _lastInteractionMs = now;
    _interactivePreview = true;
    if (_settleRenderTimer)
        _settleRenderTimer->start();
}

int CChunkedVolumeViewer::genericPreviewDownsampleFactor() const
{
    return 1;
}

bool CChunkedVolumeViewer::streamingCompositeUnsupported() const
{
    return !isSupportedStreamingCompositeMethod(_compositeSettings.params.method) ||
           _compositeSettings.params.lightingEnabled ||
           _compositeSettings.params.method == "beerLambert" ||
           _compositeSettings.postClaheEnabled ||
           _compositeSettings.postRakingEnabled ||
           _compositeSettings.postRemoveSmallComponents ||
           _compositeSettings.useVolumeGradients;
}

void CChunkedVolumeViewer::samplePlaneIntoValues(
    const cv::Vec3f& origin,
    const cv::Vec3f& vxStep,
    const cv::Vec3f& vyStep,
    const cv::Vec3f& normal,
    int startLevel,
    const vc::render::ChunkedPlaneSampler::Options& options,
    cv::Mat_<uint8_t>& values,
    cv::Mat_<uint8_t>& coverage)
{
    const bool wantComposite = _compositeSettings.planeEnabled && !streamingCompositeUnsupported();
    if (!wantComposite) {
        if (_interactivePreview) {
            vc::render::ChunkedPlaneSampler::samplePlaneLevel(
                *_chunkArray, startLevel, origin, vxStep, vyStep, values, coverage, options);
        } else {
            vc::render::ChunkedPlaneSampler::samplePlaneFineToCoarse(
                *_chunkArray, startLevel, origin, vxStep, vyStep, values, coverage, options);
        }
        return;
    }

    const int front = std::max(0, _compositeSettings.planeLayersFront);
    const int behind = std::max(0, _compositeSettings.planeLayersBehind);
    const int numLayers = front + behind + 1;
    const int zStart = -behind;
    const float zStep = _compositeSettings.reverseDirection ? -1.0f : 1.0f;
    const auto compositeOptions = vc::render::ChunkedPlaneSampler::Options(vc::Sampling::Nearest, options.tileSize);

    std::vector<cv::Mat_<uint8_t>> layerValues;
    std::vector<cv::Mat_<uint8_t>> layerCoverage;
    layerValues.reserve(numLayers);
    layerCoverage.reserve(numLayers);
    for (int i = 0; i < numLayers; ++i) {
        layerValues.emplace_back(values.rows, values.cols, uint8_t(0));
        layerCoverage.emplace_back(values.rows, values.cols, uint8_t(0));
        const cv::Vec3f layerOrigin = origin + normal * (float(zStart + i) * zStep);
        if (_interactivePreview) {
            vc::render::ChunkedPlaneSampler::samplePlaneLevel(
                *_chunkArray, startLevel, layerOrigin, vxStep, vyStep,
                layerValues.back(), layerCoverage.back(), compositeOptions);
        } else {
            vc::render::ChunkedPlaneSampler::samplePlaneFineToCoarse(
                *_chunkArray, startLevel, layerOrigin, vxStep, vyStep,
                layerValues.back(), layerCoverage.back(), compositeOptions);
        }
    }

    LayerStack stack;
    stack.values.resize(numLayers);
    for (int y = 0; y < values.rows; ++y) {
        auto* dst = values.ptr<uint8_t>(y);
        auto* cov = coverage.ptr<uint8_t>(y);
        for (int x = 0; x < values.cols; ++x) {
            stack.validCount = 0;
            for (int i = 0; i < numLayers; ++i) {
                if (!layerCoverage[i](y, x))
                    continue;
                const float value = static_cast<float>(layerValues[i](y, x));
                if (value < static_cast<float>(_compositeSettings.params.isoCutoff))
                    continue;
                stack.values[stack.validCount++] = value;
            }
            if (stack.validCount > 0) {
                dst[x] = static_cast<uint8_t>(std::clamp(
                    compositeLayerStack(stack, _compositeSettings.params), 0.0f, 255.0f));
                cov[x] = 1;
            }
        }
    }
}

void CChunkedVolumeViewer::sampleCoordsIntoValues(
    const cv::Mat_<cv::Vec3f>& coords,
    const cv::Mat_<cv::Vec3f>& normals,
    int startLevel,
    const vc::render::ChunkedPlaneSampler::Options& options,
    cv::Mat_<uint8_t>& values,
    cv::Mat_<uint8_t>& coverage)
{
    const bool wantComposite = _compositeSettings.enabled &&
                               !streamingCompositeUnsupported() &&
                               !normals.empty();
    if (!wantComposite) {
        if (_interactivePreview) {
            vc::render::ChunkedPlaneSampler::sampleCoordsLevel(
                *_chunkArray, startLevel, coords, values, coverage, options);
        } else {
            vc::render::ChunkedPlaneSampler::sampleCoordsFineToCoarse(
                *_chunkArray, startLevel, coords, values, coverage, options);
        }
        return;
    }

    const int front = std::max(0, _compositeSettings.layersFront);
    const int behind = std::max(0, _compositeSettings.layersBehind);
    const int numLayers = front + behind + 1;
    const int zStart = -behind;
    const float zStep = _compositeSettings.reverseDirection ? -1.0f : 1.0f;
    const auto compositeOptions = vc::render::ChunkedPlaneSampler::Options(vc::Sampling::Nearest, options.tileSize);

    std::vector<cv::Mat_<uint8_t>> layerValues;
    std::vector<cv::Mat_<uint8_t>> layerCoverage;
    layerValues.reserve(numLayers);
    layerCoverage.reserve(numLayers);
    cv::Mat_<cv::Vec3f> layerCoords(coords.rows, coords.cols);
    for (int i = 0; i < numLayers; ++i) {
        const float offset = float(zStart + i) * zStep;
        for (int y = 0; y < coords.rows; ++y) {
            const auto* src = coords.ptr<cv::Vec3f>(y);
            const auto* nrow = normals.ptr<cv::Vec3f>(y);
            auto* dst = layerCoords.ptr<cv::Vec3f>(y);
            for (int x = 0; x < coords.cols; ++x) {
                if (!std::isfinite(src[x][0]) || src[x][0] == -1.0f) {
                    dst[x] = src[x];
                } else {
                    dst[x] = src[x] + nrow[x] * offset;
                }
            }
        }
        layerValues.emplace_back(values.rows, values.cols, uint8_t(0));
        layerCoverage.emplace_back(values.rows, values.cols, uint8_t(0));
        if (_interactivePreview) {
            vc::render::ChunkedPlaneSampler::sampleCoordsLevel(
                *_chunkArray, startLevel, layerCoords,
                layerValues.back(), layerCoverage.back(), compositeOptions);
        } else {
            vc::render::ChunkedPlaneSampler::sampleCoordsFineToCoarse(
                *_chunkArray, startLevel, layerCoords,
                layerValues.back(), layerCoverage.back(), compositeOptions);
        }
    }

    LayerStack stack;
    stack.values.resize(numLayers);
    for (int y = 0; y < values.rows; ++y) {
        auto* dst = values.ptr<uint8_t>(y);
        auto* cov = coverage.ptr<uint8_t>(y);
        for (int x = 0; x < values.cols; ++x) {
            stack.validCount = 0;
            for (int i = 0; i < numLayers; ++i) {
                if (!layerCoverage[i](y, x))
                    continue;
                const float value = static_cast<float>(layerValues[i](y, x));
                if (value < static_cast<float>(_compositeSettings.params.isoCutoff))
                    continue;
                stack.values[stack.validCount++] = value;
            }
            if (stack.validCount > 0) {
                dst[x] = static_cast<uint8_t>(std::clamp(
                    compositeLayerStack(stack, _compositeSettings.params), 0.0f, 255.0f));
                cov[x] = 1;
            }
        }
    }
}

void CChunkedVolumeViewer::submitRender()
{
    auto surf = _surfWeak.lock();
    if (!surf || !_volume || !_chunkArray) {
        return;
    }

    resizeFramebuffer();
    const int fbW = _framebuffer.width();
    const int fbH = _framebuffer.height();
    if (fbW <= 0 || fbH <= 0)
        return;

    _values.create(fbH, fbW);
    _coverage.create(fbH, fbW);
    _values.setTo(0);
    _coverage.setTo(0);
    cv::Mat_<uint8_t> overlayValues;
    cv::Mat_<uint8_t> overlayCoverage;

    const vc::render::ChunkedPlaneSampler::Options options(_samplingMethod, 32);

    if (auto* plane = dynamic_cast<PlaneSurface*>(surf.get())) {
        const int startLevel = renderStartLevel(false);
        const cv::Vec3f vx = plane->basisX();
        const cv::Vec3f vy = plane->basisY();
        const cv::Vec3f n = plane->normal({0, 0, 0});
        const float halfW = static_cast<float>(fbW) * 0.5f / _scale;
        const float halfH = static_cast<float>(fbH) * 0.5f / _scale;
        const cv::Vec3f origin = vx * (_surfacePtrX - halfW)
                               + vy * (_surfacePtrY - halfH)
                               + plane->origin()
                               + n * _zOff;
        const cv::Vec3f vxStep = vx / _scale;
        const cv::Vec3f vyStep = vy / _scale;
        samplePlaneIntoValues(origin, vxStep, vyStep, n, startLevel, options, _values, _coverage);
        renderOverlayVolumeForPlane(origin, vxStep, vyStep, startLevel, options,
                                    overlayValues, overlayCoverage);
    } else {
        const int startLevel = renderStartLevel(true);
        const int previewFactor = genericPreviewDownsampleFactor();
        if (previewFactor > 1) {
            const int previewW = std::max(1, (fbW + previewFactor - 1) / previewFactor);
            const int previewH = std::max(1, (fbH + previewFactor - 1) / previewFactor);
            const float previewScale = std::max(kMinScale, _scale / float(previewFactor));
            const cv::Vec3f offset(_surfacePtrX * previewScale - float(previewW) * 0.5f,
                                   _surfacePtrY * previewScale - float(previewH) * 0.5f,
                                   0.0f);
            cv::Mat_<cv::Vec3f> previewCoords;
            cv::Mat_<cv::Vec3f> previewNormals;
            surf->gen(&previewCoords,
                      _zOff != 0.0f ? &previewNormals : nullptr,
                      cv::Size(previewW, previewH), {0, 0, 0}, previewScale, offset);
            if (_zOff != 0.0f &&
                _zOffWorldDir == cv::Vec3f(0.0f, 0.0f, 0.0f) &&
                !previewNormals.empty()) {
                const cv::Vec3f n = previewNormals(previewNormals.rows / 2, previewNormals.cols / 2);
                const float len = static_cast<float>(cv::norm(n));
                if (len > 1e-6f)
                    _zOffWorldDir = n / len;
            }
            if (_zOff != 0.0f && _zOffWorldDir != cv::Vec3f(0.0f, 0.0f, 0.0f)) {
                const cv::Vec3f tr = _zOffWorldDir * _zOff;
                for (int y = 0; y < previewCoords.rows; ++y) {
                    auto* row = previewCoords.ptr<cv::Vec3f>(y);
                    for (int x = 0; x < previewCoords.cols; ++x) {
                        if (row[x][0] == row[x][0] && row[x][0] != -1.0f)
                            row[x] += tr;
                    }
                }
            }

            cv::Mat_<uint8_t> previewValues(previewH, previewW, uint8_t(0));
            cv::Mat_<uint8_t> previewCoverage(previewH, previewW, uint8_t(0));
            if (_interactivePreview) {
                vc::render::ChunkedPlaneSampler::sampleCoordsLevel(
                    *_chunkArray, startLevel, previewCoords, previewValues, previewCoverage, options);
            } else {
                vc::render::ChunkedPlaneSampler::sampleCoordsFineToCoarse(
                    *_chunkArray, startLevel, previewCoords, previewValues, previewCoverage, options);
            }

            std::array<uint32_t, 256> lut{};
            vc::buildWindowLevelColormapLut(lut, _windowLow, _windowHigh, _baseColormapId);
            QImage previewImage(previewW, previewH, QImage::Format_RGB32);
            auto* previewBits = reinterpret_cast<uint32_t*>(previewImage.bits());
            const int previewStride = previewImage.bytesPerLine() / 4;
            for (int y = 0; y < previewH; ++y) {
                auto* row = previewBits + size_t(y) * size_t(previewStride);
                const auto* src = previewValues.ptr<uint8_t>(y);
                const auto* cov = previewCoverage.ptr<uint8_t>(y);
                for (int x = 0; x < previewW; ++x)
                    row[x] = cov[x] ? lut[src[x]] : 0xFF000000u;
            }
            _framebuffer = previewImage.scaled(fbW, fbH, Qt::IgnoreAspectRatio, Qt::FastTransformation);
            syncCameraTransform();
            if (_interactivePreview)
                updateIntersectionPreviewTransform();
            else
                renderIntersections();
            emit overlaysUpdated();
            _view->viewport()->update();
            updateStatusLabel();
            return;
        }

        cv::Vec3f offset(_surfacePtrX * _scale - float(fbW) * 0.5f,
                         _surfacePtrY * _scale - float(fbH) * 0.5f,
                         0.0f);
        const bool cacheHit = !_genCacheDirty &&
                              _genCacheSurfKey == surf.get() &&
                              _genCacheFbW == fbW &&
                              _genCacheFbH == fbH &&
                              _genCacheScale == _scale &&
                              _genCacheOffset == offset &&
                              _genCacheZOff == _zOff &&
                              _genCacheZOffDir == _zOffWorldDir &&
                              !_genCoords.empty();
        if (!cacheHit) {
            surf->gen(&_genCoords, &_genNormals, cv::Size(fbW, fbH), {0, 0, 0}, _scale, offset);
            if (_zOff != 0.0f &&
                _zOffWorldDir == cv::Vec3f(0.0f, 0.0f, 0.0f) &&
                !_genNormals.empty()) {
                const cv::Vec3f n = _genNormals(_genNormals.rows / 2, _genNormals.cols / 2);
                const float len = static_cast<float>(cv::norm(n));
                if (len > 1e-6f)
                    _zOffWorldDir = n / len;
            }
            if (_zOff != 0.0f && _zOffWorldDir != cv::Vec3f(0.0f, 0.0f, 0.0f)) {
                const cv::Vec3f tr = _zOffWorldDir * _zOff;
                for (int y = 0; y < _genCoords.rows; ++y) {
                    auto* row = _genCoords.ptr<cv::Vec3f>(y);
                    for (int x = 0; x < _genCoords.cols; ++x) {
                        if (row[x][0] == row[x][0] && row[x][0] != -1.0f)
                            row[x] += tr;
                    }
                }
            }
            _genCacheSurfKey = surf.get();
            _genCacheFbW = fbW;
            _genCacheFbH = fbH;
            _genCacheScale = _scale;
            _genCacheOffset = offset;
            _genCacheZOff = _zOff;
            _genCacheZOffDir = _zOffWorldDir;
            _genCacheDirty = false;
        }
        if (!_genCoords.empty()) {
            sampleCoordsIntoValues(_genCoords, _genNormals, startLevel, options, _values, _coverage);
            renderOverlayVolumeForCoords(_genCoords, startLevel, options,
                                         overlayValues, overlayCoverage);
        }
    }

    std::array<uint32_t, 256> lut{};
    vc::buildWindowLevelColormapLut(lut, _windowLow, _windowHigh, _baseColormapId);
    std::array<uint32_t, 256> overlayLut{};
    const bool hasOverlay = !overlayValues.empty() && !overlayCoverage.empty() &&
                            _overlayOpacity > 0.0f;
    if (hasOverlay) {
        vc::buildWindowLevelColormapLut(
            overlayLut, _overlayWindowLow, _overlayWindowHigh, _overlayColormapId);
    }
    auto* fbBits = reinterpret_cast<uint32_t*>(_framebuffer.bits());
    const int fbStride = _framebuffer.bytesPerLine() / 4;
    const bool planeView = dynamic_cast<PlaneSurface*>(surf.get()) != nullptr;
    const uint32_t uncoveredPixel = planeView ? 0xFF404040u : 0xFF000000u;
    for (int y = 0; y < fbH; ++y) {
        auto* row = fbBits + size_t(y) * size_t(fbStride);
        const auto* src = _values.ptr<uint8_t>(y);
        const auto* cov = _coverage.ptr<uint8_t>(y);
        const auto* overlaySrc = hasOverlay ? overlayValues.ptr<uint8_t>(y) : nullptr;
        const auto* overlayCov = hasOverlay ? overlayCoverage.ptr<uint8_t>(y) : nullptr;
        for (int x = 0; x < fbW; ++x) {
            uint32_t pixel = cov[x] ? lut[src[x]] : uncoveredPixel;
            if (hasOverlay && overlayCov[x] &&
                overlaySrc[x] >= _overlayWindowLow && overlaySrc[x] <= _overlayWindowHigh) {
                pixel = alphaBlendArgb(pixel, overlayLut[overlaySrc[x]], _overlayOpacity);
            }
            row[x] = pixel;
        }
    }

    syncCameraTransform();
    if (_interactivePreview)
        updateIntersectionPreviewTransform();
    else
        renderIntersections();
    _stableFramebuffer = _framebuffer.copy();
    _stableSurfX = _surfacePtrX;
    _stableSurfY = _surfacePtrY;
    _stableScale = _scale;
    _stableFramebufferValid = !_stableFramebuffer.isNull();
    emit overlaysUpdated();
    _view->viewport()->update();
    updateStatusLabel();
}

void CChunkedVolumeViewer::renderOverlayVolumeForPlane(
    const cv::Vec3f& origin,
    const cv::Vec3f& vxStep,
    const cv::Vec3f& vyStep,
    int startLevel,
    const vc::render::ChunkedPlaneSampler::Options& options,
    cv::Mat_<uint8_t>& overlayValues,
    cv::Mat_<uint8_t>& overlayCoverage)
{
    if (!_overlayVolume || !_overlayChunkArray || _overlayOpacity <= 0.0f ||
        _coverage.empty() || _overlayChunkArray->numLevels() <= 0)
        return;

    overlayValues.create(_coverage.rows, _coverage.cols);
    overlayCoverage.create(_coverage.rows, _coverage.cols);
    overlayValues.setTo(0);
    overlayCoverage.setTo(0);
    const int level = std::clamp(startLevel, 0, _overlayChunkArray->numLevels() - 1);
    if (_interactivePreview) {
        vc::render::ChunkedPlaneSampler::samplePlaneLevel(
            *_overlayChunkArray, level, origin, vxStep, vyStep,
            overlayValues, overlayCoverage, options);
    } else {
        vc::render::ChunkedPlaneSampler::samplePlaneFineToCoarse(
            *_overlayChunkArray, level, origin, vxStep, vyStep,
            overlayValues, overlayCoverage, options);
    }
}

void CChunkedVolumeViewer::renderOverlayVolumeForCoords(
    const cv::Mat_<cv::Vec3f>& coords,
    int startLevel,
    const vc::render::ChunkedPlaneSampler::Options& options,
    cv::Mat_<uint8_t>& overlayValues,
    cv::Mat_<uint8_t>& overlayCoverage)
{
    if (!_overlayVolume || !_overlayChunkArray || _overlayOpacity <= 0.0f ||
        _coverage.empty() || coords.empty() || _overlayChunkArray->numLevels() <= 0)
        return;

    overlayValues.create(_coverage.rows, _coverage.cols);
    overlayCoverage.create(_coverage.rows, _coverage.cols);
    overlayValues.setTo(0);
    overlayCoverage.setTo(0);
    const int level = std::clamp(startLevel, 0, _overlayChunkArray->numLevels() - 1);
    if (_interactivePreview) {
        vc::render::ChunkedPlaneSampler::sampleCoordsLevel(
            *_overlayChunkArray, level, coords, overlayValues, overlayCoverage, options);
    } else {
        vc::render::ChunkedPlaneSampler::sampleCoordsFineToCoarse(
            *_overlayChunkArray, level, coords, overlayValues, overlayCoverage, options);
    }
}

void CChunkedVolumeViewer::renderVisible(bool force)
{
    if (!force) {
        scheduleRender();
        return;
    }
    if (_renderTimer && _renderTimer->isActive())
        _renderTimer->stop();
    _renderPending = false;
    submitRender();
    updateStatusLabel();
}

void CChunkedVolumeViewer::setVolumeWindow(float low, float high)
{
    const float clampedLow = std::clamp(low, 0.0f, 255.0f);
    float clampedHigh = std::clamp(high, 0.0f, 255.0f);
    if (clampedHigh <= clampedLow)
        clampedHigh = std::min(255.0f, clampedLow + 1.0f);
    if (std::abs(_windowLow - clampedLow) < 1e-6f &&
        std::abs(_windowHigh - clampedHigh) < 1e-6f)
        return;
    _windowLow = clampedLow;
    _windowHigh = clampedHigh;
    scheduleRender();
}

void CChunkedVolumeViewer::setOverlayVolume(std::shared_ptr<Volume> volume)
{
    if (_overlayChunkCbId != 0 && _overlayChunkArray) {
        _overlayChunkArray->removeChunkReadyListener(_overlayChunkCbId);
        _overlayChunkCbId = 0;
    }
    _overlayVolume = std::move(volume);
    _overlayChunkArray.reset();
    if (_overlayVolume) {
        try {
            _overlayChunkArray = makeChunkCacheForVolume(_overlayVolume, streamingCacheCapacityBytes(_state));
        } catch (const std::exception&) {
            _overlayChunkArray.reset();
        }
        if (_overlayChunkArray) {
            QPointer<CChunkedVolumeViewer> guard(this);
            std::weak_ptr<Volume> overlayVolumeWeak = _overlayVolume;
            _overlayChunkCbId = _overlayChunkArray->addChunkReadyListener([guard, overlayVolumeWeak]() {
                QMetaObject::invokeMethod(qApp, [guard, overlayVolumeWeak]() {
                    if (!guard)
                        return;
                    auto volume = overlayVolumeWeak.lock();
                    if (!volume || guard->_overlayVolume != volume)
                        return;
                    if (guard->_interactivePreview) {
                        if (guard->_settleRenderTimer)
                            guard->_settleRenderTimer->start(kChunkReadyActiveDelayMs);
                        return;
                    }
                    guard->scheduleRender();
                }, Qt::QueuedConnection);
            });
        }
    }
    scheduleRender();
}

void CChunkedVolumeViewer::setOverlayOpacity(float opacity)
{
    _overlayOpacity = std::clamp(opacity, 0.0f, 1.0f);
    scheduleRender();
}

void CChunkedVolumeViewer::setOverlayColormap(const std::string& colormapId)
{
    _overlayColormapId = colormapId;
    scheduleRender();
}

void CChunkedVolumeViewer::setOverlayThreshold(float threshold)
{
    setOverlayWindow(std::max(threshold, 0.0f), _overlayWindowHigh);
}

void CChunkedVolumeViewer::setOverlayWindow(float low, float high)
{
    _overlayWindowLow = std::clamp(low, 0.0f, 255.0f);
    _overlayWindowHigh = std::clamp(high, _overlayWindowLow + 1.0f, 255.0f);
    scheduleRender();
}

void CChunkedVolumeViewer::panByF(float dx, float dy)
{
    markInteractiveMotion(std::hypot(double(dx), double(dy)));
    const float invScale = _panSensitivity / _scale;
    _surfacePtrX -= dx * invScale;
    _surfacePtrY -= dy * invScale;
    if (_contentMaxU > _contentMinU) {
        _surfacePtrX = std::clamp(_surfacePtrX, _contentMinU, _contentMaxU);
        _surfacePtrY = std::clamp(_surfacePtrY, _contentMinV, _contentMaxV);
    }
    if (shouldRefreshInteractivePreview())
        updateInteractivePreviewFromStableFrame(_surfacePtrX, _surfacePtrY, _scale);
    scheduleRender();
    emit overlaysUpdated();
}

void CChunkedVolumeViewer::zoomStepsAt(int steps, const QPointF& scenePos)
{
    if (steps == 0)
        return;
    const double zoomMotionPx = std::hypot(double(_view->viewport()->width()),
                                          double(_view->viewport()->height())) *
                                0.08 * std::abs(double(steps));
    markInteractiveMotion(zoomMotionPx);
    const float factor = std::pow(1.05f, static_cast<float>(steps) * _zoomSensitivity);
    const float newScale = std::clamp(_scale * factor, kMinScale, kMaxScale);
    if (std::abs(newScale - _scale) < _scale * 1e-6f)
        return;
    const float vpW = static_cast<float>(_view->viewport()->width());
    const float vpH = static_cast<float>(_view->viewport()->height());
    const float mx = static_cast<float>(scenePos.x());
    const float my = static_cast<float>(scenePos.y());
    if (mx >= 0 && mx < vpW && my >= 0 && my < vpH) {
        const float dx = mx - vpW * 0.5f;
        const float dy = my - vpH * 0.5f;
        _surfacePtrX += dx * (1.0f / _scale - 1.0f / newScale);
        _surfacePtrY += dy * (1.0f / _scale - 1.0f / newScale);
    }
    _scale = newScale;
    recalcPyramidLevel();
    _genCacheDirty = true;
    resizeFramebuffer();
    if (shouldRefreshInteractivePreview())
        updateInteractivePreviewFromStableFrame(_surfacePtrX, _surfacePtrY, _scale);
    scheduleRender();
    emit overlaysUpdated();
}

void CChunkedVolumeViewer::adjustZoomByFactor(float factor)
{
    const int steps = (factor > 1.0f) ? 1 : (factor < 1.0f ? -1 : 0);
    zoomStepsAt(steps, QPointF(_view->viewport()->width() * 0.5, _view->viewport()->height() * 0.5));
}

void CChunkedVolumeViewer::notifyInteractiveViewChange(double motionPx)
{
    if (!_volume || !_chunkArray)
        return;

    markInteractiveMotion(motionPx);
    _genCacheDirty = true;
    if (shouldRefreshInteractivePreview()) {
        _renderPending = false;
        submitRender();
    }
    scheduleRender();
    emit overlaysUpdated();
}

void CChunkedVolumeViewer::adjustSurfaceOffset(float delta)
{
    float maxZ = 10000.0f;
    if (_volume) {
        const auto [w, h, d] = _volume->shape();
        maxZ = static_cast<float>(std::max({w, h, d}));
    }
    _zOff = std::clamp(_zOff + delta, -maxZ, maxZ);
    _genCacheDirty = true;
    scheduleRender();
    updateStatusLabel();
}

void CChunkedVolumeViewer::resetSurfaceOffsets()
{
    _surfacePtrX = 0.0f;
    _surfacePtrY = 0.0f;
    _zOff = 0.0f;
    _zOffWorldDir = {0, 0, 0};
    _genCacheDirty = true;
    _stableFramebufferValid = false;
    scheduleRender();
}

void CChunkedVolumeViewer::fitSurfaceInView()
{
    _surfacePtrX = 0.0f;
    _surfacePtrY = 0.0f;
    _scale = 0.5f;
    recalcPyramidLevel();
    _genCacheDirty = true;
    _stableFramebufferValid = false;
    scheduleRender();
}

void CChunkedVolumeViewer::centerOnVolumePoint(const cv::Vec3f& point, bool forceRender)
{
    auto surf = _surfWeak.lock();
    if (!surf)
        return;
    cv::Vec2f surfacePoint(0.0f, 0.0f);
    bool haveSurfacePoint = false;
    if (auto* plane = dynamic_cast<PlaneSurface*>(surf.get())) {
        const cv::Vec3f projected = plane->project(point, 1.0, 1.0);
        surfacePoint = {projected[0], projected[1]};
        haveSurfacePoint = true;
    } else if (auto* quad = dynamic_cast<QuadSurface*>(surf.get())) {
        cv::Vec3f ptr = quad->pointer();
        auto* patchIndex = _viewerManager ? _viewerManager->surfacePatchIndex() : nullptr;
        if (quad->pointTo(ptr, point, 4.0f, 100, patchIndex) >= 0.0f) {
            const cv::Vec3f loc = quad->loc(ptr);
            surfacePoint = {loc[0], loc[1]};
            haveSurfacePoint = true;
        }
    }

    if (!haveSurfacePoint ||
        !std::isfinite(surfacePoint[0]) ||
        !std::isfinite(surfacePoint[1])) {
        return;
    }

    centerOnSurfacePoint(surfacePoint, forceRender);
}

void CChunkedVolumeViewer::centerOnSurfacePoint(const cv::Vec2f& point, bool forceRender)
{
    if (!std::isfinite(point[0]) || !std::isfinite(point[1]))
        return;

    _surfacePtrX = point[0];
    _surfacePtrY = point[1];
    _genCacheDirty = true;
    _stableFramebufferValid = false;
    if (forceRender) {
        renderVisible(true);
    } else {
        scheduleRender();
    }
    emit overlaysUpdated();
}

void CChunkedVolumeViewer::onZoom(int steps, QPointF scenePoint, Qt::KeyboardModifiers modifiers)
{
    auto surf = _surfWeak.lock();
    if (!surf)
        return;
    if (modifiers & Qt::ShiftModifier) {
        markInteractiveMotion(std::abs(double(steps)) * 96.0);
        _zOff += static_cast<float>(steps) * _zScrollSensitivity;
        _genCacheDirty = true;
        scheduleRender();
    } else if (modifiers & Qt::ControlModifier) {
        emit sendSegmentationRadiusWheel(steps, scenePoint, sceneToVolume(scenePoint));
    } else {
        zoomStepsAt(steps > 0 ? 1 : (steps < 0 ? -1 : 0), scenePoint);
    }
}

void CChunkedVolumeViewer::onResized()
{
    resizeFramebuffer();
    _genCacheDirty = true;
    if (_renderTimer && _renderTimer->isActive())
        _renderTimer->stop();
    _renderPending = false;
    if (_resizeRenderTimer)
        _resizeRenderTimer->start();
    _view->viewport()->update();
}

void CChunkedVolumeViewer::onCursorMove(QPointF scenePos)
{
    _lastScenePos = scenePos;
    updateCursorCrosshair(scenePos);
    if (_viewerManager) {
        auto surf = _surfWeak.lock();
        if (surf) {
            cv::Vec3f p = sceneToVolume(scenePos);
            if (auto* plane = dynamic_cast<PlaneSurface*>(surf.get())) {
                p += plane->normal({0, 0, 0}) * _zOff;
            } else if (_zOff != 0.0f &&
                       _zOffWorldDir != cv::Vec3f(0.0f, 0.0f, 0.0f)) {
                p += _zOffWorldDir * _zOff;
            }
            _viewerManager->broadcastLinkedCursor(this, p);
        } else {
            _viewerManager->broadcastLinkedCursor(this, std::nullopt);
        }
    }
    if (!_isPanning)
        return;
    const float dx = static_cast<float>(scenePos.x() - _lastPanSceneF.x());
    const float dy = static_cast<float>(scenePos.y() - _lastPanSceneF.y());
    _lastPanSceneF = scenePos;
    if (std::abs(dx) > 0.001f || std::abs(dy) > 0.001f) {
        if (!_panSmoothingInitialized) {
            _smoothedPanDx = dx;
            _smoothedPanDy = dy;
            _panSmoothingInitialized = true;
        } else {
            _smoothedPanDx = kPanSmoothingAlpha * dx + (1.0f - kPanSmoothingAlpha) * _smoothedPanDx;
            _smoothedPanDy = kPanSmoothingAlpha * dy + (1.0f - kPanSmoothingAlpha) * _smoothedPanDy;
        }
        panByF(_smoothedPanDx, _smoothedPanDy);
    }
}

void CChunkedVolumeViewer::onPanStart(Qt::MouseButton, Qt::KeyboardModifiers)
{
    _isPanning = true;
    _panSmoothingInitialized = false;
    _smoothedPanDx = 0.0f;
    _smoothedPanDy = 0.0f;
    _lastInteractivePreviewMs = -1;
    markInteractiveMotion(0.0);
    _lastPanSceneF = _view->mapToScene(_view->mapFromGlobal(QCursor::pos()));
}

void CChunkedVolumeViewer::onPanRelease(Qt::MouseButton, Qt::KeyboardModifiers)
{
    _isPanning = false;
    _interactivePreview = false;
    _panSmoothingInitialized = false;
    _smoothedPanDx = 0.0f;
    _smoothedPanDy = 0.0f;
    _lastInteractivePreviewMs = -1;
    if (_settleRenderTimer)
        _settleRenderTimer->stop();
    scheduleRender();
}

void CChunkedVolumeViewer::onVolumeClicked(QPointF scenePos, Qt::MouseButton button, Qt::KeyboardModifiers modifiers)
{
    auto surf = _surfWeak.lock();
    cv::Vec3f n(0, 0, 1);
    if (auto* plane = dynamic_cast<PlaneSurface*>(surf.get()))
        n = plane->normal({0, 0, 0});
    emit sendVolumeClicked(sceneToVolume(scenePos), n, surf.get(), button, modifiers);
}

void CChunkedVolumeViewer::onMousePress(QPointF scenePos, Qt::MouseButton button, Qt::KeyboardModifiers modifiers)
{
    _lastScenePos = scenePos;
    updateCursorCrosshair(scenePos);
    if (_bboxMode && _surfName == "segmentation" && button == Qt::LeftButton) {
        const cv::Vec2f sp = sceneToSurface(scenePos);
        _bboxStart = QPointF(sp[0], sp[1]);
        _activeBBoxSurfRect = QRectF(_bboxStart, QSizeF(0.0, 0.0));
        emit overlaysUpdated();
        return;
    }
    emit sendMousePressVolume(sceneToVolume(scenePos), {0, 0, 1}, button, modifiers, scenePos);
}

void CChunkedVolumeViewer::onMouseMove(QPointF scenePos, Qt::MouseButtons buttons, Qt::KeyboardModifiers modifiers)
{
    Q_UNUSED(modifiers);
    _lastScenePos = scenePos;
    updateCursorCrosshair(scenePos);
    if (_bboxMode && _activeBBoxSurfRect && (buttons & Qt::LeftButton)) {
        const cv::Vec2f sp = sceneToSurface(scenePos);
        _activeBBoxSurfRect = QRectF(_bboxStart, QPointF(sp[0], sp[1])).normalized();
        emit overlaysUpdated();
        return;
    }
    emit sendMouseMoveVolume(sceneToVolume(scenePos), buttons, modifiers, scenePos);
}

void CChunkedVolumeViewer::onMouseRelease(QPointF scenePos, Qt::MouseButton button, Qt::KeyboardModifiers modifiers)
{
    _lastScenePos = scenePos;
    updateCursorCrosshair(scenePos);
    if (_bboxMode && _surfName == "segmentation" && button == Qt::LeftButton &&
        _activeBBoxSurfRect) {
        const cv::Vec2f sp = sceneToSurface(scenePos);
        const QRectF surfRect = QRectF(_bboxStart, QPointF(sp[0], sp[1])).normalized();
        const int idx = static_cast<int>(_selections.size());
        const QColor color = QColor::fromHsv((idx * 53) % 360, 200, 255);
        _selections.push_back({surfRect, color});
        _activeBBoxSurfRect.reset();
        emit overlaysUpdated();
        return;
    }
    emit sendMouseReleaseVolume(sceneToVolume(scenePos), button, modifiers, scenePos);
}

void CChunkedVolumeViewer::onPathsChanged(const QList<ViewerOverlayControllerBase::PathPrimitive>& paths)
{
    _drawingPaths.clear();
    _drawingPaths.reserve(static_cast<std::size_t>(paths.size()));
    for (const auto& path : paths) {
        _drawingPaths.push_back(path);
    }
    emit overlaysUpdated();
}

void CChunkedVolumeViewer::onKeyPress(int key, Qt::KeyboardModifiers)
{
    constexpr float kPanPx = 64.0f;
    switch (key) {
        case Qt::Key_Left: panByF(kPanPx, 0); break;
        case Qt::Key_Right: panByF(-kPanPx, 0); break;
        case Qt::Key_Up: panByF(0, kPanPx); break;
        case Qt::Key_Down: panByF(0, -kPanPx); break;
        default: break;
    }
}

QPointF CChunkedVolumeViewer::surfaceToScene(float surfX, float surfY) const
{
    const float vpCx = static_cast<float>(_framebuffer.width()) * 0.5f;
    const float vpCy = static_cast<float>(_framebuffer.height()) * 0.5f;
    const qreal vx = (surfX - _camSurfX) * _camScale + vpCx;
    const qreal vy = (surfY - _camSurfY) * _camScale + vpCy;
    return _view->mapToScene(QPointF(vx, vy).toPoint());
}

cv::Vec2f CChunkedVolumeViewer::sceneToSurface(const QPointF& scenePos) const
{
    if (_framebuffer.isNull() || _camScale <= 0.0f)
        return {0, 0};
    const QPoint vp = _view->mapFromScene(scenePos);
    const float vpCx = static_cast<float>(_framebuffer.width()) * 0.5f;
    const float vpCy = static_cast<float>(_framebuffer.height()) * 0.5f;
    return {(static_cast<float>(vp.x()) - vpCx) / _camScale + _camSurfX,
            (static_cast<float>(vp.y()) - vpCy) / _camScale + _camSurfY};
}

QRectF CChunkedVolumeViewer::surfaceRectToSceneRect(const QRectF& surfRect) const
{
    const QPointF a = surfaceToScene(static_cast<float>(surfRect.left()),
                                     static_cast<float>(surfRect.top()));
    const QPointF b = surfaceToScene(static_cast<float>(surfRect.right()),
                                     static_cast<float>(surfRect.bottom()));
    return QRectF(a, b).normalized();
}

cv::Vec2f CChunkedVolumeViewer::sceneToSurfaceCoords(const QPointF& scenePos) const
{
    return sceneToSurface(scenePos);
}

QPointF CChunkedVolumeViewer::volumeToScene(const cv::Vec3f& volPoint)
{
    auto surf = _surfWeak.lock();
    if (!surf)
        return {};
    if (auto* plane = dynamic_cast<PlaneSurface*>(surf.get())) {
        const cv::Vec3f proj = plane->project(volPoint, 1.0, 1.0);
        return surfaceToScene(proj[0], proj[1]);
    }
    if (auto* quad = dynamic_cast<QuadSurface*>(surf.get())) {
        cv::Vec3f ptr = quad->pointer();
        auto* patchIndex = _viewerManager ? _viewerManager->surfacePatchIndex() : nullptr;
        if (quad->pointTo(ptr, volPoint, 4.0f, 100, patchIndex) < 0.0f)
            return {};
        const cv::Vec3f loc = quad->loc(ptr);
        return surfaceToScene(loc[0], loc[1]);
    }
    return {};
}

void CChunkedVolumeViewer::updateCursorCrosshair(const QPointF& scenePos)
{
    if (!_scene || !std::isfinite(scenePos.x()) || !std::isfinite(scenePos.y()))
        return;

    if (!_cursorCrosshair || !_cursorCrosshair->scene()) {
        QPainterPath path;
        constexpr qreal radius = 6.0;
        constexpr qreal arm = 14.0;
        constexpr qreal gap = 3.0;
        path.addEllipse(QPointF(0.0, 0.0), radius, radius);
        path.moveTo(-arm, 0.0);
        path.lineTo(-gap, 0.0);
        path.moveTo(gap, 0.0);
        path.lineTo(arm, 0.0);
        path.moveTo(0.0, -arm);
        path.lineTo(0.0, -gap);
        path.moveTo(0.0, gap);
        path.lineTo(0.0, arm);

        auto* marker = new QGraphicsPathItem(path);
        QPen pen(QColor(50, 255, 215), 2.0, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin);
        pen.setCosmetic(true);
        marker->setPen(pen);
        marker->setBrush(Qt::NoBrush);
        marker->setZValue(120.0);
        marker->setAcceptedMouseButtons(Qt::NoButton);
        _scene->addItem(marker);
        _cursorCrosshair = marker;
    }

    _cursorCrosshair->setPos(scenePos);
    _cursorCrosshair->show();
}

void CChunkedVolumeViewer::setLinkedCursorVolumePoint(const std::optional<cv::Vec3f>&)
{
    // The chunked viewer shows the cursor marker at its local mouse position.
    // Cross-view projection was unreliable for mixed surface/plane views and is
    // intentionally ignored here.
}

void CChunkedVolumeViewer::updateFocusMarker(POI* poi)
{
    if (!_scene)
        return;
    if (!poi && _state)
        poi = _state->poi("focus");
    if (!poi || !_surfWeak.lock()) {
        if (_focusMarker)
            _focusMarker->hide();
        return;
    }

    if (!_focusMarker || !_focusMarker->scene()) {
        auto* marker = new QGraphicsEllipseItem(-10.0, -10.0, 20.0, 20.0);
        QPen pen(QColor(50, 255, 215), 3.0, Qt::DashDotLine, Qt::RoundCap, Qt::RoundJoin);
        pen.setCosmetic(true);
        marker->setPen(pen);
        marker->setBrush(Qt::NoBrush);
        marker->setZValue(110.0);
        marker->setAcceptedMouseButtons(Qt::NoButton);
        _scene->addItem(marker);
        _focusMarker = marker;
    }

    const QPointF scenePos = volumeToScene(poi->p);
    if (!std::isfinite(scenePos.x()) || !std::isfinite(scenePos.y())) {
        _focusMarker->hide();
        return;
    }

    _focusMarker->setPos(scenePos);
    _focusMarker->show();
}

cv::Vec3f CChunkedVolumeViewer::sceneToVolume(const QPointF& scenePoint) const
{
    auto surf = _surfWeak.lock();
    if (!surf)
        return {0, 0, 0};
    const cv::Vec2f sp = sceneToSurface(scenePoint);
    return surf->coord({0, 0, 0}, {sp[0], sp[1], 0});
}

void CChunkedVolumeViewer::setOverlayGroup(const std::string& key, const std::vector<QGraphicsItem*>& items)
{
    clearOverlayGroup(key);
    _overlayGroups[key] = items;
    for (auto* item : items) {
        if (item && !item->scene())
            _scene->addItem(item);
    }
}

void CChunkedVolumeViewer::clearOverlayGroup(const std::string& key)
{
    auto it = _overlayGroups.find(key);
    if (it == _overlayGroups.end())
        return;
    for (auto* item : it->second)
        delete item;
    _overlayGroups.erase(it);
}

void CChunkedVolumeViewer::clearAllOverlayGroups()
{
    for (auto& [_, items] : _overlayGroups) {
        for (auto* item : items)
            delete item;
    }
    _overlayGroups.clear();
}

std::vector<std::pair<QRectF, QColor>> CChunkedVolumeViewer::selections() const
{
    std::vector<std::pair<QRectF, QColor>> out;
    out.reserve(_selections.size());
    for (const auto& selection : _selections) {
        out.emplace_back(surfaceRectToSceneRect(selection.surfRect), selection.color);
    }
    return out;
}

std::optional<QRectF> CChunkedVolumeViewer::activeBBoxSceneRect() const
{
    if (!_activeBBoxSurfRect)
        return std::nullopt;
    return surfaceRectToSceneRect(*_activeBBoxSurfRect);
}

void CChunkedVolumeViewer::setBBoxMode(bool enabled)
{
    _bboxMode = enabled;
    if (!enabled && _activeBBoxSurfRect) {
        _activeBBoxSurfRect.reset();
        emit overlaysUpdated();
    }
}

QuadSurface* CChunkedVolumeViewer::makeBBoxFilteredSurfaceFromSceneRect(const QRectF& sceneRect)
{
    if (_surfName != "segmentation")
        return nullptr;

    auto surf = _surfWeak.lock();
    auto* quad = dynamic_cast<QuadSurface*>(surf.get());
    if (!quad)
        return nullptr;

    const cv::Mat_<cv::Vec3f> src = quad->rawPoints();
    const int h = src.rows;
    const int w = src.cols;
    if (h <= 0 || w <= 0)
        return nullptr;

    const cv::Vec2f sp0 = sceneToSurface(sceneRect.topLeft());
    const cv::Vec2f sp1 = sceneToSurface(sceneRect.bottomRight());
    QRectF surfRect(QPointF(sp0[0], sp0[1]), QPointF(sp1[0], sp1[1]));
    surfRect = surfRect.normalized();

    const double cx = w * 0.5;
    const double cy = h * 0.5;
    const cv::Vec2f scale = quad->scale();
    if (scale[0] == 0.0f || scale[1] == 0.0f)
        return nullptr;

    const int i0 = std::max(0, static_cast<int>(std::floor(cx + surfRect.left() * scale[0])));
    const int i1 = std::min(w - 1, static_cast<int>(std::ceil(cx + surfRect.right() * scale[0])));
    const int j0 = std::max(0, static_cast<int>(std::floor(cy + surfRect.top() * scale[1])));
    const int j1 = std::min(h - 1, static_cast<int>(std::ceil(cy + surfRect.bottom() * scale[1])));
    if (i0 > i1 || j0 > j1)
        return nullptr;

    cv::Mat_<cv::Vec3f> cropped(j1 - j0 + 1, i1 - i0 + 1, cv::Vec3f(-1.0f, -1.0f, -1.0f));
    for (int j = j0; j <= j1; ++j) {
        for (int i = i0; i <= i1; ++i) {
            const cv::Vec3f& p = src(j, i);
            if (p[0] == -1.0f && p[1] == -1.0f && p[2] == -1.0f)
                continue;
            const double u = (i - cx) / scale[0];
            const double v = (j - cy) / scale[1];
            if (u >= surfRect.left() && u <= surfRect.right() &&
                v >= surfRect.top() && v <= surfRect.bottom()) {
                cropped(j - j0, i - i0) = p;
            }
        }
    }

    cv::Mat_<cv::Vec3f> cleaned = clean_surface_outliers(cropped);

    auto countValidInCol = [&](int c) {
        int count = 0;
        for (int r = 0; r < cleaned.rows; ++r) {
            if (cleaned(r, c)[0] != -1.0f)
                ++count;
        }
        return count;
    };
    auto countValidInRow = [&](int r) {
        int count = 0;
        for (int c = 0; c < cleaned.cols; ++c) {
            if (cleaned(r, c)[0] != -1.0f)
                ++count;
        }
        return count;
    };

    const int minValidCol = std::max(1, std::min(3, cleaned.rows));
    const int minValidRow = std::max(1, std::min(3, cleaned.cols));
    int left = 0;
    int right = cleaned.cols - 1;
    int top = 0;
    int bottom = cleaned.rows - 1;
    while (left <= right && countValidInCol(left) < minValidCol)
        ++left;
    while (right >= left && countValidInCol(right) < minValidCol)
        --right;
    while (top <= bottom && countValidInRow(top) < minValidRow)
        ++top;
    while (bottom >= top && countValidInRow(bottom) < minValidRow)
        --bottom;

    if (left > right || top > bottom) {
        left = cleaned.cols;
        right = -1;
        top = cleaned.rows;
        bottom = -1;
        for (int j = 0; j < cleaned.rows; ++j) {
            for (int i = 0; i < cleaned.cols; ++i) {
                if (cleaned(j, i)[0] != -1.0f) {
                    left = std::min(left, i);
                    right = std::max(right, i);
                    top = std::min(top, j);
                    bottom = std::max(bottom, j);
                }
            }
        }
        if (right < 0 || bottom < 0)
            return nullptr;
    }

    cv::Mat_<cv::Vec3f> finalPts(bottom - top + 1, right - left + 1,
                                  cv::Vec3f(-1.0f, -1.0f, -1.0f));
    for (int j = top; j <= bottom; ++j) {
        for (int i = left; i <= right; ++i) {
            finalPts(j - top, i - left) = cleaned(j, i);
        }
    }

    return new QuadSurface(finalPts, quad->scale());
}

void CChunkedVolumeViewer::clearSelections()
{
    _selections.clear();
    emit overlaysUpdated();
}

void CChunkedVolumeViewer::invalidateIntersect(const std::string&)
{
    clearIntersectionItems();
    _lastIntersectFp = {};
    _intersectionGeometryCache = {};
}

void CChunkedVolumeViewer::clearIntersectionItems()
{
    for (auto* item : _intersectionItems) {
        if (item && item->scene())
            _scene->removeItem(item);
        delete item;
    }
    _intersectionItems.clear();
    _intersectionItemsHaveCamera = false;
}

void CChunkedVolumeViewer::updateIntersectionPreviewTransform()
{
    if (_intersectionItems.empty() || !_intersectionItemsHaveCamera ||
        _intersectionItemsCamScale <= 0.0f || _camScale <= 0.0f ||
        _framebuffer.isNull()) {
        return;
    }

    const qreal vpCx = qreal(_framebuffer.width()) * 0.5;
    const qreal vpCy = qreal(_framebuffer.height()) * 0.5;
    const qreal scale = qreal(_camScale / _intersectionItemsCamScale);
    const qreal tx = (qreal(_intersectionItemsCamSurfX) - qreal(_camSurfX)) * qreal(_camScale)
                   + vpCx - vpCx * scale;
    const qreal ty = (qreal(_intersectionItemsCamSurfY) - qreal(_camSurfY)) * qreal(_camScale)
                   + vpCy - vpCy * scale;
    const QTransform transform(scale, 0.0, 0.0,
                               0.0, scale, 0.0,
                               tx, ty, 1.0);
    for (auto* item : _intersectionItems) {
        if (item)
            item->setTransform(transform);
    }
}

void CChunkedVolumeViewer::renderFlattenedIntersections(const std::shared_ptr<Surface>& surf)
{
    auto activeSeg = std::dynamic_pointer_cast<QuadSurface>(surf);
    if (!activeSeg || !_state || _state->surface("segmentation") != activeSeg) {
        invalidateIntersect();
        _lastIntersectFp = {};
        return;
    }

    auto* patchIndex = _viewerManager ? _viewerManager->surfacePatchIndex() : nullptr;
    if (!patchIndex || patchIndex->empty()) {
        invalidateIntersect();
        _lastIntersectFp = {};
        return;
    }

    struct PlaneEntry {
        std::shared_ptr<PlaneSurface> plane;
        QColor color;
    };
    const std::array<std::pair<const char*, QColor>, 3> kPlaneSpecs = {{
        {"seg xy", QColor(255, 140, 0)},
        {"seg xz", QColor(Qt::red)},
        {"seg yz", QColor(Qt::yellow)},
    }};
    std::vector<PlaneEntry> planes;
    planes.reserve(kPlaneSpecs.size());
    for (const auto& [name, color] : kPlaneSpecs) {
        if (!_intersectTgts.count(name))
            continue;
        if (auto p = std::dynamic_pointer_cast<PlaneSurface>(_state->surface(name))) {
            planes.push_back({std::move(p), color});
        }
    }
    if (planes.empty()) {
        invalidateIntersect();
        _lastIntersectFp = {};
        return;
    }

    auto mix = [](std::size_t s, std::size_t v) {
        return s ^ (v + 0x9e3779b9u + (s << 6) + (s >> 2));
    };
    auto hashVec = [&](std::size_t s, const cv::Vec3f& v) {
        for (int i = 0; i < 3; ++i)
            s = mix(s, std::hash<int>{}(int(std::lround(v[i] * 1000.0f))));
        return s;
    };

    std::size_t planesHash = 0;
    for (const auto& e : planes) {
        planesHash = hashVec(planesHash, e.plane->origin());
        planesHash = hashVec(planesHash, e.plane->normal({}, {}));
        planesHash = hashVec(planesHash, e.plane->basisX());
        planesHash = hashVec(planesHash, e.plane->basisY());
        planesHash = mix(planesHash, std::hash<uint32_t>{}(uint32_t(e.color.rgba())));
    }

    IntersectFingerprint fp;
    fp.flattenedPlanesHash = planesHash;
    fp.opacityQ = int(std::lround(_intersectionOpacity * 1000.0f));
    fp.thicknessQ = int(std::lround(_intersectionThickness * 1000.0f));
    fp.indexSamplingStride = patchIndex->samplingStride();
    fp.patchCount = patchIndex->patchCount();
    fp.surfaceCount = patchIndex->surfaceCount();
    fp.activeSegHash = std::hash<const void*>{}(activeSeg.get());
    fp.targetGenerationHash = std::hash<uint64_t>{}(patchIndex->generation(activeSeg));

    auto hashInt = [&](std::size_t s, int v) {
        return mix(s, std::hash<int>{}(v));
    };
    std::size_t cameraHash = 0;
    cameraHash = hashInt(cameraHash, int(std::lround(_camSurfX * 1000.0f)));
    cameraHash = hashInt(cameraHash, int(std::lround(_camSurfY * 1000.0f)));
    cameraHash = hashInt(cameraHash, int(std::lround(_camScale * 1000.0f)));
    cameraHash = hashInt(cameraHash, _framebuffer.width());
    cameraHash = hashInt(cameraHash, _framebuffer.height());
    if (_view) {
        const QTransform t = _view->transform();
        auto q = [](qreal v) { return int(std::lround(v * 1000.0)); };
        cameraHash = hashInt(cameraHash, q(t.m11()));
        cameraHash = hashInt(cameraHash, q(t.m12()));
        cameraHash = hashInt(cameraHash, q(t.m21()));
        cameraHash = hashInt(cameraHash, q(t.m22()));
        cameraHash = hashInt(cameraHash, q(t.dx()));
        cameraHash = hashInt(cameraHash, q(t.dy()));
    }
    fp.cameraHash = cameraHash;
    fp.valid = true;
    if (_lastIntersectFp == fp && !_intersectionItems.empty()) {
        updateIntersectionPreviewTransform();
        return;
    }

    clearIntersectionItems();
    _lastIntersectFp = fp;
    _intersectionGeometryCache = {};

    Rect3D allBounds{cv::Vec3f(0, 0, 0), cv::Vec3f(1, 1, 1)};
    if (_volume) {
        auto [w, h, d] = _volume->shape();
        allBounds.high = {static_cast<float>(w),
                          static_cast<float>(h),
                          static_cast<float>(d)};
    }

    const float clipTol = std::max(_intersectionThickness, 1e-4f);
    const float penWidth = std::max(_intersectionThickness,
                                    kActiveIntersectionMinWidthDelta);
    const float opacity = std::clamp(
        _intersectionOpacity * kActiveIntersectionOpacityScale, 0.0f, 1.0f);

    auto isFiniteScalar = [](double v) {
        uint64_t bits;
        std::memcpy(&bits, &v, sizeof(bits));
        return (bits & 0x7FF0000000000000ULL) != 0x7FF0000000000000ULL;
    };
    auto isFinitePoint = [&](const QPointF& p) {
        return isFiniteScalar(p.x()) && isFiniteScalar(p.y());
    };

    std::vector<QPainterPath> paths(planes.size());
    SurfacePatchIndex::TriangleQuery query;
    query.bounds = allBounds;
    query.surfaces.only = activeSeg;
    patchIndex->forEachTriangle(query,
        [&](const SurfacePatchIndex::TriangleCandidate& tri) {
            for (size_t idx = 0; idx < planes.size(); ++idx) {
                auto seg = SurfacePatchIndex::clipTriangleToPlane(
                    tri, *planes[idx].plane, clipTol);
                if (!seg)
                    continue;
                const cv::Vec3f a = activeSeg->loc(seg->surfaceParams[0]);
                const cv::Vec3f b = activeSeg->loc(seg->surfaceParams[1]);
                const QPointF pa = surfaceToScene(a[0], a[1]);
                const QPointF pb = surfaceToScene(b[0], b[1]);
                if (!isFinitePoint(pa) || !isFinitePoint(pb))
                    continue;
                paths[idx].moveTo(pa);
                paths[idx].lineTo(pb);
            }
        });

    _intersectionItems.reserve(paths.size());
    for (size_t idx = 0; idx < paths.size(); ++idx) {
        if (paths[idx].isEmpty())
            continue;
        QColor color = planes[idx].color;
        color.setAlphaF(opacity);
        auto* item = new QGraphicsPathItem(paths[idx]);
        QPen pen(color);
        pen.setWidthF(static_cast<qreal>(penWidth));
        pen.setCapStyle(Qt::RoundCap);
        pen.setJoinStyle(Qt::RoundJoin);
        pen.setCosmetic(true);
        item->setPen(pen);
        item->setBrush(Qt::NoBrush);
        item->setZValue(kActiveIntersectionZ);
        item->setAcceptedMouseButtons(Qt::NoButton);
        _scene->addItem(item);
        _intersectionItems.push_back(item);
    }

    _intersectionItemsCamSurfX = _camSurfX;
    _intersectionItemsCamSurfY = _camSurfY;
    _intersectionItemsCamScale = _camScale;
    _intersectionItemsHaveCamera = !_intersectionItems.empty();
    _view->viewport()->update();
}

void CChunkedVolumeViewer::renderIntersections()
{
    auto surf = _surfWeak.lock();
    auto* plane = dynamic_cast<PlaneSurface*>(surf.get());
    if (!surf || !_state || !_viewerManager || !_scene || !_view) {
        invalidateIntersect();
        _lastIntersectFp = {};
        return;
    }
    if (!plane) {
        renderFlattenedIntersections(surf);
        return;
    }

    auto* patchIndex = _viewerManager->surfacePatchIndex();
    if (!patchIndex || patchIndex->empty()) {
        invalidateIntersect();
        _lastIntersectFp = {};
        return;
    }

    std::unordered_set<SurfacePatchIndex::SurfacePtr> targets;
    auto addTarget = [&](const std::string& name) {
        if (auto quad = std::dynamic_pointer_cast<QuadSurface>(_state->surface(name)))
            targets.insert(std::move(quad));
    };
    for (const auto& name : _intersectTgts) {
        if (name == "visible_segmentation") {
            if (_highlightedSurfaceIds.empty()) {
                addTarget("segmentation");
            } else {
                for (const auto& id : _highlightedSurfaceIds)
                    addTarget(id);
            }
        } else {
            addTarget(name);
        }
    }
    if (targets.empty()) {
        invalidateIntersect();
        _lastIntersectFp = {};
        return;
    }

    QRectF sceneRect = _view->mapToScene(_view->viewport()->rect()).boundingRect();
    if (!sceneRect.isValid()) {
        invalidateIntersect();
        _lastIntersectFp = {};
        return;
    }

    float minX = std::numeric_limits<float>::max();
    float minY = std::numeric_limits<float>::max();
    float maxX = std::numeric_limits<float>::lowest();
    float maxY = std::numeric_limits<float>::lowest();
    const std::array<QPointF, 4> corners = {
        sceneRect.topLeft(), sceneRect.topRight(),
        sceneRect.bottomLeft(), sceneRect.bottomRight(),
    };
    for (const auto& c : corners) {
        cv::Vec2f sp = sceneToSurfaceCoords(c);
        minX = std::min(minX, sp[0]);
        minY = std::min(minY, sp[1]);
        maxX = std::max(maxX, sp[0]);
        maxY = std::max(maxY, sp[1]);
    }
    cv::Rect planeRoi{int(std::floor(minX)), int(std::floor(minY)),
                      std::max(1, int(std::ceil(maxX - minX))),
                      std::max(1, int(std::ceil(maxY - minY)))};

    auto activeSeg = std::dynamic_pointer_cast<QuadSurface>(_state->surface("segmentation"));

    IntersectFingerprint fp;
    fp.roiX = planeRoi.x;
    fp.roiY = planeRoi.y;
    fp.roiW = planeRoi.width;
    fp.roiH = planeRoi.height;
    auto quantizeVec = [](const cv::Vec3f& v) {
        return std::array<int, 3>{
            int(std::lround(v[0] * 1000.0f)),
            int(std::lround(v[1] * 1000.0f)),
            int(std::lround(v[2] * 1000.0f)),
        };
    };
    fp.planeOriginQ = quantizeVec(plane->origin());
    fp.planeNormalQ = quantizeVec(plane->normal({}, {}));
    fp.planeBasisXQ = quantizeVec(plane->basisX());
    fp.planeBasisYQ = quantizeVec(plane->basisY());
    fp.opacityQ = int(std::lround(_intersectionOpacity * 1000.0f));
    fp.thicknessQ = int(std::lround(_intersectionThickness * 1000.0f));
    fp.indexSamplingStride = patchIndex->samplingStride();
    fp.patchCount = patchIndex->patchCount();
    fp.surfaceCount = patchIndex->surfaceCount();
    size_t th = 0;
    size_t gh = 0;
    for (const auto& t : targets) {
        th ^= std::hash<const void*>{}(t.get()) + 0x9e3779b9u + (th << 6) + (th >> 2);
        gh ^= std::hash<const void*>{}(t.get()) ^
              (std::hash<uint64_t>{}(patchIndex->generation(t)) + 0x9e3779b9u);
    }
    fp.targetHash = th;
    fp.targetGenerationHash = gh;
    fp.activeSegHash = activeSeg ? std::hash<const void*>{}(activeSeg.get()) : 0;
    size_t hh = 0;
    for (const auto& id : _highlightedSurfaceIds)
        hh ^= std::hash<std::string>{}(id) + 0x9e3779b9u + (hh << 6) + (hh >> 2);
    fp.highlightedSurfaceHash = hh;
    fp.valid = true;
    if (_lastIntersectFp == fp && !_intersectionItems.empty()) {
        updateIntersectionPreviewTransform();
        return;
    }

    clearIntersectionItems();
    _lastIntersectFp = fp;

    auto rectContains = [](const cv::Rect& outer, const cv::Rect& inner) {
        return inner.x >= outer.x &&
               inner.y >= outer.y &&
               inner.x + inner.width <= outer.x + outer.width &&
               inner.y + inner.height <= outer.y + outer.height;
    };
    const bool geometryCacheValid =
        _intersectionGeometryCache.valid &&
        rectContains(_intersectionGeometryCache.roi, planeRoi) &&
        _intersectionGeometryCache.planeOriginQ == fp.planeOriginQ &&
        _intersectionGeometryCache.planeNormalQ == fp.planeNormalQ &&
        _intersectionGeometryCache.planeBasisXQ == fp.planeBasisXQ &&
        _intersectionGeometryCache.planeBasisYQ == fp.planeBasisYQ &&
        _intersectionGeometryCache.indexSamplingStride == fp.indexSamplingStride &&
        _intersectionGeometryCache.patchCount == fp.patchCount &&
        _intersectionGeometryCache.surfaceCount == fp.surfaceCount &&
        _intersectionGeometryCache.targetHash == fp.targetHash &&
        _intersectionGeometryCache.targetGenerationHash == fp.targetGenerationHash;

    if (!geometryCacheValid) {
        constexpr int kMinPanCachePadding = 256;
        cv::Rect cacheRoi = planeRoi;
        const int padX = std::max(kMinPanCachePadding, planeRoi.width / 2);
        const int padY = std::max(kMinPanCachePadding, planeRoi.height / 2);
        cacheRoi.x -= padX;
        cacheRoi.y -= padY;
        cacheRoi.width += padX * 2;
        cacheRoi.height += padY * 2;

        _intersectionGeometryCache = {};
        _intersectionGeometryCache.roi = cacheRoi;
        _intersectionGeometryCache.planeOriginQ = fp.planeOriginQ;
        _intersectionGeometryCache.planeNormalQ = fp.planeNormalQ;
        _intersectionGeometryCache.planeBasisXQ = fp.planeBasisXQ;
        _intersectionGeometryCache.planeBasisYQ = fp.planeBasisYQ;
        _intersectionGeometryCache.indexSamplingStride = fp.indexSamplingStride;
        _intersectionGeometryCache.patchCount = fp.patchCount;
        _intersectionGeometryCache.surfaceCount = fp.surfaceCount;
        _intersectionGeometryCache.targetHash = fp.targetHash;
        _intersectionGeometryCache.targetGenerationHash = fp.targetGenerationHash;
        _intersectionGeometryCache.intersections =
            patchIndex->computePlaneIntersections(*plane, cacheRoi, targets);
        _intersectionGeometryCache.valid = true;
    }

    const auto& intersections = _intersectionGeometryCache.intersections;
    if (intersections.empty())
        return;

    std::unordered_map<IntersectionStyle, QPainterPath, IntersectionStyleHash> groupedPaths;
    std::unordered_map<IntersectionStyle, QColor, IntersectionStyleHash> groupedColors;
    auto isFiniteScalar = [](double v) {
        uint64_t bits;
        std::memcpy(&bits, &v, sizeof(bits));
        return (bits & 0x7FF0000000000000ULL) != 0x7FF0000000000000ULL;
    };
    auto isFinitePoint = [&](const QPointF& p) {
        return isFiniteScalar(p.x()) && isFiniteScalar(p.y());
    };
    auto planeToScene = [&](const cv::Vec3f& volPoint) {
        cv::Vec3f proj = plane->project(volPoint, 1.0, 1.0);
        return surfaceToScene(proj[0], proj[1]);
    };

    for (const auto& [target, segments] : intersections) {
        if (!target || segments.empty())
            continue;

        QColor baseColor;
        int zValue = kIntersectionZ;
        float opacity = _intersectionOpacity;
        float penWidth = _intersectionThickness;
        if (target == activeSeg) {
            baseColor = activeSegmentationColorForView(_surfName);
            zValue = kActiveIntersectionZ;
            opacity *= kActiveIntersectionOpacityScale;
            penWidth = activeSegmentationIntersectionWidth(penWidth);
        } else if (_highlightedSurfaceIds.count(target->id)) {
            baseColor = QColor(0, 220, 255);
            zValue = kHighlightedIntersectionZ;
        } else {
            const auto& id = target->id;
            auto it = _surfaceColorAssignments.find(id);
            size_t idx;
            if (it != _surfaceColorAssignments.end()) {
                idx = it->second;
            } else if (_surfaceColorAssignments.size() < 500) {
                idx = _nextColorIndex++;
                _surfaceColorAssignments[id] = idx;
            } else {
                idx = std::hash<std::string>{}(id);
            }
            baseColor = QColor::fromRgba(kIntersectionPalette[idx % kIntersectionPalette.size()]);
        }
        baseColor.setAlphaF(std::clamp(opacity, 0.0f, 1.0f));
        if (baseColor.alpha() <= 0)
            continue;

        const IntersectionStyle style{
            baseColor.rgba(),
            zValue,
            int(std::lround(std::max(0.0f, penWidth) * 1000.0f)),
        };
        QPainterPath& path = groupedPaths[style];
        groupedColors[style] = baseColor;
        for (const auto& seg : segments) {
            QPointF a = planeToScene(seg.world[0]);
            QPointF b = planeToScene(seg.world[1]);
            if (!isFinitePoint(a) || !isFinitePoint(b))
                continue;
            path.moveTo(a);
            path.lineTo(b);
        }
    }

    _intersectionItems.reserve(groupedPaths.size());
    for (const auto& [style, path] : groupedPaths) {
        if (path.isEmpty())
            continue;
        auto* item = new QGraphicsPathItem(path);
        QPen pen(groupedColors[style]);
        pen.setWidthF(static_cast<qreal>(style.widthQ) / 1000.0);
        pen.setCapStyle(Qt::RoundCap);
        pen.setJoinStyle(Qt::RoundJoin);
        pen.setCosmetic(true);
        item->setPen(pen);
        item->setBrush(Qt::NoBrush);
        item->setZValue(style.z);
        item->setAcceptedMouseButtons(Qt::NoButton);
        _scene->addItem(item);
        _intersectionItems.push_back(item);
    }
    _intersectionItemsCamSurfX = _camSurfX;
    _intersectionItemsCamSurfY = _camSurfY;
    _intersectionItemsCamScale = _camScale;
    _intersectionItemsHaveCamera = !_intersectionItems.empty();

    _view->viewport()->update();
}

void CChunkedVolumeViewer::setHighlightedSurfaceIds(const std::vector<std::string>& ids)
{
    _highlightedSurfaceIds.clear();
    for (const auto& id : ids)
        _highlightedSurfaceIds.insert(id);
    renderIntersections();
}

const VolumeViewerBase::ActiveSegmentationHandle& CChunkedVolumeViewer::activeSegmentationHandle() const
{
    static ActiveSegmentationHandle handle;
    return handle;
}

const std::vector<ViewerOverlayControllerBase::PathPrimitive>& CChunkedVolumeViewer::drawingPaths() const
{
    return _drawingPaths;
}

const std::map<std::string, cv::Vec3b>& CChunkedVolumeViewer::surfaceOverlays() const
{
    return _surfaceOverlays;
}

void CChunkedVolumeViewer::updateStatusLabel()
{
    if (!_lbl)
        return;
    QString suffix;
    if ((_compositeSettings.enabled || _compositeSettings.planeEnabled) && streamingCompositeUnsupported()) {
        suffix = QString("  composite unsupported: %1").arg(QString::fromStdString(_compositeSettings.params.method));
    } else if (_compositeSettings.enabled || _compositeSettings.planeEnabled) {
        suffix = QString("  composite %1").arg(QString::fromStdString(_compositeSettings.params.method));
    }

    QString viewInfo;
    auto surf = _surfWeak.lock();
    if (auto* plane = dynamic_cast<PlaneSurface*>(surf.get())) {
        viewInfo = QString("  %1").arg(planeCoordinateText(*plane));
    } else if (dynamic_cast<QuadSurface*>(surf.get())) {
        viewInfo = QString("  normal offset %1").arg(_zOff, 0, 'f', 1);
        if (_state) {
            if (auto* poi = _state->poi("focus"))
                viewInfo += QString("  POI %1").arg(formatVec3(poi->p));
        }
    }

    _lbl->setText(QString("Streaming L%1  scale %2  %3x%4%5%6")
        .arg(_dsScaleIdx)
        .arg(_scale, 0, 'f', 2)
        .arg(_framebuffer.width())
        .arg(_framebuffer.height())
        .arg(viewInfo)
        .arg(suffix));
    _lbl->adjustSize();
}
