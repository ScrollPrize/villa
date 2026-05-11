#include "CChunkedVolumeViewer.hpp"

#include "CState.hpp"
#include "elements/ViewerStatsBar.hpp"
#include "VCSettings.hpp"
#include "ViewerManager.hpp"
#include "overlays/SegmentationOverlayController.hpp"
#include "vc/core/render/Colormaps.hpp"
#include "vc/core/render/PostProcess.hpp"
#include "render/ChunkCache.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/Geometry.hpp"
#include "vc/core/util/Logging.hpp"
#include "vc/core/util/Surface.hpp"

#include <QApplication>
#include <QCursor>
#include <QElapsedTimer>
#include <QEvent>
#include <QGraphicsEllipseItem>
#include <QGraphicsItem>
#include <QGraphicsPathItem>
#include <QGraphicsScene>
#include <QPainter>
#include <QPainterPath>
#include <QPointer>
#include <QSettings>
#include <QTimer>
#include <QTransform>
#include <QVBoxLayout>
#include <QtConcurrent/QtConcurrentRun>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <mutex>
#include <source_location>
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
constexpr qint64 kInteractivePreviewMinIntervalMs = 50;
constexpr float kPanSmoothingAlpha = 0.65f;
constexpr int kChunkPrefetchHaloPx = 128;
constexpr int kChunkPrefetchPriorityOffset = 1024;
constexpr int kNormalPrefetchSampleStridePx = 32;
constexpr int kSurfaceCellTileSize = 64;
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
constexpr float kApprovalPlaneIntersectionScale = 1.5f;

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

bool validSurfacePoint(const cv::Vec3f& p)
{
    return std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]) &&
           p[0] != -1.0f && p[1] != -1.0f && p[2] != -1.0f;
}

std::string profileCaller(const std::source_location& caller)
{
    return std::format("{}:{} {}", caller.file_name(), caller.line(), caller.function_name());
}

std::string profileVec3(const cv::Vec3f& v)
{
    return std::format("[{:.3f},{:.3f},{:.3f}]", v[0], v[1], v[2]);
}

std::string profileRect(const cv::Rect& r)
{
    return std::format("[x={},y={},w={},h={}]", r.x, r.y, r.width, r.height);
}

std::string profileRectF(const QRectF& r)
{
    return std::format("[x={:.2f},y={:.2f},w={:.2f},h={:.2f}]",
                       r.x(), r.y(), r.width(), r.height());
}

class ProfileScope {
public:
    ProfileScope(const char* event,
                 const char* reason,
                 const std::source_location& caller)
        : event_(event ? event : "")
        , enabled_(ProfileLoggingEnabled())
    {
        if (!enabled_)
            return;
        reason_ = reason ? reason : "";
        caller_ = profileCaller(caller);
        timer_.start();
        Logger()->info("[vc3d-profile] {} begin reason='{}' caller='{}'",
                       event_, reason_, caller_);
    }

    ~ProfileScope()
    {
        finish();
    }

    bool enabled() const
    {
        return enabled_;
    }

    void setDetails(const char* details)
    {
        if (!enabled_)
            return;
        details_ = details ? details : "";
    }

    void setDetails(std::string details)
    {
        if (!enabled_)
            return;
        details_ = std::move(details);
    }

    void finish()
    {
        if (!enabled_ || finished_)
            return;
        finished_ = true;
        Logger()->info("[vc3d-profile] {} end elapsed_ms={} reason='{}' caller='{}'{}{}",
                       event_, timer_.elapsed(), reason_, caller_,
                       details_.empty() ? "" : " ", details_);
    }

private:
    const char* event_ = "";
    std::string reason_;
    std::string caller_;
    bool enabled_ = false;
    bool finished_ = false;
    QElapsedTimer timer_;
    std::string details_;
};

void applyPerPixelNormalOffset(cv::Mat_<cv::Vec3f>& coords,
                               const cv::Mat_<cv::Vec3f>& normals,
                               float zOff)
{
    if (zOff == 0.0f || coords.empty() || normals.empty())
        return;

    const int h = std::min(coords.rows, normals.rows);
    const int w = std::min(coords.cols, normals.cols);
    for (int y = 0; y < h; ++y) {
        auto* coordRow = coords.ptr<cv::Vec3f>(y);
        const auto* normalRow = normals.ptr<cv::Vec3f>(y);
        for (int x = 0; x < w; ++x) {
            cv::Vec3f& p = coordRow[x];
            const cv::Vec3f& n = normalRow[x];
            if (!validSurfacePoint(p) ||
                !std::isfinite(n[0]) || !std::isfinite(n[1]) || !std::isfinite(n[2])) {
                continue;
            }
            p += n * zOff;
        }
    }
}

std::uint64_t surfaceTileKey(int tx, int ty)
{
    return (std::uint64_t(std::uint32_t(ty)) << 32) | std::uint32_t(tx);
}

std::uint64_t surfaceCellTileKey(int col, int row)
{
    const int tx = col >= 0 ? col / kSurfaceCellTileSize : (col - kSurfaceCellTileSize + 1) / kSurfaceCellTileSize;
    const int ty = row >= 0 ? row / kSurfaceCellTileSize : (row - kSurfaceCellTileSize + 1) / kSurfaceCellTileSize;
    return surfaceTileKey(tx, ty);
}

void addChunkBox(std::unordered_set<vc::render::ChunkKey, vc::render::ChunkKeyHash>& keys,
                 vc::render::IChunkedArray& array,
                 int level,
                 const cv::Vec3f& lo0,
                 const cv::Vec3f& hi0,
                 vc::Sampling sampling)
{
    const auto shape = array.shape(level);
    const auto chunkShape = array.chunkShape(level);
    if (shape[0] <= 0 || shape[1] <= 0 || shape[2] <= 0 ||
        chunkShape[0] <= 0 || chunkShape[1] <= 0 || chunkShape[2] <= 0) {
        return;
    }

    const auto transform = array.levelTransform(level);
    auto toLevel = [&transform](const cv::Vec3f& p) {
        return cv::Vec3f(
            float(double(p[0]) * transform.scaleFromLevel0[0] + transform.offsetFromLevel0[0]),
            float(double(p[1]) * transform.scaleFromLevel0[1] + transform.offsetFromLevel0[1]),
            float(double(p[2]) * transform.scaleFromLevel0[2] + transform.offsetFromLevel0[2]));
    };
    const cv::Vec3f lo = toLevel(lo0);
    const cv::Vec3f hi = toLevel(hi0);
    const float pad = sampling == vc::Sampling::Nearest ? 0.5f : 1.0f;

    const int ix0 = std::clamp(int(std::floor(std::min(lo[0], hi[0]) - pad)), 0, shape[2] - 1);
    const int iy0 = std::clamp(int(std::floor(std::min(lo[1], hi[1]) - pad)), 0, shape[1] - 1);
    const int iz0 = std::clamp(int(std::floor(std::min(lo[2], hi[2]) - pad)), 0, shape[0] - 1);
    const int ix1 = std::clamp(int(std::ceil(std::max(lo[0], hi[0]) + pad)), 0, shape[2] - 1);
    const int iy1 = std::clamp(int(std::ceil(std::max(lo[1], hi[1]) + pad)), 0, shape[1] - 1);
    const int iz1 = std::clamp(int(std::ceil(std::max(lo[2], hi[2]) + pad)), 0, shape[0] - 1);
    if (ix1 < ix0 || iy1 < iy0 || iz1 < iz0)
        return;

    for (int cz = iz0 / chunkShape[0]; cz <= iz1 / chunkShape[0]; ++cz) {
        for (int cy = iy0 / chunkShape[1]; cy <= iy1 / chunkShape[1]; ++cy) {
            for (int cx = ix0 / chunkShape[2]; cx <= ix1 / chunkShape[2]; ++cx)
                keys.insert({level, cz, cy, cx});
        }
    }
}

std::vector<vc::render::ChunkKey> collectSurfaceCellChunkKeys(
    vc::render::IChunkedArray& array,
    QuadSurface& surface,
    const cv::Rect& cellRect,
    int level,
    vc::Sampling sampling)
{
    std::vector<vc::render::ChunkKey> result;
    if (level < 0 || level >= array.numLevels())
        return result;

    const cv::Mat_<cv::Vec3f>* points = surface.rawPointsPtr();
    if (!points || points->empty() || points->cols < 2 || points->rows < 2)
        return result;

    const cv::Rect bounds(0, 0, points->cols - 1, points->rows - 1);
    const cv::Rect rect = cellRect & bounds;
    if (rect.empty())
        return result;

    std::unordered_set<vc::render::ChunkKey, vc::render::ChunkKeyHash> keys;
    keys.reserve(std::size_t(rect.area() / 4 + 16));
    for (int y = rect.y; y < rect.y + rect.height; ++y) {
        const cv::Vec3f* row0 = points->ptr<cv::Vec3f>(y);
        const cv::Vec3f* row1 = points->ptr<cv::Vec3f>(y + 1);
        for (int x = rect.x; x < rect.x + rect.width; ++x) {
            const cv::Vec3f p00 = row0[x];
            const cv::Vec3f p10 = row0[x + 1];
            const cv::Vec3f p01 = row1[x];
            const cv::Vec3f p11 = row1[x + 1];
            if (!validSurfacePoint(p00) || !validSurfacePoint(p10) ||
                !validSurfacePoint(p01) || !validSurfacePoint(p11)) {
                continue;
            }
            cv::Vec3f lo(std::min({p00[0], p10[0], p01[0], p11[0]}),
                         std::min({p00[1], p10[1], p01[1], p11[1]}),
                         std::min({p00[2], p10[2], p01[2], p11[2]}));
            cv::Vec3f hi(std::max({p00[0], p10[0], p01[0], p11[0]}),
                         std::max({p00[1], p10[1], p01[1], p11[1]}),
                         std::max({p00[2], p10[2], p01[2], p11[2]}));
            addChunkBox(keys, array, level, lo, hi, sampling);
        }
    }

    result.reserve(keys.size());
    for (const auto& key : keys)
        result.push_back(key);
    return result;
}

bool rectContains(const cv::Rect& outer, const cv::Rect& inner)
{
    return inner.x >= outer.x &&
           inner.y >= outer.y &&
           inner.x + inner.width <= outer.x + outer.width &&
           inner.y + inner.height <= outer.y + outer.height;
}

QString formatVec3(const cv::Vec3f& v)
{
    return QString("(%1, %2, %3)")
        .arg(v[0], 0, 'f', 1)
        .arg(v[1], 0, 'f', 1)
        .arg(v[2], 0, 'f', 1);
}

QString formatWholeVolumePosition(const cv::Vec3f& v)
{
    return QString("[%1, %2, %3]")
        .arg(v[0], 0, 'f', 0)
        .arg(v[1], 0, 'f', 0)
        .arg(v[2], 0, 'f', 0);
}

QString formatByteSize(std::size_t bytes)
{
    constexpr double kKiB = 1024.0;
    constexpr double kMiB = kKiB * 1024.0;
    constexpr double kGiB = kMiB * 1024.0;
    const double value = static_cast<double>(bytes);
    if (value >= kGiB)
        return QString("%1 GB").arg(value / kGiB, 0, 'f', 2);
    if (value >= kMiB)
        return QString("%1 MB").arg(value / kMiB, 0, 'f', 1);
    if (value >= kKiB)
        return QString("%1 KB").arg(value / kKiB, 0, 'f', 1);
    return QString("%1 B").arg(bytes);
}

QString formatGigabytes(std::size_t bytes)
{
    constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
    return QString("%1").arg(static_cast<double>(bytes) / kGiB, 0, 'f', 1);
}

QString formatMegabytesPerSecond(double bytesPerSecond)
{
    constexpr double kMiB = 1024.0 * 1024.0;
    return QString("%1 MB/s").arg(std::max(0.0, bytesPerSecond) / kMiB, 0, 'f', 1);
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

float scaleForCoarsestSegmentationRenderLevel(int numLevels)
{
    const int coarsestLevel = std::max(0, numLevels - 1);
    const float dsScale = static_cast<float>(std::uint64_t{1} << coarsestLevel);
    return std::clamp(0.75f / (dsScale * kSegmentationResolutionLodZoomBias), kMinScale, kMaxScale);
}

std::filesystem::path remoteCacheRootForState(const CState* state)
{
    // Suggestion order: per-volpkg setting first (so projects with an
    // explicit cache stay co-located when no host mount is present), then
    // the user's persisted setting. remoteCachePath() ignores both when
    // /volpkgs or /ephemeral is mounted.
    QString suggestion;
    if (state && state->vpkg()) {
        suggestion = QString::fromStdString(state->vpkg()->remoteCacheRootOrEmpty()).trimmed();
    }
    if (suggestion.isEmpty()) {
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        suggestion =
            settings.value(vc3d::settings::viewer::REMOTE_CACHE_DIR).toString();
    }
    return vc3d::remoteCachePath(suggestion).toStdString();
}

std::shared_ptr<vc::render::ChunkCache> makeChunkCacheForVolume(const std::shared_ptr<Volume>& volume,
                                                                std::size_t decodedByteCapacity,
                                                                const CState* state)
{
    if (!volume)
        return nullptr;

    vc::render::ChunkCache::Options options;
    options.decodedByteCapacity = decodedByteCapacity > 0
        ? decodedByteCapacity
        : streamingCacheCapacityBytes(nullptr);
    options.maxConcurrentReads = 16;
    if (volume->isRemote()) {
        const auto cacheRoot = remoteCacheRootForState(state);
        options.persistentCachePath = cacheRoot / stableHexHash(normalizedVolumeCacheIdentity(volume));
    }

    return volume->createChunkCache(std::move(options));
}

std::shared_ptr<vc::render::ChunkCache> sharedChunkCacheForVolume(const std::shared_ptr<Volume>& volume,
                                                                  std::size_t decodedByteCapacity,
                                                                  const CState* state)
{
    if (!volume)
        return nullptr;

    const std::size_t capacity = decodedByteCapacity > 0
        ? decodedByteCapacity
        : streamingCacheCapacityBytes(nullptr);
    const std::string key = normalizedVolumeCacheIdentity(volume) +
                            "|decoded=" + std::to_string(capacity) +
                            "|cache=" + (volume->isRemote() ? remoteCacheRootForState(state).string() : std::string{});

    static std::mutex cacheMutex;
    static std::unordered_map<std::string, std::weak_ptr<vc::render::ChunkCache>> caches;

    {
        std::lock_guard<std::mutex> lock(cacheMutex);
        auto it = caches.find(key);
        if (it != caches.end()) {
            if (auto cache = it->second.lock())
                return cache;
            caches.erase(it);
        }
    }

    auto cache = makeChunkCacheForVolume(volume, capacity, state);
    if (!cache)
        return nullptr;

    std::lock_guard<std::mutex> lock(cacheMutex);
    auto& slot = caches[key];
    if (auto existing = slot.lock())
        return existing;
    slot = cache;
    return cache;
}

} // namespace

struct CChunkedVolumeViewer::GeneratedSurfaceCache {
    std::mutex mutex;
    bool valid = false;
    Surface* surface = nullptr;
    int fbW = 0;
    int fbH = 0;
    float scale = 0.0f;
    cv::Vec3f offset{0, 0, 0};
    float zOff = 0.0f;
    cv::Vec3f zOffWorldDir{0, 0, 0};
    cv::Mat_<cv::Vec3f> coords;
    cv::Mat_<cv::Vec3f> normals;
};

CChunkedVolumeViewer::CChunkedVolumeViewer(CState* state, ViewerManager* manager, QWidget* parent)
    : QWidget(parent)
    , _state(state)
    , _viewerManager(manager)
    , _genSurfaceCache(std::make_shared<GeneratedSurfaceCache>())
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
    connect(_view, &CVolumeViewerView::sendMouseDoubleClick, this,
            [this](QPointF scenePos, Qt::MouseButton button, Qt::KeyboardModifiers modifiers) {
                const auto cursorPos = cursorVolumePosition(scenePos);
                cv::Vec3f volumePos;
                if (cursorPos) {
                    volumePos = *cursorPos;
                } else {
                    volumePos = sceneToVolume(scenePos);
                }
                emit sendMouseDoubleClickVolume(volumePos, button, modifiers);
            });
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
        const std::string reason = ProfileLoggingEnabled()
            ? std::format("render timer fired; scheduledReason='{}'; scheduledCaller='{}'",
                          _pendingRenderReason, _pendingRenderCaller)
            : std::string("render timer fired");
        submitRender(reason.c_str());
        updateStatusLabel();
    });

    _settleRenderTimer = new QTimer(this);
    _settleRenderTimer->setSingleShot(true);
    _settleRenderTimer->setInterval(kInteractionSettleMs);
    connect(_settleRenderTimer, &QTimer::timeout, this, [this]() {
        if (!_isPanning) {
            _interactivePreview = false;
            scheduleRender("interaction settled");
        }
    });

    _intersectionRenderTimer = new QTimer(this);
    _intersectionRenderTimer->setSingleShot(true);
    _intersectionRenderTimer->setInterval(50);
    connect(_intersectionRenderTimer, &QTimer::timeout, this, [this]() {
        const std::string reason = ProfileLoggingEnabled()
            ? std::format("intersection timer fired; scheduledReason='{}'; scheduledCaller='{}'",
                          _pendingIntersectionReason, _pendingIntersectionCaller)
            : std::string("intersection timer fired");
        renderIntersections(reason.c_str());
        emit overlaysUpdated();
    });

    _resizeRenderTimer = new QTimer(this);
    _resizeRenderTimer->setSingleShot(true);
    _resizeRenderTimer->setInterval(kResizeSettleMs);
    connect(_resizeRenderTimer, &QTimer::timeout, this, [this]() {
        scheduleRender("resize settled");
        emit overlaysUpdated();
    });

    _statusTimer = new QTimer(this);
    _statusTimer->setInterval(500);
    connect(_statusTimer, &QTimer::timeout, this, [this]() {
        updateStatusLabel();
    });
    _statusTimer->start();

    reloadPerfSettings();

    auto* layout = new QVBoxLayout;
    layout->addWidget(_view);
    setLayout(layout);

    _statsBar = new ViewerStatsBar(this);
    _statsBar->move(10, 5);
}

CChunkedVolumeViewer::~CChunkedVolumeViewer()
{
    quiesceForClose();
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

bool CChunkedVolumeViewer::eventFilter(QObject* watched, QEvent* event)
{
    if (event && event->type() == QEvent::Close && watched == parentWidget()) {
        quiesceForClose();
    }
    return QWidget::eventFilter(watched, event);
}

void CChunkedVolumeViewer::quiesceForClose()
{
    if (_closing) {
        return;
    }
    _closing = true;

    if (_viewerManager) {
        _viewerManager->unregisterViewer(this);
    }

    if (_renderTimer)
        _renderTimer->stop();
    if (_settleRenderTimer)
        _settleRenderTimer->stop();
    if (_intersectionRenderTimer)
        _intersectionRenderTimer->stop();
    if (_resizeRenderTimer)
        _resizeRenderTimer->stop();
    if (_statusTimer)
        _statusTimer->stop();

    _renderPending = false;
    _renderPendingAfterWorker = false;
    ++_renderSerial;

    if (_chunkCbId != 0 && _chunkArray) {
        _chunkArray->removeChunkReadyListener(_chunkCbId);
        _chunkCbId = 0;
    }
    if (_overlayChunkCbId != 0 && _overlayChunkArray) {
        _overlayChunkArray->removeChunkReadyListener(_overlayChunkCbId);
        _overlayChunkCbId = 0;
    }
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
    if (_closing) {
        return;
    }
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
        _chunkArray = sharedChunkCacheForVolume(_volume, streamingCacheCapacityBytes(_state), _state);
    } catch (const std::exception& e) {
        if (_statsBar)
            _statsBar->setItems({QString("Streaming unavailable: %1").arg(e.what())});
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
            if (!volume || guard->_volume != volume || guard->_closing)
                return;
            if (guard->_interactivePreview) {
                if (guard->_settleRenderTimer)
                    guard->_settleRenderTimer->start(kChunkReadyActiveDelayMs);
                return;
            }
            guard->scheduleRender("chunk ready");
        }, Qt::QueuedConnection);
    });
}

void CChunkedVolumeViewer::OnVolumeChanged(std::shared_ptr<Volume> vol)
{
    if (_closing) {
        return;
    }
    const bool hadVolume = static_cast<bool>(_volume);
    invalidateIntersect();
    if (_surfWeak.lock() == _defaultSurface) {
        _surfWeak.reset();
        _defaultSurface.reset();
    }
    _genCacheDirty = true;
    if (_genSurfaceCache) {
        std::lock_guard lock(_genSurfaceCache->mutex);
        _genSurfaceCache->valid = false;
        _genSurfaceCache->coords.release();
        _genSurfaceCache->normals.release();
    }
    _zOffWorldDir = {0, 0, 0};
    _surfaceChunkPrefetchCache = {};
    _stableFramebufferValid = false;
    if (_cursorCrosshair)
        _cursorCrosshair->hide();
    if (_focusMarker)
        _focusMarker->hide();

    _volume = std::move(vol);
    rebuildChunkArray();
    ensureDefaultSurface();
    if (_volume && isAxisAlignedView() && !hadVolume) {
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
    scheduleRender("volume changed");
    renderIntersections("volume changed");
    updateStatusLabel();
}

void CChunkedVolumeViewer::invalidateVis()
{
    if (_closing) {
        return;
    }
    _genCacheDirty = true;
    _stableFramebufferValid = false;
    _surfaceChunkPrefetchCache = {};
}

void CChunkedVolumeViewer::invalidateVisRegion(const std::string& name, const cv::Rect& changedCells)
{
    if (changedCells.empty() || name != _surfName || _surfName != "segmentation") {
        invalidateVis();
        return;
    }

    auto surf = _surfWeak.lock();
    auto* quad = dynamic_cast<QuadSurface*>(surf.get());
    if (!quad) {
        invalidateVis();
        return;
    }

    _genCacheDirty = true;
    _stableFramebufferValid = false;

    auto& cache = _surfaceChunkPrefetchCache;
    if (!cache.valid || cache.surface != quad) {
        return;
    }

    const int tx0 = changedCells.x / kSurfaceCellTileSize;
    const int ty0 = changedCells.y / kSurfaceCellTileSize;
    const int tx1 = (changedCells.x + changedCells.width - 1) / kSurfaceCellTileSize;
    const int ty1 = (changedCells.y + changedCells.height - 1) / kSurfaceCellTileSize;
    for (int ty = ty0; ty <= ty1; ++ty) {
        for (int tx = tx0; tx <= tx1; ++tx) {
            cache.tileKeys.erase(surfaceTileKey(tx, ty));
        }
    }
}

void CChunkedVolumeViewer::onSurfaceChanged(const std::string& name,
                                            const std::shared_ptr<Surface>& surf,
                                            bool isEditUpdate)
{
    if (_closing) {
        return;
    }
    const bool isCurrentSurface = (_surfName == name);
    const auto previousSurface = _surfWeak.lock();
    const bool isSameCurrentSurface = isCurrentSurface && previousSurface && previousSurface == surf;
    const bool isIntersectionTarget =
        _intersectTgts.count(name) != 0 ||
        (_intersectTgts.count("visible_segmentation") != 0 &&
         (name == "segmentation" || _highlightedSurfaceIds.count(name) != 0));

    if (!isCurrentSurface) {
        if (isIntersectionTarget) {
            if (!isEditUpdate) {
                invalidateIntersect(name);
                renderIntersections("intersection target surface changed");
            } else {
                _lastIntersectFp = {};
                if (_deferSegmentationIntersections && name == "segmentation" &&
                    dynamic_cast<PlaneSurface*>(currentSurface())) {
                    _intersectionGeometryCache = {};
                    _deferredSegmentationIntersectionsDirty = true;
                    return;
                }
                scheduleIntersectionRender("intersection target surface changed");
            }
        }
        return;
    }

    std::optional<cv::Vec3f> preservedViewCenter;
    if (!isEditUpdate && !_resetViewOnSurfaceChange && _surfName == "segmentation" &&
        surf && previousSurface && _view && !_framebuffer.isNull()) {
        const QPointF sceneCenter(static_cast<qreal>(_framebuffer.width()) * 0.5,
                                  static_cast<qreal>(_framebuffer.height()) * 0.5);
        preservedViewCenter = cursorVolumePosition(sceneCenter);
    }

    _surfWeak = surf;
    if (isSameCurrentSurface && isEditUpdate) {
        _genCacheDirty = true;
        _zOffWorldDir = {0, 0, 0};
        _stableFramebufferValid = false;
        updateContentBounds();
        updateFocusMarker();
        if (_suppressNextSurfaceEditRender) {
            _suppressNextSurfaceEditRender = false;
        } else {
            scheduleRender("current surface edit update");
        }
        scheduleIntersectionRender("current surface edit update");
        return;
    }

    _genCacheDirty = true;
    _surfaceChunkPrefetchCache = {};
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
    const bool isSegmentationQuadSurface =
        _surfName == "segmentation" && dynamic_cast<QuadSurface*>(surf.get());
    if (!isEditUpdate && isSegmentationQuadSurface && !_initializedFirstSegmentationSurface) {
        _surfacePtrX = 0.0f;
        _surfacePtrY = 0.0f;
        _zOff = 0.0f;
        const int n = _chunkArray ? _chunkArray->numLevels()
                                  : (_volume ? static_cast<int>(_volume->numScales()) : 1);
        _scale = scaleForCoarsestSegmentationRenderLevel(n);
        recalcPyramidLevel();
        _initializedFirstSegmentationSurface = true;
    } else if (!isEditUpdate && _resetViewOnSurfaceChange && isSegmentationQuadSurface) {
        _surfacePtrX = 0.0f;
        _surfacePtrY = 0.0f;
        _zOff = 0.0f;
        const int n = _chunkArray ? _chunkArray->numLevels()
                                  : (_volume ? static_cast<int>(_volume->numScales()) : 1);
        _scale = scaleForSurfaceRenderStartLevel(kInitialSegmentationSurfaceLevel, n);
        recalcPyramidLevel();
    } else if (preservedViewCenter) {
        if (auto* plane = dynamic_cast<PlaneSurface*>(surf.get())) {
            const cv::Vec3f projected = plane->project(*preservedViewCenter, 1.0, 1.0);
            if (std::isfinite(projected[0]) && std::isfinite(projected[1])) {
                _surfacePtrX = projected[0];
                _surfacePtrY = projected[1];
            }
        } else if (auto* quad = dynamic_cast<QuadSurface*>(surf.get())) {
            try {
                cv::Vec3f ptr = quad->pointer();
                auto* patchIndex = _viewerManager ? _viewerManager->surfacePatchIndex() : nullptr;
                if (quad->pointTo(ptr, *preservedViewCenter, 4.0f, 100, patchIndex) >= 0.0f) {
                    const cv::Vec3f loc = quad->loc(ptr);
                    if (std::isfinite(loc[0]) && std::isfinite(loc[1])) {
                        _surfacePtrX = loc[0];
                        _surfacePtrY = loc[1];
                    }
                }
            } catch (const std::exception& e) {
                Logger()->warn("Skipping preserved view projection for surface '{}': {}", quad->id, e.what());
            }
        }
    }
    updateFocusMarker();
    scheduleRender("current surface changed");
    renderIntersections("current surface changed");
}

void CChunkedVolumeViewer::onSurfaceWillBeDeleted(const std::string&, const std::shared_ptr<Surface>& surf)
{
    if (_closing) {
        return;
    }
    auto current = _surfWeak.lock();
    if (current && current == surf)
        _surfWeak.reset();
}

void CChunkedVolumeViewer::onVolumeClosing()
{
    if (_closing) {
        return;
    }
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
    if (_closing) {
        return;
    }
    if (name != "focus" || !poi)
        return;

    auto surf = _surfWeak.lock();
    const bool isPlaneSurface = dynamic_cast<PlaneSurface*>(surf.get()) != nullptr;
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
    scheduleRender("focus POI changed");
    if (!isPlaneSurface || !poi->suppressTransientPlaneIntersections)
        renderIntersections("focus POI changed");
}

void CChunkedVolumeViewer::ensureDefaultSurface()
{
    if (_surfWeak.lock() || !_volume || !isAxisAlignedView())
        return;
    const auto shape = _volume->shapeXyz();
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

    const auto [w, h, d] = _volume->shapeXyz();
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

void CChunkedVolumeViewer::scheduleRender(const char* reason, std::source_location caller)
{
    if (_closing) {
        return;
    }
    if (ProfileLoggingEnabled()) {
        _pendingRenderReason = reason ? reason : "";
        _pendingRenderCaller = profileCaller(caller);
    }
    ProfileScope profile("scheduleRender", reason, caller);
    if (profile.enabled()) {
        profile.setDetails(std::format(
            "surf='{}' force=false pending={} interactive={} scale={:.4f} zOff={:.3f}",
            _surfName, _renderPending, _interactivePreview, _scale, _zOff));
    }
    syncCameraTransform();
    _renderPending = true;
    if (_interactivePreview) {
        if (_settleRenderTimer)
            _settleRenderTimer->start();
        profile.setDetails("action=settle_timer");
        return;
    }
    if (_renderTimer && !_renderTimer->isActive()) {
        _renderTimer->start();
        profile.setDetails("action=render_timer_start");
    } else {
        profile.setDetails("action=already_scheduled");
    }
}

void CChunkedVolumeViewer::requestRender(const char* reason, std::source_location caller)
{
    if (reason && std::string_view(reason) == "push/pull active viewer refresh") {
        _suppressNextSurfaceEditRender = true;
    }
    scheduleRender(reason, caller);
}

void CChunkedVolumeViewer::scheduleIntersectionRender(const char* reason, std::source_location caller)
{
    if (_closing) {
        return;
    }
    if (ProfileLoggingEnabled()) {
        _pendingIntersectionReason = reason ? reason : "";
        _pendingIntersectionCaller = profileCaller(caller);
    }
    ProfileScope profile("scheduleIntersectionRender", reason, caller);
    if (profile.enabled()) {
        profile.setDetails(std::format("surf='{}' targets={}", _surfName, _intersectTgts.size()));
    }
    if (_deferSegmentationIntersections && dynamic_cast<PlaneSurface*>(currentSurface())) {
        _deferredSegmentationIntersectionsDirty = true;
        profile.setDetails("action=deferred_segmentation_edit");
        return;
    }
    if (_intersectionRenderTimer && !_intersectionRenderTimer->isActive()) {
        _intersectionRenderTimer->start();
        profile.setDetails("action=intersection_timer_start");
    } else {
        profile.setDetails("action=already_scheduled");
    }
}

void CChunkedVolumeViewer::setSegmentationIntersectionDeferral(bool active)
{
    if (_deferSegmentationIntersections == active) {
        return;
    }

    _deferSegmentationIntersections = active;
    if (!active && _deferredSegmentationIntersectionsDirty) {
        _deferredSegmentationIntersectionsDirty = false;
        scheduleIntersectionRender("deferred segmentation intersection refresh");
    }
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
    ProfileScope profile("renderInteractiveAxisAlignedSlicePreview",
                         "interactive preview requested",
                         std::source_location::current());
    if (profile.enabled()) {
        profile.setDetails(std::format(
            "surf='{}' framebuffer={}x{} scale={:.4f} zOff={:.3f}",
            _surfName, _framebuffer.width(), _framebuffer.height(), _scale, _zOff));
    }
    if (!_chunkArray || !_volume || _framebuffer.isNull()) {
        profile.setDetails("action=skip missing_input");
        return false;
    }
    if (_overlayVolume || _compositeSettings.enabled || _compositeSettings.planeEnabled) {
        profile.setDetails("action=skip overlay_or_composite");
        return false;
    }

    auto surf = _surfWeak.lock();
    auto* plane = dynamic_cast<PlaneSurface*>(surf.get());
    if (!plane) {
        profile.setDetails("action=skip non_plane");
        return false;
    }

    const cv::Vec3f vx = plane->basisX();
    const cv::Vec3f vy = plane->basisY();
    const cv::Vec3f n = plane->normal({0, 0, 0});
    const int uAxis = dominantAxis(vx);
    const int vAxis = dominantAxis(vy);
    const int fixedAxis = dominantAxis(n);
    if (uAxis < 0 || vAxis < 0 || fixedAxis < 0 ||
        uAxis == vAxis || uAxis == fixedAxis || vAxis == fixedAxis) {
        profile.setDetails("action=skip non_axis_aligned");
        return false;
    }

    const int fbW = _framebuffer.width();
    const int fbH = _framebuffer.height();
    if (fbW <= 0 || fbH <= 0) {
        profile.setDetails("action=skip invalid_framebuffer");
        return false;
    }

    const int level = renderStartLevel();
    if (level < 0 || level >= _chunkArray->numLevels()) {
        if (profile.enabled()) {
            profile.setDetails(std::format("action=skip invalid_level level={}", level));
        }
        return false;
    }

    const float halfW = static_cast<float>(fbW) * 0.5f / _scale;
    const float halfH = static_cast<float>(fbH) * 0.5f / _scale;
    const cv::Vec3f origin0 = vx * (_surfacePtrX - halfW)
                            + vy * (_surfacePtrY - halfH)
                            + plane->origin()
                            + n * _zOff;
    const cv::Vec3f vxStep0 = vx / _scale;
    const cv::Vec3f vyStep0 = vy / _scale;
    constexpr int kMaxPreviewSourcePixels = 4096 * 4096;

    const uint8_t fillByte = static_cast<uint8_t>(
        std::clamp(std::lround(_chunkArray->fillValue()), 0L, 255L));
    cv::Mat_<uint8_t> displayValues;
    cv::Mat_<uint8_t> displayCoverage(fbH, fbW, uint8_t(0));
    displayValues.create(fbH, fbW);
    displayValues.setTo(fillByte);

    int coveredChunks = 0;
    int renderedLevels = 0;
    bool emptyVisibleRegion = false;
    auto renderLevel = [&](int sampleLevel) -> bool {
        const auto shapeZyx = _chunkArray->shape(sampleLevel);
        const auto chunkZyx = _chunkArray->chunkShape(sampleLevel);
        std::array<int, 3> shapeXyz{shapeZyx[2], shapeZyx[1], shapeZyx[0]};
        std::array<int, 3> chunkXyz{chunkZyx[2], chunkZyx[1], chunkZyx[0]};
        if (shapeXyz[0] <= 0 || shapeXyz[1] <= 0 || shapeXyz[2] <= 0 ||
            chunkXyz[0] <= 0 || chunkXyz[1] <= 0 || chunkXyz[2] <= 0) {
            return false;
        }

        const auto transform = _chunkArray->levelTransform(sampleLevel);
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

        const int uBegin = std::clamp(int(std::floor(std::min(u0, u1))) - 1, 0, shapeXyz[uAxis]);
        const int uEnd = std::clamp(int(std::ceil(std::max(u0, u1))) + 2, 0, shapeXyz[uAxis]);
        const int vBegin = std::clamp(int(std::floor(std::min(v0, v1))) - 1, 0, shapeXyz[vAxis]);
        const int vEnd = std::clamp(int(std::ceil(std::max(v0, v1))) + 2, 0, shapeXyz[vAxis]);
        if (uEnd <= uBegin || vEnd <= vBegin) {
            emptyVisibleRegion = true;
            return false;
        }

        const int srcW = uEnd - uBegin;
        const int srcH = vEnd - vBegin;
        if (srcW <= 0 || srcH <= 0 || srcW * srcH > kMaxPreviewSourcePixels)
            return false;

        cv::Mat_<uint8_t> src(srcH, srcW, fillByte);
        cv::Mat_<uint8_t> srcCoverage(srcH, srcW, uint8_t(0));
        const int uChunkBegin = uBegin / chunkXyz[uAxis];
        const int uChunkEnd = (uEnd - 1) / chunkXyz[uAxis];
        const int vChunkBegin = vBegin / chunkXyz[vAxis];
        const int vChunkEnd = (vEnd - 1) / chunkXyz[vAxis];
        const int fixedChunk = fixed / chunkXyz[fixedAxis];
        int levelCoveredChunks = 0;

        for (int vc = vChunkBegin; vc <= vChunkEnd; ++vc) {
            for (int uc = uChunkBegin; uc <= uChunkEnd; ++uc) {
                std::array<int, 3> chunkXyzCoord{};
                chunkXyzCoord[uAxis] = uc;
                chunkXyzCoord[vAxis] = vc;
                chunkXyzCoord[fixedAxis] = fixedChunk;

                vc::render::ChunkResult chunk = _chunkArray->tryGetChunk(
                    sampleLevel,
                    chunkXyzCoord[2],
                    chunkXyzCoord[1],
                    chunkXyzCoord[0]);
                if (chunk.status == vc::render::ChunkStatus::MissQueued ||
                    chunk.status == vc::render::ChunkStatus::Missing ||
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

                ++levelCoveredChunks;
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
        if (levelCoveredChunks == 0)
            return false;

        cv::Mat_<uint8_t> levelValues;
        cv::Mat_<uint8_t> levelCoverage;
        const cv::Matx23f dstToSrc(
            uStep[uAxis], 0.0f, u0 - float(uBegin),
            0.0f, vStep[vAxis], v0 - float(vBegin));
        cv::warpAffine(src, levelValues, dstToSrc, cv::Size(fbW, fbH),
                       cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                       cv::BORDER_CONSTANT, cv::Scalar(fillByte));
        cv::warpAffine(srcCoverage, levelCoverage, dstToSrc, cv::Size(fbW, fbH),
                       cv::INTER_NEAREST | cv::WARP_INVERSE_MAP,
                       cv::BORDER_CONSTANT, cv::Scalar(0));
        int filledPixels = 0;
        for (int y = 0; y < fbH; ++y) {
            const auto* levelValueRow = levelValues.ptr<uint8_t>(y);
            const auto* levelCoverageRow = levelCoverage.ptr<uint8_t>(y);
            auto* valueRow = displayValues.ptr<uint8_t>(y);
            auto* coverageRow = displayCoverage.ptr<uint8_t>(y);
            for (int x = 0; x < fbW; ++x) {
                if (coverageRow[x] || !levelCoverageRow[x])
                    continue;
                valueRow[x] = levelValueRow[x];
                coverageRow[x] = 1;
                ++filledPixels;
            }
        }
        if (filledPixels == 0)
            return false;
        coveredChunks += levelCoveredChunks;
        ++renderedLevels;
        return true;
    };

    for (int sampleLevel = level; sampleLevel < _chunkArray->numLevels(); ++sampleLevel)
        (void)renderLevel(sampleLevel);

    if (renderedLevels == 0) {
        if (emptyVisibleRegion) {
            _framebuffer.fill(QColor(64, 64, 64));
            syncCameraTransform();
            _view->viewport()->update();
            profile.setDetails("action=empty_visible_region");
            return true;
        }
        profile.setDetails("action=skip no_chunks_ready");
        return false;
    }

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
        renderIntersections("interactive axis-aligned preview rendered");
    emit overlaysUpdated();
    _view->viewport()->update();
    if (profile.enabled()) {
        profile.setDetails(std::format(
            "action=rendered startLevel={} renderedLevels={} coveredChunks={} fixedAxis={}",
            level, renderedLevels, coveredChunks, fixedAxis));
    }
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
            renderIntersections("interactive preview fallback no stable frame");
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
        renderIntersections("interactive preview transformed stable frame");
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
    // to a finer level. Surface-resolution views keep their target level to
    // avoid panning blur.
    int level = _dsScaleIdx;
    if (preferSurfaceResolution && _chunkArray && level < _chunkArray->numLevels() - 1)
        level -= kSurfaceResolutionLevelBias;
    return std::clamp(level, 0, _chunkArray->numLevels() - 1);
}

void CChunkedVolumeViewer::markInteractiveMotion(double)
{
    if (!_interactionClock.isValid())
        _interactionClock.start();

    const qint64 now = _interactionClock.elapsed();
    _lastInteractionMs = now;
    _interactivePreview = true;
    if (_settleRenderTimer)
        _settleRenderTimer->start();
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

void CChunkedVolumeViewer::prefetchPlaneHalo(
    const cv::Vec3f& origin,
    const cv::Vec3f& vxStep,
    const cv::Vec3f& vyStep,
    int startLevel,
    const vc::render::ChunkedPlaneSampler::Options& options)
{
    if (!_chunkArray || _interactivePreview || _framebuffer.isNull())
        return;

    const int fbW = _framebuffer.width();
    const int fbH = _framebuffer.height();
    if (fbW <= 0 || fbH <= 0)
        return;

    const int halo = kChunkPrefetchHaloPx;
    const int expandedW = fbW + 2 * halo;
    std::unordered_set<vc::render::ChunkKey, vc::render::ChunkKeyHash> uniqueKeys;
    auto collect = [&](vc::render::IChunkedArray& array,
                       const cv::Vec3f& stripOrigin,
                       int stripW,
                       int stripH) {
        if (stripW <= 0 || stripH <= 0)
            return;
        cv::Mat_<uint8_t> stripCoverage(stripH, stripW, uint8_t(0));
        auto keys = vc::render::ChunkedPlaneSampler::collectPlaneDependencies(
            array, startLevel, stripOrigin, vxStep, vyStep, stripCoverage, options);
        uniqueKeys.insert(keys.begin(), keys.end());
    };

    collect(*_chunkArray, origin - vxStep * float(halo) - vyStep * float(halo),
            expandedW, halo);
    collect(*_chunkArray, origin - vxStep * float(halo) + vyStep * float(fbH),
            expandedW, halo);
    collect(*_chunkArray, origin - vxStep * float(halo), halo, fbH);
    collect(*_chunkArray, origin + vxStep * float(fbW), halo, fbH);

    std::vector<vc::render::ChunkKey> keys;
    keys.reserve(uniqueKeys.size());
    for (const auto& key : uniqueKeys)
        keys.push_back(key);
    _chunkArray->prefetchChunks(keys, false, kChunkPrefetchPriorityOffset);

    if (!_overlayChunkArray || !_overlayVolume || _overlayOpacity <= 0.0f)
        return;
    uniqueKeys.clear();
    collect(*_overlayChunkArray, origin - vxStep * float(halo) - vyStep * float(halo),
            expandedW, halo);
    collect(*_overlayChunkArray, origin - vxStep * float(halo) + vyStep * float(fbH),
            expandedW, halo);
    collect(*_overlayChunkArray, origin - vxStep * float(halo), halo, fbH);
    collect(*_overlayChunkArray, origin + vxStep * float(fbW), halo, fbH);
    keys.clear();
    keys.reserve(uniqueKeys.size());
    for (const auto& key : uniqueKeys)
        keys.push_back(key);
    _overlayChunkArray->prefetchChunks(keys, false, kChunkPrefetchPriorityOffset);
}

void CChunkedVolumeViewer::prefetchPlaneNormalNeighbors(
    PlaneSurface& plane,
    int startLevel,
    const vc::render::ChunkedPlaneSampler::Options& options)
{
    if (!_chunkArray || _framebuffer.isNull())
        return;

    const int fbW = _framebuffer.width();
    const int fbH = _framebuffer.height();
    if (fbW <= 0 || fbH <= 0)
        return;

    const cv::Vec3f vx = plane.basisX();
    const cv::Vec3f vy = plane.basisY();
    cv::Vec3f normal = plane.normal({0, 0, 0});
    const float normalLen = static_cast<float>(cv::norm(normal));
    if (normalLen <= 1e-6f)
        return;
    normal *= 1.0f / normalLen;

    const float halfW = static_cast<float>(fbW) * 0.5f / _scale;
    const float halfH = static_cast<float>(fbH) * 0.5f / _scale;
    const cv::Vec3f origin = vx * (_surfacePtrX - halfW)
                           + vy * (_surfacePtrY - halfH)
                           + plane.origin()
                           + normal * _zOff;
    const cv::Vec3f vxStep = vx / _scale;
    const cv::Vec3f vyStep = vy / _scale;

    const int sampleStride = std::max(1, kNormalPrefetchSampleStridePx);
    const int sampleW = std::max(1, (fbW + sampleStride - 1) / sampleStride + 1);
    const int sampleH = std::max(1, (fbH + sampleStride - 1) / sampleStride + 1);
    const cv::Vec3f sampleVxStep = vxStep * float(sampleStride);
    const cv::Vec3f sampleVyStep = vyStep * float(sampleStride);

    auto chunkDistance = [&](vc::render::IChunkedArray& array, int level) -> float {
        if (level < 0 || level >= array.numLevels())
            return 0.0f;
        const auto chunkZyx = array.chunkShape(level);
        const std::array<int, 3> chunkXyz{chunkZyx[2], chunkZyx[1], chunkZyx[0]};
        const auto transform = array.levelTransform(level);
        int axis = 0;
        float best = std::abs(normal[0]);
        for (int i = 1; i < 3; ++i) {
            const float v = std::abs(normal[i]);
            if (v > best) {
                best = v;
                axis = i;
            }
        }
        const double levelScale = std::abs(transform.scaleFromLevel0[axis]);
        if (chunkXyz[axis] <= 0 || levelScale <= 1e-12)
            return 0.0f;
        return static_cast<float>(double(chunkXyz[axis]) / levelScale);
    };

    auto collect = [&](vc::render::IChunkedArray& array, int level, float distance) {
        if (distance <= 0.0f || level < 0 || level >= array.numLevels())
            return;
        std::unordered_set<vc::render::ChunkKey, vc::render::ChunkKeyHash> uniqueKeys;
        cv::Mat_<uint8_t> coverage(sampleH, sampleW, uint8_t(0));
        for (float dir : {-1.0f, 1.0f}) {
            auto keys = vc::render::ChunkedPlaneSampler::collectPlaneDependencies(
                array, level, origin + normal * (dir * distance),
                sampleVxStep, sampleVyStep, coverage, options);
            uniqueKeys.insert(keys.begin(), keys.end());
        }

        if (uniqueKeys.empty())
            return;
        std::vector<vc::render::ChunkKey> keys;
        keys.reserve(uniqueKeys.size());
        for (const auto& key : uniqueKeys)
            keys.push_back(key);
        array.prefetchChunks(keys, false, kChunkPrefetchPriorityOffset);
    };

    if (_chunkArray->numLevels() <= 0)
        return;
    startLevel = std::clamp(startLevel, 0, _chunkArray->numLevels() - 1);
    collect(*_chunkArray, startLevel, chunkDistance(*_chunkArray, startLevel));

    if (!_overlayChunkArray || !_overlayVolume || _overlayOpacity <= 0.0f)
        return;
    if (_overlayChunkArray->numLevels() <= 0)
        return;
    const int overlayLevel = std::clamp(startLevel, 0, _overlayChunkArray->numLevels() - 1);
    collect(*_overlayChunkArray, overlayLevel, chunkDistance(*_overlayChunkArray, overlayLevel));
}

void CChunkedVolumeViewer::prefetchSurfaceHalo(
    Surface& surf,
    int startLevel,
    const vc::render::ChunkedPlaneSampler::Options& options,
    int fbW,
    int fbH)
{
    if (!_chunkArray || _interactivePreview || fbW <= 0 || fbH <= 0)
        return;

    const int halo = kChunkPrefetchHaloPx;
    const cv::Vec3f baseOffset(_surfacePtrX * _scale - float(fbW) * 0.5f,
                               _surfacePtrY * _scale - float(fbH) * 0.5f,
                               0.0f);
    std::unordered_set<vc::render::ChunkKey, vc::render::ChunkKeyHash> uniqueKeys;

    auto collect = [&](vc::render::IChunkedArray& array,
                       int sceneX,
                       int sceneY,
                       int stripW,
                       int stripH) {
        if (stripW <= 0 || stripH <= 0)
            return;

        cv::Mat_<cv::Vec3f> coords;
        cv::Mat_<cv::Vec3f> normals;
        const cv::Vec3f offset = baseOffset + cv::Vec3f(float(sceneX), float(sceneY), 0.0f);
        surf.gen(&coords, _zOff != 0.0f ? &normals : nullptr,
                 cv::Size(stripW, stripH), {0, 0, 0}, _scale, offset);

        applyPerPixelNormalOffset(coords, normals, _zOff);

        cv::Mat_<uint8_t> coverage(stripH, stripW, uint8_t(0));
        auto keys = vc::render::ChunkedPlaneSampler::collectCoordsDependencies(
            array, startLevel, coords, coverage, options);
        uniqueKeys.insert(keys.begin(), keys.end());
    };

    collect(*_chunkArray, -halo, -halo, fbW + 2 * halo, halo);
    collect(*_chunkArray, -halo, fbH, fbW + 2 * halo, halo);
    collect(*_chunkArray, -halo, 0, halo, fbH);
    collect(*_chunkArray, fbW, 0, halo, fbH);

    std::vector<vc::render::ChunkKey> keys;
    keys.reserve(uniqueKeys.size());
    for (const auto& key : uniqueKeys)
        keys.push_back(key);
    _chunkArray->prefetchChunks(keys, false, kChunkPrefetchPriorityOffset);

    if (!_overlayChunkArray || !_overlayVolume || _overlayOpacity <= 0.0f)
        return;
    uniqueKeys.clear();
    collect(*_overlayChunkArray, -halo, -halo, fbW + 2 * halo, halo);
    collect(*_overlayChunkArray, -halo, fbH, fbW + 2 * halo, halo);
    collect(*_overlayChunkArray, -halo, 0, halo, fbH);
    collect(*_overlayChunkArray, fbW, 0, halo, fbH);
    keys.clear();
    keys.reserve(uniqueKeys.size());
    for (const auto& key : uniqueKeys)
        keys.push_back(key);
    _overlayChunkArray->prefetchChunks(keys, false, kChunkPrefetchPriorityOffset);
}

void CChunkedVolumeViewer::prefetchVisibleSurfaceChunks(int priorityOffset)
{
    if (!_chunkArray || _framebuffer.isNull() || _framebuffer.width() <= 0 ||
        _framebuffer.height() <= 0) {
        return;
    }

    auto surf = _surfWeak.lock();
    auto* quad = dynamic_cast<QuadSurface*>(surf.get());
    if (!quad) {
        _surfaceChunkPrefetchCache = {};
        return;
    }

    const cv::Mat_<cv::Vec3f>* points = quad->rawPointsPtr();
    if (!points || points->cols < 2 || points->rows < 2)
        return;

    const int level = renderStartLevel(true);
    if (level < 0 || level >= _chunkArray->numLevels())
        return;

    const float halfW = static_cast<float>(_framebuffer.width()) * 0.5f / std::max(_scale, kMinScale);
    const float halfH = static_cast<float>(_framebuffer.height()) * 0.5f / std::max(_scale, kMinScale);
    const cv::Vec2f g0 = quad->ptrToGrid({_surfacePtrX - halfW, _surfacePtrY - halfH, 0.0f});
    const cv::Vec2f g1 = quad->ptrToGrid({_surfacePtrX + halfW, _surfacePtrY + halfH, 0.0f});
    const float minX = std::min(g0[0], g1[0]);
    const float maxX = std::max(g0[0], g1[0]);
    const float minY = std::min(g0[1], g1[1]);
    const float maxY = std::max(g0[1], g1[1]);

    cv::Rect visibleCells(
        int(std::floor(minX)) - 1,
        int(std::floor(minY)) - 1,
        std::max(1, int(std::ceil(maxX)) - int(std::floor(minX)) + 3),
        std::max(1, int(std::ceil(maxY)) - int(std::floor(minY)) + 3));
    const cv::Rect cellBounds(0, 0, points->cols - 1, points->rows - 1);
    visibleCells &= cellBounds;
    if (visibleCells.empty())
        return;

    const float cellsPerPixelX =
        float(visibleCells.width) / float(std::max(1, _framebuffer.width()));
    const float cellsPerPixelY =
        float(visibleCells.height) / float(std::max(1, _framebuffer.height()));
    const int padX = std::max(1, int(std::ceil(float(kChunkPrefetchHaloPx) * cellsPerPixelX)) + 2);
    const int padY = std::max(1, int(std::ceil(float(kChunkPrefetchHaloPx) * cellsPerPixelY)) + 2);
    cv::Rect paddedCells(
        visibleCells.x - padX,
        visibleCells.y - padY,
        visibleCells.width + padX * 2,
        visibleCells.height + padY * 2);
    paddedCells &= cellBounds;
    if (paddedCells.empty())
        return;

    auto& cache = _surfaceChunkPrefetchCache;
    if (!cache.valid || cache.surface != quad || cache.level != level ||
        cache.sampling != _samplingMethod) {
        cache = {};
        cache.surface = quad;
        cache.level = level;
        cache.sampling = _samplingMethod;
    }

    auto collectKeysForCells = [&](const cv::Rect& cells,
                                   std::unordered_set<vc::render::ChunkKey,
                                                      vc::render::ChunkKeyHash>& keys) {
        if (cells.empty())
            return;
        const int tx0 = cells.x / kSurfaceCellTileSize;
        const int ty0 = cells.y / kSurfaceCellTileSize;
        const int tx1 = (cells.x + cells.width - 1) / kSurfaceCellTileSize;
        const int ty1 = (cells.y + cells.height - 1) / kSurfaceCellTileSize;
        for (int ty = ty0; ty <= ty1; ++ty) {
            for (int tx = tx0; tx <= tx1; ++tx) {
                const std::uint64_t key = surfaceTileKey(tx, ty);
                auto it = cache.tileKeys.find(key);
                if (it == cache.tileKeys.end()) {
                    cv::Rect tileCells(tx * kSurfaceCellTileSize,
                                       ty * kSurfaceCellTileSize,
                                       kSurfaceCellTileSize,
                                       kSurfaceCellTileSize);
                    tileCells &= cellBounds;
                    it = cache.tileKeys.emplace(
                        key,
                        collectSurfaceCellChunkKeys(*_chunkArray, *quad, tileCells, level, _samplingMethod)).first;
                }
                keys.insert(it->second.begin(), it->second.end());
            }
        }
    };

    std::unordered_set<vc::render::ChunkKey, vc::render::ChunkKeyHash> visibleKeys;
    std::unordered_set<vc::render::ChunkKey, vc::render::ChunkKeyHash> speculativeKeys;
    collectKeysForCells(visibleCells, visibleKeys);
    collectKeysForCells(paddedCells, speculativeKeys);
    for (const auto& key : visibleKeys)
        speculativeKeys.erase(key);

    cache.valid = true;
    cache.prefetchedCellRect = rectContains(cache.prefetchedCellRect, paddedCells)
        ? cache.prefetchedCellRect
        : paddedCells;

    std::vector<vc::render::ChunkKey> keys;
    keys.reserve(visibleKeys.size());
    for (const auto& key : visibleKeys)
        keys.push_back(key);
    _chunkArray->prefetchChunks(keys, false, priorityOffset);

    keys.clear();
    keys.reserve(speculativeKeys.size());
    for (const auto& key : speculativeKeys)
        keys.push_back(key);
    _chunkArray->prefetchChunks(keys, false, kChunkPrefetchPriorityOffset);
}

struct CChunkedVolumeViewer::RenderContext {
    std::uint64_t serial = 0;
    int fbW = 0;
    int fbH = 0;
    float surfacePtrX = 0.0f;
    float surfacePtrY = 0.0f;
    float scale = 1.0f;
    float zOff = 0.0f;
    cv::Vec3f zOffWorldDir{0, 0, 0};
    int startLevel = 0;
    vc::Sampling samplingMethod = vc::Sampling::Trilinear;
    CompositeRenderSettings compositeSettings;
    float windowLow = 0.0f;
    float windowHigh = 255.0f;
    std::string baseColormapId;
    std::shared_ptr<Surface> surf;
    std::shared_ptr<vc::render::ChunkCache> chunkArray;
    std::shared_ptr<Volume> overlayVolume;
    std::shared_ptr<vc::render::ChunkCache> overlayChunkArray;
    float overlayOpacity = 0.0f;
    std::string overlayColormapId;
    float overlayWindowLow = 0.0f;
    float overlayWindowHigh = 255.0f;
    bool interactivePreview = false;
    std::shared_ptr<GeneratedSurfaceCache> genCache;
    bool genCacheDirty = false;
    std::string profileReason;
    std::string profileCaller;
};

struct CChunkedVolumeViewer::RenderResult {
    std::uint64_t serial = 0;
    QImage framebuffer;
    float surfacePtrX = 0.0f;
    float surfacePtrY = 0.0f;
    float scale = 1.0f;
    qint64 renderFrameElapsedMs = 0;
};

CChunkedVolumeViewer::RenderResult CChunkedVolumeViewer::renderFrame(RenderContext ctx)
{
    QElapsedTimer renderTimer;
    if (ProfileLoggingEnabled()) {
        renderTimer.start();
        Logger()->info("[vc3d-profile] renderFrame begin reason='{}' caller='{}' serial={} surf='{}' size={}x{} level={} interactive={} overlay={} composite={} planeComposite={}",
                       ctx.profileReason, ctx.profileCaller, ctx.serial,
                       ctx.surf ? ctx.surf->id : std::string(""),
                       ctx.fbW, ctx.fbH, ctx.startLevel, ctx.interactivePreview,
                       bool(ctx.overlayChunkArray && ctx.overlayVolume && ctx.overlayOpacity > 0.0f),
                       ctx.compositeSettings.enabled, ctx.compositeSettings.planeEnabled);
    }
    RenderResult result;
    result.serial = ctx.serial;
    result.surfacePtrX = ctx.surfacePtrX;
    result.surfacePtrY = ctx.surfacePtrY;
    result.scale = ctx.scale;
    result.framebuffer = QImage(std::max(1, ctx.fbW), std::max(1, ctx.fbH), QImage::Format_RGB32);
    result.framebuffer.fill(QColor(64, 64, 64));

    auto finishRenderFrameProfile = [&]() {
        if (!ProfileLoggingEnabled() || !renderTimer.isValid())
            return;
        result.renderFrameElapsedMs = renderTimer.elapsed();
        Logger()->info("[vc3d-profile] renderFrame end elapsed_ms={} reason='{}' caller='{}' serial={} framebuffer={}x{}",
                       result.renderFrameElapsedMs, ctx.profileReason, ctx.profileCaller,
                       result.serial, result.framebuffer.width(), result.framebuffer.height());
    };

    if (!ctx.surf || !ctx.chunkArray || ctx.fbW <= 0 || ctx.fbH <= 0) {
        finishRenderFrameProfile();
        return result;
    }

    cv::Mat_<uint8_t> values(ctx.fbH, ctx.fbW, uint8_t(0));
    cv::Mat_<uint8_t> coverage(ctx.fbH, ctx.fbW, uint8_t(0));
    cv::Mat_<uint8_t> overlayValues;
    cv::Mat_<uint8_t> overlayCoverage;
    const vc::render::ChunkedPlaneSampler::Options options(ctx.samplingMethod, 32);

    auto streamingCompositeUnsupported = [&]() {
        return !isSupportedStreamingCompositeMethod(ctx.compositeSettings.params.method) ||
               ctx.compositeSettings.params.lightingEnabled ||
               ctx.compositeSettings.params.method == "beerLambert" ||
               ctx.compositeSettings.postClaheEnabled ||
               ctx.compositeSettings.postRakingEnabled ||
               ctx.compositeSettings.postRemoveSmallComponents ||
               ctx.compositeSettings.useVolumeGradients;
    };

    auto samplePlane = [&](const cv::Vec3f& origin,
                           const cv::Vec3f& vxStep,
                           const cv::Vec3f& vyStep,
                           const cv::Vec3f& normal,
                           cv::Mat_<uint8_t>& dst,
                           cv::Mat_<uint8_t>& cov,
                           vc::render::ChunkCache& array) {
        const bool wantComposite = ctx.compositeSettings.planeEnabled && !streamingCompositeUnsupported();
        if (!wantComposite) {
            if (ctx.interactivePreview) {
                vc::render::ChunkedPlaneSampler::samplePlaneCoarseToFine(
                    array, ctx.startLevel, origin, vxStep, vyStep, dst, cov, options);
            } else {
                vc::render::ChunkedPlaneSampler::samplePlaneFineToCoarse(
                    array, ctx.startLevel, origin, vxStep, vyStep, dst, cov, options);
            }
            return;
        }

        const int front = std::max(0, ctx.compositeSettings.planeLayersFront);
        const int behind = std::max(0, ctx.compositeSettings.planeLayersBehind);
        const int numLayers = front + behind + 1;
        const int zStart = -behind;
        const float zStep = ctx.compositeSettings.reverseDirection ? -1.0f : 1.0f;
        const auto compositeOptions = vc::render::ChunkedPlaneSampler::Options(vc::Sampling::Nearest, options.tileSize);
        std::vector<cv::Mat_<uint8_t>> layerValues;
        std::vector<cv::Mat_<uint8_t>> layerCoverage;
        layerValues.reserve(numLayers);
        layerCoverage.reserve(numLayers);
        for (int i = 0; i < numLayers; ++i) {
            layerValues.emplace_back(dst.rows, dst.cols, uint8_t(0));
            layerCoverage.emplace_back(dst.rows, dst.cols, uint8_t(0));
            const cv::Vec3f layerOrigin = origin + normal * (float(zStart + i) * zStep);
            if (ctx.interactivePreview) {
                vc::render::ChunkedPlaneSampler::samplePlaneCoarseToFine(
                    array, ctx.startLevel, layerOrigin, vxStep, vyStep,
                    layerValues.back(), layerCoverage.back(), compositeOptions);
            } else {
                vc::render::ChunkedPlaneSampler::samplePlaneFineToCoarse(
                    array, ctx.startLevel, layerOrigin, vxStep, vyStep,
                    layerValues.back(), layerCoverage.back(), compositeOptions);
            }
        }
        LayerStack stack;
        stack.values.resize(numLayers);
        for (int y = 0; y < dst.rows; ++y) {
            auto* dstRow = dst.ptr<uint8_t>(y);
            auto* covRow = cov.ptr<uint8_t>(y);
            for (int x = 0; x < dst.cols; ++x) {
                stack.validCount = 0;
                for (int i = 0; i < numLayers; ++i) {
                    if (!layerCoverage[i](y, x))
                        continue;
                    const float value = static_cast<float>(layerValues[i](y, x));
                    if (value < static_cast<float>(ctx.compositeSettings.params.isoCutoff))
                        continue;
                    stack.values[stack.validCount++] = value;
                }
                if (stack.validCount > 0) {
                    dstRow[x] = static_cast<uint8_t>(std::clamp(
                        compositeLayerStack(stack, ctx.compositeSettings.params), 0.0f, 255.0f));
                    covRow[x] = 1;
                }
            }
        }
    };

    auto sampleCoords = [&](const cv::Mat_<cv::Vec3f>& coords,
                            const cv::Mat_<cv::Vec3f>& normals,
                            cv::Mat_<uint8_t>& dst,
                            cv::Mat_<uint8_t>& cov,
                            vc::render::ChunkCache& array) {
        const bool wantComposite = ctx.compositeSettings.enabled &&
                                   !streamingCompositeUnsupported() &&
                                   !normals.empty();
        if (!wantComposite) {
            if (ctx.interactivePreview) {
                vc::render::ChunkedPlaneSampler::sampleCoordsCoarseToFine(
                    array, ctx.startLevel, coords, dst, cov, options);
            } else {
                vc::render::ChunkedPlaneSampler::sampleCoordsFineToCoarse(
                    array, ctx.startLevel, coords, dst, cov, options);
            }
            return;
        }

        const int front = std::max(0, ctx.compositeSettings.layersFront);
        const int behind = std::max(0, ctx.compositeSettings.layersBehind);
        const int numLayers = front + behind + 1;
        const int zStart = -behind;
        const float zStep = ctx.compositeSettings.reverseDirection ? -1.0f : 1.0f;
        const auto compositeOptions = vc::render::ChunkedPlaneSampler::Options(vc::Sampling::Nearest, options.tileSize);
        std::vector<cv::Mat_<uint8_t>> layerValues;
        std::vector<cv::Mat_<uint8_t>> layerCoverage;
        cv::Mat_<cv::Vec3f> layerCoords(coords.rows, coords.cols);
        layerValues.reserve(numLayers);
        layerCoverage.reserve(numLayers);
        for (int i = 0; i < numLayers; ++i) {
            const float offset = float(zStart + i) * zStep;
            for (int y = 0; y < coords.rows; ++y) {
                const auto* src = coords.ptr<cv::Vec3f>(y);
                const auto* nrow = normals.ptr<cv::Vec3f>(y);
                auto* dstRow = layerCoords.ptr<cv::Vec3f>(y);
                for (int x = 0; x < coords.cols; ++x) {
                    if (!std::isfinite(src[x][0]) || src[x][0] == -1.0f)
                        dstRow[x] = src[x];
                    else
                        dstRow[x] = src[x] + nrow[x] * offset;
                }
            }
            layerValues.emplace_back(dst.rows, dst.cols, uint8_t(0));
            layerCoverage.emplace_back(dst.rows, dst.cols, uint8_t(0));
            if (ctx.interactivePreview) {
                vc::render::ChunkedPlaneSampler::sampleCoordsCoarseToFine(
                    array, ctx.startLevel, layerCoords,
                    layerValues.back(), layerCoverage.back(), compositeOptions);
            } else {
                vc::render::ChunkedPlaneSampler::sampleCoordsFineToCoarse(
                    array, ctx.startLevel, layerCoords,
                    layerValues.back(), layerCoverage.back(), compositeOptions);
            }
        }
        LayerStack stack;
        stack.values.resize(numLayers);
        for (int y = 0; y < dst.rows; ++y) {
            auto* dstRow = dst.ptr<uint8_t>(y);
            auto* covRow = cov.ptr<uint8_t>(y);
            for (int x = 0; x < dst.cols; ++x) {
                stack.validCount = 0;
                for (int i = 0; i < numLayers; ++i) {
                    if (!layerCoverage[i](y, x))
                        continue;
                    const float value = static_cast<float>(layerValues[i](y, x));
                    if (value < static_cast<float>(ctx.compositeSettings.params.isoCutoff))
                        continue;
                    stack.values[stack.validCount++] = value;
                }
                if (stack.validCount > 0) {
                    dstRow[x] = static_cast<uint8_t>(std::clamp(
                        compositeLayerStack(stack, ctx.compositeSettings.params), 0.0f, 255.0f));
                    covRow[x] = 1;
                }
            }
        }
    };

    const bool planeView = dynamic_cast<PlaneSurface*>(ctx.surf.get()) != nullptr;
    if (auto* plane = dynamic_cast<PlaneSurface*>(ctx.surf.get())) {
        const cv::Vec3f vx = plane->basisX();
        const cv::Vec3f vy = plane->basisY();
        const cv::Vec3f n = plane->normal({0, 0, 0});
        const float halfW = static_cast<float>(ctx.fbW) * 0.5f / ctx.scale;
        const float halfH = static_cast<float>(ctx.fbH) * 0.5f / ctx.scale;
        const cv::Vec3f origin = vx * (ctx.surfacePtrX - halfW)
                               + vy * (ctx.surfacePtrY - halfH)
                               + plane->origin()
                               + n * ctx.zOff;
        const cv::Vec3f vxStep = vx / ctx.scale;
        const cv::Vec3f vyStep = vy / ctx.scale;
        samplePlane(origin, vxStep, vyStep, n, values, coverage, *ctx.chunkArray);
        if (ctx.overlayChunkArray && ctx.overlayVolume && ctx.overlayOpacity > 0.0f) {
            overlayValues.create(ctx.fbH, ctx.fbW);
            overlayCoverage.create(ctx.fbH, ctx.fbW);
            overlayValues.setTo(0);
            overlayCoverage.setTo(0);
            const int level = std::clamp(ctx.startLevel, 0, ctx.overlayChunkArray->numLevels() - 1);
            if (ctx.interactivePreview) {
                vc::render::ChunkedPlaneSampler::samplePlaneCoarseToFine(
                    *ctx.overlayChunkArray, level, origin, vxStep, vyStep,
                    overlayValues, overlayCoverage, options);
            } else {
                vc::render::ChunkedPlaneSampler::samplePlaneFineToCoarse(
                    *ctx.overlayChunkArray, level, origin, vxStep, vyStep,
                    overlayValues, overlayCoverage, options);
            }
        }
    } else {
        cv::Mat_<cv::Vec3f> coords;
        cv::Mat_<cv::Vec3f> normals;
        const cv::Vec3f offset(ctx.surfacePtrX * ctx.scale - float(ctx.fbW) * 0.5f,
                               ctx.surfacePtrY * ctx.scale - float(ctx.fbH) * 0.5f,
                               0.0f);
        const bool needSurfaceNormals =
            ctx.zOff != 0.0f ||
            (ctx.compositeSettings.enabled && !streamingCompositeUnsupported());

        bool genCacheHit = false;
        if (ctx.genCache) {
            std::lock_guard lock(ctx.genCache->mutex);
            if (ctx.genCacheDirty) {
                ctx.genCache->valid = false;
                ctx.genCache->coords.release();
                ctx.genCache->normals.release();
            }
            genCacheHit =
                ctx.genCache->valid &&
                ctx.genCache->surface == ctx.surf.get() &&
                ctx.genCache->fbW == ctx.fbW &&
                ctx.genCache->fbH == ctx.fbH &&
                ctx.genCache->scale == ctx.scale &&
                ctx.genCache->offset == offset &&
                ctx.genCache->zOff == ctx.zOff &&
                ctx.genCache->zOffWorldDir == ctx.zOffWorldDir &&
                !ctx.genCache->coords.empty() &&
                (!needSurfaceNormals || !ctx.genCache->normals.empty());
            if (genCacheHit) {
                coords = ctx.genCache->coords;
                if (needSurfaceNormals)
                    normals = ctx.genCache->normals;
            }
        }

        if (!genCacheHit) {
            ctx.surf->gen(&coords, needSurfaceNormals ? &normals : nullptr,
                          cv::Size(ctx.fbW, ctx.fbH), {0, 0, 0}, ctx.scale, offset);
            applyPerPixelNormalOffset(coords, normals, ctx.zOff);

            if (ctx.genCache && !coords.empty()) {
                std::lock_guard lock(ctx.genCache->mutex);
                ctx.genCache->valid = true;
                ctx.genCache->surface = ctx.surf.get();
                ctx.genCache->fbW = ctx.fbW;
                ctx.genCache->fbH = ctx.fbH;
                ctx.genCache->scale = ctx.scale;
                ctx.genCache->offset = offset;
                ctx.genCache->zOff = ctx.zOff;
                ctx.genCache->zOffWorldDir = ctx.zOffWorldDir;
                ctx.genCache->coords = coords;
                ctx.genCache->normals = normals;
            }
        }
        if (!coords.empty()) {
            sampleCoords(coords, normals, values, coverage, *ctx.chunkArray);
            if (ctx.overlayChunkArray && ctx.overlayVolume && ctx.overlayOpacity > 0.0f) {
                overlayValues.create(ctx.fbH, ctx.fbW);
                overlayCoverage.create(ctx.fbH, ctx.fbW);
                overlayValues.setTo(0);
                overlayCoverage.setTo(0);
                const int level = std::clamp(ctx.startLevel, 0, ctx.overlayChunkArray->numLevels() - 1);
                if (ctx.interactivePreview) {
                    vc::render::ChunkedPlaneSampler::sampleCoordsCoarseToFine(
                        *ctx.overlayChunkArray, level, coords, overlayValues, overlayCoverage, options);
                } else {
                    vc::render::ChunkedPlaneSampler::sampleCoordsFineToCoarse(
                        *ctx.overlayChunkArray, level, coords, overlayValues, overlayCoverage, options);
                }
            }
        }
    }

    std::array<uint32_t, 256> lut{};
    vc::buildWindowLevelColormapLut(lut, ctx.windowLow, ctx.windowHigh, ctx.baseColormapId);
    std::array<uint32_t, 256> overlayLut{};
    const bool hasOverlay = !overlayValues.empty() && !overlayCoverage.empty() &&
                            ctx.overlayOpacity > 0.0f;
    if (hasOverlay) {
        vc::buildWindowLevelColormapLut(
            overlayLut, ctx.overlayWindowLow, ctx.overlayWindowHigh, ctx.overlayColormapId);
    }
    auto* fbBits = reinterpret_cast<uint32_t*>(result.framebuffer.bits());
    const int fbStride = result.framebuffer.bytesPerLine() / 4;
    const uint32_t uncoveredPixel = planeView ? 0xFF404040u : 0xFF000000u;
    for (int y = 0; y < ctx.fbH; ++y) {
        auto* row = fbBits + size_t(y) * size_t(fbStride);
        const auto* src = values.ptr<uint8_t>(y);
        const auto* cov = coverage.ptr<uint8_t>(y);
        const auto* overlaySrc = hasOverlay ? overlayValues.ptr<uint8_t>(y) : nullptr;
        const auto* overlayCov = hasOverlay ? overlayCoverage.ptr<uint8_t>(y) : nullptr;
        for (int x = 0; x < ctx.fbW; ++x) {
            uint32_t pixel = cov[x] ? lut[src[x]] : uncoveredPixel;
            if (hasOverlay && overlayCov[x] &&
                overlaySrc[x] >= ctx.overlayWindowLow && overlaySrc[x] <= ctx.overlayWindowHigh) {
                pixel = alphaBlendArgb(pixel, overlayLut[overlaySrc[x]], ctx.overlayOpacity);
            }
            row[x] = pixel;
        }
    }
    finishRenderFrameProfile();
    return result;
}

void CChunkedVolumeViewer::submitRender(const char* reason, std::source_location caller)
{
    ProfileScope profile("submitRender", reason, caller);
    if (profile.enabled()) {
        profile.setDetails(std::format(
            "surf='{}' volume={} chunkArray={} busy={}",
            _surfName, bool(_volume), bool(_chunkArray),
            _renderWorkerBusy.load(std::memory_order_acquire)));
    }
    if (_closing) {
        profile.setDetails("action=skip closing");
        return;
    }

    auto surf = _surfWeak.lock();
    if (!surf || !_volume || !_chunkArray) {
        profile.setDetails("action=skip missing_input");
        return;
    }

    resizeFramebuffer();
    const int fbW = _framebuffer.width();
    const int fbH = _framebuffer.height();
    if (fbW <= 0 || fbH <= 0) {
        if (profile.enabled()) {
            profile.setDetails(std::format("action=skip invalid_framebuffer size={}x{}", fbW, fbH));
        }
        return;
    }

    if (_renderWorkerBusy.exchange(true, std::memory_order_acq_rel)) {
        ++_renderSerial;
        _renderPendingAfterWorker = true;
        profile.setDetails("action=queued_after_worker");
        return;
    }

    prefetchVisibleSurfaceChunks();

    RenderContext ctx;
    ctx.serial = ++_renderSerial;
    ctx.fbW = fbW;
    ctx.fbH = fbH;
    ctx.surfacePtrX = _surfacePtrX;
    ctx.surfacePtrY = _surfacePtrY;
    ctx.scale = _scale;
    ctx.zOff = _zOff;
    ctx.zOffWorldDir = _zOffWorldDir;
    ctx.startLevel = renderStartLevel(dynamic_cast<PlaneSurface*>(surf.get()) == nullptr);
    ctx.samplingMethod = _samplingMethod;
    ctx.compositeSettings = _compositeSettings;
    ctx.windowLow = _windowLow;
    ctx.windowHigh = _windowHigh;
    ctx.baseColormapId = _baseColormapId;
    ctx.surf = std::move(surf);
    ctx.chunkArray = _chunkArray;
    ctx.overlayVolume = _overlayVolume;
    ctx.overlayChunkArray = _overlayChunkArray;
    ctx.overlayOpacity = _overlayOpacity;
    ctx.overlayColormapId = _overlayColormapId;
    ctx.overlayWindowLow = _overlayWindowLow;
    ctx.overlayWindowHigh = _overlayWindowHigh;
    ctx.interactivePreview = _interactivePreview;
    ctx.genCache = _genSurfaceCache;
    ctx.genCacheDirty = _genCacheDirty;
    if (ProfileLoggingEnabled()) {
        ctx.profileReason = reason ? reason : "";
        ctx.profileCaller = profileCaller(caller);
    }
    _genCacheDirty = false;
    if (profile.enabled()) {
        profile.setDetails(std::format(
            "action=worker_start serial={} size={}x{} level={} interactive={}",
            ctx.serial, ctx.fbW, ctx.fbH, ctx.startLevel, ctx.interactivePreview));
    }

    QPointer<CChunkedVolumeViewer> guard(this);
    (void)QtConcurrent::run([guard, ctx = std::move(ctx)]() mutable {
        auto result = std::make_shared<RenderResult>(renderFrame(std::move(ctx)));
        QMetaObject::invokeMethod(qApp, [guard, result = std::move(result)]() mutable {
            if (guard)
                guard->finishRenderOnMainThread(std::move(result));
        }, Qt::QueuedConnection);
    });
}

void CChunkedVolumeViewer::finishRenderOnMainThread(std::shared_ptr<RenderResult> result)
{
    ProfileScope profile("finishRenderOnMainThread", "render worker finished",
                         std::source_location::current());
    if (profile.enabled()) {
        profile.setDetails(std::format(
            "result={} serial={} currentSerial={} pendingAfterWorker={}",
            bool(result), result ? result->serial : 0, _renderSerial,
            _renderPendingAfterWorker));
    }
    _renderWorkerBusy.store(false, std::memory_order_release);
    if (_closing) {
        profile.setDetails("action=drop_closing");
        return;
    }
    if (!result || result->serial != _renderSerial) {
        if (_renderPendingAfterWorker) {
            _renderPendingAfterWorker = false;
            scheduleRender("stale render result had pending worker");
        }
        profile.setDetails("action=drop_stale_result");
        return;
    }

    _framebuffer = std::move(result->framebuffer);
    syncCameraTransform();
    if (_interactivePreview)
        updateIntersectionPreviewTransform();
    else
        scheduleIntersectionRender("stable render finished");
    if (!_interactivePreview) {
        if (auto surf = _surfWeak.lock()) {
            const vc::render::ChunkedPlaneSampler::Options options(_samplingMethod, 32);
            if (auto* plane = dynamic_cast<PlaneSurface*>(surf.get())) {
                const int startLevel = renderStartLevel(false);
                const cv::Vec3f vx = plane->basisX();
                const cv::Vec3f vy = plane->basisY();
                const cv::Vec3f n = plane->normal({0, 0, 0});
                const float halfW = static_cast<float>(_framebuffer.width()) * 0.5f / _scale;
                const float halfH = static_cast<float>(_framebuffer.height()) * 0.5f / _scale;
                const cv::Vec3f origin = vx * (_surfacePtrX - halfW)
                                       + vy * (_surfacePtrY - halfH)
                                       + plane->origin()
                                       + n * _zOff;
                prefetchPlaneHalo(origin, vx / _scale, vy / _scale, startLevel, options);
                prefetchPlaneNormalNeighbors(*plane, startLevel, options);
            } else {
                prefetchSurfaceHalo(*surf, renderStartLevel(true), options,
                                    _framebuffer.width(), _framebuffer.height());
            }
        }
    }
    _stableFramebuffer = _framebuffer.copy();
    _stableSurfX = result->surfacePtrX;
    _stableSurfY = result->surfacePtrY;
    _stableScale = result->scale;
    _stableFramebufferValid = !_stableFramebuffer.isNull();
    emit overlaysUpdated();
    _view->viewport()->update();
    updateStatusLabel();

    if (_renderPendingAfterWorker) {
        _renderPendingAfterWorker = false;
        scheduleRender("worker finished with pending render");
    }
    if (profile.enabled()) {
        profile.setDetails(std::format(
            "action=display serial={} worker_elapsed_ms={} framebuffer={}x{}",
            result->serial, result->renderFrameElapsedMs,
            _framebuffer.width(), _framebuffer.height()));
    }
}

void CChunkedVolumeViewer::renderVisible(bool force, const char* reason, std::source_location caller)
{
    ProfileScope profile("renderVisible", reason, caller);
    if (profile.enabled()) {
        profile.setDetails(std::format(
            "surf='{}' force={} pending={} interactive={} scale={:.4f} zOff={:.3f}",
            _surfName, force, _renderPending, _interactivePreview, _scale, _zOff));
    }
    if (_closing) {
        profile.setDetails("action=skip closing");
        return;
    }
    if (!force) {
        scheduleRender("renderVisible non-force", caller);
        profile.setDetails("action=schedule");
        return;
    }
    if (_renderTimer && _renderTimer->isActive())
        _renderTimer->stop();
    _renderPending = false;
    submitRender("renderVisible force", caller);
    updateStatusLabel();
    profile.setDetails("action=submit");
}

void CChunkedVolumeViewer::setVolumeWindow(float low, float high)
{
    if (_closing) {
        return;
    }
    const float clampedLow = std::clamp(low, 0.0f, 255.0f);
    float clampedHigh = std::clamp(high, 0.0f, 255.0f);
    if (clampedHigh <= clampedLow)
        clampedHigh = std::min(255.0f, clampedLow + 1.0f);
    if (std::abs(_windowLow - clampedLow) < 1e-6f &&
        std::abs(_windowHigh - clampedHigh) < 1e-6f)
        return;
    _windowLow = clampedLow;
    _windowHigh = clampedHigh;
    scheduleRender("volume window changed");
}

void CChunkedVolumeViewer::setOverlayVolume(std::shared_ptr<Volume> volume)
{
    if (_closing) {
        return;
    }
    if (_overlayChunkCbId != 0 && _overlayChunkArray) {
        _overlayChunkArray->removeChunkReadyListener(_overlayChunkCbId);
        _overlayChunkCbId = 0;
    }
    _overlayVolume = std::move(volume);
    _overlayChunkArray.reset();
    if (_overlayVolume) {
        try {
            _overlayChunkArray = sharedChunkCacheForVolume(_overlayVolume, streamingCacheCapacityBytes(_state), _state);
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
                    if (!volume || guard->_overlayVolume != volume || guard->_closing)
                        return;
                    if (guard->_interactivePreview) {
                        if (guard->_settleRenderTimer)
                            guard->_settleRenderTimer->start(kChunkReadyActiveDelayMs);
                        return;
                    }
                    guard->scheduleRender("overlay chunk ready");
                }, Qt::QueuedConnection);
            });
        }
    }
    scheduleRender("overlay volume changed");
}

void CChunkedVolumeViewer::setOverlayOpacity(float opacity)
{
    if (_closing) {
        return;
    }
    _overlayOpacity = std::clamp(opacity, 0.0f, 1.0f);
    scheduleRender("overlay opacity changed");
}

void CChunkedVolumeViewer::setOverlayColormap(const std::string& colormapId)
{
    if (_closing) {
        return;
    }
    _overlayColormapId = colormapId;
    scheduleRender("overlay colormap changed");
}

void CChunkedVolumeViewer::setOverlayThreshold(float threshold)
{
    if (_closing) {
        return;
    }
    setOverlayWindow(std::max(threshold, 0.0f), _overlayWindowHigh);
}

void CChunkedVolumeViewer::setOverlayWindow(float low, float high)
{
    if (_closing) {
        return;
    }
    _overlayWindowLow = std::clamp(low, 0.0f, 255.0f);
    _overlayWindowHigh = std::clamp(high, _overlayWindowLow + 1.0f, 255.0f);
    scheduleRender("overlay window changed");
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
    prefetchVisibleSurfaceChunks();
    if (shouldRefreshInteractivePreview())
        updateInteractivePreviewFromStableFrame(_surfacePtrX, _surfacePtrY, _scale);
    scheduleRender("pan");
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
    prefetchVisibleSurfaceChunks();
    if (shouldRefreshInteractivePreview())
        updateInteractivePreviewFromStableFrame(_surfacePtrX, _surfacePtrY, _scale);
    scheduleRender("zoom");
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
    prefetchVisibleSurfaceChunks();
    if (shouldRefreshInteractivePreview()) {
        _renderPending = false;
        submitRender("interactive view change preview");
    }
    scheduleRender("interactive view change final render");
    emit overlaysUpdated();
}

void CChunkedVolumeViewer::adjustSurfaceOffset(float delta)
{
    float maxZ = 10000.0f;
    if (_volume) {
        const auto [w, h, d] = _volume->shapeXyz();
        maxZ = static_cast<float>(std::max({w, h, d}));
    }
    _zOff = std::clamp(_zOff + delta, -maxZ, maxZ);
    _genCacheDirty = true;
    scheduleRender("surface offset changed");
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
    scheduleRender("surface offsets reset");
}

void CChunkedVolumeViewer::fitSurfaceInView()
{
    _surfacePtrX = 0.0f;
    _surfacePtrY = 0.0f;
    _scale = 0.5f;
    recalcPyramidLevel();
    _genCacheDirty = true;
    _stableFramebufferValid = false;
    scheduleRender("fit surface in view");
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
        try {
            cv::Vec3f ptr = quad->pointer();
            auto* patchIndex = _viewerManager ? _viewerManager->surfacePatchIndex() : nullptr;
            if (quad->pointTo(ptr, point, 4.0f, 100, patchIndex) >= 0.0f) {
                const cv::Vec3f loc = quad->loc(ptr);
                surfacePoint = {loc[0], loc[1]};
                haveSurfacePoint = true;
            }
        } catch (const std::exception& e) {
            Logger()->warn("Skipping focus projection for surface '{}': {}", quad->id, e.what());
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

    const float oldSurfacePtrX = _surfacePtrX;
    const float oldSurfacePtrY = _surfacePtrY;
    _surfacePtrX = point[0];
    _surfacePtrY = point[1];
    _genCacheDirty = true;
    if (forceRender) {
        _stableFramebufferValid = false;
        renderVisible(true, "center on surface point");
    } else {
        const double motionPx = std::hypot(double(_surfacePtrX - oldSurfacePtrX),
                                          double(_surfacePtrY - oldSurfacePtrY)) *
                                double(std::max(_scale, kMinScale));
        markInteractiveMotion(motionPx);
        if (shouldRefreshInteractivePreview())
            updateInteractivePreviewFromStableFrame(_surfacePtrX, _surfacePtrY, _scale);
        scheduleRender("center on surface point");
    }
    emit overlaysUpdated();
}

void CChunkedVolumeViewer::onZoom(int steps, QPointF scenePoint, Qt::KeyboardModifiers modifiers)
{
    auto surf = _surfWeak.lock();
    if (!surf)
        return;
    if (modifiers & Qt::ShiftModifier) {
        if (auto* plane = dynamic_cast<PlaneSurface*>(surf.get())) {
            const cv::Vec3f normal = plane->normal({0, 0, 0});
            if (std::isfinite(normal[0]) && std::isfinite(normal[1]) &&
                std::isfinite(normal[2]) && cv::norm(normal) > 0.0f) {
                const float delta = static_cast<float>(steps) * _zScrollSensitivity;
                auto shiftedPlane = std::make_shared<PlaneSurface>(*plane);
                shiftedPlane->setOrigin(plane->origin() + normal * (delta + _zOff));
                _zOff = 0.0f;
                _zOffWorldDir = {0, 0, 0};
                if (_state) {
                    _state->setSurface(_surfName, shiftedPlane, false, true);
                } else {
                    _defaultSurface = shiftedPlane;
                    _surfWeak = _defaultSurface;
                    updateContentBounds();
                    _genCacheDirty = true;
                    _stableFramebufferValid = false;
                    scheduleRender("plane slice mouse wheel");
                }
            }
        } else {
            _zOff += static_cast<float>(steps) * _zScrollSensitivity;
            _genCacheDirty = true;
            scheduleRender("z offset mouse wheel");
        }
    } else if (modifiers & Qt::ControlModifier) {
        emit sendSegmentationRadiusWheel(steps, scenePoint, sceneToVolume(scenePoint));
    } else {
        zoomStepsAt(steps > 0 ? 1 : (steps < 0 ? -1 : 0), scenePoint);
    }
}

void CChunkedVolumeViewer::onResized()
{
    if (_closing) {
        return;
    }
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
    _lastCursorVolumePos = cursorVolumePosition(scenePos);
    updateCursorCrosshair(scenePos);
    updateStatusLabel();
    if (_viewerManager) {
        _viewerManager->broadcastLinkedCursor(this, _lastCursorVolumePos);
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
    scheduleRender("pan released");
}

void CChunkedVolumeViewer::onVolumeClicked(QPointF scenePos, Qt::MouseButton button, Qt::KeyboardModifiers modifiers)
{
    auto surf = _surfWeak.lock();
    const auto cursorPos = cursorVolumePosition(scenePos);
    cv::Vec3f volumePos;
    if (cursorPos) {
        volumePos = *cursorPos;
    } else {
        volumePos = sceneToVolume(scenePos);
    }
    cv::Vec3f n(0, 0, 1);
    if (auto* plane = dynamic_cast<PlaneSurface*>(surf.get())) {
        n = plane->normal({0, 0, 0});
    } else if (surf) {
        const cv::Vec2f sp = sceneToSurface(scenePos);
        const cv::Vec3f surfaceNormal = surf->normal({0, 0, 0}, {sp[0], sp[1], 0.0f});
        if (std::isfinite(surfaceNormal[0]) &&
            std::isfinite(surfaceNormal[1]) &&
            std::isfinite(surfaceNormal[2])) {
            n = surfaceNormal;
        }
    }
    emit sendVolumeClicked(volumePos, n, surf.get(), button, modifiers);
}

void CChunkedVolumeViewer::onMousePress(QPointF scenePos, Qt::MouseButton button, Qt::KeyboardModifiers modifiers)
{
    _lastScenePos = scenePos;
    _lastCursorVolumePos = cursorVolumePosition(scenePos);
    updateCursorCrosshair(scenePos);
    updateStatusLabel();
    if (_bboxMode && _surfName == "segmentation" && button == Qt::LeftButton) {
        const cv::Vec2f sp = sceneToSurface(scenePos);
        _bboxStart = QPointF(sp[0], sp[1]);
        _activeBBoxSurfRect = QRectF(_bboxStart, QSizeF(0.0, 0.0));
        emit overlaysUpdated();
        return;
    }
    cv::Vec3f volumePos;
    if (_lastCursorVolumePos) {
        volumePos = *_lastCursorVolumePos;
    } else {
        volumePos = sceneToVolume(scenePos);
    }
    emit sendMousePressVolume(volumePos, {0, 0, 1}, button, modifiers, scenePos);
}

void CChunkedVolumeViewer::onMouseMove(QPointF scenePos, Qt::MouseButtons buttons, Qt::KeyboardModifiers modifiers)
{
    Q_UNUSED(modifiers);
    const bool reusedCursorSample = (_lastScenePos == scenePos && _lastCursorVolumePos.has_value());
    if (!reusedCursorSample) {
        _lastCursorVolumePos = cursorVolumePosition(scenePos);
    }
    _lastScenePos = scenePos;
    updateCursorCrosshair(scenePos);
    updateStatusLabel();
    if (_bboxMode && _activeBBoxSurfRect && (buttons & Qt::LeftButton)) {
        const cv::Vec2f sp = sceneToSurface(scenePos);
        _activeBBoxSurfRect = QRectF(_bboxStart, QPointF(sp[0], sp[1])).normalized();
        emit overlaysUpdated();
        return;
    }
    cv::Vec3f volumePos;
    if (_lastCursorVolumePos) {
        volumePos = *_lastCursorVolumePos;
    } else {
        volumePos = sceneToVolume(scenePos);
    }
    emit sendMouseMoveVolume(volumePos, buttons, modifiers, scenePos);
}

void CChunkedVolumeViewer::onMouseRelease(QPointF scenePos, Qt::MouseButton button, Qt::KeyboardModifiers modifiers)
{
    _lastScenePos = scenePos;
    _lastCursorVolumePos = cursorVolumePosition(scenePos);
    updateCursorCrosshair(scenePos);
    updateStatusLabel();
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
    cv::Vec3f volumePos;
    if (_lastCursorVolumePos) {
        volumePos = *_lastCursorVolumePos;
    } else {
        volumePos = sceneToVolume(scenePos);
    }
    emit sendMouseReleaseVolume(volumePos, button, modifiers, scenePos);
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
    const qreal vx = (surfX - _surfacePtrX) * _scale + vpCx;
    const qreal vy = (surfY - _surfacePtrY) * _scale + vpCy;
    return QPointF(vx, vy);
}

cv::Vec2f CChunkedVolumeViewer::sceneToSurface(const QPointF& scenePos) const
{
    if (_framebuffer.isNull() || _scale <= 0.0f)
        return {0, 0};
    const float vpCx = static_cast<float>(_framebuffer.width()) * 0.5f;
    const float vpCy = static_cast<float>(_framebuffer.height()) * 0.5f;
    return {(static_cast<float>(scenePos.x()) - vpCx) / _scale + _surfacePtrX,
            (static_cast<float>(scenePos.y()) - vpCy) / _scale + _surfacePtrY};
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
        try {
            cv::Vec3f ptr = quad->pointer();
            auto* patchIndex = _viewerManager ? _viewerManager->surfacePatchIndex() : nullptr;
            if (quad->pointTo(ptr, volPoint, 4.0f, 100, patchIndex) < 0.0f)
                return {};
            const cv::Vec3f loc = quad->loc(ptr);
            return surfaceToScene(loc[0], loc[1]);
        } catch (const std::exception& e) {
            Logger()->warn("Skipping volume-to-surface projection for surface '{}': {}", quad->id, e.what());
            return {};
        }
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

std::optional<cv::Vec3f> CChunkedVolumeViewer::cursorVolumePosition(const QPointF& scenePos) const
{
    auto surf = _surfWeak.lock();
    if (!surf)
        return std::nullopt;

    cv::Vec3f p = sceneToVolume(scenePos);
    if (auto* plane = dynamic_cast<PlaneSurface*>(surf.get())) {
        p += plane->normal({0, 0, 0}) * _zOff;
    } else if (_zOff != 0.0f) {
        const cv::Vec2f sp = sceneToSurface(scenePos);
        const cv::Vec3f n = surf->normal({0, 0, 0}, {sp[0], sp[1], 0.0f});
        if (std::isfinite(n[0]) && std::isfinite(n[1]) && std::isfinite(n[2]))
            p += n * _zOff;
    }
    return p;
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

void CChunkedVolumeViewer::invalidateIntersect(const std::string& name)
{
    if (_closing) {
        return;
    }
    if (_segmentationEditActive && name == "segmentation" &&
        dynamic_cast<PlaneSurface*>(currentSurface())) {
        _lastIntersectFp = {};
        _intersectionGeometryCache = {};
        if (_deferSegmentationIntersections) {
            _deferredSegmentationIntersectionsDirty = true;
            return;
        }
        scheduleIntersectionRender("segmentation intersection invalidated during edit");
        return;
    }

    clearIntersectionItems();
    _lastIntersectFp = {};
    _intersectionGeometryCache = {};
    _flattenedIntersectionCache = {};
    _flattenedIntersectionDirtyCells.reset();
}

void CChunkedVolumeViewer::invalidateIntersectRegion(const std::string& name,
                                                     const cv::Rect& changedCells)
{
    if (changedCells.empty()) {
        return;
    }
    if (name.empty() || name != "segmentation") {
        invalidateIntersect(name);
        return;
    }
    if (_surfName != "segmentation") {
        invalidateIntersect(name);
        return;
    }

    _flattenedIntersectionDirtyCells =
        _flattenedIntersectionDirtyCells
            ? rectContains(*_flattenedIntersectionDirtyCells, changedCells)
                  ? *_flattenedIntersectionDirtyCells
                  : (*_flattenedIntersectionDirtyCells | changedCells)
            : changedCells;
    _lastIntersectFp = {};
    scheduleIntersectionRender("segmentation intersection region invalidated");
}

void CChunkedVolumeViewer::setIntersects(const std::set<std::string>& names)
{
    if (_closing || _intersectTgts == names) {
        return;
    }
    _intersectTgts = names;
    invalidateIntersect();
    renderIntersections("setIntersects");
}

void CChunkedVolumeViewer::setIntersectionOpacity(float v)
{
    if (_closing) {
        return;
    }
    const float clamped = std::clamp(v, 0.0f, 1.0f);
    if (std::abs(_intersectionOpacity - clamped) < 1e-6f) {
        return;
    }
    _intersectionOpacity = clamped;
    renderIntersections("setIntersectionOpacity");
}

void CChunkedVolumeViewer::setIntersectionThickness(float v)
{
    if (_closing) {
        return;
    }
    const float clamped = std::max(0.0f, v);
    if (std::abs(_intersectionThickness - clamped) < 1e-6f) {
        return;
    }
    _intersectionThickness = clamped;
    renderIntersections("setIntersectionThickness");
}

void CChunkedVolumeViewer::setSurfacePatchSamplingStride(int s)
{
    if (_closing) {
        return;
    }
    const int stride = std::max(1, s);
    if (_surfacePatchSamplingStride == stride) {
        return;
    }
    _surfacePatchSamplingStride = stride;
    invalidateIntersect();
    renderIntersections("setSurfacePatchSamplingStride");
}

void CChunkedVolumeViewer::clearIntersectionItems()
{
    for (auto* item : _intersectionItems) {
        if (item && item->scene())
            _scene->removeItem(item);
        delete item;
    }
    _intersectionItems.clear();
    _flattenedIntersectionCache.tileItems.clear();
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

void CChunkedVolumeViewer::renderFlattenedIntersections(const std::shared_ptr<Surface>& surf,
                                                        const char* reason,
                                                        std::source_location caller)
{
    ProfileScope profile("renderFlattenedIntersections", reason, caller);
    if (profile.enabled()) {
        profile.setDetails(std::format("surf='{}' targets={}", _surfName, _intersectTgts.size()));
    }
    auto activeSeg = std::dynamic_pointer_cast<QuadSurface>(surf);
    if (!activeSeg || !_state || _state->surface("segmentation") != activeSeg) {
        invalidateIntersect();
        _lastIntersectFp = {};
        profile.setDetails("action=skip inactive_segmentation");
        return;
    }

    auto* patchIndex = _viewerManager ? _viewerManager->surfacePatchIndex() : nullptr;
    if (!patchIndex || patchIndex->empty()) {
        invalidateIntersect();
        _lastIntersectFp = {};
        profile.setDetails("action=skip empty_patch_index");
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
        profile.setDetails("action=skip no_planes");
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
    fp.indexSamplingStride = 1;
    fp.patchCount = patchIndex->patchCount();
    fp.surfaceCount = patchIndex->surfaceCount();
    fp.activeSegHash = std::hash<const void*>{}(activeSeg.get());
    const uint64_t activeGeneration = patchIndex->generation(activeSeg);
    fp.targetGenerationHash = 0;

    // Flattened intersections are built in surface-view scene coordinates.
    // Pan/zoom can reuse the same paths by transforming the existing items.
    fp.cameraHash = 0;
    fp.valid = true;
    if (_lastIntersectFp == fp && !_intersectionItems.empty() &&
        !_flattenedIntersectionDirtyCells) {
        updateIntersectionPreviewTransform();
        if (profile.enabled()) {
            profile.setDetails(std::format("action=cache_hit planes={} items={}",
                                           planes.size(), _intersectionItems.size()));
        }
        return;
    }

    Rect3D allBounds{cv::Vec3f(0, 0, 0), cv::Vec3f(1, 1, 1)};
    if (_volume) {
        auto [w, h, d] = _volume->shapeXyz();
        allBounds.high = {static_cast<float>(w),
                          static_cast<float>(h),
                          static_cast<float>(d)};
    }

    const float clipTol = 1e-4f;
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

    const cv::Mat_<cv::Vec3f>* points = activeSeg->rawPointsPtr();
    if (!points || points->cols < 2 || points->rows < 2) {
        invalidateIntersect();
        _lastIntersectFp = {};
        profile.setDetails("action=skip no_points");
        return;
    }

    auto cellBounds = [&]() {
        return cv::Rect(0, 0, points->cols - 1, points->rows - 1);
    };
    const int stride = 1;
    const bool cacheCompatible =
        _flattenedIntersectionCache.valid &&
        _flattenedIntersectionCache.surface == activeSeg.get() &&
        _flattenedIntersectionCache.planesHash == planesHash &&
        _flattenedIntersectionCache.indexSamplingStride == stride;
    const bool displayOnlyRefresh =
        cacheCompatible &&
        !_flattenedIntersectionDirtyCells;
    const bool needsFullRebuild = !cacheCompatible;
    std::unordered_map<std::uint64_t, std::unordered_set<int>> dirtyTilePlanes;

    auto removeFlattenedTilePlaneItem = [&](std::uint64_t tileKey, int planeIndex) {
        auto it = _flattenedIntersectionCache.tileItems.find(tileKey);
        if (it == _flattenedIntersectionCache.tileItems.end() ||
            planeIndex < 0 || static_cast<std::size_t>(planeIndex) >= it->second.size()) {
            return;
        }
        auto*& item = it->second[static_cast<std::size_t>(planeIndex)];
        if (!item) {
            return;
        }
        if (item->scene()) {
            _scene->removeItem(item);
        }
        auto vecIt = std::find(_intersectionItems.begin(), _intersectionItems.end(), item);
        if (vecIt != _intersectionItems.end()) {
            _intersectionItems.erase(vecIt);
        }
        delete item;
        item = nullptr;
    };

    auto lineMatchesPlane = [](const FlattenedIntersectionLine& line, int planeIndex) {
        return line.planeIndex == planeIndex;
    };
    auto markChangedCellPlanes = [&](std::uint64_t cellKey,
                                     const std::vector<FlattenedIntersectionLine>& newLines) {
        const int col = int(std::uint32_t(cellKey));
        const int row = int(std::uint32_t(cellKey >> 32));
        const std::uint64_t tileKey = surfaceCellTileKey(col, row);
        const auto oldIt = _flattenedIntersectionCache.cellLines.find(cellKey);
        const auto* oldLines = oldIt == _flattenedIntersectionCache.cellLines.end()
            ? nullptr
            : &oldIt->second;

        for (std::size_t planeIdx = 0; planeIdx < planes.size(); ++planeIdx) {
            const int idx = static_cast<int>(planeIdx);
            std::vector<const FlattenedIntersectionLine*> oldPlaneLines;
            std::vector<const FlattenedIntersectionLine*> newPlaneLines;
            if (oldLines) {
                for (const auto& line : *oldLines) {
                    if (lineMatchesPlane(line, idx)) {
                        oldPlaneLines.push_back(&line);
                    }
                }
            }
            for (const auto& line : newLines) {
                if (lineMatchesPlane(line, idx)) {
                    newPlaneLines.push_back(&line);
                }
            }

            bool changed = oldPlaneLines.size() != newPlaneLines.size();
            if (!changed) {
                for (std::size_t i = 0; i < oldPlaneLines.size(); ++i) {
                    const auto& oldLine = *oldPlaneLines[i];
                    const auto& newLine = *newPlaneLines[i];
                    if (oldLine.a != newLine.a || oldLine.b != newLine.b) {
                        changed = true;
                        break;
                    }
                }
            }
            if (changed) {
                dirtyTilePlanes[tileKey].insert(idx);
            }
        }
    };

    auto rebuildFlattenedCells = [&](const cv::Rect& requestedCells) {
        const cv::Rect cells = requestedCells & cellBounds();
        if (cells.empty()) {
            return;
        }

        const cv::Vec3f center = activeSeg->center();
        const cv::Vec2f gridScale = activeSeg->scale();
        const float cx = center[0] * gridScale[0];
        const float cy = center[1] * gridScale[1];

        const int rowStart = (cells.y / stride) * stride;
        const int colStart = (cells.x / stride) * stride;
        const int rowEnd = cells.y + cells.height;
        const int colEnd = cells.x + cells.width;
        for (int row = rowStart; row < rowEnd && row < points->rows - 1; row += stride) {
            for (int col = colStart; col < colEnd && col < points->cols - 1; col += stride) {
                const std::uint64_t key = surfaceTileKey(col, row);
                std::vector<FlattenedIntersectionLine> lines;

                const int strideX = std::min(stride, points->cols - 1 - col);
                const int strideY = std::min(stride, points->rows - 1 - row);
                if (strideX <= 0 || strideY <= 0) {
                    markChangedCellPlanes(key, lines);
                    _flattenedIntersectionCache.cellLines.erase(key);
                    continue;
                }

                const std::array<cv::Vec3f, 4> corners = {
                    (*points)(row, col),
                    (*points)(row, col + strideX),
                    (*points)(row + strideY, col + strideX),
                    (*points)(row + strideY, col),
                };
                if (!validSurfacePoint(corners[0]) || !validSurfacePoint(corners[1]) ||
                    !validSurfacePoint(corners[2]) || !validSurfacePoint(corners[3])) {
                    markChangedCellPlanes(key, lines);
                    _flattenedIntersectionCache.cellLines.erase(key);
                    continue;
                }

                const float baseX = static_cast<float>(col);
                const float baseY = static_cast<float>(row);
                const std::array<cv::Vec3f, 4> params = {
                    cv::Vec3f(baseX - cx, baseY - cy, 0.0f),
                    cv::Vec3f(baseX + float(strideX) - cx, baseY - cy, 0.0f),
                    cv::Vec3f(baseX + float(strideX) - cx, baseY + float(strideY) - cy, 0.0f),
                    cv::Vec3f(baseX - cx, baseY + float(strideY) - cy, 0.0f),
                };

                for (int triIdx = 0; triIdx < 2; ++triIdx) {
                    SurfacePatchIndex::TriangleCandidate tri;
                    tri.surface = activeSeg;
                    tri.i = col;
                    tri.j = row;
                    tri.triangleIndex = triIdx;
                    if (triIdx == 0) {
                        tri.world = {corners[0], corners[1], corners[3]};
                        tri.surfaceParams = {params[0], params[1], params[3]};
                    } else {
                        tri.world = {corners[1], corners[2], corners[3]};
                        tri.surfaceParams = {params[1], params[2], params[3]};
                    }

                    if ((tri.world[0][0] < allBounds.low[0] && tri.world[1][0] < allBounds.low[0] && tri.world[2][0] < allBounds.low[0]) ||
                        (tri.world[0][0] > allBounds.high[0] && tri.world[1][0] > allBounds.high[0] && tri.world[2][0] > allBounds.high[0]) ||
                        (tri.world[0][1] < allBounds.low[1] && tri.world[1][1] < allBounds.low[1] && tri.world[2][1] < allBounds.low[1]) ||
                        (tri.world[0][1] > allBounds.high[1] && tri.world[1][1] > allBounds.high[1] && tri.world[2][1] > allBounds.high[1]) ||
                        (tri.world[0][2] < allBounds.low[2] && tri.world[1][2] < allBounds.low[2] && tri.world[2][2] < allBounds.low[2]) ||
                        (tri.world[0][2] > allBounds.high[2] && tri.world[1][2] > allBounds.high[2] && tri.world[2][2] > allBounds.high[2])) {
                        continue;
                    }

                    for (size_t idx = 0; idx < planes.size(); ++idx) {
                        auto seg = SurfacePatchIndex::clipTriangleToPlane(
                            tri, *planes[idx].plane, clipTol);
                        if (!seg) {
                            continue;
                        }
                        const cv::Vec3f a = activeSeg->loc(seg->surfaceParams[0]);
                        const cv::Vec3f b = activeSeg->loc(seg->surfaceParams[1]);
                        const QPointF pa = surfaceToScene(a[0], a[1]);
                        const QPointF pb = surfaceToScene(b[0], b[1]);
                        if (!isFinitePoint(pa) || !isFinitePoint(pb)) {
                            continue;
                        }
                        lines.push_back(FlattenedIntersectionLine{static_cast<int>(idx), pa, pb});
                    }
                }

                if (lines.empty()) {
                    markChangedCellPlanes(key, lines);
                    _flattenedIntersectionCache.cellLines.erase(key);
                } else {
                    markChangedCellPlanes(key, lines);
                    _flattenedIntersectionCache.cellLines[key] = std::move(lines);
                }
            }
        }
    };

    if (needsFullRebuild) {
        clearIntersectionItems();
        _flattenedIntersectionCache = {};
        _flattenedIntersectionCache.surface = activeSeg.get();
        _flattenedIntersectionCache.planesHash = planesHash;
        _flattenedIntersectionCache.indexSamplingStride = stride;
        _flattenedIntersectionCache.cellLines.reserve(
            std::size_t(points->rows / stride + 1) * std::size_t(points->cols / stride + 1) / 8 + 1024);
        rebuildFlattenedCells(cellBounds());
        _flattenedIntersectionCache.valid = true;
    } else if (_flattenedIntersectionDirtyCells) {
        rebuildFlattenedCells(*_flattenedIntersectionDirtyCells);
    } else if (!cacheCompatible) {
        invalidateIntersect();
        _lastIntersectFp = {};
        profile.setDetails("action=skip incompatible_cache");
        return;
    }
    _flattenedIntersectionCache.generation = activeGeneration;
    _flattenedIntersectionDirtyCells.reset();

    _lastIntersectFp = fp;
    _intersectionGeometryCache = {};

    const bool rebuildAllTileItems =
        needsFullRebuild ||
        displayOnlyRefresh ||
        _flattenedIntersectionCache.tileItems.empty();
    if (rebuildAllTileItems) {
        dirtyTilePlanes.clear();
        for (const auto& [cellKey, _] : _flattenedIntersectionCache.cellLines) {
            const int col = int(std::uint32_t(cellKey));
            const int row = int(std::uint32_t(cellKey >> 32));
            const std::uint64_t tileKey = surfaceCellTileKey(col, row);
            for (const auto& line : _) {
                dirtyTilePlanes[tileKey].insert(line.planeIndex);
            }
        }
    }

    if (dirtyTilePlanes.empty()) {
        _intersectionItemsCamSurfX = _camSurfX;
        _intersectionItemsCamSurfY = _camSurfY;
        _intersectionItemsCamScale = _camScale;
        _intersectionItemsHaveCamera = !_intersectionItems.empty();
        _view->viewport()->update();
        if (profile.enabled()) {
            profile.setDetails(std::format(
                "action=no_dirty_tiles planes={} cells={} items={}",
                planes.size(), _flattenedIntersectionCache.cellLines.size(),
                _intersectionItems.size()));
        }
        return;
    }

    std::unordered_map<std::uint64_t, std::vector<QPainterPath>> tilePaths;
    tilePaths.reserve(dirtyTilePlanes.size());
    for (const auto& [tileKey, _] : dirtyTilePlanes) {
        tilePaths.emplace(tileKey, std::vector<QPainterPath>(planes.size()));
    }

    for (const auto& [cellKey, lines] : _flattenedIntersectionCache.cellLines) {
        const int col = int(std::uint32_t(cellKey));
        const int row = int(std::uint32_t(cellKey >> 32));
        const std::uint64_t tileKey = surfaceCellTileKey(col, row);
        auto pathsIt = tilePaths.find(tileKey);
        if (pathsIt == tilePaths.end()) {
            continue;
        }
        auto& paths = pathsIt->second;
        for (const auto& line : lines) {
            if (line.planeIndex < 0 || static_cast<size_t>(line.planeIndex) >= paths.size()) {
                continue;
            }
            const auto dirtyIt = dirtyTilePlanes.find(tileKey);
            if (dirtyIt == dirtyTilePlanes.end() ||
                dirtyIt->second.count(line.planeIndex) == 0) {
                continue;
            }
            paths[line.planeIndex].moveTo(line.a);
            paths[line.planeIndex].lineTo(line.b);
        }
    }

    for (auto& [tileKey, paths] : tilePaths) {
        auto& tileItems = _flattenedIntersectionCache.tileItems[tileKey];
        if (tileItems.size() < paths.size()) {
            tileItems.resize(paths.size(), nullptr);
        }
        const auto dirtyIt = dirtyTilePlanes.find(tileKey);
        for (std::size_t idx = 0; idx < paths.size(); ++idx) {
            if (dirtyIt == dirtyTilePlanes.end() ||
                dirtyIt->second.count(static_cast<int>(idx)) == 0) {
                continue;
            }
            QColor color = planes[idx].color;
            color.setAlphaF(opacity);
            QPen pen(color);
            pen.setWidthF(static_cast<qreal>(penWidth));
            pen.setCapStyle(Qt::RoundCap);
            pen.setJoinStyle(Qt::RoundJoin);
            pen.setCosmetic(true);
            if (paths[idx].isEmpty()) {
                removeFlattenedTilePlaneItem(tileKey, static_cast<int>(idx));
                continue;
            }
            auto*& item = tileItems[idx];
            if (!item) {
                item = new QGraphicsPathItem();
                item->setBrush(Qt::NoBrush);
                item->setZValue(kActiveIntersectionZ);
                item->setAcceptedMouseButtons(Qt::NoButton);
                _scene->addItem(item);
                _intersectionItems.push_back(item);
            }
            item->setTransform(QTransform());
            item->setPath(paths[idx]);
            item->setPen(pen);
        }
    }

    for (auto* item : _intersectionItems) {
        if (item) {
            item->setTransform(QTransform());
        }
    }

    if (_intersectionItems.empty()) {
        _intersectionItemsHaveCamera = false;
        _view->viewport()->update();
        if (profile.enabled()) {
            profile.setDetails(std::format(
                "action=empty_result planes={} cells={} dirtyTiles={}",
                planes.size(), _flattenedIntersectionCache.cellLines.size(),
                dirtyTilePlanes.size()));
        }
        return;
    }

    _intersectionItemsCamSurfX = _camSurfX;
    _intersectionItemsCamSurfY = _camSurfY;
    _intersectionItemsCamScale = _camScale;
    _intersectionItemsHaveCamera = !_intersectionItems.empty();
    _view->viewport()->update();
    if (profile.enabled()) {
        profile.setDetails(std::format(
            "action=rendered planes={} cells={} dirtyTiles={} items={} fullRebuild={} dirtyCells={}",
            planes.size(), _flattenedIntersectionCache.cellLines.size(),
            dirtyTilePlanes.size(), _intersectionItems.size(), needsFullRebuild,
            bool(_flattenedIntersectionDirtyCells)));
    }
}

void CChunkedVolumeViewer::renderIntersections(const char* reason, std::source_location caller)
{
    ProfileScope profile("renderIntersections", reason, caller);
    if (profile.enabled()) {
        profile.setDetails(std::format(
            "surf='{}' targets={} viewport={} scale={:.4f} zOff={:.3f}",
            _surfName, _intersectTgts.size(),
            _view ? profileRectF(_view->mapToScene(_view->viewport()->rect()).boundingRect())
                  : std::string("[]"),
            _scale, _zOff));
    }
    if (_closing) {
        profile.setDetails("action=skip closing");
        return;
    }

    auto surf = _surfWeak.lock();
    auto* plane = dynamic_cast<PlaneSurface*>(surf.get());
    if (!surf || !_state || !_viewerManager || !_scene || !_view) {
        invalidateIntersect();
        _lastIntersectFp = {};
        profile.setDetails("action=skip missing_input");
        return;
    }
    if (!plane) {
        try {
            renderFlattenedIntersections(surf, reason, caller);
        } catch (const std::exception& e) {
            Logger()->warn("Skipping flattened intersections for surface '{}': {}",
                           surf ? surf->id : std::string{}, e.what());
            invalidateIntersect();
            _lastIntersectFp = {};
        }
        profile.setDetails("action=delegated_flattened");
        return;
    }

    auto* patchIndex = _viewerManager->surfacePatchIndex();
    if (!patchIndex || patchIndex->empty()) {
        invalidateIntersect();
        _lastIntersectFp = {};
        profile.setDetails("action=skip empty_patch_index");
        return;
    }

    auto activeSeg = std::dynamic_pointer_cast<QuadSurface>(_state->surface("segmentation"));
    std::unordered_set<SurfacePatchIndex::SurfacePtr> targets;
    auto addTarget = [&](const std::string& name) {
        if (auto quad = std::dynamic_pointer_cast<QuadSurface>(_state->surface(name))) {
            if (activeSeg && quad != activeSeg && !activeSeg->id.empty() &&
                quad->id == activeSeg->id) {
                return;
            }
            targets.insert(std::move(quad));
        }
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
        profile.setDetails("action=skip no_targets");
        return;
    }

    QRectF sceneRect = _view->mapToScene(_view->viewport()->rect()).boundingRect();
    if (!sceneRect.isValid()) {
        invalidateIntersect();
        _lastIntersectFp = {};
        profile.setDetails("action=skip invalid_scene_rect");
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
    fp.cameraHash = (std::hash<int>{}(_framebuffer.width()) + 0x9e3779b9u) ^
                    (std::hash<int>{}(_framebuffer.height()) << 1);
    fp.valid = true;
    if (_lastIntersectFp == fp && !_intersectionItems.empty()) {
        updateIntersectionPreviewTransform();
        if (profile.enabled()) {
            profile.setDetails(std::format(
                "action=cache_hit targetSurfaces={} items={} roi={}",
                targets.size(), _intersectionItems.size(), profileRect(planeRoi)));
        }
        return;
    }

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

    auto samePlaneRenderInputsExceptRoi = [](const IntersectFingerprint& a,
                                             const IntersectFingerprint& b) {
        return a.valid && b.valid &&
               a.planeOriginQ == b.planeOriginQ &&
               a.planeNormalQ == b.planeNormalQ &&
               a.planeBasisXQ == b.planeBasisXQ &&
               a.planeBasisYQ == b.planeBasisYQ &&
               a.opacityQ == b.opacityQ &&
               a.thicknessQ == b.thicknessQ &&
               a.indexSamplingStride == b.indexSamplingStride &&
               a.patchCount == b.patchCount &&
               a.surfaceCount == b.surfaceCount &&
               a.targetHash == b.targetHash &&
               a.targetGenerationHash == b.targetGenerationHash &&
               a.activeSegHash == b.activeSegHash &&
               a.highlightedSurfaceHash == b.highlightedSurfaceHash &&
               a.cameraHash == b.cameraHash;
    };
    if (geometryCacheValid && !_intersectionItems.empty() &&
        samePlaneRenderInputsExceptRoi(_lastIntersectFp, fp)) {
        _lastIntersectFp = fp;
        updateIntersectionPreviewTransform();
        if (profile.enabled()) {
            profile.setDetails(std::format(
                "action=camera_cache_hit targetSurfaces={} items={} roi={} cacheRoi={}",
                targets.size(), _intersectionItems.size(), profileRect(planeRoi),
                profileRect(_intersectionGeometryCache.roi)));
        }
        return;
    }

    _lastIntersectFp = fp;

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
        try {
            _intersectionGeometryCache.intersections =
                patchIndex->computePlaneIntersections(*plane, cacheRoi, targets);
        } catch (const std::exception& e) {
            Logger()->warn("Skipping plane intersections: {}", e.what());
            invalidateIntersect();
            _lastIntersectFp = {};
            _intersectionGeometryCache = {};
            return;
        }
        _intersectionGeometryCache.valid = true;
    }

    const auto& intersections = _intersectionGeometryCache.intersections;
    if (intersections.empty()) {
        clearIntersectionItems();
        if (profile.enabled()) {
            profile.setDetails(std::format(
                "action=empty_result targetSurfaces={} roi={} geometryCacheHit={}",
                targets.size(), profileRect(planeRoi), geometryCacheValid));
        }
        return;
    }

    std::size_t segmentCount = 0;
    for (const auto& [target, segments] : intersections) {
        (void)target;
        segmentCount += segments.size();
    }

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
    auto* approvalOverlay = _viewerManager ? _viewerManager->segmentationOverlay() : nullptr;
    const bool showApprovalMaskIntersections =
        approvalOverlay && activeSeg && approvalOverlay->hasApprovalMaskData();
    auto paramToApprovalGrid = [&](const cv::Vec3f& param) {
        if (!activeSeg) {
            return QPointF();
        }
        const cv::Vec2f grid = activeSeg->ptrToGrid(param);
        return QPointF(grid[0], grid[1]);
    };
    auto addApprovalMaskIntersection = [&](const SurfacePatchIndex::TriangleSegment& seg,
                                           float renderedOpacity,
                                           float renderedPenWidth) {
        if (!showApprovalMaskIntersections) {
            return;
        }

        const QPointF gridA = paramToApprovalGrid(seg.surfaceParams[0]);
        const QPointF gridB = paramToApprovalGrid(seg.surfaceParams[1]);
        if (!isFinitePoint(gridA) || !isFinitePoint(gridB)) {
            return;
        }

        const int steps = std::max(1, int(std::ceil(std::max(std::abs(gridB.x() - gridA.x()),
                                                             std::abs(gridB.y() - gridA.y())))));
        const float opacity = std::clamp(renderedOpacity * kApprovalPlaneIntersectionScale,
                                         0.0f, 1.0f);
        const float penWidth = std::max(0.0f, renderedPenWidth) *
                               kApprovalPlaneIntersectionScale;

        for (int step = 0; step < steps; ++step) {
            const float t0 = static_cast<float>(step) / static_cast<float>(steps);
            const float t1 = static_cast<float>(step + 1) / static_cast<float>(steps);
            const float tm = (t0 + t1) * 0.5f;

            const float row = static_cast<float>(gridA.y() + (gridB.y() - gridA.y()) * tm);
            const float col = static_cast<float>(gridA.x() + (gridB.x() - gridA.x()) * tm);
            int approvalStatus = 0;
            if (approvalOverlay->queryApprovalBilinear(row, col, &approvalStatus) <= 0.0f ||
                approvalStatus == 0) {
                continue;
            }

            QColor approvalColor = approvalOverlay->queryApprovalColor(
                static_cast<int>(std::round(row)),
                static_cast<int>(std::round(col)));
            if (!approvalColor.isValid()) {
                approvalColor = QColor(0, 255, 0);
            }
            approvalColor.setAlphaF(opacity);
            if (approvalColor.alpha() <= 0) {
                continue;
            }

            const cv::Vec3f world0 = seg.world[0] + (seg.world[1] - seg.world[0]) * t0;
            const cv::Vec3f world1 = seg.world[0] + (seg.world[1] - seg.world[0]) * t1;
            const QPointF a = planeToScene(world0);
            const QPointF b = planeToScene(world1);
            if (!isFinitePoint(a) || !isFinitePoint(b)) {
                continue;
            }

            const IntersectionStyle approvalStyle{
                approvalColor.rgba(),
                kActiveIntersectionZ + 1,
                int(std::lround(penWidth * 1000.0f)),
            };
            QPainterPath& approvalPath = groupedPaths[approvalStyle];
            groupedColors[approvalStyle] = approvalColor;
            approvalPath.moveTo(a);
            approvalPath.lineTo(b);
        }
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
        groupedColors[style] = baseColor;
        for (const auto& seg : segments) {
            QPointF a = planeToScene(seg.world[0]);
            QPointF b = planeToScene(seg.world[1]);
            if (!isFinitePoint(a) || !isFinitePoint(b))
                continue;
            groupedPaths[style].moveTo(a);
            groupedPaths[style].lineTo(b);
            if (target == activeSeg) {
                addApprovalMaskIntersection(seg, opacity, penWidth);
            }
        }
    }

    std::size_t itemIndex = 0;
    _intersectionItems.reserve(std::max(_intersectionItems.size(), groupedPaths.size()));
    for (const auto& [style, path] : groupedPaths) {
        if (path.isEmpty())
            continue;
        QGraphicsPathItem* item = nullptr;
        if (itemIndex < _intersectionItems.size()) {
            item = dynamic_cast<QGraphicsPathItem*>(_intersectionItems[itemIndex]);
        }
        if (!item) {
            item = new QGraphicsPathItem();
            item->setBrush(Qt::NoBrush);
            item->setAcceptedMouseButtons(Qt::NoButton);
            _scene->addItem(item);
            if (itemIndex < _intersectionItems.size()) {
                if (_intersectionItems[itemIndex] && _intersectionItems[itemIndex]->scene()) {
                    _scene->removeItem(_intersectionItems[itemIndex]);
                }
                delete _intersectionItems[itemIndex];
                _intersectionItems[itemIndex] = item;
            } else {
                _intersectionItems.push_back(item);
            }
        }
        QPen pen(groupedColors[style]);
        pen.setWidthF(static_cast<qreal>(style.widthQ) / 1000.0);
        pen.setCapStyle(Qt::RoundCap);
        pen.setJoinStyle(Qt::RoundJoin);
        pen.setCosmetic(true);
        item->setTransform(QTransform());
        item->setPath(path);
        item->setPen(pen);
        item->setZValue(style.z);
        ++itemIndex;
    }
    while (_intersectionItems.size() > itemIndex) {
        auto* item = _intersectionItems.back();
        _intersectionItems.pop_back();
        if (item && item->scene()) {
            _scene->removeItem(item);
        }
        delete item;
    }
    _intersectionItemsCamSurfX = _camSurfX;
    _intersectionItemsCamSurfY = _camSurfY;
    _intersectionItemsCamScale = _camScale;
    _intersectionItemsHaveCamera = !_intersectionItems.empty();

    _view->viewport()->update();
    if (profile.enabled()) {
        profile.setDetails(std::format(
            "action=rendered targetSurfaces={} intersectingSurfaces={} segments={} groupedPaths={} items={} roi={} cacheRoi={} geometryCacheHit={}",
            targets.size(), intersections.size(), segmentCount,
            groupedPaths.size(), _intersectionItems.size(), profileRect(planeRoi),
            profileRect(_intersectionGeometryCache.roi), geometryCacheValid));
    }
}

void CChunkedVolumeViewer::setHighlightedSurfaceIds(const std::vector<std::string>& ids)
{
    if (_closing) {
        return;
    }
    std::unordered_set<std::string> next;
    next.reserve(ids.size());
    for (const auto& id : ids) {
        next.insert(id);
    }
    if (_highlightedSurfaceIds == next) {
        return;
    }
    _highlightedSurfaceIds.clear();
    _highlightedSurfaceIds = std::move(next);
    renderIntersections("highlighted surface ids changed");
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
    if (!_statsBar)
        return;

    QStringList items;
    items << QString("L%1").arg(_dsScaleIdx);
    items << QString("scale %1").arg(_scale, 0, 'f', 2);
    items << QString("%1x%2").arg(_framebuffer.width()).arg(_framebuffer.height());

    if ((_compositeSettings.enabled || _compositeSettings.planeEnabled) && streamingCompositeUnsupported()) {
        items << QString("composite unsupported: %1").arg(QString::fromStdString(_compositeSettings.params.method));
    } else if (_compositeSettings.enabled || _compositeSettings.planeEnabled) {
        items << QString("composite %1").arg(QString::fromStdString(_compositeSettings.params.method));
    }

    if (_chunkArray) {
        const auto stats = _chunkArray->stats();
        items << QString("RAM %1/%2 GB")
            .arg(formatGigabytes(stats.decodedBytes))
            .arg(formatGigabytes(stats.decodedByteCapacity));
        items << QString("disk %1").arg(formatByteSize(stats.persistentCacheBytes));
        if (stats.remoteFetchesInFlight > 0) {
            items << QString("downloading %1 @ %2")
                .arg(stats.remoteFetchesInFlight)
                .arg(formatMegabytesPerSecond(stats.remoteDownloadBytesPerSecond));
        }
    }

    auto surf = _surfWeak.lock();
    if (_lastCursorVolumePos)
        items << formatWholeVolumePosition(*_lastCursorVolumePos);
    if (dynamic_cast<QuadSurface*>(surf.get())) {
        items << QString("normal offset %1").arg(_zOff, 0, 'f', 1);
        if (_state) {
            if (auto* poi = _state->poi("focus"))
                items << QString("POI %1").arg(formatVec3(poi->p));
        }
    }

    _statsBar->setItems(items);
}
