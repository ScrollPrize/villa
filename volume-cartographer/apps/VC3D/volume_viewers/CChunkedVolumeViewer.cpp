#include "CChunkedVolumeViewer.hpp"

#include "CState.hpp"
#include "elements/ViewerStatsBar.hpp"
#include "RemoteVolumeCachePaths.hpp"
#include "VCSettings.hpp"
#include "ViewerManager.hpp"
#include "overlays/SegmentationOverlayController.hpp"
#include "vc/core/render/Colormaps.hpp"
#include "vc/core/render/PostProcess.hpp"
#include "render/ChunkCache.hpp"
#include "vc/core/render/PersistentZarrCacheBudget.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/Geometry.hpp"
#include "vc/core/util/Logging.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/ui/VCCollection.hpp"

#include <QApplication>
#include <QCursor>
#include <QElapsedTimer>
#include <QEvent>
#include <QFrame>
#include <QShowEvent>
#include <QGraphicsEllipseItem>
#include <QGraphicsItem>
#include <QGraphicsPathItem>
#include <QGraphicsScene>
#include <QGraphicsSimpleTextItem>
#include <QMessageBox>
#include <QPainter>
#include <QPainterPath>
#include <QPointer>
#include <QSettings>
#include <QTimer>
#include <QTransform>
#include <QVBoxLayout>
#include <QtConcurrent/QtConcurrentRun>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <mutex>
#include <optional>
#include <queue>
#include <source_location>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#include <opencv2/imgproc.hpp>

namespace {

constexpr float kMinScale = 0.002f;
constexpr float kMaxScale = 128.0f;
// Global clock is ~16ms/tick. Durations expressed in ticks for serviceRenderTick.
constexpr int kGlobalTickMs = 16;
constexpr int kStatusRefreshTicks = 5000 / kGlobalTickMs;           // ~313 ticks (5s)
// Non-interactive chunk-ready coalescing. Chunks for one viewpoint trickle in
// over many milliseconds; the default 16ms render debounce re-renders the whole
// frame per chunk (~9-11x per viewpoint). Debounce chunk-ready re-renders on a
// longer window so a burst collapses into ~1 render once chunks stop arriving.
constexpr float kResolutionLodZoomBias = 0.5f;
constexpr float kSegmentationResolutionLodZoomBias = 1.0f;
constexpr int kSurfaceResolutionLevelBias = 1;
constexpr int kInitialSegmentationSurfaceLevel = 5;
constexpr std::size_t kOverlayTargetPageMaximumBytes = 128ULL * 1024ULL * 1024ULL;
constexpr float kPanSmoothingAlpha = 0.65f;
constexpr int kSurfaceCellTileSize = 64;
constexpr std::array<QRgb, 12> kIntersectionPalette = {
    qRgb(255, 120, 120), qRgb(120, 200, 255), qRgb(120, 255, 140),
    qRgb(255, 220, 100), qRgb(220, 140, 255), qRgb(255, 160, 200),
    qRgb(140, 255, 220), qRgb(200, 255, 140), qRgb(255, 180, 120),
    qRgb(180, 200, 255), qRgb(255, 140, 180), qRgb(160, 255, 180),
};
constexpr int kIntersectionZ = 100;
constexpr const char* kMeasurementOverlayKey = "segmentation_measurement";

constexpr int kHighlightedIntersectionZ = 110;
constexpr int kActiveIntersectionZ = 120;
constexpr float kActiveIntersectionOpacityScale = 1.2f;
constexpr float kActiveIntersectionWidthScale = 1.3f;
constexpr float kActiveIntersectionMinWidthDelta = 0.75f;
constexpr float kApprovalPlaneIntersectionScale = 1.5f;
constexpr float kFocusProjectionThreshold = 4.0f;

std::optional<QColor> segmentFolderColorOverride(const QuadSurface* surface)
{
    if (!surface || surface->meta.is_null() || !surface->meta.contains("vc3d_segment_folder_color")) {
        return std::nullopt;
    }
    const auto& color = surface->meta["vc3d_segment_folder_color"];
    if (!color.is_array() || color.size() < 3) {
        return std::nullopt;
    }
    auto channel = [&](size_t index) {
        return std::clamp(static_cast<int>(std::lround(color.at(index).get_double())), 0, 255);
    };
    return QColor(channel(0), channel(1), channel(2));
}

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

bool finiteVec2(const cv::Vec2f& v)
{
    return std::isfinite(v[0]) && std::isfinite(v[1]);
}

bool finiteVec3(const cv::Vec3f& v)
{
    return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]);
}

float distance3(const cv::Vec3f& a, const cv::Vec3f& b)
{
    const cv::Vec3f d = a - b;
    return std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
}

double volumeDistanceMicrometers(const cv::Vec3f& a, const cv::Vec3f& b, double voxelSizeUm)
{
    if (!(voxelSizeUm > 0.0)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const cv::Vec3d d(static_cast<double>(a[0]) - static_cast<double>(b[0]),
                      static_cast<double>(a[1]) - static_cast<double>(b[1]),
                      static_cast<double>(a[2]) - static_cast<double>(b[2]));
    return std::sqrt(d.dot(d)) * voxelSizeUm;
}

struct SafeFocusProjection {
    bool valid = false;
    const char* reason = "unchecked";
    float dist = -1.0f;
    cv::Vec3f ptr{NAN, NAN, NAN};
    cv::Vec3f loc{NAN, NAN, NAN};
    cv::Vec3f coord{NAN, NAN, NAN};
    cv::Vec2f grid{NAN, NAN};
    int row = -1;
    int col = -1;
    int nearestRow = -1;
    int nearestCol = -1;
};

SafeFocusProjection validateSafeFocusProjection(
    QuadSurface& surface,
    const cv::Vec3f& ptr,
    const cv::Vec3f* focusWorld,
    std::optional<float> pointToDistance = std::nullopt)
{
    SafeFocusProjection result;
    result.ptr = ptr;

    if (!finiteVec3(result.ptr)) {
        result.reason = "non_finite_ptr";
        return result;
    }

    result.loc = surface.loc(result.ptr);
    if (!finiteVec3(result.loc)) {
        result.reason = "non_finite_loc";
        return result;
    }

    if (pointToDistance) {
        result.dist = *pointToDistance;
        if (!std::isfinite(result.dist) ||
            result.dist < 0.0f ||
            result.dist >= kFocusProjectionThreshold) {
            result.reason = "distance_out_of_range";
            return result;
        }
    }

    result.coord = surface.coord(result.ptr);
    if (!validSurfacePoint(result.coord)) {
        result.reason = "invalid_coord";
        return result;
    }

    if (!pointToDistance) {
        result.dist = focusWorld ? distance3(result.coord, *focusWorld) : 0.0f;
        if (!std::isfinite(result.dist) ||
            result.dist < 0.0f ||
            result.dist >= kFocusProjectionThreshold) {
            result.reason = "distance_out_of_range";
            return result;
        }
    }

    result.grid = surface.ptrToGrid(result.ptr);
    if (!finiteVec2(result.grid)) {
        result.reason = "non_finite_grid";
        return result;
    }

    const cv::Mat_<cv::Vec3f>* points = surface.rawPointsPtr();
    if (!points || points->cols < 2 || points->rows < 2) {
        result.reason = "missing_surface_points";
        return result;
    }
    if (result.grid[0] <= 0.0f || result.grid[1] <= 0.0f ||
        result.grid[0] >= static_cast<float>(points->cols - 1) ||
        result.grid[1] >= static_cast<float>(points->rows - 1)) {
        result.reason = "grid_on_or_outside_border";
        return result;
    }

    result.col = static_cast<int>(std::floor(result.grid[0]));
    result.row = static_cast<int>(std::floor(result.grid[1]));
    if (!surface.isQuadValid(result.row, result.col)) {
        result.reason = "invalid_interpolation_quad";
        return result;
    }

    result.nearestCol = static_cast<int>(std::lround(result.grid[0]));
    result.nearestRow = static_cast<int>(std::lround(result.grid[1]));
    if (!surface.isPointValid(result.nearestRow, result.nearestCol)) {
        result.reason = "invalid_nearest_point";
        return result;
    }
    const cv::Vec3f normal = surface.gridNormal(result.nearestRow, result.nearestCol);
    if (!finiteVec3(normal)) {
        result.reason = "non_finite_grid_normal";
        return result;
    }

    result.valid = true;
    result.reason = "valid";
    return result;
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
    // v is in coordinate/UI order [x, y, z] = [width, height, slices]. Label each
    // component so the scroll Z (slice/depth) axis is unambiguous in the status bar.
    return QString("[X=%1, Y=%2, Z=%3]")
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

constexpr std::size_t kOverlayDecodedCacheCapacity =
    1ULL * 1024ULL * 1024ULL * 1024ULL;

std::shared_ptr<vc::render::DecodedChunkCacheBudget> overlayDecodedCacheBudget()
{
    static auto budget =
        std::make_shared<vc::render::DecodedChunkCacheBudget>(
            kOverlayDecodedCacheCapacity);
    return budget;
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

float fitScaleForExtent(float extentU, float extentV, int viewportW, int viewportH)
{
    constexpr float kFitPadding = 0.88f;
    if (!(extentU > 0.0f) || !(extentV > 0.0f) || viewportW <= 0 || viewportH <= 0) {
        return 1.0f;
    }

    const float scaleU = static_cast<float>(viewportW) / extentU;
    const float scaleV = static_cast<float>(viewportH) / extentV;
    return std::clamp(std::min(scaleU, scaleV) * kFitPadding, kMinScale, kMaxScale);
}

std::shared_ptr<vc::render::ChunkCache> makeOverlayChunkCacheForVolume(
    const std::shared_ptr<Volume>& volume,
    const CState* state)
{
    if (!volume)
        return nullptr;

    vc::render::ChunkCache::Options options;
    options.decodedByteCapacity = kOverlayDecodedCacheCapacity;
    options.decodedByteBudget = overlayDecodedCacheBudget();
    options.maxConcurrentReads = 16;
    options.detectAllFillChunks = volume->isRemote();
    if (volume->isRemote()) {
        const auto budgetRoot = volume->remoteCacheRoot().empty()
            ? vc3d::remoteCacheRootForState(state)
            : volume->remoteCacheRoot();
        options.persistentCachePath = vc3d::persistentCacheDirForVolume(volume, state);
        options.persistentCacheBudgetRoot = budgetRoot;
        using namespace vc3d::settings;
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        constexpr std::uint64_t gib = 1024ULL * 1024ULL * 1024ULL;
        vc::render::PersistentZarrCacheBudget::Limits limits;
        const auto maximumGiB = settings.value(
            perf::REMOTE_CACHE_MAX_GIB, perf::REMOTE_CACHE_MAX_GIB_DEFAULT).toULongLong();
        if (maximumGiB > 0)
            limits.maximumBytes = maximumGiB * gib;
        limits.minimumFreeBytes = settings.value(
            perf::REMOTE_CACHE_MIN_FREE_GIB,
            perf::REMOTE_CACHE_MIN_FREE_GIB_DEFAULT).toULongLong() * gib;
        vc::render::PersistentZarrCacheBudget::configure(budgetRoot, limits);
    }

    return volume->createChunkCache(std::move(options));
}

struct OverlayChunkCacheLease {
    ~OverlayChunkCacheLease()
    {
        if (cache)
            cache->invalidate();
    }

    std::shared_ptr<vc::render::ChunkCache> cache;
};

std::shared_ptr<OverlayChunkCacheLease> sharedOverlayChunkCacheForVolume(
    const std::shared_ptr<Volume>& volume,
    const CState* state)
{
    if (!volume)
        return nullptr;

    const std::string key =
        vc3d::normalizedVolumeCacheIdentity(volume) +
        "|overlay-cache=" +
        (volume->isRemote()
             ? vc3d::remoteCacheRootForState(state).string()
             : std::string{});

    static std::mutex cacheMutex;
    static std::unordered_map<std::string, std::weak_ptr<OverlayChunkCacheLease>> caches;

    {
        std::lock_guard<std::mutex> lock(cacheMutex);
        auto it = caches.find(key);
        if (it != caches.end()) {
            if (auto lease = it->second.lock())
                return lease;
            caches.erase(it);
        }
    }

    auto cache = makeOverlayChunkCacheForVolume(volume, state);
    if (!cache)
        return nullptr;
    auto lease = std::make_shared<OverlayChunkCacheLease>();
    lease->cache = std::move(cache);

    std::lock_guard<std::mutex> lock(cacheMutex);
    auto& slot = caches[key];
    if (auto existing = slot.lock())
        return existing;
    slot = lease;
    return lease;
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
    _view->setFrameShape(QFrame::NoFrame);
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
    connect(_view, &CVolumeViewerView::sendMouseLeftView, this, [this]() {
        clearLineAnnotationPlacementMarker();
    });
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

    // No per-viewer timers: render requests submit immediately. ViewerManager's
    // global clock still drains intersection requests and status refreshes.
    _statusRefreshTicks = kStatusRefreshTicks;

    reloadPerfSettings();

    auto* layout = new QVBoxLayout;
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);
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

void CChunkedVolumeViewer::showEvent(QShowEvent* event)
{
    QWidget::showEvent(event);
    // Render whatever went stale while this viewer was hidden.
    if (_renderStaleWhileHidden && !_closing) {
        _renderStaleWhileHidden = false;
        submitRender("shown after hidden");
    }
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

    // Clear all pending maintenance work (the global clock will see nothing to do).
    _intersectionPending = false;

    _activeRenderJob.reset();
    _pendingRenderJob.reset();
    _displayedRenderJob.reset();
    _pendingRenderDirty = false;
    ++_renderSerial;

    if (_chunkCbId != 0 && _chunkArray) {
        _chunkArray->removeChunkReadyListener(_chunkCbId);
        _chunkCbId = 0;
    }
    if (_overlayChunkCbId != 0 && _overlayChunkArray) {
        _overlayChunkArray->removeChunkReadyListener(_overlayChunkCbId);
        _overlayChunkCbId = 0;
    }
    _overlayChunkArray.reset();
    _overlayChunkCacheOwner.reset();
}

void CChunkedVolumeViewer::reloadPerfSettings()
{
    QSettings s(vc3d::settingsFilePath(), QSettings::IniFormat);
    using namespace vc3d::settings;
    _panSensitivity = std::max(0.01f, s.value(viewer::PAN_SENSITIVITY, viewer::PAN_SENSITIVITY_DEFAULT).toFloat());
    _zoomSensitivity = std::max(0.01f, s.value(viewer::ZOOM_SENSITIVITY, viewer::ZOOM_SENSITIVITY_DEFAULT).toFloat());
    _zScrollSensitivity = std::max(0.01f, s.value(viewer::ZSCROLL_SENSITIVITY, viewer::ZSCROLL_SENSITIVITY_DEFAULT).toFloat());
    _linkedCursorViewTolerance = std::max(
        0.0f,
        s.value(viewer::POINT_COLLECTION_VIEW_TOLERANCE,
                viewer::POINT_COLLECTION_VIEW_TOLERANCE_DEFAULT).toFloat());
    const int interpIdx = s.value(perf::INTERPOLATION_METHOD, perf::INTERPOLATION_METHOD_DEFAULT).toInt();
    _samplingMethod = static_cast<vc::Sampling>(std::clamp(interpIdx, 0, 1));
    _maxDisplayedResolution = std::clamp(
        s.value(viewer::MAX_DISPLAYED_RESOLUTION, viewer::MAX_DISPLAYED_RESOLUTION_DEFAULT).toInt(),
        0,
        5);
}

void CChunkedVolumeViewer::setSurface(const std::string& name)
{
    if (_closing) {
        return;
    }
    if (_measurement.active || _measurement.first || _measurement.second) {
        clearMeasurement();
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

CChunkedVolumeViewer::CameraState CChunkedVolumeViewer::cameraState() const
{
    CameraState state;
    state.surfacePtrX = _surfacePtrX;
    state.surfacePtrY = _surfacePtrY;
    state.scale = _scale;
    state.zOffset = _zOff;
    state.zOffsetWorldDir = _zOffWorldDir;
    return state;
}

float CChunkedVolumeViewer::clampCameraScale(float scale)
{
    return std::clamp(scale, kMinScale, kMaxScale);
}

void CChunkedVolumeViewer::applyCameraState(const CameraState& state, bool forceRender)
{
    if (_closing) {
        return;
    }
    _surfacePtrX = state.surfacePtrX;
    _surfacePtrY = state.surfacePtrY;
    _scale = state.scale;
    _zOff = state.zOffset;
    _zOffWorldDir = state.zOffsetWorldDir;
    recalcPyramidLevel();
    _genCacheDirty = true;
    if (forceRender) {
        renderVisible(true, "annotation camera state applied");
    } else {
        submitRender("annotation camera state applied");
    }
    refreshMeasurementOverlay();
    emit overlaysUpdated();
}

void CChunkedVolumeViewer::applyCameraStateForReplayRepaint(const CameraState& state)
{
    if (_closing) {
        return;
    }
    _surfacePtrX = state.surfacePtrX;
    _surfacePtrY = state.surfacePtrY;
    _scale = state.scale;
    _zOff = state.zOffset;
    _zOffWorldDir = state.zOffsetWorldDir;
    recalcPyramidLevel();
    _genCacheDirty = true;
    syncCameraTransform();
    refreshMeasurementOverlay();
    emit overlaysUpdated();
}

bool CChunkedVolumeViewer::isRenderQuiescent() const
{
    return !_renderWorkerBusy.load(std::memory_order_acquire)
        && !_pendingRenderJob.has_value()
        && !_pendingRenderDirty;
}

std::size_t CChunkedVolumeViewer::chunkFetchesInFlight() const
{
    return _chunkArray ? _chunkArray->stats().remoteFetchesInFlight : 0;
}

void CChunkedVolumeViewer::rebuildChunkArray()
{
    if (_chunkCbId != 0 && _chunkArray) {
        _chunkArray->removeChunkReadyListener(_chunkCbId);
        _chunkCbId = 0;
    }
    _chunkArray.reset();
    _lastRenderResult.reset();
    if (!_volume)
        return;

    try {
        _chunkArray = _volume->sharedChunkCache();
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
            ++guard->_chunkContentEpoch;
            guard->submitRender("chunk ready");
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
    if (_cursorCrosshair)
        _cursorCrosshair->hide();
    clearLineAnnotationPlacementMarker();
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
    recalcPyramidLevel();   // also pushes the scalebar µm/px (updateScalebarScale)
    updateContentBounds();
    resizeFramebuffer();
    submitRender("volume changed");
    renderIntersections("volume changed");
    updateStatusLabel();
}

void CChunkedVolumeViewer::invalidateVis()
{
    if (_closing) {
        return;
    }
    _genCacheDirty = true;
}

void CChunkedVolumeViewer::invalidateVisRegion(const std::string& name, const cv::Rect& changedCells)
{
    if (changedCells.empty() || name != _surfName || _surfName != "segmentation") {
        invalidateVis();
        return;
    }


    _genCacheDirty = true;
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
        updateContentBounds();
        updateFocusMarker();
        if (_suppressNextSurfaceEditRender) {
            _suppressNextSurfaceEditRender = false;
        } else {
            submitRender("current surface edit update");
        }
        scheduleIntersectionRender("current surface edit update");
        return;
    }

    _genCacheDirty = true;
    _zOffWorldDir = {0, 0, 0};
    invalidateIntersect(name);
    if (!surf) {
        clearIntersectionItems();
        _measurement = {};
        _scene->clear();
        _overlayGroups.clear();
        _cursorCrosshair = nullptr;
        _lineAnnotationPlacementMarker = nullptr;
        _focusMarker = nullptr;
        return;
    }
    updateContentBounds();
    const bool isSegmentationQuadSurface =
        _surfName == "segmentation" && dynamic_cast<QuadSurface*>(surf.get());

    auto setSegmentationPointerFromFocus = [&]() {
        auto* quad = dynamic_cast<QuadSurface*>(surf.get());
        auto* poi = _state ? _state->poi("focus") : nullptr;
        if (!quad || !poi) {
            return false;
        }
        if (!poi->surfaceId.empty() && !quad->id.empty() &&
            poi->surfaceId != quad->id && poi->surfaceId != _surfName) {
            return false;
        }

        if (poi->surfacePtr) {
            const SafeFocusProjection projected =
                validateSafeFocusProjection(*quad, *poi->surfacePtr, &poi->p);
            if (projected.valid) {
                _surfacePtrX = projected.loc[0];
                _surfacePtrY = projected.loc[1];
                return true;
            }
        }

        cv::Vec3f ptr = quad->pointer();
        const float dist = quad->pointTo(ptr, poi->p, kFocusProjectionThreshold, 100, nullptr);
        const SafeFocusProjection projected =
            validateSafeFocusProjection(*quad, ptr, &poi->p, dist);
        if (!projected.valid) {
            return false;
        }

        _surfacePtrX = projected.loc[0];
        _surfacePtrY = projected.loc[1];
        return true;
    };

    if (!isEditUpdate && isSegmentationQuadSurface && !_initializedFirstSegmentationSurface) {
        if (_resetViewOnSurfaceChange) {
            (void)setSegmentationPointerFromFocus();
        }
        _zOff = 0.0f;
        const int n = _chunkArray ? _chunkArray->numLevels()
                                  : (_volume ? static_cast<int>(_volume->numScales()) : 1);
        if (_resetViewOnSurfaceChange) {
            _scale = scaleForCoarsestSegmentationRenderLevel(n);
            recalcPyramidLevel();
        }
        _initializedFirstSegmentationSurface = true;
    } else if (!isEditUpdate && _resetViewOnSurfaceChange && isSegmentationQuadSurface) {
        (void)setSegmentationPointerFromFocus();
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
            cv::Vec3f ptr = quad->pointer();
            auto* patchIndex = _viewerManager ? _viewerManager->surfacePatchIndex() : nullptr;
            if (quad->pointTo(ptr, *preservedViewCenter, 4.0f, 100, patchIndex) >= 0.0f) {
                const cv::Vec3f loc = quad->loc(ptr);
                if (std::isfinite(loc[0]) && std::isfinite(loc[1])) {
                    _surfacePtrX = loc[0];
                    _surfacePtrY = loc[1];
                }
            }
        }
    }
    updateFocusMarker();
    submitRender("current surface changed");
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
    if (property("vc_viewer_role").toString() == QStringLiteral("annotation")) {
        if (_focusMarker) {
            _focusMarker->hide();
        }
        return;
    }

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
    submitRender("focus POI changed");
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
    updateScalebarScale();
}

void CChunkedVolumeViewer::updateScalebarScale()
{
    // Scene pixels are framebuffer pixels. `_scale` is the camera zoom in
    // framebuffer pixels per level-0 voxel-space unit; render LOD changes which
    // zarr level is sampled, not the physical size of the view.
    if (!_view || !_volume)
        return;
    double voxel = _volume->voxelSize();
    if (!(voxel > 0.0) || !(_scale > 0.0f))
        return;
    const double umPerScenePx = voxel / static_cast<double>(_scale);
    _view->setVoxelSize(umPerScenePx, umPerScenePx);
}

void CChunkedVolumeViewer::resizeFramebuffer()
{
    const QSize vpSize = _view->viewport()->size();
    const int w = std::max(1, vpSize.width());
    const int h = std::max(1, vpSize.height());
    if (_framebuffer.isNull() || _framebuffer.width() != w || _framebuffer.height() != h) {
        _framebuffer = QImage(w, h, QImage::Format_RGB32);
        _framebuffer.fill(QColor(64, 64, 64));
    }
    _scene->setSceneRect(0, 0, w, h);
}

void CChunkedVolumeViewer::requestRender(const char* reason, std::source_location caller)
{
    if (reason && std::string_view(reason) == "push/pull active viewer refresh") {
        _suppressNextSurfaceEditRender = true;
    }
    submitRender(reason, caller);
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
    _intersectionPending = true;   // serviceRenderTick() renders it next tick
    profile.setDetails("action=intersection_pending");
}

// Driven by ViewerManager's single global clock (~60Hz). Rendering is submitted
// immediately by view/data updates; this tick only drains non-render maintenance.
void CChunkedVolumeViewer::serviceRenderTick()
{
    if (_closing)
        return;

    if (_intersectionPending) {
        _intersectionPending = false;
        renderIntersections("global tick intersection");
        emit overlaysUpdated();
    }

    // Periodic status refresh (background download progress).
    if (--_statusRefreshTicks <= 0) {
        _statusRefreshTicks = kStatusRefreshTicks;
        updateStatusLabel();
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
    updateDisplayedFramebufferMapping();
    updateIntersectionPreviewTransform();
    refreshMeasurementOverlay();
    updateFocusMarker();
    requestDirectPaint();
}

void CChunkedVolumeViewer::requestDirectPaint()
{
    if (_closing || !_view || !_view->viewport() || !_view->isVisible()) {
        return;
    }
    _view->viewport()->update();
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
    level = std::max(level, _maxDisplayedResolution);
    return std::clamp(level, 0, _chunkArray->numLevels() - 1);
}

int CChunkedVolumeViewer::overlayRenderStartLevel(bool preferSurfaceResolution) const
{
    if (!_overlayChunkArray) {
        return 0;
    }

    int level = _dsScaleIdx;
    if (preferSurfaceResolution && level < _overlayChunkArray->numLevels() - 1) {
        level -= kSurfaceResolutionLevelBias;
    }
    level = std::max(level, _overlayMaxDisplayedResolution);
    return std::clamp(level, 0, _overlayChunkArray->numLevels() - 1);
}

void CChunkedVolumeViewer::markInteractiveMotion(double)
{
    // Interactions schedule a normal full-quality render directly (scheduleRender in
    // the handler). The old interactive-preview + settle-timer fast path is gone.
}

bool CChunkedVolumeViewer::streamingCompositeUnsupported() const
{
    return !isSupportedStreamingCompositeMethod(_compositeSettings.params.method) ||
           _compositeSettings.params.lightingEnabled ||
           _compositeSettings.params.method == "beerLambert" ||
           _compositeSettings.useVolumeGradients;
}

struct CChunkedVolumeViewer::RenderContext {
    PendingRenderJob renderJob;
    std::uint64_t serial = 0;
    int fbW = 0;
    int fbH = 0;
    float surfacePtrX = 0.0f;
    float surfacePtrY = 0.0f;
    float scale = 1.0f;
    float zOff = 0.0f;
    cv::Vec3f zOffWorldDir{0, 0, 0};
    int startLevel = 0;
    int overlayStartLevel = 0;
    vc::Sampling samplingMethod = vc::Sampling::Trilinear;
    vc::Sampling overlaySamplingMethod = vc::Sampling::Nearest;
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
    OverlayCompositeSettings overlayComposite;
    std::shared_ptr<GeneratedSurfaceCache> genCache;
    bool genCacheDirty = false;
    std::shared_ptr<const RenderResult> prevResult;
    std::string profileReason;
    std::string profileCaller;
};

struct CChunkedVolumeViewer::RenderResult {
    std::uint64_t serial = 0;
    PendingRenderJob renderJob;
    QImage framebuffer;
    // Pre-LUT sample values and their coverage, kept so the next
    // same-geometry render can fill chunk holes from this frame.
    cv::Mat_<uint8_t> values;
    cv::Mat_<uint8_t> coverage;
    cv::Mat_<uint8_t> overlayValues;
    cv::Mat_<uint8_t> overlayCoverage;
    // Pixels already sampled at overlayStartLevel. Unlike overlayCoverage,
    // this excludes coarse fallback pixels so a static view can progressively
    // complete without needing its entire 3D overlay working set resident.
    cv::Mat_<uint8_t> overlayTargetCoverage;
    float surfacePtrX = 0.0f;
    float surfacePtrY = 0.0f;
    float scale = 1.0f;
    double renderFrameElapsedMs = 0.0;
    std::chrono::steady_clock::time_point submittedAt;
    std::chrono::steady_clock::time_point workerStartedAt;
    std::chrono::steady_clock::time_point workerFinishedAt;
};

CChunkedVolumeViewer::RenderResult CChunkedVolumeViewer::renderFrame(RenderContext ctx)
{
    QElapsedTimer renderTimer;
    renderTimer.start();
    // Sub-phase timing (only meaningful under --profile). gen = surface coords,
    // sample = chunk sample/decode, blit = colormap LUT + framebuffer write.
    const bool profilePhases = ProfileLoggingEnabled();
    qint64 phaseGenMs = -1, phaseSampleMs = 0, phaseBlitMs = 0;
    bool phaseGenCached = false;
    QElapsedTimer phaseTimer;
    if (ProfileLoggingEnabled()) {
        Logger()->info("[vc3d-profile] renderFrame begin reason='{}' caller='{}' serial={} surf='{}' size={}x{} level={} overlay_level={} overlay={} composite={} planeComposite={}",
                       ctx.profileReason, ctx.profileCaller, ctx.serial,
                       ctx.surf ? ctx.surf->id : std::string(""),
                       ctx.fbW, ctx.fbH, ctx.startLevel, ctx.overlayStartLevel,
                       bool(ctx.overlayChunkArray && ctx.overlayVolume && ctx.overlayOpacity > 0.0f),
                       ctx.compositeSettings.enabled, ctx.compositeSettings.planeEnabled);
    }
    RenderResult result;
    result.serial = ctx.serial;
    result.renderJob = ctx.renderJob;
    result.surfacePtrX = ctx.surfacePtrX;
    result.surfacePtrY = ctx.surfacePtrY;
    result.scale = ctx.scale;
    result.submittedAt = ctx.renderJob.submittedAt;
    result.workerStartedAt = std::chrono::steady_clock::now();
    result.framebuffer = QImage(std::max(1, ctx.fbW), std::max(1, ctx.fbH), QImage::Format_RGB32);
    result.framebuffer.fill(QColor(64, 64, 64));

    auto finishRenderFrameProfile = [&]() {
        result.renderFrameElapsedMs = static_cast<double>(renderTimer.nsecsElapsed()) / 1000000.0;
        result.workerFinishedAt = std::chrono::steady_clock::now();
        if (!ProfileLoggingEnabled())
            return;
        Logger()->info("[vc3d-profile] renderFrame end elapsed_ms={} gen_ms={} genCached={} sample_ms={} blit_ms={} reason='{}' caller='{}' serial={} framebuffer={}x{}",
                       result.renderFrameElapsedMs, phaseGenMs, phaseGenCached,
                       phaseSampleMs, phaseBlitMs, ctx.profileReason, ctx.profileCaller,
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
    cv::Mat_<uint8_t> overlayTargetCoverage;
    const bool continuingSameGeometry =
        ctx.prevResult && !ctx.genCacheDirty &&
        renderJobsSameGeometry(ctx.renderJob, ctx.prevResult->renderJob);
    vc::render::ChunkedPlaneSampler::Options options(ctx.samplingMethod, 32);
    vc::render::ChunkedPlaneSampler::Options overlayOptions(
        ctx.overlaySamplingMethod, options.tileSize);
    // A new view may queue one adjacent coarse level for a quick preview.
    // Once that geometry has produced a frame, only the requested level may
    // create more cache work. Resident fallback chunks can still fill holes,
    // but those reads do not promote them over the target working set.
    options.queuedFallbackLevels = continuingSameGeometry ? 0 : 1;
    overlayOptions.queuedFallbackLevels = 0;

    auto initializeOverlayProgress = [&]() {
        overlayValues.create(ctx.fbH, ctx.fbW);
        overlayCoverage.create(ctx.fbH, ctx.fbW);
        overlayTargetCoverage.create(ctx.fbH, ctx.fbW);
        overlayValues.setTo(0);
        overlayCoverage.setTo(0);
        overlayTargetCoverage.setTo(0);

        const RenderResult* prev = ctx.prevResult.get();
        if (!continuingSameGeometry || !prev ||
            prev->overlayValues.size() != overlayValues.size() ||
            prev->overlayCoverage.size() != overlayCoverage.size() ||
            prev->overlayTargetCoverage.size() != overlayTargetCoverage.size()) {
            return;
        }

        for (int y = 0; y < overlayValues.rows; ++y) {
            auto* dstValue = overlayValues.ptr<uint8_t>(y);
            auto* dstCoverage = overlayCoverage.ptr<uint8_t>(y);
            auto* dstTarget = overlayTargetCoverage.ptr<uint8_t>(y);
            const auto* srcValue = prev->overlayValues.ptr<uint8_t>(y);
            const auto* srcCoverage = prev->overlayCoverage.ptr<uint8_t>(y);
            const auto* srcTarget = prev->overlayTargetCoverage.ptr<uint8_t>(y);
            for (int x = 0; x < overlayValues.cols; ++x) {
                if (!srcTarget[x])
                    continue;
                dstValue[x] = srcValue[x];
                dstCoverage[x] = srcCoverage[x];
                dstTarget[x] = 1;
            }
        }
    };

    auto overlayFallbackOptions = [&]() {
        auto fallback = overlayOptions;
        fallback.queueMisses = false;
        fallback.queuedFallbackLevels = 0;
        return fallback;
    };

    auto overlayResidentOptions = overlayOptions;
    overlayResidentOptions.queueMisses = false;

    std::size_t overlayRequestBytesRemaining = 0;
    if (ctx.overlayChunkArray) {
        const std::size_t capacity =
            ctx.overlayChunkArray->stats().decodedByteCapacity;
        overlayRequestBytesRemaining = std::min(
            kOverlayTargetPageMaximumBytes, capacity / 8);
    }
    std::unordered_set<vc::render::ChunkKey, vc::render::ChunkKeyHash>
        overlayRequestsThisFrame;
    auto queueOverlayTargetPage =
        [&](const std::vector<vc::render::ChunkKey>& dependencies) {
            if (!ctx.overlayChunkArray || overlayRequestBytesRemaining == 0)
                return;

            auto orderedDependencies = dependencies;
            std::sort(
                orderedDependencies.begin(), orderedDependencies.end(),
                [](const vc::render::ChunkKey& lhs,
                   const vc::render::ChunkKey& rhs) {
                    return std::tie(lhs.level, lhs.iz, lhs.iy, lhs.ix) <
                           std::tie(rhs.level, rhs.iz, rhs.iy, rhs.ix);
                });
            std::vector<vc::render::ChunkKey> page;
            for (const auto& key : orderedDependencies) {
                if (!overlayRequestsThisFrame.insert(key).second)
                    continue;
                const auto chunkShape = ctx.overlayChunkArray->chunkShape(key.level);
                std::size_t chunkBytes =
                    ctx.overlayChunkArray->dtype() == vc::render::ChunkDtype::UInt16 ? 2 : 1;
                for (const int dimension : chunkShape) {
                    if (dimension <= 0 ||
                        chunkBytes > std::numeric_limits<std::size_t>::max() /
                                         static_cast<std::size_t>(dimension)) {
                        chunkBytes = std::numeric_limits<std::size_t>::max();
                        break;
                    }
                    chunkBytes *= static_cast<std::size_t>(dimension);
                }
                if (chunkBytes > overlayRequestBytesRemaining) {
                    if (!page.empty())
                        break;
                    // A single valid chunk must be admitted even if its decoded
                    // size exceeds the normal page target.
                    page.push_back(key);
                    overlayRequestBytesRemaining = 0;
                    break;
                }
                page.push_back(key);
                overlayRequestBytesRemaining -= chunkBytes;
                if (overlayRequestBytesRemaining == 0)
                    break;
            }
            if (!page.empty())
                ctx.overlayChunkArray->prefetchChunks(page, false);
        };

    auto streamingCompositeUnsupported = [&]() {
        return !isSupportedStreamingCompositeMethod(ctx.compositeSettings.params.method) ||
               ctx.compositeSettings.params.lightingEnabled ||
               ctx.compositeSettings.params.method == "beerLambert" ||
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
            vc::render::ChunkedPlaneSampler::samplePlaneFineToCoarse(
                array, ctx.startLevel, origin, vxStep, vyStep, dst, cov, options);
            return;
        }

        const int front = std::max(0, ctx.compositeSettings.planeLayersFront);
        const int behind = std::max(0, ctx.compositeSettings.planeLayersBehind);
        const int numLayers = front + behind + 1;
        const int zStart = -behind;
        const float zStep = ctx.compositeSettings.reverseDirection ? -1.0f : 1.0f;
        auto compositeOptions = vc::render::ChunkedPlaneSampler::Options(
            vc::Sampling::Nearest, options.tileSize);
        compositeOptions.queuedFallbackLevels = options.queuedFallbackLevels;
        std::vector<cv::Mat_<uint8_t>> layerValues;
        std::vector<cv::Mat_<uint8_t>> layerCoverage;
        layerValues.reserve(numLayers);
        layerCoverage.reserve(numLayers);
        for (int i = 0; i < numLayers; ++i) {
            layerValues.emplace_back(dst.rows, dst.cols, uint8_t(0));
            layerCoverage.emplace_back(dst.rows, dst.cols, uint8_t(0));
            const cv::Vec3f layerOrigin = origin + normal * (float(zStart + i) * zStep);
            vc::render::ChunkedPlaneSampler::samplePlaneFineToCoarse(
                array, ctx.startLevel, layerOrigin, vxStep, vyStep,
                layerValues.back(), layerCoverage.back(), compositeOptions);
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

    auto sampleCoordsComposite = [&](const cv::Mat_<cv::Vec3f>& coords,
                                     const cv::Mat_<cv::Vec3f>& normals,
                                     cv::Mat_<uint8_t>& dst,
                                     cv::Mat_<uint8_t>& cov,
                                     vc::render::ChunkCache& array,
                                     int startLevel,
                                     int layersFront,
                                     int layersBehind,
                                     float zStep,
                                     const CompositeParams& params,
                                     const vc::render::ChunkedPlaneSampler::Options& samplingOptions,
                                     cv::Mat_<uint8_t>* targetResolved,
                                     const vc::render::ChunkedPlaneSampler::Options* fallbackOptions) {
        const int front = std::max(0, layersFront);
        const int behind = std::max(0, layersBehind);
        const int numLayers = front + behind + 1;
        const int zStart = -behind;
        std::vector<cv::Mat_<uint8_t>> layerValues;
        std::vector<cv::Mat_<uint8_t>> layerCoverage;
        std::vector<cv::Mat_<uint8_t>> layerTargetCoverage;
        cv::Mat_<uint8_t> resolvedBefore;
        if (targetResolved)
            resolvedBefore = targetResolved->clone();
        cv::Mat_<cv::Vec3f> layerCoords(coords.rows, coords.cols);
        layerValues.reserve(numLayers);
        layerCoverage.reserve(numLayers);
        if (targetResolved)
            layerTargetCoverage.reserve(numLayers);
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
            if (targetResolved) {
                layerTargetCoverage.push_back(resolvedBefore.clone());
                vc::render::ChunkedPlaneSampler::sampleCoordsLevel(
                    array, startLevel, layerCoords,
                    layerValues.back(), layerTargetCoverage.back(), samplingOptions);
                queueOverlayTargetPage(
                    vc::render::ChunkedPlaneSampler::collectCoordsDependencies(
                        array, startLevel, layerCoords,
                        layerTargetCoverage.back(), samplingOptions));
                layerCoverage.back() = layerTargetCoverage.back().clone();
                if (fallbackOptions && startLevel + 1 < array.numLevels()) {
                    vc::render::ChunkedPlaneSampler::sampleCoordsFineToCoarse(
                        array, startLevel + 1, layerCoords,
                        layerValues.back(), layerCoverage.back(), *fallbackOptions);
                }
            } else {
                vc::render::ChunkedPlaneSampler::sampleCoordsFineToCoarse(
                    array, startLevel, layerCoords,
                    layerValues.back(), layerCoverage.back(), samplingOptions);
            }
        }
        LayerStack stack;
        stack.values.resize(numLayers);
        for (int y = 0; y < dst.rows; ++y) {
            auto* dstRow = dst.ptr<uint8_t>(y);
            auto* covRow = cov.ptr<uint8_t>(y);
            for (int x = 0; x < dst.cols; ++x) {
                if (targetResolved && resolvedBefore(y, x))
                    continue;
                stack.validCount = 0;
                bool allLayersAtTarget = targetResolved != nullptr;
                for (int i = 0; i < numLayers; ++i) {
                    if (targetResolved && !layerTargetCoverage[i](y, x))
                        allLayersAtTarget = false;
                    if (!layerCoverage[i](y, x))
                        continue;
                    const float value = static_cast<float>(layerValues[i](y, x));
                    if (value < static_cast<float>(params.isoCutoff))
                        continue;
                    stack.values[stack.validCount++] = value;
                }
                if (stack.validCount > 0) {
                    dstRow[x] = static_cast<uint8_t>(std::clamp(
                        compositeLayerStack(stack, params), 0.0f, 255.0f));
                    covRow[x] = 1;
                }
                if (allLayersAtTarget)
                    (*targetResolved)(y, x) = 1;
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
            vc::render::ChunkedPlaneSampler::sampleCoordsFineToCoarse(
                array, ctx.startLevel, coords, dst, cov, options);
            return;
        }

        const float zStep = ctx.compositeSettings.reverseDirection ? -1.0f : 1.0f;
        auto compositeOptions = vc::render::ChunkedPlaneSampler::Options(
            vc::Sampling::Nearest, options.tileSize);
        compositeOptions.queuedFallbackLevels = options.queuedFallbackLevels;
        sampleCoordsComposite(coords, normals, dst, cov, array, ctx.startLevel,
                              ctx.compositeSettings.layersFront,
                              ctx.compositeSettings.layersBehind,
                              zStep, ctx.compositeSettings.params, compositeOptions,
                              nullptr, nullptr);
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
        if (profilePhases) phaseTimer.restart();
        samplePlane(origin, vxStep, vyStep, n, values, coverage, *ctx.chunkArray);
        if (profilePhases) phaseSampleMs += phaseTimer.elapsed();
        if (ctx.overlayChunkArray && ctx.overlayVolume && ctx.overlayOpacity > 0.0f) {
            initializeOverlayProgress();
            const int level = std::clamp(ctx.overlayStartLevel, 0, ctx.overlayChunkArray->numLevels() - 1);
            vc::render::ChunkedPlaneSampler::samplePlaneLevel(
                *ctx.overlayChunkArray, level, origin, vxStep, vyStep,
                overlayValues, overlayTargetCoverage, overlayResidentOptions);
            queueOverlayTargetPage(
                vc::render::ChunkedPlaneSampler::collectPlaneDependencies(
                    *ctx.overlayChunkArray, level, origin, vxStep, vyStep,
                    overlayTargetCoverage, overlayResidentOptions));
            overlayCoverage.setTo(1, overlayTargetCoverage);
            if (level + 1 < ctx.overlayChunkArray->numLevels()) {
                const auto fallback = overlayFallbackOptions();
                vc::render::ChunkedPlaneSampler::samplePlaneFineToCoarse(
                    *ctx.overlayChunkArray, level + 1, origin, vxStep, vyStep,
                    overlayValues, overlayCoverage, fallback);
            }
        }
    } else {
        cv::Mat_<cv::Vec3f> coords;
        cv::Mat_<cv::Vec3f> normals;
        const cv::Vec3f offset(ctx.surfacePtrX * ctx.scale - float(ctx.fbW) * 0.5f,
                               ctx.surfacePtrY * ctx.scale - float(ctx.fbH) * 0.5f,
                               0.0f);
        const bool overlayActive =
            ctx.overlayChunkArray && ctx.overlayVolume && ctx.overlayOpacity > 0.0f;
        const bool overlayWantsComposite =
            overlayActive && ctx.overlayComposite.enabled &&
            (ctx.overlayComposite.method == "max" ||
             ctx.overlayComposite.method == "mean" ||
             ctx.overlayComposite.method == "min") &&
            (ctx.overlayComposite.layersFront > 0 || ctx.overlayComposite.layersBehind > 0);
        const bool needSurfaceNormals =
            ctx.zOff != 0.0f ||
            (ctx.compositeSettings.enabled && !streamingCompositeUnsupported()) ||
            overlayWantsComposite;

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

        phaseGenCached = genCacheHit;
        if (!genCacheHit) {
            if (profilePhases) phaseTimer.restart();
            ctx.surf->gen(&coords, needSurfaceNormals ? &normals : nullptr,
                          cv::Size(ctx.fbW, ctx.fbH), {0, 0, 0}, ctx.scale, offset);
            applyPerPixelNormalOffset(coords, normals, ctx.zOff);
            if (profilePhases) phaseGenMs = phaseTimer.elapsed();

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
            if (profilePhases) phaseTimer.restart();
            sampleCoords(coords, normals, values, coverage, *ctx.chunkArray);
            if (profilePhases) phaseSampleMs += phaseTimer.elapsed();
            if (overlayActive) {
                initializeOverlayProgress();
                const int level = std::clamp(ctx.overlayStartLevel, 0, ctx.overlayChunkArray->numLevels() - 1);
                if (overlayWantsComposite && !normals.empty()) {
                    // Keep "front" pointing the same physical direction as the primary composite.
                    const float zStep = ctx.compositeSettings.reverseDirection ? -1.0f : 1.0f;
                    CompositeParams overlayParams;
                    overlayParams.method = ctx.overlayComposite.method;
                    const auto fallback = overlayFallbackOptions();
                    sampleCoordsComposite(coords, normals, overlayValues, overlayCoverage,
                                          *ctx.overlayChunkArray, level,
                                          ctx.overlayComposite.layersFront,
                                          ctx.overlayComposite.layersBehind,
                                          zStep, overlayParams, overlayResidentOptions,
                                          &overlayTargetCoverage, &fallback);
                } else {
                    vc::render::ChunkedPlaneSampler::sampleCoordsLevel(
                        *ctx.overlayChunkArray, level, coords, overlayValues,
                        overlayTargetCoverage, overlayResidentOptions);
                    queueOverlayTargetPage(
                        vc::render::ChunkedPlaneSampler::collectCoordsDependencies(
                            *ctx.overlayChunkArray, level, coords,
                            overlayTargetCoverage, overlayResidentOptions));
                    overlayCoverage.setTo(1, overlayTargetCoverage);
                    if (level + 1 < ctx.overlayChunkArray->numLevels()) {
                        const auto fallback = overlayFallbackOptions();
                        vc::render::ChunkedPlaneSampler::sampleCoordsFineToCoarse(
                            *ctx.overlayChunkArray, level + 1, coords,
                            overlayValues, overlayCoverage, fallback);
                    }
                }
            }
        }
    }

    // Carry forward the previous frame's pixels where this frame has no
    // chunk data yet (evicted or still in flight). Chunk-ready re-renders
    // arrive constantly while streaming, and presenting a from-scratch
    // frame that lost a chunk flashes the region blank until the refetch
    // lands. Sampling above always runs first, so refined data still
    // replaces carried pixels as soon as it exists.
    if (const RenderResult* prev = ctx.prevResult.get();
        prev && !ctx.genCacheDirty &&
        renderJobsSameGeometry(ctx.renderJob, prev->renderJob)) {
        auto fillFromPrev = [](cv::Mat_<uint8_t>& dst, cv::Mat_<uint8_t>& cov,
                               const cv::Mat_<uint8_t>& prevDst,
                               const cv::Mat_<uint8_t>& prevCov) {
            if (dst.empty() || prevDst.empty() ||
                dst.size() != prevDst.size() || cov.size() != prevCov.size())
                return;
            for (int y = 0; y < dst.rows; ++y) {
                uint8_t* dstRow = dst.ptr<uint8_t>(y);
                uint8_t* covRow = cov.ptr<uint8_t>(y);
                const uint8_t* prevDstRow = prevDst.ptr<uint8_t>(y);
                const uint8_t* prevCovRow = prevCov.ptr<uint8_t>(y);
                for (int x = 0; x < dst.cols; ++x) {
                    if (!covRow[x] && prevCovRow[x]) {
                        dstRow[x] = prevDstRow[x];
                        covRow[x] = 1;
                    }
                }
            }
        };
        fillFromPrev(values, coverage, prev->values, prev->coverage);
        fillFromPrev(overlayValues, overlayCoverage,
                     prev->overlayValues, prev->overlayCoverage);
    }
    result.values = values;
    result.coverage = coverage;
    result.overlayValues = overlayValues;
    result.overlayCoverage = overlayCoverage;
    result.overlayTargetCoverage = overlayTargetCoverage;

    if (profilePhases) phaseTimer.restart();
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
    if (profilePhases) phaseBlitMs = phaseTimer.elapsed();
    finishRenderFrameProfile();
    return result;
}

std::optional<CChunkedVolumeViewer::PendingRenderJob> CChunkedVolumeViewer::captureRenderJob(
    const char* reason,
    std::source_location caller,
    const std::shared_ptr<Surface>& surf,
    int fbW,
    int fbH,
    std::chrono::steady_clock::time_point submittedAt)
{
    if (!surf || !_volume || !_chunkArray || fbW <= 0 || fbH <= 0) {
        return std::nullopt;
    }

    PendingRenderJob job;
    job.requestId = ++_renderRequestSerial;
    job.fbW = fbW;
    job.fbH = fbH;
    job.surfacePtrX = _surfacePtrX;
    job.surfacePtrY = _surfacePtrY;
    job.scale = _scale;
    job.zOff = _zOff;
    job.zOffWorldDir = _zOffWorldDir;
    const bool preferSurfaceResolution = dynamic_cast<PlaneSurface*>(surf.get()) == nullptr;
    job.startLevel = renderStartLevel(preferSurfaceResolution);
    job.overlayStartLevel = overlayRenderStartLevel(preferSurfaceResolution);
    job.samplingMethod = _samplingMethod;
    job.overlaySamplingMethod = _overlaySamplingMethod;
    job.compositeSettings = _compositeSettings;
    job.windowLow = _windowLow;
    job.windowHigh = _windowHigh;
    job.baseColormapId = _baseColormapId;
    job.surf = surf;
    job.surfaceName = _surfName;
    job.chunkArray = _chunkArray;
    job.overlayVolume = _overlayVolume;
    job.overlayChunkArray = _overlayChunkArray;
    job.overlayOpacity = _overlayOpacity;
    job.overlayColormapId = _overlayColormapId;
    job.overlayWindowLow = _overlayWindowLow;
    job.overlayWindowHigh = _overlayWindowHigh;
    job.overlayComposite = _overlayComposite;
    job.chunkContentEpoch = _chunkContentEpoch;
    job.surfaceGeometryEpoch = _surfaceGeometryEpoch;
    job.genCache = _genSurfaceCache;
    job.genCacheDirty = _genCacheDirty;
    job.profileReason = reason ? reason : "";
    if (ProfileLoggingEnabled()) {
        job.profileCaller = profileCaller(caller);
    }
    job.submittedAt = submittedAt;
    return job;
}

bool CChunkedVolumeViewer::renderJobsEquivalentForDisplay(const PendingRenderJob& a,
                                                          const PendingRenderJob& b)
{
    return renderJobsSameGeometry(a, b) &&
           a.chunkContentEpoch == b.chunkContentEpoch &&
           a.genCacheDirty == b.genCacheDirty;
}

bool CChunkedVolumeViewer::renderJobsSameGeometry(const PendingRenderJob& a,
                                                  const PendingRenderJob& b)
{
    auto vecEqual = [](const cv::Vec3f& lhs, const cv::Vec3f& rhs) {
        return lhs[0] == rhs[0] && lhs[1] == rhs[1] && lhs[2] == rhs[2];
    };
    return a.fbW == b.fbW &&
           a.fbH == b.fbH &&
           a.surfacePtrX == b.surfacePtrX &&
           a.surfacePtrY == b.surfacePtrY &&
           a.scale == b.scale &&
           a.zOff == b.zOff &&
           vecEqual(a.zOffWorldDir, b.zOffWorldDir) &&
           a.startLevel == b.startLevel &&
           a.overlayStartLevel == b.overlayStartLevel &&
           a.samplingMethod == b.samplingMethod &&
           a.overlaySamplingMethod == b.overlaySamplingMethod &&
           a.compositeSettings == b.compositeSettings &&
           a.windowLow == b.windowLow &&
           a.windowHigh == b.windowHigh &&
           a.baseColormapId == b.baseColormapId &&
           a.surf.get() == b.surf.get() &&
           a.surfaceName == b.surfaceName &&
           a.chunkArray.get() == b.chunkArray.get() &&
           a.overlayVolume.get() == b.overlayVolume.get() &&
           a.overlayChunkArray.get() == b.overlayChunkArray.get() &&
           a.overlayOpacity == b.overlayOpacity &&
           a.overlayColormapId == b.overlayColormapId &&
           a.overlayWindowLow == b.overlayWindowLow &&
           a.overlayWindowHigh == b.overlayWindowHigh &&
           a.overlayComposite == b.overlayComposite &&
           a.surfaceGeometryEpoch == b.surfaceGeometryEpoch;
}

void CChunkedVolumeViewer::updateDisplayedFramebufferMapping()
{
    if (!_view || !_displayedRenderJob || _displayedRenderJob->scale <= 0.0f) {
        if (_view) {
            _view->setDirectFramebufferMapping(1.0, QPointF(0.0, 0.0));
        }
        return;
    }

    const auto& job = *_displayedRenderJob;
    const qreal ratio = static_cast<qreal>(_scale) / static_cast<qreal>(job.scale);
    const qreal currentHalfW = static_cast<qreal>(_framebuffer.width()) * 0.5;
    const qreal currentHalfH = static_cast<qreal>(_framebuffer.height()) * 0.5;
    const qreal renderedHalfW = static_cast<qreal>(job.fbW) * 0.5;
    const qreal renderedHalfH = static_cast<qreal>(job.fbH) * 0.5;
    const QPointF offset(
        currentHalfW +
            (static_cast<qreal>(job.surfacePtrX) - static_cast<qreal>(_surfacePtrX)) *
                static_cast<qreal>(_scale) -
            renderedHalfW * ratio,
        currentHalfH +
            (static_cast<qreal>(job.surfacePtrY) - static_cast<qreal>(_surfacePtrY)) *
                static_cast<qreal>(_scale) -
            renderedHalfH * ratio);
    _view->setDirectFramebufferMapping(ratio, offset);
}

void CChunkedVolumeViewer::startRenderJob(PendingRenderJob job)
{
    if (_closing || !job.chunkArray) {
        return;
    }

    _renderWorkerBusy.store(true, std::memory_order_release);
    _activeRenderJob = job;
    _pendingRenderDirty = false;

    RenderContext ctx;
    job.chunkArray->beginViewRequest();
    if (job.overlayChunkArray)
        job.overlayChunkArray->beginViewRequest();
    ctx.renderJob = job;
    ctx.serial = ++_renderSerial;
    emit renderFrameSubmitted(ctx.serial);
    ctx.fbW = job.fbW;
    ctx.fbH = job.fbH;
    ctx.surfacePtrX = job.surfacePtrX;
    ctx.surfacePtrY = job.surfacePtrY;
    ctx.scale = job.scale;
    ctx.zOff = job.zOff;
    ctx.zOffWorldDir = job.zOffWorldDir;
    ctx.startLevel = job.startLevel;
    ctx.overlayStartLevel = job.overlayStartLevel;
    ctx.samplingMethod = job.samplingMethod;
    ctx.overlaySamplingMethod = job.overlaySamplingMethod;
    ctx.compositeSettings = job.compositeSettings;
    ctx.windowLow = job.windowLow;
    ctx.windowHigh = job.windowHigh;
    ctx.baseColormapId = job.baseColormapId;
    ctx.surf = job.surf;
    ctx.chunkArray = job.chunkArray;
    ctx.overlayVolume = job.overlayVolume;
    ctx.overlayChunkArray = job.overlayChunkArray;
    ctx.overlayOpacity = job.overlayOpacity;
    ctx.overlayColormapId = job.overlayColormapId;
    ctx.overlayWindowLow = job.overlayWindowLow;
    ctx.overlayWindowHigh = job.overlayWindowHigh;
    ctx.overlayComposite = job.overlayComposite;
    ctx.genCache = job.genCache;
    ctx.genCacheDirty = job.genCacheDirty;
    ctx.prevResult = _lastRenderResult;
    ctx.profileReason = job.profileReason;
    ctx.profileCaller = job.profileCaller;
    _genCacheDirty = false;

    QPointer<CChunkedVolumeViewer> guard(this);
    (void)QtConcurrent::run([guard, ctx = std::move(ctx)]() mutable {
        auto result = std::make_shared<RenderResult>(renderFrame(std::move(ctx)));
        QMetaObject::invokeMethod(qApp, [guard, result = std::move(result)]() mutable {
            if (guard) {
                guard->finishRenderOnMainThread(std::move(result));
            }
        }, Qt::QueuedConnection);
    });
}

void CChunkedVolumeViewer::submitPendingRenderJobIfNeeded()
{
    if (_closing || _renderWorkerBusy.load(std::memory_order_acquire)) {
        return;
    }
    if (!_pendingRenderJob) {
        _pendingRenderDirty = false;
        return;
    }

    auto job = std::move(*_pendingRenderJob);
    _pendingRenderJob.reset();
    if (_displayedRenderJob &&
        renderJobsEquivalentForDisplay(job, *_displayedRenderJob)) {
        _pendingRenderDirty = false;
        return;
    }
    startRenderJob(std::move(job));
}

void CChunkedVolumeViewer::submitRender(const char* reason, std::source_location caller)
{
    ProfileScope profile("submitRender", reason, caller);
    if (profile.enabled()) {
        profile.setDetails(std::format(
            "surf='{}' volume={} chunkArray={} busy={} pending={}",
            _surfName, bool(_volume), bool(_chunkArray),
            _renderWorkerBusy.load(std::memory_order_acquire),
            _pendingRenderJob.has_value()));
    }
    if (_closing) {
        profile.setDetails("action=skip closing");
        return;
    }
    syncCameraTransform();
    if (ProfileLoggingEnabled()) {
        _pendingRenderReason = reason ? reason : "";
        _pendingRenderCaller = profileCaller(caller);
    }
    // Hidden viewer (minimized subwindow / background tab): skip the render.
    // Mark stale so showEvent renders it when it becomes visible.
    if (!isVisible()) {
        _renderStaleWhileHidden = true;
        profile.setDetails("action=skip hidden");
        return;
    }

    auto surf = _surfWeak.lock();
    if (!surf || !_volume || !_chunkArray) {
        profile.setDetails("action=skip missing_input");
        return;
    }

    const bool busy = _renderWorkerBusy.load(std::memory_order_acquire);
    if (!busy) {
        resizeFramebuffer();
    }
    const int fbW = !_framebuffer.isNull()
        ? _framebuffer.width()
        : (_view && _view->viewport() ? std::max(1, _view->viewport()->width()) : 1);
    const int fbH = !_framebuffer.isNull()
        ? _framebuffer.height()
        : (_view && _view->viewport() ? std::max(1, _view->viewport()->height()) : 1);
    auto job = captureRenderJob(reason, caller, surf, fbW, fbH, std::chrono::steady_clock::now());
    if (!job) {
        profile.setDetails("action=skip invalid_job");
        return;
    }

    if (busy) {
        if (_activeRenderJob &&
            renderJobsEquivalentForDisplay(*job, *_activeRenderJob)) {
            profile.setDetails("action=skip_duplicate_active");
            return;
        }
        if (_pendingRenderJob &&
            renderJobsEquivalentForDisplay(*job, *_pendingRenderJob)) {
            _pendingRenderDirty = true;
            profile.setDetails("action=skip_duplicate_pending");
            return;
        }
        _pendingRenderJob = std::move(*job);
        _pendingRenderDirty = true;
        updateStatusLabel();
        profile.setDetails("action=queued_latest_after_worker");
        return;
    }

    if (_displayedRenderJob &&
        renderJobsEquivalentForDisplay(*job, *_displayedRenderJob)) {
        _pendingRenderDirty = false;
        profile.setDetails("action=skip_duplicate_displayed");
        return;
    }

    if (profile.enabled()) {
        profile.setDetails(std::format(
            "action=worker_start serial={} request={} size={}x{} level={}",
            _renderSerial + 1, job->requestId, job->fbW, job->fbH, job->startLevel));
    }
    startRenderJob(std::move(*job));
    updateStatusLabel();
}

void CChunkedVolumeViewer::finishRenderOnMainThread(std::shared_ptr<RenderResult> result)
{
    ProfileScope profile("finishRenderOnMainThread", "render worker finished",
                         std::source_location::current());
    if (profile.enabled()) {
        profile.setDetails(std::format(
            "result={} serial={} currentSerial={} pending={}",
            bool(result), result ? result->serial : 0, _renderSerial,
            _pendingRenderJob.has_value()));
    }
    _renderWorkerBusy.store(false, std::memory_order_release);
    _activeRenderJob.reset();
    if (_closing) {
        profile.setDetails("action=drop_closing");
        return;
    }
    if (!result) {
        submitPendingRenderJobIfNeeded();
        profile.setDetails("action=drop_empty_result");
        return;
    }

    _framebuffer = std::move(result->framebuffer);
    _displayedRenderJob = result->renderJob;
    _lastRenderResult = result;
    syncCameraTransform();
    scheduleIntersectionRender("stable render finished");
    emit overlaysUpdated();
    _view->viewport()->update();
    emit renderFrameCompleted(result->serial, result->renderFrameElapsedMs);
    updateStatusLabel();

    if (_pendingRenderJob &&
        renderJobsEquivalentForDisplay(*_pendingRenderJob, result->renderJob)) {
        _pendingRenderJob.reset();
        _pendingRenderDirty = false;
    }
    if (!_pendingRenderJob) {
        auto surf = _surfWeak.lock();
        if (surf && _volume && _chunkArray) {
            const int fbW = !_framebuffer.isNull()
                ? _framebuffer.width()
                : (_view && _view->viewport() ? std::max(1, _view->viewport()->width()) : 1);
            const int fbH = !_framebuffer.isNull()
                ? _framebuffer.height()
                : (_view && _view->viewport() ? std::max(1, _view->viewport()->height()) : 1);
            auto latest = captureRenderJob("catch up after stale presentation",
                                           std::source_location::current(),
                                           surf,
                                           fbW,
                                           fbH,
                                           std::chrono::steady_clock::now());
            if (latest &&
                !renderJobsEquivalentForDisplay(*latest, result->renderJob)) {
                _pendingRenderJob = std::move(*latest);
                _pendingRenderDirty = true;
            }
        }
    }
    submitPendingRenderJobIfNeeded();

    if (profile.enabled()) {
        profile.setDetails(std::format(
            "action=display serial={} request={} worker_elapsed_ms={} framebuffer={}x{} pending={}",
            result->serial, result->renderJob.requestId, result->renderFrameElapsedMs,
            _framebuffer.width(), _framebuffer.height(), _pendingRenderJob.has_value()));
    }
}

void CChunkedVolumeViewer::renderVisible(bool force, const char* reason, std::source_location caller)
{
    ProfileScope profile("renderVisible", reason, caller);
    if (profile.enabled()) {
        profile.setDetails(std::format(
            "surf='{}' force={} scale={:.4f} zOff={:.3f}",
            _surfName, force, _scale, _zOff));
    }
    if (_closing) {
        profile.setDetails("action=skip closing");
        return;
    }
    submitRender(reason ? reason : (force ? "renderVisible force" : "renderVisible non-force"), caller);
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
    submitRender("volume window changed");
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
    _overlayChunkCacheOwner.reset();
    if (_overlayVolume) {
        try {
            auto lease =
                sharedOverlayChunkCacheForVolume(_overlayVolume, _state);
            if (lease) {
                _overlayChunkArray = lease->cache;
                _overlayChunkCacheOwner = std::move(lease);
            }
        } catch (const std::exception&) {
            _overlayChunkArray.reset();
            _overlayChunkCacheOwner.reset();
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
                    ++guard->_chunkContentEpoch;
                    guard->submitRender("overlay chunk ready");
                }, Qt::QueuedConnection);
            });
        }
    }
    submitRender("overlay volume changed");
}

void CChunkedVolumeViewer::setOverlayOpacity(float opacity)
{
    if (_closing) {
        return;
    }
    _overlayOpacity = std::clamp(opacity, 0.0f, 1.0f);
    submitRender("overlay opacity changed");
}

void CChunkedVolumeViewer::setOverlayColormap(const std::string& colormapId)
{
    if (_closing) {
        return;
    }
    _overlayColormapId = colormapId;
    submitRender("overlay colormap changed");
}

void CChunkedVolumeViewer::setOverlaySamplingMethod(vc::Sampling method)
{
    if (_closing) {
        return;
    }

    const vc::Sampling sanitized = method == vc::Sampling::Trilinear
        ? vc::Sampling::Trilinear
        : vc::Sampling::Nearest;
    if (_overlaySamplingMethod == sanitized) {
        return;
    }

    _overlaySamplingMethod = sanitized;
    submitRender("overlay sampling changed");
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
    submitRender("overlay window changed");
}

void CChunkedVolumeViewer::setOverlayMaxDisplayedResolution(int level)
{
    if (_closing) {
        return;
    }

    const int clamped = std::clamp(level, 0, 5);
    if (_overlayMaxDisplayedResolution == clamped) {
        return;
    }

    _overlayMaxDisplayedResolution = clamped;
    submitRender("overlay max displayed resolution changed");
}

void CChunkedVolumeViewer::setOverlayComposite(const OverlayCompositeSettings& settings)
{
    if (_closing) {
        return;
    }

    if (_overlayComposite == settings) {
        return;
    }

    _overlayComposite = settings;
    submitRender("overlay composite changed");
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
    submitRender("pan");
    refreshSameWrapAnnotationOverlay();
    refreshMeasurementOverlay();
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
    submitRender("zoom");
    refreshSameWrapAnnotationOverlay();
    refreshMeasurementOverlay();
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
    submitRender("interactive view change");
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
    submitRender("surface offset changed");
    updateStatusLabel();
}

void CChunkedVolumeViewer::resetSurfaceOffsets()
{
    _surfacePtrX = 0.0f;
    _surfacePtrY = 0.0f;
    _zOff = 0.0f;
    _zOffWorldDir = {0, 0, 0};
    _genCacheDirty = true;
    submitRender("surface offsets reset");
}

void CChunkedVolumeViewer::fitSurfaceInView()
{
    _surfacePtrX = 0.0f;
    _surfacePtrY = 0.0f;
    _scale = 0.5f;
    recalcPyramidLevel();
    _genCacheDirty = true;
    submitRender("fit surface in view");
}

void CChunkedVolumeViewer::resetViewForCurrentContent(bool forceRender)
{
    if (_closing || !_view) {
        return;
    }

    auto surf = _surfWeak.lock();
    if (!_volume || !surf) {
        return;
    }

    resizeFramebuffer();
    const QSize viewportSize = _view->viewport()->size();
    float minU = 0.0f;
    float maxU = 0.0f;
    float minV = 0.0f;
    float maxV = 0.0f;
    bool haveBounds = false;

    if (dynamic_cast<PlaneSurface*>(surf.get())) {
        updateContentBounds();
        if (_contentMaxU > _contentMinU && _contentMaxV > _contentMinV) {
            minU = _contentMinU;
            maxU = _contentMaxU;
            minV = _contentMinV;
            maxV = _contentMaxV;
            haveBounds = true;
        }
    } else if (auto* quad = dynamic_cast<QuadSurface*>(surf.get())) {
        try {
            quad->ensureLoaded();
            if (const cv::Mat_<cv::Vec3f>* points = quad->rawPointsPtr();
                points && !points->empty()) {
                const cv::Vec3f center = quad->center();
                const cv::Vec2f gridScale = quad->scale();
                minU = -center[0] * gridScale[0];
                minV = -center[1] * gridScale[1];
                maxU = static_cast<float>(points->cols - 1) - center[0] * gridScale[0];
                maxV = static_cast<float>(points->rows - 1) - center[1] * gridScale[1];
                haveBounds = maxU > minU && maxV > minV;
            }
        } catch (const std::exception& e) {
            qWarning() << "Could not fit surface view:" << e.what();
        }
    }

    if (!haveBounds) {
        return;
    }

    _surfacePtrX = (minU + maxU) * 0.5f;
    _surfacePtrY = (minV + maxV) * 0.5f;
    _scale = fitScaleForExtent(maxU - minU,
                               maxV - minV,
                               viewportSize.width(),
                               viewportSize.height());
    _zOff = 0.0f;
    _zOffWorldDir = {0, 0, 0};
    recalcPyramidLevel();
    _genCacheDirty = true;
    updateFocusMarker();
    if (forceRender) {
        renderVisible(true, "reset view for current content");
    } else {
        submitRender("reset view for current content");
    }
    refreshSameWrapAnnotationOverlay();
    emit overlaysUpdated();
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

    const float oldSurfacePtrX = _surfacePtrX;
    const float oldSurfacePtrY = _surfacePtrY;
    _surfacePtrX = point[0];
    _surfacePtrY = point[1];
    _genCacheDirty = true;
    if (forceRender) {
        renderVisible(true, "center on surface point");
    } else {
        const double motionPx = std::hypot(double(_surfacePtrX - oldSurfacePtrX),
                                          double(_surfacePtrY - oldSurfacePtrY)) *
                                double(std::max(_scale, kMinScale));
        markInteractiveMotion(motionPx);
        submitRender("center on surface point");
    }
    emit overlaysUpdated();
}

void CChunkedVolumeViewer::onZoom(int steps, QPointF scenePoint, Qt::KeyboardModifiers modifiers)
{
    auto surf = _surfWeak.lock();
    if (!surf)
        return;
    if (modifiers & Qt::ShiftModifier) {
        if (_shiftScrollOverride && _shiftScrollOverride(steps, scenePoint, modifiers)) {
            refreshCursorPositionAt(scenePoint);
            return;
        }
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
                    submitRender("plane slice mouse wheel");
                }
            }
        } else {
            _zOff += static_cast<float>(steps) * _zScrollSensitivity;
            _genCacheDirty = true;
            submitRender("z offset mouse wheel");
        }
        refreshCursorPositionAt(scenePoint);
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
    submitRender("resized");
    refreshMeasurementOverlay();
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
    markInteractiveMotion(0.0);
    _lastPanSceneF = _view->mapToScene(_view->mapFromGlobal(QCursor::pos()));
}

void CChunkedVolumeViewer::onPanRelease(Qt::MouseButton, Qt::KeyboardModifiers)
{
    _isPanning = false;
    _panSmoothingInitialized = false;
    _smoothedPanDx = 0.0f;
    _smoothedPanDy = 0.0f;
    submitRender("pan released");
}

void CChunkedVolumeViewer::onVolumeClicked(QPointF scenePos, Qt::MouseButton button, Qt::KeyboardModifiers modifiers)
{
    if (_sameWrapAnnotation.enabled() && button == Qt::LeftButton && modifiers.testFlag(Qt::ShiftModifier)) {
        if (_sameWrapAnnotation.manualPathType()) {
            return;
        }
        const bool appendToPreview = _sameWrapAnnotation.hasPreview() &&
                                     !_sameWrapAnnotation.shiftReleasedSincePreview();
        if (_sameWrapAnnotation.generatePreview(
                _framebuffer,
                scenePos,
                appendToPreview,
                _scale,
                _pointCollection,
                [this](const QPointF& point) { return sceneToVolume(point); },
                [this](const cv::Vec3f& point) { return volumeToScene(point); },
                [this](const std::string& key, const std::vector<QGraphicsItem*>& items) {
                    setOverlayGroup(key, items);
                },
                [this](const std::string& key) { clearOverlayGroup(key); })) {
            return;
        }
    }

    if (button == Qt::LeftButton && modifiers == Qt::NoModifier) {
        if (const auto hit = pointAtScenePosition(scenePos)) {
            _selectedCollectionId = hit->first;
            _selectedPointId = hit->second;
            emit pointClicked(_selectedPointId);
            emit overlaysUpdated();
            return;
        }
    }

    const auto sample = sampleSceneVolume(scenePos);
    if (!sample) {
        return;
    }
    emit sendVolumeClicked(sample->position, sample->normal, sample->surface, button, modifiers);
}

void CChunkedVolumeViewer::onCollectionSelected(uint64_t collectionId)
{
    _selectedCollectionId = collectionId;
    if (_selectedPointId != 0) {
        const auto point = _pointCollection ? _pointCollection->getPoint(_selectedPointId)
                                            : std::optional<ColPoint>{};
        if (!point || point->collectionId != collectionId) {
            _selectedPointId = 0;
        }
    }
    emit overlaysUpdated();
}

void CChunkedVolumeViewer::onPointSelected(uint64_t pointId)
{
    _selectedPointId = pointId;
    if (_pointCollection && pointId != 0) {
        if (const auto point = _pointCollection->getPoint(pointId)) {
            _selectedCollectionId = point->collectionId;
        }
    }
    emit overlaysUpdated();
}

void CChunkedVolumeViewer::setSameWrapAnnotationMode(bool enabled)
{
    _sameWrapManualPathDragActive = false;
    _sameWrapAnnotation.setEnabled(enabled);
    if (!enabled) {
        clearSameWrapAnnotationPreview();
    }
    emit overlaysUpdated();
}

void CChunkedVolumeViewer::setSameWrapAnnotationSpacing(double spacingVx)
{
    _sameWrapAnnotation.setSpacing(spacingVx);
}

void CChunkedVolumeViewer::setSameWrapAnnotationPolylineOpacity(double opacity)
{
    _sameWrapAnnotationPolylineOpacity = std::clamp(opacity, 0.0, 1.0);
    emit overlaysUpdated();
}

void CChunkedVolumeViewer::setSameWrapAnnotationMergeExisting(bool enabled)
{
    _sameWrapAnnotation.setMergeExistingAnnotations(enabled);
}

void CChunkedVolumeViewer::setSameWrapAnnotationPathType(int pathType)
{
    SameWrapAnnotationTool::PathType toolPathType = SameWrapAnnotationTool::PathType::ConnectedComponents;
    if (pathType == static_cast<int>(SameWrapAnnotationTool::PathType::ShortestPath)) {
        toolPathType = SameWrapAnnotationTool::PathType::ShortestPath;
    } else if (pathType == static_cast<int>(SameWrapAnnotationTool::PathType::Manual)) {
        toolPathType = SameWrapAnnotationTool::PathType::Manual;
    }
    _sameWrapManualPathDragActive = false;
    _sameWrapAnnotation.setPathType(toolPathType);
    clearSameWrapAnnotationPreview();
}

void CChunkedVolumeViewer::setSameWrapAnnotationFilterType(int filterType)
{
    SameWrapAnnotationTool::ImageFilterType toolFilterType = SameWrapAnnotationTool::ImageFilterType::None;
    if (filterType == static_cast<int>(SameWrapAnnotationTool::ImageFilterType::Median)) {
        toolFilterType = SameWrapAnnotationTool::ImageFilterType::Median;
    } else if (filterType == static_cast<int>(SameWrapAnnotationTool::ImageFilterType::Gaussian)) {
        toolFilterType = SameWrapAnnotationTool::ImageFilterType::Gaussian;
    }
    _sameWrapAnnotation.setImageFilterType(toolFilterType);
    clearSameWrapAnnotationPreview();
}

void CChunkedVolumeViewer::setSameWrapAnnotationFilterKernelSize(int kernelSize)
{
    _sameWrapAnnotation.setImageFilterKernelSize(kernelSize);
    clearSameWrapAnnotationPreview();
}

bool CChunkedVolumeViewer::hasSameWrapAnnotationPreview() const
{
    return _sameWrapAnnotation.hasPreview();
}

void CChunkedVolumeViewer::clearSameWrapAnnotationPreview()
{
    _sameWrapManualPathDragActive = false;
    _sameWrapAnnotation.clear([this](const std::string& key) { clearOverlayGroup(key); });
}

bool CChunkedVolumeViewer::commitSameWrapAnnotationPreview()
{
    return _sameWrapAnnotation.commit(
        _pointCollection,
        [this](const cv::Vec3f& point) { return volumeToScene(point); },
        QRectF(0.0, 0.0, _framebuffer.width(), _framebuffer.height()),
        [this](const std::string& key) { clearOverlayGroup(key); });
}

bool CChunkedVolumeViewer::undoSameWrapAnnotation()
{
    if (_sameWrapAnnotation.hasPreview()) {
        clearSameWrapAnnotationPreview();
        return true;
    }
    return _sameWrapAnnotation.undoLastCommit(_pointCollection);
}

void CChunkedVolumeViewer::refreshSameWrapAnnotationOverlay()
{
    if (!_sameWrapAnnotation.hasPreview()) {
        return;
    }

    _sameWrapAnnotation.refreshOverlay(
        [this](const cv::Vec3f& point) { return volumeToScene(point); },
        [this](const std::string& key, const std::vector<QGraphicsItem*>& items) {
            setOverlayGroup(key, items);
        },
        [this](const std::string& key) { clearOverlayGroup(key); });
}

void CChunkedVolumeViewer::onMousePress(QPointF scenePos, Qt::MouseButton button, Qt::KeyboardModifiers modifiers)
{
    _lastScenePos = scenePos;
    _lastCursorVolumePos = cursorVolumePosition(scenePos);
    updateCursorCrosshair(scenePos);
    updateStatusLabel();
    if (_viewerManager) {
        _viewerManager->broadcastLinkedCursor(this, _lastCursorVolumePos);
    }
    _sameWrapManualMergePressConsumed = false;

    if (handleMeasurementClick(scenePos, button, modifiers)) {
        return;
    }

    if (_sameWrapAnnotation.enabled() && button == Qt::RightButton &&
        modifiers.testFlag(Qt::ShiftModifier)) {
        if (const auto hit = pointAtScenePosition(scenePos)) {
            if (_sameWrapAnnotation.manualMergePointClicked(
                    _pointCollection,
                    hit->first,
                    hit->second,
                    [this](const cv::Vec3f& point) { return volumeToScene(point); },
                    QRectF(0.0, 0.0, _framebuffer.width(), _framebuffer.height()),
                    [this](const SameWrapAnnotationTool::MixedDirectionMergeWarning& warning) {
                        const auto directionText = [this](const std::string& key) {
                            return key == "z" ? tr("mostly up/down in Z") : tr("mostly in XY");
                        };
                        const QMessageBox::StandardButton reply = QMessageBox::warning(
                            this,
                            tr("Same-wrap Merge"),
                            tr("You are about to merge two same-wrap point collections with different dominant directions:\n\n"
                               "%1: %2\n"
                               "%3: %4\n\n"
                               "Continue merging?")
                                .arg(QString::fromStdString(warning.firstCollectionName),
                                     directionText(warning.firstDirectionKey),
                                     QString::fromStdString(warning.secondCollectionName),
                                     directionText(warning.secondDirectionKey)),
                            QMessageBox::Yes | QMessageBox::No,
                            QMessageBox::No);
                        return reply == QMessageBox::Yes;
                    })) {
                if (_pointCollection && _pointCollection->getPoint(hit->second)) {
                    _selectedCollectionId = hit->first;
                    _selectedPointId = hit->second;
                } else {
                    _selectedCollectionId = 0;
                    _selectedPointId = 0;
                }
                _sameWrapManualMergePressConsumed = true;
                if (_selectedPointId != 0) {
                    emit pointClicked(_selectedPointId);
                }
                emit overlaysUpdated();
                return;
            }
        }
    }

    if (_sameWrapAnnotation.enabled() && button == Qt::LeftButton &&
        modifiers.testFlag(Qt::ShiftModifier)) {
        if (_sameWrapAnnotation.manualPathType()) {
            const bool appendToPreview = _sameWrapAnnotation.hasPreview() &&
                                         !_sameWrapAnnotation.shiftReleasedSincePreview();
            _sameWrapManualPathDragActive = _sameWrapAnnotation.beginManualPreview(
                scenePos,
                appendToPreview,
                [this](const QPointF& point) { return sceneToVolume(point); },
                [this](const cv::Vec3f& point) { return volumeToScene(point); },
                [this](const std::string& key, const std::vector<QGraphicsItem*>& items) {
                    setOverlayGroup(key, items);
                },
                [this](const std::string& key) { clearOverlayGroup(key); });
        }
        return;
    }

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
    if (_lineAnnotationPlacementPreviewEnabled && button == Qt::LeftButton &&
        modifiers == Qt::NoModifier &&
        std::isfinite(volumePos[0]) && std::isfinite(volumePos[1]) &&
        std::isfinite(volumePos[2])) {
        emit sendLineAnnotationSeedRequested(volumePos, scenePos);
    }
}

void CChunkedVolumeViewer::onMouseMove(QPointF scenePos, Qt::MouseButtons buttons, Qt::KeyboardModifiers modifiers)
{
    const bool reusedCursorSample = (_lastScenePos == scenePos && _lastCursorVolumePos.has_value());
    if (!reusedCursorSample) {
        _lastCursorVolumePos = cursorVolumePosition(scenePos);
    }
    _lastScenePos = scenePos;
    updateCursorCrosshair(scenePos);
    updateStatusLabel();

    const uint64_t previousHighlight = _highlightedPointId;
    if (const auto hit = pointAtScenePosition(scenePos)) {
        _highlightedPointId = hit->second;
    } else {
        _highlightedPointId = 0;
    }
    if (_highlightedPointId != previousHighlight) {
        emit overlaysUpdated();
    }

    if (_sameWrapManualPathDragActive && (buttons & Qt::LeftButton)) {
        _sameWrapAnnotation.appendManualPreview(
            scenePos,
            _scale,
            _pointCollection,
            [this](const QPointF& point) { return sceneToVolume(point); },
            [this](const cv::Vec3f& point) { return volumeToScene(point); },
            [this](const std::string& key, const std::vector<QGraphicsItem*>& items) {
                setOverlayGroup(key, items);
            },
            [this](const std::string& key) { clearOverlayGroup(key); });
        return;
    }
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
    if (_lineAnnotationPlacementPreviewEnabled &&
        std::isfinite(volumePos[0]) && std::isfinite(volumePos[1]) &&
        std::isfinite(volumePos[2])) {
        updateLineAnnotationPlacementMarker(scenePos);
    } else {
        clearLineAnnotationPlacementMarker();
    }
    emit sendMouseMoveVolume(volumePos, buttons, modifiers, scenePos);
}

void CChunkedVolumeViewer::onMouseRelease(QPointF scenePos, Qt::MouseButton button, Qt::KeyboardModifiers modifiers)
{
    _lastScenePos = scenePos;
    _lastCursorVolumePos = cursorVolumePosition(scenePos);
    updateCursorCrosshair(scenePos);
    updateStatusLabel();
    if (_viewerManager) {
        _viewerManager->broadcastLinkedCursor(this, _lastCursorVolumePos);
    }
    if (_sameWrapManualMergePressConsumed && button == Qt::RightButton) {
        _sameWrapManualMergePressConsumed = false;
        return;
    }
    if (_sameWrapManualPathDragActive && button == Qt::LeftButton) {
        _sameWrapAnnotation.appendManualPreview(
            scenePos,
            _scale,
            _pointCollection,
            [this](const QPointF& point) { return sceneToVolume(point); },
            [this](const cv::Vec3f& point) { return volumeToScene(point); },
            [this](const std::string& key, const std::vector<QGraphicsItem*>& items) {
                setOverlayGroup(key, items);
            },
            [this](const std::string& key) { clearOverlayGroup(key); });
        _sameWrapManualPathDragActive = false;
        return;
    }
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
    if (key == Qt::Key_Escape && isMeasurementActive()) {
        clearMeasurement();
        return;
    }

    constexpr float kPanPx = 64.0f;
    switch (key) {
        case Qt::Key_Left: panByF(kPanPx, 0); break;
        case Qt::Key_Right: panByF(-kPanPx, 0); break;
        case Qt::Key_Up: panByF(0, kPanPx); break;
        case Qt::Key_Down: panByF(0, -kPanPx); break;
        default: break;
    }
}

void CChunkedVolumeViewer::onKeyRelease(int key, Qt::KeyboardModifiers)
{
    if (key == Qt::Key_Shift) {
        _sameWrapAnnotation.noteShiftReleased();
    }
}

QPointF CChunkedVolumeViewer::surfaceToScene(float surfX, float surfY) const
{
    // Keep the camera arithmetic in qreal. Overlay code derives an affine
    // surface-to-scene transform from this function; doing these operations in
    // float makes its one-unit basis lose precision far from surface origin,
    // and that error is then multiplied by the absolute surface coordinate.
    const qreal vpCx = static_cast<qreal>(_framebuffer.width()) * 0.5;
    const qreal vpCy = static_cast<qreal>(_framebuffer.height()) * 0.5;
    const qreal scale = static_cast<qreal>(_scale);
    const qreal vx = (static_cast<qreal>(surfX) - static_cast<qreal>(_surfacePtrX))
                   * scale + vpCx;
    const qreal vy = (static_cast<qreal>(surfY) - static_cast<qreal>(_surfacePtrY))
                   * scale + vpCy;
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
        cv::Vec3f ptr = quad->pointer();
        auto* patchIndex = _viewerManager ? _viewerManager->surfacePatchIndex() : nullptr;
        // Match the points-overlay default view tolerance so points that are meant to
        // render faded still project to their true location rather than collapsing.
        constexpr float kQuadProjectTolerance = 10.0f;
        // pointTo() with a patch index signals "no surface point within tolerance" by
        // returning a positive value (~the tolerance) WITHOUT updating ptr, so a bare
        // `< 0.0f` check would silently keep ptr at {0,0,0} and map the point to the
        // segment center. Treat anything outside the tolerance as a failed projection.
        const float dist = quad->pointTo(ptr, volPoint, kQuadProjectTolerance, 100, patchIndex);
        if (dist < 0.0f || dist > kQuadProjectTolerance)
            return {NAN, NAN};
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

void CChunkedVolumeViewer::setSegmentationCursorMirroring(bool enabled)
{
    _segmentationCursorMirroring = enabled;
    if (!enabled && _cursorCrosshair) {
        _cursorCrosshair->hide();
    }
}

void CChunkedVolumeViewer::setLineAnnotationPlacementPreviewEnabled(bool enabled)
{
    _lineAnnotationPlacementPreviewEnabled = enabled;
    if (!enabled) {
        clearLineAnnotationPlacementMarker();
    }
}

bool CChunkedVolumeViewer::lineAnnotationPlacementMarkerVisible() const
{
    return _lineAnnotationPlacementMarker && _lineAnnotationPlacementMarker->isVisible();
}

void CChunkedVolumeViewer::markSurfaceGeometryChanged()
{
    if (_closing) {
        return;
    }
    ++_surfaceGeometryEpoch;
    _genCacheDirty = true;
}

void CChunkedVolumeViewer::updateLineAnnotationPlacementMarker(const QPointF& scenePos)
{
    if (!_lineAnnotationPlacementPreviewEnabled || !_scene ||
        !std::isfinite(scenePos.x()) || !std::isfinite(scenePos.y())) {
        clearLineAnnotationPlacementMarker();
        return;
    }

    if (!_lineAnnotationPlacementMarker || !_lineAnnotationPlacementMarker->scene()) {
        auto* marker = new QGraphicsEllipseItem(-6.0, -6.0, 12.0, 12.0);
        QPen pen(QColor(Qt::yellow), 2.0);
        pen.setCosmetic(true);
        marker->setPen(pen);
        marker->setBrush(QBrush(QColor(255, 255, 0, 70)));
        marker->setZValue(135.0);
        marker->setAcceptedMouseButtons(Qt::NoButton);
        _scene->addItem(marker);
        _lineAnnotationPlacementMarker = marker;
    }

    _lineAnnotationPlacementMarker->setPos(scenePos);
    _lineAnnotationPlacementMarker->show();
}

void CChunkedVolumeViewer::clearLineAnnotationPlacementMarker()
{
    if (_lineAnnotationPlacementMarker) {
        _lineAnnotationPlacementMarker->hide();
    }
}

void CChunkedVolumeViewer::startMeasurementMode()
{
    if (!measurementSupported()) {
        return;
    }
    _measurement = {};
    _measurement.active = true;
    clearOverlayGroup(kMeasurementOverlayKey);
    if (_view) {
        _view->setFocus(Qt::MouseFocusReason);
        _view->setCursor(Qt::CrossCursor);
    }
}

bool CChunkedVolumeViewer::measurementSupported() const
{
    auto surf = _surfWeak.lock();
    if (!surf) {
        return false;
    }
    return _surfName == "segmentation" ||
           dynamic_cast<PlaneSurface*>(surf.get()) != nullptr;
}

bool CChunkedVolumeViewer::isMeasurementActive() const
{
    return _measurement.active;
}

void CChunkedVolumeViewer::clearMeasurement()
{
    _measurement = {};
    clearOverlayGroup(kMeasurementOverlayKey);
    if (_view) {
        _view->unsetCursor();
        _view->viewport()->update();
    }
}

bool CChunkedVolumeViewer::handleMeasurementClick(const QPointF& scenePos,
                                                  Qt::MouseButton button,
                                                  Qt::KeyboardModifiers modifiers)
{
    if (!_measurement.active || !measurementSupported()) {
        return false;
    }
    if (button != Qt::LeftButton || modifiers != Qt::NoModifier) {
        return true;
    }

    const cv::Vec3f volumePos = _lastCursorVolumePos ? *_lastCursorVolumePos : sceneToVolume(scenePos);
    if (!std::isfinite(volumePos[0]) ||
        !std::isfinite(volumePos[1]) ||
        !std::isfinite(volumePos[2])) {
        return true;
    }

    MeasurementPoint point;
    point.surface = sceneToSurface(scenePos);
    point.volume = volumePos;
    if (!_measurement.first || _measurement.second) {
        _measurement.first = point;
        _measurement.second.reset();
    } else {
        _measurement.second = point;
    }
    refreshMeasurementOverlay();
    return true;
}

void CChunkedVolumeViewer::refreshMeasurementOverlay()
{
    if (!_measurement.active && !_measurement.first && !_measurement.second) {
        clearOverlayGroup(kMeasurementOverlayKey);
        return;
    }

    std::vector<QGraphicsItem*> items;
    auto makeMarker = [](const QPointF& pos) {
        auto* marker = new QGraphicsEllipseItem(-4.0, -4.0, 8.0, 8.0);
        QPen pen(QColor(255, 230, 80), 1.5);
        pen.setCosmetic(true);
        marker->setPen(pen);
        marker->setBrush(QBrush(QColor(255, 230, 80, 90)));
        marker->setPos(pos);
        marker->setZValue(145.0);
        marker->setAcceptedMouseButtons(Qt::NoButton);
        return marker;
    };

    std::optional<QPointF> firstScene;
    std::optional<QPointF> secondScene;
    if (_measurement.first) {
        firstScene = surfaceToScene(_measurement.first->surface[0], _measurement.first->surface[1]);
        if (std::isfinite(firstScene->x()) && std::isfinite(firstScene->y())) {
            items.push_back(makeMarker(*firstScene));
        }
    }
    if (_measurement.second) {
        secondScene = surfaceToScene(_measurement.second->surface[0], _measurement.second->surface[1]);
        if (std::isfinite(secondScene->x()) && std::isfinite(secondScene->y())) {
            items.push_back(makeMarker(*secondScene));
        }
    }

    if (firstScene && secondScene &&
        std::isfinite(firstScene->x()) && std::isfinite(firstScene->y()) &&
        std::isfinite(secondScene->x()) && std::isfinite(secondScene->y())) {
        QPainterPath path;
        path.moveTo(*firstScene);
        path.lineTo(*secondScene);
        auto* line = new QGraphicsPathItem(path);
        QPen pen(QColor(255, 230, 80), 2.0, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin);
        pen.setCosmetic(true);
        line->setPen(pen);
        line->setBrush(Qt::NoBrush);
        line->setZValue(144.0);
        line->setAcceptedMouseButtons(Qt::NoButton);
        items.push_back(line);

        const double um = _volume
            ? volumeDistanceMicrometers(_measurement.first->volume,
                                        _measurement.second->volume,
                                        _volume->voxelSize())
            : std::numeric_limits<double>::quiet_NaN();
        if (std::isfinite(um)) {
            const QPointF mid = (*firstScene + *secondScene) * 0.5;
            const QPointF delta = *secondScene - *firstScene;
            const double len = std::hypot(delta.x(), delta.y());
            QPointF offset(0.0, -16.0);
            if (len > 1.0) {
                offset = QPointF(-delta.y() / len * 16.0, delta.x() / len * 16.0);
                if (offset.y() > 0.0) {
                    offset = -offset;
                }
            }
            const QString label = CVolumeViewerView::formatScaleBarLength(um).text;
            QFont font;
            font.setPointSize(12);
            font.setBold(true);

            auto* shadow = new QGraphicsSimpleTextItem(label);
            shadow->setFont(font);
            shadow->setBrush(QBrush(Qt::black));
            shadow->setPos(mid + offset + QPointF(1.0, 1.0));
            shadow->setZValue(145.0);
            shadow->setAcceptedMouseButtons(Qt::NoButton);
            items.push_back(shadow);

            auto* text = new QGraphicsSimpleTextItem(label);
            text->setFont(font);
            text->setBrush(QBrush(Qt::white));
            text->setPos(mid + offset);
            text->setZValue(146.0);
            text->setAcceptedMouseButtons(Qt::NoButton);
            items.push_back(text);
        }
    }

    setOverlayGroup(kMeasurementOverlayKey, items);
    if (_view) {
        _view->viewport()->update();
    }
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

void CChunkedVolumeViewer::refreshCursorPositionAt(const QPointF& scenePos)
{
    _lastScenePos = scenePos;
    _lastCursorVolumePos = cursorVolumePosition(scenePos);
    updateCursorCrosshair(scenePos);
    updateStatusLabel();
    if (_viewerManager) {
        _viewerManager->broadcastLinkedCursor(this, _lastCursorVolumePos);
    }
}

std::optional<std::pair<uint64_t, uint64_t>> CChunkedVolumeViewer::pointAtScenePosition(const QPointF& scenePos)
{
    if (!_pointCollection || !std::isfinite(scenePos.x()) || !std::isfinite(scenePos.y())) {
        return std::nullopt;
    }

    const auto& collections = _pointCollection->getAllCollections();
    if (collections.empty()) {
        return std::nullopt;
    }

    constexpr qreal kPointHitRadius = 4.0;
    constexpr qreal kPointHitRadiusSq = kPointHitRadius * kPointHitRadius;

    // volumeToScene() is expensive on a QuadSurface (a per-point pointTo search) and
    // this runs on every mouse-move for hover highlighting. Reject points that are
    // obviously too far before paying for the projection: the cursor's volume position
    // lies on the current surface, so a marker can only fall within kPointHitRadius
    // scene px if it is within ~kPointHitRadius/_scale surface units laterally, plus a
    // normal-direction margin covering the depth band that is actually drawn (markers
    // fade out past the view tolerance, whose generous ceiling we bound below).
    const std::optional<cv::Vec3f> cursorVol = cursorVolumePosition(scenePos);
    constexpr float kHoverDepthMarginVx = 256.0f;
    const float lateralBoundVx = static_cast<float>(kPointHitRadius) / std::max(_scale, 1e-3f);
    const float prefilterRadiusVx = lateralBoundVx + kHoverDepthMarginVx;
    const float prefilterRadiusSqVx = prefilterRadiusVx * prefilterRadiusVx;

    qreal bestDistSq = kPointHitRadiusSq;
    std::optional<std::pair<uint64_t, uint64_t>> bestHit;

    for (const auto& [collectionId, collection] : collections) {
        for (const auto& [pointId, point] : collection.points) {
            if (cursorVol) {
                const cv::Vec3f delta = point.p - *cursorVol;
                if (delta.dot(delta) > prefilterRadiusSqVx) {
                    continue;
                }
            }

            const QPointF pointScenePos = volumeToScene(point.p);
            if (!std::isfinite(pointScenePos.x()) || !std::isfinite(pointScenePos.y())) {
                continue;
            }

            const qreal dx = pointScenePos.x() - scenePos.x();
            const qreal dy = pointScenePos.y() - scenePos.y();
            const qreal distSq = dx * dx + dy * dy;
            if (distSq <= bestDistSq) {
                bestDistSq = distSq;
                bestHit = std::make_pair(collectionId, pointId);
            }
        }
    }

    return bestHit;
}

void CChunkedVolumeViewer::setLinkedCursorVolumePoint(const std::optional<cv::Vec3f>& point)
{
    auto hideCrosshair = [this]() {
        if (_cursorCrosshair) {
            _cursorCrosshair->hide();
        }
    };

    if (!_segmentationCursorMirroring || !point) {
        hideCrosshair();
        return;
    }

    QPointF scenePos;
    if (auto* plane = dynamic_cast<PlaneSurface*>(currentSurface())) {
        const cv::Vec3f projected = plane->project(*point, 1.0f, 1.0f);
        if (!std::isfinite(projected[0]) ||
            !std::isfinite(projected[1]) ||
            !std::isfinite(projected[2]) ||
            std::abs(projected[2] - _zOff) > _linkedCursorViewTolerance) {
            hideCrosshair();
            return;
        }
        scenePos = surfaceToScene(projected[0], projected[1]);
    } else {
        scenePos = volumeToScene(*point);
    }

    if (!std::isfinite(scenePos.x()) || !std::isfinite(scenePos.y())) {
        hideCrosshair();
        return;
    }

    updateCursorCrosshair(scenePos);
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

std::optional<CChunkedVolumeViewer::SceneVolumeSample> CChunkedVolumeViewer::sampleSceneVolume(
    const QPointF& scenePoint) const
{
    auto surf = _surfWeak.lock();
    if (!surf) {
        return std::nullopt;
    }

    const auto cursorPos = cursorVolumePosition(scenePoint);
    SceneVolumeSample sample;
    sample.position = cursorPos ? *cursorPos : sceneToVolume(scenePoint);
    sample.surface = surf.get();
    if (!std::isfinite(sample.position[0]) ||
        !std::isfinite(sample.position[1]) ||
        !std::isfinite(sample.position[2])) {
        return std::nullopt;
    }

    if (auto* plane = dynamic_cast<PlaneSurface*>(surf.get())) {
        sample.normal = plane->normal({0, 0, 0});
    } else {
        const cv::Vec2f sp = sceneToSurface(scenePoint);
        const cv::Vec3f surfaceNormal = surf->normal({0, 0, 0}, {sp[0], sp[1], 0.0f});
        if (std::isfinite(surfaceNormal[0]) &&
            std::isfinite(surfaceNormal[1]) &&
            std::isfinite(surfaceNormal[2]) &&
            cv::norm(surfaceNormal) > 0.0f) {
            sample.normal = surfaceNormal;
        }
    }
    if (!std::isfinite(sample.normal[0]) ||
        !std::isfinite(sample.normal[1]) ||
        !std::isfinite(sample.normal[2]) ||
        cv::norm(sample.normal) <= 0.0f) {
        sample.normal = {0, 0, 1};
    }
    return sample;
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
    _resolvedIntersectTargets = {};
    invalidateIntersect();
    renderIntersections("setIntersects");
}

void CChunkedVolumeViewer::setPlaneIntersectionLinesVisible(bool visible)
{
    if (_closing || _planeIntersectionLinesVisible == visible) {
        return;
    }
    _planeIntersectionLinesVisible = visible;
    _lastIntersectFp = {};
    auto surf = _surfWeak.lock();
    const bool isPlaneViewer = dynamic_cast<PlaneSurface*>(surf.get()) != nullptr;
    if (!visible && !isPlaneViewer) {
        clearIntersectionItems();
        if (_view) {
            _view->viewport()->update();
        }
        return;
    }
    if (visible && !isPlaneViewer) {
        renderIntersections("setPlaneIntersectionLinesVisible");
    }
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
        const std::size_t sampledRows =
            std::size_t(points->rows / stride + 1);
        const std::size_t sampledColumns =
            std::size_t(points->cols / stride + 1);
        const std::size_t sampledCells = sampledRows * sampledColumns;
        // A plane normally crosses a surface along a curve, so populated cells
        // scale with the sampled perimeter rather than with the entire grid.
        // Let pathological surfaces grow the map naturally instead of eagerly
        // reserving buckets for a fixed fraction of every cell.
        const std::size_t expectedIntersectedCells =
            planes.size() * 2 * (sampledRows + sampledColumns) + 1024;
        _flattenedIntersectionCache.cellLines.reserve(
            std::min(sampledCells, expectedIntersectedCells));
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

    auto appendLinesForCell = [&](std::uint64_t tileKey, std::uint64_t cellKey) {
        auto pathsIt = tilePaths.find(tileKey);
        if (pathsIt == tilePaths.end()) {
            return;
        }
        const auto linesIt = _flattenedIntersectionCache.cellLines.find(cellKey);
        if (linesIt == _flattenedIntersectionCache.cellLines.end()) {
            return;
        }
        const auto dirtyIt = dirtyTilePlanes.find(tileKey);
        if (dirtyIt == dirtyTilePlanes.end()) {
            return;
        }

        auto& paths = pathsIt->second;
        for (const auto& line : linesIt->second) {
            if (line.planeIndex < 0 || static_cast<size_t>(line.planeIndex) >= paths.size()) {
                continue;
            }
            if (dirtyIt->second.count(line.planeIndex) == 0) {
                continue;
            }
            paths[line.planeIndex].moveTo(line.a);
            paths[line.planeIndex].lineTo(line.b);
        }
    };

    if (rebuildAllTileItems) {
        for (const auto& [cellKey, lines] : _flattenedIntersectionCache.cellLines) {
            const int col = int(std::uint32_t(cellKey));
            const int row = int(std::uint32_t(cellKey >> 32));
            appendLinesForCell(surfaceCellTileKey(col, row), cellKey);
        }
    } else {
        for (const auto& [tileKey, _] : dirtyTilePlanes) {
            const int tileCol = int(std::uint32_t(tileKey));
            const int tileRow = int(std::uint32_t(tileKey >> 32));
            const int colStart = std::max(0, tileCol * kSurfaceCellTileSize);
            const int rowStart = std::max(0, tileRow * kSurfaceCellTileSize);
            const int colEnd = std::min(points->cols - 1, colStart + kSurfaceCellTileSize);
            const int rowEnd = std::min(points->rows - 1, rowStart + kSurfaceCellTileSize);
            for (int row = rowStart; row < rowEnd; ++row) {
                for (int col = colStart; col < colEnd; ++col) {
                    appendLinesForCell(tileKey, surfaceTileKey(col, row));
                }
            }
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
    if (property("vc_viewer_role").toString() == QStringLiteral("annotation")) {
        clearIntersectionItems();
        _lastIntersectFp = {};
        profile.setDetails("action=skip annotation_viewer");
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
    if (!plane && !_planeIntersectionLinesVisible) {
        clearIntersectionItems();
        _lastIntersectFp = {};
        profile.setDetails("action=skip flattened_disabled");
        return;
    }

    auto* patchIndex = _viewerManager->surfacePatchIndex();
    auto* editPatchIndex = _viewerManager->activeSegmentationEditSurfacePatchIndex();
    if ((!patchIndex || patchIndex->empty()) && (!editPatchIndex || editPatchIndex->empty())) {
        invalidateIntersect();
        _lastIntersectFp = {};
        profile.setDetails("action=skip empty_patch_index");
        return;
    }

    auto activeSeg = std::dynamic_pointer_cast<QuadSurface>(_state->surface("segmentation"));
    if (!plane) {
        renderFlattenedIntersections(surf, reason, caller);
        profile.setDetails("action=delegated_flattened");
        return;
    }

    const std::uint64_t surfacesVersion = _state->surfacesVersion();
    if (!_resolvedIntersectTargets.valid ||
        _resolvedIntersectTargets.activeSeg != activeSeg.get() ||
        _resolvedIntersectTargets.surfacesVersion != surfacesVersion) {
        auto& resolved = _resolvedIntersectTargets;
        resolved.targets.clear();
        auto addTarget = [&](const std::string& name) {
            if (auto quad = std::dynamic_pointer_cast<QuadSurface>(_state->surface(name))) {
                if (activeSeg && quad != activeSeg && !activeSeg->id.empty() &&
                    quad->id == activeSeg->id) {
                    // During editing, named/highlighted surface lookups may resolve
                    // to the saved surface object while the live geometry is in the
                    // active segmentation preview. Target the preview instead of
                    // dropping the surface for this frame.
                    resolved.targets.insert(activeSeg);
                    return;
                }
                resolved.targets.insert(std::move(quad));
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
        std::size_t th = 0;
        for (const auto& t : resolved.targets)
            th ^= std::hash<const void*>{}(t.get()) + 0x9e3779b9u + (th << 6) + (th >> 2);
        resolved.targetHash = th;
        resolved.activeSeg = activeSeg.get();
        resolved.surfacesVersion = surfacesVersion;
        resolved.valid = true;
    }
    const auto& targets = _resolvedIntersectTargets.targets;
    if (targets.empty()) {
        invalidateIntersect();
        _lastIntersectFp = {};
        profile.setDetails("action=skip no_targets");
        return;
    }

    const bool useEditIndexForActive = _segmentationEditActive && editPatchIndex && !editPatchIndex->empty() &&
        activeSeg && targets.find(activeSeg) != targets.end();
    if (useEditIndexForActive) {
        // Flush before the fingerprint below so it observes the post-flush
        // edit-index generation.
        editPatchIndex->flushPendingUpdates(activeSeg);
    }
    auto* primaryPatchIndex = patchIndex ? patchIndex : editPatchIndex;

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
    fp.indexSamplingStride = primaryPatchIndex ? primaryPatchIndex->samplingStride() : 1;
    fp.patchCount = (patchIndex ? patchIndex->patchCount() : 0) +
                    (useEditIndexForActive ? editPatchIndex->patchCount() : 0);
    fp.surfaceCount = (patchIndex ? patchIndex->surfaceCount() : 0) +
                      (useEditIndexForActive ? editPatchIndex->surfaceCount() : 0);
    fp.targetHash = _resolvedIntersectTargets.targetHash;
    // One O(1) probe per index instead of a mutex-guarded generation()
    // lookup per target (~100k per tick on large sessions). Slightly
    // over-invalidates — an update to a non-target surface also bumps the
    // global generation and forces a recompute — but non-target updates are
    // rare compared to render ticks.
    fp.targetGenerationHash =
        std::hash<uint64_t>{}(patchIndex ? patchIndex->globalGeneration() : 0) ^
        (std::hash<uint64_t>{}(useEditIndexForActive ? editPatchIndex->globalGeneration() : 0) +
         0x9e3779b9u);
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

        if (useEditIndexForActive) {
            // The edit index lives in SegmentationEditManager and is mutated on the
            // main thread during correction-drag; it is not covered by the index-read
            // gate, so reading it off-thread would race. Compute synchronously here
            // (global targets on the global index, the active surface on the edit
            // index) and fall through to the scene-item build below.
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
            auto globalTargets = targets;
            globalTargets.erase(activeSeg);
            if (patchIndex && !globalTargets.empty()) {
                _intersectionGeometryCache.intersections =
                    patchIndex->computePlaneIntersections(*plane, cacheRoi, globalTargets);
            }
            const std::unordered_set<SurfacePatchIndex::SurfacePtr> editTargets{activeSeg};
            auto editIntersections = editPatchIndex->computePlaneIntersections(*plane, cacheRoi, editTargets);
            for (auto& [surface, segments] : editIntersections) {
                auto& out = _intersectionGeometryCache.intersections[surface];
                out.insert(out.end(), segments.begin(), segments.end());
            }
            _intersectionGeometryCache.valid = true;
        } else {
            // Only one compute in flight at a time; while it runs the current scene items
            // stand as the preview. A later render with changed inputs waits for the
            // in-flight result (dropped if stale via the gen token) and re-kicks.
            if (_intersectComputeInFlight) {
                if (profile.enabled()) profile.setDetails("action=compute_already_in_flight");
                return;
            }
            // Heavy r-tree query + triangle clip -> worker. The QGraphicsItem build (Qt,
            // main-thread-only) happens in finishPlaneIntersectionCompute when it lands.
            const std::uint64_t gen = ++_intersectGen;
            auto planeCopy = std::make_shared<PlaneSurface>(*plane);
            auto targetsCopy = std::make_shared<std::unordered_set<SurfacePatchIndex::SurfacePtr>>(targets);
            _intersectComputeInFlight = true;
            _viewerManager->beginIndexRead();
            QPointer<CChunkedVolumeViewer> guard(this);
            ViewerManager* vm = _viewerManager;            // outlives viewers; always end the read
            SurfacePatchIndex* idx = patchIndex;
            (void)QtConcurrent::run([guard, vm, gen, cacheRoi, fp, planeCopy, targetsCopy, idx]() mutable {
                auto out = std::make_shared<std::unordered_map<SurfacePatchIndex::SurfacePtr,
                    std::vector<SurfacePatchIndex::TriangleSegment>>>(
                        idx->computePlaneIntersections(*planeCopy, cacheRoi, *targetsCopy));
                QMetaObject::invokeMethod(qApp, [guard, vm, gen, cacheRoi, fp, out = std::move(out)]() mutable {
                    if (guard)
                        guard->finishPlaneIntersectionCompute(gen, cacheRoi, fp, std::move(out));
                    else if (vm)
                        vm->endIndexRead();                // viewer gone: still release the gate
                }, Qt::QueuedConnection);
            });
            if (profile.enabled())
                profile.setDetails(std::format("action=async_compute_kick gen={} roi={}",
                                               gen, profileRect(cacheRoi)));
            return;   // scene items rebuilt by finishPlaneIntersectionCompute -> re-entry
        }
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
        } else if (auto folderColor = segmentFolderColorOverride(target.get())) {
            baseColor = *folderColor;
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

// Worker finished computePlaneIntersections (main thread, via invokeMethod). Release
// the index read-gate; then -- unless superseded -- publish the geometry into the
// cache and re-run renderIntersections, which now hits the geometry cache and builds
// the QGraphicsItems synchronously (cheap). A stale result (gen advanced because the
// camera/surface/targets changed) is dropped; the next render re-kicks.
void CChunkedVolumeViewer::finishPlaneIntersectionCompute(
    std::uint64_t gen, cv::Rect cacheRoi, IntersectFingerprint fp,
    std::shared_ptr<std::unordered_map<SurfacePatchIndex::SurfacePtr,
        std::vector<SurfacePatchIndex::TriangleSegment>>> result)
{
    _intersectComputeInFlight = false;
    if (_viewerManager)
        _viewerManager->endIndexRead();              // release the gate (may apply a deferred swap)
    if (_closing || gen != _intersectGen || !result)
        return;                                      // superseded or shutting down

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
    _intersectionGeometryCache.intersections = std::move(*result);
    _intersectionGeometryCache.valid = true;
    // Clear the fp memo so the cache-hit early-out doesn't skip the scene-item build.
    _lastIntersectFp = {};
    renderIntersections("async intersection ready");
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
    _resolvedIntersectTargets = {};
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
    QStringList sharedCacheItems;
    items << QString("L%1").arg(_dsScaleIdx);
    if (const QString viewerLabel = property("vc_viewer_label").toString(); !viewerLabel.isEmpty())
        items << viewerLabel;
    items << QString("scale %1").arg(_scale, 0, 'f', 2);
    items << QString("%1x%2").arg(_framebuffer.width()).arg(_framebuffer.height());

    if ((_compositeSettings.enabled || _compositeSettings.planeEnabled) && streamingCompositeUnsupported()) {
        items << QString("composite unsupported: %1").arg(QString::fromStdString(_compositeSettings.params.method));
    } else if (_compositeSettings.enabled || _compositeSettings.planeEnabled) {
        items << QString("composite %1").arg(QString::fromStdString(_compositeSettings.params.method));
    }

    if (_chunkArray) {
        const auto stats = _chunkArray->stats();
        sharedCacheItems << QString("RAM %1/%2 GB")
            .arg(formatGigabytes(stats.decodedBytes))
            .arg(formatGigabytes(stats.decodedByteCapacity));
        if (stats.persistentCacheEnabled) {
            QString disk = QString("disk %1").arg(formatByteSize(stats.persistentCacheBytes));
            if (stats.persistentCacheMaximumBytes)
                disk += QString("/%1").arg(formatByteSize(*stats.persistentCacheMaximumBytes));
            if (stats.persistentCacheScanInFlight)
                disk += QStringLiteral(" (scanning)");
            else if (stats.persistentCacheTrimInFlight)
                disk += QStringLiteral(" (trimming)");
            sharedCacheItems << disk;
            if (stats.persistentCacheLowSpace) {
                sharedCacheItems << QString("⚠ low disk: %1 free")
                    .arg(formatByteSize(stats.persistentCacheFreeBytes));
            }
        } else {
            sharedCacheItems << QStringLiteral("disk off");
        }
        if (stats.remoteFetchesInFlight > 0) {
            sharedCacheItems << QString("network %1 @ %2")
                .arg(stats.remoteFetchesInFlight)
                .arg(formatMegabytesPerSecond(stats.remoteDownloadBytesPerSecond));
        } else {
            sharedCacheItems << QStringLiteral("network idle");
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
    } else if (property("vc_show_custom_normal_offset").toBool()) {
        items << QString("normal offset %1")
                     .arg(property("vc_custom_normal_offset_vx").toDouble(), 0, 'f', 1);
    }

    _statsBar->setItems(items);
    emit sharedCacheStatsChanged(sharedCacheItems);
}
