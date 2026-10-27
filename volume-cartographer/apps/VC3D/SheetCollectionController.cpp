#include "SheetCollectionController.hpp"

#include "ViewerManager.hpp"
#include "VCSettings.hpp"
#include "CWindow.hpp"
#include "volume_viewers/CVolumeViewerView.hpp"
#include "volume_viewers/CChunkedVolumeViewer.hpp"
#include "volume_viewers/VolumeViewerBase.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/types/SheetStitch.hpp"

#include <QAction>
#include <QMenu>

#include <QCheckBox>
#include <QColor>
#include <QDockWidget>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QFutureWatcher>
#include <QImage>
#include <QLabel>
#include <QMainWindow>
#include <QPainter>
#include <QPainterPath>
#include <QPen>
#include <QPixmap>
#include <QPointer>
#include <QPolygonF>
#include <QPushButton>
#include <QtMath>
#include <QSettings>
#include <QSpinBox>
#include <QTimer>
#include <QVBoxLayout>
#include <QWidget>
#include <QtConcurrent/QtConcurrentRun>

#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <utility>
#include <vector>

using vc::sheets::Bounds;
using vc::sheets::Point;
using vc::sheets::SheetCollection;
using vc::sheets::SheetMesh;
using vc::sheets::ViewRegion;

namespace
{
constexpr int kRainbowBuckets = 64;

Point sub(const Point& a, const Point& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
double dot(const Point& a, const Point& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
Point normalized(const Point& a)
{
    const double n = std::sqrt(dot(a, a));
    if (!(n > 0)) throw std::runtime_error("degenerate slice axis");
    return {a[0] / n, a[1] / n, a[2] / n};
}
Point toPoint(const cv::Vec3f& p, double scale) { return {p[0] * scale, p[1] * scale, p[2] * scale}; }
cv::Vec3f toVec(const Point& p, double scale)
{
    return {float(p[0] * scale), float(p[1] * scale), float(p[2] * scale)};
}

int bucketOf(int64_t id)
{
    return int(((uint64_t(id) * 0x9e3779b97f4a7c15ULL) >> 32) % kRainbowBuckets);
}
QColor hueOf(int bucket) { return QColor::fromHsvF(double(bucket) / kRainbowBuckets, 0.7, 1.0); }

// Chain a sheet's crossing segments into connected polylines. Adjacent triangles share
// the exact crossing point (same edge, same interpolation), so endpoints match by value;
// chaining lets the rasteriser stroke each curve once instead of many round-capped
// segments. Appends the resulting polylines to `out`.
void chainInto(const std::vector<std::pair<QPointF, QPointF>>& segs, std::vector<QPolygonF>& out)
{
    const size_t n = segs.size();
    if (n == 0)
        return;
    using Key = std::pair<qint64, qint64>;
    auto key = [](const QPointF& p) {
        qint64 x, y;
        const double a = p.x(), b = p.y();
        std::memcpy(&x, &a, 8);
        std::memcpy(&y, &b, 8);
        return Key{x, y};
    };
    std::map<Key, std::vector<size_t>> at;
    for (size_t i = 0; i < n; ++i) {
        at[key(segs[i].first)].push_back(i);
        at[key(segs[i].second)].push_back(i);
    }
    std::vector<bool> used(n, false);
    for (size_t i = 0; i < n; ++i) {
        if (used[i])
            continue;
        std::deque<QPointF> pts{segs[i].first, segs[i].second};
        used[i] = true;
        auto extend = [&](bool front) {
            for (;;) {
                const QPointF end = front ? pts.front() : pts.back();
                size_t next = n;
                bool nextIsFirst = false;
                for (size_t j : at[key(end)]) {
                    if (used[j])
                        continue;
                    nextIsFirst = (key(segs[j].first) == key(end));
                    next = j;
                    break;
                }
                if (next == n)
                    break;
                const QPointF add = nextIsFirst ? segs[next].second : segs[next].first;
                if (front)
                    pts.push_front(add);
                else
                    pts.push_back(add);
                used[next] = true;
            }
        };
        extend(false);  // forward from the back
        extend(true);   // backward from the front
        QPolygonF poly;
        poly.reserve(int(pts.size()));
        for (const auto& p : pts)
            poly << p;
        out.push_back(std::move(poly));
    }
}

// Prefetch geometry. Coverage is an oriented prism (same axes as the slice) whose
// in-plane extent has a pan margin and whose half-thickness NC along the slice normal
// pre-caches NC voxels of neighbouring slices, so scrolling within it never re-queries.
// Re-query fires predictively when the slice leaves the CORE (coverage shrunk by a
// buffer), so the async query completes before the view runs out of cached slices.
// Cache tuned for the interaction priority zoom > pan > scroll.
// - Scroll is cheapest to give up: a crossing sheet straddles the plane, so a thin
//   normal band still captures every visible sheet; thin also keeps the fetch count
//   down. Scroll re-queries sooner (kCoreNormal small).
// - Pan matters, so the in-plane margin is generous and pan re-queries only near the
//   (large) margin edge (kCoreInPlane).
// - Zoom matters most; it re-queries on a real zoom change (see collectPrimitives) and
//   reuses the cache between.
// The fetch is NOT capped by sheet count: a top-N-by-size cap spends the budget on the
// largest sheets anywhere in the region (including the off-screen pan margin) and never
// decodes the viewport's smaller sheets, so they can never be drawn. Fetch EVERY sheet
// in the region (bounded only by the decode byte budget) and apply maxDisplayed only
// when drawing, which is viewport-culled, so the per-frame cost still tracks the screen.
constexpr double kPrefetchNormal = 8.0;    // half-thickness along the normal (scroll axis), L0 vx
constexpr double kPrefetchInPlaneFrac = 0.4;
constexpr double kPrefetchInPlaneMin = 48.0;
constexpr double kCoreNormal = 4.0;        // re-query after ~this much scroll (vx)
constexpr double kCoreInPlane = 24.0;      // re-query only this close (vx) to the pan-margin edge
constexpr size_t kFetchByteBudget = 512u * 1024u * 1024u;  // decode cap per query
constexpr int kQueryDebounceMs = 1000;     // settle time before a motion fires a read
constexpr int kControlsDebounceMs = 1000;  // settle time before a dock-control edit applies

// Build the coverage prism and its world AABB (for the rtree query) around a slice.
ViewRegion makeCoverage(const ViewRegion& s)
{
    ViewRegion c;
    c.origin = s.origin;
    c.u = s.u;
    c.v = s.v;
    c.n = s.n;
    const double mu = std::max(kPrefetchInPlaneFrac * (s.local.upper[0] - s.local.lower[0]), kPrefetchInPlaneMin);
    const double mv = std::max(kPrefetchInPlaneFrac * (s.local.upper[1] - s.local.lower[1]), kPrefetchInPlaneMin);
    c.local.lower = {s.local.lower[0] - mu, s.local.lower[1] - mv, -kPrefetchNormal};
    c.local.upper = {s.local.upper[0] + mu, s.local.upper[1] + mv, kPrefetchNormal};
    c.box.lower.fill(std::numeric_limits<double>::infinity());
    c.box.upper.fill(-std::numeric_limits<double>::infinity());
    for (double a : {c.local.lower[0], c.local.upper[0]})
        for (double b : {c.local.lower[1], c.local.upper[1]})
            for (double w : {c.local.lower[2], c.local.upper[2]})
                for (int i = 0; i < 3; ++i) {
                    const double p = c.origin[i] + a * c.u[i] + b * c.v[i] + w * c.n[i];
                    c.box.lower[i] = std::min(c.box.lower[i], p);
                    c.box.upper[i] = std::max(c.box.upper[i], p);
                }
    return c;
}

// Do two world AABBs overlap?
bool overlaps(const Bounds& a, const Bounds& b)
{
    for (int i = 0; i < 3; ++i)
        if (a.upper[i] < b.lower[i] || a.lower[i] > b.upper[i])
            return false;
    return true;
}

// Is slice s inside the coverage prism, shrunk by (su, sv, sn) per axis? Requires the
// same axes (a rotation invalidates coverage). Shrink 0 = the full prism.
bool prismContains(const ViewRegion& c, const ViewRegion& s, double su = 0, double sv = 0, double sn = 0)
{
    // Tolerant axis match: the slice axes are computed from the viewport each frame, so
    // for an oblique plane (e.g. a segmentation's xz/yz cut) exact equality jitters and
    // the cache would never be reused. A real rotation moves them far more than this.
    auto axisClose = [](const Point& a, const Point& b) {
        return std::abs(a[0] - b[0]) < 1e-6 && std::abs(a[1] - b[1]) < 1e-6 && std::abs(a[2] - b[2]) < 1e-6;
    };
    if (!axisClose(c.u, s.u) || !axisClose(c.v, s.v) || !axisClose(c.n, s.n))
        return false;
    const Point off = c.project(s.origin);
    const double lo[3] = {c.local.lower[0] + su, c.local.lower[1] + sv, c.local.lower[2] + sn};
    const double hi[3] = {c.local.upper[0] - su, c.local.upper[1] - sv, c.local.upper[2] - sn};
    for (int i = 0; i < 3; ++i)
        if (!(s.local.lower[i] + off[i] >= lo[i] && s.local.upper[i] + off[i] <= hi[i]))
            return false;
    return true;
}
// True if the sheet's world AABB can contribute a visible curve: its corners must
// straddle the slice plane (n-range crosses 0) and overlap the visible (u,v) rect
// (with a small margin). The u,v test culls off-screen sheets so the per-frame cost
// tracks what is on screen, not the whole cached prism.
bool bboxVisible(const Bounds& b, const ViewRegion& s)
{
    double umin = 1e300, umax = -1e300, vmin = 1e300, vmax = -1e300, nmin = 1e300, nmax = -1e300;
    for (double x : {b.lower[0], b.upper[0]})
        for (double y : {b.lower[1], b.upper[1]})
            for (double z : {b.lower[2], b.upper[2]}) {
                const Point p = s.project({x, y, z});
                umin = std::min(umin, p[0]); umax = std::max(umax, p[0]);
                vmin = std::min(vmin, p[1]); vmax = std::max(vmax, p[1]);
                nmin = std::min(nmin, p[2]); nmax = std::max(nmax, p[2]);
            }
    if (nmin > 0.0 || nmax < 0.0)
        return false;  // does not cross the plane
    const double mu = 0.15 * (s.local.upper[0] - s.local.lower[0]);
    const double mv = 0.15 * (s.local.upper[1] - s.local.lower[1]);
    if (umax < s.local.lower[0] - mu || umin > s.local.upper[0] + mu)
        return false;
    if (vmax < s.local.lower[1] - mv || vmin > s.local.upper[1] + mv)
        return false;
    return true;
}
}  // namespace

// Per-viewport decoded-mesh cache plus the in-flight query that refreshes it.
struct SheetCollectionController::View {
    std::vector<SheetMesh> meshes;
    ViewRegion coverage;        // oriented prism the cached meshes cover
    bool hasCoverage{false};
    double builtExtentU{0}, builtExtentV{0};  // the slice's in-plane size when queried
    uint64_t cacheQueryRev{0};
    QPointer<QFutureWatcher<std::vector<SheetMesh>>> watcher;
    std::shared_ptr<std::atomic_bool> cancel;
    ViewRegion pendingCoverage;  // prism of the in-flight query
    bool pending{false};
    uint64_t pendingQueryRev{0};
    std::unique_ptr<QTimer> queryTimer;  // debounces re-queries during motion
};

SheetCollectionController::SheetCollectionController(ViewerManager* manager, QMainWindow* window)
    : ViewerOverlayControllerBase("sheet-collection", window), manager_(manager)
{
    cwin_ = dynamic_cast<CWindow*>(window);
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);

    dock_ = new QDockWidget(tr("Automated Sheet Volume"), window);
    dock_->setObjectName("sheetCollectionDock");
    auto* body = new QWidget(dock_);
    auto* layout = new QVBoxLayout(body);

    auto* openBtn = new QPushButton(tr("Open .asv…"), body);
    openBtn->setToolTip(tr("Attach a read-only Automated Sheet Volume (.asv) to the current volume."));
    layout->addWidget(openBtn);
    connect(openBtn, &QPushButton::clicked, this, [this]() {
        const auto p = QFileDialog::getOpenFileName(
            dock_, tr("Open Automated Sheet Volume (.asv)"), {},
            tr("Automated Sheet Volume (*.asv);;All files (*)"));
        if (!p.isEmpty()) openCollection(p);
    });

    status_ = new QLabel(tr("Open an Automated Sheet Volume (.asv)."), body);
    status_->setObjectName("sheetCollectionStatus");
    status_->setWordWrap(true);
    layout->addWidget(status_);

    visible_ = new QCheckBox(tr("Show volume · read only"), body);
    visible_->setChecked(true);
    layout->addWidget(visible_);

    rainbow_ = new QCheckBox(tr("Rainbow colors"), body);
    rainbow_->setChecked(settings.value("sheetCollections/rainbow", true).toBool());
    layout->addWidget(rainbow_);

    auto* form = new QFormLayout;
    form->setLabelAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    form->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);

    maxDisplayed_ = new QSpinBox(body);
    maxDisplayed_->setRange(0, 10000000);
    maxDisplayed_->setSpecialValueText(tr("All"));
    maxDisplayed_->setSingleStep(100);
    maxDisplayed_->setValue(settings.value("sheetCollections/maxDisplayed", 2000).toInt());
    maxDisplayed_->setToolTip(tr("Maximum sheets fetched for each viewport among those near its slice. "
                                 "0 fetches all. Larger sheets are kept first."));

    minArea_ = new QDoubleSpinBox(body);
    minArea_->setRange(0, 1000000);
    minArea_->setDecimals(2);
    minArea_->setSingleStep(0.5);
    minArea_->setValue(settings.value("sheetCollections/minAreaMm2", 0.0).toDouble());
    minArea_->setToolTip(tr("Hide sheets smaller than this surface area (mm²)."));

    distance_ = new QDoubleSpinBox(body);
    distance_->setRange(0.1, 1000);
    distance_->setValue(settings.value("sheetCollections/distance", 1.0).toDouble());
    distance_->setToolTip(tr("Half-thickness of the slice in native L0 voxels (fetch range)."));

    lineWidth_ = new QDoubleSpinBox(body);
    lineWidth_->setRange(0.5, 12);
    lineWidth_->setDecimals(1);
    lineWidth_->setSingleStep(0.5);
    lineWidth_->setValue(settings.value("sheetCollections/lineWidth", 2.0).toDouble());

    form->addRow(tr("Max displayed sheets"), maxDisplayed_);
    form->addRow(tr("Min sheet size (mm²)"), minArea_);
    form->addRow(tr("Slice distance ± (L0 vx)"), distance_);
    form->addRow(tr("Line width (px)"), lineWidth_);
    layout->addLayout(form);
    layout->addStretch(1);

    dock_->setWidget(body);
    window->addDockWidget(Qt::RightDockWidgetArea, dock_);

    auto persist = [this]() {
        QSettings s(vc3d::settingsFilePath(), QSettings::IniFormat);
        s.setValue("sheetCollections/rainbow", rainbow_->isChecked());
        s.setValue("sheetCollections/maxDisplayed", maxDisplayed_->value());
        s.setValue("sheetCollections/minAreaMm2", minArea_->value());
        s.setValue("sheetCollections/distance", distance_->value());
        s.setValue("sheetCollections/lineWidth", lineWidth_->value());
    };
    // Changes that alter which sheets are fetched force a re-query; draw-only changes
    // just repaint from the cache.
    // Checkboxes are single clicks -> apply instantly.
    connect(visible_, &QCheckBox::toggled, this, [this]() { invalidateDraw(); });
    connect(rainbow_, &QCheckBox::toggled, this, [this, persist]() { persist(); invalidateDraw(); });
    // Spinboxes debounce (1 s) so dragging a value does not re-render on every step. max-displayed and line width
    // only change the draw; min-size and slice distance change which sheets are fetched.
    connect(lineWidth_, qOverload<double>(&QDoubleSpinBox::valueChanged), this, [this, persist]() { persist(); scheduleControlsApply(false); });
    connect(maxDisplayed_, qOverload<int>(&QSpinBox::valueChanged), this, [this, persist]() { persist(); scheduleControlsApply(false); });
    connect(minArea_, qOverload<double>(&QDoubleSpinBox::valueChanged), this, [this, persist]() { persist(); scheduleControlsApply(true); });
    connect(distance_, qOverload<double>(&QDoubleSpinBox::valueChanged), this, [this, persist]() { persist(); scheduleControlsApply(true); });

    bindToViewerManager(manager_);
}

SheetCollectionController::~SheetCollectionController()
{
    for (auto& [viewer, v] : views_)
        if (v && v->cancel)
            v->cancel->store(true);
}

void SheetCollectionController::openCollection(const QString& path)
{
    // Drop caches and cancel in-flight queries before swapping readers.
    for (auto& [viewer, v] : views_)
        if (v && v->cancel)
            v->cancel->store(true);
    views_.clear();
    reader_.reset();
    path_.clear();
    if (!path.isEmpty()) {
        try {
            reader_ = std::make_unique<SheetCollection>(path.toStdString());
            path_ = path;
            bool ok = false;
            const double g = QString::fromStdString(reader_->metadata("grid_step_um")).toDouble(&ok);
            if (ok && g > 0)
                gridStepUm_ = g;
            const int64_t n = reader_->sheetCount();
            status_->setText(tr("%1 sheets · %2").arg(n).arg(QFileInfo(path).fileName()));
        } catch (const std::exception& e) {
            status_->setText(tr("Could not open %1: %2").arg(path, QString::fromUtf8(e.what())));
        }
    } else {
        status_->setText(tr("Open an Automated Sheet Volume (.asv)."));
    }
    invalidateQuery();
    emit collectionOpened(path_);  // persisted next to the project by the window
}

void SheetCollectionController::invalidateQuery()
{
    ++queryRevision_;
    ++drawRevision_;
    ViewerOverlayControllerBase::refreshAll();
}

void SheetCollectionController::invalidateDraw()
{
    ++drawRevision_;
    ViewerOverlayControllerBase::refreshAll();
}

void SheetCollectionController::scheduleControlsApply(bool needQuery)
{
    controlsNeedQuery_ = controlsNeedQuery_ || needQuery;
    if (!controlsTimer_) {
        controlsTimer_ = new QTimer(this);
        controlsTimer_->setSingleShot(true);
        controlsTimer_->setInterval(kControlsDebounceMs);
        connect(controlsTimer_, &QTimer::timeout, this, [this]() {
            if (controlsNeedQuery_)
                invalidateQuery();
            else
                invalidateDraw();
            controlsNeedQuery_ = false;
        });
    }
    controlsTimer_->start();  // (re)start: apply once the control settles
}

int64_t SheetCollectionController::minVertsThreshold() const
{
    const double mm = gridStepUm_ / 1000.0;  // grid step in mm
    const double cell = mm * mm;             // ~surface area per footprint vertex (mm^2)
    if (!minArea_ || cell <= 0.0)
        return 0;
    return int64_t(minArea_->value() / cell + 0.5);
}

bool SheetCollectionController::isOverlayEnabledFor(VolumeViewerBase* viewer) const
{
    return reader_ && visible_ && visible_->isChecked() && viewer &&
           dynamic_cast<PlaneSurface*>(viewer->currentSurface()) != nullptr;
}

std::optional<SheetCollectionController::Slice> SheetCollectionController::slice(VolumeViewerBase* viewer) const
{
    if (!viewer || !dynamic_cast<PlaneSurface*>(viewer->currentSurface()))
        return std::nullopt;
    const QRectF rect = visibleSceneRect(viewer);
    if (rect.isEmpty())
        return std::nullopt;
    Slice s;
    const double inv = 1.0 / nativeToViewer_;
    s.origin = toPoint(viewer->sceneToVolume(rect.topLeft()), inv);
    const Point x = toPoint(viewer->sceneToVolume(rect.topRight()), inv);
    const Point y = toPoint(viewer->sceneToVolume(rect.bottomLeft()), inv);
    try {
        s.u = normalized(sub(x, s.origin));
        s.v = normalized(sub(y, s.origin));
        s.n = normalized({s.u[1] * s.v[2] - s.u[2] * s.v[1],
                          s.u[2] * s.v[0] - s.u[0] * s.v[2],
                          s.u[0] * s.v[1] - s.u[1] * s.v[0]});
    } catch (...) {
        return std::nullopt;
    }
    const double d = distance_->value();
    s.local = {{0, 0, -d},
               {std::sqrt(dot(sub(x, s.origin), sub(x, s.origin))),
                std::sqrt(dot(sub(y, s.origin), sub(y, s.origin))), d}};
    s.box.lower.fill(std::numeric_limits<double>::infinity());
    s.box.upper.fill(-std::numeric_limits<double>::infinity());
    for (double a : {0.0, s.local.upper[0]})
        for (double b : {0.0, s.local.upper[1]})
            for (double c : {-d, d})
                for (int i = 0; i < 3; ++i) {
                    const double p = s.origin[i] + a * s.u[i] + b * s.v[i] + c * s.n[i];
                    s.box.lower[i] = std::min(s.box.lower[i], p);
                    s.box.upper[i] = std::max(s.box.upper[i], p);
                }
    return s;
}

bool SheetCollectionController::needsOverlayRebuild(VolumeViewerBase* viewer) const
{
    const auto it = built_.find(viewer);
    if (it == built_.end())
        return true;
    if (it->second.drawRevision != drawRevision_)
        return true;
    const double dpr =
        viewer && viewer->graphicsView() ? viewer->graphicsView()->viewport()->devicePixelRatioF() : 1.0;
    if (it->second.dpr != dpr)
        return true;
    return it->second.slice != slice(viewer);
}

void SheetCollectionController::ensureRegion(VolumeViewerBase* viewer, const Slice& s)
{
    auto& vptr = views_[viewer];
    if (!vptr)
        vptr = std::make_unique<View>();
    View& v = *vptr;
    // Let a running query for this revision finish while it still overlaps the view --
    // cancelling and restarting on every gesture frame would starve a moving view of
    // all results. Only abandon a query the view has moved entirely away from.
    if (v.pending && v.pendingQueryRev == queryRevision_ && overlaps(v.pendingCoverage.box, s.box))
        return;
    if (v.cancel)
        v.cancel->store(true);  // supersede the previous (now irrelevant) query

    const ViewRegion cov = makeCoverage(s);
    const Bounds q = cov.box;
    auto cancel = std::make_shared<std::atomic_bool>(false);
    v.cancel = cancel;
    v.pending = true;
    v.pendingCoverage = cov;
    v.pendingQueryRev = queryRevision_;

    const std::string path = path_.toStdString();
    const size_t maxSheets = 0;  // fetch every sheet in the region; maxDisplayed caps the draw
    const int64_t minV = minVertsThreshold();
    const uint64_t rev = queryRevision_;
    const double extU = s.local.upper[0] - s.local.lower[0];
    const double extV = s.local.upper[1] - s.local.lower[1];

    auto* watcher = new QFutureWatcher<std::vector<SheetMesh>>(this);
    v.watcher = watcher;
    connect(watcher, &QFutureWatcher<std::vector<SheetMesh>>::finished, this,
            [this, viewer, watcher, cov, rev, cancel, extU, extV]() {
                std::vector<SheetMesh> meshes;
                try { meshes = watcher->future().result(); } catch (...) {}
                watcher->deleteLater();
                if (cancel->load())
                    return;  // superseded by a newer query
                auto it = views_.find(viewer);
                if (it == views_.end() || !it->second)
                    return;
                View& vv = *it->second;
                vv.meshes = std::move(meshes);
                vv.coverage = cov;
                vv.builtExtentU = extU;
                vv.builtExtentV = extV;
                vv.hasCoverage = true;
                vv.cacheQueryRev = rev;
                vv.pending = false;
                ++drawRevision_;
                refreshViewer(viewer);
            });
    // Worker thread: its own read-only SQLite connection, so it never races the
    // UI-thread reader.
    watcher->setFuture(QtConcurrent::run([path, q, maxSheets, minV, cancel]() {
        std::vector<SheetMesh> out;
        try {
            SheetCollection rdr(path);
            out = rdr.region(q, maxSheets, minV, vc::sheets::FamilyFilter::All, cancel.get(),
                             kFetchByteBudget);
        } catch (...) {}
        return out;
    }));
}

void SheetCollectionController::scheduleQuery(VolumeViewerBase* viewer)
{
    auto& vptr = views_[viewer];
    if (!vptr)
        vptr = std::make_unique<View>();
    View& v = *vptr;
    if (!v.queryTimer) {
        v.queryTimer = std::make_unique<QTimer>();
        v.queryTimer->setSingleShot(true);
        v.queryTimer->setInterval(kQueryDebounceMs);
        connect(v.queryTimer.get(), &QTimer::timeout, this, [this, viewer]() {
            if (!views_.count(viewer))
                return;
            if (auto s = slice(viewer))
                ensureRegion(viewer, *s);
        });
    }
    v.queryTimer->start();  // (re)start: the read fires only once the motion settles
}

void SheetCollectionController::collectPrimitives(VolumeViewerBase* viewer, OverlayBuilder& builder)
{
    auto current = slice(viewer);
    const qreal dpr = viewer->graphicsView() ? viewer->graphicsView()->viewport()->devicePixelRatioF() : 1.0;
    built_[viewer] = Built{current, drawRevision_, dpr};
    installInteractions(viewer);  // connect mouse/menu once this viewer is actually drawn on
    if (!reader_ || !current)
        return;

    auto& vptr = views_[viewer];
    if (!vptr)
        vptr = std::make_unique<View>();
    View& v = *vptr;

    // Predictive refresh: re-query once the slice leaves the core (the coverage prism
    // shrunk by the buffer), so the async query lands before the cached slices run out.
    const bool revOk = v.hasCoverage && v.cacheQueryRev == queryRevision_;
    const bool inCore = revOk && prismContains(v.coverage, *current, kCoreInPlane, kCoreInPlane, kCoreNormal);
    // Re-query when the zoom changed enough that the cap would now sample a very
    // different sheet density (zoom in -> fetch finer sheets; zoom out -> wider area).
    const double curU = current->local.upper[0] - current->local.lower[0];
    const bool zoomMismatch = revOk && v.builtExtentU > 0 &&
                              (curU < 0.7 * v.builtExtentU || curU > 1.45 * v.builtExtentU);
    if (!inCore || zoomMismatch) {
        if (!v.hasCoverage)
            ensureRegion(viewer, *current);  // first load for this viewport: do not wait
        else
            scheduleQuery(viewer);           // debounce re-queries while the view moves
    }
    if (!v.hasCoverage)
        return;  // nothing cached yet; the async result will refresh this viewport

    const bool useRainbow = rainbow_->isChecked();
    const qreal width = lineWidth_->value();
    const int64_t minV = minVertsThreshold();
    const size_t cap = size_t(maxDisplayed_->value());
    // Intersect the cached meshes with the plane, chaining each sheet's crossing into
    // connected polylines (per colour) so the rasteriser strokes whole curves, not
    // thousands of round-capped segments.
    std::array<std::vector<QPolygonF>, kRainbowBuckets> rainbow;
    std::vector<QPolygonF> plain;
    std::vector<QPolygonF> sheetPolys, stitchPolys, seedPolys;  // stitch highlight buckets
    std::vector<SheetHit> localHits;                            // for click hit-testing
    std::array<Point, 2> seg;
    std::vector<std::pair<QPointF, QPointF>> sheetSegs;
    size_t drawn = 0;
    for (const auto& sm : v.meshes) {
        if (cap && drawn >= cap)
            break;
        if (int64_t(sm.mesh.verts.size()) < minV)
            continue;
        if (!bboxVisible(sm.bounds, *current))
            continue;
        sheetSegs.clear();
        for (const auto& t : sm.mesh.tris) {
            if (vc::sheets::intersectTrianglePlane(
                    sm.mesh.verts[size_t(t[0])], sm.mesh.verts[size_t(t[1])], sm.mesh.verts[size_t(t[2])],
                    current->origin, current->n, seg)) {
                sheetSegs.emplace_back(viewer->volumeToScene(toVec(seg[0], nativeToViewer_)),
                                       viewer->volumeToScene(toVec(seg[1], nativeToViewer_)));
            }
        }
        if (sheetSegs.empty())
            continue;
        ++drawn;
        sheetPolys.clear();
        chainInto(sheetSegs, sheetPolys);
        localHits.push_back(SheetHit{sm.id, sheetPolys});
        if (stitchMode_ && stitched_.count(sm.id)) {
            auto& dst = (sm.id == seedId_) ? seedPolys : stitchPolys;
            dst.insert(dst.end(), sheetPolys.begin(), sheetPolys.end());
        } else if (useRainbow) {
            auto& dst = rainbow[bucketOf(sm.id)];
            dst.insert(dst.end(), sheetPolys.begin(), sheetPolys.end());
        } else {
            plain.insert(plain.end(), sheetPolys.begin(), sheetPolys.end());
        }
    }
    hits_[viewer] = std::move(localHits);

    // Rasterise the curves into one image sized to the visible rect, then hand the base
    // a pixmap. Repaints (CT tiles, cursor, other overlays) then blit it instead of
    // re-stroking on the UI thread, so cost no longer scales with pen width or segment
    // count. A vector fallback covers a rect too large to raster.
    const QRectF rect = visibleSceneRect(viewer);
    const QSize pixels(qCeil(rect.width() * dpr), qCeil(rect.height() * dpr));
    auto strokePolys = [width](QPainter& p, const std::vector<QPolygonF>& polys, const QColor& c) {
        if (polys.empty()) return;
        p.setPen(QPen(c, width, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
        for (const auto& poly : polys)
            p.drawPolyline(poly);
    };
    if (!rect.isEmpty() && pixels.width() > 0 && pixels.height() > 0 &&
        qint64(pixels.width()) * pixels.height() * 4 <= 64 * 1024 * 1024) {
        QImage raster(pixels, QImage::Format_ARGB32_Premultiplied);
        raster.setDevicePixelRatio(dpr);
        raster.fill(Qt::transparent);
        QPainter p(&raster);
        p.setRenderHints(viewer->graphicsView()->renderHints());
        p.translate(-rect.topLeft());
        p.setBrush(Qt::NoBrush);
        if (useRainbow)
            for (int b = 0; b < kRainbowBuckets; ++b)
                strokePolys(p, rainbow[b], hueOf(b));
        else
            strokePolys(p, plain, QColor("#40c0ff"));
        // Stitch highlight drawn on top: members yellow, the seed magenta, both thicker.
        auto strokeHi = [&](const std::vector<QPolygonF>& polys, const QColor& c, qreal w) {
            if (polys.empty()) return;
            p.setPen(QPen(c, w, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
            for (const auto& poly : polys) p.drawPolyline(poly);
        };
        strokeHi(stitchPolys, QColor("#000000"), width + 5.0);   // halo
        strokeHi(seedPolys, QColor("#ffffff"), width + 8.0);     // halo (seed extra bold)
        strokeHi(stitchPolys, QColor("#ffee00"), width + 3.0);   // members: yellow
        strokeHi(seedPolys, QColor("#ff30ff"), width + 5.0);     // seed: magenta
        p.end();
        builder.addImage(raster, QPixmap::fromImage(raster), rect.topLeft(), 1, 1, 1, 45);
    } else {
        OverlayStyle style;
        style.penWidth = width;
        style.z = 45;
        style.brushColor = Qt::transparent;
        auto toPath = [](const std::vector<QPolygonF>& polys) {
            QPainterPath pp;
            for (const auto& poly : polys)
                pp.addPolygon(poly);
            return pp;
        };
        if (useRainbow) {
            for (int b = 0; b < kRainbowBuckets; ++b) {
                QPainterPath pp = toPath(rainbow[b]);
                if (!pp.isEmpty()) { style.penColor = hueOf(b); builder.addPainterPath(pp, style); }
            }
        } else {
            QPainterPath pp = toPath(plain);
            if (!pp.isEmpty()) { style.penColor = QColor("#40c0ff"); builder.addPainterPath(pp, style); }
        }
        auto addHi = [&](const std::vector<QPolygonF>& polys, const QColor& c, qreal w) {
            QPainterPath pp = toPath(polys);
            if (pp.isEmpty()) return;
            style.penColor = c; style.penWidth = w;
            builder.addPainterPath(pp, style);
        };
        addHi(stitchPolys, QColor("#000000"), width + 5.0);
        addHi(seedPolys, QColor("#ffffff"), width + 8.0);
        addHi(stitchPolys, QColor("#ffee00"), width + 3.0);
        addHi(seedPolys, QColor("#ff30ff"), width + 5.0);
    }
}

void SheetCollectionController::detachViewer(VolumeViewerBase* viewer)
{
    if (auto it = views_.find(viewer); it != views_.end()) {
        if (it->second && it->second->cancel)
            it->second->cancel->store(true);
        views_.erase(it);
    }
    built_.erase(viewer);
    hits_.erase(viewer);
    if (wired_.erase(viewer)) {
        if (auto* chunked = qobject_cast<CChunkedVolumeViewer*>(viewer->asQObject()))
            disconnect(chunked, nullptr, this, nullptr);
    }
    ViewerOverlayControllerBase::detachViewer(viewer);
}

// --- manual sheet stitching -------------------------------------------------

void SheetCollectionController::installInteractions(VolumeViewerBase* viewer)
{
    if (!viewer || wired_.count(viewer))
        return;
    auto* chunked = qobject_cast<CChunkedVolumeViewer*>(viewer->asQObject());
    if (!chunked)
        return;
    wired_.insert(viewer);
    // Left-click a sheet to toggle its membership while stitching. (Plain right-click
    // is pan in VC3D and the Ctrl+right menu is routed via CWindow's context seam.)
    connect(chunked, &CChunkedVolumeViewer::sendMousePressVolume, this,
            [this, viewer](cv::Vec3f, cv::Vec3f, Qt::MouseButton button, Qt::KeyboardModifiers mods, QPointF scenePos) {
                if (button == Qt::LeftButton && mods == Qt::NoModifier)
                    onSheetLeftPress(viewer, scenePos);
            });
}

namespace
{
// Distance from scene point p to segment (a, b).
double segDist(const QPointF& p, const QPointF& a, const QPointF& b)
{
    const QPointF ab = b - a, ap = p - a;
    const double len2 = ab.x() * ab.x() + ab.y() * ab.y();
    double t = len2 > 0 ? (ap.x() * ab.x() + ap.y() * ab.y()) / len2 : 0.0;
    t = std::clamp(t, 0.0, 1.0);
    const QPointF q = a + t * ab;
    const QPointF d = p - q;
    return std::sqrt(d.x() * d.x() + d.y() * d.y());
}
}  // namespace

int64_t SheetCollectionController::sheetAt(VolumeViewerBase* viewer, QPointF scenePos, double tolScene) const
{
    const auto it = hits_.find(viewer);
    if (it == hits_.end())
        return 0;
    int64_t best = 0;
    double bestD = tolScene;
    for (const auto& h : it->second)
        for (const auto& poly : h.curves)
            for (int i = 0; i + 1 < poly.size(); ++i) {
                const double d = segDist(scenePos, poly[i], poly[i + 1]);
                if (d < bestD) { bestD = d; best = h.id; }
            }
    return best;
}

double SheetCollectionController::clickToleranceScene(VolumeViewerBase* viewer) const
{
    const double m = (viewer && viewer->graphicsView()) ? viewer->graphicsView()->transform().m11() : 1.0;
    return 7.0 / std::max(m, 1e-6);  // ~7 px, independent of zoom
}

bool SheetCollectionController::maybeShowStitchMenu(VolumeViewerBase* viewer, QPointF scenePos, QPoint globalPos)
{
    if (!reader_)
        return false;
    const int64_t hit = sheetAt(viewer, scenePos, clickToleranceScene(viewer));
    if (!stitchMode_ && !hit)
        return false;  // nothing of ours here; let VC3D's own menu handle it

    QMenu menu;
    if (!stitchMode_) {
        QAction* a = menu.addAction(tr("Stitch into segment"));
        connect(a, &QAction::triggered, this, [this, hit]() { enterStitch(hit); });
    } else {
        QAction* save = menu.addAction(tr("Save stitched segment (%1 sheets)").arg(int(stitched_.size())));
        connect(save, &QAction::triggered, this, [this]() { saveStitch(); });
        if (hit && stitched_.count(hit) && hit != seedId_) {
            QAction* rem = menu.addAction(tr("Remove sheet from stitch"));
            connect(rem, &QAction::triggered, this, [this, hit]() { toggleSheet(hit); });
        } else if (hit && !stitched_.count(hit)) {
            QAction* add = menu.addAction(tr("Add sheet to stitch"));
            connect(add, &QAction::triggered, this, [this, hit]() { toggleSheet(hit); });
        }
        menu.addSeparator();
        QAction* cancel = menu.addAction(tr("Cancel stitching"));
        connect(cancel, &QAction::triggered, this, [this]() { exitStitch(); });
    }
    menu.exec(globalPos);
    return true;
}

void SheetCollectionController::onSheetLeftPress(VolumeViewerBase* viewer, QPointF scenePos)
{
    if (!stitchMode_)
        return;
    const int64_t hit = sheetAt(viewer, scenePos, clickToleranceScene(viewer));
    if (hit)
        toggleSheet(hit);
}

void SheetCollectionController::toggleSheet(int64_t id)
{
    if (!id)
        return;
    if (stitched_.count(id)) {
        if (stitched_.size() <= 1)
            return;  // keep at least one sheet in the stitch
        stitched_.erase(id);
        if (id == seedId_)
            seedId_ = *stitched_.begin();  // keep a valid seed for naming/anchor
    } else {
        stitched_.insert(id);
    }
    updateStitchBanner();
    rebuildStitchPreview();
    invalidateDraw();  // repaint the highlight
}

void SheetCollectionController::updateStitchBanner()
{
    if (!cwin_ || !stitchMode_)
        return;
    cwin_->sheetStitchSetBanner(
        tr("Stitching into segment — seed sheet #%1 (magenta), %2 sheet(s) selected.  "
           "Left-click a sheet to add/remove · Ctrl+right-click to save or cancel.")
            .arg(seedId_)
            .arg(int(stitched_.size())));
}

void SheetCollectionController::enterStitch(int64_t seedId)
{
    if (!seedId)
        return;
    stitchMode_ = true;
    seedId_ = seedId;
    stitched_ = {seedId};
    if (cwin_) {
        cwin_->sheetStitchEnterLayout();
        updateStitchBanner();
    }
    rebuildStitchPreview();
    invalidateDraw();
}

void SheetCollectionController::exitStitch()
{
    if (!stitchMode_)
        return;
    stitchMode_ = false;
    stitched_.clear();
    seedId_ = 0;
    if (cwin_) {
        cwin_->sheetStitchHideBanner();
        cwin_->sheetStitchExitLayout();
    }
    invalidateDraw();
}

std::shared_ptr<QuadSurface> SheetCollectionController::blendSelected() const
{
    if (!reader_ || stitched_.empty())
        return nullptr;
    std::vector<vc::sheets::Mesh> meshes;
    meshes.reserve(stitched_.size());
    for (int64_t id : stitched_) {
        try {
            vc::sheets::Mesh m = reader_->mesh(id);
            if (!m.verts.empty())
                meshes.push_back(std::move(m));
        } catch (...) {}
    }
    if (meshes.empty())
        return nullptr;
    const vc::sheets::StitchResult r = vc::sheets::stitchSheets(meshes);
    if (!r.ok())
        return nullptr;
    // Native-L0 XYZ -> QuadSurface grid (invalid cells = (-1,-1,-1)); scale = 1/cell.
    cv::Mat_<cv::Vec3f> pts(r.rows, r.cols, cv::Vec3f(-1.f, -1.f, -1.f));
    for (int rr = 0; rr < r.rows; ++rr)
        for (int cc = 0; cc < r.cols; ++cc)
            if (r.validAt(rr, cc)) {
                const vc::sheets::Point& p = r.at(rr, cc);
                pts(rr, cc) = cv::Vec3f(float(p[0] * nativeToViewer_),
                                        float(p[1] * nativeToViewer_),
                                        float(p[2] * nativeToViewer_));
            }
    const float s = float(1.0 / r.cell);
    return std::make_shared<QuadSurface>(pts, cv::Vec2f(s, s));
}

void SheetCollectionController::rebuildStitchPreview()
{
    if (!stitchMode_ || !cwin_)
        return;
    if (auto surf = blendSelected())
        cwin_->sheetStitchShowPreview(surf);
}

void SheetCollectionController::saveStitch()
{
    if (!cwin_)
        return;
    auto surf = blendSelected();
    if (!surf) {
        status_->setText(tr("Nothing to save — no usable sheets selected."));
        return;
    }
    QString err;
    const QString id = cwin_->sheetStitchSaveSegment(surf, QStringLiteral("stitch_%1").arg(seedId_), err);
    if (id.isEmpty())
        status_->setText(tr("Save failed: %1").arg(err));
    else {
        status_->setText(tr("Saved stitched segment %1 (%2 sheets).").arg(id).arg(int(stitched_.size())));
        exitStitch();
    }
}
