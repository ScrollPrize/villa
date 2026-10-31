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
#include <QComboBox>
#include <QDockWidget>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QFutureWatcher>
#include <QHBoxLayout>
#include <QImage>
#include <QLabel>
#include <QMainWindow>
#include <QMessageBox>
#include <QPainter>
#include <QPainterPath>
#include <QPen>
#include <QPixmap>
#include <QPointer>
#include <QPolygonF>
#include <QPushButton>
#include <QtMath>
#include <QSettings>
#include <QStandardItemModel>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QTimer>
#include <QToolButton>
#include <QVBoxLayout>
#include <QWidget>
#include <QtConcurrent/QtConcurrentRun>

#include <array>
#include <atomic>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include <queue>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <cstdio>
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

// --- colour-by-metric --------------------------------------------------------
constexpr int kMetricBuckets = 24;      // quantise a 0..1 metric into colour buckets
constexpr double kFitRmsFullUm = 100.0; // fit-RMS colour full-scale: 0 um -> good, >= this -> bad

double clamp01(double x) { return x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x); }
// 0 -> red (bad), 0.5 -> yellow, 1 -> green (good). Hue 0..0.34 of the HSV wheel.
QColor metricColour(double good) { return QColor::fromHsvF(float(clamp01(good) * 0.34), 0.85, 0.95); }
int metricBucket(double good)
{
    const int b = int(clamp01(good) * kMetricBuckets);
    return b >= kMetricBuckets ? kMetricBuckets - 1 : b;
}
// Colour at the centre of bucket b (so the legend/stroke colour matches the binning).
QColor metricBucketColour(int b) { return metricColour((b + 0.5) / kMetricBuckets); }
// Neutral grey for sheets whose chosen metric is missing (NaN: NULL or older .asv).
QColor kMissingColour() { return QColor(150, 150, 150); }

// --- affinity colour ramp ----------------------------------------------------
// 0 -> blue, 1 -> red (distinct from the red->green metric ramp). Hue swept from 240deg
// (blue) up through magenta to 360deg (= red), so the ramp carries no green.
QColor affinityColour(double a)
{
    float h = 0.6667f + float(clamp01(a)) * 0.3333f;  // 0.667 (blue) .. 1.0 (== 0.0, red)
    if (h >= 1.0f)
        h -= 1.0f;
    return QColor::fromHsvF(h, 0.9f, 1.0f);
}
// Colour at the centre of metric bucket b on the affinity ramp (reuses the metric buckets).
QColor affinityBucketColour(int b) { return affinityColour((b + 0.5) / kMetricBuckets); }

// --- GASP agglomeration ------------------------------------------------------
// Average-linkage GASP (Generalized Algorithm for Signed graph Partitioning) over a signed
// weighted graph: repeatedly merge the pair of clusters with the highest AVERAGE inter-cluster
// edge weight while that average is positive; a negative (repulsive) accumulation blocks a
// merge, so no threshold is needed. `n` nodes labelled 0..n-1; `edges` are {u, v, w} triples
// (pre-aggregate duplicate pairs by mean before calling). Returns each node's cluster label
// (a representative node id). Lazy priority queue: a stale entry (whose recomputed average no
// longer matches, or has gone non-positive) is skipped.
std::vector<int> gaspAverageLinkage(int n, const std::vector<std::tuple<int, int, double>>& edges)
{
    std::vector<int> parent(n);
    std::iota(parent.begin(), parent.end(), 0);
    auto find = [&](int x) {
        while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    };
    // Current cluster adjacency: root -> (neighbour root -> {summed weight, edge count}).
    std::vector<std::unordered_map<int, std::pair<double, long long>>> adj(n);
    for (const auto& [u, v, w] : edges) {
        if (u == v || u < 0 || v < 0 || u >= n || v >= n)
            continue;
        auto& a = adj[u][v]; a.first += w; a.second += 1;
        auto& b = adj[v][u]; b.first += w; b.second += 1;
    }
    using QE = std::tuple<double, int, int>;  // (average weight, root u, root v), max-first
    std::priority_queue<QE> pq;
    for (int u = 0; u < n; ++u)
        for (const auto& [v, sc] : adj[u])
            if (u < v) {
                const double a = sc.first / double(sc.second);
                if (a > 0.0)
                    pq.emplace(a, u, v);
            }
    while (!pq.empty()) {
        const auto [w, u, v] = pq.top();
        pq.pop();
        const int ru = find(u), rv = find(v);
        if (ru == rv)
            continue;
        const auto it = adj[ru].find(rv);
        if (it == adj[ru].end())
            continue;
        const double cur = it->second.first / double(it->second.second);
        if (cur <= 0.0 || std::abs(cur - w) > 1e-9)
            continue;  // stale: a fresher entry exists, or the merge turned repulsive
        // Merge rv into ru: fold rv's adjacency into ru's and redirect its neighbours.
        for (const auto& [x, sc] : adj[rv]) {
            if (x == ru)
                continue;
            auto& into = adj[ru][x]; into.first += sc.first; into.second += sc.second;
            auto& back = adj[x]; auto& be = back[ru]; be.first += sc.first; be.second += sc.second;
            back.erase(rv);
        }
        adj[ru].erase(rv);
        adj[rv].clear();
        parent[rv] = ru;
        for (const auto& [x, sc] : adj[ru]) {
            const double a = sc.first / double(sc.second);
            if (a > 0.0)
                pq.emplace(a, ru, x);
        }
    }
    std::vector<int> label(n);
    for (int i = 0; i < n; ++i)
        label[i] = find(i);
    return label;
}

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
// Resident (decoded) mesh ceiling per viewport query. Generous so it does not bind at
// normal zoom (binding truncates largest-sheet-first and starves small sheets on pan);
// it only caps a pathological zoom-out so several panels cannot OOM. Watch the
// "[sheet-stitch] ... sheet-cache X MB" log to tune this to real data.
constexpr size_t kFetchByteBudget = 1024u * 1024u * 1024u;
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

    // Colour mode: Solid, per-sheet Rainbow, or colour-by one of the quality metrics
    // (red = low/poor, green = high/good; unscored sheets drawn grey). One exclusive
    // choice. Stored by its stable enum index so reordering the labels is safe.
    auto* colourRow = new QWidget(body);
    auto* colourLay = new QHBoxLayout(colourRow);
    colourLay->setContentsMargins(0, 0, 0, 0);
    colourLay->addWidget(new QLabel(tr("Colour by"), colourRow));
    colourBy_ = new QComboBox(colourRow);
    colourBy_->addItem(tr("Solid"));                     // ColourMode::Solid
    colourBy_->addItem(tr("Rainbow (per sheet)"));       // ColourMode::Rainbow
    colourBy_->addItem(tr("Confidence"));                // ColourMode::Confidence
    colourBy_->addItem(tr("Coverage"));                  // ColourMode::Coverage
    colourBy_->addItem(tr("Mass fraction"));             // ColourMode::MassFrac
    colourBy_->addItem(tr("Fit RMS"));                   // ColourMode::FitRms
    colourBy_->addItem(tr("Affinity (stitch)"));         // ColourMode::Affinity — enabled only while stitching
    colourBy_->setToolTip(tr("How to colour the sheet curves. The metric modes shade each sheet "
                             "red (low/poor) → green (high/good); unscored sheets are grey."));
    // Restore the saved mode, but NEVER into Affinity: it needs an active stitch on an
    // affinities-bearing .asv, neither of which holds at startup. Falling back to the default
    // here is what keeps a previously-saved Affinity selection from colouring an empty map.
    int savedColour = settings.value("sheetCollections/colourMode", int(ColourMode::Rainbow)).toInt();
    if (savedColour < int(ColourMode::Solid) || savedColour >= int(ColourMode::Affinity))
        savedColour = int(ColourMode::Rainbow);
    colourBy_->setCurrentIndex(savedColour);
    colourLay->addWidget(colourBy_, 1);
    layout->addWidget(colourRow);

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

    // Per-sheet quality filters (the .asv quality columns). 0 = off for each; older .asv
    // files without the columns simply ignore them (the reader keeps such sheets).
    auto makeUnitSpin = [&](const char* key) {
        auto* sp = new QDoubleSpinBox(body);
        sp->setRange(0.0, 1.0);
        sp->setDecimals(2);
        sp->setSingleStep(0.05);
        sp->setSpecialValueText(tr("Off"));  // shown at 0
        sp->setValue(settings.value(QString("sheetCollections/") + key, 0.0).toDouble());
        return sp;
    };
    minConfidence_ = makeUnitSpin("minConfidence");
    minConfidence_->setToolTip(tr("Hide sheets whose mean own-membership (confidence) is below this. 0 = off."));
    minCoverage_ = makeUnitSpin("minCoverage");
    minCoverage_->setToolTip(tr("Hide sheets whose mask coverage is below this. 0 = off."));
    minMassFrac_ = makeUnitSpin("minMassFrac");
    minMassFrac_->setToolTip(tr("Hide sheets whose normalised mass fraction is below this. 0 = off."));
    maxFitRms_ = new QDoubleSpinBox(body);
    maxFitRms_->setRange(0.0, 100000.0);
    maxFitRms_->setDecimals(0);
    maxFitRms_->setSingleStep(10.0);
    maxFitRms_->setSuffix(tr(" µm"));
    maxFitRms_->setSpecialValueText(tr("Off"));  // shown at 0
    maxFitRms_->setValue(settings.value("sheetCollections/maxFitRmsUm", 0.0).toDouble());
    maxFitRms_->setToolTip(tr("Hide sheets whose mesh-vs-mask fit RMS is larger than this — a SMALLER number keeps "
                              "only the better-fitted sheets. On load this is scaled to the file's range and starts "
                              "at the max (nothing hidden); pull it down to cull the worst fits."));

    // Section header for the quality filters, with a help button explaining the columns.
    auto* qualHeader = new QWidget(body);
    auto* qualHdrLay = new QHBoxLayout(qualHeader);
    qualHdrLay->setContentsMargins(0, 6, 0, 0);
    auto* qualLabel = new QLabel(tr("Quality filters"), qualHeader);
    QFont qf = qualLabel->font();
    qf.setBold(true);
    qualLabel->setFont(qf);
    auto* qualHelp = new QToolButton(qualHeader);
    qualHelp->setText(QStringLiteral("?"));
    qualHelp->setAutoRaise(true);
    qualHelp->setToolTip(tr("What do these quality values mean?"));
    connect(qualHelp, &QToolButton::clicked, this, [this]() { showQualityHelp(); });
    qualHdrLay->addWidget(qualLabel);
    qualHdrLay->addStretch(1);
    qualHdrLay->addWidget(qualHelp);

    form->addRow(tr("Max displayed sheets"), maxDisplayed_);
    form->addRow(tr("Min sheet size (mm²)"), minArea_);
    form->addRow(tr("Slice distance ± (L0 vx)"), distance_);
    form->addRow(tr("Line width (px)"), lineWidth_);
    form->addRow(qualHeader);
    form->addRow(tr("Min confidence"), minConfidence_);
    form->addRow(tr("Min coverage"), minCoverage_);
    form->addRow(tr("Min mass fraction"), minMassFrac_);
    form->addRow(tr("Max fit RMS"), maxFitRms_);
    layout->addLayout(form);

    // Auto-stitch: GASP-agglomerate the latest selected sheet's affinity graph over a window
    // neighbourhood and add its cluster to the stitch. Only usable in stitch mode on an
    // affinities-bearing .asv (enabled by refreshAffinityAvailability()).
    auto* autoRow = new QWidget(body);
    auto* autoLay = new QHBoxLayout(autoRow);
    autoLay->setContentsMargins(0, 0, 0, 0);
    autoStitch_ = new QPushButton(tr("Auto-stitch"), autoRow);
    autoStitch_->setToolTip(tr("Grow the stitch from the latest selected sheet: agglomerate its stitcher "
                               "affinity graph (GASP, average linkage over the signed logits) across the "
                               "surrounding windows and add the sheet's resulting cluster. Needs an .asv with "
                               "an affinities table, in stitch mode."));
    autoRadius_ = new QSpinBox(autoRow);
    autoRadius_->setRange(1, 5);
    autoRadius_->setValue(1);
    autoRadius_->setPrefix(QStringLiteral("±"));
    autoRadius_->setSuffix(tr(" win"));
    autoRadius_->setToolTip(tr("Window neighbourhood radius: ±1 = the 3×3×3 block of windows around the "
                               "selected sheet's window, ±2 = 5×5×5, and so on."));
    autoLay->addWidget(autoStitch_, 1);
    autoLay->addWidget(autoRadius_);
    layout->addWidget(autoRow);

    layout->addStretch(1);

    dock_->setWidget(body);
    window->addDockWidget(Qt::RightDockWidgetArea, dock_);

    auto persist = [this]() {
        QSettings s(vc3d::settingsFilePath(), QSettings::IniFormat);
        s.setValue("sheetCollections/colourMode", colourBy_->currentIndex());
        s.setValue("sheetCollections/maxDisplayed", maxDisplayed_->value());
        s.setValue("sheetCollections/minAreaMm2", minArea_->value());
        s.setValue("sheetCollections/distance", distance_->value());
        s.setValue("sheetCollections/lineWidth", lineWidth_->value());
        s.setValue("sheetCollections/minConfidence", minConfidence_->value());
        s.setValue("sheetCollections/minCoverage", minCoverage_->value());
        s.setValue("sheetCollections/minMassFrac", minMassFrac_->value());
        s.setValue("sheetCollections/maxFitRmsUm", maxFitRms_->value());
    };
    // Changes that alter which sheets are fetched force a re-query; draw-only changes
    // just repaint from the cache.
    // Checkboxes are single clicks -> apply instantly.
    connect(visible_, &QCheckBox::toggled, this, [this]() { invalidateDraw(); });
    // Colour mode is a draw-only change (the cached meshes already carry the metrics).
    connect(colourBy_, qOverload<int>(&QComboBox::currentIndexChanged), this,
            [this, persist]() { persist(); invalidateDraw(); });
    connect(autoStitch_, &QPushButton::clicked, this, [this]() { autoStitch(); });
    // Spinboxes debounce (1 s) so dragging a value does not re-render on every step. max-displayed and line width
    // only change the draw; min-size and slice distance change which sheets are fetched.
    connect(lineWidth_, qOverload<double>(&QDoubleSpinBox::valueChanged), this, [this, persist]() { persist(); scheduleControlsApply(false); });
    connect(maxDisplayed_, qOverload<int>(&QSpinBox::valueChanged), this, [this, persist]() { persist(); scheduleControlsApply(false); });
    connect(minArea_, qOverload<double>(&QDoubleSpinBox::valueChanged), this, [this, persist]() { persist(); scheduleControlsApply(true); });
    connect(distance_, qOverload<double>(&QDoubleSpinBox::valueChanged), this, [this, persist]() { persist(); scheduleControlsApply(true); });
    // Quality thresholds change which sheets are fetched -> re-query (debounced).
    for (QDoubleSpinBox* sp : {minConfidence_, minCoverage_, minMassFrac_, maxFitRms_})
        connect(sp, qOverload<double>(&QDoubleSpinBox::valueChanged), this, [this, persist]() { persist(); scheduleControlsApply(true); });

    refreshAffinityAvailability();  // Affinity starts disabled (no stitch, no .asv yet)

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
    hasAffinities_ = false;
    affinityRef_ = 0;
    affinityVals_.clear();
    if (!path.isEmpty()) {
        try {
            reader_ = std::make_unique<SheetCollection>(path.toStdString());
            path_ = path;
            hasAffinities_ = reader_->hasAffinities();  // offer affinity colouring only if the table exists
            bool ok = false;
            const double g = QString::fromStdString(reader_->metadata("grid_step_um")).toDouble(&ok);
            if (ok && g > 0)
                gridStepUm_ = g;
            const int64_t n = reader_->sheetCount();
            status_->setText(tr("%1 sheets · %2").arg(n).arg(QFileInfo(path).fileName()));
            // Scale the fit-RMS filter to THIS file: floor to the data min, cap to the data max, and default to
            // the max so nothing is hidden on load (smaller = better-fitted; pull it down to cull the worst fits).
            // There is a hard fit-RMS floor (~1.5 voxels), so a fixed 0-based range would sit entirely below it.
            if (maxFitRms_) {
                QSignalBlocker block(maxFitRms_);
                if (auto rng = reader_->columnRange("fit_rms_um")) {
                    maxFitRms_->setSpecialValueText(QString());          // no "Off" at min; the max is the keep-all end
                    maxFitRms_->setRange(std::floor(rng->first), std::ceil(rng->second));
                    maxFitRms_->setValue(maxFitRms_->maximum());         // default = max = nothing hidden
                } else {                                                 // older .asv w/o the column: 0 = Off
                    maxFitRms_->setSpecialValueText(tr("Off"));
                    maxFitRms_->setRange(0.0, 100000.0);
                    maxFitRms_->setValue(0.0);
                }
            }
        } catch (const std::exception& e) {
            status_->setText(tr("Could not open %1: %2").arg(path, QString::fromUtf8(e.what())));
        }
    } else {
        status_->setText(tr("Open an Automated Sheet Volume (.asv)."));
    }
    refreshAffinityAvailability();  // enable/disable Affinity for THIS file's affinities table
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

SheetCollectionController::ColourMode SheetCollectionController::colourMode() const
{
    const int i = colourBy_ ? colourBy_->currentIndex() : int(ColourMode::Rainbow);
    if (i < int(ColourMode::Solid) || i > int(ColourMode::Affinity))
        return ColourMode::Rainbow;
    const ColourMode m = ColourMode(i);
    // Affinity only applies during a stitch on a .asv that has the affinities table; otherwise
    // fall back to the default so it is never drawn (and never reads an empty affinity map).
    if (m == ColourMode::Affinity && !(stitchMode_ && hasAffinities_))
        return ColourMode::Rainbow;
    return m;
}

int64_t SheetCollectionController::affinityReference() const
{
    // The "latest selected sheet": the most-recently added one, else the anchor.
    return stitchMode_ ? (lastAdded_ ? lastAdded_ : seedId_) : 0;
}

void SheetCollectionController::ensureAffinityValues()
{
    const int64_t ref = affinityReference();
    if (ref == affinityRef_)
        return;  // already computed for this reference (changes only when the selection changes)
    affinityRef_ = ref;
    affinityVals_.clear();
    if (!ref || !reader_ || !hasAffinities_)
        return;
    const auto edges = reader_->affinities(ref);  // (dst, logit), both directions stored
    if (edges.empty())
        return;
    // Softmax over the reference's affinity logits, then normalise so the strongest neighbour
    // reads 1 (red); every rendered sheet not in this set stays 0 (blue).
    double maxLogit = -std::numeric_limits<double>::infinity();
    for (const auto& e : edges)
        maxLogit = std::max(maxLogit, e.second);
    std::vector<double> w(edges.size());
    double sum = 0.0;
    for (size_t i = 0; i < edges.size(); ++i) {
        w[i] = std::exp(edges[i].second - maxLogit);
        sum += w[i];
    }
    double peak = 0.0;
    for (size_t i = 0; i < edges.size(); ++i) {
        w[i] = sum > 0.0 ? w[i] / sum : 0.0;
        peak = std::max(peak, w[i]);
    }
    const double inv = peak > 0.0 ? 1.0 / peak : 1.0;
    for (size_t i = 0; i < edges.size(); ++i)
        affinityVals_[edges[i].first] = clamp01(w[i] * inv);
}

void SheetCollectionController::refreshAffinityAvailability()
{
    if (!colourBy_)
        return;
    const bool avail = hasAffinities_ && stitchMode_;
    if (auto* model = qobject_cast<QStandardItemModel*>(colourBy_->model())) {
        if (QStandardItem* item = model->item(int(ColourMode::Affinity)))
            item->setEnabled(avail);
    }
    if (autoStitch_)
        autoStitch_->setEnabled(avail);  // auto-stitch needs the same (affinities × stitch)
    if (autoRadius_)
        autoRadius_->setEnabled(avail);
    // Never leave Affinity selected when it is not usable: pull it back to the default so the
    // next repaint colours by the default mode (a previously-saved Affinity choice can't crash).
    if (!avail && colourBy_->currentIndex() == int(ColourMode::Affinity))
        colourBy_->setCurrentIndex(int(ColourMode::Rainbow));
}

void SheetCollectionController::autoStitch()
{
    if (!stitchMode_ || !reader_ || !hasAffinities_)
        return;
    const int64_t ref = affinityReference();  // the latest selected sheet
    if (!ref)
        return;
    const int radius = autoRadius_ ? autoRadius_->value() : 1;

    auto parseWin = [](const std::string& s, long long& z, long long& y, long long& x) {
        return std::sscanf(s.c_str(), "z%lld_y%lld_x%lld", &z, &y, &x) == 3;
    };
    long long sz = 0, sy = 0, sx = 0;
    if (!parseWin(reader_->summary(ref).window, sz, sy, sx))
        return;

    // Parse every sheet's window and build sorted unique per-axis coordinates, so a window's
    // distance in window-STEPS is well defined even if the grid has gaps.
    const auto sw = reader_->sheetWindows();
    std::set<long long> zs, ys, xs;
    std::vector<std::tuple<int64_t, long long, long long, long long>> parsed;
    parsed.reserve(sw.size());
    for (const auto& [id, win] : sw) {
        long long z = 0, y = 0, x = 0;
        if (!parseWin(win, z, y, x))
            continue;
        zs.insert(z); ys.insert(y); xs.insert(x);
        parsed.emplace_back(id, z, y, x);
    }
    auto rank = [](const std::set<long long>& u, long long v) {
        return int(std::distance(u.begin(), u.lower_bound(v)));
    };
    const int rz = rank(zs, sz), ry = rank(ys, sy), rx = rank(xs, sx);

    // Candidate sheets: window within `radius` steps of the reference's window on every axis
    // (the (2*radius+1)^3 block). The reference is always included.
    std::unordered_set<int64_t> cand;
    for (const auto& [id, z, y, x] : parsed)
        if (std::abs(rank(zs, z) - rz) <= radius && std::abs(rank(ys, y) - ry) <= radius &&
            std::abs(rank(xs, x) - rx) <= radius)
            cand.insert(id);
    cand.insert(ref);
    if (cand.size() < 2)
        return;

    // Dense node ids for the candidates.
    std::vector<int64_t> nodes(cand.begin(), cand.end());
    std::sort(nodes.begin(), nodes.end());
    std::unordered_map<int64_t, int> idx;
    idx.reserve(nodes.size() * 2);
    for (int i = 0; i < int(nodes.size()); ++i)
        idx[nodes[i]] = i;

    // Edges among candidates, with duplicate pairs (both directions + several corners)
    // aggregated to the mean logit (docs §9.1).
    const long long M = (long long)nodes.size();
    std::unordered_map<long long, std::pair<double, long long>> acc;  // key = a*M + b with a<b
    for (int64_t c : nodes) {
        const int iu = idx[c];
        for (const auto& [dst, logit] : reader_->affinities(c)) {
            const auto jt = idx.find(dst);
            if (jt == idx.end() || jt->second == iu)
                continue;
            const int a = std::min(iu, jt->second), b = std::max(iu, jt->second);
            auto& e = acc[(long long)a * M + b];
            e.first += logit; e.second += 1;
        }
    }
    std::vector<std::tuple<int, int, double>> edges;
    edges.reserve(acc.size());
    for (const auto& [k, e] : acc)
        edges.emplace_back(int(k / M), int(k % M), e.first / double(e.second));

    const std::vector<int> label = gaspAverageLinkage(int(nodes.size()), edges);
    const int seedLabel = label[idx[ref]];
    int added = 0;
    for (int i = 0; i < int(nodes.size()); ++i)
        if (label[i] == seedLabel && stitched_.insert(nodes[i]).second)
            ++added;

    refreshSelectionMeshes();
    updateStitchBanner();
    rebuildStitchPreview();
    invalidateDraw();
    if (status_)
        status_->setText(tr("Auto-stitch (±%1 win): added %2 sheet(s) → %3 in stitch.")
                             .arg(radius).arg(added).arg(int(stitched_.size())));
}

vc::sheets::QualityFilter SheetCollectionController::qualityFilter() const
{
    vc::sheets::QualityFilter q;
    if (minConfidence_) q.minConfidence = minConfidence_->value();
    if (minCoverage_)   q.minCoverage = minCoverage_->value();
    if (minMassFrac_)   q.minMassFrac = minMassFrac_->value();
    // At the cap (the file's max, the load default) the fit-RMS filter is OFF — keep every sheet, including the
    // one exactly at the max, with no rounding-boundary exclusion. Below the cap it culls the worse fits.
    if (maxFitRms_ && maxFitRms_->value() < maxFitRms_->maximum())
        q.maxFitRmsUm = maxFitRms_->value();
    return q;
}

void SheetCollectionController::showQualityHelp()
{
    // Definitions mirror the .asv quality columns (hercunet/meshes/asv.py). Nullable:
    // an older .asv without a column ignores that filter.
    const QString html = tr(
        "<p>Each sheet in an <b>.asv</b> carries optional quality statistics, written per window "
        "when the mesh is fitted. Use them to hide spurious or poorly-fitted sheets. A filter at "
        "<i>0</i> (<i>Off</i>) imposes no constraint, and sheets from an older .asv that predates a "
        "column are always kept.</p>"
        "<table cellspacing='0' cellpadding='4'>"
        "<tr><td valign='top'><b>confidence</b></td>"
        "<td><i>self_mass / n_voxels</i> — the mean own-membership of the sheet's voxels (0–1). "
        "How strongly its voxels actually claim it; the cleanest \"is this a real sheet?\" score.</td></tr>"
        "<tr><td valign='top'><b>coverage</b></td>"
        "<td>Fraction of the sheet's mask voxels within one grid step of the fitted mesh (0–1). "
        "Low = the mesh misses part of its support (holes, a bad grid).</td></tr>"
        "<tr><td valign='top'><b>mass fraction</b></td>"
        "<td><i>self_mass / max(self_mass in the window)</i> (0–1) — the sheet's confidence-weighted "
        "size relative to the biggest sheet in its window; the window's size-prune term.</td></tr>"
        "<tr><td valign='top'><b>fit RMS</b></td>"
        "<td>RMS distance from the gated mask voxels to the fitted mesh, in µm. "
        "Low = a tight fit; high = noisy / curved / under-resolved geometry.</td></tr>"
        "</table>"
        "<p style='margin-top:8px'>Also stored but not filtered here: <b>area_mm2</b> (fitted surface area — "
        "use <i>Min sheet size</i>), <b>n_voxels</b> (raw voxel support) and <b>self_mass</b> "
        "(Σ own-membership, the soft voxel count).</p>"
        "<p><b>Colour by</b> shades each sheet by one of these metrics: "
        "<span style='color:#c00'>red = low / poor</span> → "
        "<span style='color:#090'>green = high / good</span> (fit RMS inverts: low RMS is good). "
        "Sheets with no value for the chosen metric are drawn grey.</p>");
    QMessageBox box(dock_);
    box.setWindowTitle(tr("Sheet quality metrics"));
    box.setTextFormat(Qt::RichText);
    box.setText(html);
    box.setIcon(QMessageBox::Information);
    box.exec();
}

bool SheetCollectionController::isOverlayEnabledFor(VolumeViewerBase* viewer) const
{
    if (!reader_ || !visible_ || !visible_->isChecked() || !viewer)
        return false;
    if (dynamic_cast<PlaneSurface*>(viewer->currentSurface()))
        return true;  // cross-section slice panes
    // During stitching, also outline the selected sheets on the flattened segment pane.
    if (stitchMode_ && viewer->surfName() == "segmentation")
        return true;
    return false;
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
    // The flattened segment pane has no slice; redraw it cheaply each cycle so the
    // sheet outlines follow its pan/zoom and surface changes.
    if (stitchMode_ && viewer && viewer->surfName() == "segmentation")
        return true;
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
    const vc::sheets::QualityFilter qf = qualityFilter();
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
                {   // diagnostics: fetched count + decoded MB; flag if the budget truncated (starvation risk)
                    size_t b = 0;
                    for (const auto& sm : vv.meshes)
                        b += sm.mesh.verts.size() * sizeof(vc::sheets::Point) +
                             sm.mesh.tris.size() * sizeof(std::array<int64_t, 3>) + sm.mesh.foot.size();
                    std::fprintf(stderr, "[sheet-stitch] query -> %zu sheets, %.1f MB decoded%s\n",
                                 vv.meshes.size(), b / 1048576.0,
                                 b >= kFetchByteBudget ? "  (BUDGET HIT - small sheets may be dropped)" : "");
                    std::fflush(stderr);
                }
                ++drawRevision_;
                refreshViewer(viewer);
            });
    // Worker thread: its own read-only SQLite connection, so it never races the
    // UI-thread reader.
    watcher->setFuture(QtConcurrent::run([path, q, maxSheets, minV, qf, cancel]() {
        std::vector<SheetMesh> out;
        try {
            SheetCollection rdr(path);
            out = rdr.region(q, maxSheets, minV, vc::sheets::FamilyFilter::All, cancel.get(),
                             kFetchByteBudget, qf);
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
    if (!reader_)
        return;
    if (!current) {
        // Not a cross-section: the flattened segment pane, where we outline selected sheets.
        if (stitchMode_ && viewer->surfName() == "segmentation")
            drawSegmentPaneOutlines(viewer, builder);
        return;
    }

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

    const ColourMode mode = colourMode();
    if (mode == ColourMode::Affinity)
        ensureAffinityValues();  // (re)compute the reference sheet's softmaxed affinities if it changed
    const qreal width = lineWidth_->value();
    const int64_t minV = minVertsThreshold();
    const size_t cap = size_t(maxDisplayed_->value());
    // Per-sheet "goodness" (0 poor -> 1 good) for the active mode, or NaN when the sheet
    // lacks that value (NULL column / older .asv) -> drawn grey. Affinity maps the softmaxed
    // affinity to the reference sheet (missing from the map = 0 = not in range -> blue end).
    auto goodness = [this, mode](const SheetMesh& sm) -> double {
        if (mode == ColourMode::Affinity) {
            const auto it = affinityVals_.find(sm.id);
            return it != affinityVals_.end() ? it->second : 0.0;
        }
        switch (mode) {
            case ColourMode::Confidence: return sm.confidence;
            case ColourMode::Coverage:   return sm.coverage;
            case ColourMode::MassFrac:   return sm.massFrac;
            case ColourMode::FitRms:
                return std::isnan(sm.fitRmsUm) ? std::nan("")
                                               : clamp01(1.0 - sm.fitRmsUm / kFitRmsFullUm);  // low RMS = good
            default: return std::nan("");
        }
    };
    // Intersect the cached meshes with the plane, chaining each sheet's crossing into
    // connected polylines (per colour) so the rasteriser strokes whole curves, not
    // thousands of round-capped segments.
    std::array<std::vector<QPolygonF>, kRainbowBuckets> rainbow;
    std::array<std::vector<QPolygonF>, kMetricBuckets> metricB;  // colour-by-metric buckets
    std::vector<QPolygonF> missing;                              // sheets with no value for the metric
    std::vector<QPolygonF> plain;
    std::vector<QPolygonF> sheetPolys, stitchPolys, lastPolys, seedPolys;  // stitch highlight buckets
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
            auto& dst = (sm.id == seedId_) ? seedPolys
                        : (sm.id == lastAdded_ ? lastPolys : stitchPolys);
            dst.insert(dst.end(), sheetPolys.begin(), sheetPolys.end());
        } else if (mode == ColourMode::Rainbow) {
            auto& dst = rainbow[bucketOf(sm.id)];
            dst.insert(dst.end(), sheetPolys.begin(), sheetPolys.end());
        } else if (mode == ColourMode::Solid) {
            plain.insert(plain.end(), sheetPolys.begin(), sheetPolys.end());
        } else {
            const double g = goodness(sm);
            auto& dst = std::isnan(g) ? missing : metricB[metricBucket(g)];
            dst.insert(dst.end(), sheetPolys.begin(), sheetPolys.end());
        }
    }
    hits_[viewer] = std::move(localHits);

    // Consolidated base: draw its intersection with the slice as a prominent cyan line
    // so the growing segment path stays visible in the cross-sections after consolidation.
    std::vector<QPolygonF> basePolys;
    if (!consolidatedBase_.verts.empty() && !consolidatedBase_.tris.empty()) {
        std::vector<std::pair<QPointF, QPointF>> baseSegs;
        std::array<Point, 2> bseg;
        for (const auto& t : consolidatedBase_.tris) {
            if (vc::sheets::intersectTrianglePlane(
                    consolidatedBase_.verts[size_t(t[0])], consolidatedBase_.verts[size_t(t[1])],
                    consolidatedBase_.verts[size_t(t[2])], current->origin, current->n, bseg)) {
                baseSegs.emplace_back(viewer->volumeToScene(toVec(bseg[0], nativeToViewer_)),
                                      viewer->volumeToScene(toVec(bseg[1], nativeToViewer_)));
            }
        }
        chainInto(baseSegs, basePolys);
    }

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
        if (mode == ColourMode::Rainbow) {
            for (int b = 0; b < kRainbowBuckets; ++b)
                strokePolys(p, rainbow[b], hueOf(b));
        } else if (mode == ColourMode::Solid) {
            strokePolys(p, plain, QColor("#40c0ff"));
        } else if (mode == ColourMode::Affinity) {
            for (int b = 0; b < kMetricBuckets; ++b)        // blue (0) -> red (1), no "missing" (0 = blue)
                strokePolys(p, metricB[b], affinityBucketColour(b));
        } else {
            strokePolys(p, missing, kMissingColour());  // unscored first (underneath)
            for (int b = 0; b < kMetricBuckets; ++b)
                strokePolys(p, metricB[b], metricBucketColour(b));
        }
        // Stitch highlight drawn on top: members yellow, the seed magenta, both thicker.
        auto strokeHi = [&](const std::vector<QPolygonF>& polys, const QColor& c, qreal w) {
            if (polys.empty()) return;
            p.setPen(QPen(c, w, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
            for (const auto& poly : polys) p.drawPolyline(poly);
        };
        strokeHi(basePolys, QColor("#063b40"), width + 4.5);     // consolidated base: dark halo
        strokeHi(basePolys, QColor("#00e5ff"), width + 2.5);     // consolidated base: cyan
        strokeHi(stitchPolys, QColor("#ffffff"), width + 5.5);   // white outlines
        strokeHi(lastPolys, QColor("#ffffff"), width + 5.5);
        strokeHi(seedPolys, QColor("#ffffff"), width + 6.5);
        strokeHi(stitchPolys, QColor("#ffee00"), width + 3.0);   // members: yellow
        strokeHi(lastPolys, QColor("#ff30ff"), width + 3.0);     // last added: magenta
        strokeHi(seedPolys, QColor("#ff2020"), width + 3.5);     // anchor: red
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
        if (mode == ColourMode::Rainbow) {
            for (int b = 0; b < kRainbowBuckets; ++b) {
                QPainterPath pp = toPath(rainbow[b]);
                if (!pp.isEmpty()) { style.penColor = hueOf(b); builder.addPainterPath(pp, style); }
            }
        } else if (mode == ColourMode::Solid) {
            QPainterPath pp = toPath(plain);
            if (!pp.isEmpty()) { style.penColor = QColor("#40c0ff"); builder.addPainterPath(pp, style); }
        } else if (mode == ColourMode::Affinity) {
            for (int b = 0; b < kMetricBuckets; ++b) {       // blue (0) -> red (1)
                QPainterPath pp = toPath(metricB[b]);
                if (!pp.isEmpty()) { style.penColor = affinityBucketColour(b); builder.addPainterPath(pp, style); }
            }
        } else {
            QPainterPath pm = toPath(missing);
            if (!pm.isEmpty()) { style.penColor = kMissingColour(); builder.addPainterPath(pm, style); }
            for (int b = 0; b < kMetricBuckets; ++b) {
                QPainterPath pp = toPath(metricB[b]);
                if (!pp.isEmpty()) { style.penColor = metricBucketColour(b); builder.addPainterPath(pp, style); }
            }
        }
        auto addHi = [&](const std::vector<QPolygonF>& polys, const QColor& c, qreal w) {
            QPainterPath pp = toPath(polys);
            if (pp.isEmpty()) return;
            style.penColor = c; style.penWidth = w;
            builder.addPainterPath(pp, style);
        };
        addHi(basePolys, QColor("#00e5ff"), width + 2.5);     // consolidated base: cyan
        addHi(stitchPolys, QColor("#ffffff"), width + 5.5);
        addHi(lastPolys, QColor("#ffffff"), width + 5.5);
        addHi(seedPolys, QColor("#ffffff"), width + 6.5);
        addHi(stitchPolys, QColor("#ffee00"), width + 3.0);   // members: yellow
        addHi(lastPolys, QColor("#ff30ff"), width + 3.0);     // last added: magenta
        addHi(seedPolys, QColor("#ff2020"), width + 3.5);     // anchor: red
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
        if (hasAffinities_) {
            QAction* autoA = menu.addAction(tr("Auto-stitch from latest (±%1 win)")
                                                .arg(autoRadius_ ? autoRadius_->value() : 1));
            autoA->setToolTip(tr("Agglomerate the latest selected sheet's affinity graph (GASP) over the "
                                 "surrounding windows and add its cluster."));
            connect(autoA, &QAction::triggered, this, [this]() { autoStitch(); });
            menu.addSeparator();
        }
        QAction* interp = menu.addAction(tr("Interpolate gaps && consolidate"));
        interp->setToolTip(tr("Fill the gaps between the selected sheets and bake the result as a base "
                              "you can keep stitching onto."));
        connect(interp, &QAction::triggered, this, [this]() { consolidateInterpolate(); });
        QAction* save = menu.addAction(tr("Save stitched segment"));
        connect(save, &QAction::triggered, this, [this]() { saveStitch(); });
        menu.addSeparator();
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
        if (stitched_.size() <= 1 && consolidatedBase_.verts.empty())
            return;  // keep at least one sheet when there is no consolidated base
        stitched_.erase(id);
        if (id == seedId_)
            seedId_ = stitched_.empty() ? 0 : *stitched_.begin();  // keep a valid anchor if any
        if (id == lastAdded_)
            lastAdded_ = 0;                 // the magenta "just-added" sheet is gone
    } else {
        stitched_.insert(id);
        lastAdded_ = id;                     // the newly appended sheet is the magenta one
    }
    refreshSelectionMeshes();
    updateStitchBanner();
    rebuildStitchPreview();
    invalidateDraw();  // repaint the highlight
}

void SheetCollectionController::updateStitchBanner()
{
    if (!cwin_ || !stitchMode_)
        return;
    QString txt;
    if (!consolidatedBase_.verts.empty())
        txt = tr("Stitching onto consolidated base — %1 new sheet(s). ").arg(int(stitched_.size()));
    else
        txt = tr("Stitching into segment — anchor sheet #%1 (red), %2 sheet(s). ")
                  .arg(seedId_)
                  .arg(int(stitched_.size()));
    txt += tr("Left-click a sheet to add/remove · Ctrl+right-click to interpolate, save or cancel.");
    cwin_->sheetStitchSetBanner(txt);
}

void SheetCollectionController::enterStitch(int64_t seedId)
{
    if (!seedId)
        return;
    stitchMode_ = true;
    seedId_ = seedId;
    lastAdded_ = 0;  // nothing appended yet; only the red anchor
    stitched_ = {seedId};
    refreshSelectionMeshes();
    refreshAffinityAvailability();  // Affinity colouring becomes selectable now (if the .asv has it)
    if (cwin_) {
        cwin_->sheetStitchEnterLayout();
        updateStitchBanner();
    }
    rebuildStitchPreview();
    invalidateDraw();
}

size_t SheetCollectionController::cachedMeshBytes() const
{
    size_t bytes = 0;
    for (const auto& [viewer, v] : views_)
        if (v)
            for (const auto& sm : v->meshes)
                bytes += sm.mesh.verts.size() * sizeof(vc::sheets::Point) +
                         sm.mesh.tris.size() * sizeof(std::array<int64_t, 3>) + sm.mesh.foot.size();
    return bytes;
}

void SheetCollectionController::reclaimCaches()
{
    // Drop the decoded-sheet caches (up to kFetchByteBudget per viewer) so memory
    // recedes after a stitch session; viewers re-query lazily when next drawn.
    for (auto& [viewer, v] : views_) {
        if (!v)
            continue;
        if (v->cancel)
            v->cancel->store(true);  // abandon any in-flight query
        v->meshes.clear();
        v->meshes.shrink_to_fit();
        v->hasCoverage = false;
        v->pending = false;
    }
    hits_.clear();
    ++queryRevision_;  // invalidate so stale in-flight results are ignored
}

void SheetCollectionController::exitStitch()
{
    if (!stitchMode_)
        return;
    std::fprintf(stderr, "[sheet-stitch] exit: sheet-cache was %.1f MB (reclaiming)\n",
                 cachedMeshBytes() / (1024.0 * 1024.0));
    std::fflush(stderr);
    stitchMode_ = false;
    stitched_.clear();
    seedId_ = 0;
    lastAdded_ = 0;
    selMeshes_.clear();
    consolidatedBase_ = {};
    affinityRef_ = 0;
    affinityVals_.clear();
    refreshAffinityAvailability();  // disable Affinity + pull the selection back to the default
    if (cwin_) {
        cwin_->sheetStitchHideBanner();
        cwin_->sheetStitchExitLayout();  // restores layout AND the pre-stitch segmentation surface
    }
    reclaimCaches();
    invalidateQuery();  // force a fresh, bounded re-query for the still-visible panels
}

void SheetCollectionController::refreshSelectionMeshes()
{
    if (!reader_) {
        selMeshes_.clear();
        return;
    }
    for (auto it = selMeshes_.begin(); it != selMeshes_.end();)
        it = stitched_.count(it->first) ? std::next(it) : selMeshes_.erase(it);
    for (int64_t id : stitched_) {
        if (selMeshes_.count(id))
            continue;
        try {
            vc::sheets::Mesh m = reader_->mesh(id);
            if (!m.verts.empty())
                selMeshes_.emplace(id, std::move(m));
        } catch (...) {}
    }
}

vc::sheets::StitchResult SheetCollectionController::blendResult() const
{
    std::vector<vc::sheets::Mesh> meshes;
    if (!consolidatedBase_.verts.empty())
        meshes.push_back(consolidatedBase_);  // the baked, interpolated base, as a pseudo-sheet
    for (const auto& [id, m] : selMeshes_)
        meshes.push_back(m);
    if (meshes.empty())
        return {};  // ok() == false
    const vc::sheets::StitchResult r = vc::sheets::stitchSheets(meshes);
    std::fprintf(stderr,
                 "[sheet-stitch] blend %zu parts -> grid %dx%d (cell %.2f vx, %d filled, %d fold) | "
                 "sheet-cache %.1f MB\n",
                 meshes.size(), r.cols, r.rows, r.cell, r.filledCells, r.foldCells,
                 cachedMeshBytes() / (1024.0 * 1024.0));
    std::fflush(stderr);
    return r;
}

std::shared_ptr<QuadSurface> SheetCollectionController::toQuadSurface(const vc::sheets::StitchResult& r) const
{
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

std::shared_ptr<QuadSurface> SheetCollectionController::blendSelected() const
{
    return toQuadSurface(blendResult());
}

void SheetCollectionController::rebuildStitchPreview()
{
    if (!stitchMode_ || !cwin_)
        return;
    if (auto surf = blendSelected())
        cwin_->sheetStitchShowPreview(surf);
}

namespace
{
// A stitched grid as a pseudo-sheet so it can be fed back into stitchSheets as the
// consolidated base: nu = rows, nv = cols, foot = valid mask, verts row-major over
// present cells (the order stitchSheets expects). Tris are unused by the blender.
vc::sheets::Mesh meshFromGrid(const vc::sheets::StitchResult& r)
{
    vc::sheets::Mesh m;
    m.nu = r.rows;
    m.nv = r.cols;
    const int C = r.cols;
    m.foot.assign(size_t(r.rows) * C, 0);
    std::vector<int64_t> idxOf(size_t(r.rows) * C, -1);
    int64_t n = 0;
    for (int row = 0; row < r.rows; ++row)
        for (int col = 0; col < C; ++col)
            if (r.validAt(row, col)) {
                m.foot[size_t(row) * C + col] = 1;
                idxOf[size_t(row) * C + col] = n++;
                m.verts.push_back(r.at(row, col));
            }
    // Triangulate present quads so the base can be intersected with a slice plane
    // (winding is irrelevant for the intersection).
    auto vid = [&](int i, int j) { return idxOf[size_t(i) * C + j]; };
    for (int i = 0; i + 1 < r.rows; ++i)
        for (int j = 0; j + 1 < C; ++j) {
            const int64_t a = vid(i, j), b = vid(i, j + 1), c = vid(i + 1, j), d = vid(i + 1, j + 1);
            if (a >= 0 && b >= 0 && c >= 0) m.tris.push_back({a, c, b});
            if (b >= 0 && c >= 0 && d >= 0) m.tris.push_back({b, c, d});
        }
    return m;
}
}  // namespace

void SheetCollectionController::consolidateInterpolate()
{
    const vc::sheets::StitchResult r = blendResult();
    if (!r.ok()) {
        if (status_)
            status_->setText(tr("Nothing to consolidate."));
        return;
    }
    const vc::sheets::StitchResult filled = vc::sheets::interpolateGaps(r);
    consolidatedBase_ = meshFromGrid(filled);
    std::fprintf(stderr, "[sheet-stitch] consolidate: %d -> %d filled cells (base %zu verts)\n",
                 r.filledCells, filled.filledCells, consolidatedBase_.verts.size());
    std::fflush(stderr);
    // Reset the sheet selection: the interpolated surface is now the base, and the user
    // keeps stitching more (possibly sparse) sheets onto it.
    stitched_.clear();
    selMeshes_.clear();
    lastAdded_ = 0;
    seedId_ = 0;
    rebuildStitchPreview();  // preview is now the consolidated base alone
    updateStitchBanner();
    invalidateDraw();
}

namespace
{
// Ordered loop of a sheet's footprint-grid perimeter (top row, right col, bottom row,
// left col), using present cells. verts are row-major over present cells.
std::vector<Point> sheetPerimeter(const vc::sheets::Mesh& m)
{
    std::vector<int64_t> idxOf(size_t(m.nu * m.nv), -1);
    int64_t n = 0;
    for (size_t k = 0; k < idxOf.size(); ++k)
        if (m.foot[k])
            idxOf[k] = n++;
    auto vid = [&](int64_t i, int64_t j) {
        return (i >= 0 && j >= 0 && i < m.nu && j < m.nv) ? idxOf[size_t(i * m.nv + j)] : int64_t(-1);
    };
    std::vector<Point> loop;
    auto add = [&](int64_t i, int64_t j) {
        const int64_t a = vid(i, j);
        if (a >= 0)
            loop.push_back(m.verts[size_t(a)]);
    };
    for (int64_t j = 0; j < m.nv; ++j) add(0, j);
    for (int64_t i = 1; i < m.nu; ++i) add(i, m.nv - 1);
    for (int64_t j = m.nv - 2; j >= 0; --j) add(m.nu - 1, j);
    for (int64_t i = m.nu - 2; i >= 1; --i) add(i, 0);
    return loop;
}
}  // namespace

void SheetCollectionController::drawSegmentPaneOutlines(VolumeViewerBase* viewer, OverlayBuilder& builder)
{
    if (selMeshes_.empty())
        return;
    for (const auto& [id, m] : selMeshes_) {
        const auto loop = sheetPerimeter(m);
        if (loop.size() < 3)
            continue;
        QPolygonF poly;
        poly.reserve(int(loop.size()) + 1);
        for (const auto& p : loop)
            poly << viewer->volumeToScene(toVec(p, nativeToViewer_));
        poly << poly.front();  // close the loop
        QPainterPath pp;
        pp.addPolygon(poly);
        const bool isSeed = (id == seedId_);
        const QColor fill = isSeed ? QColor("#ff2020")                 // anchor: red
                            : (id == lastAdded_ ? QColor("#ff30ff")    // last added: magenta
                                                : QColor("#ffee00"));  // earlier members: yellow
        OverlayStyle style;
        style.brushColor = Qt::transparent;
        style.z = 46;
        style.penColor = QColor("#ffffff");         // white outline
        style.penWidth = isSeed ? 6.0 : 5.0;
        builder.addPainterPath(pp, style);
        style.penColor = fill;
        style.penWidth = isSeed ? 3.5 : 3.0;
        builder.addPainterPath(pp, style);
    }
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
