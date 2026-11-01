#include "Fiber3DSliceControls.hpp"
#include "Fiber3DSlices.hpp"
#include "scene3d/Scene3DCanvas.hpp"
#include "CState.hpp"
#include "ViewerManager.hpp"
#include "VCSettings.hpp"
#include "volume_viewers/VolumeViewerBase.hpp"
#include "volume_viewers/CVolumeViewerView.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include <QCheckBox>
#include <QFutureWatcher>
#include <QLabel>
#include <QSettings>
#include <QDoubleSpinBox>
#include <limits>
#include <QSignalBlocker>
#include <QSlider>
#include <QSpinBox>
#include <QTimer>
#include <QToolButton>
#include <QVBoxLayout>
#include <QtConcurrent/QtConcurrentRun>
#include <algorithm>
#include <cmath>

namespace vc3d::fibers3d
{
namespace
{
const std::array<std::string, 3> names{"xy plane", "seg xz", "seg yz"};
Point point(cv::Vec3f p)
{
    return {p[0], p[1], p[2]};
}
struct Result {
    slices::Images images;
    QString error;
};
}  // namespace
struct Fiber3DSliceControls::Impl {
    Fiber3DSliceControls* q;
    CState* state;
    QString settingsGroup;
    ViewerManager* viewers;
    scene3d::Scene3DCanvas* canvas;
    QToolButton* button;
    std::array<QCheckBox*, 3> axes{};
    QCheckBox* volumetric;
    QWidget* xyPositionRow = nullptr;
    QSlider* xyPosition = nullptr;
    QDoubleSpinBox* xyPositionValue = nullptr;
    QCheckBox* followMain = nullptr;
    QLabel* status;
    QTimer poll, debounce;
    QFutureWatcher<Result> watcher;
    std::shared_ptr<std::atomic_bool> cancelled;
    std::shared_ptr<Volume> volume;
    std::vector<slices::Plane> requested;
    slices::Images cache;
    std::optional<Point> center;
    std::optional<Point> meshCenter, lastMeshCenter;
    std::optional<scene3d::Bounds> meshBounds;
    std::array<bool, 3> savedAxes{};
    bool busy = false, pending = false;

    Impl(Fiber3DSliceControls* widget, CState* s, ViewerManager* vm, scene3d::Scene3DCanvas* c, QToolButton* b, QString group)
        : q(widget), state(s), settingsGroup(std::move(group)), viewers(vm), canvas(c), button(b)
    {
        q->setObjectName(QStringLiteral("fibers3DSliceControls"));
        auto* layout = new QVBoxLayout(q);
        layout->setContentsMargins(2, 2, 2, 2);
        auto* row = new QHBoxLayout;
        QSettings prefs(vc3d::settingsFilePath(), QSettings::IniFormat);
        for (int i = 0; i < 3; ++i) {
            axes[i] = new QCheckBox(QStringList{"XY", "XZ", "YZ"}[i], q);
            axes[i]->setObjectName(QStringLiteral("fibers3DSlice") + axes[i]->text());
            axes[i]->setChecked(prefs.value(settingsGroup + QStringLiteral("/") + axes[i]->objectName(), i != 1).toBool());
            axes[i]->setToolTip(q->tr("Follow the position, orientation and visible area of this CT view"));
            row->addWidget(axes[i]);
            QObject::connect(axes[i], &QCheckBox::toggled, q, [this, i](bool on) {
                QSettings prefs(vc3d::settingsFilePath(), QSettings::IniFormat);
                prefs.setValue(settingsGroup + QStringLiteral("/") + axes[i]->objectName(), on);
                refresh();
            });
        }
        volumetric = new QCheckBox(q->tr("Volumetric"), q);
        volumetric->setObjectName(QStringLiteral("fibers3DSliceVolumetric"));
        volumetric->setToolTip(
            q->tr("Use Volumetric mode from Composite View. Layer thickness and material settings are shared; the 3D camera controls the viewing angle."));
        row->addWidget(volumetric);
        QObject::connect(volumetric, &QCheckBox::toggled, q, [this](bool on) {
            const auto linked = linkedViewers();
            for (auto* manager : ViewerManager::allManagers()) {
                const auto& owned = manager->baseViewers();
                if (std::any_of(linked.begin(), linked.end(), [&](auto* view) {
                        return view && std::find(owned.begin(), owned.end(), view) != owned.end();
                    })) {
                    manager->setCompositeMethod(on ? "volumetric" : "max");
                    emit manager->volumetricModeRequested(on);
                }
            }
            refresh();
        });
        row->addSpacing(8);
        auto* label = new QLabel(q->tr("Opacity"), q);
        row->addWidget(label);
        auto* opacity = new QSpinBox(q);
        opacity->setObjectName(QStringLiteral("fibers3DSliceOpacity"));
        opacity->setRange(0, 100);
        opacity->setSingleStep(5);
        opacity->setSuffix(QStringLiteral(" %"));
        opacity->setValue(prefs.value(settingsGroup + QStringLiteral("/fibers3DSliceOpacity"), 70).toInt());
        auto* slider = new QSlider(Qt::Horizontal, q);
        slider->setObjectName(QStringLiteral("fibers3DSliceOpacitySlider"));
        slider->setRange(0, 100);
        slider->setValue(opacity->value());
        slider->setAccessibleName(q->tr("CT slice opacity"));
        label->setBuddy(opacity);
        QObject::connect(slider, &QSlider::valueChanged, opacity, &QSpinBox::setValue);
        QObject::connect(opacity, &QSpinBox::valueChanged, slider, &QSlider::setValue);
        QObject::connect(opacity, &QSpinBox::valueChanged, q, [this](int value) {
            QSettings prefs(vc3d::settingsFilePath(), QSettings::IniFormat);
            prefs.setValue(settingsGroup + QStringLiteral("/fibers3DSliceOpacity"), value);
            canvas->setCTSliceOpacity(value / 100.f);
        });
        canvas->setCTSliceOpacity(opacity->value() / 100.f);
        row->addWidget(slider, 1);
        row->addWidget(opacity);
        layout->addLayout(row);
        status = new QLabel(q);
        status->setTextFormat(Qt::PlainText);
        status->setWordWrap(true);
        status->setObjectName(QStringLiteral("fibers3DSliceStatus"));
        layout->addWidget(status);
        button->setCheckable(true);
        button->setToolTip(q->tr("Show the linked CT slice planes in 3D"));
        button->setEnabled(bool(state->currentVolume()));
        q->hide();
        poll.setInterval(150);
        debounce.setSingleShot(true);
        debounce.setInterval(150);
        QObject::connect(&poll, &QTimer::timeout, q, [this] { refresh(); });
        QObject::connect(&debounce, &QTimer::timeout, q, [this] { start(); });
        QObject::connect(button, &QToolButton::toggled, q, [this](bool on) {
            q->setVisible(on);
            if (on)
                q->resume();
            else
                q->cancel();
        });
        QObject::connect(state, &CState::volumeChanged, q, [this] { invalidate(); });
        QObject::connect(state, &CState::vpkgChanged, q, [this] { invalidate(); });
        QObject::connect(state, &CState::volumeClosing, q, [this] {
            resetXYPosition();
            if (xyPositionRow) xyPositionRow->setEnabled(false);
            q->cancel();
            volume.reset();
            cache.clear();
            button->setEnabled(button->isChecked());
        });
        QObject::connect(viewers, &ViewerManager::volumeWindowChanged, canvas, &scene3d::Scene3DCanvas::setCTWindow);
        canvas->setCTWindow(viewers->volumeWindowLow(), viewers->volumeWindowHigh());
        QObject::connect(&watcher, &QFutureWatcher<Result>::finished, q, [this] {
            busy = false;
            const bool stale = cancelled->load() || pending || !q->isVisible() || !button->isChecked();
            auto result = watcher.future().takeResult();
            if (!stale) {
                if (result.error.isEmpty()) {
                    cache = std::move(result.images);
                    present();
                    pending = std::any_of(cache.begin(), cache.end(), [](const auto& image) { return image->preview; });
                } else
                    status->setText(q->tr("CT slices: %1").arg(result.error));
            }
            if (pending)
                debounce.start();
        });
    }
    void enableXYPositionControl()
    {
        if (xyPositionRow)
            return;
        xyPositionRow = new QWidget(q);
        auto* row = new QHBoxLayout(xyPositionRow);
        row->setContentsMargins(0, 0, 0, 0);
        auto* label = new QLabel(q->tr("XY slice Z"), xyPositionRow);
        xyPosition = new QSlider(Qt::Horizontal, xyPositionRow);
        xyPosition->setObjectName(QStringLiteral("scene3DSliceZ"));
        xyPosition->setAccessibleName(q->tr("XY slice Z position"));
        xyPosition->setSingleStep(1);
        xyPosition->setPageStep(10);
        xyPosition->setToolTip(q->tr("Move the XY slice through the mesh along the volume Z axis. The mesh and Main views stay fixed."));
        xyPositionValue = new QDoubleSpinBox(xyPositionRow);
        xyPositionValue->setObjectName(QStringLiteral("scene3DSliceZValue"));
        xyPositionValue->setDecimals(2);
        xyPositionValue->setSuffix(q->tr(" vx"));
        xyPositionValue->setKeyboardTracking(false);
        label->setBuddy(xyPositionValue);
        followMain = new QCheckBox(q->tr("Follow Main"), xyPositionRow);
        followMain->setObjectName(QStringLiteral("scene3DFollowMainXY"));
        followMain->setChecked(true);
        row->addWidget(label);
        row->addWidget(xyPosition, 1);
        row->addWidget(xyPositionValue);
        row->addWidget(followMain);
        static_cast<QVBoxLayout*>(q->layout())->insertWidget(1, xyPositionRow);
        QObject::connect(xyPosition, &QSlider::valueChanged, q, [this](int value) {
            xyPositionValue->setValue(value);
        });
        QObject::connect(xyPositionValue, &QDoubleSpinBox::valueChanged, q, [this](double value) {
            const QSignalBlocker sliderBlock(xyPosition);
            const QSignalBlocker followBlock(followMain);
            xyPosition->setValue(int(std::lround(value)));
            followMain->setChecked(false);
            refresh();
        });
        QObject::connect(followMain, &QCheckBox::toggled, q, [this] { refresh(); });
        refresh();
    }
    void resetXYPosition()
    {
        if (!followMain) return;
        const QSignalBlocker blocker(followMain);
        followMain->setChecked(true);
    }
    void updateXYPosition(double mainZ)
    {
        if (!xyPositionRow) return;
        const auto current = state->currentVolume();
        xyPositionRow->setEnabled(bool(current) && axes[0]->isChecked());
        if (!current) return;
        const int last = int(std::clamp<double>(double(current->shapeXyz()[2]) - 1, 0, std::numeric_limits<int>::max()));
        const QSignalBlocker sliderBlock(xyPosition);
        const QSignalBlocker valueBlock(xyPositionValue);
        xyPosition->setRange(0, last);
        xyPositionValue->setRange(0, last);
        if (followMain->isChecked())
            xyPositionValue->setValue(mainZ);
        xyPosition->setValue(int(std::lround(xyPositionValue->value())));
    }
    void invalidate()
    {
        resetXYPosition();
        if (xyPositionRow)
            xyPositionRow->setEnabled(bool(state->currentVolume()) && axes[0]->isChecked());
        q->cancel();
        volume.reset();
        cache.clear();
        button->setEnabled(bool(state->currentVolume()) || button->isChecked());
        if (button->isChecked() && q->isVisible())
            q->resume();
    }
    std::array<VolumeViewerBase*, 3> linkedViewers() const
    {
        std::array<VolumeViewerBase*, 3> linked{};
        auto find = [&](ViewerManager* manager) {
            for (auto* viewer : manager->baseViewers())
                for (int i = 0; i < 3; ++i)
                    if (!linked[i] && viewer->surfName() == names[i] && viewer->currentVolume() == state->currentVolume())
                        linked[i] = viewer;
        };
        find(viewers);
        // Line Annotation owns a separate manager with generated cuts, not
        // the main workspace's named CT planes and Composite View settings.
        for (auto* manager : ViewerManager::allManagers())
            find(manager);
        return linked;
    }
    std::vector<slices::Plane> snapshot()
    {
        std::vector<slices::Plane> planes;
        const auto center = meshCenter ? meshCenter : this->center;
        const auto linkedViews = linkedViewers();
        for (int i = 0; i < 3; ++i) {
            if (!axes[i]->isChecked())
                continue;
            auto plane = std::dynamic_pointer_cast<PlaneSurface>(state->surface(names[i]));
            if (!plane)
                continue;
            auto* linked = linkedViews[i];
            cv::Vec3f a, b, d;
            auto* view = linked ? linked->graphicsView() : nullptr;
            if (!center && view && view->isVisible() && view->viewport()->width() > 1 && view->viewport()->height() > 1) {
                const auto size = view->viewport()->size();
                const auto offset = plane->normal({}, {}) * linked->normalOffset();
                // Perspective maps the viewport to a trapezoid. Bound all
                // four corners in the physical plane basis, not three corners
                // pretending to be a parallelogram.
                const auto x = plane->basisX(), y = plane->basisY(), origin = plane->origin();
                float x0 = INFINITY, x1 = -INFINITY, y0 = INFINITY, y1 = -INFINITY;
                for (const auto& pixel :
                     {QPoint(0, 0), QPoint(size.width() - 1, 0), QPoint(0, size.height() - 1), QPoint(size.width() - 1, size.height() - 1)}) {
                    const auto p = linked->sceneToVolume(view->mapToScene(pixel)) - origin;
                    x0 = std::min(x0, p.dot(x));
                    x1 = std::max(x1, p.dot(x));
                    y0 = std::min(y0, p.dot(y));
                    y1 = std::max(y1, p.dot(y));
                }
                a = origin + x * x0 + y * y0 + offset;
                b = origin + x * x1 + y * y0 + offset;
                d = origin + x * x0 + y * y1 + offset;
            } else {
                // A hidden CT pane has no useful viewport; keep its actual
                // plane and show a bounded patch around the shared focus.
                auto position = plane->origin();
                if (center)
                    position = cv::Vec3f(float((*center)[0]), float((*center)[1]), float((*center)[2]));
                else if (const auto* focus = state->poi("focus")) {
                    const auto n = plane->normal({}, {});
                    position = focus->p - n * (focus->p - position).dot(n);
                }
                if (linked && !center)
                    position += plane->normal({}, {}) * linked->normalOffset();
                const auto x = plane->basisX(), y = plane->basisY();
                double halfWidth = 512, halfHeight = 512;
                if (meshBounds) {
                    // Project the entire mesh AABB onto the plane, including
                    // its Z extent when tilted. Texture resolution must not
                    // limit the physical extent of the displayed slice.
                    halfWidth = halfHeight = 0;
                    for (int k = 0; k < 3; ++k) {
                        const double halfExtent = .5 * (meshBounds->upper[k] - meshBounds->lower[k]);
                        halfWidth += std::abs(x[k]) * halfExtent;
                        halfHeight += std::abs(y[k]) * halfExtent;
                    }
                    halfWidth = std::max(1., halfWidth * 1.05);
                    halfHeight = std::max(1., halfHeight * 1.05);
                }
                a = position - x * float(halfWidth) - y * float(halfHeight);
                b = a + x * float(2 * halfWidth);
                d = a + y * float(2 * halfHeight);
            }
            if (i == 0 && xyPositionRow) {
                const double mainZ = .5 * (double(b[2]) + double(d[2]));
                updateXYPosition(mainZ);
                if (!followMain->isChecked()) {
                    const float shift = float(xyPositionValue->value() - mainZ);
                    a[2] += shift;
                    b[2] += shift;
                    d[2] += shift;
                }
            }
            const double w = cv::norm(b - a), h = cv::norm(d - a);
            if (!std::isfinite(w) || !std::isfinite(h) || w < 1e-6 || h < 1e-6)
                continue;
            // Preserve aspect and world coordinates while bounding texture work.
            const double step = std::max(1., std::max(w, h) / 1023.);
            planes.push_back(
                {i, point(a), point(b - a), point(d - a), std::clamp(int(std::ceil(w / step)) + 1, 2, 1024), std::clamp(int(std::ceil(h / step)) + 1, 2, 1024)});
            if (volumetric->isChecked()) {
                auto& p = planes.back();
                const auto cs = linked ? linked->compositeRenderSettings() : CompositeRenderSettings{};
                p.volumetric = true;
                p.normal = point(plane->normal({}, {}) * (cs.planeReverseDirection ? -1.f : 1.f));
                p.front = std::max(0, cs.planeLayersFront);
                p.behind = std::max(0, cs.planeLayersBehind);
            }
        }
        return planes;
    }
    slices::Images matching() const
    {
        slices::Images images;
        for (const auto& plane : requested)
            for (const auto& image : cache)
                if (image->plane == plane) {
                    images.push_back(image);
                    break;
                }
        return images;
    }
    void present()
    {
        auto images = matching();
        canvas->setCTSlices(images);
        if (images.empty())
            status->setText(q->tr("Choose a CT plane."));
        else if (std::any_of(images.begin(), images.end(), [](const auto& image) { return image->preview; }))
            status->setText(q->tr("Refining CT slices…"));
        else if (volumetric->isChecked()) {
            const auto& p = images.front()->plane;
            const auto& slab = *images.front()->slab;
            status->setText(q->tr("Volumetric · %1 front / %2 behind · %3 × %4 × %5 samples · settings in Composite View")
                                .arg(p.front)
                                .arg(p.behind)
                                .arg(slab.width)
                                .arg(slab.height)
                                .arg(slab.depth));
        } else if (followMain && !followMain->isChecked() && axes[0]->isChecked())
            status->setText(meshCenter ? q->tr("XY at Z %1").arg(xyPositionValue->value(), 0, 'f', 2)
                                      : q->tr("XY at Z %1 · XZ/YZ linked to Main").arg(xyPositionValue->value(), 0, 'f', 2));
        else if (meshCenter)
            status->setText(q->tr("XY centered on mesh"));
        else
            status->setText(q->tr("Linked to CT views · single slices"));
    }
    void refresh()
    {
        if (!button->isChecked() || !q->isVisible())
            return;
        const auto current = state->currentVolume();
        if (current != volume) {
            cache.clear();
            requested.clear();
            volume = current;
        }
        if (xyPositionRow)
            xyPositionRow->setEnabled(bool(volume) && axes[0]->isChecked());
        if (!volume) {
            q->cancel();
            status->setText(q->tr("Open a CT volume."));
            return;
        }
        std::array<CompositeParams, 3> transfer;
        bool volumeMode = false;
        const auto linked = linkedViewers();
        for (int i = 0; i < 3; ++i)
            if (linked[i]) {
                transfer[i] = linked[i]->compositeRenderSettings().params;
                volumeMode = volumeMode || transfer[i].method == "volumetric";
            }
        {
            const QSignalBlocker blocker(volumetric);
            volumetric->setChecked(volumeMode);
        }
        canvas->setCTSliceTransfer(transfer);
        const auto planes = snapshot();
        if (planes == requested)
            return;
        if (cancelled)
            cancelled->store(true);
        requested = planes;
        auto ready = matching();
        canvas->setCTSlices(ready);  // never put old CT at a new plane position
        pending =
            ready.size() != requested.size() || std::any_of(ready.begin(), ready.end(), [](const auto& image) { return image->preview; });
        if (pending) {
            status->setText(q->tr("Loading CT slices…"));
            debounce.start();
        } else {
            debounce.stop();
            present();
        }
    }
    void start()
    {
        if (busy || !pending || !volume)
            return;
        busy = true;
        pending = false;
        cancelled = std::make_shared<std::atomic_bool>(false);
        const auto token = cancelled;
        const auto source = volume;
        const auto planes = requested;
        const auto ready = matching();
        const bool previewPass = ready.size() != planes.size();
        watcher.setFuture(QtConcurrent::run([=] {
            Result result;
            try {
                for (const auto& plane : planes) {
                    if (token->load())
                        return result;
                    auto found = std::find_if(ready.begin(), ready.end(), [&](const auto& image) { return image->plane == plane; });
                    if (found != ready.end() && (previewPass || !(*found)->preview))
                        result.images.push_back(*found);
                    else
                        result.images.push_back(slices::sample(plane, *source, token.get(), previewPass ? 256 : 1024));
                }
            } catch (const std::exception& e) {
                result.error = QString::fromUtf8(e.what());
            }
            return result;
        }));
    }
};
Fiber3DSliceControls::Fiber3DSliceControls(CState* s, ViewerManager* v, scene3d::Scene3DCanvas* c, QToolButton* b, QWidget* parent, QString settingsGroup)
    : QWidget(parent), d(std::make_unique<Impl>(this, s, v, c, b, std::move(settingsGroup)))
{
}
Fiber3DSliceControls::~Fiber3DSliceControls()
{
    // The parent may already have destroyed the canvas. Workers own snapshots.
    if (d->cancelled)
        d->cancelled->store(true);
}
void Fiber3DSliceControls::cancel()
{
    if (d->cancelled)
        d->cancelled->store(true);
    d->poll.stop();
    d->debounce.stop();
    d->pending = false;
    d->requested.clear();
    d->canvas->setCTSlices({});
}
void Fiber3DSliceControls::enableXYPositionControl()
{
    d->enableXYPositionControl();
}
void Fiber3DSliceControls::resume()
{
    if (d->button->isChecked() && isVisible()) {
        d->poll.start();
        d->refresh();
    }
}
void Fiber3DSliceControls::setCenter(std::optional<Point> center)
{
    if (d->center == center)
        return;
    d->center = center;
    d->refresh();
}
void Fiber3DSliceControls::setMeshBounds(std::optional<scene3d::Bounds> bounds)
{
    if (d->meshBounds.has_value() == bounds.has_value() &&
        (!bounds || (d->meshBounds->lower == bounds->lower && d->meshBounds->upper == bounds->upper)))
        return;
    d->meshBounds = bounds;
    std::optional<Point> center;
    if (bounds) {
        Point midpoint;
        for (int k = 0; k < 3; ++k)
            midpoint[k] = .5 * (bounds->lower[k] + bounds->upper[k]);
        center = midpoint;
    }
    const bool entering = center && !d->meshCenter;
    const bool leaving = !center && d->meshCenter;
    for (int i = 0; i < 3; ++i) {
        const QSignalBlocker blocker(d->axes[i]);
        if (entering) d->savedAxes[i] = d->axes[i]->isChecked();
        if (center) d->axes[i]->setChecked(i == 0);
        else if (leaving) d->axes[i]->setChecked(d->savedAxes[i]);
        d->axes[i]->setEnabled(!center);
    }
    d->meshCenter = center;
    if (d->followMain) {
        d->followMain->setText(center ? tr("Center on mesh") : tr("Follow Main"));
        if (center && center != d->lastMeshCenter) {
            const QSignalBlocker blocker(d->followMain);
            d->followMain->setChecked(true);
        }
    }
    if (center) d->lastMeshCenter = center;
    d->refresh();
}
}  // namespace vc3d::fibers3d
