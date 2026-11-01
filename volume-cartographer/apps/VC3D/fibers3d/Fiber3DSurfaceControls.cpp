#include "Fiber3DSurfaceControls.hpp"
#include "Fiber3DSurfaceMesh.hpp"
#include "scene3d/Scene3DCanvas.hpp"
#include "scene3d/VolumePreview.hpp"
#include "scene3d/MemoryBudgetInput.hpp"
#include "CState.hpp"
#include "VCSettings.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/types/Volume.hpp"
#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QSignalBlocker>
#include <algorithm>
#include <cmath>
#include <QFutureWatcher>
#include <QLabel>
#include <QPushButton>
#include <QSettings>
#include <QSlider>
#include <QSpinBox>
#include <QTimer>
#include <QVBoxLayout>
#include <QtConcurrent/QtConcurrentRun>
#include <stdexcept>

namespace vc3d::fibers3d
{
namespace
{
struct Result {
    std::shared_ptr<const surface::Mesh> mesh;
    std::shared_ptr<const scene3d::VolumeData> volume;
    QString error;
};
}  // namespace
struct Fiber3DSurfaceControls::Impl {
    Fiber3DSurfaceControls* q;
    CState* state;
    QString settingsGroup;
    scene3d::Scene3DCanvas* canvas;
    QCheckBox* enabled;
    QCheckBox* textured;
    QComboBox* textureLevel;
    int preferredTextureLevel = 3;
    bool volumeSource = false;
    QLabel* title;
    QWidget* densityRow;
    QDoubleSpinBox* density;
    std::function<void(bool, double)> applyFilter;
    QCheckBox* nearby;
    QSpinBox* distance;
    QLabel* status;
    QPushButton* fit;
    QTimer debounce;
    QFutureWatcher<Result> watcher;
    std::shared_ptr<std::atomic_bool> cancelled;
    bool suspended = true, busy = false, pending = false;
    size_t maxBytes = scene3d::defaultMemoryBudget;

    Impl(Fiber3DSurfaceControls* widget, CState* s, scene3d::Scene3DCanvas* c, QString group, std::function<void(bool, double)> filter)
        : q(widget), state(s), settingsGroup(std::move(group)), canvas(c), applyFilter(std::move(filter))
    {
        auto* layout = new QVBoxLayout(q);
        layout->setContentsMargins(0, 0, 0, 0);
        layout->setSpacing(10);
        title = new QLabel(q->tr("Surface selected in Main"), q);
        auto font = title->font();
        font.setBold(true);
        title->setFont(font);
        layout->addWidget(title);
        status = new QLabel(q);
        status->setObjectName(QStringLiteral("fibers3DSurfaceStatus"));
        status->setTextFormat(Qt::PlainText);
        status->setWordWrap(true);
        layout->addWidget(status);
        layout->addSpacing(5);
        QSettings prefs(vc3d::settingsFilePath(), QSettings::IniFormat);
        enabled = new QCheckBox(q->tr("Display surface mesh"), q);
        enabled->setObjectName(QStringLiteral("fibers3DSurfaceMesh"));
        enabled->setToolTip(q->tr("Display the surface selected in Main, in the active CT coordinate frame."));
        enabled->setChecked(prefs.value(settingsGroup + QStringLiteral("/fibers3DSurfaceMesh"), !applyFilter).toBool());
        canvas->setSurfaceVisible(enabled->isChecked());
        layout->addWidget(enabled);
        textured = new QCheckBox(q->tr("CT texture"), q);
        textured->setObjectName(QStringLiteral("scene3DSurfaceTexture"));
        textured->setToolTip(q->tr("Sample CT on the surface. Uses the active CT window/level; turn off to show the shaded mesh."));
        textured->setChecked(prefs.value(settingsGroup + QStringLiteral("/surfaceTextured"), true).toBool());
        canvas->setSurfaceTextured(textured->isChecked());
        layout->addWidget(textured);
        auto* levelRow = new QHBoxLayout;
        auto* levelLabel = new QLabel(q->tr("CT texture OME-Zarr level"), q);
        textureLevel = new QComboBox(q);
        textureLevel->setObjectName(QStringLiteral("scene3DSurfaceTextureLevel"));
        textureLevel->setToolTip(q->tr("Higher levels sample coarser CT data and read fewer volume chunks. Only levels present in the active volume are listed."));
        levelLabel->setBuddy(textureLevel);
        levelRow->addWidget(levelLabel);
        levelRow->addWidget(textureLevel, 1);
        layout->addLayout(levelRow);
        preferredTextureLevel = std::max(0, prefs.value(settingsGroup + QStringLiteral("/surfaceTextureLevel"), 3).toInt());
        updateTextureLevels();
        QObject::connect(textureLevel, &QComboBox::currentIndexChanged, q, [this](int index) {
            if (index < 0) return;
            preferredTextureLevel = textureLevel->currentData().toInt();
            QSettings prefs(vc3d::settingsFilePath(), QSettings::IniFormat);
            prefs.setValue(settingsGroup + QStringLiteral("/surfaceTextureLevel"), preferredTextureLevel);
            invalidate();
        });
        QObject::connect(textured, &QCheckBox::toggled, q, [this](bool on) {
            canvas->setSurfaceTextured(on);
            textureLevel->setEnabled(on && textureLevel->count() > 0);
            QSettings prefs(vc3d::settingsFilePath(), QSettings::IniFormat);
            prefs.setValue(settingsGroup + QStringLiteral("/surfaceTextured"), on);
            if ((on && (!canvas->surfaceMesh() || !canvas->surfaceMesh()->texture)) || busy)
                invalidate();
        });
        densityRow = new QWidget(q);
        auto* densityLayout = new QHBoxLayout(densityRow);
        densityLayout->setContentsMargins(0,0,0,0);
        auto* densityLabel = new QLabel(q->tr("Volume density"), densityRow);
        density = new QDoubleSpinBox(densityRow);
        density->setObjectName(QStringLiteral("scene3DVolumeDensity"));
        density->setRange(0, 100);
        density->setSingleStep(.25);
        density->setKeyboardTracking(false);
        density->setValue(prefs.value(settingsGroup + QStringLiteral("/volumeDensity"), 8.).toDouble());
        density->setToolTip(q->tr("Control absorption through the CT volume. Lower density reveals more internal structure; zero makes the volume transparent."));
        densityLabel->setBuddy(density);
        densityLayout->addWidget(densityLabel);
        densityLayout->addWidget(density, 1);
        layout->addWidget(densityRow);
        densityRow->hide();
        canvas->setCTVolumeDensity(float(density->value()));
        QObject::connect(density, &QDoubleSpinBox::valueChanged, q, [this](double value) {
            QSettings prefs(vc3d::settingsFilePath(), QSettings::IniFormat);
            prefs.setValue(settingsGroup + QStringLiteral("/volumeDensity"), value);
            canvas->setCTVolumeDensity(float(value));
        });
        auto* options = new QWidget(q);
        auto* line = new QHBoxLayout(options);
        line->setContentsMargins(0, 0, 0, 0);
        auto* label = new QLabel(q->tr("Opacity"), q);
        auto* opacity = new QSpinBox(q);
        opacity->setObjectName(QStringLiteral("fibers3DSurfaceOpacity"));
        opacity->setRange(0, 100);
        opacity->setSingleStep(5);
        opacity->setSuffix(QStringLiteral(" %"));
        opacity->setValue(prefs.value(settingsGroup + QStringLiteral("/fibers3DSurfaceOpacity"), 100).toInt());
        auto* slider = new QSlider(Qt::Horizontal, q);
        slider->setObjectName(QStringLiteral("fibers3DSurfaceOpacitySlider"));
        slider->setRange(0, 100);
        slider->setSingleStep(5);
        slider->setValue(opacity->value());
        slider->setAccessibleName(q->tr("Surface mesh opacity"));
        label->setBuddy(opacity);
        line->addWidget(label);
        line->addWidget(slider, 1);
        line->addWidget(opacity);
        layout->addWidget(options);
        layout->addSpacing(5);
        auto* fiberOptions = new QWidget(q);
        auto* fiberLayout = new QVBoxLayout(fiberOptions);
        fiberLayout->setContentsMargins(0, 0, 0, 0);
        fiberLayout->addWidget(new QLabel(q->tr("Nearby fibers"), fiberOptions));
        nearby = new QCheckBox(q->tr("Only fibers near surface"), q);
        nearby->setObjectName(QStringLiteral("fibers3DNearSurface"));
        nearby->setChecked(prefs.value(settingsGroup + QStringLiteral("/fibers3DNearSurface"), false).toBool());
        nearby->setToolTip(
            q->tr("Keep whole loaded fibers that pass within this distance of the selected surface, on either side. Works even when the mesh is hidden. Combines with other fiber filters."));
        fiberLayout->addWidget(nearby);
        auto* distanceRow = new QHBoxLayout;
        auto* distanceLabel = new QLabel(q->tr("Distance to surface"), q);
        distance = new QSpinBox(q);
        distance->setObjectName(QStringLiteral("fibers3DSurfaceDistance"));
        distance->setRange(0, 100000);
        distance->setSingleStep(8);
        distance->setSuffix(q->tr(" vx"));
        distance->setKeyboardTracking(false);
        distance->setValue(prefs.value(settingsGroup + QStringLiteral("/fibers3DSurfaceDistance"), 64).toInt());
        distance->setEnabled(nearby->isChecked());
        distanceLabel->setBuddy(distance);
        distanceRow->addWidget(distanceLabel);
        distanceRow->addWidget(distance, 1);
        fiberLayout->addLayout(distanceRow);
        auto* help = new QLabel(q->tr("Keeps whole fibers within this distance, on either side. Also works with the mesh hidden."), q);
        help->setWordWrap(true);
        fiberLayout->addWidget(help);
        layout->addWidget(fiberOptions);
        fiberOptions->setVisible(bool(applyFilter));
        if (applyFilter)
            applyFilter(nearby->isChecked(), distance->value());
        fit = new QPushButton(q->tr("Fit surface"), q);
        fit->setObjectName(QStringLiteral("fibers3DFitSurface"));
        fit->setAutoDefault(false);
        fit->setEnabled(false);
        layout->addWidget(fit, 0, Qt::AlignLeft);
        QObject::connect(slider, &QSlider::valueChanged, opacity, &QSpinBox::setValue);
        QObject::connect(opacity, &QSpinBox::valueChanged, slider, &QSlider::setValue);
        QObject::connect(opacity, &QSpinBox::valueChanged, q, [this](int value) {
            canvas->setSurfaceOpacity(value / 100.f);
            QSettings prefs(vc3d::settingsFilePath(), QSettings::IniFormat);
            prefs.setValue(settingsGroup + QStringLiteral("/fibers3DSurfaceOpacity"), value);
        });
        canvas->setSurfaceOpacity(opacity->value() / 100.f);
        QObject::connect(fit, &QPushButton::clicked, q, [this] {
            if (volumeSource) canvas->fitAll();
            else canvas->fitSurface();
        });
        options->setEnabled(enabled->isChecked());
        QObject::connect(enabled, &QCheckBox::toggled, q, [this, options](bool on) {
            options->setEnabled(on);
            canvas->setSurfaceVisible(on);
            QSettings prefs(vc3d::settingsFilePath(), QSettings::IniFormat);
            prefs.setValue(settingsGroup + (volumeSource ? QStringLiteral("/volumeVisible") : QStringLiteral("/fibers3DSurfaceMesh")), on);
            refreshOptions();
        });
        auto filterChanged = [this] {
            distance->setEnabled(nearby->isChecked());
            QSettings prefs(vc3d::settingsFilePath(), QSettings::IniFormat);
            prefs.setValue(settingsGroup + QStringLiteral("/fibers3DNearSurface"), nearby->isChecked());
            prefs.setValue(settingsGroup + QStringLiteral("/fibers3DSurfaceDistance"), distance->value());
            if (applyFilter)
                applyFilter(nearby->isChecked(), distance->value());
            refreshOptions();
        };
        QObject::connect(nearby, &QCheckBox::toggled, q, filterChanged);
        QObject::connect(distance, &QSpinBox::valueChanged, q, filterChanged);
        debounce.setSingleShot(true);
        debounce.setInterval(150);
        QObject::connect(&debounce, &QTimer::timeout, q, [this] { start(); });
        QObject::connect(state, &CState::surfaceChanged, q, [this](const std::string& name) {
            if (!volumeSource && (name.empty() || name == "segmentation"))
                invalidate();
        });
        QObject::connect(state, &CState::surfaceWillBeDeleted, q, [this](const std::string& name) {
            if (!volumeSource && name == "segmentation")
                clear();
        });
        QObject::connect(state, &CState::volumeClosing, q, [this] {
            clear();
            const QSignalBlocker blocker(textureLevel);
            textureLevel->clear();
            textureLevel->setEnabled(false);
        });
        QObject::connect(state, &CState::volumeChanged, q, [this] { updateTextureLevels(); invalidate(); });
        QObject::connect(state, &CState::vpkgChanged, q, [this] { invalidate(); });
        QObject::connect(&watcher, &QFutureWatcher<Result>::finished, q, [this] {
            busy = false;
            auto result = watcher.future().takeResult();
            if (!cancelled->load() && !suspended && wanted()) {
                canvas->setSurfaceMesh(result.mesh);
                canvas->setCTVolume(result.volume);
                const bool valid = bool(result.volume) || (result.mesh && !result.mesh->indices.empty());
                fit->setEnabled(valid);
                if (!result.error.isEmpty())
                    status->setText(result.error);
                else if (valid) {
                    status->setText(volumeSource ? q->tr("CT volume · Zarr level %1 · %2 × %3 × %4 preview")
                        .arg(result.volume->images.front()->level).arg(result.volume->images.front()->slab->width)
                        .arg(result.volume->images.front()->slab->height).arg(result.volume->images.front()->slab->depth) :
                        (state->activeSurfaceId().empty() ? q->tr("Surface selected in Main")
                                                         : q->tr("Surface %1").arg(QString::fromStdString(state->activeSurfaceId()))));
                    if (result.volume && result.volume->images.front()->level != result.volume->requestedLevel)
                        status->setText(status->text() + q->tr(" (requested %1; preview uses a coarser level)").arg(result.volume->requestedLevel));
                    if (result.volume && result.volume->resampled)
                        status->setText(status->text() + q->tr(" · downsampled overview"));
                    if (result.mesh && result.mesh->texture)
                        status->setText(status->text() + q->tr(" · CT level %1").arg(result.mesh->texture->level));
                } else
                    status->setText(volumeSource ? q->tr("No visible CT intensity at this preview resolution.") : q->tr("The selected surface has no valid mesh cells."));
            }
            if (pending)
                debounce.start();
        });
        status->setText(q->tr("Surface selected in Main"));
    }
    void updateTextureLevels()
    {
        const QSignalBlocker blocker(textureLevel);
        textureLevel->clear();
        if (const auto volume = state->currentVolume()) {
            auto levels = volume->presentScaleLevels();
            std::sort(levels.begin(), levels.end());
            for (int level : levels)
                textureLevel->addItem(q->tr("Level %1").arg(level), level);
            // Prefer the requested level, then the nearest coarser available
            // level; if the pyramid is shorter, use its coarsest level. Do not
            // overwrite the preference when temporarily using another volume.
            const auto found = std::lower_bound(levels.begin(), levels.end(), preferredTextureLevel);
            if (!levels.empty())
                textureLevel->setCurrentIndex(found == levels.end() ? int(levels.size()-1) : int(found-levels.begin()));
        }
        textureLevel->setEnabled((volumeSource || textured->isChecked()) && textureLevel->count() > 0);
    }
    bool wanted() const { return enabled->isChecked() || (applyFilter && nearby->isChecked()); }
    void refreshOptions()
    {
        if (!wanted() || !canvas->surfaceMesh() ||
            (enabled->isChecked() && textured->isChecked() && state->currentVolume() && !canvas->surfaceMesh()->texture))
            invalidate();
    }
    void clear()
    {
        if (cancelled)
            cancelled->store(true);
        pending = false;
        debounce.stop();
        fit->setEnabled(false);
        canvas->setSurfaceMesh({});
        canvas->setCTVolume({});
    }
    void invalidate()
    {
        clear();
        if (!wanted()) {
            status->setText(volumeSource ? q->tr("CT volume hidden") : q->tr("Surface selected in Main"));
            return;
        }
        pending = true;
        if (!suspended) {
            status->setText(volumeSource ? q->tr("Preparing CT volume preview…") : q->tr("Preparing surface mesh…"));
            debounce.start();
        }
    }
    void start()
    {
        if (busy || suspended || !pending || !wanted())
            return;
        pending = false;
        auto quad = std::dynamic_pointer_cast<QuadSurface>(state->surface("segmentation"));
        if (!volumeSource && !quad) {
            status->setText(q->tr("Select a surface in Main."));
            return;
        }
        try {
            // QuadSurface's live grid can be edited in place. Copy on its GUI
            // owner thread; workers never access mutable surface state.
            cv::Mat_<cv::Vec3f> points;
            cv::Vec2f scale{1,1};
            if (!volumeSource) {
                const auto grid = quad->gridSize();
                if (uint64_t(grid.width) * uint64_t(grid.height) > maxBytes / (sizeof(surface::Vertex) + 6 * sizeof(uint32_t)))
                    throw std::runtime_error("Surface exceeds the 3D memory budget. Increase Memory budget in Settings.");
                points = quad->rawPoints().clone();
                scale = quad->scale();
            }
            const bool overview = volumeSource, withTexture = !volumeSource && textured->isChecked() && enabled->isChecked();
            auto volume = overview || withTexture ? state->currentVolume() : nullptr;
            if (overview && !volume) { status->setText(q->tr("Attach a CT volume.")); return; }
            const int level = textureLevel->currentData().toInt();
            cancelled = std::make_shared<std::atomic_bool>(false);
            busy = true;
            watcher.setFuture(QtConcurrent::run([points = std::move(points), stop = cancelled, budget = maxBytes, scale, volume, level, overview, withTexture] {
                Result r;
                try {
                    if (overview) {
                        r.volume = scene3d::sampleVolume(*volume, level, budget, *stop);
                        return r;
                    }
                    auto mesh = surface::build(points, budget, *stop);
                    r.mesh = mesh;
                    if (mesh && !mesh->indices.empty() && volume && withTexture && !stop->load()) {
                        mesh->texture = surface::sampleTexture(points, scale, *volume, *stop, level);
                    }
                } catch (const std::exception& e) {
                    r.error = QString::fromUtf8(e.what());
                }
                return r;
            }));
        } catch (const std::exception& e) {
            status->setText(QString::fromUtf8(e.what()));
        }
    }
};
Fiber3DSurfaceControls::Fiber3DSurfaceControls(CState* state, scene3d::Scene3DCanvas* canvas, QWidget* parent,
                                               QString settingsGroup, std::function<void(bool, double)> filterChanged)
    : QWidget(parent), d(std::make_unique<Impl>(this, state, canvas, std::move(settingsGroup), std::move(filterChanged)))
{
}
Fiber3DSurfaceControls::~Fiber3DSurfaceControls()
{
    if (d->cancelled)
        d->cancelled->store(true);
}
void Fiber3DSurfaceControls::resume()
{
    d->suspended = false;
    d->invalidate();
}
void Fiber3DSurfaceControls::suspend()
{
    d->suspended = true;
    d->clear();
}
void Fiber3DSurfaceControls::setMemoryBudget(size_t bytes)
{
    if (d->maxBytes == bytes)
        return;
    d->maxBytes = bytes;
    d->invalidate();
}
void Fiber3DSurfaceControls::setVolumeSource(bool volume)
{
    if (d->volumeSource == volume) return;
    d->volumeSource = volume;
    d->title->setText(volume ? tr("CT volume") : tr("Surface selected in Main"));
    d->enabled->setText(volume ? tr("Display CT volume") : tr("Display surface mesh"));
    d->enabled->setToolTip(volume ? tr("Display a bounded, rotatable CT volume preview.") : tr("Display the surface selected in Main."));
    d->fit->setText(volume ? tr("Fit volume") : tr("Fit surface"));
    d->densityRow->setVisible(volume);
    d->textured->setVisible(!volume);
    d->textureLevel->setEnabled((volume || d->textured->isChecked()) && d->textureLevel->count() > 0);
    QSettings prefs(vc3d::settingsFilePath(), QSettings::IniFormat);
    d->enabled->setChecked(prefs.value(d->settingsGroup + (volume ? QStringLiteral("/volumeVisible") : QStringLiteral("/fibers3DSurfaceMesh")), !d->applyFilter).toBool());
    d->canvas->setSurfaceVisible(d->enabled->isChecked());
    d->invalidate();
}
}  // namespace vc3d::fibers3d
