#include "Fiber3DRibbonControls.hpp"
#include "Fiber3DRibbon.hpp"
#include "Fiber3DCanvas.hpp"
#include "CState.hpp"
#include "ViewerManager.hpp"
#include "VCSettings.hpp"
#include "OpenDataLasagna.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/lasagna/LasagnaNormalSampler.hpp"
#include <QFutureWatcher>
#include <QLabel>
#include <QSettings>
#include <QSlider>
#include <QSpinBox>
#include <QTimer>
#include <QToolButton>
#include <QVBoxLayout>
#include <QtConcurrent/QtConcurrentRun>

namespace vc3d::fibers3d
{
namespace
{
struct Result {
    std::shared_ptr<ribbon::Frames> frames;
    std::shared_ptr<ribbon::Image> image;
    QString warning, error;
};
}  // namespace
struct Fiber3DRibbonControls::Impl {
    Fiber3DRibbonControls* q;
    CState* state;
    Fiber3DCanvas* canvas;
    QToolButton* button;
    QSpinBox *width, *rotation;
    QLabel* status;
    QTimer debounce;
    QFutureWatcher<Result> watcher;
    std::shared_ptr<std::atomic_bool> cancelled;
    std::shared_ptr<const Scene> scene;
    QString key, warning;
    QStringList normalSignature;
    std::shared_ptr<ribbon::Frames> frames;
    std::shared_ptr<ribbon::Image> image;
    bool busy = false, pending = false;

    Impl(Fiber3DRibbonControls* widget, CState* s, ViewerManager* viewers, Fiber3DCanvas* c, QToolButton* b)
        : q(widget), state(s), canvas(c), button(b)
    {
        q->setObjectName(QStringLiteral("fibers3DRibbonControls"));
        button->setCheckable(true);
        button->setToolTip(q->tr("Show a CT ribbon along the selected fiber in 3D"));
        auto* layout = new QVBoxLayout(q);
        layout->setContentsMargins(2, 2, 2, 2);
        layout->setSpacing(4);
        QSettings prefs(vc3d::settingsFilePath(), QSettings::IniFormat);
        auto row = [&](const QString& label, const char* name, int lo, int hi, int value, const QString& suffix) {
            auto* line = new QHBoxLayout;
            auto* title = new QLabel(label, q);
            title->setMinimumWidth(58);
            auto* spin = new QSpinBox(q);
            spin->setObjectName(QString::fromLatin1(name));
            spin->setRange(lo, hi);
            spin->setValue(prefs.value(QStringLiteral("fibers3d/") + QString::fromLatin1(name), value).toInt());
            spin->setSuffix(suffix);
            spin->setKeyboardTracking(false);
            auto* slider = new QSlider(Qt::Horizontal, q);
            slider->setObjectName(QString::fromLatin1(name) + QStringLiteral("Slider"));
            slider->setRange(lo, hi);
            slider->setValue(spin->value());
            slider->setAccessibleName(label);
            title->setBuddy(spin);
            QObject::connect(slider, &QSlider::valueChanged, spin, &QSpinBox::setValue);
            QObject::connect(spin, &QSpinBox::valueChanged, slider, &QSlider::setValue);
            line->addWidget(title);
            line->addWidget(slider, 1);
            line->addWidget(spin);
            layout->addLayout(line);
            return spin;
        };
        width = row(q->tr("Width"), "fibers3DRibbonWidth", 1, 512, 32, q->tr(" vx"));
        width->setToolTip(q->tr("Full ribbon width, centered on the fiber, in active CT voxels"));
        rotation = row(q->tr("Rotation"), "fibers3DRibbonRotation", -180, 180, 0, QString::fromUtf8("°"));
        rotation->setToolTip(q->tr("Rotate the sampling plane around the fiber. Without Lasagna normals the starting plane is arbitrary."));
        status = new QLabel(q);
        status->setWordWrap(true);
        status->setTextFormat(Qt::PlainText);
        status->setObjectName(QStringLiteral("fibers3DRibbonStatus"));
        layout->addWidget(status);
        q->hide();
        button->setEnabled(false);
        debounce.setSingleShot(true);
        debounce.setInterval(180);
        QObject::connect(&debounce, &QTimer::timeout, q, [this] { start(); });
        QObject::connect(button, &QToolButton::toggled, q, [this](bool enabled) {
            q->setVisible(enabled);
            button->setEnabled(enabled || (fiber() && state->currentVolume()));
            request();
        });
        for (auto* spin : {width, rotation})
            QObject::connect(spin, &QSpinBox::valueChanged, q, [this, spin] {
                QSettings prefs(vc3d::settingsFilePath(), QSettings::IniFormat);
                prefs.setValue(QStringLiteral("fibers3d/") + spin->objectName(), spin->value());
                request();
            });
        canvas->setCTWindow(viewers->volumeWindowLow(), viewers->volumeWindowHigh());
        QObject::connect(viewers, &ViewerManager::volumeWindowChanged, canvas, &Fiber3DCanvas::setCTWindow);
        QObject::connect(&watcher, &QFutureWatcher<Result>::finished, q, [this] {
            busy = false;
            const bool stale = cancelled->load() || pending;
            auto result = watcher.future().takeResult();  // worker catches all std::exceptions
            if (!stale) {
                if (result.frames)
                    frames = result.frames;
                warning = result.warning;
                image = result.image;
                if (!result.error.isEmpty())
                    status->setText(q->tr("CT ribbon: %1").arg(result.error));
                else if (image)
                    present();
            }
            if (pending)
                debounce.start();
        });
    }

    const Fiber* fiber() const
    {
        if (scene)
            for (const auto& f : scene->fibers)
                if (f.key == key)
                    return &f;
        return nullptr;
    }
    void stop()
    {
        if (cancelled)
            cancelled->store(true);
        pending = false;
        debounce.stop();
        canvas->setCTRibbon({});
    }
    void present()
    {
        if (!button->isChecked())
            return;
        canvas->setCTRibbon(image);
        QString text;
        if (!image->sampledNormals)
            text = q->tr("No Lasagna normals · adjust rotation");
        else if (image->sampledNormals == image->rows)
            text = q->tr("Lasagna orientation");
        else
            text = q->tr("Lasagna orientation · transported frame where normals are missing");
        if (!warning.isEmpty())
            text += QStringLiteral("\n") + warning;
        status->setText(text);
    }
    void request()
    {
        stop();
        if (!button->isChecked())
            return;
        if (!fiber() || !state->currentVolume()) {
            status->setText(q->tr("Select a displayed fiber."));
            return;
        }
        // A newly attached/selected normal dataset must invalidate the fallback
        // cache even when the selected fiber and its display settings stay put.
        QStringList signature;
        if (const auto pkg = state->vpkg()) {
            signature << QString::fromStdString(pkg->path().string()) << QString::fromStdString(pkg->selectedLasagnaDataset())
                      << QString::fromStdString(state->currentVolumeId());
            for (const auto& entry : pkg->lasagnaDatasetEntries()) {
                signature << QStringLiteral("location:") + QString::fromStdString(entry.location);
                for (const auto& tag : entry.tags)
                    signature << QStringLiteral("tag:") + QString::fromStdString(tag);
            }
        }
        if (signature != normalSignature) {
            normalSignature = std::move(signature);
            frames.reset();
            image.reset();
            warning.clear();
        }
        if (image && image->widthVoxels == width->value() && image->rotationDegrees == rotation->value()) {
            present();
            return;
        }
        status->setText(q->tr("Loading CT ribbon…"));
        pending = true;
        debounce.start();
    }
    void start()
    {
        if (busy || !pending || !fiber())
            return;
        pending = false;
        busy = true;
        cancelled = std::make_shared<std::atomic_bool>(false);
        const auto token = cancelled;
        const auto volume = state->currentVolume();
        const Fiber selected = *fiber();
        const auto clip = scene->clipBounds;
        const auto cachedFrames = frames;
        const auto previousWarning = warning;
        const int w = width->value(), angle = rotation->value();
        opendata::LasagnaResolutionSnapshot normalSource;
        if (const auto pkg = state->vpkg())
            normalSource =
                {pkg->path(), pkg->lasagnaDatasetEntries(), pkg->selectedLasagnaDataset(), pkg->volumeTags(state->currentVolumeId()), volume->shape()};
        watcher.setFuture(QtConcurrent::run([=] {
            Result result;
            try {
                result.frames = cachedFrames;
                result.warning = previousWarning;
                if (!result.frames) {
                    std::shared_ptr<vc::lasagna::LasagnaNormalSampler> sampler;
                    try {
                        if (const auto resolved = opendata::resolveLasagnaSnapshot(normalSource)) {
                            vc::lasagna::LasagnaDatasetOpenOptions options;
                            options.workingToBaseScale = resolved->workingToBaseScale;
                            options.remoteCacheRoot = vc3d::remoteCachePathFs();
                            const auto location =
                                resolved->manifestBacked
                                    ? resolved->manifestPath.string()
                                    : (vc::project::isLocationRemote(resolved->sourceManifestLocation)
                                           ? resolved->sourceManifestLocation
                                           : vc::project::resolveLocalPath(resolved->sourceManifestLocation, normalSource.projectPath.parent_path())
                                                 .string());
                            auto dataset = vc::lasagna::LasagnaDataset::openLocation(location, options);
                            sampler = std::make_shared<vc::lasagna::LasagnaNormalSampler>(dataset);
                        }
                    } catch (const std::exception& e) {
                        result.warning = QStringLiteral("Lasagna unavailable: %1").arg(QString::fromUtf8(e.what()));
                    }
                    ribbon::NormalAt normals;
                    if (sampler)
                        normals = [sampler](const cv::Vec3d& p) -> std::optional<cv::Vec3d> {
                            const auto n = sampler->sampleNormal(p);
                            return n.valid ? std::optional<cv::Vec3d>(n.normal) : std::nullopt;
                        };
                    result.frames = std::make_shared<ribbon::Frames>(ribbon::makeFrames(selected, clip, normals, token.get()));
                }
                result.image = ribbon::sample(*result.frames, *volume, w, angle, token.get());
                result.image->key = selected.key;
            } catch (const std::exception& e) {
                result.error = QString::fromUtf8(e.what());
            }
            return result;
        }));
    }
};
Fiber3DRibbonControls::Fiber3DRibbonControls(CState* s, ViewerManager* v, Fiber3DCanvas* c, QToolButton* b, QWidget* parent)
    : QWidget(parent), d(std::make_unique<Impl>(this, s, v, c, b))
{
}
Fiber3DRibbonControls::~Fiber3DRibbonControls()
{
    if (d->cancelled)
        d->cancelled->store(true);
}
void Fiber3DRibbonControls::cancel()
{
    d->stop();
}
void Fiber3DRibbonControls::setSelection(std::shared_ptr<const Scene> scene, const QString& key)
{
    if (scene == d->scene && key == d->key)
        return;
    d->scene = std::move(scene);
    d->key = key;
    d->frames.reset();
    d->image.reset();
    d->warning.clear();
    d->button->setEnabled(d->button->isChecked() || (d->fiber() && d->state->currentVolume()));
    d->request();
}
}  // namespace vc3d::fibers3d
