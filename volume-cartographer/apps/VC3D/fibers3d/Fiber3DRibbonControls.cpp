#include "Fiber3DRibbonControls.hpp"
#include "Fiber3DRibbon.hpp"
#include "Fiber3DCanvas.hpp"
#include "CState.hpp"
#include "LineAnnotationController.hpp"
#include "FiberCollectionController.hpp"
#include "Fiber3DData.hpp"
#include <QSignalBlocker>
#include "ViewerManager.hpp"
#include "VCSettings.hpp"
#include "OpenDataLasagna.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/lasagna/LasagnaNormalSampler.hpp"
#include "vc/core/render/ChunkCache.hpp"
#include "vc/core/render/ChunkedPlaneSampler.hpp"
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
    LineAnnotationController* annotations;
    FiberCollectionController* automated;
    Fiber3DCanvas* canvas;
    QToolButton* button;
    QSpinBox *width, *rotation;
    QLabel* status;
    QTimer debounce, refresh;
    std::shared_ptr<vc::render::ChunkCache> cache;
    vc::render::IChunkedArray::ChunkReadyCallbackId listener = 0;
    std::shared_ptr<std::atomic_bool> chunksReady = std::make_shared<std::atomic_bool>(false);
    int requestedLevel = -1;
    bool suspended = false;
    QFutureWatcher<Result> watcher;
    std::shared_ptr<std::atomic_bool> cancelled;
    std::shared_ptr<const Scene> scene;
    QString key, warning;
    FramesFactory canonicalSurface;
    QStringList normalSignature;
    std::shared_ptr<ribbon::Frames> frames;
    std::shared_ptr<ribbon::Image> image;
    bool busy = false, pending = false;

    Impl(Fiber3DRibbonControls* widget, CState* s, ViewerManager* viewers, LineAnnotationController* a, FiberCollectionController* f, Fiber3DCanvas* c, QToolButton* b)
        : q(widget), state(s), annotations(a), automated(f), canvas(c), button(b)
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
        rotation->setToolTip(
            q->tr("Rotation is available only without Lasagna normals. With Lasagna, this is the Line Annotation surface."));
        status = new QLabel(q);
        status->setWordWrap(true);
        status->setTextFormat(Qt::PlainText);
        status->setObjectName(QStringLiteral("fibers3DRibbonStatus"));
        layout->addWidget(status);
        q->hide();
        button->setEnabled(false);
        refresh.setInterval(500);
        QObject::connect(&refresh, &QTimer::timeout, q, [this] {
            if (busy || suspended || !button->isChecked() || !image || image->complete || image->errors)
                return;
            if (chunksReady->exchange(false)) {
                pending = true;
                start();
            }
        });
        QObject::connect(canvas, &Fiber3DCanvas::viewScaleChanged, q, [this] {
            if (button->isChecked() && cache && levelForView() != requestedLevel)
                request();
        });
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
            else if (!stale && image && !image->complete && !image->errors && !suspended)
                refresh.start();
        });
    }

    ~Impl() { releaseCache(); }

    void releaseCache()
    {
        if (cache && listener)
            cache->removeChunkReadyListener(listener);
        listener = 0;
        cache.reset();
        chunksReady->store(false);
    }
    int levelForView() const { return vc::render::ChunkedPlaneSampler::sourceLevelForView(*cache, canvas->pixelsPerVoxel()); }
    void observeCache()
    {
        const auto next = state->currentVolume()->sharedChunkCache();
        if (next == cache)
            return;
        releaseCache();
        cache = next;
        const auto ready = chunksReady;
        listener = cache->addChunkReadyListener([ready] { ready->store(true); });
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
        refresh.stop();
        canvas->setCTRibbon({});
    }
    void present()
    {
        if (!button->isChecked())
            return;
        canvas->setCTRibbon(image);
        const bool manual = !canonicalSurface && image->sampledNormals == 0;
        rotation->setEnabled(manual);
        q->findChild<QSlider*>(QStringLiteral("fibers3DRibbonRotationSlider"))->setEnabled(manual);
        if (!manual && rotation->value() != 0) {
            QSignalBlocker blocked(rotation);
            rotation->setValue(0);
            q->findChild<QSlider*>(QStringLiteral("fibers3DRibbonRotationSlider"))->setValue(0);
        }
        QString text;
        if (canonicalSurface)
            text = q->tr("Line Annotation surface · synchronized");
        else if (!image->sampledNormals)
            text = q->tr("No Lasagna normals · adjust rotation");
        else
            text = q->tr("Line Annotation surface · Lasagna");
        if (!warning.isEmpty())
            text += QStringLiteral("\n") + warning;
        text += q->tr(" · CT L%1").arg(image->level);
        if (!image->complete) {
            const int percent = image->validTexels ? int(100 * image->requestedLevelTexels / image->validTexels) : 100;
            text += image->errors ? q->tr(" · CT read error; toggle ribbon to retry") : q->tr(" · loading details %1%").arg(percent);
        }
        status->setText(text);
    }
    void request()
    {
        stop();
        if (!button->isChecked() || suspended)
            return;
        if (!fiber() || !state->currentVolume()) {
            status->setText(q->tr("Select a displayed fiber."));
            return;
        }
        observeCache();
        requestedLevel = levelForView();
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
        if (image && image->widthVoxels == width->value() && image->rotationDegrees == rotation->value() && image->level == requestedLevel) {
            present();
            if (image->complete)
                return;
        }
        if (image && image->key == key && image->widthVoxels == width->value() && image->rotationDegrees == rotation->value())
            present();  // Retain the previous zoom's image while finer data arrive.
        else
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
        const auto source = automated->sourceDescriptor();
        const auto cachedFrames = frames;
        const auto surfaceFactory = canonicalSurface;
        std::optional<LineAnnotationController::FiberMapSnapshot> annotationSource;
        if (!cachedFrames && clip && selected.source == Source::Annotation)
            annotationSource = annotations->fiberMapSnapshot(false, selected.id);
        const auto previousWarning = warning;
        const auto centerReference = annotations->displayCenterReference();
        const int w = width->value(), angle = rotation->value(), level = requestedLevel;
        chunksReady->store(false);
        opendata::LasagnaResolutionSnapshot normalSource;
        if (const auto pkg = state->vpkg())
            normalSource =
                {pkg->path(), pkg->lasagnaDatasetEntries(), pkg->selectedLasagnaDataset(), pkg->volumeTags(state->currentVolumeId()), volume->shape()};
        watcher.setFuture(QtConcurrent::run([=] {
            Result result;
            try {
                result.frames = cachedFrames;
                result.warning = previousWarning;
                if (!result.frames && surfaceFactory)
                    result.frames = std::make_shared<ribbon::Frames>(surfaceFactory(token.get()));
                if (!result.frames) {
                    Fiber complete = selected;
                    if (clip && selected.source == Source::Automated && source) {
                        const auto shape = volume->shapeXyz();
                        AutomatedSource
                            fullSource{source->path, source->uuid, source->nativeToViewer, source->nativeVoxelMm, Bounds{{0, 0, 0}, {double(shape[0] - 1), double(shape[1] - 1), double(shape[2] - 1)}}};
                        QueryOptions options;
                        options.maximumFibers = 1;
                        options.selectedKey = selected.key;
                        const auto full = loadAutomated(fullSource, options, token.get());
                        if (full.fibers.empty() || full.fibers.front().key != selected.key)
                            throw std::runtime_error("Cannot load the complete fiber for its Line Annotation surface");
                        complete = full.fibers.front();
                        vc::fibers::FiberCollection reader(source->path.toStdString());
                        const auto expected = reader.summary(int64_t(selected.id)).pointCount;
                        std::sort(complete.runs.begin(), complete.runs.end(), [](const auto& a, const auto& b) {
                            return a.firstPoint < b.firstPoint;
                        });
                        int64_t next = 0;
                        for (const auto& run : complete.runs) {
                            if (run.firstPoint != next || run.points.size() < 2)
                                throw std::runtime_error("Incomplete selected fiber geometry");
                            next += int64_t(run.points.size()) - 1;
                        }
                        if (next + 1 != expected)
                            throw std::runtime_error("Incomplete selected fiber geometry");
                    }
                    if (annotationSource) {
                        QueryOptions options;
                        options.maximumFibers = 1;
                        const auto full = loadAnnotations(*annotationSource, {}, options, token.get());
                        if (!full.complete || full.fibers.size() != 1 || full.fibers.front().key != selected.key)
                            throw std::runtime_error("Cannot load the complete annotation for its Line Annotation surface");
                        complete = full.fibers.front();
                    }
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
                    ribbon::ModelForPoints model;
                    if (sampler)
                        model = [sampler](const std::vector<cv::Vec3d>& points) {
                            return vc::lasagna::lineModelFromSampledPoints(points, sampler.get());
                        };
                    if (!result.warning.isEmpty())
                        throw std::runtime_error(result.warning.toStdString());
                    result.frames = std::make_shared<ribbon::Frames>(ribbon::makeFrames(complete, clip, {}, token.get(), model, centerReference));
                }
                result.image = ribbon::sampleProgressive(
                    *result.frames, *volume, w, (surfaceFactory || result.frames->sampledNormals) ? 0 : angle, level, token.get());
                result.image->key = selected.key;
            } catch (const std::exception& e) {
                result.error = QString::fromUtf8(e.what());
            }
            return result;
        }));
    }
};
Fiber3DRibbonControls::Fiber3DRibbonControls(
    CState* s, ViewerManager* v, LineAnnotationController* a, FiberCollectionController* f, Fiber3DCanvas* c, QToolButton* b, QWidget* parent)
    : QWidget(parent), d(std::make_unique<Impl>(this, s, v, a, f, c, b))
{
}
Fiber3DRibbonControls::~Fiber3DRibbonControls()
{
    if (d->cancelled)
        d->cancelled->store(true);
}
void Fiber3DRibbonControls::cancel()
{
    d->suspended = true;
    d->stop();
}
void Fiber3DRibbonControls::resume()
{
    d->suspended = false;
    d->request();
}
void Fiber3DRibbonControls::setSelection(std::shared_ptr<const Scene> scene, const QString& key, FramesFactory canonicalSurface)
{
    if (scene == d->scene && key == d->key)
        return;
    d->canonicalSurface = std::move(canonicalSurface);
    d->scene = std::move(scene);
    d->key = key;
    d->frames.reset();
    d->image.reset();
    d->warning.clear();
    d->button->setEnabled(d->button->isChecked() || (d->fiber() && d->state->currentVolume()));
    d->request();
    // A cleared/closed scene must not retain its volume's CT cache or listener.
    // Hiding a valid scene still keeps its cache for immediate resumption.
    if (!d->fiber())
        d->releaseCache();
}
}  // namespace vc3d::fibers3d
