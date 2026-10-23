#include "Fiber3DPanel.hpp"
#include "Fiber3DCanvas.hpp"
#include "Fiber3DData.hpp"
#include "CState.hpp"
#include "FiberCollectionController.hpp"
#include "VCSettings.hpp"
#include "ViewerManager.hpp"
#include "vc/core/types/Volume.hpp"

#include <QCheckBox>
#include <QColorDialog>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QFutureWatcher>
#include <QHideEvent>
#include <QLabel>
#include <QMenu>
#include <QPushButton>
#include <QScrollArea>
#include <QSettings>
#include <QShowEvent>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QTimer>
#include <QToolButton>
#include <QVBoxLayout>
#include <QWidgetAction>
#include <QtConcurrent/QtConcurrentRun>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>

namespace vc3d::fibers3d
{
namespace
{
struct LoadResult {
    std::shared_ptr<Scene> scene;
    QString error;
    uint64_t revision = 0;
};

void appendScene(Scene& out, Scene part)
{
    out.complete = out.complete && part.complete;
    if (!part.message.isEmpty()) {
        if (!out.message.isEmpty())
            out.message += QLatin1Char('\n');
        out.message += part.message;
    }
    if (std::isfinite(part.voxelSizeMm))
        out.voxelSizeMm = part.voxelSizeMm;
    out.fibers.insert(out.fibers.end(), std::make_move_iterator(part.fibers.begin()), std::make_move_iterator(part.fibers.end()));
}
}  // namespace

struct Fiber3DPanel::Impl {
    Fiber3DPanel* q;
    CState* state;
    ViewerManager* viewers;
    LineAnnotationController* annotations;
    FiberCollectionController* automated;
    Fiber3DCanvas* canvas;
    QLabel *status, *selection;
    QCheckBox *nativeSource, *afvSource, *h, *v, *unknown;
    QComboBox *region, *colors, *direction, *palette;
    QSpinBox *boxWidth, *limit;
    QDoubleSpinBox *minimum, *rangeMin, *rangeMax;
    QPushButton *customColor, *hColor, *vColor;
    QCheckBox* fixedRange;
    DisplaySettings display;
    QTimer debounce;
    QFutureWatcher<LoadResult> watcher;
    std::shared_ptr<std::atomic_bool> cancelled;
    std::shared_ptr<const Scene> scene;
    QString selectedKey;
    uint64_t revision = 0;
    bool pending = false;
    bool syncingFilters = false;
    bool fitted = false;

    Impl(Fiber3DPanel* panel, CState* s, ViewerManager* vm, LineAnnotationController* a, FiberCollectionController* f)
        : q(panel), state(s), viewers(vm), annotations(a), automated(f)
    {
        q->setObjectName(QStringLiteral("fibers3DPanel"));
        q->setMinimumSize(280, 240);
        auto* layout = new QVBoxLayout(q);
        layout->setContentsMargins(4, 4, 4, 4);
        layout->setSpacing(4);
        auto* bar = new QHBoxLayout;
        auto button = [&](const QString& text, const QString& name) {
            auto* b = new QToolButton(q);
            b->setText(text);
            b->setObjectName(name);
            bar->addWidget(b);
            return b;
        };
        auto* fit = button(q->tr("Fit"), QStringLiteral("fibers3DFit"));
        auto* fitSelected = button(q->tr("Fit selected"), QStringLiteral("fibers3DFitSelected"));
        bar->addStretch();
        auto* detach = button(q->tr("Detach"), QStringLiteral("fibers3DDetach"));
        auto* gear = button(QString::fromUtf8("⚙"), QStringLiteral("fibers3DSettings"));
        gear->setToolTip(q->tr("3D display settings"));
        gear->setAccessibleName(q->tr("3D display settings"));
        layout->addLayout(bar);
        canvas = new Fiber3DCanvas(q);
        canvas->setObjectName(QStringLiteral("fibers3DCanvas"));
        layout->addWidget(canvas, 1);
        selection = new QLabel(q);
        selection->setWordWrap(true);
        selection->setTextFormat(Qt::PlainText);
        selection->setObjectName(QStringLiteral("fibers3DSelection"));
        layout->addWidget(selection);
        status = new QLabel(q);
        status->setWordWrap(true);
        status->setTextFormat(Qt::PlainText);
        status->setObjectName(QStringLiteral("fibers3DStatus"));
        layout->addWidget(status);
        canvas->setToolTip(
            q->tr("Drag to orbit · right/middle drag to pan · wheel to zoom\nClick to select · right-click for CT navigation"));
        QObject::connect(fit, &QToolButton::clicked, canvas, &Fiber3DCanvas::fitAll);
        QObject::connect(fitSelected, &QToolButton::clicked, canvas, &Fiber3DCanvas::fitSelected);
        QObject::connect(detach, &QToolButton::clicked, q, &Fiber3DPanel::detachRequested);
        createSettings(gear);

        debounce.setSingleShot(true);
        debounce.setInterval(100);
        QObject::connect(&debounce, &QTimer::timeout, q, [this] { startLoad(); });
        QObject::connect(&watcher, &QFutureWatcher<LoadResult>::finished, q, [this] {
            const auto result = watcher.result();
            if (result.revision == revision && q->isVisible()) {
                if (!result.error.isEmpty()) {
                    canvas->clear();
                    scene.reset();
                    status->setText(q->tr("Cannot display fibers: %1").arg(result.error));
                    updateSelection();
                } else if (result.scene) {
                    scene = result.scene;
                    canvas->setScene(scene);
                    canvas->setSelectedKey(selectedKey);
                    if (!fitted && !scene->fibers.empty()) {
                        canvas->fitAll();
                        fitted = true;
                    }
                    const auto text =
                        q->tr("%1 fibers displayed%2").arg(scene->fibers.size()).arg(scene->complete ? QString{} : q->tr(" · partial view"));
                    status->setText(text + (scene->message.isEmpty() ? QString{} : QLatin1Char('\n') + scene->message));
                    updateSelection();
                }
            }
            if (pending && q->isVisible())
                debounce.start();
        });
        QObject::connect(canvas, &Fiber3DCanvas::fiberPicked, q, [this](const QString& key, qint64, double, double, double) {
            select(key, true);
        });
        QObject::connect(canvas, &Fiber3DCanvas::fiberContextRequested, q, [this](const QString& key, qint64 index, double x, double y, double z, QPoint p) {
            select(key, true);
            contextMenu(p, key, index, {x, y, z});
        });
        QObject::connect(canvas, &Fiber3DCanvas::emptyContextRequested, q, [this](QPoint p) { contextMenu(p, {}, 0, {}); });
        QObject::connect(annotations, &LineAnnotationController::fibersChanged, q, [this] { request(); });
        QObject::connect(automated, &FiberCollectionController::collectionChanged, q, [this] {
            afvSource->setEnabled(automated->sourceDescriptor().has_value());
            if (!selectedKey.startsWith(QStringLiteral("annotation:")))
                selectedKey.clear();
            fitted = false;
            request(true);
        });
        QObject::connect(automated, &FiberCollectionController::fiberFiltersChanged, q, [this] {
            if (syncingFilters)
                return;
            const QSignalBlocker b1(minimum), b2(h), b3(v);
            minimum->setValue(automated->minimumLengthMm());
            h->setChecked(automated->showHorizontal());
            v->setChecked(automated->showVertical());
            request();
        });
        QObject::connect(automated, &FiberCollectionController::fiberSelectionChanged, q, [this](qint64 id) {
            if (const auto source = automated->sourceDescriptor(); source && id > 0) {
                select(source->uuid + QLatin1Char(':') + QString::number(id), false);
            }
        });
        auto reset = [this] {
            selectedKey.clear();
            fitted = false;
            request(true);
        };
        QObject::connect(state, &CState::volumeChanged, q, reset);
        QObject::connect(state, &CState::vpkgChanged, q, reset);
        QObject::connect(state, &CState::volumeClosing, q, reset);
        QObject::connect(state, &CState::poiChanged, q, [this](const std::string& name, POI*) {
            if (name == "focus" && region->currentIndex() == 0)
                request();
        });
        afvSource->setEnabled(automated->sourceDescriptor().has_value());
        canvas->setDisplaySettings(display);
        updateSelection();
    }

    void createSettings(QToolButton* gear)
    {
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        settings.beginGroup(QStringLiteral("fibers3d"));
        auto* menu = new QMenu(gear);
        auto* scroll = new QScrollArea(menu);
        scroll->setWidgetResizable(true);
        scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        scroll->setFixedSize(360, 560);
        auto* contents = new QWidget(scroll);
        auto* form = new QFormLayout(contents);
        form->setRowWrapPolicy(QFormLayout::WrapLongRows);
        form->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
        scroll->setWidget(contents);
        auto* action = new QWidgetAction(menu);
        action->setDefaultWidget(scroll);
        menu->addAction(action);
        gear->setMenu(menu);
        gear->setPopupMode(QToolButton::InstantPopup);
        auto check = [&](const char* key, const QString& label, bool value) {
            auto* w = new QCheckBox(contents);
            w->setObjectName(QString::fromLatin1(key));
            w->setChecked(settings.value(QString::fromLatin1(key), value).toBool());
            form->addRow(label, w);
            return w;
        };
        auto combo = [&](const char* key, const QString& label, const QStringList& choices, int defaultIndex = 0) {
            auto* w = new QComboBox(contents);
            w->setObjectName(QString::fromLatin1(key));
            w->addItems(choices);
            w->setCurrentIndex(std::clamp(settings.value(QString::fromLatin1(key), defaultIndex).toInt(), 0, int(choices.size()) - 1));
            form->addRow(label, w);
            return w;
        };
        auto real = [&](const char* key, const QString& label, double lo, double hi, double value, int decimals) {
            auto* w = new QDoubleSpinBox(contents);
            w->setObjectName(QString::fromLatin1(key));
            w->setRange(lo, hi);
            w->setDecimals(decimals);
            w->setValue(settings.value(QString::fromLatin1(key), value).toDouble());
            form->addRow(label, w);
            return w;
        };
        auto integer = [&](const char* key, const QString& label, int lo, int hi, int value) {
            auto* w = new QSpinBox(contents);
            w->setObjectName(QString::fromLatin1(key));
            w->setRange(lo, hi);
            w->setValue(settings.value(QString::fromLatin1(key), value).toInt());
            form->addRow(label, w);
            return w;
        };
        nativeSource = check("fibers3DNative", q->tr("Project annotations"), true);
        afvSource = check("fibers3DAutomated", q->tr("Automated collection"), true);
        h = check("fibers3DH", q->tr("Horizontal"), automated->showHorizontal());
        v = check("fibers3DV", q->tr("Vertical"), automated->showVertical());
        unknown = check("fibers3DUnknown", q->tr("Unclassified"), true);
        minimum = real("fibers3DMinLength", q->tr("Minimum length (mm)"), 0, 10000, automated->minimumLengthMm(), 2);
        // The shared collection controls own these filters; never restore a
        // second, conflicting filter state from the 3D preferences.
        h->setChecked(automated->showHorizontal());
        v->setChecked(automated->showVertical());
        minimum->setValue(automated->minimumLengthMm());
        region = combo("fibers3DRegion", q->tr("Region"), {q->tr("Around CT focus"), q->tr("Whole volume")}, 1);
        boxWidth = integer("fibers3DBoxWidth", q->tr("Box width (CT voxels)"), 16, 100000, 1024);
        limit = integer("fibers3DLimit", q->tr("Maximum fibers per source"), 1, 50000, 10000);
        auto* budget = new QLabel(q->tr("Geometry is limited to 16 MiB per source. Narrow the region if the view is partial."), contents);
        budget->setWordWrap(true);
        form->addRow(budget);
        colors = combo("fibers3DColorMode", q->tr("Color"), {q->tr("Custom"), q->tr("H / V"), q->tr("Direction"), q->tr("Length")}, 1);
        auto colorButton = [&](const char* key, const QString& label, QColor& color) {
            const QColor stored(settings.value(QString::fromLatin1(key), color.name()).toString());
            if (stored.isValid())
                color = stored;
            auto* b = new QPushButton(color.name(), contents);
            b->setObjectName(QString::fromLatin1(key));
            form->addRow(label, b);
            QObject::connect(b, &QPushButton::clicked, q, [this, b, &color] {
                const auto chosen = QColorDialog::getColor(color, q, q->tr("Fiber color"));
                if (chosen.isValid()) {
                    color = chosen;
                    b->setText(chosen.name());
                    applyDisplay();
                }
            });
            return b;
        };
        customColor = colorButton("fibers3DCustomColor", q->tr("Custom color"), display.custom);
        hColor = colorButton("fibers3DHColor", q->tr("H color"), display.horizontal);
        vColor = colorButton("fibers3DVColor", q->tr("V color"), display.vertical);
        direction =
            combo("fibers3DDirection", q->tr("Direction"), {q->tr("XYZ orientation"), q->tr("Angle to X"), q->tr("Angle to Y"), q->tr("Angle to Z")});
        palette =
            combo("fibers3DPalette", q->tr("Palette"), {QStringLiteral("Viridis"), QStringLiteral("Magma"), QStringLiteral("Cividis"), QStringLiteral("Turbo")});
        fixedRange = check("fibers3DFixedRange", q->tr("Fixed length range"), false);
        rangeMin = real("fibers3DRangeMin", q->tr("Length minimum (mm)"), 0, 99999.99, 0, 2);
        rangeMax = real("fibers3DRangeMax", q->tr("Length maximum (mm)"), 0.01, 100000, 10, 2);
        auto* width = real("fibers3DLineWidth", q->tr("Line width (px)"), 0.5, 10, 2, 1);
        auto* opacity = real("fibers3DOpacity", q->tr("Unselected opacity"), 0.05, 1, 1, 2);
        auto* gaps = check("fibers3DGaps", q->tr("Dash gap spans"), true);
        auto* annotationsVisible = check("fibers3DAnnotations", q->tr("Show annotation markers / tags"), false);
        auto* selectedOnly = check("fibers3DAnnotationsSelected", q->tr("Annotations on selection only"), true);
        auto* perspective = check("fibers3DPerspective", q->tr("Perspective"), false);
        auto* bounds = check("fibers3DBounds", q->tr("Show region bounds"), true);
        auto changedDisplay = [=, this] {
            display.lineWidth = float(width->value());
            display.unselectedOpacity = float(opacity->value());
            display.showGaps = gaps->isChecked();
            display.showAnnotations = annotationsVisible->isChecked();
            display.annotationsSelectedOnly = selectedOnly->isChecked();
            display.perspective = perspective->isChecked();
            display.showBounds = bounds->isChecked();
            applyDisplay();
        };
        for (auto* w : {nativeSource, afvSource, unknown})
            QObject::connect(w, &QCheckBox::toggled, q, [this] {
                request();
                saveSettings();
            });
        for (auto* w : {h, v})
            QObject::connect(w, &QCheckBox::toggled, q, [this] { sharedFilters(); });
        QObject::connect(minimum, &QDoubleSpinBox::valueChanged, q, [this] { sharedFilters(); });
        QObject::connect(region, &QComboBox::currentIndexChanged, q, [this] {
            boxWidth->setEnabled(region->currentIndex() == 0);
            request();
            saveSettings();
        });
        for (auto* w : {boxWidth, limit})
            QObject::connect(w, &QSpinBox::valueChanged, q, [this] {
                request();
                saveSettings();
            });
        for (auto* w : {colors, direction, palette})
            QObject::connect(w, &QComboBox::currentIndexChanged, q, changedDisplay);
        for (auto* w : {width, opacity, rangeMin, rangeMax})
            QObject::connect(w, &QDoubleSpinBox::valueChanged, q, changedDisplay);
        for (auto* w : {fixedRange, gaps, annotationsVisible, selectedOnly, perspective, bounds})
            QObject::connect(w, &QCheckBox::toggled, q, changedDisplay);
        boxWidth->setEnabled(region->currentIndex() == 0);
        changedDisplay();
    }

    void saveSettings()
    {
        QSettings s(vc3d::settingsFilePath(), QSettings::IniFormat);
        s.beginGroup(QStringLiteral("fibers3d"));
        // All editable settings have stable object names; the popup is a child.
        for (auto* w : q->findChildren<QCheckBox*>())
            s.setValue(w->objectName(), w->isChecked());
        for (auto* w : q->findChildren<QComboBox*>())
            s.setValue(w->objectName(), w->currentIndex());
        for (auto* w : q->findChildren<QSpinBox*>())
            s.setValue(w->objectName(), w->value());
        for (auto* w : q->findChildren<QDoubleSpinBox*>())
            s.setValue(w->objectName(), w->value());
        s.setValue(QStringLiteral("fibers3DCustomColor"), display.custom.name());
        s.setValue(QStringLiteral("fibers3DHColor"), display.horizontal.name());
        s.setValue(QStringLiteral("fibers3DVColor"), display.vertical.name());
    }

    void applyDisplay()
    {
        display.colorMode = ColorMode(colors->currentIndex());
        display.directionMode = DirectionMode(direction->currentIndex());
        display.palette = Palette(palette->currentIndex());
        display.fixedLengthRange = fixedRange->isChecked();
        display.lengthMinMm = rangeMin->value();
        if (rangeMax->value() <= display.lengthMinMm) {
            const QSignalBlocker blocker(rangeMax);
            rangeMax->setValue(display.lengthMinMm + 0.01);
        }
        display.lengthMaxMm = rangeMax->value();
        const bool length = display.colorMode == ColorMode::Length;
        customColor->setEnabled(display.colorMode == ColorMode::Custom);
        hColor->setEnabled(display.colorMode == ColorMode::Family);
        vColor->setEnabled(display.colorMode == ColorMode::Family);
        direction->setEnabled(display.colorMode == ColorMode::Direction);
        palette->setEnabled(length || (display.colorMode == ColorMode::Direction && display.directionMode != DirectionMode::XYZ));
        fixedRange->setEnabled(length);
        rangeMin->setEnabled(length && display.fixedLengthRange);
        rangeMax->setEnabled(length && display.fixedLengthRange);
        canvas->setDisplaySettings(display);
        updateSelection();
        saveSettings();
    }

    void sharedFilters()
    {
        syncingFilters = true;
        automated->setFiberFilters(minimum->value(), h->isChecked(), v->isChecked());
        syncingFilters = false;
        request();
    }

    void request(bool clear = false)
    {
        ++revision;
        pending = true;
        if (cancelled)
            cancelled->store(true, std::memory_order_relaxed);
        if (clear) {
            scene.reset();
            canvas->clear();
            updateSelection();
        }
        if (q->isVisible())
            debounce.start();
    }

    void startLoad()
    {
        if (!q->isVisible() || watcher.isRunning())
            return;
        pending = false;
        const auto volume = state->currentVolume();
        if (!volume) {
            canvas->clear();
            status->setText(q->tr("Open a volume to display fibers."));
            return;
        }
        QueryOptions options;
        options.minLengthMm = minimum->value();
        options.horizontal = h->isChecked();
        options.vertical = v->isChecked();
        options.unclassified = unknown->isChecked();
        options.maximumFibers = size_t(limit->value());
        options.selectedKey = selectedKey;
        const auto shape = volume->shapeXyz();
        Bounds volumeBounds;
        for (int i = 0; i < 3; ++i)
            volumeBounds.upper[i] = std::max(0, shape[i] - 1);
        if (region->currentIndex() == 0) {
            auto* focus = state->poi("focus");
            Bounds b;
            for (int i = 0; i < 3; ++i) {
                const double center = focus ? double(focus->p[i]) : volumeBounds.upper[i] / 2;
                b.lower[i] = std::max(0.0, center - boxWidth->value() / 2.0);
                b.upper[i] = std::min(volumeBounds.upper[i], center + boxWidth->value() / 2.0);
            }
            options.bounds = b;
        }
        std::optional<AutomatedSource> source;
        if (afvSource->isChecked())
            if (const auto afv = automated->sourceDescriptor())
                source = AutomatedSource{afv->path, afv->uuid, afv->nativeToViewer, afv->nativeVoxelMm, volumeBounds};
        // Copy live annotation state only on the GUI thread. The worker owns
        // this snapshot and its own read-only AFV connection, never a controller.
        std::optional<LineAnnotationController::FiberMapSnapshot> snapshot;
        std::vector<LineAnnotationController::FiberSummary> summaries;
        if (nativeSource->isChecked()) {
            snapshot = annotations->fiberMapSnapshot(false);
            summaries = annotations->fiberSummaries();
        }
        cancelled = std::make_shared<std::atomic_bool>(false);
        status->setText(q->tr("Loading fibers…"));
        const auto token = cancelled;
        const auto generation = revision;
        watcher.setFuture(QtConcurrent::run([options, source, snapshot = std::move(snapshot), summaries = std::move(summaries), token, generation]() {
            LoadResult result;
            result.revision = generation;
            try {
                result.scene = std::make_shared<Scene>();
                result.scene->clipBounds = options.bounds;
                if (snapshot && !snapshot->fibers.empty())
                    appendScene(*result.scene, loadAnnotations(*snapshot, summaries, options, token.get()));
                if (source)
                    appendScene(*result.scene, loadAutomated(*source, options, token.get()));
                if (token->load(std::memory_order_relaxed))
                    result.scene.reset();
            } catch (const std::exception& e) {
                result.scene.reset();
                result.error = QString::fromUtf8(e.what());
            }
            return result;
        }));
    }

    const Fiber* find(const QString& key) const
    {
        if (scene)
            for (const auto& fiber : scene->fibers)
                if (fiber.key == key)
                    return &fiber;
        return nullptr;
    }

    void select(const QString& key, bool notify)
    {
        if (selectedKey == key)
            return;
        selectedKey = key;
        canvas->setSelectedKey(key);
        updateSelection();
        // A selected fibre has priority within an otherwise limited region.
        if (!key.isEmpty() && !find(key))
            request();
        if (notify)
            if (const auto* fiber = find(key)) {
                if (fiber->source == Source::Annotation)
                    emit q->nativeFiberSelected(fiber->id);
                else
                    automated->selectForInspection(qint64(fiber->id));
            }
    }

    void updateSelection()
    {
        const auto* fiber = find(selectedKey);
        if (!fiber) {
            selection->setText(selectedKey.isEmpty() ? q->tr("Click a fiber to select it.") : q->tr("Selected fiber is outside the displayed region or filters."));
            return;
        }
        QString text =
            q->tr("%1 · %2 · %3")
                .arg(fiber->label, QString(QChar::fromLatin1(fiber->family)), std::isfinite(fiber->lengthMm) ? q->tr("%1 mm").arg(fiber->lengthMm, 0, 'f', 2) : q->tr("scale unavailable"));
        if (display.showAnnotations && !fiber->tags.empty())
            text += QLatin1Char('\n') + fiber->tags.join(QStringLiteral(", "));
        selection->setText(text);
    }

    void contextMenu(QPoint position, const QString& key, qint64 index, Point point)
    {
        QMenu menu(q);
        const auto menuRevision = revision;
        menu.addAction(q->tr("Fit displayed fibers"), canvas, &Fiber3DCanvas::fitAll);
        if (const auto* fiber = find(key)) {
            // Copy identifiers: nested menus can process scene-reload signals.
            const auto id = fiber->id;
            const auto file = fiber->fileName;
            const auto source = fiber->source;
            menu.addAction(q->tr("Fit selected fiber"), canvas, &Fiber3DCanvas::fitSelected);
            menu.addAction(q->tr("Center CT views here"), q, [this, point, menuRevision] {
                if (menuRevision != revision)
                    return;
                viewers->centerFocusAt({float(point[0]), float(point[1]), float(point[2])}, {0, 0, 1}, "fibers-3d");
            });
            menu.addAction(q->tr("Open in Line Annotation"), q, [this, id, file, source, index, menuRevision] {
                if (menuRevision != revision)
                    return;
                if (source == Source::Automated)
                    automated->selectForInspection(qint64(id), true);
                else if (annotations->hasLoadedFiber(id, file.toStdString()))
                    annotations->openFiberAtLinePointIndex(id, int(std::clamp<qint64>(index, 0, std::numeric_limits<int>::max())));
            });
        }
        menu.exec(position);
    }
};

Fiber3DPanel::Fiber3DPanel(CState* s, ViewerManager* vm, LineAnnotationController* a, FiberCollectionController* f, QWidget* parent)
    : QWidget(parent), d(std::make_unique<Impl>(this, s, vm, a, f))
{
}
Fiber3DPanel::~Fiber3DPanel()
{
    // No wait on the GUI thread: all worker inputs are owned value snapshots.
    if (d->cancelled)
        d->cancelled->store(true, std::memory_order_relaxed);
}
void Fiber3DPanel::showEvent(QShowEvent* e)
{
    QWidget::showEvent(e);
    d->request();
}
void Fiber3DPanel::hideEvent(QHideEvent* e)
{
    QWidget::hideEvent(e);
    ++d->revision;
    d->debounce.stop();
    d->pending = true;
    if (d->cancelled)
        d->cancelled->store(true, std::memory_order_relaxed);
}
void Fiber3DPanel::selectNativeFiber(uint64_t id)
{
    if (!id) {
        if (d->selectedKey.startsWith(QStringLiteral("annotation:")))
            d->select({}, false);
        return;
    }
    d->select(QStringLiteral("annotation:") + QString::number(id), false);
}
}  // namespace vc3d::fibers3d
