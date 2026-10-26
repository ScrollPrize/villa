#include "Fiber3DPanel.hpp"
#include "Fiber3DCanvas.hpp"
#include "Fiber3DData.hpp"
#include "Fiber3DRibbonControls.hpp"
#include "LineAnnotationGeneratedViews.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "Fiber3DSliceControls.hpp"
#include "Fiber3DSurfaceControls.hpp"
#include "CState.hpp"
#include "FiberCollectionController.hpp"
#include "VCSettings.hpp"
#include "ViewerManager.hpp"
#include "vc/core/types/Volume.hpp"

#include <QCheckBox>
#include <QColorDialog>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QDialog>
#include <QDialogButtonBox>
#include <QIcon>
#include <QPixmap>
#include <QScreen>
#include <QFutureWatcher>
#include <QHideEvent>
#include <QLabel>
#include <QMenu>
#include <QPushButton>
#include <QPromise>
#include <QSettings>
#include <QShowEvent>
#include <QSignalBlocker>
#include <QSlider>
#include <QSpinBox>
#include <QTabWidget>
#include <QTimer>
#include <QToolButton>
#include <QVBoxLayout>
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
    bool preview = false;
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
    Fiber3DRibbonControls* ribbonControls;
    Fiber3DSliceControls* sliceControls;
    Fiber3DSurfaceControls* surfaceControls;
    QLabel *status, *selection;
    QCheckBox *nativeSource, *afvSource, *h, *v, *unknown;
    QComboBox *region, *colors, *direction, *palette, *memoryBudget;
    QSpinBox *boxWidth, *limit;
    QDoubleSpinBox *minimum, *rangeMin, *rangeMax;
    QPushButton *customColor, *hColor, *vColor;
    QCheckBox* fixedRange;
    QWidget *customRow, *familyRow, *directionRow, *paletteRow, *fixedRangeRow, *lengthRangeRow, *boxWidthRow, *annotationOptions;
    QDialog* settingsDialog;
    QTabWidget* settingsTabs;
    std::vector<QWidget*> preferenceFields;
    DisplaySettings display;
    QTimer debounce;
    QFutureWatcher<LoadResult> watcher;
    std::shared_ptr<std::atomic_bool> cancelled;
    std::shared_ptr<const Scene> scene;
    QString selectedKey;
    uint64_t revision = 0;
    bool loading = false, pending = false;
    bool preview = false;
    bool syncingFilters = false;
    bool fitted = false;
    bool lineAnnotation = false;
    std::vector<cv::Vec3f> linePoints;
    std::vector<double> lineArcs;
    double linePosition = 0;
    Fiber3DRibbonControls::FramesFactory canonicalSurface;

    Impl(Fiber3DPanel* panel, CState* s, ViewerManager* vm, LineAnnotationController* a, FiberCollectionController* f, bool inLineAnnotation)
        : q(panel), state(s), viewers(vm), annotations(a), automated(f), lineAnnotation(inLineAnnotation)
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
        auto* ribbon = button(q->tr("CT ribbon"), QStringLiteral("fibers3DRibbon"));
        auto* slices = button(q->tr("CT slices"), QStringLiteral("fibers3DSlices"));
        bar->addStretch();
        auto* detach = button(q->tr("Detach"), QStringLiteral("fibers3DDetach"));
        auto* settingsButton = button(q->tr("Settings"), QStringLiteral("fibers3DSettings"));
        settingsButton->setToolTip(q->tr("3D display settings"));
        settingsButton->setAccessibleName(q->tr("3D display settings"));
        layout->addLayout(bar);
        canvas = new Fiber3DCanvas(q);
        canvas->setObjectName(QStringLiteral("fibers3DCanvas"));
        ribbonControls = new Fiber3DRibbonControls(state, viewers, annotations, automated, canvas, ribbon, q);
        layout->addWidget(ribbonControls);
        sliceControls = new Fiber3DSliceControls(state, viewers, canvas, slices, q);
        layout->addWidget(sliceControls);
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
        auto* annotationStatus = new QLabel(q);
        annotationStatus->setObjectName(QStringLiteral("fibers3DAnnotationStatus"));
        annotationStatus->setTextFormat(Qt::PlainText);
        annotationStatus->setWordWrap(true);
        annotationStatus->hide();
        layout->addWidget(annotationStatus);
        QObject::connect(automated, &FiberCollectionController::annotationStatusChanged, annotationStatus, [annotationStatus](const QString& text) {
            annotationStatus->setText(text);
            annotationStatus->setVisible(!text.isEmpty());
        });
        canvas->setToolTip(
            q->tr("Drag to orbit · right/middle drag to pan · wheel to zoom\nClick to select · Escape or click empty space to clear selection\nRight-click for CT navigation"));
        QObject::connect(canvas, &Fiber3DCanvas::visibilityChanged, q, [this] { updateStatus(); });
        QObject::connect(fit, &QToolButton::clicked, canvas, &Fiber3DCanvas::fitAll);
        QObject::connect(fitSelected, &QToolButton::clicked, q, [this] {
            if (lineAnnotation)
                canvas->fitAll();
            else
                canvas->fitSelected();
        });
        QObject::connect(detach, &QToolButton::clicked, q, &Fiber3DPanel::detachRequested);
        createSettings(settingsButton);
        if (lineAnnotation) {
            fitSelected->hide();
            for (int i = 0; i < settingsTabs->count(); ++i) {
                const auto name = settingsTabs->widget(i)->objectName();
                if (name == QStringLiteral("fibers3DSettingsFibers") || name == QStringLiteral("fibers3DSettingsMemory"))
                    settingsTabs->setTabVisible(i, false);
            }
            QObject::connect(canvas, &Fiber3DCanvas::fiberPositionPicked, q, [this](const QString&, double position) {
                emit q->linePositionPicked(position);
            });
        }

        debounce.setSingleShot(true);
        debounce.setInterval(100);
        QObject::connect(&debounce, &QTimer::timeout, q, [this] { startLoad(); });
        QObject::connect(&watcher, &QFutureWatcher<LoadResult>::resultReadyAt, q, [this](int index) {
            const auto result = watcher.resultAt(index);
            if (result.revision == revision && q->isVisible()) {
                if (!result.error.isEmpty()) {
                    canvas->clear();
                    scene.reset();
                    status->setText(q->tr("Cannot display fibers: %1").arg(result.error));
                    updateSelection();
                } else if (result.scene) {
                    preview = result.preview;
                    scene = result.scene;
                    canvas->setScene(scene);
                    syncCursor();
                    canvas->setSelectedKey(selectedKey);
                    if (!fitted && !scene->fibers.empty()) {
                        canvas->fitAll();
                        fitted = true;
                    }
                    updateStatus();
                    updateSelection();
                }
            }
        });
        QObject::connect(&watcher, &QFutureWatcher<LoadResult>::finished, q, [this] {
            loading = false;
            if (pending && q->isVisible())
                debounce.start();
        });
        QObject::connect(canvas, &Fiber3DCanvas::fiberPicked, q, [this](const QString& key, qint64, double, double, double) {
            if (!lineAnnotation)
                select(key, true);
        });
        QObject::connect(canvas, &Fiber3DCanvas::clearSelectionRequested, q, [this] { clearSelection(); });
        QObject::connect(canvas, &Fiber3DCanvas::fiberContextRequested, q, [this](const QString& key, qint64 index, double x, double y, double z, QPoint p) {
            if (!lineAnnotation)
                select(key, true);
            contextMenu(p, key, index, {x, y, z});
        });
        QObject::connect(canvas, &Fiber3DCanvas::emptyContextRequested, q, [this](QPoint p) { contextMenu(p, {}, 0, {}); });
        QObject::connect(annotations, &LineAnnotationController::fibersChanged, q, [this] {
            // Inspecting a read-only AFV fiber also changes the annotation
            // controller. It must not restart an unrelated automated-only load.
            if (nativeSource->isChecked())
                request();
        });
        QObject::connect(automated, &FiberCollectionController::collectionChanged, q, [this] {
            if (lineAnnotation)
                return;
            afvSource->setEnabled(automated->sourceDescriptor().has_value());
            if (!selectedKey.startsWith(QStringLiteral("annotation:")))
                selectedKey.clear();
            fitted = false;
            request(true);
        });
        QObject::connect(automated, &FiberCollectionController::fiberFiltersChanged, q, [this] {
            if (lineAnnotation || syncingFilters)
                return;
            const QSignalBlocker b1(minimum), b2(h), b3(v);
            minimum->setValue(automated->minimumLengthMm());
            h->setChecked(automated->showHorizontal());
            v->setChecked(automated->showVertical());
            request();
        });
        QObject::connect(automated, &FiberCollectionController::fiberSelectionChanged, q, [this](qint64 id) {
            if (lineAnnotation)
                return;
            if (const auto source = automated->sourceDescriptor(); source && id > 0) {
                select(source->uuid + QLatin1Char(':') + QString::number(id), false);
            } else if (id == 0 && !selectedKey.startsWith(QStringLiteral("annotation:"))) {
                select({}, false);
            }
        });
        auto reset = [this] {
            if (lineAnnotation) {
                q->clearLineAnnotation();
                return;
            }
            selectedKey.clear();
            fitted = false;
            request(true);
        };
        QObject::connect(state, &CState::volumeChanged, q, [this, reset, previous = std::weak_ptr<Volume>(state->currentVolume())](const std::shared_ptr<Volume>& volume) mutable {
            // CState also emits on reselection; Line Annotation does not
            // regenerate its views when the active volume is unchanged.
            if (lineAnnotation && previous.lock() == volume)
                return;
            previous = volume;
            reset();
        });
        QObject::connect(state, &CState::vpkgChanged, q, reset);
        QObject::connect(state, &CState::volumeClosing, q, reset);
        QObject::connect(state, &CState::poiChanged, q, [this](const std::string& name, POI*) {
            if (name != "focus")
                return;
            syncCursor();
            if (region->currentIndex() == 0)
                request();
        });
        afvSource->setEnabled(automated->sourceDescriptor().has_value());
        canvas->setDisplaySettings(display);
        syncCursor();
        updateSelection();
    }

    void syncCursor()
    {
        if (lineAnnotation)
            return;
        const auto* focus = state->poi("focus");
        if (!state->currentVolume() || !focus) {
            canvas->setCursorPosition(std::nullopt);
            return;
        }
        // Like the slice crosshair, the focus is already in the current CT
        // grid. Copy its position; CState owns and may replace the POI.
        canvas->setCursorPosition(Point{double(focus->p[0]), double(focus->p[1]), double(focus->p[2])});
    }

    void createSettings(QToolButton* settingsButton)
    {
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        settings.beginGroup(lineAnnotation ? QStringLiteral("lineAnnotation/fibers3d") : QStringLiteral("fibers3d"));
        settingsDialog = new QDialog(q, Qt::Tool);
        settingsDialog->setObjectName(QStringLiteral("fibers3DSettingsDialog"));
        settingsDialog->setWindowTitle(q->tr("Fibers 3D · Settings"));
        settingsDialog->setModal(false);
        settingsDialog->resize(520, 460);
        auto* dialogLayout = new QVBoxLayout(settingsDialog);
        dialogLayout->setContentsMargins(12, 12, 12, 12);
        settingsTabs = new QTabWidget(settingsDialog);
        settingsTabs->setObjectName(QStringLiteral("fibers3DSettingsTabs"));
        settingsTabs->setUsesScrollButtons(false);
        // The application's dark theme gives all tabs the same background.
        // Keep the current settings page unmistakable in either theme.
        settingsTabs->setStyleSheet(
            QStringLiteral("QTabBar::tab:selected { background-color: palette(highlight); color: palette(highlighted-text); }"));
        dialogLayout->addWidget(settingsTabs);
        auto* closeButtons = new QDialogButtonBox(QDialogButtonBox::Close, settingsDialog);
        closeButtons->button(QDialogButtonBox::Close)->setAutoDefault(false);
        dialogLayout->addWidget(closeButtons);
        QObject::connect(closeButtons, &QDialogButtonBox::rejected, settingsDialog, &QDialog::hide);
        QObject::connect(settingsButton, &QToolButton::clicked, q, [this, settingsButton] {
            resizeSettings();
            if (!settingsDialog->isVisible()) {
                // Keep the last position when reopened, but recover a window
                // left on a disconnected monitor. Use the button's screen for
                // a new dialog, also when the 3D dock has been detached.
                const auto available = settingsButton->screen()->availableGeometry();
                if (!settingsDialog->property("positioned").toBool() || !available.contains(settingsDialog->frameGeometry())) {
                    auto position =
                        settingsButton->mapToGlobal(QPoint(settingsButton->width() - settingsDialog->width(), settingsButton->height()));
                    position.setX(std::clamp(position.x(), available.left(), std::max(available.left(), available.right() - settingsDialog->width())));
                    position.setY(
                        std::clamp(position.y(), available.top() + 30, std::max(available.top() + 30, available.bottom() - settingsDialog->height())));
                    settingsDialog->move(position);
                    settingsDialog->setProperty("positioned", true);
                }
                settingsDialog->show();
            }
            settingsDialog->raise();
            settingsDialog->activateWindow();
        });

        QWidget* settingsContents = nullptr;
        QVBoxLayout* body = nullptr;
        auto page = [&](const QString& title, const QString& name) {
            if (body)
                body->addStretch();
            settingsContents = new QWidget(settingsTabs);
            settingsContents->setObjectName(name);
            body = new QVBoxLayout(settingsContents);
            body->setContentsMargins(16, 16, 16, 16);
            body->setSpacing(10);
            settingsTabs->addTab(settingsContents, title);
        };

        auto section = [&](const QString& title) {
            if (body->count())
                body->addSpacing(5);
            auto* label = new QLabel(title, settingsContents);
            auto font = label->font();
            font.setBold(true);
            label->setFont(font);
            body->addWidget(label);
        };
        auto row = [&](const QString& title, QWidget* field, QVBoxLayout* destination = nullptr) {
            auto* container = new QWidget(settingsContents);
            auto* line = new QHBoxLayout(container);
            line->setContentsMargins(0, 0, 0, 0);
            line->setSpacing(10);
            auto* label = new QLabel(title, container);
            label->setMinimumWidth(128);
            label->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Preferred);
            label->setBuddy(field);
            line->addWidget(label);
            line->addWidget(field, 1);
            (destination ? destination : body)->addWidget(container);
            return container;
        };
        auto check = [&](const char* key, const QString& label, bool value) {
            auto* w = new QCheckBox(label, settingsContents);
            preferenceFields.push_back(w);
            w->setObjectName(QString::fromLatin1(key));
            w->setChecked(settings.value(QString::fromLatin1(key), value).toBool());
            return w;
        };
        auto combo = [&](const char* key, const QStringList& choices, int defaultIndex = 0) {
            auto* w = new QComboBox(settingsContents);
            preferenceFields.push_back(w);
            w->setObjectName(QString::fromLatin1(key));
            w->addItems(choices);
            w->setSizeAdjustPolicy(QComboBox::AdjustToMinimumContentsLengthWithIcon);
            w->setMinimumContentsLength(8);
            w->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
            w->setCurrentIndex(std::clamp(settings.value(QString::fromLatin1(key), defaultIndex).toInt(), 0, int(choices.size()) - 1));
            return w;
        };
        auto real = [&](const char* key, double lo, double hi, double value, int decimals, const QString& suffix = {}) {
            auto* w = new QDoubleSpinBox(settingsContents);
            preferenceFields.push_back(w);
            w->setObjectName(QString::fromLatin1(key));
            w->setRange(lo, hi);
            w->setDecimals(decimals);
            w->setSuffix(suffix);
            w->setValue(settings.value(QString::fromLatin1(key), value).toDouble());
            return w;
        };
        auto integer = [&](const char* key, int lo, int hi, int value) {
            auto* w = new QSpinBox(settingsContents);
            preferenceFields.push_back(w);
            w->setObjectName(QString::fromLatin1(key));
            w->setRange(lo, hi);
            w->setValue(settings.value(QString::fromLatin1(key), value).toInt());
            return w;
        };
        auto horizontal = [&](const QList<QWidget*>& widgets) {
            auto* container = new QWidget(settingsContents);
            auto* line = new QHBoxLayout(container);
            line->setContentsMargins(0, 0, 0, 0);
            line->setSpacing(8);
            for (auto* widget : widgets)
                line->addWidget(widget);
            return container;
        };
        auto colorButton = [&](const char* key, const QString& label, QColor& color) {
            const QColor stored(settings.value(QString::fromLatin1(key), color.name()).toString());
            if (stored.isValid())
                color = stored;
            auto* button = new QPushButton(label, settingsContents);
            button->setAutoDefault(false);
            button->setObjectName(QString::fromLatin1(key));
            const auto refresh = [button, &color] {
                QPixmap swatch(32, 32);
                swatch.setDevicePixelRatio(2);
                swatch.fill(color);
                button->setIcon(QIcon(swatch));
                button->setIconSize(QSize(16, 16));
                button->setToolTip(color.name());
            };
            refresh();
            QObject::connect(button, &QPushButton::clicked, q, [this, &color, refresh] {
                const auto chosen = QColorDialog::getColor(color, q, q->tr("Fiber color"));
                if (chosen.isValid()) {
                    color = chosen;
                    refresh();
                    applyDisplay();
                }
            });
            return button;
        };

        page(q->tr("Fibers"), QStringLiteral("fibers3DSettingsFibers"));
        section(q->tr("Sources"));
        nativeSource = check("fibers3DNative", q->tr("Project annotations"), true);
        afvSource = check("fibers3DAutomated", q->tr("Automated collection"), true);
        body->addWidget(nativeSource);
        body->addWidget(afvSource);
        section(q->tr("Filters"));
        h = check("fibers3DH", q->tr("H"), automated->showHorizontal());
        v = check("fibers3DV", q->tr("V"), automated->showVertical());
        unknown = check("fibers3DUnknown", q->tr("Unclassified"), true);
        h->setToolTip(q->tr("Horizontal fibers"));
        v->setToolTip(q->tr("Vertical fibers"));
        row(q->tr("Families"), horizontal({h, v, unknown}));
        minimum = real("fibers3DMinLength", 0, 10000, automated->minimumLengthMm(), 2, q->tr(" mm"));
        // The collection controls own these shared filters.
        h->setChecked(automated->showHorizontal());
        v->setChecked(automated->showVertical());
        minimum->setValue(automated->minimumLengthMm());
        row(q->tr("Minimum length"), minimum);
        region = combo("fibers3DRegion", {q->tr("Around CT focus"), q->tr("Whole volume")}, 1);
        row(q->tr("Region"), region);
        boxWidth = integer("fibers3DBoxWidth", 16, 100000, 1024);
        boxWidth->setSuffix(q->tr(" vx"));
        boxWidth->setToolTip(q->tr("Width in current CT volume voxels"));
        boxWidthRow = row(q->tr("Box width"), boxWidth);

        page(q->tr("Surface"), QStringLiteral("fibers3DSettingsSurface"));
        surfaceControls = new Fiber3DSurfaceControls(state, canvas, settingsContents);
        body->addWidget(surfaceControls);

        page(q->tr("Memory"), QStringLiteral("fibers3DSettingsMemory"));
        section(q->tr("Loading limits"));

        memoryBudget = new QComboBox(settingsContents);
        preferenceFields.push_back(memoryBudget);
        memoryBudget->setObjectName(QStringLiteral("fibers3DMemoryBudgetMiB"));
        for (int mib = 64; mib <= 32768; mib *= 2) {
            // Omit values that cannot be represented on this build's address space.
            if (uint64_t(mib) * 1024 * 1024 > std::numeric_limits<size_t>::max())
                break;
            memoryBudget->addItem(mib < 1024 ? q->tr("%1 MiB").arg(mib) : q->tr("%1 GiB").arg(mib / 1024), mib);
        }
        const int savedBudget = memoryBudget->findData(settings.value(memoryBudget->objectName(), 512).toInt());
        memoryBudget->setCurrentIndex(savedBudget >= 0 ? savedBudget : memoryBudget->findData(512));
        surfaceControls->setMemoryBudget(size_t(memoryBudget->currentData().toInt()) * 1024 * 1024);
        memoryBudget->setToolTip(
            q->tr("Maximum decoded fiber coordinates per source, and maximum surface mesh buffer size. Additional rendering buffers, snapshots and CT use more memory. This is a limit, not a preallocation."));
        row(q->tr("Memory budget"), memoryBudget);
        auto* memoryHelp =
            new QLabel(q->tr("Budget per fiber source and for the surface mesh. Rendering buffers and CT use additional memory. Memory is allocated only as needed."), settingsContents);
        memoryHelp->setWordWrap(true);
        body->addWidget(memoryHelp);
        limit = integer("fibers3DLimit", 0, 1000000, int(QueryOptions{}.maximumFibers));
        limit->setSpecialValueText(q->tr("No limit"));
        limit->setToolTip(
            q->tr("Maximum displayed fibers per source. No limit uses only the memory budget. The selected fiber has priority, then the longest fibers in the region."));
        row(q->tr("Fiber limit"), limit);
        auto* limitHelp =
            new QLabel(q->tr("Set to 0 for no count limit. The selected fiber loads first, then the longest fibers in the region."), settingsContents);
        limitHelp->setWordWrap(true);
        body->addWidget(limitHelp);

        page(q->tr("Appearance"), QStringLiteral("fibers3DSettingsAppearance"));
        colors = combo("fibers3DColorMode", {q->tr("Custom"), q->tr("H / V"), q->tr("Direction"), q->tr("Length")}, 1);
        row(q->tr("Color by"), colors);
        customColor = colorButton("fibers3DCustomColor", q->tr("Choose…"), display.custom);
        customRow = row(q->tr("Color"), customColor);
        hColor = colorButton("fibers3DHColor", q->tr("H"), display.horizontal);
        vColor = colorButton("fibers3DVColor", q->tr("V"), display.vertical);
        familyRow = row(q->tr("Colors"), horizontal({hColor, vColor}));
        direction = combo("fibers3DDirection", {q->tr("XYZ orientation"), q->tr("Angle to X"), q->tr("Angle to Y"), q->tr("Angle to Z")});
        directionRow = row(q->tr("Orientation"), direction);
        palette = combo("fibers3DPalette", {QStringLiteral("Viridis"), QStringLiteral("Magma"), QStringLiteral("Cividis"), QStringLiteral("Turbo")});
        paletteRow = row(q->tr("Palette"), palette);
        fixedRange = check("fibers3DFixedRange", q->tr("Fixed range"), false);
        fixedRangeRow = row(q->tr("Length scale"), fixedRange);
        rangeMin = real("fibers3DRangeMin", 0, 99999.99, 0, 2);
        rangeMax = real("fibers3DRangeMax", 0.01, 100000, 10, 2);
        rangeMin->setAccessibleName(q->tr("Minimum length in millimeters"));
        rangeMax->setAccessibleName(q->tr("Maximum length in millimeters"));
        lengthRangeRow = new QWidget(settingsContents);
        auto* rangeLayout = new QVBoxLayout(lengthRangeRow);
        rangeLayout->setContentsMargins(0, 0, 0, 0);
        rangeLayout->setSpacing(7);
        rangeMin->setSuffix(q->tr(" mm"));
        rangeMax->setSuffix(q->tr(" mm"));
        row(q->tr("Minimum"), rangeMin, rangeLayout);
        row(q->tr("Maximum"), rangeMax, rangeLayout);
        body->addWidget(lengthRangeRow);
        auto* width = real("fibers3DLineWidth", 0.5, 10, display.lineWidth, 1, q->tr(" px"));
        width->setSingleStep(0.5);
        row(q->tr("Line width"), width);

        section(q->tr("View"));
        auto* perspective = check("fibers3DPerspective", q->tr("Perspective"), false);
        auto* bounds = check("fibers3DBounds", q->tr("Show region bounds"), display.showBounds);
        auto* cursor = check("fibers3DCursor", q->tr("Show cursor"), true);
        body->addWidget(perspective);
        body->addWidget(horizontal({cursor, bounds}));
        auto* gaps = check("fibers3DGaps", q->tr("Show annotated gaps as dashed lines"), false);
        gaps->setToolTip(q->tr("Display annotated gaps and inferred connections as dashed lines."));
        body->addWidget(gaps);

        page(q->tr("Selection"), QStringLiteral("fibers3DSettingsSelection"));
        if (!lineAnnotation)
            section(q->tr("Nearby fibers"));
        auto* nearby = check("fibers3DNearbyOnly", q->tr("Only fibers near selection"), false);
        nearby->setToolTip(
            q->tr("Keep whole loaded fibers that pass within this distance of any part of the selected fiber. Without a selection, show all loaded fibers. This only filters the 3D view."));
        body->addWidget(nearby);
        auto* nearbyDistance = integer("fibers3DNearbyDistance", 1, 100000, 64);
        nearbyDistance->setSuffix(q->tr(" vx"));
        nearbyDistance->setSingleStep(8);
        nearbyDistance->setKeyboardTracking(false);
        nearbyDistance->setToolTip(
            q->tr("Shortest 3D distance between fiber centerlines, in active CT voxels. Computed from original segments, independently of zoom."));
        auto* nearbyRow = row(q->tr("Distance"), nearbyDistance);
        if (lineAnnotation) {
            nearby->hide();
            nearbyRow->hide();
        }

        section(q->tr("Highlight"));
        // Keep the stored 0..1 preference compatible with previous versions;
        // the controls expose a percentage with useful keyboard/button steps.
        double storedOpacity = settings.value(QStringLiteral("fibers3DOpacity"), display.unselectedOpacity).toDouble();
        if (!std::isfinite(storedOpacity))
            storedOpacity = display.unselectedOpacity;
        auto* opacity = new QSpinBox(settingsContents);
        preferenceFields.push_back(opacity);
        opacity->setObjectName(QStringLiteral("fibers3DOpacity"));
        opacity->setRange(0, 100);
        opacity->setSingleStep(5);
        opacity->setSuffix(QStringLiteral(" %"));
        opacity->setKeyboardTracking(false);
        opacity->setValue(int(std::lround(std::clamp(storedOpacity, 0.0, 1.0) * 100.0)));
        auto* opacitySlider = new QSlider(Qt::Horizontal, settingsContents);
        opacitySlider->setObjectName(QStringLiteral("fibers3DOpacitySlider"));
        opacitySlider->setRange(0, 100);
        opacitySlider->setSingleStep(5);
        opacitySlider->setPageStep(10);
        opacitySlider->setValue(opacity->value());
        auto* opacityLabel = new QLabel(q->tr("Other fibers opacity"), settingsContents);
        opacityLabel->setBuddy(opacity);
        opacityLabel->setWordWrap(true);
        opacitySlider->setAccessibleName(opacityLabel->text());
        opacity->setAccessibleName(opacityLabel->text());
        const auto opacityHelp =
            q->tr("Opacity of other fibers when a displayed fiber is selected. Without a selection, all fibers are opaque.");
        opacity->setToolTip(opacityHelp);
        opacitySlider->setToolTip(opacityHelp);
        body->addWidget(opacityLabel);
        body->addWidget(horizontal({opacitySlider, opacity}));
        QObject::connect(opacitySlider, &QSlider::valueChanged, opacity, &QSpinBox::setValue);
        QObject::connect(opacity, &QSpinBox::valueChanged, opacitySlider, &QSlider::setValue);
        auto* opacityHelpLabel = new QLabel(q->tr("Applied when a fiber is selected. All fibers stay opaque without a selection."), settingsContents);
        opacityHelpLabel->setWordWrap(true);
        body->addWidget(opacityHelpLabel);
        section(q->tr("Annotation details"));
        auto* annotationsVisible = check("fibers3DAnnotations", q->tr("Show markers and tags"), false);
        auto* selectedOnly = check("fibers3DAnnotationsSelected", q->tr("Selected fiber only"), true);
        body->addWidget(annotationsVisible);
        annotationOptions = selectedOnly;
        body->addWidget(annotationOptions);
        body->addStretch();
        // Memory is deliberately last; the everyday controls come first.
        auto* memoryPage = settingsTabs->widget(2);
        settingsTabs->removeTab(2);
        settingsTabs->addTab(memoryPage, q->tr("Memory"));

        auto changedDisplay = [=, this] {
            display.lineWidth = float(width->value());
            display.unselectedOpacity = float(opacity->value() / 100.0);
            display.showGaps = gaps->isChecked();
            display.showAnnotations = annotationsVisible->isChecked();
            display.annotationsSelectedOnly = selectedOnly->isChecked();
            display.perspective = perspective->isChecked();
            display.showBounds = bounds->isChecked();
            display.showCursor = cursor->isChecked();
            display.nearbyOnly = !lineAnnotation && nearby->isChecked();
            display.nearbyDistanceVoxels = nearbyDistance->value();
            nearbyDistance->setEnabled(display.nearbyOnly);
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
            boxWidthRow->setVisible(region->currentIndex() == 0);
            resizeSettings();
            request();
            saveSettings();
        });
        QObject::connect(memoryBudget, &QComboBox::currentIndexChanged, q, [this] {
            surfaceControls->setMemoryBudget(size_t(memoryBudget->currentData().toInt()) * 1024 * 1024);
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
        QObject::connect(opacity, &QSpinBox::valueChanged, q, changedDisplay);
        QObject::connect(nearbyDistance, &QSpinBox::valueChanged, q, changedDisplay);
        for (auto* w : {width, rangeMin, rangeMax})
            QObject::connect(w, &QDoubleSpinBox::valueChanged, q, changedDisplay);
        for (auto* w : {fixedRange, gaps, annotationsVisible, selectedOnly, perspective, bounds, cursor, nearby})
            QObject::connect(w, &QCheckBox::toggled, q, changedDisplay);
        boxWidthRow->setVisible(region->currentIndex() == 0);
        changedDisplay();
    }

    void resizeSettings()
    {
        settingsDialog->layout()->activate();
        // Never shrink or jump while changing a mode or tab. Grow only if
        // a longer translation/font or newly visible control requires it.
        settingsDialog->resize(settingsDialog->size().expandedTo(settingsDialog->sizeHint()).expandedTo(QSize(520, 460)));
    }

    void saveSettings()
    {
        QSettings s(vc3d::settingsFilePath(), QSettings::IniFormat);
        s.beginGroup(lineAnnotation ? QStringLiteral("lineAnnotation/fibers3d") : QStringLiteral("fibers3d"));
        // Only persist fields owned by this panel. Ribbon, slice and surface
        // controls manage their own preferences and may have another live view.
        for (auto* field : preferenceFields) {
            if (auto* w = qobject_cast<QCheckBox*>(field))
                s.setValue(w->objectName(), w->isChecked());
            else if (auto* w = qobject_cast<QComboBox*>(field))
                s.setValue(w->objectName(), w == memoryBudget ? w->currentData().toInt() : w->currentIndex());
            else if (auto* w = qobject_cast<QSpinBox*>(field))
                s.setValue(w->objectName(), w->objectName() == QStringLiteral("fibers3DOpacity") ? w->value() / 100.0 : w->value());
            else if (auto* w = qobject_cast<QDoubleSpinBox*>(field))
                s.setValue(w->objectName(), w->value());
        }
        s.setValue(QStringLiteral("fibers3DCustomColor"), display.custom.name());
        s.setValue(QStringLiteral("fibers3DHColor"), display.horizontal.name());
        s.setValue(QStringLiteral("fibers3DVColor"), display.vertical.name());
    }

    void applyDisplay()
    {
        display.highlightSelection = !lineAnnotation;
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
        customRow->setVisible(display.colorMode == ColorMode::Custom);
        familyRow->setVisible(display.colorMode == ColorMode::Family);
        directionRow->setVisible(display.colorMode == ColorMode::Direction);
        paletteRow->setVisible(length || (display.colorMode == ColorMode::Direction && display.directionMode != DirectionMode::XYZ));
        fixedRangeRow->setVisible(length);
        lengthRangeRow->setVisible(length && display.fixedLengthRange);
        annotationOptions->setVisible(display.showAnnotations);
        resizeSettings();
        canvas->setDisplaySettings(display);
        updateSelection();
        saveSettings();
    }

    void sharedFilters()
    {
        if (lineAnnotation)
            return;
        syncingFilters = true;
        automated->setFiberFilters(minimum->value(), h->isChecked(), v->isChecked());
        syncingFilters = false;
        request();
    }

    void request(bool clear = false)
    {
        if (lineAnnotation)
            return;
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
        if (!q->isVisible() || loading)
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
        options.byteBudget = size_t(memoryBudget->currentData().toUInt()) * 1024 * 1024;
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
        loading = true;
        watcher.setFuture(
            QtConcurrent::run([options, source, snapshot = std::move(snapshot), summaries = std::move(summaries), token, generation](QPromise<LoadResult>& promise) {
                LoadResult result;
                result.revision = generation;
                try {
                    // Publish a bounded, selectable first view without changing the
                    // user's final fiber or memory limits. Full loading stays on
                    // this worker, with the same revision/cancellation safeguards.
                    if (source && !options.bounds && options.byteBudget > size_t(64) * 1024 * 1024 &&
                        (!options.maximumFibers || options.maximumFibers > 2000)) {
                        auto initial = options;
                        initial.maximumFibers = 2000;
                        initial.byteBudget = size_t(64) * 1024 * 1024;
                        auto first = std::make_shared<Scene>(loadAutomated(*source, initial, token.get()));
                        if (token->load(std::memory_order_relaxed))
                            return;
                        if (!first->fibers.empty()) {
                            // Budget notices belong to the final view; this one is
                            // explicitly labelled as a preview still being loaded.
                            first->message.clear();
                            first->complete = true;
                            promise.addResult(LoadResult{std::move(first), {}, generation, true});
                        }
                    }
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
                if (!token->load(std::memory_order_relaxed))
                    promise.addResult(std::move(result));
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

    void clearSelection()
    {
        if (lineAnnotation)
            return;
        // Clear the shared selections as well as this view: slice overlays and
        // asynchronous inspection must not restore the old highlight.
        select({}, false);
        automated->clearSelection();
        emit q->nativeFiberSelected(0);
    }

    void updateStatus()
    {
        if (lineAnnotation) {
            status->setText(q->tr("Click the fiber to navigate · drag to orbit · wheel to zoom"));
            return;
        }
        if (!scene)
            return;
        const auto current = canvas->scene();
        if (current != scene) {
            status->setText(q->tr("Preparing %1 fibers…").arg(scene->fibers.size()));
            return;
        }
        QString text = canvas->isFiltering()
                           ? q->tr("Finding fibers near selection…")
                           : q->tr("%1 fibers displayed / %2 loaded").arg(canvas->visibleFiberCount()).arg(scene->fibers.size());
        if (preview)
            text += q->tr(" · preview — loading remaining fibers…");
        else if (!scene->complete)
            text += q->tr(" · partial view");
        if (!scene->message.isEmpty())
            text += QLatin1Char('\n') + scene->message;
        status->setText(text);
    }

    void updateSelection()
    {
        ribbonControls->setSelection(scene, selectedKey, canonicalSurface);
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
        selection->setToolTip(q->tr("Escape in the 3D view or click empty space to clear selection."));
        if (lineAnnotation)
            q->setLineAnnotationPosition(linePosition);
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
            menu.addAction(q->tr("Fit selected fiber"), q, [this] {
                if (lineAnnotation)
                    canvas->fitAll();
                else
                    canvas->fitSelected();
            });
            menu.addAction(q->tr("Center CT views here"), q, [this, point, menuRevision] {
                if (menuRevision != revision)
                    return;
                viewers->centerFocusAt({float(point[0]), float(point[1]), float(point[2])}, {0, 0, 1}, "fibers-3d");
            });
            if (lineAnnotation) {
                menu.addAction(q->tr("Go here in Line Annotation"), q, [this, index] { emit q->linePositionPicked(double(index)); });
            } else
                menu.addAction(q->tr("Open in Line Annotation"), q, [this, id, file, source, index, menuRevision] {
                    if (menuRevision != revision)
                        return;
                    if (source == Source::Automated)
                        automated->selectForInspection(qint64(id), true);
                    else if (annotations->hasLoadedFiber(id, file.toStdString()))
                        annotations->openFiberAtLinePointIndex(id, int(std::clamp<qint64>(index, 0, std::numeric_limits<int>::max())));
                });
        }
        if (!lineAnnotation && !selectedKey.isEmpty()) {
            menu.addSeparator();
            menu.addAction(q->tr("Clear selection"), q, [this] { clearSelection(); });
        }
        menu.exec(position);
    }
};

Fiber3DPanel::Fiber3DPanel(CState* s, ViewerManager* vm, LineAnnotationController* a, FiberCollectionController* f, QWidget* parent, bool lineAnnotation)
    : QWidget(parent), d(std::make_unique<Impl>(this, s, vm, a, f, lineAnnotation))
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
    d->syncCursor();
    d->request();
    d->ribbonControls->resume();
    d->sliceControls->resume();
    d->surfaceControls->resume();
}
void Fiber3DPanel::hideEvent(QHideEvent* e)
{
    QWidget::hideEvent(e);
    d->settingsDialog->hide();
    d->ribbonControls->cancel();
    d->sliceControls->cancel();
    d->surfaceControls->suspend();
    ++d->revision;
    d->debounce.stop();
    d->pending = true;
    if (d->cancelled)
        d->cancelled->store(true, std::memory_order_relaxed);
}
void Fiber3DPanel::setLineAnnotationViews(const vc3d::line_annotation::GeneratedViews& views, const QString& name, const QString& family)
{
    if (!d->lineAnnotation)
        return;
    d->linePoints = views.linePoints;
    d->lineArcs = vc3d::line_annotation::generatedCumulativeArcLength(d->linePoints);
    auto scene = std::make_shared<Scene>();
    if (const auto volume = d->state->currentVolume())
        scene->voxelSizeMm = volume->voxelSize() / 1000.;
    Fiber fiber = generatedAnnotationFiber(views);
    fiber.key = QStringLiteral("line-annotation:current");
    fiber.label = name;
    fiber.family = family == QStringLiteral("H") ? 'H' : family == QStringLiteral("V") ? 'V' : '?';
    fiber.lengthMm = d->lineArcs.empty() ? 0 : d->lineArcs.back() * scene->voxelSizeMm;
    d->selectedKey = fiber.key;
    scene->fibers.push_back(std::move(fiber));
    d->scene = scene;
    d->canonicalSurface = {};
    if (views.lineSurface && views.lineSideSlice) {
        // Deep copies: a live annotation can replace/edit surfaces while the worker samples CT.
        const cv::Mat_<cv::Vec3f> grid = views.lineSurface->rawPoints().clone();
        const cv::Mat_<cv::Vec3f> side = views.lineSideSlice->rawPoints().clone();
        const double spacing = views.stripPositionMap.stripGridSpacingBaseVoxels;
        d->canonicalSurface = [grid, side, spacing](const std::atomic_bool* cancel) {
            return ribbon::framesFromSurface(grid, side, spacing, false, cancel);
        };
    }
    d->canvas->setScene(scene);
    d->canvas->setSelectedKey(d->selectedKey);
    if (!d->fitted) {
        d->canvas->fitAll();
        d->fitted = true;
    }
    d->updateSelection();
    d->updateStatus();
}

void Fiber3DPanel::setLineAnnotationPosition(double position)
{
    if (!d->lineAnnotation || d->linePoints.empty() || !std::isfinite(position))
        return;
    d->linePosition = position;
    const auto p = vc3d::line_annotation::interpolatedGeneratedLinePoint(d->linePoints, position);
    const double arc = vc3d::line_annotation::generatedArcLengthAt(d->lineArcs, position);
    const double total = d->lineArcs.back();
    const QString label = tr("Position %1%").arg(total > 0 ? 100 * arc / total : 0, 0, 'f', 1);
    d->selection->setText(
        label + (d->scene && std::isfinite(d->scene->voxelSizeMm)
                     ? tr(" · %1 / %2 mm").arg(arc * d->scene->voxelSizeMm, 0, 'f', 2).arg(total * d->scene->voxelSizeMm, 0, 'f', 2)
                     : QString()));
    d->selection->setToolTip(tr("Distance from the first stored point, measured along the fiber."));
    d->canvas->setLinePosition(vc3d::line_annotation::finiteGeneratedPoint(p) ? std::optional<Point>{{p[0], p[1], p[2]}} : std::nullopt);
}

void Fiber3DPanel::clearLineAnnotation()
{
    if (!d->lineAnnotation)
        return;
    d->linePoints.clear();
    d->lineArcs.clear();
    d->scene.reset();
    d->canonicalSurface = {};
    d->fitted = false;
    d->selectedKey.clear();
    d->canvas->clear();
    d->canvas->setLinePosition({});
    d->updateSelection();
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
