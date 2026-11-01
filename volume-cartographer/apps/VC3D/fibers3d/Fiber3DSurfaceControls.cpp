#include "Fiber3DSurfaceControls.hpp"
#include "Fiber3DSurfaceMesh.hpp"
#include "Fiber3DCanvas.hpp"
#include "CState.hpp"
#include "VCSettings.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include <QCheckBox>
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
    QString error;
};
}  // namespace
struct Fiber3DSurfaceControls::Impl {
    Fiber3DSurfaceControls* q;
    CState* state;
    Fiber3DCanvas* canvas;
    QCheckBox* enabled;
    QCheckBox* nearby;
    QSpinBox* distance;
    QLabel* status;
    QPushButton* fit;
    QTimer debounce;
    QFutureWatcher<Result> watcher;
    std::shared_ptr<std::atomic_bool> cancelled;
    bool suspended = true, busy = false, pending = false;
    size_t maxBytes = 512ull * 1024 * 1024;

    Impl(Fiber3DSurfaceControls* widget, CState* s, Fiber3DCanvas* c) : q(widget), state(s), canvas(c)
    {
        auto* layout = new QVBoxLayout(q);
        layout->setContentsMargins(0, 0, 0, 0);
        layout->setSpacing(10);
        auto heading = [&](const QString& text) {
            auto* title = new QLabel(text, q);
            auto font = title->font();
            font.setBold(true);
            title->setFont(font);
            layout->addWidget(title);
        };
        heading(q->tr("Surface selected in Main"));
        status = new QLabel(q);
        status->setObjectName(QStringLiteral("fibers3DSurfaceStatus"));
        status->setTextFormat(Qt::PlainText);
        status->setWordWrap(true);
        layout->addWidget(status);
        layout->addSpacing(5);
        QSettings prefs(vc3d::settingsFilePath(), QSettings::IniFormat);
        enabled = new QCheckBox(q->tr("Display surface mesh"), q);
        enabled->setObjectName(QStringLiteral("fibers3DSurfaceMesh"));
        enabled->setToolTip(q->tr("Display the surface selected in Main, in the same CT coordinates as the fibers."));
        enabled->setChecked(prefs.value(QStringLiteral("fibers3d/fibers3DSurfaceMesh"), false).toBool());
        canvas->setSurfaceVisible(enabled->isChecked());
        layout->addWidget(enabled);
        auto* options = new QWidget(q);
        auto* line = new QHBoxLayout(options);
        line->setContentsMargins(0, 0, 0, 0);
        auto* label = new QLabel(q->tr("Opacity"), q);
        auto* opacity = new QSpinBox(q);
        opacity->setObjectName(QStringLiteral("fibers3DSurfaceOpacity"));
        opacity->setRange(0, 100);
        opacity->setSingleStep(5);
        opacity->setSuffix(QStringLiteral(" %"));
        opacity->setValue(prefs.value(QStringLiteral("fibers3d/fibers3DSurfaceOpacity"), 100).toInt());
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
        heading(q->tr("Nearby fibers"));
        nearby = new QCheckBox(q->tr("Only fibers near surface"), q);
        nearby->setObjectName(QStringLiteral("fibers3DNearSurface"));
        nearby->setChecked(prefs.value(QStringLiteral("fibers3d/fibers3DNearSurface"), false).toBool());
        nearby->setToolTip(
            q->tr("Keep whole loaded fibers that pass within this distance of the selected surface, on either side. Works even when the mesh is hidden. Combines with other fiber filters."));
        layout->addWidget(nearby);
        auto* distanceRow = new QHBoxLayout;
        auto* distanceLabel = new QLabel(q->tr("Distance to surface"), q);
        distance = new QSpinBox(q);
        distance->setObjectName(QStringLiteral("fibers3DSurfaceDistance"));
        distance->setRange(0, 100000);
        distance->setSingleStep(8);
        distance->setSuffix(q->tr(" vx"));
        distance->setKeyboardTracking(false);
        distance->setValue(prefs.value(QStringLiteral("fibers3d/fibers3DSurfaceDistance"), 64).toInt());
        distance->setEnabled(nearby->isChecked());
        distanceLabel->setBuddy(distance);
        distanceRow->addWidget(distanceLabel);
        distanceRow->addWidget(distance, 1);
        layout->addLayout(distanceRow);
        auto* help = new QLabel(q->tr("Keeps whole fibers within this distance, on either side. Also works with the mesh hidden."), q);
        help->setWordWrap(true);
        layout->addWidget(help);
        canvas->setSurfaceFilter(nearby->isChecked(), distance->value());
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
            prefs.setValue(QStringLiteral("fibers3d/fibers3DSurfaceOpacity"), value);
        });
        canvas->setSurfaceOpacity(opacity->value() / 100.f);
        QObject::connect(fit, &QPushButton::clicked, canvas, &Fiber3DCanvas::fitSurface);
        options->setEnabled(enabled->isChecked());
        QObject::connect(enabled, &QCheckBox::toggled, q, [this, options](bool on) {
            options->setEnabled(on);
            canvas->setSurfaceVisible(on);
            QSettings prefs(vc3d::settingsFilePath(), QSettings::IniFormat);
            prefs.setValue(QStringLiteral("fibers3d/fibers3DSurfaceMesh"), on);
            refreshOptions();
        });
        auto filterChanged = [this] {
            distance->setEnabled(nearby->isChecked());
            QSettings prefs(vc3d::settingsFilePath(), QSettings::IniFormat);
            prefs.setValue(QStringLiteral("fibers3d/fibers3DNearSurface"), nearby->isChecked());
            prefs.setValue(QStringLiteral("fibers3d/fibers3DSurfaceDistance"), distance->value());
            canvas->setSurfaceFilter(nearby->isChecked(), distance->value());
            refreshOptions();
        };
        QObject::connect(nearby, &QCheckBox::toggled, q, filterChanged);
        QObject::connect(distance, &QSpinBox::valueChanged, q, filterChanged);
        debounce.setSingleShot(true);
        debounce.setInterval(150);
        QObject::connect(&debounce, &QTimer::timeout, q, [this] { start(); });
        QObject::connect(state, &CState::surfaceChanged, q, [this](const std::string& name) {
            if (name.empty() || name == "segmentation")
                invalidate();
        });
        QObject::connect(state, &CState::surfaceWillBeDeleted, q, [this](const std::string& name) {
            if (name == "segmentation")
                clear();
        });
        QObject::connect(state, &CState::volumeClosing, q, [this] { clear(); });
        QObject::connect(state, &CState::volumeChanged, q, [this] { invalidate(); });
        QObject::connect(state, &CState::vpkgChanged, q, [this] { invalidate(); });
        QObject::connect(&watcher, &QFutureWatcher<Result>::finished, q, [this] {
            busy = false;
            auto result = watcher.future().takeResult();
            if (!cancelled->load() && !suspended && wanted()) {
                canvas->setSurfaceMesh(result.mesh);
                const bool valid = result.mesh && !result.mesh->indices.empty();
                fit->setEnabled(valid);
                if (!result.error.isEmpty())
                    status->setText(result.error);
                else if (valid)
                    status->setText(
                        state->activeSurfaceId().empty() ? q->tr("Surface selected in Main")
                                                         : q->tr("Surface %1").arg(QString::fromStdString(state->activeSurfaceId())));
                else
                    status->setText(q->tr("The selected surface has no valid mesh cells."));
            }
            if (pending)
                debounce.start();
        });
        status->setText(q->tr("Surface selected in Main"));
    }
    bool wanted() const { return enabled->isChecked() || nearby->isChecked(); }
    void refreshOptions()
    {
        if (!wanted() || !canvas->surfaceMesh())
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
    }
    void invalidate()
    {
        clear();
        if (!wanted()) {
            status->setText(q->tr("Surface selected in Main"));
            return;
        }
        pending = true;
        if (!suspended) {
            status->setText(q->tr("Preparing surface mesh…"));
            debounce.start();
        }
    }
    void start()
    {
        if (busy || suspended || !pending || !wanted())
            return;
        pending = false;
        auto quad = std::dynamic_pointer_cast<QuadSurface>(state->surface("segmentation"));
        if (!quad) {
            status->setText(q->tr("Select a surface in Main."));
            return;
        }
        try {
            // QuadSurface's live grid can be edited in place. Copy on its GUI
            // owner thread; workers never access mutable surface state.
            const auto grid = quad->gridSize();
            if (uint64_t(grid.width) * uint64_t(grid.height) > maxBytes / (sizeof(surface::Vertex) + 6 * sizeof(uint32_t)))
                throw std::runtime_error("Surface exceeds the 3D memory budget. Increase Memory budget in Settings.");
            auto points = quad->rawPoints().clone();
            cancelled = std::make_shared<std::atomic_bool>(false);
            busy = true;
            watcher.setFuture(QtConcurrent::run([points = std::move(points), stop = cancelled, budget = maxBytes] {
                Result r;
                try {
                    r.mesh = surface::build(points, budget, *stop);
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
Fiber3DSurfaceControls::Fiber3DSurfaceControls(CState* state, Fiber3DCanvas* canvas, QWidget* parent)
    : QWidget(parent), d(std::make_unique<Impl>(this, state, canvas))
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
}  // namespace vc3d::fibers3d
