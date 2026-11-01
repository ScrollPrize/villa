#include "Scene3DPanel.hpp"
#include "Scene3DCanvas.hpp"
#include "MemoryBudgetInput.hpp"
#include "CState.hpp"
#include "VCSettings.hpp"
#include "fibers3d/Fiber3DSliceControls.hpp"
#include "fibers3d/Fiber3DSurfaceControls.hpp"
#include "vc/core/types/Volume.hpp"
#include <QCheckBox>
#include <QComboBox>
#include <QHideEvent>
#include <QLabel>
#include <QPushButton>
#include <QScrollArea>
#include <QSettings>
#include <QShowEvent>
#include <QToolButton>
#include <QVBoxLayout>

namespace vc3d::scene3d
{
Scene3DPanel::Scene3DPanel(CState* state, ViewerManager* viewers, QWidget* parent) : QWidget(parent)
{
    setObjectName(QStringLiteral("scene3DPanel"));
    auto* layout = new QVBoxLayout(this);
    auto* toolbar = new QHBoxLayout;
    auto button = [&](const QString& label, const char* name) {
        auto* result = new QToolButton(this);
        result->setText(label);
        result->setObjectName(QString::fromLatin1(name));
        toolbar->addWidget(result);
        return result;
    };
    auto* ct = button(tr("CT"), "scene3DCT");
    displayMode_ = new QComboBox(this);
    displayMode_->setObjectName(QStringLiteral("scene3DDisplayMode"));
    displayMode_->addItem(tr("Surface + slices"));
    displayMode_->addItem(tr("CT volume"));
    displayMode_->setToolTip(tr("Show the selected surface and CT slices, or the translucent CT volume without a surface or slices."));
    toolbar->insertWidget(0, displayMode_);
    auto* fit = button(tr("Fit all"), "scene3DFitAll");
    auto* perspective = button(tr("Perspective"), "scene3DPerspective");
    perspective->setCheckable(true);
    auto* settings = button(tr("Settings"), "scene3DSettings");
    settings->setCheckable(true);
    toolbar->addStretch();
    auto* detach = button(tr("Detach"), "scene3DDetach");
    layout->addLayout(toolbar);
    auto* volumeLabel = new QLabel(this);
    volumeLabel->setTextFormat(Qt::PlainText);
    volumeLabel->setWordWrap(true);
    layout->addWidget(volumeLabel);
    auto refreshVolume = [state, volumeLabel] {
        const auto volume = state->currentVolume();
        volumeLabel->setText(volume ? tr("CT volume: %1").arg(QString::fromStdString(volume->id())) : tr("Attach a CT volume to view its slices."));
    };
    connect(state, &CState::volumeChanged, this, refreshVolume);
    refreshVolume();

    canvas_ = new Scene3DCanvas(this);
    canvas_->setObjectName(QStringLiteral("scene3DCanvas"));
    layout->addWidget(canvas_, 1);
    slices_ = new fibers3d::Fiber3DSliceControls(state, viewers, canvas_, ct, this, QStringLiteral("scene3d"));
    slices_->enableXYPositionControl();
    connect(canvas_, &Scene3DCanvas::surfaceDisplayChanged, this, [this] {
        std::optional<Bounds> bounds;
        if (const auto mesh = canvas_->surfaceMesh(); displayMode_->currentIndex() == 0 && mesh && !mesh->indices.empty() && canvas_->isSurfaceVisible())
            bounds = mesh->bounds;
        slices_->setMeshBounds(bounds);
    });
    layout->addWidget(slices_);

    auto* scroll = new QScrollArea(this);
    scroll->setWidgetResizable(true);
    scroll->setMaximumHeight(300);
    auto* controls = new QWidget(scroll);
    auto* options = new QVBoxLayout(controls);
    auto* sources = new QHBoxLayout;
    auto* attach = new QPushButton(tr("Attach CT volume…"), controls);
    auto* import = new QPushButton(tr("Import tifxyz surface…"), controls);
    sources->addWidget(attach);
    sources->addWidget(import);
    options->addLayout(sources);
    auto* help = new QLabel(tr("While a mesh is displayed, only the XY slice is shown, centered on the mesh. Use XY slice Z to move through it, or Center on mesh to return to its center. Volumetric thickness and material settings are in Composite View. Select a surface in Main's surface list to display its mesh."), controls);
    help->setWordWrap(true);
    options->addWidget(help);
    surface_ = new fibers3d::Fiber3DSurfaceControls(state, canvas_, controls, QStringLiteral("scene3d"));
    options->addWidget(surface_);
    QSettings prefs(vc3d::settingsFilePath(), QSettings::IniFormat);
    auto* budget = makeMemoryBudgetInput(controls, prefs.value("scene3d/memoryMiB", 2048).toDouble());
    budget->setObjectName(QStringLiteral("scene3DMemoryBudgetMiB"));
    budget->setToolTip(tr("Memory allowance for 3D loading, allocated only as needed. Type a larger value for large surfaces. The CT overview keeps its preview size bounded automatically."));
    auto* budgetRow = new QHBoxLayout;
    auto* budgetLabel = new QLabel(tr("3D memory budget"), controls);
    budgetLabel->setBuddy(budget);
    budgetRow->addWidget(budgetLabel);
    budgetRow->addWidget(budget, 1);
    surface_->setMemoryBudget(memoryBudgetBytes(budget));
    options->addLayout(budgetRow);
    scroll->setWidget(controls);
    layout->addWidget(scroll);
    // Surface sampling follows panel visibility, independently of whether its
    // settings are expanded. Slice controls remain visible while CT is enabled.
    settings->setChecked(true);
    connect(settings, &QToolButton::toggled, scroll, &QWidget::setVisible);
    connect(attach, &QPushButton::clicked, this, &Scene3DPanel::attachVolumeRequested);
    connect(import, &QPushButton::clicked, this, &Scene3DPanel::importSurfaceRequested);
    connect(detach, &QToolButton::clicked, this, &Scene3DPanel::detachRequested);
    connect(fit, &QToolButton::clicked, canvas_, &Scene3DCanvas::fitAll);
    connect(perspective, &QToolButton::toggled, canvas_, &Scene3DCanvas::setPerspective);
    perspective->setChecked(prefs.value("scene3d/perspective", false).toBool());
    connect(perspective, &QToolButton::toggled, this, [](bool enabled) {
        QSettings prefs(vc3d::settingsFilePath(), QSettings::IniFormat);
        prefs.setValue("scene3d/perspective", enabled);
    });
    connect(budget, &QDoubleSpinBox::valueChanged, this, [this, budget] {
        surface_->setMemoryBudget(memoryBudgetBytes(budget));
        QSettings prefs(vc3d::settingsFilePath(), QSettings::IniFormat);
        prefs.setValue("scene3d/memoryMiB", budget->value());
    });
    connect(state, &CState::volumeClosing, canvas_, &Scene3DCanvas::clearContent);
    connect(state, &CState::volumeClosing, this, [volumeLabel] { volumeLabel->setText(tr("Attach a CT volume to view its slices.")); });
    connect(state, &CState::volumeChanged, canvas_, &Scene3DCanvas::clearContent);
    ct->setChecked(true);
    const auto surfaceHelp = help->text();
    connect(displayMode_, &QComboBox::currentIndexChanged, this, [this, ct, help, surfaceHelp, import](int index) {
        const bool volume = index == 1;
        if (volume) {
            savedCT_ = ct->isChecked();
            ct->setChecked(false);
            slices_->cancel();
            slices_->hide();
        }
        ct->setVisible(!volume);
        import->setVisible(!volume);
        canvas_->clearContent();
        surface_->setVolumeSource(volume);
        if (!volume) ct->setChecked(savedCT_);
        help->setText(volume ? tr("Drag to rotate the CT volume. Volume density controls how much internal structure shows through. Preview resolution adjusts automatically to keep memory bounded; no surface mesh is needed.") : surfaceHelp);
    });
}
void Scene3DPanel::showEvent(QShowEvent* event)
{
    QWidget::showEvent(event);
    if (displayMode_->currentIndex() == 0) slices_->resume();
    surface_->resume();
}
void Scene3DPanel::hideEvent(QHideEvent* event)
{
    QWidget::hideEvent(event);
    slices_->cancel();
    surface_->suspend();
}
}

#ifdef VC_TESTING
#include "Scene3DPanelTest.inc"
#endif
