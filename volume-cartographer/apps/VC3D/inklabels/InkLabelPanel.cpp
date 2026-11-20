#include "inklabels/InkLabelPanel.hpp"

#include "elements/LabeledControlRow.hpp"
#include "inklabels/InkLabelController.hpp"

#include <QButtonGroup>
#include <QCheckBox>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QInputDialog>
#include <QLabel>
#include <QLineEdit>
#include <QLocale>
#include <QPushButton>
#include <QSignalBlocker>
#include <QSlider>
#include <QVBoxLayout>

#include <filesystem>
#include <string>


InkLabelPanel::InkLabelPanel(InkLabelController* controller, QWidget* parent)
    : QWidget(parent)
    , _controller(controller)
{
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(8);

    _enabled = new QCheckBox(tr("Label ink on the segment view"), this);
    _enabled->setToolTip(tr("Left drag paints; Alt+drag paints the other class. Unpainted pixels stay unlabelled "
                            "and do not train."));
    layout->addWidget(_enabled);

    auto* toolRow = new LabeledControlRow(tr("Tool"), this);
    auto* tools = new QButtonGroup(this);
    _ink = new QPushButton(tr("Ink (B)"), toolRow);
    _noInk = new QPushButton(tr("No ink (N)"), toolRow);
    _erase = new QPushButton(tr("Erase (U)"), toolRow);
    _ink->setToolTip(tr("Paint ink: positive examples."));
    _noInk->setToolTip(tr("Paint no ink: negative examples. Keep them off letters, row gaps and ruled lines."));
    _erase->setToolTip(tr("Return pixels to unlabelled."));
    for (auto* button : {_ink, _noInk, _erase}) {
        button->setCheckable(true);
        tools->addButton(button);
        toolRow->addControl(button);
    }
    toolRow->addStretch(1);
    layout->addWidget(toolRow);

    auto* radiusRow = new LabeledControlRow(tr("Radius"), this);
    _radius = new QDoubleSpinBox(radiusRow);
    _radius->setRange(0.5, 400.0);
    _radius->setDecimals(1);
    _radius->setSuffix(tr(" px"));
    _radius->setToolTip(tr("Brush radius in render pixels ([ and ] change it)."));
    radiusRow->addControl(_radius);
    radiusRow->addStretch(1);
    layout->addWidget(radiusRow);

    auto* opacityRow = new LabeledControlRow(tr("Opacity"), this);
    _opacity = new QSlider(Qt::Horizontal, opacityRow);
    _opacity->setRange(0, 100);
    _opacity->setSingleStep(5);
    _opacityValue = new QLabel(opacityRow);
    _opacityValue->setMinimumWidth(36);
    _visible = new QCheckBox(tr("Show (L)"), opacityRow);
    opacityRow->addControl(_opacity, 1);
    opacityRow->addControl(_opacityValue);
    opacityRow->addControl(_visible);
    layout->addWidget(opacityRow);

    auto* historyRow = new LabeledControlRow(tr("History"), this);
    _undo = new QPushButton(tr("Undo"), historyRow);
    _redo = new QPushButton(tr("Redo"), historyRow);
    historyRow->addControl(_undo);
    historyRow->addControl(_redo);
    historyRow->addStretch(1);
    layout->addWidget(historyRow);

    auto* filesRow = new LabeledControlRow(tr("Labels"), this);
    _save = new QPushButton(tr("Save"), filesRow);
    _saveTo = new QPushButton(tr("Save to Folder..."), filesRow);
    _saveTo->setToolTip(tr("Write <segment>_inklabels.tif and <segment>_supervision_mask.tif for training."));
    _load = new QPushButton(tr("Load..."), filesRow);
    filesRow->addControl(_save);
    filesRow->addControl(_saveTo);
    filesRow->addControl(_load);
    filesRow->addStretch(1);
    layout->addWidget(filesRow);

    _status = new QLabel(this);
    _status->setWordWrap(true);
    _status->setTextInteractionFlags(Qt::TextSelectableByMouse);
    layout->addWidget(_status);
    layout->addStretch(1);

    if (!_controller) {
        setEnabled(false);
        return;
    }
    connect(_enabled, &QCheckBox::toggled, _controller, &InkLabelController::setLabellingEnabled);
    connect(_ink, &QPushButton::clicked, this, [this]() { _controller->setTool(InkLabelController::Tool::Ink); });
    connect(_noInk, &QPushButton::clicked, this, [this]() { _controller->setTool(InkLabelController::Tool::NoInk); });
    connect(_erase, &QPushButton::clicked, this, [this]() { _controller->setTool(InkLabelController::Tool::Erase); });
    connect(_radius, qOverload<double>(&QDoubleSpinBox::valueChanged), this,
            [this](double value) { _controller->setBrushRadius(static_cast<float>(value)); });
    connect(_opacity, &QSlider::valueChanged, _controller, &InkLabelController::setOpacity);
    connect(_visible, &QCheckBox::toggled, _controller, &InkLabelController::setLabelsVisible);
    connect(_undo, &QPushButton::clicked, _controller, &InkLabelController::undo);
    connect(_redo, &QPushButton::clicked, _controller, &InkLabelController::redo);
    connect(_save, &QPushButton::clicked, _controller, &InkLabelController::save);
    connect(_saveTo, &QPushButton::clicked, this, &InkLabelPanel::saveToFolder);
    connect(_load, &QPushButton::clicked, this, &InkLabelPanel::loadFromFile);
    connect(_controller, &InkLabelController::stateChanged, this, &InkLabelPanel::updateFromController);
    updateFromController();
}

void InkLabelPanel::updateFromController()
{
    const QSignalBlocker b0(_enabled), b1(_radius), b2(_opacity), b3(_visible);
    _enabled->setChecked(_controller->labellingEnabled());
    _ink->setChecked(_controller->tool() == InkLabelController::Tool::Ink);
    _noInk->setChecked(_controller->tool() == InkLabelController::Tool::NoInk);
    _erase->setChecked(_controller->tool() == InkLabelController::Tool::Erase);
    _radius->setValue(_controller->brushRadius());
    _opacity->setValue(_controller->opacity());
    _opacityValue->setText(QStringLiteral("%1%").arg(_controller->opacity()));
    _visible->setChecked(_controller->labelsVisible());

    const bool surface = _controller->hasSurface();
    const auto& layer = _controller->layer();
    _undo->setEnabled(surface && layer.canUndo());
    _redo->setEnabled(surface && layer.canRedo());
    _undo->setText(layer.undoDepth() ? tr("Undo (%1)").arg(layer.undoDepth()) : tr("Undo"));
    _save->setEnabled(surface && !_controller->labelDirectory().empty() && !_controller->savedLabelsBlocked());
    _save->setToolTip(_controller->labelDirectory().empty()
                          ? tr("Only for a surface in a store; use Save to Folder.")
                          : tr("Save to %1 (also done on switching surface).")
                                .arg(QString::fromStdString(_controller->labelDirectory().string())));
    _saveTo->setEnabled(surface);
    _load->setEnabled(surface);
    if (!surface) {
        _status->setText(tr("Open a segment to label it."));
        return;
    }
    const auto counts = layer.counts();
    const QLocale locale;
    _status->setText(tr("%1: %2 x %3 px. Ink %4 px, no ink %5 px.%6")
                         .arg(QString::fromStdString(_controller->surfaceName()))
                         .arg(layer.size().width)
                         .arg(layer.size().height)
                         .arg(locale.toString(static_cast<qulonglong>(counts[2])))
                         .arg(locale.toString(static_cast<qulonglong>(counts[1])))
                         .arg(layer.modified() ? tr(" Unsaved changes.") : QString()));
    if (_controller->savedLabelsBlocked()) {
        _status->setText(_status->text() +
                         tr(" The labels saved for this surface do not match its grid (it was edited): "
                            "not drawing or saving over them."));
    }
}

void InkLabelPanel::saveToFolder()
{
    const QString directory = QFileDialog::getExistingDirectory(this, tr("Save Ink Labels to Folder"));
    if (directory.isEmpty()) {
        return;
    }
    bool ok = false;
    const QString prefix = QInputDialog::getText(
        this, tr("Label File Prefix"),
        tr("Prefix (the trainer needs the segment folder's name):"), QLineEdit::Normal,
        QString::fromStdString(_controller->surfaceName()), &ok);
    if (!ok || prefix.trimmed().isEmpty()) {
        return;
    }
    _controller->saveTo(directory.toStdString(), prefix.trimmed().toStdString());
}

void InkLabelPanel::loadFromFile()
{
    const QString path = QFileDialog::getOpenFileName(this, tr("Load Ink Labels"), QString(),
                                                      tr("Ink labels (*_inklabels*.tif *_inklabels*.tiff *_inklabels*.png)"));
    if (path.isEmpty()) {
        return;
    }
    const std::filesystem::path ink(path.toStdString());
    std::string supervision = ink.filename().string();
    const auto at = supervision.rfind("_inklabels");
    if (at == std::string::npos) {
        return;
    }
    supervision.replace(at, std::string("_inklabels").size(), "_supervision_mask");
    _controller->load({ink, ink.parent_path() / supervision});
}
