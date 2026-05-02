#include "viewer_controls/panels/ViewerTransformsPanel.hpp"

#include "elements/LabeledControlRow.hpp"

#include <QCheckBox>
#include <QLabel>
#include <QPushButton>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QVBoxLayout>

ViewerTransformsPanel::ViewerTransformsPanel(QWidget* parent)
    : QWidget(parent)
{
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(8);

    _preview = new QCheckBox(tr("Preview Result"), this);
    _preview->setToolTip(
        tr("Preview the scaled and/or affine-transformed segmentation."));
    layout->addWidget(_preview);

    _scaleOnly = new QCheckBox(tr("Scale Only"), this);
    _scaleOnly->setToolTip(
        tr("Ignore any loaded or volume affine and apply scale only."));
    layout->addWidget(_scaleOnly);

    _invert = new QCheckBox(tr("Invert Affine"), this);
    _invert->setToolTip(
        tr("Invert the loaded affine. Ignored when no affine transform is loaded."));
    layout->addWidget(_invert);

    auto* scaleRow = new LabeledControlRow(tr("Scale"), this);
    _scale = new QSpinBox(scaleRow);
    _scale->setMinimum(1);
    _scale->setMaximum(1000);
    _scale->setValue(1);
    _scale->setToolTip(
        tr("Multiply segmentation points by this integer. Works with or without an affine transform."));
    scaleRow->setLabelToolTip(_scale->toolTip());
    scaleRow->addControl(_scale);
    scaleRow->addStretch(1);
    layout->addWidget(scaleRow);

    _loadAffine = new QPushButton(tr("Load Affine (Optional)"), this);
    _loadAffine->setToolTip(
        tr("Load an affine JSON from a local path or URL. Leave the dialog blank to return to the current volume transform."));
    layout->addWidget(_loadAffine);

    _saveTransformed = new QPushButton(tr("Save Transformed"), this);
    _saveTransformed->setToolTip(
        tr("Save a new surface using the current scale and optional affine transform."));
    layout->addWidget(_saveTransformed);

    _status = new QLabel(this);
    _status->setWordWrap(true);
    _status->setTextInteractionFlags(Qt::TextSelectableByMouse);
    layout->addWidget(_status);

    connect(_preview, &QCheckBox::toggled, this, &ViewerTransformsPanel::previewToggled);
    connect(_scaleOnly, &QCheckBox::toggled, this, &ViewerTransformsPanel::stateChanged);
    connect(_invert, &QCheckBox::toggled, this, &ViewerTransformsPanel::stateChanged);
    connect(_scale, QOverload<int>::of(&QSpinBox::valueChanged),
            this, [this](int) { emit stateChanged(); });
    connect(_loadAffine, &QPushButton::clicked, this, &ViewerTransformsPanel::loadAffineRequested);
    connect(_saveTransformed, &QPushButton::clicked, this, &ViewerTransformsPanel::saveTransformedRequested);
}

bool ViewerTransformsPanel::previewChecked() const
{
    return _preview && _preview->isChecked();
}

bool ViewerTransformsPanel::scaleOnlyChecked() const
{
    return _scaleOnly && _scaleOnly->isChecked();
}

bool ViewerTransformsPanel::invertChecked() const
{
    return _invert && _invert->isChecked();
}

int ViewerTransformsPanel::scaleValue() const
{
    return _scale ? _scale->value() : 1;
}

void ViewerTransformsPanel::setPreviewChecked(bool checked, bool blockSignals)
{
    if (!_preview) {
        return;
    }
    if (blockSignals) {
        const QSignalBlocker blocker(_preview);
        _preview->setChecked(checked);
    } else {
        _preview->setChecked(checked);
    }
}

void ViewerTransformsPanel::applyUiState(const UiState& state)
{
    if (_preview) {
        _preview->setEnabled(state.previewAvailable);
    }
    if (_scaleOnly) {
        _scaleOnly->setEnabled(state.sourceAvailable && !state.editingEnabled);
    }
    if (_invert) {
        _invert->setEnabled(!state.editingEnabled && state.affineAvailable && !state.scaleOnly);
    }
    if (_scale) {
        _scale->setEnabled(state.sourceAvailable && !state.editingEnabled);
    }
    if (_loadAffine) {
        _loadAffine->setEnabled(!state.editingEnabled);
    }
    if (_saveTransformed) {
        _saveTransformed->setEnabled(state.saveAvailable);
    }
    if (_status) {
        _status->setText(state.statusText);
    }
}
