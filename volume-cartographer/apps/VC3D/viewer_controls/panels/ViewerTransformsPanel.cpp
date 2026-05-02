#include "viewer_controls/panels/ViewerTransformsPanel.hpp"

#include "elements/LabeledControlRow.hpp"

#include <QCheckBox>
#include <QLabel>
#include <QPushButton>
#include <QSpinBox>
#include <QVBoxLayout>

ViewerTransformsPanel::ViewerTransformsPanel(QWidget* parent)
    : QWidget(parent)
{
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(8);

    _controls.preview = new QCheckBox(tr("Preview Result"), this);
    _controls.preview->setToolTip(
        tr("Preview the scaled and/or affine-transformed segmentation."));
    layout->addWidget(_controls.preview);

    _controls.scaleOnly = new QCheckBox(tr("Scale Only"), this);
    _controls.scaleOnly->setToolTip(
        tr("Ignore any loaded or volume affine and apply scale only."));
    layout->addWidget(_controls.scaleOnly);

    _controls.invert = new QCheckBox(tr("Invert Affine"), this);
    _controls.invert->setToolTip(
        tr("Invert the loaded affine. Ignored when no affine transform is loaded."));
    layout->addWidget(_controls.invert);

    auto* scaleRow = new LabeledControlRow(tr("Scale"), this);
    _controls.scale = new QSpinBox(scaleRow);
    _controls.scale->setMinimum(1);
    _controls.scale->setMaximum(1000);
    _controls.scale->setValue(1);
    _controls.scale->setToolTip(
        tr("Multiply segmentation points by this integer. Works with or without an affine transform."));
    scaleRow->setLabelToolTip(_controls.scale->toolTip());
    scaleRow->addControl(_controls.scale);
    scaleRow->addStretch(1);
    layout->addWidget(scaleRow);

    _controls.loadAffine = new QPushButton(tr("Load Affine (Optional)"), this);
    _controls.loadAffine->setToolTip(
        tr("Load an affine JSON from a local path or URL. Leave the dialog blank to return to the current volume transform."));
    layout->addWidget(_controls.loadAffine);

    _controls.saveTransformed = new QPushButton(tr("Save Transformed"), this);
    _controls.saveTransformed->setToolTip(
        tr("Save a new surface using the current scale and optional affine transform."));
    layout->addWidget(_controls.saveTransformed);

    _controls.status = new QLabel(this);
    _controls.status->setWordWrap(true);
    _controls.status->setTextInteractionFlags(Qt::TextSelectableByMouse);
    layout->addWidget(_controls.status);
}
