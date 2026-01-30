#include "SegmentationDirectionFieldPanel.hpp"

#include "elements/CollapsibleSettingsGroup.hpp"
#include "segmentation/SegmentationGrowth.hpp"

#include <QAbstractItemView>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QPushButton>
#include <QToolButton>
#include <QVBoxLayout>

SegmentationDirectionFieldPanel::SegmentationDirectionFieldPanel(QWidget* parent)
    : QWidget(parent)
{
    auto* panelLayout = new QVBoxLayout(this);
    panelLayout->setContentsMargins(0, 0, 0, 0);
    panelLayout->setSpacing(0);

    _groupDirectionField = new CollapsibleSettingsGroup(tr("Direction Fields"), this);

    auto* directionParent = _groupDirectionField->contentWidget();

    _groupDirectionField->addRow(tr("Zarr folder:"), [&](QHBoxLayout* row) {
        _directionFieldPathEdit = new QLineEdit(directionParent);
        _directionFieldPathEdit->setToolTip(tr("Filesystem path to the direction field zarr folder."));
        _directionFieldBrowseButton = new QToolButton(directionParent);
        _directionFieldBrowseButton->setText(QStringLiteral("..."));
        _directionFieldBrowseButton->setToolTip(tr("Browse for a direction field dataset on disk."));
        row->addWidget(_directionFieldPathEdit, 1);
        row->addWidget(_directionFieldBrowseButton);
    }, tr("Filesystem path to the direction field zarr folder."));

    _groupDirectionField->addRow(tr("Orientation:"), [&](QHBoxLayout* row) {
        _comboDirectionFieldOrientation = new QComboBox(directionParent);
        _comboDirectionFieldOrientation->setToolTip(tr("Select which axis the direction field describes."));
        _comboDirectionFieldOrientation->addItem(tr("Normal"), static_cast<int>(SegmentationDirectionFieldOrientation::Normal));
        _comboDirectionFieldOrientation->addItem(tr("Horizontal"), static_cast<int>(SegmentationDirectionFieldOrientation::Horizontal));
        _comboDirectionFieldOrientation->addItem(tr("Vertical"), static_cast<int>(SegmentationDirectionFieldOrientation::Vertical));
        row->addWidget(_comboDirectionFieldOrientation);
        row->addSpacing(12);

        auto* scaleLabel = new QLabel(tr("Scale level:"), directionParent);
        _comboDirectionFieldScale = new QComboBox(directionParent);
        _comboDirectionFieldScale->setToolTip(tr("Choose the multiscale level sampled from the direction field."));
        for (int scale = 0; scale <= 5; ++scale) {
            _comboDirectionFieldScale->addItem(QString::number(scale), scale);
        }
        row->addWidget(scaleLabel);
        row->addWidget(_comboDirectionFieldScale);
        row->addSpacing(12);

        auto* weightLabel = new QLabel(tr("Weight:"), directionParent);
        _spinDirectionFieldWeight = new QDoubleSpinBox(directionParent);
        _spinDirectionFieldWeight->setDecimals(2);
        _spinDirectionFieldWeight->setToolTip(tr("Relative influence of this direction field during growth."));
        _spinDirectionFieldWeight->setRange(0.0, 10.0);
        _spinDirectionFieldWeight->setSingleStep(0.1);
        row->addWidget(weightLabel);
        row->addWidget(_spinDirectionFieldWeight);
        row->addStretch(1);
    });

    _groupDirectionField->addRow(QString(), [&](QHBoxLayout* row) {
        _directionFieldAddButton = new QPushButton(tr("Add"), directionParent);
        _directionFieldAddButton->setToolTip(tr("Save the current direction field parameters to the list."));
        _directionFieldRemoveButton = new QPushButton(tr("Remove"), directionParent);
        _directionFieldRemoveButton->setToolTip(tr("Delete the selected direction field entry."));
        _directionFieldRemoveButton->setEnabled(false);
        _directionFieldClearButton = new QPushButton(tr("Clear"), directionParent);
        _directionFieldClearButton->setToolTip(tr("Clear selection and reset the form for adding a new entry."));
        row->addWidget(_directionFieldAddButton);
        row->addWidget(_directionFieldRemoveButton);
        row->addWidget(_directionFieldClearButton);
        row->addStretch(1);
    });

    _directionFieldList = new QListWidget(directionParent);
    _directionFieldList->setToolTip(tr("Direction field configurations applied during growth."));
    _directionFieldList->setSelectionMode(QAbstractItemView::SingleSelection);
    _groupDirectionField->addFullWidthWidget(_directionFieldList);

    panelLayout->addWidget(_groupDirectionField);
}
