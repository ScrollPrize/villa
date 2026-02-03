#include "SegmentationHeaderRow.hpp"

#include <QCheckBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QSignalBlocker>

SegmentationHeaderRow::SegmentationHeaderRow(QWidget* parent)
    : QWidget(parent)
{
    auto* layout = new QHBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);

    _chkEditing = new QCheckBox(tr("Enable editing"), this);
    _chkEditing->setToolTip(tr("Start or stop segmentation editing so brush tools can modify surfaces."));

    _lblStatus = new QLabel(this);
    _lblStatus->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);

    layout->addWidget(_chkEditing);
    layout->addSpacing(8);
    layout->addWidget(_lblStatus, 1);

    connect(_chkEditing, &QCheckBox::toggled, this, &SegmentationHeaderRow::editingToggled);
}

void SegmentationHeaderRow::setEditingChecked(bool checked)
{
    if (!_chkEditing) {
        return;
    }
    const QSignalBlocker blocker(_chkEditing);
    _chkEditing->setChecked(checked);
}

bool SegmentationHeaderRow::isEditingChecked() const
{
    return _chkEditing && _chkEditing->isChecked();
}

void SegmentationHeaderRow::setStatusText(const QString& text)
{
    if (_lblStatus) {
        _lblStatus->setText(text);
    }
}
