#include "SegmentationHeaderRow.hpp"

#include <QCheckBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QSignalBlocker>
#include <QVBoxLayout>

SegmentationHeaderRow::SegmentationHeaderRow(QWidget* parent)
    : QWidget(parent)
{
    auto* layout = new QHBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);

    _chkAnnotate = new QCheckBox(tr("Annotate"), this);
    _chkAnnotate->setChecked(true);
    _chkAnnotate->setToolTip(tr("Toggle annotation mode for placing correction points on surfaces."));

    auto* checkLayout = new QVBoxLayout();
    checkLayout->setContentsMargins(0, 0, 0, 0);
    checkLayout->setSpacing(2);

    _chkEditing = new QCheckBox(tr("Enable editing"), this);
    _chkEditing->setToolTip(tr("Start or stop segmentation editing so brush tools can modify surfaces."));

    _chkDrawMask = new QCheckBox(tr("Draw mask"), this);
    _chkDrawMask->setToolTip(tr("Right-drag on the surface view to mark grid cells invalid in the surface."));

    _lblStatus = new QLabel(this);
    _lblStatus->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);

    checkLayout->addWidget(_chkAnnotate);
    checkLayout->addWidget(_chkEditing);
    checkLayout->addWidget(_chkDrawMask);
    layout->addLayout(checkLayout);
    layout->addSpacing(8);
    layout->addWidget(_lblStatus, 1);

    connect(_chkAnnotate, &QCheckBox::toggled, this, &SegmentationHeaderRow::annotateToggled);
    connect(_chkEditing, &QCheckBox::toggled, this, &SegmentationHeaderRow::editingToggled);
    connect(_chkDrawMask, &QCheckBox::toggled, this, &SegmentationHeaderRow::drawMaskToggled);
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

void SegmentationHeaderRow::setAnnotateChecked(bool checked)
{
    if (!_chkAnnotate) {
        return;
    }
    const QSignalBlocker blocker(_chkAnnotate);
    _chkAnnotate->setChecked(checked);
}

bool SegmentationHeaderRow::isAnnotateChecked() const
{
    return _chkAnnotate && _chkAnnotate->isChecked();
}

void SegmentationHeaderRow::setDrawMaskChecked(bool checked)
{
    if (!_chkDrawMask) {
        return;
    }
    const QSignalBlocker blocker(_chkDrawMask);
    _chkDrawMask->setChecked(checked);
}

bool SegmentationHeaderRow::isDrawMaskChecked() const
{
    return _chkDrawMask && _chkDrawMask->isChecked();
}

void SegmentationHeaderRow::setStatusText(const QString& text)
{
    if (_lblStatus) {
        _lblStatus->setText(text);
    }
}
