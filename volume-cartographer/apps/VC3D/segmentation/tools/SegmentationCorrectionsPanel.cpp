#include "SegmentationCorrectionsPanel.hpp"

#include <QCheckBox>
#include <QComboBox>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>

SegmentationCorrectionsPanel::SegmentationCorrectionsPanel(QWidget* parent)
    : QWidget(parent)
{
    auto* panelLayout = new QVBoxLayout(this);
    panelLayout->setContentsMargins(0, 0, 0, 0);
    panelLayout->setSpacing(0);

    _groupCorrections = new QGroupBox(tr("Corrections"), this);
    auto* correctionsLayout = new QVBoxLayout(_groupCorrections);

    auto* correctionsComboRow = new QHBoxLayout();
    auto* correctionsLabel = new QLabel(tr("Active set:"), _groupCorrections);
    _comboCorrections = new QComboBox(_groupCorrections);
    _comboCorrections->setEnabled(false);
    _comboCorrections->setToolTip(tr("Choose an existing correction set to apply."));
    correctionsComboRow->addWidget(correctionsLabel);
    correctionsComboRow->addStretch(1);
    correctionsComboRow->addWidget(_comboCorrections, 1);
    correctionsLayout->addLayout(correctionsComboRow);

    _btnCorrectionsNew = new QPushButton(tr("New correction set"), _groupCorrections);
    _btnCorrectionsNew->setToolTip(tr("Create a new, empty correction set for this segmentation."));
    correctionsLayout->addWidget(_btnCorrectionsNew);

    _chkCorrectionsAnnotate = new QCheckBox(tr("Annotate corrections"), _groupCorrections);
    _chkCorrectionsAnnotate->setToolTip(tr("Toggle annotation overlay while reviewing corrections."));
    correctionsLayout->addWidget(_chkCorrectionsAnnotate);

    _groupCorrections->setLayout(correctionsLayout);
    panelLayout->addWidget(_groupCorrections);
}
