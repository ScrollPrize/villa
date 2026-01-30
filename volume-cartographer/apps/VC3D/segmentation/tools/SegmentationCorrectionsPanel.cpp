#include "SegmentationCorrectionsPanel.hpp"

#include "VCSettings.hpp"

#include <QCheckBox>
#include <QComboBox>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QSettings>
#include <QSignalBlocker>
#include <QVBoxLayout>
#include <QVariant>

SegmentationCorrectionsPanel::SegmentationCorrectionsPanel(const QString& settingsGroup,
                                                           QWidget* parent)
    : QWidget(parent)
    , _settingsGroup(settingsGroup)
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

    // --- Signal wiring ---

    connect(_comboCorrections, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index) {
        if (index < 0) {
            emit correctionsCollectionSelected(0);
            return;
        }
        const QVariant data = _comboCorrections->itemData(index);
        emit correctionsCollectionSelected(data.toULongLong());
    });

    connect(_btnCorrectionsNew, &QPushButton::clicked, this, [this]() {
        emit correctionsCreateRequested();
    });

    connect(_chkCorrectionsAnnotate, &QCheckBox::toggled, this, [this](bool enabled) {
        emit correctionsAnnotateToggled(enabled);
    });
}

void SegmentationCorrectionsPanel::writeSetting(const QString& key, const QVariant& value)
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.beginGroup(_settingsGroup);
    settings.setValue(key, value);
    settings.endGroup();
}

void SegmentationCorrectionsPanel::setCorrectionsEnabled(bool enabled)
{
    if (_correctionsEnabled == enabled) {
        return;
    }
    _correctionsEnabled = enabled;
    writeSetting(QStringLiteral("corrections_enabled"), _correctionsEnabled);
    if (!enabled) {
        _correctionsAnnotateChecked = false;
        if (_chkCorrectionsAnnotate) {
            const QSignalBlocker blocker(_chkCorrectionsAnnotate);
            _chkCorrectionsAnnotate->setChecked(false);
        }
    }
}

void SegmentationCorrectionsPanel::setCorrectionsAnnotateChecked(bool enabled)
{
    _correctionsAnnotateChecked = enabled;
    if (_chkCorrectionsAnnotate) {
        const QSignalBlocker blocker(_chkCorrectionsAnnotate);
        _chkCorrectionsAnnotate->setChecked(enabled);
    }
}

void SegmentationCorrectionsPanel::setCorrectionCollections(
    const QVector<QPair<uint64_t, QString>>& collections,
    std::optional<uint64_t> activeId)
{
    if (!_comboCorrections) {
        return;
    }
    const QSignalBlocker blocker(_comboCorrections);
    _comboCorrections->clear();
    for (const auto& pair : collections) {
        _comboCorrections->addItem(pair.second, QVariant::fromValue(static_cast<qulonglong>(pair.first)));
    }
    if (activeId) {
        int idx = _comboCorrections->findData(QVariant::fromValue(static_cast<qulonglong>(*activeId)));
        if (idx >= 0) {
            _comboCorrections->setCurrentIndex(idx);
        }
    } else {
        _comboCorrections->setCurrentIndex(-1);
    }
}

void SegmentationCorrectionsPanel::restoreSettings(QSettings& settings)
{
    using namespace vc3d::settings;
    _restoringSettings = true;

    _correctionsEnabled = settings.value(segmentation::CORRECTIONS_ENABLED, segmentation::CORRECTIONS_ENABLED_DEFAULT).toBool();

    _restoringSettings = false;
}

void SegmentationCorrectionsPanel::syncUiState(bool editingEnabled, bool growthInProgress)
{
    const bool allowCorrections = editingEnabled && _correctionsEnabled && !growthInProgress;

    if (_groupCorrections) {
        _groupCorrections->setEnabled(allowCorrections);
    }
    if (_comboCorrections) {
        const QSignalBlocker blocker(_comboCorrections);
        _comboCorrections->setEnabled(allowCorrections && _comboCorrections->count() > 0);
    }
    if (_btnCorrectionsNew) {
        _btnCorrectionsNew->setEnabled(editingEnabled && !growthInProgress);
    }
    if (_chkCorrectionsAnnotate) {
        _chkCorrectionsAnnotate->setEnabled(allowCorrections);
        const QSignalBlocker blocker(_chkCorrectionsAnnotate);
        _chkCorrectionsAnnotate->setChecked(_correctionsAnnotateChecked);
    }
}
