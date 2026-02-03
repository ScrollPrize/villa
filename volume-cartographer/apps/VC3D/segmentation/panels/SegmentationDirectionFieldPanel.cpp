#include "SegmentationDirectionFieldPanel.hpp"

#include "VCSettings.hpp"
#include "elements/CollapsibleSettingsGroup.hpp"

#include <QAbstractItemView>
#include <QComboBox>
#include <QDir>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QPushButton>
#include <QScrollBar>
#include <QSettings>
#include <QSignalBlocker>
#include <QToolButton>
#include <QVBoxLayout>
#include <QVariant>

#include <algorithm>
#include <cmath>
#include <utility>

namespace
{
constexpr int kCompactDirectionFieldRowLimit = 3;
} // namespace

SegmentationDirectionFieldPanel::SegmentationDirectionFieldPanel(const QString& settingsGroup,
                                                                 QWidget* parent)
    : QWidget(parent)
    , _settingsGroup(settingsGroup)
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

    // --- Signal wiring ---

    connect(_directionFieldPathEdit, &QLineEdit::textChanged, this, [this](const QString& text) {
        _directionFieldPath = text.trimmed();
        if (!_updatingDirectionFieldForm) {
            applyDirectionFieldDraftToSelection(_directionFieldList ? _directionFieldList->currentRow() : -1);
        }
    });

    connect(_directionFieldBrowseButton, &QToolButton::clicked, this, [this]() {
        const QString initial = _directionFieldPath.isEmpty() ? QDir::homePath() : _directionFieldPath;
        const QString dir = QFileDialog::getExistingDirectory(this, tr("Select direction field"), initial);
        if (dir.isEmpty()) {
            return;
        }
        _directionFieldPath = dir;
        _directionFieldPathEdit->setText(dir);
    });

    connect(_comboDirectionFieldOrientation, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index) {
        _directionFieldOrientation = segmentationDirectionFieldOrientationFromInt(
            _comboDirectionFieldOrientation->itemData(index).toInt());
        if (!_updatingDirectionFieldForm) {
            applyDirectionFieldDraftToSelection(_directionFieldList ? _directionFieldList->currentRow() : -1);
        }
    });

    connect(_comboDirectionFieldScale, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index) {
        _directionFieldScale = _comboDirectionFieldScale->itemData(index).toInt();
        if (!_updatingDirectionFieldForm) {
            applyDirectionFieldDraftToSelection(_directionFieldList ? _directionFieldList->currentRow() : -1);
        }
    });

    connect(_spinDirectionFieldWeight, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        _directionFieldWeight = value;
        if (!_updatingDirectionFieldForm) {
            applyDirectionFieldDraftToSelection(_directionFieldList ? _directionFieldList->currentRow() : -1);
        }
    });

    connect(_directionFieldAddButton, &QPushButton::clicked, this, [this]() {
        auto config = buildDirectionFieldDraft();
        if (!config.isValid()) {
            qCInfo(lcSegWidget) << "Ignoring direction field add; path empty";
            return;
        }
        _directionFields.push_back(std::move(config));
        refreshDirectionFieldList();
        persistDirectionFields();
        clearDirectionFieldForm();
    });

    connect(_directionFieldRemoveButton, &QPushButton::clicked, this, [this]() {
        const int row = _directionFieldList ? _directionFieldList->currentRow() : -1;
        if (row < 0 || row >= static_cast<int>(_directionFields.size())) {
            return;
        }
        _directionFields.erase(_directionFields.begin() + row);
        refreshDirectionFieldList();
        persistDirectionFields();
    });

    connect(_directionFieldClearButton, &QPushButton::clicked, this, [this]() {
        clearDirectionFieldForm();
    });

    connect(_directionFieldList, &QListWidget::currentRowChanged, this, [this](int row) {
        updateDirectionFieldFormFromSelection(row);
        if (_directionFieldRemoveButton) {
            _directionFieldRemoveButton->setEnabled(_editingEnabled && row >= 0);
        }
    });

    // Remember group expand/collapse state
    connect(_groupDirectionField, &CollapsibleSettingsGroup::toggled, this, [this](bool expanded) {
        if (_restoringSettings) {
            return;
        }
        writeSetting(QStringLiteral("group_direction_field_expanded"), expanded);
    });
}

void SegmentationDirectionFieldPanel::writeSetting(const QString& key, const QVariant& value)
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.beginGroup(_settingsGroup);
    settings.setValue(key, value);
    settings.endGroup();
}

std::vector<SegmentationDirectionFieldConfig> SegmentationDirectionFieldPanel::directionFieldConfigs() const
{
    std::vector<SegmentationDirectionFieldConfig> configs;
    configs.reserve(_directionFields.size());
    for (const auto& config : _directionFields) {
        if (config.isValid()) {
            configs.push_back(config);
        }
    }
    return configs;
}

SegmentationDirectionFieldConfig SegmentationDirectionFieldPanel::buildDirectionFieldDraft() const
{
    SegmentationDirectionFieldConfig config;
    config.path = _directionFieldPath.trimmed();
    config.orientation = _directionFieldOrientation;
    config.scale = std::clamp(_directionFieldScale, 0, 5);
    config.weight = std::clamp(_directionFieldWeight, 0.0, 10.0);
    return config;
}

void SegmentationDirectionFieldPanel::refreshDirectionFieldList()
{
    if (!_directionFieldList) {
        return;
    }
    const QSignalBlocker blocker(_directionFieldList);
    const int previousRow = _directionFieldList->currentRow();
    _directionFieldList->clear();

    for (const auto& config : _directionFields) {
        QString orientationLabel = segmentationDirectionFieldOrientationKey(config.orientation);
        const QString weightText = QString::number(std::clamp(config.weight, 0.0, 10.0), 'f', 2);
        const QString itemText = tr("%1 — %2 (scale %3, weight %4)")
                                     .arg(config.path,
                                          orientationLabel,
                                          QString::number(std::clamp(config.scale, 0, 5)),
                                          weightText);
        auto* item = new QListWidgetItem(itemText, _directionFieldList);
        item->setToolTip(config.path);
    }

    if (!_directionFields.empty()) {
        const int clampedRow = std::clamp(previousRow, 0, static_cast<int>(_directionFields.size()) - 1);
        _directionFieldList->setCurrentRow(clampedRow);
    }
    if (_directionFieldRemoveButton) {
        _directionFieldRemoveButton->setEnabled(_editingEnabled && !_directionFields.empty() && _directionFieldList->currentRow() >= 0);
    }

    updateDirectionFieldFormFromSelection(_directionFieldList->currentRow());
    updateDirectionFieldListGeometry();
}

void SegmentationDirectionFieldPanel::updateDirectionFieldFormFromSelection(int row)
{
    const bool previousUpdating = _updatingDirectionFieldForm;
    _updatingDirectionFieldForm = true;

    if (row >= 0 && row < static_cast<int>(_directionFields.size())) {
        const auto& config = _directionFields[static_cast<std::size_t>(row)];
        _directionFieldPath = config.path;
        _directionFieldOrientation = config.orientation;
        _directionFieldScale = config.scale;
        _directionFieldWeight = config.weight;
    }

    if (_directionFieldPathEdit) {
        const QSignalBlocker blocker(_directionFieldPathEdit);
        _directionFieldPathEdit->setText(_directionFieldPath);
    }
    if (_comboDirectionFieldOrientation) {
        const QSignalBlocker blocker(_comboDirectionFieldOrientation);
        int idx = _comboDirectionFieldOrientation->findData(static_cast<int>(_directionFieldOrientation));
        if (idx >= 0) {
            _comboDirectionFieldOrientation->setCurrentIndex(idx);
        }
    }
    if (_comboDirectionFieldScale) {
        const QSignalBlocker blocker(_comboDirectionFieldScale);
        int idx = _comboDirectionFieldScale->findData(_directionFieldScale);
        if (idx >= 0) {
            _comboDirectionFieldScale->setCurrentIndex(idx);
        }
    }
    if (_spinDirectionFieldWeight) {
        const QSignalBlocker blocker(_spinDirectionFieldWeight);
        _spinDirectionFieldWeight->setValue(_directionFieldWeight);
    }

    _updatingDirectionFieldForm = previousUpdating;
}

void SegmentationDirectionFieldPanel::applyDirectionFieldDraftToSelection(int row)
{
    if (row < 0 || row >= static_cast<int>(_directionFields.size())) {
        return;
    }

    auto config = buildDirectionFieldDraft();
    if (!config.isValid()) {
        return;
    }

    auto& target = _directionFields[static_cast<std::size_t>(row)];
    if (target.path == config.path &&
        target.orientation == config.orientation &&
        target.scale == config.scale &&
        std::abs(target.weight - config.weight) < 1e-4) {
        return;
    }

    target = std::move(config);
    updateDirectionFieldListItem(row);
    persistDirectionFields();
}

void SegmentationDirectionFieldPanel::updateDirectionFieldListItem(int row)
{
    if (!_directionFieldList) {
        return;
    }
    if (row < 0 || row >= _directionFieldList->count()) {
        return;
    }
    if (row >= static_cast<int>(_directionFields.size())) {
        return;
    }

    const auto& config = _directionFields[static_cast<std::size_t>(row)];
    QString orientationLabel = segmentationDirectionFieldOrientationKey(config.orientation);
    const QString weightText = QString::number(std::clamp(config.weight, 0.0, 10.0), 'f', 2);
    const QString itemText = tr("%1 — %2 (scale %3, weight %4)")
                                 .arg(config.path,
                                      orientationLabel,
                                      QString::number(std::clamp(config.scale, 0, 5)),
                                      weightText);

    if (auto* item = _directionFieldList->item(row)) {
        item->setText(itemText);
        item->setToolTip(config.path);
    }
}

void SegmentationDirectionFieldPanel::updateDirectionFieldListGeometry()
{
    if (!_directionFieldList) {
        return;
    }

    auto policy = _directionFieldList->sizePolicy();
    const int itemCount = _directionFieldList->count();

    if (itemCount <= kCompactDirectionFieldRowLimit) {
        const int sampleRowHeight = _directionFieldList->sizeHintForRow(0);
        const int rowHeight = sampleRowHeight > 0 ? sampleRowHeight : _directionFieldList->fontMetrics().height() + 8;
        const int visibleRows = std::max(1, itemCount);
        const int frameHeight = 2 * _directionFieldList->frameWidth();
        const auto* hScroll = _directionFieldList->horizontalScrollBar();
        const int scrollHeight = (hScroll && hScroll->isVisible()) ? hScroll->sizeHint().height() : 0;
        const int targetHeight = rowHeight * visibleRows + frameHeight + scrollHeight;

        policy.setVerticalPolicy(QSizePolicy::Fixed);
        policy.setVerticalStretch(0);
        _directionFieldList->setSizePolicy(policy);
        _directionFieldList->setMinimumHeight(targetHeight);
        _directionFieldList->setMaximumHeight(targetHeight);
    } else {
        policy.setVerticalPolicy(QSizePolicy::Expanding);
        policy.setVerticalStretch(1);
        _directionFieldList->setSizePolicy(policy);
        _directionFieldList->setMinimumHeight(0);
        _directionFieldList->setMaximumHeight(QWIDGETSIZE_MAX);
    }

    _directionFieldList->updateGeometry();
}

void SegmentationDirectionFieldPanel::clearDirectionFieldForm()
{
    // Clear the list selection
    if (_directionFieldList) {
        _directionFieldList->setCurrentRow(-1);
    }

    // Reset member variables to defaults
    _directionFieldPath.clear();
    _directionFieldOrientation = SegmentationDirectionFieldOrientation::Normal;
    _directionFieldScale = 0;
    _directionFieldWeight = 1.0;

    // Update the form fields to reflect the cleared state
    const bool previousUpdating = _updatingDirectionFieldForm;
    _updatingDirectionFieldForm = true;

    if (_directionFieldPathEdit) {
        _directionFieldPathEdit->clear();
    }
    if (_comboDirectionFieldOrientation) {
        int idx = _comboDirectionFieldOrientation->findData(static_cast<int>(SegmentationDirectionFieldOrientation::Normal));
        if (idx >= 0) {
            _comboDirectionFieldOrientation->setCurrentIndex(idx);
        }
    }
    if (_comboDirectionFieldScale) {
        int idx = _comboDirectionFieldScale->findData(0);
        if (idx >= 0) {
            _comboDirectionFieldScale->setCurrentIndex(idx);
        }
    }
    if (_spinDirectionFieldWeight) {
        _spinDirectionFieldWeight->setValue(1.0);
    }

    _updatingDirectionFieldForm = previousUpdating;

    // Update button states
    if (_directionFieldRemoveButton) {
        _directionFieldRemoveButton->setEnabled(false);
    }
}

void SegmentationDirectionFieldPanel::persistDirectionFields()
{
    QVariantList serialized;
    serialized.reserve(static_cast<int>(_directionFields.size()));
    for (const auto& config : _directionFields) {
        QVariantMap map;
        map.insert(QStringLiteral("path"), config.path);
        map.insert(QStringLiteral("orientation"), static_cast<int>(config.orientation));
        map.insert(QStringLiteral("scale"), std::clamp(config.scale, 0, 5));
        map.insert(QStringLiteral("weight"), std::clamp(config.weight, 0.0, 10.0));
        serialized.push_back(map);
    }
    writeSetting(QStringLiteral("direction_fields"), serialized);
}

void SegmentationDirectionFieldPanel::restoreSettings(QSettings& settings)
{
    using namespace vc3d::settings;
    _restoringSettings = true;

    QVariantList serialized = settings.value(segmentation::DIRECTION_FIELDS, QVariantList{}).toList();
    _directionFields.clear();
    for (const QVariant& entry : serialized) {
        const QVariantMap map = entry.toMap();
        SegmentationDirectionFieldConfig config;
        config.path = map.value(QStringLiteral("path")).toString();
        config.orientation = segmentationDirectionFieldOrientationFromInt(
            map.value(QStringLiteral("orientation"), 0).toInt());
        config.scale = map.value(QStringLiteral("scale"), 0).toInt();
        config.weight = map.value(QStringLiteral("weight"), 1.0).toDouble();
        if (config.isValid()) {
            _directionFields.push_back(std::move(config));
        }
    }

    const bool directionExpanded = settings.value(segmentation::GROUP_DIRECTION_FIELD_EXPANDED, segmentation::GROUP_DIRECTION_FIELD_EXPANDED_DEFAULT).toBool();
    if (_groupDirectionField) {
        _groupDirectionField->setExpanded(directionExpanded);
    }

    _restoringSettings = false;
}

void SegmentationDirectionFieldPanel::syncUiState(bool editingEnabled)
{
    _editingEnabled = editingEnabled;

    refreshDirectionFieldList();

    if (_directionFieldPathEdit) {
        const QSignalBlocker blocker(_directionFieldPathEdit);
        _directionFieldPathEdit->setText(_directionFieldPath);
    }
    if (_comboDirectionFieldOrientation) {
        const QSignalBlocker blocker(_comboDirectionFieldOrientation);
        int idx = _comboDirectionFieldOrientation->findData(static_cast<int>(_directionFieldOrientation));
        if (idx >= 0) {
            _comboDirectionFieldOrientation->setCurrentIndex(idx);
        }
    }
    if (_comboDirectionFieldScale) {
        const QSignalBlocker blocker(_comboDirectionFieldScale);
        int idx = _comboDirectionFieldScale->findData(_directionFieldScale);
        if (idx >= 0) {
            _comboDirectionFieldScale->setCurrentIndex(idx);
        }
    }
    if (_spinDirectionFieldWeight) {
        const QSignalBlocker blocker(_spinDirectionFieldWeight);
        _spinDirectionFieldWeight->setValue(_directionFieldWeight);
    }

    if (_directionFieldAddButton) {
        _directionFieldAddButton->setEnabled(_editingEnabled);
    }
    if (_directionFieldRemoveButton) {
        const bool hasSelection = _directionFieldList && _directionFieldList->currentRow() >= 0;
        _directionFieldRemoveButton->setEnabled(_editingEnabled && hasSelection);
    }
    if (_directionFieldList) {
        _directionFieldList->setEnabled(_editingEnabled);
    }
}
