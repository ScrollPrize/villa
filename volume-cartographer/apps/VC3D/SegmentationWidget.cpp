#include "SegmentationWidget.hpp"

#include <QAbstractItemView>
#include <QByteArray>
#include <QCheckBox>
#include <QComboBox>
#include <QDir>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QGroupBox>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QListWidgetItem>
#include <QLoggingCategory>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QRegularExpression>
#include <QScrollBar>
#include <QSettings>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QToolButton>
#include <QVariant>
#include <QVBoxLayout>
#include <QHBoxLayout>

#include <algorithm>
#include <cmath>
#include <exception>

#include <nlohmann/json.hpp>

namespace
{
Q_LOGGING_CATEGORY(lcSegWidget, "vc.segmentation.widget")

constexpr int kGrowDirUpBit = 1 << 0;
constexpr int kGrowDirDownBit = 1 << 1;
constexpr int kGrowDirLeftBit = 1 << 2;
constexpr int kGrowDirRightBit = 1 << 3;
constexpr int kGrowDirAllMask = kGrowDirUpBit | kGrowDirDownBit | kGrowDirLeftBit | kGrowDirRightBit;
constexpr int kCompactDirectionFieldRowLimit = 3;

bool containsSurfKeyword(const QString& text)
{
    if (text.isEmpty()) {
        return false;
    }
    const QString lowered = text.toLower();
    return lowered.contains(QStringLiteral("surface")) || lowered.contains(QStringLiteral("surf"));
}

std::optional<int> trailingNumber(const QString& text)
{
    static const QRegularExpression numberSuffix(QStringLiteral("(\\d+)$"));
    const auto match = numberSuffix.match(text.trimmed());
    if (match.hasMatch()) {
        return match.captured(1).toInt();
    }
    return std::nullopt;
}

QString settingsGroup()
{
    return QStringLiteral("segmentation_edit");
}
}

QString SegmentationWidget::determineDefaultVolumeId(const QVector<QPair<QString, QString>>& volumes,
                                                     const QString& requestedId) const
{
    const auto hasId = [&volumes](const QString& id) {
        return std::any_of(volumes.cbegin(), volumes.cend(), [&](const auto& entry) {
            return entry.first == id;
        });
    };

    QString numericCandidate;
    int numericValue = -1;
    QString keywordCandidate;

    for (const auto& entry : volumes) {
        const QString& id = entry.first;
        const QString& label = entry.second;

        if (!containsSurfKeyword(id) && !containsSurfKeyword(label)) {
            continue;
        }

        const auto numberFromId = trailingNumber(id);
        const auto numberFromLabel = trailingNumber(label);
        const std::optional<int> number = numberFromId ? numberFromId : numberFromLabel;

        if (number) {
            if (*number > numericValue) {
                numericValue = *number;
                numericCandidate = id;
            }
        } else if (keywordCandidate.isEmpty()) {
            keywordCandidate = id;
        }
    }

    if (!numericCandidate.isEmpty()) {
        return numericCandidate;
    }
    if (!keywordCandidate.isEmpty()) {
        return keywordCandidate;
    }
    if (!requestedId.isEmpty() && hasId(requestedId)) {
        return requestedId;
    }
    if (!volumes.isEmpty()) {
        return volumes.front().first;
    }
    return {};
}

SegmentationWidget::SegmentationWidget(QWidget* parent)
    : QWidget(parent)
{
    _growthDirectionMask = kGrowDirAllMask;
    buildUi();
    restoreSettings();
    syncUiState();
}

void SegmentationWidget::buildUi()
{
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(8, 8, 8, 8);
    layout->setSpacing(12);

    auto* editingRow = new QHBoxLayout();
    _chkEditing = new QCheckBox(tr("Enable editing"), this);
    _lblStatus = new QLabel(this);
    _lblStatus->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    editingRow->addWidget(_chkEditing);
    editingRow->addSpacing(8);
    editingRow->addWidget(_lblStatus, 1);
    layout->addLayout(editingRow);

    auto* brushRow = new QHBoxLayout();
    brushRow->addSpacing(4);
    _chkEraseBrush = new QCheckBox(tr("Invalidation brush (Shift)"), this);
    _chkEraseBrush->setEnabled(false);
    brushRow->addWidget(_chkEraseBrush);
    brushRow->addStretch(1);
    layout->addLayout(brushRow);

    _groupGrowth = new QGroupBox(tr("Surface Growth"), this);
    auto* growthLayout = new QVBoxLayout(_groupGrowth);

    auto* dirRow = new QHBoxLayout();
    auto* stepsLabel = new QLabel(tr("Steps:"), _groupGrowth);
    _spinGrowthSteps = new QSpinBox(_groupGrowth);
    _spinGrowthSteps->setRange(1, 1024);
    _spinGrowthSteps->setSingleStep(1);
    dirRow->addWidget(stepsLabel);
    dirRow->addWidget(_spinGrowthSteps);
    dirRow->addSpacing(16);

    auto* dirLabel = new QLabel(tr("Allowed directions:"), _groupGrowth);
    dirRow->addWidget(dirLabel);
    auto addDirectionCheckbox = [&](const QString& text) {
        auto* box = new QCheckBox(text, _groupGrowth);
        dirRow->addWidget(box);
        return box;
    };
    _chkGrowthDirUp = addDirectionCheckbox(tr("Up"));
    _chkGrowthDirDown = addDirectionCheckbox(tr("Down"));
    _chkGrowthDirLeft = addDirectionCheckbox(tr("Left"));
    _chkGrowthDirRight = addDirectionCheckbox(tr("Right"));
    dirRow->addStretch(1);
    growthLayout->addLayout(dirRow);

    auto* zRow = new QHBoxLayout();
    _chkCorrectionsUseZRange = new QCheckBox(tr("Limit Z range"), _groupGrowth);
    zRow->addWidget(_chkCorrectionsUseZRange);
    zRow->addSpacing(12);
    auto* zMinLabel = new QLabel(tr("Z min"), _groupGrowth);
    _spinCorrectionsZMin = new QSpinBox(_groupGrowth);
    _spinCorrectionsZMin->setRange(-100000, 100000);
    auto* zMaxLabel = new QLabel(tr("Z max"), _groupGrowth);
    _spinCorrectionsZMax = new QSpinBox(_groupGrowth);
    _spinCorrectionsZMax->setRange(-100000, 100000);
    zRow->addWidget(zMinLabel);
    zRow->addWidget(_spinCorrectionsZMin);
    zRow->addSpacing(8);
    zRow->addWidget(zMaxLabel);
    zRow->addWidget(_spinCorrectionsZMax);
    zRow->addStretch(1);
    growthLayout->addLayout(zRow);

    _btnGrow = new QPushButton(tr("Grow"), _groupGrowth);
    growthLayout->addWidget(_btnGrow);

    auto* volumeRow = new QHBoxLayout();
    auto* volumeLabel = new QLabel(tr("Volume:"), _groupGrowth);
    _comboVolumes = new QComboBox(_groupGrowth);
    _comboVolumes->setEnabled(false);
    volumeRow->addWidget(volumeLabel);
    volumeRow->addWidget(_comboVolumes, 1);
    growthLayout->addLayout(volumeRow);

    _groupGrowth->setLayout(growthLayout);
    layout->addWidget(_groupGrowth);

    _lblNormalGrid = new QLabel(this);
    _lblNormalGrid->setTextFormat(Qt::RichText);
    _lblNormalGrid->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    layout->addWidget(_lblNormalGrid);

    auto* falloffGroup = new QGroupBox(tr("Editing"), this);
    auto* falloffLayout = new QVBoxLayout(falloffGroup);

    auto* radiusSigmaRow = new QHBoxLayout();
    auto* radiusLabel = new QLabel(tr("Max radius"), falloffGroup);
    _spinRadius = new QDoubleSpinBox(falloffGroup);
    _spinRadius->setDecimals(2);
    _spinRadius->setRange(0.25, 128.0);
    _spinRadius->setSingleStep(0.25);
    radiusSigmaRow->addWidget(radiusLabel);
    radiusSigmaRow->addWidget(_spinRadius);
    radiusSigmaRow->addSpacing(12);
    auto* sigmaLabel = new QLabel(tr("Sigma"), falloffGroup);
    _spinSigma = new QDoubleSpinBox(falloffGroup);
    _spinSigma->setDecimals(2);
    _spinSigma->setRange(0.05, 64.0);
    _spinSigma->setSingleStep(0.1);
    radiusSigmaRow->addWidget(sigmaLabel);
    radiusSigmaRow->addWidget(_spinSigma);
    radiusSigmaRow->addStretch(1);
    falloffLayout->addLayout(radiusSigmaRow);

    auto* pushPullRow = new QHBoxLayout();
    auto* pushPullLabel = new QLabel(tr("Push/Pull step"), falloffGroup);
    _spinPushPullStep = new QDoubleSpinBox(falloffGroup);
    _spinPushPullStep->setDecimals(2);
    _spinPushPullStep->setRange(0.05, 10.0);
    _spinPushPullStep->setSingleStep(0.05);
    pushPullRow->addWidget(pushPullLabel);
    pushPullRow->addWidget(_spinPushPullStep);
    pushPullRow->addStretch(1);
    falloffLayout->addLayout(pushPullRow);

    falloffGroup->setLayout(falloffLayout);
    layout->addWidget(falloffGroup);

    _groupDirectionField = new QGroupBox(tr("Direction Fields"), this);
    auto* dfLayout = new QVBoxLayout(_groupDirectionField);

    auto* pathRow = new QHBoxLayout();
    auto* pathLabel = new QLabel(tr("Zarr folder:"), _groupDirectionField);
    _directionFieldPathEdit = new QLineEdit(_groupDirectionField);
    _directionFieldBrowseButton = new QToolButton(_groupDirectionField);
    _directionFieldBrowseButton->setText(QStringLiteral("..."));
    pathRow->addWidget(pathLabel);
    pathRow->addWidget(_directionFieldPathEdit, 1);
    pathRow->addWidget(_directionFieldBrowseButton);
    dfLayout->addLayout(pathRow);

    auto* orientationRow = new QHBoxLayout();
    auto* orientationLabel = new QLabel(tr("Orientation:"), _groupDirectionField);
    _comboDirectionFieldOrientation = new QComboBox(_groupDirectionField);
    _comboDirectionFieldOrientation->addItem(tr("Normal"), static_cast<int>(SegmentationDirectionFieldOrientation::Normal));
    _comboDirectionFieldOrientation->addItem(tr("Horizontal"), static_cast<int>(SegmentationDirectionFieldOrientation::Horizontal));
    _comboDirectionFieldOrientation->addItem(tr("Vertical"), static_cast<int>(SegmentationDirectionFieldOrientation::Vertical));
    orientationRow->addWidget(orientationLabel);
    orientationRow->addWidget(_comboDirectionFieldOrientation);
    orientationRow->addSpacing(12);
    auto* scaleLabel = new QLabel(tr("Scale level:"), _groupDirectionField);
    _comboDirectionFieldScale = new QComboBox(_groupDirectionField);
    for (int scale = 0; scale <= 5; ++scale) {
        _comboDirectionFieldScale->addItem(QString::number(scale), scale);
    }
    orientationRow->addWidget(scaleLabel);
    orientationRow->addWidget(_comboDirectionFieldScale);
    orientationRow->addSpacing(12);
    auto* weightLabel = new QLabel(tr("Weight:"), _groupDirectionField);
    _spinDirectionFieldWeight = new QDoubleSpinBox(_groupDirectionField);
    _spinDirectionFieldWeight->setDecimals(2);
    _spinDirectionFieldWeight->setRange(0.0, 10.0);
    _spinDirectionFieldWeight->setSingleStep(0.1);
    orientationRow->addWidget(weightLabel);
    orientationRow->addWidget(_spinDirectionFieldWeight);
    orientationRow->addStretch(1);
    dfLayout->addLayout(orientationRow);

    auto* buttonsRow = new QHBoxLayout();
    _directionFieldAddButton = new QPushButton(tr("Add"), _groupDirectionField);
    _directionFieldRemoveButton = new QPushButton(tr("Remove"), _groupDirectionField);
    _directionFieldRemoveButton->setEnabled(false);
    buttonsRow->addWidget(_directionFieldAddButton);
    buttonsRow->addWidget(_directionFieldRemoveButton);
    buttonsRow->addStretch(1);
    dfLayout->addLayout(buttonsRow);

    _directionFieldList = new QListWidget(_groupDirectionField);
    _directionFieldList->setSelectionMode(QAbstractItemView::SingleSelection);
    dfLayout->addWidget(_directionFieldList);

    _groupDirectionField->setLayout(dfLayout);
    layout->addWidget(_groupDirectionField);

    _groupCorrections = new QGroupBox(tr("Corrections"), this);
    auto* correctionsLayout = new QVBoxLayout(_groupCorrections);

    auto* correctionsComboRow = new QHBoxLayout();
    auto* correctionsLabel = new QLabel(tr("Active set:"), _groupCorrections);
    _comboCorrections = new QComboBox(_groupCorrections);
    _comboCorrections->setEnabled(false);
    correctionsComboRow->addWidget(correctionsLabel);
    correctionsComboRow->addStretch(1);
    correctionsComboRow->addWidget(_comboCorrections, 1);
    correctionsLayout->addLayout(correctionsComboRow);

    _btnCorrectionsNew = new QPushButton(tr("New correction set"), _groupCorrections);
    correctionsLayout->addWidget(_btnCorrectionsNew);

    _chkCorrectionsAnnotate = new QCheckBox(tr("Annotate corrections"), _groupCorrections);
    correctionsLayout->addWidget(_chkCorrectionsAnnotate);

    _groupCorrections->setLayout(correctionsLayout);
    layout->addWidget(_groupCorrections);

    _groupCustomParams = new QGroupBox(tr("Custom Params"), this);
    auto* customParamsLayout = new QVBoxLayout(_groupCustomParams);

    auto* customParamsDescription = new QLabel(
        tr("Additional JSON fields merge into the tracer params. Leave empty for defaults."), _groupCustomParams);
    customParamsDescription->setWordWrap(true);
    customParamsLayout->addWidget(customParamsDescription);

    _editCustomParams = new QPlainTextEdit(_groupCustomParams);
    _editCustomParams->setPlaceholderText(QStringLiteral("{\n    \"example_param\": 1\n}"));
    _editCustomParams->setTabChangesFocus(true);
    customParamsLayout->addWidget(_editCustomParams);

    _lblCustomParamsStatus = new QLabel(_groupCustomParams);
    _lblCustomParamsStatus->setWordWrap(true);
    _lblCustomParamsStatus->setVisible(false);
    _lblCustomParamsStatus->setStyleSheet(QStringLiteral("color: #c0392b;"));
    customParamsLayout->addWidget(_lblCustomParamsStatus);

    _groupCustomParams->setLayout(customParamsLayout);
    layout->addWidget(_groupCustomParams);

    auto* buttons = new QHBoxLayout();
    _btnApply = new QPushButton(tr("Apply"), this);
    _btnReset = new QPushButton(tr("Reset"), this);
    _btnStop = new QPushButton(tr("Stop tools"), this);
    buttons->addWidget(_btnApply);
    buttons->addWidget(_btnReset);
    buttons->addWidget(_btnStop);
    layout->addLayout(buttons);

    layout->addStretch(1);

    connect(_chkEditing, &QCheckBox::toggled, this, [this](bool enabled) {
        updateEditingState(enabled, true);
    });

    auto connectDirectionCheckbox = [this](QCheckBox* box) {
        if (!box) {
            return;
        }
        connect(box, &QCheckBox::toggled, this, [this, box](bool) {
            updateGrowthDirectionMaskFromUi(box);
        });
    };
    connectDirectionCheckbox(_chkGrowthDirUp);
    connectDirectionCheckbox(_chkGrowthDirDown);
    connectDirectionCheckbox(_chkGrowthDirLeft);
    connectDirectionCheckbox(_chkGrowthDirRight);

    connect(_spinGrowthSteps, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        value = std::clamp(value, 1, 1024);
        if (_growthSteps == value) {
            return;
        }
        _growthSteps = value;
        writeSetting(QStringLiteral("growth_steps"), _growthSteps);
    });

    connect(_btnGrow, &QPushButton::clicked, this, [this]() {
        const auto allowed = allowedGrowthDirections();
        SegmentationGrowthDirection direction = SegmentationGrowthDirection::All;
        if (allowed.size() == 1) {
            direction = allowed.front();
        }
        qCInfo(lcSegWidget) << "Grow pressed" << segmentationGrowthMethodToString(_growthMethod)
                            << segmentationGrowthDirectionToString(direction)
                            << "steps" << _growthSteps;
        emit growSurfaceRequested(_growthMethod, direction, _growthSteps);
    });

    connect(_comboVolumes, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index) {
        if (index < 0) {
            return;
        }
        const QString volumeId = _comboVolumes->itemData(index).toString();
        if (volumeId.isEmpty() || volumeId == _activeVolumeId) {
            return;
        }
        _activeVolumeId = volumeId;
        emit volumeSelectionChanged(volumeId);
    });

    connect(_spinRadius, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        setRadius(static_cast<float>(value));
        emit radiusChanged(_radiusSteps);
    });

    connect(_spinSigma, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        setSigma(static_cast<float>(value));
        emit sigmaChanged(_sigmaSteps);
    });

    connect(_spinPushPullStep, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        setPushPullStep(static_cast<float>(value));
        emit pushPullStepChanged(_pushPullStep);
    });

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

    connect(_directionFieldList, &QListWidget::currentRowChanged, this, [this](int row) {
        updateDirectionFieldFormFromSelection(row);
        if (_directionFieldRemoveButton) {
            _directionFieldRemoveButton->setEnabled(_editingEnabled && row >= 0);
        }
    });

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

    connect(_editCustomParams, &QPlainTextEdit::textChanged, this, [this]() {
        handleCustomParamsEdited();
    });

    connect(_chkCorrectionsAnnotate, &QCheckBox::toggled, this, [this](bool enabled) {
        emit correctionsAnnotateToggled(enabled);
    });

    connect(_chkCorrectionsUseZRange, &QCheckBox::toggled, this, [this](bool enabled) {
        _correctionsZRangeEnabled = enabled;
        writeSetting(QStringLiteral("corrections_z_range_enabled"), _correctionsZRangeEnabled);
        updateGrowthUiState();
        emit correctionsZRangeChanged(enabled, _correctionsZMin, _correctionsZMax);
    });

    connect(_spinCorrectionsZMin, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        if (_correctionsZMin == value) {
            return;
        }
        _correctionsZMin = value;
        writeSetting(QStringLiteral("corrections_z_min"), _correctionsZMin);
        if (_correctionsZRangeEnabled) {
            emit correctionsZRangeChanged(true, _correctionsZMin, _correctionsZMax);
        }
    });

    connect(_spinCorrectionsZMax, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        if (_correctionsZMax == value) {
            return;
        }
        _correctionsZMax = value;
        writeSetting(QStringLiteral("corrections_z_max"), _correctionsZMax);
        if (_correctionsZRangeEnabled) {
            emit correctionsZRangeChanged(true, _correctionsZMin, _correctionsZMax);
        }
    });

    connect(_btnApply, &QPushButton::clicked, this, &SegmentationWidget::applyRequested);
    connect(_btnReset, &QPushButton::clicked, this, &SegmentationWidget::resetRequested);
    connect(_btnStop, &QPushButton::clicked, this, &SegmentationWidget::stopToolsRequested);
}

void SegmentationWidget::syncUiState()
{
    if (_chkEditing) {
        const QSignalBlocker blocker(_chkEditing);
        _chkEditing->setChecked(_editingEnabled);
    }

    if (_lblStatus) {
        if (_editingEnabled) {
            _lblStatus->setText(_pending ? tr("Editing enabled – pending changes")
                                         : tr("Editing enabled"));
        } else {
            _lblStatus->setText(tr("Editing disabled"));
        }
    }

    if (_chkEraseBrush) {
        const QSignalBlocker blocker(_chkEraseBrush);
        _chkEraseBrush->setChecked(_eraseBrushActive);
        _chkEraseBrush->setEnabled(_editingEnabled);
    }

    if (_spinRadius) {
        const QSignalBlocker blocker(_spinRadius);
        _spinRadius->setValue(static_cast<double>(_radiusSteps));
    }
    if (_spinSigma) {
        const QSignalBlocker blocker(_spinSigma);
        _spinSigma->setValue(static_cast<double>(_sigmaSteps));
    }
    if (_spinPushPullStep) {
        const QSignalBlocker blocker(_spinPushPullStep);
        _spinPushPullStep->setValue(static_cast<double>(_pushPullStep));
    }

    if (_editCustomParams) {
        if (_editCustomParams->toPlainText() != _customParamsText) {
            const QSignalBlocker blocker(_editCustomParams);
            _editCustomParams->setPlainText(_customParamsText);
        }
    }
    updateCustomParamsStatus();

    if (_spinGrowthSteps) {
        const QSignalBlocker blocker(_spinGrowthSteps);
        _spinGrowthSteps->setValue(_growthSteps);
    }

    applyGrowthDirectionMaskToUi();
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

    if (_comboCorrections) {
        const QSignalBlocker blocker(_comboCorrections);
        _comboCorrections->setEnabled(_correctionsEnabled && !_growthInProgress && _comboCorrections->count() > 0);
    }
    if (_chkCorrectionsAnnotate) {
        const QSignalBlocker blocker(_chkCorrectionsAnnotate);
        _chkCorrectionsAnnotate->setChecked(_correctionsAnnotateChecked);
    }
    if (_chkCorrectionsUseZRange) {
        const QSignalBlocker blocker(_chkCorrectionsUseZRange);
        _chkCorrectionsUseZRange->setChecked(_correctionsZRangeEnabled);
    }
    if (_spinCorrectionsZMin) {
        const QSignalBlocker blocker(_spinCorrectionsZMin);
        _spinCorrectionsZMin->setValue(_correctionsZMin);
    }
    if (_spinCorrectionsZMax) {
        const QSignalBlocker blocker(_spinCorrectionsZMax);
        _spinCorrectionsZMax->setValue(_correctionsZMax);
    }

    if (_lblNormalGrid) {
        const QString icon = _normalGridAvailable
            ? QStringLiteral("<span style=\"color:#2e7d32; font-size:16px;\">&#10003;</span>")
            : QStringLiteral("<span style=\"color:#c62828; font-size:16px;\">&#10007;</span>");
        const bool hasExplicitLocation = !_normalGridDisplayPath.isEmpty() && _normalGridDisplayPath != _normalGridHint;
        QString message;
        if (hasExplicitLocation) {
            message = _normalGridAvailable
                ? tr("Normal grids found at %1").arg(_normalGridDisplayPath)
                : tr("Normal grids not found at %1").arg(_normalGridDisplayPath);
        } else {
            message = _normalGridAvailable ? tr("Normal grids found.") : tr("Normal grids not found.");
            if (!_normalGridHint.isEmpty()) {
                message.append(QStringLiteral(" ("));
                message.append(_normalGridHint);
                message.append(QLatin1Char(')'));
            }
        }

        QString tooltip = message;
        if (hasExplicitLocation && !_normalGridHint.isEmpty()) {
            tooltip.append(QStringLiteral("\n"));
            tooltip.append(_normalGridHint);
        }
        if (!_volumePackagePath.isEmpty()) {
            tooltip.append(QStringLiteral("\n"));
            tooltip.append(tr("Volume package: %1").arg(_volumePackagePath));
        }

        _lblNormalGrid->setText(icon + QStringLiteral("&nbsp;") + message);
        _lblNormalGrid->setToolTip(tooltip);
        _lblNormalGrid->setAccessibleDescription(message);
    }

    updateGrowthUiState();
}

void SegmentationWidget::restoreSettings()
{
    QSettings settings(QStringLiteral("VC.ini"), QSettings::IniFormat);
    settings.beginGroup(settingsGroup());

    _radiusSteps = settings.value(QStringLiteral("radius_steps"), _radiusSteps).toFloat();
    _sigmaSteps = settings.value(QStringLiteral("sigma_steps"), _sigmaSteps).toFloat();
    _pushPullStep = settings.value(QStringLiteral("push_pull_step"), _pushPullStep).toFloat();
    _growthMethod = segmentationGrowthMethodFromInt(
        settings.value(QStringLiteral("growth_method"), static_cast<int>(_growthMethod)).toInt());
    _growthSteps = settings.value(QStringLiteral("growth_steps"), _growthSteps).toInt();
    _growthSteps = std::clamp(_growthSteps, 1, 1024);
    _growthDirectionMask = normalizeGrowthDirectionMask(
        settings.value(QStringLiteral("growth_direction_mask"), kGrowDirAllMask).toInt());

    QVariantList serialized = settings.value(QStringLiteral("direction_fields"), QVariantList{}).toList();
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

    _correctionsEnabled = settings.value(QStringLiteral("corrections_enabled"), false).toBool();
    _correctionsZRangeEnabled = settings.value(QStringLiteral("corrections_z_range_enabled"), false).toBool();
    _correctionsZMin = settings.value(QStringLiteral("corrections_z_min"), 0).toInt();
   _correctionsZMax = settings.value(QStringLiteral("corrections_z_max"), _correctionsZMin).toInt();
    if (_correctionsZMax < _correctionsZMin) {
        _correctionsZMax = _correctionsZMin;
    }

    _customParamsText = settings.value(QStringLiteral("custom_params_text"), QString()).toString();
    validateCustomParamsText();

    settings.endGroup();
}

void SegmentationWidget::writeSetting(const QString& key, const QVariant& value)
{
    QSettings settings(QStringLiteral("VC.ini"), QSettings::IniFormat);
    settings.beginGroup(settingsGroup());
    settings.setValue(key, value);
    settings.endGroup();
}

void SegmentationWidget::updateEditingState(bool enabled, bool notifyListeners)
{
    if (_editingEnabled == enabled) {
        return;
    }

    _editingEnabled = enabled;
    if (!_editingEnabled && _eraseBrushActive) {
        _eraseBrushActive = false;
    }
    syncUiState();

    if (notifyListeners) {
        emit editingModeChanged(_editingEnabled);
    }
}

void SegmentationWidget::setEraseBrushActive(bool active)
{
    const bool sanitized = _editingEnabled && active;
    if (_eraseBrushActive == sanitized) {
        return;
    }
    _eraseBrushActive = sanitized;
    syncUiState();
}

void SegmentationWidget::setPendingChanges(bool pending)
{
    if (_pending == pending) {
        return;
    }
    _pending = pending;
    syncUiState();
}

void SegmentationWidget::setEditingEnabled(bool enabled)
{
    updateEditingState(enabled, false);
}

void SegmentationWidget::setRadius(float value)
{
    const float clamped = std::clamp(value, 0.25f, 128.0f);
    if (std::fabs(clamped - _radiusSteps) < 1e-4f) {
        return;
    }
    _radiusSteps = clamped;
    writeSetting(QStringLiteral("radius_steps"), _radiusSteps);
    if (_spinRadius) {
        const QSignalBlocker blocker(_spinRadius);
        _spinRadius->setValue(static_cast<double>(_radiusSteps));
    }
}

void SegmentationWidget::setSigma(float value)
{
    const float clamped = std::clamp(value, 0.05f, 64.0f);
    if (std::fabs(clamped - _sigmaSteps) < 1e-4f) {
        return;
    }
    _sigmaSteps = clamped;
    writeSetting(QStringLiteral("sigma_steps"), _sigmaSteps);
    if (_spinSigma) {
        const QSignalBlocker blocker(_spinSigma);
        _spinSigma->setValue(static_cast<double>(_sigmaSteps));
    }
}

void SegmentationWidget::setPushPullStep(float value)
{
    const float clamped = std::clamp(value, 0.05f, 10.0f);
    if (std::fabs(clamped - _pushPullStep) < 1e-4f) {
        return;
    }
    _pushPullStep = clamped;
    writeSetting(QStringLiteral("push_pull_step"), _pushPullStep);
    if (_spinPushPullStep) {
        const QSignalBlocker blocker(_spinPushPullStep);
        _spinPushPullStep->setValue(static_cast<double>(_pushPullStep));
    }
}

void SegmentationWidget::handleCustomParamsEdited()
{
    if (!_editCustomParams) {
        return;
    }
    _customParamsText = _editCustomParams->toPlainText();
    writeSetting(QStringLiteral("custom_params_text"), _customParamsText);
    validateCustomParamsText();
    updateCustomParamsStatus();
}

void SegmentationWidget::validateCustomParamsText()
{
    QString error;
    parseCustomParams(&error);
    _customParamsError = error;
}

void SegmentationWidget::updateCustomParamsStatus()
{
    if (!_lblCustomParamsStatus) {
        return;
    }
    if (_customParamsError.isEmpty()) {
        _lblCustomParamsStatus->clear();
        _lblCustomParamsStatus->setVisible(false);
        return;
    }
    _lblCustomParamsStatus->setText(_customParamsError);
    _lblCustomParamsStatus->setVisible(true);
}

std::optional<nlohmann::json> SegmentationWidget::parseCustomParams(QString* error) const
{
    if (error) {
        error->clear();
    }

    const QString trimmed = _customParamsText.trimmed();
    if (trimmed.isEmpty()) {
        return std::nullopt;
    }

    try {
        const QByteArray utf8 = trimmed.toUtf8();
        nlohmann::json parsed = nlohmann::json::parse(utf8.constData(), utf8.constData() + utf8.size());
        if (!parsed.is_object()) {
            if (error) {
                *error = tr("Custom params must be a JSON object.");
            }
            return std::nullopt;
        }
        return parsed;
    } catch (const nlohmann::json::parse_error& ex) {
        if (error) {
            *error = tr("Custom params JSON parse error (byte %1): %2")
                         .arg(static_cast<qulonglong>(ex.byte))
                         .arg(QString::fromStdString(ex.what()));
        }
    } catch (const std::exception& ex) {
        if (error) {
            *error = tr("Custom params JSON parse error: %1")
                         .arg(QString::fromStdString(ex.what()));
        }
    } catch (...) {
        if (error) {
            *error = tr("Custom params JSON parse error: unknown error");
        }
    }

    return std::nullopt;
}

std::optional<nlohmann::json> SegmentationWidget::customParamsJson() const
{
    QString error;
    auto parsed = parseCustomParams(&error);
    if (!error.isEmpty()) {
        return std::nullopt;
    }
    return parsed;
}

void SegmentationWidget::setGrowthMethod(SegmentationGrowthMethod method)
{
    if (_growthMethod == method) {
        return;
    }
    _growthMethod = method;
    writeSetting(QStringLiteral("growth_method"), static_cast<int>(_growthMethod));
    syncUiState();
    emit growthMethodChanged(_growthMethod);
}

void SegmentationWidget::setGrowthInProgress(bool running)
{
    if (_growthInProgress == running) {
        return;
    }
    _growthInProgress = running;
    updateGrowthUiState();
}

void SegmentationWidget::setNormalGridAvailable(bool available)
{
    _normalGridAvailable = available;
    syncUiState();
}

void SegmentationWidget::setNormalGridPathHint(const QString& hint)
{
    _normalGridHint = hint;
    QString display = hint.trimmed();
    const int colonIndex = display.indexOf(QLatin1Char(':'));
    if (colonIndex >= 0 && colonIndex + 1 < display.size()) {
        display = display.mid(colonIndex + 1).trimmed();
    }
    _normalGridDisplayPath = display;
    syncUiState();
}

void SegmentationWidget::setVolumePackagePath(const QString& path)
{
    _volumePackagePath = path;
    syncUiState();
}

void SegmentationWidget::setAvailableVolumes(const QVector<QPair<QString, QString>>& volumes,
                                              const QString& activeId)
{
    _volumeEntries = volumes;
    _activeVolumeId = determineDefaultVolumeId(_volumeEntries, activeId);
    if (_comboVolumes) {
        const QSignalBlocker blocker(_comboVolumes);
        _comboVolumes->clear();
        for (const auto& entry : _volumeEntries) {
            const QString& id = entry.first;
            const QString& label = entry.second.isEmpty() ? id : entry.second;
            _comboVolumes->addItem(label, id);
        }
        int idx = _comboVolumes->findData(_activeVolumeId);
        if (idx < 0 && !_volumeEntries.isEmpty()) {
            _activeVolumeId = _comboVolumes->itemData(0).toString();
            idx = 0;
        }
        if (idx >= 0) {
            _comboVolumes->setCurrentIndex(idx);
        }
        _comboVolumes->setEnabled(!_volumeEntries.isEmpty());
    }
}

void SegmentationWidget::setActiveVolume(const QString& volumeId)
{
    if (_activeVolumeId == volumeId) {
        return;
    }
    _activeVolumeId = volumeId;
    if (_comboVolumes) {
        const QSignalBlocker blocker(_comboVolumes);
        int idx = _comboVolumes->findData(_activeVolumeId);
        if (idx >= 0) {
            _comboVolumes->setCurrentIndex(idx);
        }
    }
}

void SegmentationWidget::setCorrectionsEnabled(bool enabled)
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
    updateGrowthUiState();
}

void SegmentationWidget::setCorrectionsAnnotateChecked(bool enabled)
{
    _correctionsAnnotateChecked = enabled;
    if (_chkCorrectionsAnnotate) {
        const QSignalBlocker blocker(_chkCorrectionsAnnotate);
        _chkCorrectionsAnnotate->setChecked(enabled);
    }
    updateGrowthUiState();
}

void SegmentationWidget::setCorrectionCollections(const QVector<QPair<uint64_t, QString>>& collections,
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
    _comboCorrections->setEnabled(_correctionsEnabled && !_growthInProgress && _comboCorrections->count() > 0);
}

std::optional<std::pair<int, int>> SegmentationWidget::correctionsZRange() const
{
    if (!_correctionsZRangeEnabled) {
        return std::nullopt;
    }
    return std::make_pair(_correctionsZMin, _correctionsZMax);
}

std::vector<SegmentationGrowthDirection> SegmentationWidget::allowedGrowthDirections() const
{
    std::vector<SegmentationGrowthDirection> dirs;
    if (_growthDirectionMask & kGrowDirUpBit) {
        dirs.push_back(SegmentationGrowthDirection::Up);
    }
    if (_growthDirectionMask & kGrowDirDownBit) {
        dirs.push_back(SegmentationGrowthDirection::Down);
    }
    if (_growthDirectionMask & kGrowDirLeftBit) {
        dirs.push_back(SegmentationGrowthDirection::Left);
    }
    if (_growthDirectionMask & kGrowDirRightBit) {
        dirs.push_back(SegmentationGrowthDirection::Right);
    }
    if (dirs.empty()) {
        dirs = {
            SegmentationGrowthDirection::Up,
            SegmentationGrowthDirection::Down,
            SegmentationGrowthDirection::Left,
            SegmentationGrowthDirection::Right
        };
    }
    return dirs;
}

std::vector<SegmentationDirectionFieldConfig> SegmentationWidget::directionFieldConfigs() const
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

SegmentationDirectionFieldConfig SegmentationWidget::buildDirectionFieldDraft() const
{
    SegmentationDirectionFieldConfig config;
    config.path = _directionFieldPath.trimmed();
    config.orientation = _directionFieldOrientation;
    config.scale = std::clamp(_directionFieldScale, 0, 5);
    config.weight = std::clamp(_directionFieldWeight, 0.0, 10.0);
    return config;
}

void SegmentationWidget::refreshDirectionFieldList()
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

void SegmentationWidget::updateDirectionFieldFormFromSelection(int row)
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

void SegmentationWidget::applyDirectionFieldDraftToSelection(int row)
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

void SegmentationWidget::updateDirectionFieldListItem(int row)
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

void SegmentationWidget::updateDirectionFieldListGeometry()
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

void SegmentationWidget::persistDirectionFields()
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

void SegmentationWidget::setGrowthDirectionMask(int mask)
{
    mask = normalizeGrowthDirectionMask(mask);
    if (_growthDirectionMask == mask) {
        return;
    }
    _growthDirectionMask = mask;
    writeSetting(QStringLiteral("growth_direction_mask"), _growthDirectionMask);
    applyGrowthDirectionMaskToUi();
}

void SegmentationWidget::updateGrowthDirectionMaskFromUi(QCheckBox* changedCheckbox)
{
    int mask = 0;
    if (_chkGrowthDirUp && _chkGrowthDirUp->isChecked()) {
        mask |= kGrowDirUpBit;
    }
    if (_chkGrowthDirDown && _chkGrowthDirDown->isChecked()) {
        mask |= kGrowDirDownBit;
    }
    if (_chkGrowthDirLeft && _chkGrowthDirLeft->isChecked()) {
        mask |= kGrowDirLeftBit;
    }
    if (_chkGrowthDirRight && _chkGrowthDirRight->isChecked()) {
        mask |= kGrowDirRightBit;
    }

    if (mask == 0) {
        if (changedCheckbox) {
            const QSignalBlocker blocker(changedCheckbox);
            changedCheckbox->setChecked(true);
        }
        mask = kGrowDirAllMask;
    }

    setGrowthDirectionMask(mask);
}

void SegmentationWidget::applyGrowthDirectionMaskToUi()
{
    if (_chkGrowthDirUp) {
        const QSignalBlocker blocker(_chkGrowthDirUp);
        _chkGrowthDirUp->setChecked((_growthDirectionMask & kGrowDirUpBit) != 0);
    }
    if (_chkGrowthDirDown) {
        const QSignalBlocker blocker(_chkGrowthDirDown);
        _chkGrowthDirDown->setChecked((_growthDirectionMask & kGrowDirDownBit) != 0);
    }
    if (_chkGrowthDirLeft) {
        const QSignalBlocker blocker(_chkGrowthDirLeft);
        _chkGrowthDirLeft->setChecked((_growthDirectionMask & kGrowDirLeftBit) != 0);
    }
    if (_chkGrowthDirRight) {
        const QSignalBlocker blocker(_chkGrowthDirRight);
        _chkGrowthDirRight->setChecked((_growthDirectionMask & kGrowDirRightBit) != 0);
    }
}

void SegmentationWidget::updateGrowthUiState()
{
    const bool enableGrowth = _editingEnabled && !_growthInProgress;
    if (_spinGrowthSteps) {
        _spinGrowthSteps->setEnabled(enableGrowth);
    }
    if (_btnGrow) {
        _btnGrow->setEnabled(enableGrowth);
    }
    const bool enableDirCheckbox = enableGrowth;
    if (_chkGrowthDirUp) {
        _chkGrowthDirUp->setEnabled(enableDirCheckbox);
    }
    if (_chkGrowthDirDown) {
        _chkGrowthDirDown->setEnabled(enableDirCheckbox);
    }
    if (_chkGrowthDirLeft) {
        _chkGrowthDirLeft->setEnabled(enableDirCheckbox);
    }
    if (_chkGrowthDirRight) {
        _chkGrowthDirRight->setEnabled(enableDirCheckbox);
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

    const bool allowZRange = _editingEnabled && !_growthInProgress;
    if (_chkCorrectionsUseZRange) {
        _chkCorrectionsUseZRange->setEnabled(allowZRange);
    }
    if (_spinCorrectionsZMin) {
        _spinCorrectionsZMin->setEnabled(allowZRange && _correctionsZRangeEnabled);
    }
    if (_spinCorrectionsZMax) {
        _spinCorrectionsZMax->setEnabled(allowZRange && _correctionsZRangeEnabled);
    }
    if (_groupCustomParams) {
        _groupCustomParams->setEnabled(_editingEnabled);
    }

    const bool allowCorrections = _editingEnabled && _correctionsEnabled && !_growthInProgress;
    if (_groupCorrections) {
        _groupCorrections->setEnabled(allowCorrections);
    }
    if (_comboCorrections) {
        const QSignalBlocker blocker(_comboCorrections);
        _comboCorrections->setEnabled(allowCorrections && _comboCorrections->count() > 0);
    }
    if (_btnCorrectionsNew) {
        _btnCorrectionsNew->setEnabled(_editingEnabled && !_growthInProgress);
    }
    if (_chkCorrectionsAnnotate) {
        _chkCorrectionsAnnotate->setEnabled(allowCorrections);
    }
}

int SegmentationWidget::normalizeGrowthDirectionMask(int mask)
{
    mask &= kGrowDirAllMask;
    if (mask == 0) {
        // If no directions are selected, enable all directions by default.
        // This ensures that growth is not unintentionally disabled.
        mask = kGrowDirAllMask;
    }
    return mask;
}
