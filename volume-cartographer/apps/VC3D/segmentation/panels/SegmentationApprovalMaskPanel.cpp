#include "SegmentationApprovalMaskPanel.hpp"

#include "VCSettings.hpp"
#include "elements/CollapsibleSettingsGroup.hpp"

#include <QCheckBox>
#include <QComboBox>
#include <QColorDialog>
#include <QDoubleSpinBox>
#include <QHBoxLayout>
#include <QInputDialog>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QSettings>
#include <QSignalBlocker>
#include <QSlider>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>

SegmentationApprovalMaskPanel::SegmentationApprovalMaskPanel(const QString& settingsGroup,
                                                             QWidget* parent)
    : QWidget(parent)
    , _settingsGroup(settingsGroup)
{
    auto* panelLayout = new QVBoxLayout(this);
    panelLayout->setContentsMargins(0, 0, 0, 0);
    panelLayout->setSpacing(0);

    _groupApprovalMask = new CollapsibleSettingsGroup(tr("Approval Mask"), this);
    auto* approvalLayout = _groupApprovalMask->contentLayout();
    auto* approvalParent = _groupApprovalMask->contentWidget();

    // Show approval mask checkbox
    _chkShowApprovalMask = new QCheckBox(tr("Show Approval Mask"), approvalParent);
    _chkShowApprovalMask->setToolTip(tr("Display the approval mask overlay on the surface."));
    approvalLayout->addWidget(_chkShowApprovalMask);

    // Approval mask selection row
    auto* maskSelectionRow = new QHBoxLayout();
    maskSelectionRow->setSpacing(8);
    auto* maskSelectionLabel = new QLabel(tr("Mask:"), approvalParent);
    _comboApprovalMask = new QComboBox(approvalParent);
    _comboApprovalMask->setToolTip(tr("Select which mask file is shown and edited."));
    _btnNewApprovalMask = new QPushButton(tr("New"), approvalParent);
    _btnNewApprovalMask->setToolTip(tr("Create a new named mask file for this segment."));
    maskSelectionRow->addWidget(maskSelectionLabel);
    maskSelectionRow->addWidget(_comboApprovalMask, 1);
    maskSelectionRow->addWidget(_btnNewApprovalMask);
    approvalLayout->addLayout(maskSelectionRow);

    auto* maskedCopyRow = new QHBoxLayout();
    maskedCopyRow->setSpacing(8);
    _btnCopyMaskedForward = new QPushButton(tr("Copy Masked Forward"), approvalParent);
    _btnCopyMaskedForward->setToolTip(tr("Run masked neighbor copy forward on the active segment."));
    _btnCopyMaskedBackward = new QPushButton(tr("Copy Masked Backward"), approvalParent);
    _btnCopyMaskedBackward->setToolTip(tr("Run masked neighbor copy backward on the active segment."));
    _btnGrowAllInMasked = new QPushButton(tr("Grow All in Masked"), approvalParent);
    _btnGrowAllInMasked->setToolTip(
        tr("Run regular growth constrained to masked valid cells from the selected approval mask."));
    maskedCopyRow->addWidget(_btnCopyMaskedForward);
    maskedCopyRow->addWidget(_btnCopyMaskedBackward);
    maskedCopyRow->addWidget(_btnGrowAllInMasked);
    maskedCopyRow->addStretch(1);
    approvalLayout->addLayout(maskedCopyRow);

    // Edit checkboxes row - mutually exclusive approve/unapprove modes
    auto* editRow = new QHBoxLayout();
    editRow->setSpacing(8);

    _chkEditApprovedMask = new QCheckBox(tr("Edit Approved (B)"), approvalParent);
    _chkEditApprovedMask->setToolTip(tr("Paint regions as approved. Saves to disk when toggled off."));
    _chkEditApprovedMask->setEnabled(false);

    _chkEditUnapprovedMask = new QCheckBox(tr("Edit Unapproved (N)"), approvalParent);
    _chkEditUnapprovedMask->setToolTip(tr("Paint regions as unapproved. Saves to disk when toggled off."));
    _chkEditUnapprovedMask->setEnabled(false);

    editRow->addWidget(_chkEditApprovedMask);
    editRow->addWidget(_chkEditUnapprovedMask);
    editRow->addStretch(1);
    approvalLayout->addLayout(editRow);

    // Auto-approve edits checkbox
    _chkAutoApproveEdits = new QCheckBox(tr("Auto-Approve Edits"), approvalParent);
    _chkAutoApproveEdits->setToolTip(tr("Automatically add edited surface regions to the approval mask."));
    approvalLayout->addWidget(_chkAutoApproveEdits);

    // Auto-approval settings row: radius, threshold, max distance
    auto* autoApprovalRow = new QHBoxLayout();
    autoApprovalRow->setSpacing(8);

    auto* autoRadiusLabel = new QLabel(tr("Overpaint:"), approvalParent);
    _spinAutoApprovalRadius = new QDoubleSpinBox(approvalParent);
    _spinAutoApprovalRadius->setDecimals(1);
    _spinAutoApprovalRadius->setRange(0.0, 10.0);
    _spinAutoApprovalRadius->setSingleStep(0.1);
    _spinAutoApprovalRadius->setToolTip(tr("Overpainting radius in grid steps around edited vertices.\n"
                                           "0 = only exact edited vertices\n"
                                           "0.5 = small overpainting (default)\n"
                                           ">1 = larger overpainting region"));
    autoApprovalRow->addWidget(autoRadiusLabel);
    autoApprovalRow->addWidget(_spinAutoApprovalRadius);

    auto* autoThresholdLabel = new QLabel(tr("Min Change:"), approvalParent);
    _spinAutoApprovalThreshold = new QDoubleSpinBox(approvalParent);
    _spinAutoApprovalThreshold->setDecimals(1);
    _spinAutoApprovalThreshold->setRange(0.0, 10.0);
    _spinAutoApprovalThreshold->setSingleStep(0.1);
    _spinAutoApprovalThreshold->setToolTip(tr("Minimum vertex displacement (world units) to auto-approve.\n"
                                              "0 = approve all edited vertices\n"
                                              ">0 = skip vertices that moved less than this amount"));
    autoApprovalRow->addWidget(autoThresholdLabel);
    autoApprovalRow->addWidget(_spinAutoApprovalThreshold);

    auto* autoMaxDistLabel = new QLabel(tr("Max Dist:"), approvalParent);
    _spinAutoApprovalMaxDistance = new QDoubleSpinBox(approvalParent);
    _spinAutoApprovalMaxDistance->setDecimals(0);
    _spinAutoApprovalMaxDistance->setRange(0.0, 500.0);
    _spinAutoApprovalMaxDistance->setSingleStep(1.0);
    _spinAutoApprovalMaxDistance->setToolTip(tr("Maximum grid distance from drag center to auto-approve.\n"
                                               "0 = unlimited (approve all edited vertices)\n"
                                               ">0 = skip vertices farther than this from drag center"));
    autoApprovalRow->addWidget(autoMaxDistLabel);
    autoApprovalRow->addWidget(_spinAutoApprovalMaxDistance);
    autoApprovalRow->addStretch(1);
    approvalLayout->addLayout(autoApprovalRow);

    // Cylinder brush controls: radius and depth
    auto* approvalBrushRow = new QHBoxLayout();
    approvalBrushRow->setSpacing(8);

    auto* brushRadiusLabel = new QLabel(tr("Radius:"), approvalParent);
    _spinApprovalBrushRadius = new QDoubleSpinBox(approvalParent);
    _spinApprovalBrushRadius->setDecimals(0);
    _spinApprovalBrushRadius->setRange(1.0, 1000.0);
    _spinApprovalBrushRadius->setSingleStep(10.0);
    _spinApprovalBrushRadius->setToolTip(tr("Cylinder radius: circle size in plane views, rectangle width in flattened view (native voxels)."));
    approvalBrushRow->addWidget(brushRadiusLabel);
    approvalBrushRow->addWidget(_spinApprovalBrushRadius);

    auto* brushDepthLabel = new QLabel(tr("Depth:"), approvalParent);
    _spinApprovalBrushDepth = new QDoubleSpinBox(approvalParent);
    _spinApprovalBrushDepth->setDecimals(0);
    _spinApprovalBrushDepth->setRange(1.0, 500.0);
    _spinApprovalBrushDepth->setSingleStep(5.0);
    _spinApprovalBrushDepth->setToolTip(tr("Cylinder depth: rectangle height in flattened view, painting thickness from plane views (native voxels)."));
    approvalBrushRow->addWidget(brushDepthLabel);
    approvalBrushRow->addWidget(_spinApprovalBrushDepth);
    approvalBrushRow->addStretch(1);
    approvalLayout->addLayout(approvalBrushRow);

    auto* brushShapeRow = new QHBoxLayout();
    brushShapeRow->setSpacing(8);
    auto* brushShapeLabel = new QLabel(tr("Shape:"), approvalParent);
    _comboApprovalBrushShape = new QComboBox(approvalParent);
    _comboApprovalBrushShape->addItem(tr("Rectangle"), static_cast<int>(ApprovalBrushShape::Rectangle));
    _comboApprovalBrushShape->addItem(tr("Circle"), static_cast<int>(ApprovalBrushShape::Circle));
    _comboApprovalBrushShape->setToolTip(
        tr("Brush shape used in the flattened segmentation view.\n"
           "Plane views continue using the existing cylinder footprint."));
    brushShapeRow->addWidget(brushShapeLabel);
    brushShapeRow->addWidget(_comboApprovalBrushShape);
    brushShapeRow->addStretch(1);
    approvalLayout->addLayout(brushShapeRow);

    // Opacity slider row
    auto* opacityRow = new QHBoxLayout();
    opacityRow->setSpacing(8);

    auto* opacityLabel = new QLabel(tr("Opacity:"), approvalParent);
    _sliderApprovalMaskOpacity = new QSlider(Qt::Horizontal, approvalParent);
    _sliderApprovalMaskOpacity->setRange(0, 100);
    _sliderApprovalMaskOpacity->setToolTip(tr("Mask overlay transparency (0 = transparent, 100 = opaque)."));

    _lblApprovalMaskOpacity = new QLabel(approvalParent);
    _lblApprovalMaskOpacity->setMinimumWidth(35);

    opacityRow->addWidget(opacityLabel);
    opacityRow->addWidget(_sliderApprovalMaskOpacity, 1);
    opacityRow->addWidget(_lblApprovalMaskOpacity);
    approvalLayout->addLayout(opacityRow);

    // Color picker row
    auto* colorRow = new QHBoxLayout();
    colorRow->setSpacing(8);

    auto* colorLabel = new QLabel(tr("Brush Color:"), approvalParent);
    _btnApprovalColor = new QPushButton(approvalParent);
    _btnApprovalColor->setFixedSize(60, 24);
    _btnApprovalColor->setToolTip(tr("Click to choose the color for approval mask painting."));

    colorRow->addWidget(colorLabel);
    colorRow->addWidget(_btnApprovalColor);
    colorRow->addStretch(1);
    approvalLayout->addLayout(colorRow);

    // Undo button
    auto* buttonRow = new QHBoxLayout();
    buttonRow->setSpacing(8);
    _btnUndoApprovalStroke = new QPushButton(tr("Undo (Ctrl+B)"), approvalParent);
    _btnUndoApprovalStroke->setToolTip(tr("Undo the last approval mask brush stroke."));
    buttonRow->addWidget(_btnUndoApprovalStroke);
    buttonRow->addStretch(1);
    approvalLayout->addLayout(buttonRow);

    panelLayout->addWidget(_groupApprovalMask);

    setApprovalMaskOptions(_approvalMaskOptions);

    // --- Signal wiring (moved from SegmentationWidget::buildUi) ---

    connect(_chkShowApprovalMask, &QCheckBox::toggled, this, [this](bool enabled) {
        setShowApprovalMask(enabled);
        // If show is being unchecked and edit modes are active, turn them off
        if (!enabled) {
            if (_editApprovedMask) {
                setEditApprovedMask(false);
            }
            if (_editUnapprovedMask) {
                setEditUnapprovedMask(false);
            }
        }
    });

    connect(_chkEditApprovedMask, &QCheckBox::toggled, this, [this](bool enabled) {
        setEditApprovedMask(enabled);
    });

    connect(_chkEditUnapprovedMask, &QCheckBox::toggled, this, [this](bool enabled) {
        setEditUnapprovedMask(enabled);
    });

    connect(_chkAutoApproveEdits, &QCheckBox::toggled, this, [this](bool enabled) {
        setAutoApprovalEnabled(enabled);
    });

    connect(_spinAutoApprovalRadius, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        setAutoApprovalRadius(static_cast<float>(value));
    });

    connect(_spinAutoApprovalThreshold, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        setAutoApprovalThreshold(static_cast<float>(value));
    });

    connect(_spinAutoApprovalMaxDistance, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        setAutoApprovalMaxDistance(static_cast<float>(value));
    });

    connect(_spinApprovalBrushRadius, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        setApprovalBrushRadius(static_cast<float>(value));
    });

    connect(_spinApprovalBrushDepth, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        setApprovalBrushDepth(static_cast<float>(value));
    });

    connect(_comboApprovalBrushShape, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index) {
        if (index < 0 || !_comboApprovalBrushShape) {
            return;
        }
        const QVariant data = _comboApprovalBrushShape->itemData(index);
        const ApprovalBrushShape shape = static_cast<ApprovalBrushShape>(data.toInt());
        setApprovalBrushShape(shape);
    });

    connect(_sliderApprovalMaskOpacity, &QSlider::valueChanged, this, [this](int value) {
        setApprovalMaskOpacity(value);
    });

    connect(_btnApprovalColor, &QPushButton::clicked, this, [this]() {
        QColor newColor = QColorDialog::getColor(_approvalBrushColor, this, tr("Choose Approval Mask Color"));
        if (newColor.isValid()) {
            setApprovalBrushColor(newColor);
        }
    });

    connect(_btnUndoApprovalStroke, &QPushButton::clicked, this, &SegmentationApprovalMaskPanel::approvalStrokesUndoRequested);

    connect(_comboApprovalMask, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index) {
        if (index < 0 || !_comboApprovalMask) {
            return;
        }
        const QString maskId = _comboApprovalMask->itemData(index).toString();
        setSelectedApprovalMaskId(maskId);
    });

    connect(_btnNewApprovalMask, &QPushButton::clicked, this, [this]() {
        bool accepted = false;
        const QString entered = QInputDialog::getText(
            this,
            tr("Create Mask"),
            tr("Mask name:"),
            QLineEdit::Normal,
            QString(),
            &accepted);
        if (!accepted) {
            return;
        }
        const QString trimmed = entered.trimmed();
        if (trimmed.isEmpty()) {
            return;
        }
        emit approvalMaskCreateRequested(trimmed);
    });

    connect(_btnCopyMaskedForward, &QPushButton::clicked, this, [this]() {
        emit copyMaskedForwardRequested();
    });
    connect(_btnCopyMaskedBackward, &QPushButton::clicked, this, [this]() {
        emit copyMaskedBackwardRequested();
    });
    connect(_btnGrowAllInMasked, &QPushButton::clicked, this, [this]() {
        emit growAllInMaskedRequested();
    });

    connect(_groupApprovalMask, &CollapsibleSettingsGroup::toggled, this, [this](bool expanded) {
        if (_restoringSettings) {
            return;
        }
        writeSetting(vc3d::settings::segmentation::GROUP_APPROVAL_MASK_EXPANDED, expanded);
    });
}

void SegmentationApprovalMaskPanel::writeSetting(const QString& key, const QVariant& value)
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.beginGroup(_settingsGroup);
    settings.setValue(key, value);
    settings.endGroup();
}

void SegmentationApprovalMaskPanel::setShowApprovalMask(bool enabled)
{
    if (_showApprovalMask == enabled) {
        return;
    }
    _showApprovalMask = enabled;
    qInfo() << "SegmentationWidget: Show approval mask changed to:" << enabled;
    if (!_restoringSettings) {
        writeSetting(QStringLiteral("show_approval_mask"), _showApprovalMask);
        qInfo() << "  Emitting showApprovalMaskChanged signal";
        emit showApprovalMaskChanged(_showApprovalMask);
    }
    if (_chkShowApprovalMask) {
        const QSignalBlocker blocker(_chkShowApprovalMask);
        _chkShowApprovalMask->setChecked(_showApprovalMask);
    }
    syncUiState();
}

void SegmentationApprovalMaskPanel::setEditApprovedMask(bool enabled)
{
    if (_editApprovedMask == enabled) {
        return;
    }
    _editApprovedMask = enabled;
    qInfo() << "SegmentationWidget: Edit approved mask changed to:" << enabled;

    // Mutual exclusion: if enabling approved, disable unapproved
    if (enabled && _editUnapprovedMask) {
        setEditUnapprovedMask(false);
    }

    if (!_restoringSettings) {
        writeSetting(QStringLiteral("edit_approved_mask"), _editApprovedMask);
        qInfo() << "  Emitting editApprovedMaskChanged signal";
        emit editApprovedMaskChanged(_editApprovedMask);
    }
    if (_chkEditApprovedMask) {
        const QSignalBlocker blocker(_chkEditApprovedMask);
        _chkEditApprovedMask->setChecked(_editApprovedMask);
    }
    syncUiState();
}

void SegmentationApprovalMaskPanel::setEditUnapprovedMask(bool enabled)
{
    if (_editUnapprovedMask == enabled) {
        return;
    }
    _editUnapprovedMask = enabled;
    qInfo() << "SegmentationWidget: Edit unapproved mask changed to:" << enabled;

    // Mutual exclusion: if enabling unapproved, disable approved
    if (enabled && _editApprovedMask) {
        setEditApprovedMask(false);
    }

    if (!_restoringSettings) {
        writeSetting(QStringLiteral("edit_unapproved_mask"), _editUnapprovedMask);
        qInfo() << "  Emitting editUnapprovedMaskChanged signal";
        emit editUnapprovedMaskChanged(_editUnapprovedMask);
    }
    if (_chkEditUnapprovedMask) {
        const QSignalBlocker blocker(_chkEditUnapprovedMask);
        _chkEditUnapprovedMask->setChecked(_editUnapprovedMask);
    }
    syncUiState();
}

void SegmentationApprovalMaskPanel::setAutoApproveEdits(bool enabled)
{
    setAutoApprovalEnabled(enabled);
}

void SegmentationApprovalMaskPanel::setAutoApprovalEnabled(bool enabled)
{
    if (_autoApprovalEnabled == enabled) {
        return;
    }
    _autoApprovalEnabled = enabled;
    qInfo() << "SegmentationWidget: Auto-approve edits changed to:" << enabled;
    if (!_restoringSettings) {
        writeSetting(QStringLiteral("auto_approval_enabled"), _autoApprovalEnabled);
        emit autoApprovalEnabledChanged(_autoApprovalEnabled);
        emit autoApproveEditsChanged(_autoApprovalEnabled);
    }
    if (_chkAutoApproveEdits) {
        const QSignalBlocker blocker(_chkAutoApproveEdits);
        _chkAutoApproveEdits->setChecked(_autoApprovalEnabled);
    }
}

void SegmentationApprovalMaskPanel::setAutoApprovalRadius(float radius)
{
    const float sanitized = std::clamp(radius, 0.0f, 10.0f);
    if (std::abs(_autoApprovalRadius - sanitized) < 1e-4f) {
        return;
    }
    _autoApprovalRadius = sanitized;
    if (!_restoringSettings) {
        writeSetting(QStringLiteral("auto_approval_radius"), _autoApprovalRadius);
        emit autoApprovalRadiusChanged(_autoApprovalRadius);
    }
    if (_spinAutoApprovalRadius) {
        const QSignalBlocker blocker(_spinAutoApprovalRadius);
        _spinAutoApprovalRadius->setValue(static_cast<double>(_autoApprovalRadius));
    }
}

void SegmentationApprovalMaskPanel::setAutoApprovalThreshold(float threshold)
{
    const float sanitized = std::clamp(threshold, 0.0f, 10.0f);
    if (std::abs(_autoApprovalThreshold - sanitized) < 1e-4f) {
        return;
    }
    _autoApprovalThreshold = sanitized;
    if (!_restoringSettings) {
        writeSetting(QStringLiteral("auto_approval_threshold"), _autoApprovalThreshold);
        emit autoApprovalThresholdChanged(_autoApprovalThreshold);
    }
    if (_spinAutoApprovalThreshold) {
        const QSignalBlocker blocker(_spinAutoApprovalThreshold);
        _spinAutoApprovalThreshold->setValue(static_cast<double>(_autoApprovalThreshold));
    }
}

void SegmentationApprovalMaskPanel::setAutoApprovalMaxDistance(float distance)
{
    const float sanitized = std::clamp(distance, 0.0f, 500.0f);
    if (std::abs(_autoApprovalMaxDistance - sanitized) < 1e-4f) {
        return;
    }
    _autoApprovalMaxDistance = sanitized;
    if (!_restoringSettings) {
        writeSetting(QStringLiteral("auto_approval_max_distance"), _autoApprovalMaxDistance);
        emit autoApprovalMaxDistanceChanged(_autoApprovalMaxDistance);
    }
    if (_spinAutoApprovalMaxDistance) {
        const QSignalBlocker blocker(_spinAutoApprovalMaxDistance);
        _spinAutoApprovalMaxDistance->setValue(static_cast<double>(_autoApprovalMaxDistance));
    }
}



void SegmentationApprovalMaskPanel::setApprovalBrushRadius(float radius)
{
    const float sanitized = std::clamp(radius, 1.0f, 1000.0f);
    if (std::abs(_approvalBrushRadius - sanitized) < 1e-4f) {
        return;
    }
    _approvalBrushRadius = sanitized;
    if (!_restoringSettings) {
        writeSetting(QStringLiteral("approval_brush_radius"), _approvalBrushRadius);
        emit approvalBrushRadiusChanged(_approvalBrushRadius);
    }
    if (_spinApprovalBrushRadius) {
        const QSignalBlocker blocker(_spinApprovalBrushRadius);
        _spinApprovalBrushRadius->setValue(static_cast<double>(_approvalBrushRadius));
    }
}

void SegmentationApprovalMaskPanel::setApprovalBrushDepth(float depth)
{
    const float sanitized = std::clamp(depth, 1.0f, 500.0f);
    if (std::abs(_approvalBrushDepth - sanitized) < 1e-4f) {
        return;
    }
    _approvalBrushDepth = sanitized;
    if (!_restoringSettings) {
        writeSetting(QStringLiteral("approval_brush_depth"), _approvalBrushDepth);
        emit approvalBrushDepthChanged(_approvalBrushDepth);
    }
    if (_spinApprovalBrushDepth) {
        const QSignalBlocker blocker(_spinApprovalBrushDepth);
        _spinApprovalBrushDepth->setValue(static_cast<double>(_approvalBrushDepth));
    }
}

void SegmentationApprovalMaskPanel::setApprovalBrushShape(ApprovalBrushShape shape)
{
    if (_approvalBrushShape == shape) {
        return;
    }
    _approvalBrushShape = shape;
    if (!_restoringSettings) {
        writeSetting(QStringLiteral("approval_brush_shape"), static_cast<int>(_approvalBrushShape));
        emit approvalBrushShapeChanged(_approvalBrushShape);
    }
    if (_comboApprovalBrushShape) {
        const QSignalBlocker blocker(_comboApprovalBrushShape);
        const int idx = _comboApprovalBrushShape->findData(static_cast<int>(_approvalBrushShape));
        if (idx >= 0) {
            _comboApprovalBrushShape->setCurrentIndex(idx);
        }
    }
}

void SegmentationApprovalMaskPanel::setApprovalMaskOptions(const QVector<QPair<QString, QString>>& options)
{
    QVector<QPair<QString, QString>> sanitized = options;
    if (sanitized.isEmpty()) {
        sanitized.push_back({QStringLiteral("default"), QStringLiteral("approval")});
    }
    _approvalMaskOptions = sanitized;

    if (_comboApprovalMask) {
        const QSignalBlocker blocker(_comboApprovalMask);
        _comboApprovalMask->clear();
        for (const auto& option : _approvalMaskOptions) {
            _comboApprovalMask->addItem(option.second, option.first);
        }
    }

    bool selectedFound = false;
    for (const auto& option : _approvalMaskOptions) {
        if (option.first == _selectedApprovalMaskId) {
            selectedFound = true;
            break;
        }
    }
    if (!selectedFound) {
        _selectedApprovalMaskId = _approvalMaskOptions.front().first;
    }
    syncUiState();
}

void SegmentationApprovalMaskPanel::setSelectedApprovalMaskId(const QString& maskId)
{
    QString resolved = maskId;
    bool found = false;
    for (const auto& option : _approvalMaskOptions) {
        if (option.first == resolved) {
            found = true;
            break;
        }
    }
    if (!found && !_approvalMaskOptions.isEmpty()) {
        resolved = _approvalMaskOptions.front().first;
    }

    if (_selectedApprovalMaskId == resolved) {
        return;
    }
    _selectedApprovalMaskId = resolved;
    if (!_restoringSettings) {
        writeSetting(QStringLiteral("approval_selected_mask_id"), _selectedApprovalMaskId);
        emit approvalMaskSelected(_selectedApprovalMaskId);
    }
    if (_comboApprovalMask) {
        const QSignalBlocker blocker(_comboApprovalMask);
        const int idx = _comboApprovalMask->findData(_selectedApprovalMaskId);
        if (idx >= 0) {
            _comboApprovalMask->setCurrentIndex(idx);
        }
    }
}

void SegmentationApprovalMaskPanel::setApprovalMaskOpacity(int opacity)
{
    const int sanitized = std::clamp(opacity, 0, 100);
    if (_approvalMaskOpacity == sanitized) {
        return;
    }
    _approvalMaskOpacity = sanitized;
    if (!_restoringSettings) {
        writeSetting(QStringLiteral("approval_mask_opacity"), _approvalMaskOpacity);
        emit approvalMaskOpacityChanged(_approvalMaskOpacity);
    }
    if (_sliderApprovalMaskOpacity) {
        const QSignalBlocker blocker(_sliderApprovalMaskOpacity);
        _sliderApprovalMaskOpacity->setValue(_approvalMaskOpacity);
    }
    if (_lblApprovalMaskOpacity) {
        _lblApprovalMaskOpacity->setText(QString::number(_approvalMaskOpacity) + QStringLiteral("%"));
    }
}

void SegmentationApprovalMaskPanel::setApprovalBrushColor(const QColor& color)
{
    if (!color.isValid() || _approvalBrushColor == color) {
        return;
    }
    _approvalBrushColor = color;
    if (!_restoringSettings) {
        writeSetting(QStringLiteral("approval_brush_color"), _approvalBrushColor.name());
        emit approvalBrushColorChanged(_approvalBrushColor);
    }
    if (_btnApprovalColor) {
        _btnApprovalColor->setStyleSheet(
            QStringLiteral("background-color: %1; border: 1px solid #888;").arg(_approvalBrushColor.name()));
    }

    const bool hasMaskSelection = !_approvalMaskOptions.isEmpty();
    const bool canCopyMasked = _showApprovalMask && hasMaskSelection;
    if (_btnCopyMaskedForward) {
        _btnCopyMaskedForward->setEnabled(canCopyMasked);
    }
    if (_btnCopyMaskedBackward) {
        _btnCopyMaskedBackward->setEnabled(canCopyMasked);
    }
    if (_btnGrowAllInMasked) {
        _btnGrowAllInMasked->setEnabled(canCopyMasked);
    }
}

void SegmentationApprovalMaskPanel::restoreSettings(QSettings& settings)
{
    using namespace vc3d::settings;

    _restoringSettings = true;

    _approvalBrushRadius = settings.value(segmentation::APPROVAL_BRUSH_RADIUS, _approvalBrushRadius).toFloat();
    _approvalBrushRadius = std::clamp(_approvalBrushRadius, 1.0f, 1000.0f);
    _approvalBrushDepth = settings.value(segmentation::APPROVAL_BRUSH_DEPTH, _approvalBrushDepth).toFloat();
    _approvalBrushDepth = std::clamp(_approvalBrushDepth, 1.0f, 500.0f);
    const int brushShapeValue = settings.value(segmentation::APPROVAL_BRUSH_SHAPE,
                                               static_cast<int>(_approvalBrushShape)).toInt();
    if (brushShapeValue == static_cast<int>(ApprovalBrushShape::Circle)) {
        _approvalBrushShape = ApprovalBrushShape::Circle;
    } else {
        _approvalBrushShape = ApprovalBrushShape::Rectangle;
    }

    _approvalMaskOpacity = settings.value(segmentation::APPROVAL_MASK_OPACITY, _approvalMaskOpacity).toInt();
    _approvalMaskOpacity = std::clamp(_approvalMaskOpacity, 0, 100);
    const QString colorName = settings.value(segmentation::APPROVAL_BRUSH_COLOR, _approvalBrushColor.name()).toString();
    if (QColor::isValidColorName(colorName)) {
        _approvalBrushColor = QColor::fromString(colorName);
    }
    _selectedApprovalMaskId = settings.value(segmentation::APPROVAL_SELECTED_MASK_ID,
                                             _selectedApprovalMaskId).toString();
    _showApprovalMask = settings.value(segmentation::SHOW_APPROVAL_MASK, _showApprovalMask).toBool();

    // Auto-approval settings — fall back to legacy key if new key not found
    if (settings.contains(segmentation::AUTO_APPROVAL_ENABLED)) {
        _autoApprovalEnabled = settings.value(segmentation::AUTO_APPROVAL_ENABLED, _autoApprovalEnabled).toBool();
    } else {
        _autoApprovalEnabled = settings.value(segmentation::APPROVAL_AUTO_APPROVE_EDITS, _autoApprovalEnabled).toBool();
    }
    _autoApprovalRadius = settings.value(segmentation::AUTO_APPROVAL_RADIUS, _autoApprovalRadius).toFloat();
    _autoApprovalRadius = std::clamp(_autoApprovalRadius, 0.0f, 10.0f);
    _autoApprovalThreshold = settings.value(segmentation::AUTO_APPROVAL_THRESHOLD, _autoApprovalThreshold).toFloat();
    _autoApprovalThreshold = std::clamp(_autoApprovalThreshold, 0.0f, 10.0f);
    _autoApprovalMaxDistance = settings.value(segmentation::AUTO_APPROVAL_MAX_DISTANCE, _autoApprovalMaxDistance).toFloat();
    _autoApprovalMaxDistance = std::clamp(_autoApprovalMaxDistance, 0.0f, 500.0f);
    const bool approvalMaskExpanded = settings.value(segmentation::GROUP_APPROVAL_MASK_EXPANDED,
                                                     segmentation::GROUP_APPROVAL_MASK_EXPANDED_DEFAULT).toBool();
    if (_groupApprovalMask) {
        _groupApprovalMask->setExpanded(approvalMaskExpanded);
    }

    // Don't restore edit states - user must explicitly enable editing each session

    _restoringSettings = false;
}

void SegmentationApprovalMaskPanel::syncUiState()
{
    if (_chkShowApprovalMask) {
        const QSignalBlocker blocker(_chkShowApprovalMask);
        _chkShowApprovalMask->setChecked(_showApprovalMask);
    }
    if (_chkEditApprovedMask) {
        const QSignalBlocker blocker(_chkEditApprovedMask);
        _chkEditApprovedMask->setChecked(_editApprovedMask);
        _chkEditApprovedMask->setEnabled(_showApprovalMask);
    }
    if (_chkEditUnapprovedMask) {
        const QSignalBlocker blocker(_chkEditUnapprovedMask);
        _chkEditUnapprovedMask->setChecked(_editUnapprovedMask);
        _chkEditUnapprovedMask->setEnabled(_showApprovalMask);
    }
    if (_chkAutoApproveEdits) {
        const QSignalBlocker blocker(_chkAutoApproveEdits);
        _chkAutoApproveEdits->setChecked(_autoApprovalEnabled);
    }
    if (_spinAutoApprovalRadius) {
        const QSignalBlocker blocker(_spinAutoApprovalRadius);
        _spinAutoApprovalRadius->setValue(static_cast<double>(_autoApprovalRadius));
    }
    if (_spinAutoApprovalThreshold) {
        const QSignalBlocker blocker(_spinAutoApprovalThreshold);
        _spinAutoApprovalThreshold->setValue(static_cast<double>(_autoApprovalThreshold));
    }
    if (_spinAutoApprovalMaxDistance) {
        const QSignalBlocker blocker(_spinAutoApprovalMaxDistance);
        _spinAutoApprovalMaxDistance->setValue(static_cast<double>(_autoApprovalMaxDistance));
    }
    if (_comboApprovalMask) {
        const QSignalBlocker blocker(_comboApprovalMask);
        const int idx = _comboApprovalMask->findData(_selectedApprovalMaskId);
        if (idx >= 0) {
            _comboApprovalMask->setCurrentIndex(idx);
        } else if (_comboApprovalMask->count() > 0) {
            _comboApprovalMask->setCurrentIndex(0);
        }
    }
    if (_comboApprovalBrushShape) {
        const QSignalBlocker blocker(_comboApprovalBrushShape);
        const int idx = _comboApprovalBrushShape->findData(static_cast<int>(_approvalBrushShape));
        if (idx >= 0) {
            _comboApprovalBrushShape->setCurrentIndex(idx);
        }
    }
    if (_sliderApprovalMaskOpacity) {
        const QSignalBlocker blocker(_sliderApprovalMaskOpacity);
        _sliderApprovalMaskOpacity->setValue(_approvalMaskOpacity);
    }
    if (_lblApprovalMaskOpacity) {
        _lblApprovalMaskOpacity->setText(QString::number(_approvalMaskOpacity) + QStringLiteral("%"));
    }
    if (_btnApprovalColor) {
        _btnApprovalColor->setStyleSheet(
            QStringLiteral("background-color: %1; border: 1px solid #888;").arg(_approvalBrushColor.name()));
    }

    const bool hasMaskSelection = !_approvalMaskOptions.isEmpty();
    const bool canCopyMasked = _showApprovalMask && hasMaskSelection;
    if (_btnCopyMaskedForward) {
        _btnCopyMaskedForward->setEnabled(canCopyMasked);
    }
    if (_btnCopyMaskedBackward) {
        _btnCopyMaskedBackward->setEnabled(canCopyMasked);
    }
    if (_btnGrowAllInMasked) {
        _btnGrowAllInMasked->setEnabled(canCopyMasked);
    }
}
