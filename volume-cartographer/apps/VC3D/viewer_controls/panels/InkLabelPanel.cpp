#include "viewer_controls/panels/InkLabelPanel.hpp"

#include "elements/CollapsibleSettingsGroup.hpp"
#include "elements/LabeledControlRow.hpp"
#include "overlays/InkLabelOverlayController.hpp"
#include "ink_labels/InkLabelController.hpp"

#include <QApplication>
#include <QButtonGroup>
#include <QClipboard>
#include <QFileDialog>
#include <QMenu>
#include <QMessageBox>
#include <QCheckBox>
#include <QComboBox>
#include <QFileInfo>
#include <QFontMetrics>
#include <QGridLayout>
#include <QLabel>
#include <QLocale>
#include <QPainter>
#include <QPixmap>
#include <QPushButton>
#include <QSignalBlocker>
#include <QSlider>
#include <QSpinBox>
#include <QTimer>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>
#include <initializer_list>

using namespace vc3d::ink_labels;

namespace {

// The radius slider is logarithmic: 0 is kMinRadius, kRadiusSteps is kMaxRadius.
constexpr int kRadiusSteps = 1000;

int radiusToSlider(double r)
{
    const double t = std::log(r / kMinRadius) / std::log(kMaxRadius / kMinRadius);
    return int(std::lround(std::clamp(t, 0.0, 1.0) * kRadiusSteps));
}

double sliderToRadius(int value)
{
    return kMinRadius * std::pow(kMaxRadius / kMinRadius, double(value) / kRadiusSteps);
}

// A small square of a label color for the class buttons.
QIcon swatch(const QColor& color)
{
    QPixmap pixmap(12, 12);
    pixmap.fill(Qt::transparent);
    QPainter painter(&pixmap);
    painter.setPen(color.darker(160));
    painter.setBrush(color);
    painter.drawRect(0, 0, 11, 11);
    return QIcon(pixmap);
}

QPushButton* makeButton(const QString& text, QWidget* parent, bool checkable)
{
    auto* button = new QPushButton(text, parent);
    button->setCheckable(checkable);
    button->setFocusPolicy(Qt::NoFocus);
    if (checkable) {
        button->setProperty("inkLabelBaseText", text);
        button->setAccessibleName(text);
        // A shape and text cue supplement color in both light and dark themes.
        // Scope this style to label toggles; ordinary action buttons stay native.
        button->setStyleSheet(QStringLiteral(
            "QPushButton { background-color: palette(button); color: palette(button-text);"
            " border: 2px solid transparent; border-radius: 3px; padding: 2px 4px; }"
            "QPushButton:checked { border-color: palette(highlight); font-weight: bold; }"
            "QPushButton:disabled { color: palette(mid); }"));
    }
    return button;
}

// Refresh from the controller even when QSignalBlocker suppresses toggled().
void setCheckedFeedback(QPushButton* button, bool checked)
{
    button->setChecked(checked);
    const QString text = button->property("inkLabelBaseText").toString();
    button->setText(checked ? text + QStringLiteral(" \u2713") : text);
    button->setAccessibleDescription(checked ? InkLabelPanel::tr("Selected") : InkLabelPanel::tr("Not selected"));
}

QCheckBox* makeCheck(const QString& text, QWidget* parent)
{
    auto* check = new QCheckBox(text, parent);
    check->setFocusPolicy(Qt::NoFocus);
    return check;
}

QComboBox* makeCombo(const QStringList& items, QWidget* parent)
{
    auto* combo = new QComboBox(parent);
    combo->addItems(items);
    combo->setFocusPolicy(Qt::NoFocus);
    return combo;
}

QSpinBox* makeSpin(int min, int max, QWidget* parent)
{
    auto* spin = new QSpinBox(parent);
    spin->setRange(min, max);
    // Arrows and the wheel still work; the label keys keep working.
    spin->setFocusPolicy(Qt::NoFocus);
    return spin;
}

}  // namespace

InkLabelPanel::InkLabelPanel(InkLabelController* labels, QWidget* parent)
    : QWidget(parent)
    , _labels(labels)
{
    setObjectName(QStringLiteral("inkLabelPanel"));
    auto* outer = new QVBoxLayout(this);
    outer->setContentsMargins(0, 8, 0, 0);
    outer->setSpacing(6);
    _labelsGroup = new CollapsibleSettingsGroup(tr("Ink labels"), this);
    _labelsGroup->setObjectName(QStringLiteral("inkLabelGroup"));
    _labelsGroup->setToolTip(tr("Draw ink, background and validation labels on the segment view and save them as the ink "
                                "detection trainer's label TIFFs."));
    outer->addWidget(_labelsGroup);
    auto* layout = _labelsGroup->contentLayout();
    layout->setSpacing(6);

    auto* modeRow = new LabeledControlRow(tr("Labels"), this);
    _modeButton = makeButton(tr("Label mode (I)"), modeRow, true);
    _modeButton->setObjectName(QStringLiteral("inkLabelModeButton"));
    _modeButton->setToolTip(tr("Paint labels on the segment view with the left mouse button. Right drag still pans."));
    modeRow->addControl(_modeButton);
    modeRow->addStretch(1);
    layout->addWidget(modeRow);

    _folderLabel = new QLabel(this);
    _folderLabel->setObjectName(QStringLiteral("inkLabelFolder"));
    _folderLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
    _folderLabel->setWordWrap(true);
    _folderLabel->setTextFormat(Qt::PlainText);
    _folderLabel->setMinimumWidth(40);
    layout->addWidget(_folderLabel);

    _statusLabel = new QLabel(this);
    _statusLabel->setObjectName(QStringLiteral("inkLabelStatus"));
    layout->addWidget(_statusLabel);

    _bannerLabel = new QLabel(this);
    _bannerLabel->setObjectName(QStringLiteral("inkLabelBanner"));
    _bannerLabel->setWordWrap(true);
    _bannerLabel->setTextFormat(Qt::PlainText);
    // Palette colors, so the banner reads in light and dark themes.
    _bannerLabel->setFrameShape(QFrame::StyledPanel);
    _bannerLabel->setMargin(4);
    _bannerLabel->setAutoFillBackground(true);
    _bannerLabel->setBackgroundRole(QPalette::ToolTipBase);
    _bannerLabel->setForegroundRole(QPalette::ToolTipText);
    _bannerLabel->hide();
    layout->addWidget(_bannerLabel);
    _newVersionButton = makeButton(tr("Start empty labels in a new version"), this, false);
    _newVersionButton->setObjectName(QStringLiteral("inkLabelNewVersion"));
    _newVersionButton->setToolTip(tr("Keep the locked labels as they are and label this canvas as the next version."));
    _newVersionButton->hide();
    layout->addWidget(_newVersionButton);

    auto* classRow = new LabeledControlRow(tr("Class"), this);
    _inkButton = makeButton(tr("Ink (W)"), classRow, true);
    _inkButton->setToolTip(tr("Ink, drawn red: the trainer learns these pixels as ink."));
    _inkButton->setIcon(swatch(InkLabelController::colorFor(PaintClass::Ink, false)));
    _backgroundButton = makeButton(tr("Background (F)"), classRow, true);
    _backgroundButton->setToolTip(tr("Background, drawn blue: supervised pixels that are not ink."));
    _backgroundButton->setIcon(swatch(InkLabelController::colorFor(PaintClass::Background, false)));
    _validationButton = makeButton(tr("Validation (V)"), classRow, true);
    _validationButton->setToolTip(tr("Validation area, drawn yellow: written to the validation mask."));
    _validationButton->setIcon(swatch(InkLabelController::colorFor(PaintClass::Validation, false)));
    auto* classGroup = new QButtonGroup(this);
    classGroup->setExclusive(true);
    classGroup->addButton(_inkButton, int(PaintClass::Ink));
    classGroup->addButton(_backgroundButton, int(PaintClass::Background));
    classGroup->addButton(_validationButton, int(PaintClass::Validation));
    classRow->addControl(_inkButton);
    classRow->addControl(_backgroundButton);
    classRow->addControl(_validationButton);
    classRow->addStretch(1);
    layout->addWidget(classRow);

    auto* eraseRow = new LabeledControlRow(QString(), this);
    _eraserButton = makeButton(tr("Eraser (Z)"), eraseRow, true);
    _eraserButton->setToolTip(tr("Ink and background go back to unknown (not supervised); validation is cleared."));
    _eraserButton->setIcon(swatch(InkLabelController::colorFor(PaintClass::Ink, true)));
    _keepsInkCheck = makeCheck(tr("Background keeps ink"), eraseRow);
    _keepsInkCheck->setToolTip(tr("Painting background over ink marks it supervised without clearing the ink."));
    eraseRow->addControl(_eraserButton);
    eraseRow->addControl(_keepsInkCheck);
    eraseRow->addStretch(1);
    layout->addWidget(eraseRow);

    auto* toolRow = new LabeledControlRow(tr("Tool"), this);
    _brushButton = makeButton(tr("Brush"), toolRow, true);
    _brushButton->setToolTip(tr("Paint with a round brush; Shift+Click draws a straight stroke. L switches tools."));
    _polygonButton = makeButton(tr("Polygon"), toolRow, true);
    _polygonButton->setToolTip(tr("Click points and press Enter, or drag a lasso. Backspace removes a point, Esc cancels. L "
                                  "switches tools."));
    _fillButton = makeButton(tr("Fill"), toolRow, true);
    _fillButton->setToolTip(tr("Click to fill the connected region the Assist settings describe (prediction above or below the "
                               "threshold, or equal labels). L switches tools."));
    auto* toolGroup = new QButtonGroup(this);
    toolGroup->setExclusive(true);
    toolGroup->addButton(_brushButton, int(InkLabelController::Tool::Brush));
    toolGroup->addButton(_polygonButton, int(InkLabelController::Tool::Polygon));
    toolGroup->addButton(_fillButton, int(InkLabelController::Tool::Fill));
    toolRow->addControl(_brushButton);
    toolRow->addControl(_polygonButton);
    toolRow->addControl(_fillButton);
    toolRow->addStretch(1);
    layout->addWidget(toolRow);

    auto* radiusRow = new LabeledControlRow(tr("Radius"), this);
    _radiusSlider = new QSlider(Qt::Horizontal, radiusRow);
    _radiusSlider->setObjectName(QStringLiteral("inkLabelRadius"));
    _radiusSlider->setRange(0, kRadiusSteps);
    _radiusSlider->setFocusPolicy(Qt::NoFocus);
    _radiusSlider->setToolTip(tr("Brush radius in label pixels (voxels at full resolution). Ctrl+Mouse Wheel in the view."));
    _radiusLabel = new QLabel(radiusRow);
    _radiusLabel->setToolTip(_radiusSlider->toolTip());
    _radiusLabel->setMinimumWidth(QFontMetrics(_radiusLabel->font()).horizontalAdvance(QStringLiteral("512 px")));
    radiusRow->addControl(_radiusSlider, 1);
    radiusRow->addControl(_radiusLabel);
    layout->addWidget(radiusRow);

    auto* showRow = new LabeledControlRow(tr("Show"), this);
    _showCheck = makeCheck(tr("Labels (H)"), showRow);
    _showInkCheck = makeCheck(tr("Ink"), showRow);
    _showSupervisionCheck = makeCheck(tr("Supervision"), showRow);
    _showValidationCheck = makeCheck(tr("Validation"), showRow);
    showRow->addControl(_showCheck);
    showRow->addControl(_showInkCheck);
    showRow->addControl(_showSupervisionCheck);
    showRow->addControl(_showValidationCheck);
    showRow->addStretch(1);
    layout->addWidget(showRow);

    auto* opacityRow = new LabeledControlRow(tr("Opacity"), this);
    _opacitySlider = new QSlider(Qt::Horizontal, opacityRow);
    _opacitySlider->setRange(0, 100);
    _opacitySlider->setFocusPolicy(Qt::NoFocus);
    _opacitySlider->setAccessibleName(tr("Label opacity"));
    _opacityLabel = new QLabel(opacityRow);
    _opacityLabel->setObjectName(QStringLiteral("inkLabelOpacityValue"));
    _opacityLabel->setAccessibleName(tr("Label opacity percentage"));
    _opacityLabel->setMinimumWidth(QFontMetrics(_opacityLabel->font()).horizontalAdvance(QStringLiteral("100%")));
    _opacityLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    opacityRow->addControl(_opacitySlider, 1);
    opacityRow->addControl(_opacityLabel);
    layout->addWidget(opacityRow);

    auto* historyRow = new LabeledControlRow(tr("History"), this);
    _undoButton = makeButton(tr("Undo"), historyRow, false);
    _redoButton = makeButton(tr("Redo"), historyRow, false);
    historyRow->addControl(_undoButton);
    historyRow->addControl(_redoButton);
    historyRow->addStretch(1);
    layout->addWidget(historyRow);

    auto* fileRow = new LabeledControlRow(tr("File"), this);
    _saveButton = makeButton(tr("Save (Ctrl+S)"), fileRow, false);
    _saveButton->setObjectName(QStringLiteral("inkLabelSave"));
    _saveButton->setToolTip(tr("Write <segment>_inklabels.tif, <segment>_supervision_mask.tif and, if used, "
                               "<segment>_validation_mask.tif."));
    fileRow->addControl(_saveButton);
    fileRow->addStretch(1);
    layout->addWidget(fileRow);

    // Versions, imports and the trainer handoff, folded until needed (added
    // below the Assist group).
    _filesGroup = new CollapsibleSettingsGroup(tr("Versions and import"), this);
    _filesGroup->setObjectName(QStringLiteral("inkLabelFilesGroup"));
    _filesGroup->setExpanded(false);
    auto* files = _filesGroup->contentLayout();
    files->setSpacing(6);
    auto* versionRow = new LabeledControlRow(tr("Version"), this);
    _versionCombo = new QComboBox(versionRow);
    _versionCombo->setObjectName(QStringLiteral("inkLabelVersion"));
    _versionCombo->setFocusPolicy(Qt::NoFocus);
    _versionCombo->setSizeAdjustPolicy(QComboBox::AdjustToContents);
    _versionCombo->setToolTip(tr("Label version: v1 is the unnumbered <segment>_inklabels.tif, v2 is "
                                 "<segment>_inklabels_v2.tif and so on. Save rewrites the open version."));
    _saveVersionButton = makeButton(tr("Save as new version"), versionRow, false);
    _saveVersionButton->setToolTip(tr("Write all three label files as the next version (_v2, _v3, ...)."));
    versionRow->addControl(_versionCombo);
    versionRow->addControl(_saveVersionButton);
    versionRow->addStretch(1);
    files->addWidget(versionRow);

    auto* handoffRow = new LabeledControlRow(QString(), this);
    _importButton = makeButton(tr("Import..."), handoffRow, false);
    _importButton->setObjectName(QStringLiteral("inkLabelImport"));
    _importButton->setToolTip(tr("Read a label TIFF, PNG or label zarr into ink, supervision or validation (by its file name)."));
    auto* importMenu = new QMenu(_importButton);
    importMenu->addAction(tr("Label image (TIFF or PNG)..."), this, [this] { chooseImport(false); });
    importMenu->addAction(tr("Label zarr folder..."), this, [this] { chooseImport(true); });
    _importButton->setMenu(importMenu);
    _commandsButton = makeButton(tr("Copy dataset entry"), handoffRow, false);
    _commandsButton->setObjectName(QStringLiteral("inkLabelDatasetEntry"));
    _commandsButton->setToolTip(tr("Copy the ink detection trainer's dataset entry (JSON) for these labels. The commands that "
                                   "make the label zarrs and the surface volume are in docs/ink_labels.md."));
    handoffRow->addControl(_importButton);
    handoffRow->addControl(_commandsButton);
    handoffRow->addStretch(1);
    files->addWidget(handoffRow);

    _countsLabel = new QLabel(this);
    _countsLabel->setObjectName(QStringLiteral("inkLabelCounts"));
    _countsLabel->setWordWrap(true);
    layout->addWidget(_countsLabel);

    // Assist: fill, suggestions and cleanup, folded away until needed.
    _assistGroup = new CollapsibleSettingsGroup(tr("Assist"), this);
    _assistGroup->setObjectName(QStringLiteral("inkLabelAssistGroup"));
    _assistGroup->setToolTip(tr("Fill, suggestions and cleanup work in the visible part of the segment view and add one undo "
                                "entry each."));
    _assistGroup->setExpanded(false);
    outer->addWidget(_assistGroup);
    auto* assist = _assistGroup->contentLayout();
    assist->setSpacing(6);

    auto* guideRow = new LabeledControlRow(tr("Guide"), this);
    _guideCombo = makeCombo({tr("Prediction"), tr("Labels")}, guideRow);
    _guideCombo->setObjectName(QStringLiteral("inkLabelGuide"));
    _guideCombo->setToolTip(tr("Fill from the selected ink prediction (single-channel only), or across equal labels."));
    _sideCombo = makeCombo({tr("above"), tr("below")}, guideRow);
    _sideCombo->setToolTip(tr("Prediction values at or above (or at or below) the threshold pass."));
    guideRow->addControl(_guideCombo);
    guideRow->addControl(_sideCombo);
    guideRow->addStretch(1);
    assist->addWidget(guideRow);

    auto* fillRow = new LabeledControlRow(tr("Fill"), this);
    _connectivityCombo = makeCombo({tr("4-connected"), tr("8-connected")}, fillRow);
    _unlabeledOnlyCheck = makeCheck(tr("Unlabeled only"), fillRow);
    _unlabeledOnlyCheck->setToolTip(tr("The fill tool fills only pixels that are neither ink nor background yet. (Suggestions "
                                       "always leave labeled pixels alone.)"));
    fillRow->addControl(_connectivityCombo);
    fillRow->addControl(_unlabeledOnlyCheck);
    fillRow->addStretch(1);
    assist->addWidget(fillRow);

    auto* thresholdRow = new LabeledControlRow(tr("Threshold"), this);
    _thresholdSlider = new QSlider(Qt::Horizontal, thresholdRow);
    _thresholdSlider->setObjectName(QStringLiteral("inkLabelThreshold"));
    _thresholdSlider->setRange(0, 255);
    _thresholdSlider->setFocusPolicy(Qt::NoFocus);
    _thresholdSlider->setToolTip(tr("In the prediction's display units: 8-bit values as they are, other depths scaled to 0-255."));
    _thresholdLabel = new QLabel(thresholdRow);
    _thresholdLabel->setMinimumWidth(QFontMetrics(_thresholdLabel->font()).horizontalAdvance(QStringLiteral("255")));
    thresholdRow->addControl(_thresholdSlider, 1);
    thresholdRow->addControl(_thresholdLabel);
    assist->addWidget(thresholdRow);

    auto* suggestRow = new LabeledControlRow(tr("Suggest"), this);
    _medianCombo = makeCombo({tr("no smoothing"), tr("median 3"), tr("median 5")}, suggestRow);
    _minAreaSpin = makeSpin(0, 1000000, suggestRow);
    _minAreaSpin->setSuffix(tr(" px min"));
    _minAreaSpin->setSingleStep(10);
    _minAreaSpin->setToolTip(tr("Smallest suggested component, in label pixels."));
    suggestRow->addControl(_medianCombo);
    suggestRow->addControl(_minAreaSpin);
    suggestRow->addStretch(1);
    assist->addWidget(suggestRow);

    auto* suggestButtons = new LabeledControlRow(QString(), this);
    _suggestButton = makeButton(tr("Suggest (U)"), suggestButtons, false);
    _suggestButton->setObjectName(QStringLiteral("inkLabelSuggest"));
    _suggestButton->setIcon(swatch(QColor::fromRgb(InkLabelOverlayController::suggestionColorTable()[1])));
    _pickAllButton = makeButton(tr("Pick all (Shift+U)"), suggestButtons, false);
    _pickAllButton->setObjectName(QStringLiteral("inkLabelPickAll"));
    suggestButtons->addControl(_suggestButton);
    suggestButtons->addControl(_pickAllButton);
    suggestButtons->addStretch(1);
    assist->addWidget(suggestButtons);

    auto* acceptRow = new LabeledControlRow(QString(), this);
    _acceptButton = makeButton(tr("Accept (Enter)"), acceptRow, false);
    _acceptButton->setToolTip(tr("Paint the picked suggestions as ink, as one undo step."));
    _discardButton = makeButton(tr("Discard (Esc)"), acceptRow, false);
    acceptRow->addControl(_acceptButton);
    acceptRow->addControl(_discardButton);
    acceptRow->addStretch(1);
    assist->addWidget(acceptRow);

    _assistLabel = new QLabel(this);
    _assistLabel->setObjectName(QStringLiteral("inkLabelAssistStatus"));
    _assistLabel->setWordWrap(true);
    assist->addWidget(_assistLabel);

    auto* cleanupRow = new LabeledControlRow(tr("Cleanup"), this);
    _cleanupSpin = makeSpin(1, 256, cleanupRow);
    _cleanupSpin->setSuffix(tr(" px"));
    _cleanupSpin->setToolTip(tr("Size n: specks and holes under n px, or grow, shrink and ring by n px."));
    cleanupRow->addControl(_cleanupSpin);
    cleanupRow->addStretch(1);
    assist->addWidget(cleanupRow);
    // Three columns, so the buttons fit the dock bar's pop-up.
    auto* cleanupButtons = new QWidget(this);
    auto* cleanupGrid = new QGridLayout(cleanupButtons);
    cleanupGrid->setContentsMargins(0, 0, 0, 0);
    const std::pair<Cleanup, QString> cleanups[] = {
        {Cleanup::RemoveSpecks, tr("Remove specks")},
        {Cleanup::FillHoles, tr("Fill holes")},
        {Cleanup::Grow, tr("Grow")},
        {Cleanup::Shrink, tr("Shrink")},
        {Cleanup::BackgroundRing, tr("Background ring")},
    };
    int cleanupIndex = 0;
    for (const auto& [kind, text] : cleanups) {
        auto* button = makeButton(text, cleanupButtons, false);
        button->setProperty("cleanup", int(kind));
        cleanupGrid->addWidget(button, cleanupIndex / 3, cleanupIndex % 3);
        _cleanupButtons.append(button);
        ++cleanupIndex;
    }
    assist->addWidget(cleanupButtons);
    outer->addWidget(_filesGroup);

    _inkButton->setObjectName(QStringLiteral("inkLabelControl_ink"));
    _backgroundButton->setObjectName(QStringLiteral("inkLabelControl_background"));
    _validationButton->setObjectName(QStringLiteral("inkLabelControl_validation"));
    _eraserButton->setObjectName(QStringLiteral("inkLabelControl_erase"));
    _keepsInkCheck->setObjectName(QStringLiteral("inkLabelControl_keep_ink"));
    _brushButton->setObjectName(QStringLiteral("inkLabelControl_brush"));
    _polygonButton->setObjectName(QStringLiteral("inkLabelControl_polygon"));
    _fillButton->setObjectName(QStringLiteral("inkLabelControl_fill"));
    _showCheck->setObjectName(QStringLiteral("inkLabelControl_show"));
    _showInkCheck->setObjectName(QStringLiteral("inkLabelControl_show_ink"));
    _showSupervisionCheck->setObjectName(QStringLiteral("inkLabelControl_show_supervision"));
    _showValidationCheck->setObjectName(QStringLiteral("inkLabelControl_show_validation"));
    _opacitySlider->setObjectName(QStringLiteral("inkLabelControl_opacity"));
    _undoButton->setObjectName(QStringLiteral("inkLabelControl_undo"));
    _redoButton->setObjectName(QStringLiteral("inkLabelControl_redo"));

    if (!_labels) {
        setEnabled(false);
        return;
    }

    _countsTimer = new QTimer(this);
    _countsTimer->setSingleShot(true);
    _countsTimer->setInterval(150);
    connect(_countsTimer, &QTimer::timeout, this, &InkLabelPanel::refreshCounts);

    connect(_modeButton, &QPushButton::clicked, this, [this](bool checked) {
        _labels->setLabelModeEnabled(checked);
        refreshSession();
    });
    connect(classGroup, &QButtonGroup::idClicked, this, [this](int id) {
        _labels->setErase(false);
        _labels->setPaintClass(PaintClass(id));
    });
    connect(_eraserButton, &QPushButton::clicked, this, [this](bool checked) { _labels->setErase(checked); });
    connect(_keepsInkCheck, &QCheckBox::toggled, this, [this](bool checked) { _labels->setBackgroundKeepsInk(checked); });
    connect(toolGroup, &QButtonGroup::idClicked, this, [this](int id) { _labels->setTool(InkLabelController::Tool(id)); });
    connect(_radiusSlider, &QSlider::valueChanged, this, [this](int value) { _labels->setRadius(sliderToRadius(value)); });
    for (QCheckBox* check : {_showCheck, _showInkCheck, _showSupervisionCheck, _showValidationCheck})
        connect(check, &QCheckBox::toggled, this, &InkLabelPanel::pushDisplay);
    connect(_opacitySlider, &QSlider::valueChanged, this, &InkLabelPanel::pushDisplay);
    connect(_undoButton, &QPushButton::clicked, this, [this] { _labels->undo(Actor::User, true); });
    connect(_redoButton, &QPushButton::clicked, this, [this] { _labels->redo(Actor::User); });
    connect(_saveButton, &QPushButton::clicked, this, [this] { _labels->saveInteractive(); });
    connect(_saveVersionButton, &QPushButton::clicked, this, [this] { _labels->saveAsNewVersion(); });
    connect(_commandsButton, &QPushButton::clicked, this, [this] {
        const QString text = _labels->datasetEntryText();
        if (text.isEmpty())
            return;
        QApplication::clipboard()->setText(text);
        _assistLabel->setText(tr("Dataset entry copied to the clipboard; set volume_scale as in your config."));
    });
    connect(_newVersionButton, &QPushButton::clicked, this, [this] {
        const auto out = _labels->startNewVersion();
        if (!out.ok && !out.message.isEmpty())
            _bannerLabel->setText(out.message);
    });
    connect(_versionCombo, &QComboBox::activated, this, [this](int index) {
        const int version = _versionCombo->itemData(index).toInt();
        const InkLabelController::SessionInfo* session = _labels->session();
        if (version > 0 && session && version != session->paths.version)
            _labels->openVersion(version);
    });
    connect(_labels, &InkLabelController::importFinished, this, &InkLabelPanel::onImportFinished);
    for (QComboBox* combo : {_guideCombo, _sideCombo, _connectivityCombo, _medianCombo})
        connect(combo, &QComboBox::currentIndexChanged, this, &InkLabelPanel::pushAssist);
    connect(_thresholdSlider, &QSlider::valueChanged, this, &InkLabelPanel::pushAssist);
    connect(_unlabeledOnlyCheck, &QCheckBox::toggled, this, &InkLabelPanel::pushAssist);
    connect(_minAreaSpin, &QSpinBox::valueChanged, this, &InkLabelPanel::pushAssist);
    connect(_cleanupSpin, &QSpinBox::valueChanged, this, &InkLabelPanel::pushAssist);
    connect(_suggestButton, &QPushButton::clicked, this, [this] { _labels->suggest(); });
    connect(_pickAllButton, &QPushButton::clicked, this, [this] { _labels->pickAllSuggestions(); });
    connect(_acceptButton, &QPushButton::clicked, this, [this] { _labels->acceptSuggestions(); });
    connect(_discardButton, &QPushButton::clicked, this, [this] { _labels->cancelAssist(); });
    for (QPushButton* button : std::as_const(_cleanupButtons))
        connect(button, &QPushButton::clicked, this, [this, button] { _labels->runCleanup(Cleanup(button->property("cleanup").toInt())); });
    connect(_labels, &InkLabelController::assistChanged, this, &InkLabelPanel::refreshAssist);

    connect(_labels, &InkLabelController::sessionChanged, this, &InkLabelPanel::refreshSession);
    connect(_labels, &InkLabelController::labelModeChanged, this, &InkLabelPanel::refreshSession);
    connect(_labels, &InkLabelController::labelModeRefusalChanged, this, &InkLabelPanel::refreshSession);
    connect(_labels, &InkLabelController::dirtyChanged, this, &InkLabelPanel::refreshSession);
    connect(_labels, &InkLabelController::saveFinished, this, &InkLabelPanel::refreshSession);
    connect(_labels, &InkLabelController::toolStateChanged, this, &InkLabelPanel::refreshTools);
    connect(_labels, &InkLabelController::toolStateChanged, this, &InkLabelPanel::refreshAssist);
    connect(_labels, &InkLabelController::displayChanged, this, &InkLabelPanel::refreshDisplay);
    connect(_labels, &InkLabelController::historyChanged, this, &InkLabelPanel::refreshHistory);
    connect(_labels, &InkLabelController::rasterChanged, this, [this](QRect) { _countsTimer->start(); });

    refreshSession();
    refreshTools();
    refreshDisplay();
    refreshHistory();
    refreshAssist();
}

bool InkLabelPanel::clickUndoForAgent()
{
    if (!_labels || !_undoButton || !_undoButton->isEnabled())
        return false;
    const auto* history = _labels->history();
    const auto* top = history ? history->top() : nullptr;
    if (!top || top->actor != Actor::Agent)
        return false;
    _undoButton->click();
    return true;
}

void InkLabelPanel::refreshSession()
{
    if (!_labels)
        return;
    const InkLabelController::SessionInfo* session = _labels->session();
    const QString refusal = _labels->labelModeRefusal();
    {
        const QSignalBlocker blocker(_modeButton);
        setCheckedFeedback(_modeButton, _labels->labelModeEnabled());
    }
    if (!session) {
        _folderLabel->setText(tr("No labels open. Select a segment and press I."));
        _folderLabel->setToolTip(QString());
        _statusLabel->clear();
        _bannerLabel->setText(refusal);
        _bannerLabel->setVisible(!refusal.isEmpty());
    } else {
        const QString dir = QString::fromStdString(session->paths.dir.string());
        _folderLabel->setText(dir);
        _folderLabel->setToolTip(dir);

        QStringList parts;
        // The open version, when there is more than one (the combo box sits
        // in the folded group).
        if (!session->locked && (session->paths.version > 1 || _labels->versions().size() > 1))
            parts << tr("v%1").arg(session->paths.version);
        if (session->locked)
            parts << tr("locked");
        else if (session->readOnly)
            parts << tr("read-only");
        else
            parts << (session->dirty ? tr("unsaved changes") : tr("no unsaved changes"));
        if (const auto& last = _labels->lastSave(); last && last->atUtc.isValid())
            parts << tr("saved %1").arg(last->atUtc.toLocalTime().toString(QStringLiteral("hh:mm")));
        if (_labels->lastAutosave().isValid())
            parts << tr("autosaved %1").arg(_labels->lastAutosave().toLocalTime().toString(QStringLiteral("hh:mm")));
        if (_labels->saveInFlight())
            parts << tr("saving...");
        _statusLabel->setText(parts.join(QStringLiteral(", ")));

        const QString reason = refusal.isEmpty() ? session->lockReason : refusal;
        _bannerLabel->setVisible(!reason.isEmpty());
        _bannerLabel->setText(reason);
    }

    const bool editable = session && !session->locked && !session->readOnly;
    for (QWidget* w : std::initializer_list<QWidget*>{_inkButton,
                                                      _backgroundButton,
                                                      _validationButton,
                                                      _eraserButton,
                                                      _keepsInkCheck,
                                                      _brushButton,
                                                      _polygonButton,
                                                      _fillButton,
                                                      _radiusSlider})
        w->setEnabled(editable || !session);
    _saveButton->setEnabled(editable);
    _saveVersionButton->setEnabled(editable);
    _importButton->setEnabled(editable);
    _commandsButton->setEnabled(session && !session->locked);
    _newVersionButton->setVisible(session && session->locked);
    refreshVersions();
    refreshAssist();
    refreshHistory();
    refreshCounts();
}

void InkLabelPanel::refreshTools()
{
    if (!_labels)
        return;
    const QSignalBlocker b1(_inkButton), b2(_backgroundButton), b3(_validationButton), b4(_eraserButton), b5(_keepsInkCheck),
        b6(_brushButton), b7(_polygonButton), b8(_radiusSlider), b9(_fillButton);
    setCheckedFeedback(_inkButton, _labels->paintClass() == PaintClass::Ink);
    setCheckedFeedback(_backgroundButton, _labels->paintClass() == PaintClass::Background);
    setCheckedFeedback(_validationButton, _labels->paintClass() == PaintClass::Validation);
    setCheckedFeedback(_eraserButton, _labels->erase());
    _keepsInkCheck->setChecked(_labels->backgroundKeepsInk());
    setCheckedFeedback(_brushButton, _labels->tool() == InkLabelController::Tool::Brush);
    setCheckedFeedback(_polygonButton, _labels->tool() == InkLabelController::Tool::Polygon);
    setCheckedFeedback(_fillButton, _labels->tool() == InkLabelController::Tool::Fill);
    _radiusSlider->setValue(radiusToSlider(_labels->radius()));
    _radiusLabel->setText(tr("%1 px").arg(QString::number(_labels->radius(), 'g', 3)));
}

void InkLabelPanel::refreshDisplay()
{
    if (!_labels)
        return;
    const QSignalBlocker b1(_showCheck), b2(_showInkCheck), b3(_showSupervisionCheck), b4(_showValidationCheck),
        b5(_opacitySlider);
    _showCheck->setChecked(_labels->showLabels());
    _showInkCheck->setChecked(_labels->showInk());
    _showSupervisionCheck->setChecked(_labels->showSupervision());
    _showValidationCheck->setChecked(_labels->showValidation());
    _opacitySlider->setValue(int(std::lround(_labels->opacity() * 100.0)));
    _opacityLabel->setText(tr("%1%").arg(_opacitySlider->value()));
}

void InkLabelPanel::refreshHistory()
{
    if (!_labels)
        return;
    const History* history = _labels->history();
    const InkLabelController::SessionInfo* session = _labels->session();
    const bool editable = session && !session->locked && !session->readOnly;
    const std::size_t undoDepth = history ? history->undoDepth() : 0;
    const std::size_t redoDepth = history ? history->redoDepth() : 0;
    _undoButton->setText(tr("Undo (%1)").arg(undoDepth));
    _redoButton->setText(tr("Redo (%1)").arg(redoDepth));
    _undoButton->setEnabled(editable && undoDepth > 0);
    _redoButton->setEnabled(editable && redoDepth > 0);
}

void InkLabelPanel::refreshCounts()
{
    if (!_labels)
        return;
    const InkLabelController::SessionInfo* session = _labels->session();
    if (!session || session->locked) {
        _countsLabel->clear();
        return;
    }
    const LabelCounts c = LabelCounts::from(session->counts);
    const QLocale locale;
    _countsLabel->setText(tr("ink %1 px, background %2 px, validation %3 px")
                              .arg(locale.toString(qulonglong(c.ink)),
                                   locale.toString(qulonglong(c.background)),
                                   locale.toString(qulonglong(c.validation))));
}

void InkLabelPanel::pushDisplay()
{
    if (!_labels)
        return;
    _labels->setDisplay(_showCheck->isChecked(),
                        _showInkCheck->isChecked(),
                        _showSupervisionCheck->isChecked(),
                        _showValidationCheck->isChecked(),
                        _opacitySlider->value() / 100.0);
}

void InkLabelPanel::refreshAssist()
{
    if (!_labels)
        return;
    const InkLabelController::AssistSettings& a = _labels->assistSettings();
    {
        const QSignalBlocker b1(_guideCombo), b2(_sideCombo), b3(_connectivityCombo), b4(_medianCombo), b5(_thresholdSlider),
            b6(_unlabeledOnlyCheck), b7(_minAreaSpin), b8(_cleanupSpin);
        _guideCombo->setCurrentIndex(a.guide == GuideKind::Prediction ? 0 : 1);
        _sideCombo->setCurrentIndex(a.above ? 0 : 1);
        _connectivityCombo->setCurrentIndex(a.connectivity == 8 ? 1 : 0);
        _medianCombo->setCurrentIndex(a.median <= 1 ? 0 : (a.median >= 5 ? 2 : 1));
        _thresholdSlider->setValue(a.threshold);
        _unlabeledOnlyCheck->setChecked(a.unlabeledOnly);
        _minAreaSpin->setValue(a.minArea);
        _cleanupSpin->setValue(a.cleanupSize);
    }
    _thresholdLabel->setText(QString::number(a.threshold));

    const InkLabelController::SessionInfo* session = _labels->session();
    const bool editable = session && !session->locked && !session->readOnly;
    const bool running = _labels->assistRunning();
    const bool hasGuide = _labels->predictionGuide().has_value();
    const SuggestionSet* suggestions = _labels->suggestions();
    _suggestButton->setEnabled(editable && !running && hasGuide);
    _suggestButton->setToolTip(hasGuide ? tr("Components of the thresholded prediction in view, drawn hatched in cyan; pixels "
                                             "already labeled are left out. Click them to pick, Enter accepts the picked ones "
                                             "as ink, Esc discards.")
                                        : tr("Select a single-channel ink prediction above; color predictions have no values to "
                                             "threshold."));
    _acceptButton->setEnabled(editable && suggestions && !_labels->selectedSuggestions().empty());
    _pickAllButton->setEnabled(suggestions && !suggestions->empty());
    _discardButton->setEnabled(running || suggestions);
    for (QPushButton* button : std::as_const(_cleanupButtons))
        button->setEnabled(editable && !running);
    if (running)
        _assistLabel->setText(tr("Working... (Esc drops the result)"));
    else if (suggestions)
        _assistLabel->setText(tr("%1 suggestions, %2 picked").arg(suggestions->items.size()).arg(_labels->selectedSuggestions().size()));
    else
        _assistLabel->clear();
}

void InkLabelPanel::pushAssist()
{
    if (!_labels)
        return;
    InkLabelController::AssistSettings a = _labels->assistSettings();
    a.guide = _guideCombo->currentIndex() == 1 ? GuideKind::Labels : GuideKind::Prediction;
    a.above = _sideCombo->currentIndex() == 0;
    a.connectivity = _connectivityCombo->currentIndex() == 1 ? 8 : 4;
    a.median = _medianCombo->currentIndex() == 0 ? 1 : (_medianCombo->currentIndex() == 2 ? 5 : 3);
    a.threshold = uint8_t(_thresholdSlider->value());
    a.unlabeledOnly = _unlabeledOnlyCheck->isChecked();
    a.minArea = _minAreaSpin->value();
    a.cleanupSize = _cleanupSpin->value();
    _labels->setAssistSettings(a);
}

void InkLabelPanel::refreshVersions()
{
    if (!_labels)
        return;
    const QSignalBlocker blocker(_versionCombo);
    _versionCombo->clear();
    const InkLabelController::SessionInfo* session = _labels->session();
    if (!session || session->locked) {
        _versionCombo->setEnabled(false);
        return;
    }
    for (const int v : _labels->versions())
        _versionCombo->addItem(tr("v%1").arg(v), v);
    _versionCombo->setCurrentIndex(_versionCombo->findData(session->paths.version));
    _versionCombo->setEnabled(_versionCombo->count() > 1);
}

void InkLabelPanel::chooseImport(bool zarrFolder)
{
    if (!_labels || !_labels->session())
        return;
    const QString start = QString::fromStdString(_labels->session()->segmentDir.string());
    const QString path = zarrFolder ? QFileDialog::getExistingDirectory(this, tr("Import label zarr"), start)
                                    : QFileDialog::getOpenFileName(this, tr("Import label image"), start,
                                                                   tr("Label images (*.tif *.tiff *.png)"));
    if (path.isEmpty())
        return;
    ImportOptions options;
    options.kind = kindForFileName(path.toStdString());
    _labels->importFile(path.toStdString(), options);
}

void InkLabelPanel::onImportFinished(const QString& path, const ImportOutcome& outcome, const ImportOptions& options)
{
    if (!_labels)
        return;
    if (outcome.ok) {
        _assistLabel->setText(tr("Imported %1 px of %2 from %3")
                                  .arg(QLocale().toString(qulonglong(outcome.setPixels)), QString::fromLatin1(kindName(options.kind)),
                                       QFileInfo(path).fileName()));
        return;
    }
    if (!outcome.needsDecision) {
        _assistLabel->setText(outcome.error);
        return;
    }
    // A legacy mask or a lower-resolution render: ask, then import again.
    ImportOptions next = options;
    if (outcome.format == ImportFormat::LegacyPngPair && options.legacyMaskAs == LegacyMaskAs::Ask) {
        const auto answer = QMessageBox::question(this, tr("Legacy label mask"), outcome.error,
                                                  QMessageBox::Yes | QMessageBox::No | QMessageBox::Cancel, QMessageBox::Yes);
        if (answer == QMessageBox::Cancel)
            return;
        next.legacyMaskAs = answer == QMessageBox::Yes ? LegacyMaskAs::Supervision : LegacyMaskAs::Ignore;
    } else {
        const auto answer = QMessageBox::question(this, tr("Resample label image"), outcome.error, QMessageBox::Yes | QMessageBox::Cancel);
        if (answer != QMessageBox::Yes)
            return;
        next.allowResample = true;
    }
    _labels->importFile(path.toStdString(), next);
}
