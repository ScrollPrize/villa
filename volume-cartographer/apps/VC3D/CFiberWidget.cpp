#include "CFiberWidget.hpp"

#include "FiberLengthDisplay.hpp"
#include "FiberNameDisplay.hpp"
#include "LineAnnotationFiberSegments.hpp"

#include <QAbstractItemView>
#include <QAction>
#include <QApplication>
#include <QButtonGroup>
#include <QCheckBox>
#include <QColor>
#include <QDoubleSpinBox>
#include <QHeaderView>
#include <QHBoxLayout>
#include <QItemSelectionModel>
#include <QLabel>
#include <QLineEdit>
#include <QMenu>
#include <QMessageBox>
#include <QPainter>
#include <QScrollArea>
#include <QScrollBar>
#include <QSlider>
#include <QSpinBox>
#include <QSignalBlocker>
#include <QStyle>
#include <QStyledItemDelegate>
#include <QStyleOptionViewItem>
#include <QStringList>
#include <QTimer>
#include <QTreeView>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <unordered_map>
#include <utility>

namespace {

enum FiberColumn {
    kNameColumn = 0,
    kDirectionColumn,
    kLinkColumn,
    kPendingColumn,
    kLengthColumn,
    kControlPointsColumn,
    kLinePointsColumn,
    kTagsColumn,
    kInterpStatusColumn,
    kColumnCount,
};

constexpr int kFiberIdRole = Qt::UserRole + 1;
constexpr int kIsSpanRole = Qt::UserRole + 2;
constexpr int kSpanFirstControlIndexRole = Qt::UserRole + 3;
constexpr int kSpanSecondControlIndexRole = Qt::UserRole + 4;
// Marks the single dummy child that keeps a collapsed fiber's expand arrow
// visible; the real span rows are built on first expansion (they are ~90% of
// the model's items, and most fibers stay collapsed).
constexpr int kSpanPlaceholderRole = Qt::UserRole + 5;
constexpr double kHighlightThresholdDegrees = 45.0;

void addUniqueSorted(std::vector<std::string>& values, const std::string& value)
{
    if (value.empty()) {
        return;
    }
    if (std::find(values.begin(), values.end(), value) != values.end()) {
        return;
    }
    values.push_back(value);
    std::sort(values.begin(), values.end());
}

bool containsTag(const std::vector<std::string>& tags, const std::string& tag)
{
    return std::find(tags.begin(), tags.end(), tag) != tags.end();
}

class FiberNameDelegate final : public QStyledItemDelegate
{
public:
    using QStyledItemDelegate::QStyledItemDelegate;

    void paint(QPainter* painter,
               const QStyleOptionViewItem& option,
               const QModelIndex& index) const override
    {
        QStyleOptionViewItem adjusted(option);
        initStyleOption(&adjusted, index);
        const QString fullText = adjusted.text;

        QStyle* style = adjusted.widget ? adjusted.widget->style() : QApplication::style();
        adjusted.text.clear();
        adjusted.textElideMode = Qt::ElideNone;
        style->drawControl(QStyle::CE_ItemViewItem, &adjusted, painter, adjusted.widget);

        const QRect textRect =
            style->subElementRect(QStyle::SE_ItemViewItemText, &adjusted, adjusted.widget);
        const QString displayText = vc3d::adaptFiberNameToWidth(fullText,
                                                                adjusted.fontMetrics,
                                                                textRect.width());
        const QPalette::ColorRole textRole =
            adjusted.state & QStyle::State_Selected ? QPalette::HighlightedText : QPalette::Text;

        style->drawItemText(painter,
                            textRect,
                            adjusted.displayAlignment,
                            adjusted.palette,
                            adjusted.state & QStyle::State_Enabled,
                            displayText,
                            textRole);
    }
};

std::vector<QCheckBox*> tagCheckboxesInLayout(QVBoxLayout* layout)
{
    std::vector<QCheckBox*> checkboxes;
    if (!layout) {
        return checkboxes;
    }

    for (int i = 0; i < layout->count(); ++i) {
        QLayoutItem* item = layout->itemAt(i);
        if (!item) {
            continue;
        }
        if (auto* checkbox = qobject_cast<QCheckBox*>(item->widget())) {
            checkboxes.push_back(checkbox);
        }
    }
    return checkboxes;
}

QStandardItem* readOnlyItem(const QString& text = QString())
{
    auto* item = new QStandardItem(text);
    item->setEditable(false);
    return item;
}

QString formatTags(const std::vector<std::string>& tags)
{
    QStringList parts;
    parts.reserve(static_cast<qsizetype>(tags.size()));
    for (const auto& tag : tags) {
        const QString text = QString::fromStdString(tag).trimmed();
        if (!text.isEmpty()) {
            parts.push_back(text);
        }
    }
    parts.removeDuplicates();
    parts.sort();
    return parts.join(QStringLiteral(", "));
}

QString metricTooltip(const CFiberWidget::FiberEntry::AlignmentMetrics& metric,
                      bool showMetrics)
{
    if (!showMetrics) {
        return QObject::tr("Enable Calc metrics to sample Lasagna normal alignment errors.");
    }
    if (metric.pending) {
        return QObject::tr("Sampling Lasagna normals.");
    }
    if (!metric.error.empty()) {
        return QString::fromStdString(metric.error);
    }
    if (!metric.available) {
        return QObject::tr("Metric has not been calculated.");
    }
    return QObject::tr("%1 samples").arg(metric.sampleCount);
}

bool shouldHighlight(const CFiberWidget::FiberEntry::AlignmentMetrics& metric,
                     bool showMetrics)
{
    return showMetrics &&
           metric.available &&
           std::isfinite(metric.maxErrorDegrees) &&
           metric.maxErrorDegrees > kHighlightThresholdDegrees;
}

void applyRowMetadata(const QList<QStandardItem*>& row,
                      uint64_t fiberId,
                      bool isSpan,
                      const CFiberWidget::FiberEntry::AlignmentMetrics& metric,
                      bool showMetrics)
{
    const bool highlight = shouldHighlight(metric, showMetrics);
    const QColor warningColor(255, 232, 232);
    const QString tooltip = metricTooltip(metric, showMetrics);
    for (int column = 0; column < row.size(); ++column) {
        QStandardItem* item = row[column];
        if (!item) {
            continue;
        }
        item->setData(QVariant::fromValue(fiberId), kFiberIdRole);
        item->setData(isSpan, kIsSpanRole);
        if (highlight) {
            item->setBackground(warningColor);
        } else {
            item->setData(QVariant(), Qt::BackgroundRole);
        }
        if (column != kInterpStatusColumn) {
            // The interp cell keeps its predictions-provenance tooltip;
            // metric refreshes must not overwrite it.
            item->setToolTip(tooltip);
        }
    }
}

void applySpanMetadata(const QList<QStandardItem*>& row, int firstControlIndex, int secondControlIndex)
{
    for (QStandardItem* item : row) {
        if (!item) {
            continue;
        }
        item->setData(firstControlIndex, kSpanFirstControlIndexRole);
        item->setData(secondControlIndex, kSpanSecondControlIndexRole);
    }
}

} // namespace

CFiberWidget::CFiberWidget(QWidget* parent)
    : QDockWidget(tr("Fibers"), parent)
{
    setupUi();
}

CFiberWidget::~CFiberWidget() = default;

void CFiberWidget::setupUi()
{
    auto* scrollArea = new QScrollArea(this);
    scrollArea->setWidgetResizable(true);
    scrollArea->setFrameShape(QFrame::NoFrame);
    auto* mainWidget = new QWidget(scrollArea);
    auto* layout = new QVBoxLayout(mainWidget);

    // Store: the attached object store every listed and drawn fiber comes from.
    auto* storeLayout = new QHBoxLayout();
    _attachButton = new QPushButton(tr("Attach store…"), mainWidget);
    _attachButton->setObjectName(QStringLiteral("fiberAttachStoreButton"));
    _detachButton = new QPushButton(tr("Detach"), mainWidget);
    _detachButton->setObjectName(QStringLiteral("fiberDetachStoreButton"));
    _detachButton->setEnabled(false);
    storeLayout->addWidget(_attachButton);
    storeLayout->addWidget(_detachButton);
    storeLayout->addStretch(1);
    layout->addLayout(storeLayout);
    _storeStatusLabel = new QLabel(tr("No object store attached."), mainWidget);
    _storeStatusLabel->setObjectName(QStringLiteral("fiberStoreStatusLabel"));
    _storeStatusLabel->setWordWrap(true);
    layout->addWidget(_storeStatusLabel);
    connect(_attachButton, &QPushButton::clicked, this, &CFiberWidget::attachStoreRequested);
    connect(_detachButton, &QPushButton::clicked, this, &CFiberWidget::detachStoreRequested);

    // Filters: which fibers are listed and drawn.
    auto* filterLayout = new QHBoxLayout();
    auto* minLengthLabel = new QLabel(tr("Min length:"), mainWidget);
    filterLayout->addWidget(minLengthLabel);
    _minLengthSpinBox = new QDoubleSpinBox(mainWidget);
    _minLengthSpinBox->setObjectName(QStringLiteral("fiberMinLengthSpinBox"));
    _minLengthSpinBox->setRange(0.0, 1000000.0);
    _minLengthSpinBox->setDecimals(2);
    _minLengthSpinBox->setSingleStep(1.0);
    _minLengthSpinBox->setSuffix(tr(" mm"));
    _minLengthSpinBox->setKeyboardTracking(false);
    _minLengthSpinBox->setToolTip(
        tr("Minimum length of the complete fiber in millimeters, inclusive. 0 shows all lengths. "
           "Applies to the list and the drawn fibers before the display limit."));
    minLengthLabel->setBuddy(_minLengthSpinBox);
    filterLayout->addWidget(_minLengthSpinBox);
    _horizontalCheckBox = new QCheckBox(tr("H"), mainWidget);
    _horizontalCheckBox->setObjectName(QStringLiteral("fiberHorizontalCheckBox"));
    _horizontalCheckBox->setToolTip(tr("List and draw horizontal fibers."));
    filterLayout->addWidget(_horizontalCheckBox);
    _verticalCheckBox = new QCheckBox(tr("V"), mainWidget);
    _verticalCheckBox->setObjectName(QStringLiteral("fiberVerticalCheckBox"));
    _verticalCheckBox->setToolTip(tr("List and draw vertical fibers."));
    filterLayout->addWidget(_verticalCheckBox);
    filterLayout->addStretch(1);
    layout->addLayout(filterLayout);

    // Display: how the store's fibers are drawn in the volume viewers.
    auto* displayLayout = new QHBoxLayout();
    _showFibersCheckBox = new QCheckBox(tr("Show fibers"), mainWidget);
    _showFibersCheckBox->setObjectName(QStringLiteral("fiberShowFibersCheckBox"));
    _showFibersCheckBox->setToolTip(
        tr("Draw the store's fibers, and the control points of annotated fibers, near each slice."));
    displayLayout->addWidget(_showFibersCheckBox);
    _showLinkedCheckBox = new QCheckBox(tr("Show linked"), mainWidget);
    _showLinkedCheckBox->setObjectName(QStringLiteral("fiberShowLinkedCheckBox"));
    _showLinkedCheckBox->setToolTip(
        tr("Color linked fibers as one group and mark linked control points "
           "(blue = pending, purple = approved)."));
    displayLayout->addWidget(_showLinkedCheckBox);
    _rainbowCheckBox = new QCheckBox(tr("Rainbow"), mainWidget);
    _rainbowCheckBox->setObjectName(QStringLiteral("fiberRainbowCheckBox"));
    _rainbowCheckBox->setToolTip(tr("Color each fiber individually instead of by H/V family."));
    displayLayout->addWidget(_rainbowCheckBox);
    displayLayout->addStretch(1);
    layout->addLayout(displayLayout);

    auto* displayValuesLayout = new QHBoxLayout();
    auto* sliceDistanceLabel = new QLabel(tr("Slice ±"), mainWidget);
    displayValuesLayout->addWidget(sliceDistanceLabel);
    _sliceDistanceSpinBox = new QDoubleSpinBox(mainWidget);
    _sliceDistanceSpinBox->setObjectName(QStringLiteral("fiberSliceDistanceSpinBox"));
    _sliceDistanceSpinBox->setRange(0.1, 1000.0);
    _sliceDistanceSpinBox->setDecimals(1);
    _sliceDistanceSpinBox->setSingleStep(1.0);
    _sliceDistanceSpinBox->setSuffix(tr(" vx"));
    _sliceDistanceSpinBox->setKeyboardTracking(false);
    _sliceDistanceSpinBox->setToolTip(
        tr("Maximum distance from the current slice, in volume voxels, at which fibers are drawn."));
    sliceDistanceLabel->setBuddy(_sliceDistanceSpinBox);
    displayValuesLayout->addWidget(_sliceDistanceSpinBox);
    auto* maxDisplayedLabel = new QLabel(tr("Max shown"), mainWidget);
    displayValuesLayout->addWidget(maxDisplayedLabel);
    _maxDisplayedSpinBox = new QSpinBox(mainWidget);
    _maxDisplayedSpinBox->setObjectName(QStringLiteral("fiberMaxDisplayedSpinBox"));
    _maxDisplayedSpinBox->setRange(0, 10000000);
    _maxDisplayedSpinBox->setSpecialValueText(tr("All"));
    _maxDisplayedSpinBox->setSingleStep(500);
    _maxDisplayedSpinBox->setKeyboardTracking(false);
    _maxDisplayedSpinBox->setToolTip(
        tr("Maximum fibers drawn in each viewport, sampled only among fibers intersecting its "
           "current slice. Stable UUID order. 0 shows all. A visible selected fiber is included "
           "within this limit."));
    maxDisplayedLabel->setBuddy(_maxDisplayedSpinBox);
    displayValuesLayout->addWidget(_maxDisplayedSpinBox);
    auto* lineWidthLabel = new QLabel(tr("Width"), mainWidget);
    displayValuesLayout->addWidget(lineWidthLabel);
    _lineWidthSpinBox = new QDoubleSpinBox(mainWidget);
    _lineWidthSpinBox->setObjectName(QStringLiteral("fiberLineWidthSpinBox"));
    _lineWidthSpinBox->setRange(0.5, 12.0);
    _lineWidthSpinBox->setDecimals(1);
    _lineWidthSpinBox->setSingleStep(0.5);
    _lineWidthSpinBox->setSuffix(tr(" px"));
    lineWidthLabel->setBuddy(_lineWidthSpinBox);
    displayValuesLayout->addWidget(_lineWidthSpinBox);
    displayValuesLayout->addStretch(1);
    layout->addLayout(displayValuesLayout);

    setDisplayOptions(FiberDisplayOptions{});
    for (auto* checkBox : {_horizontalCheckBox, _verticalCheckBox, _showFibersCheckBox,
                           _showLinkedCheckBox, _rainbowCheckBox}) {
        connect(checkBox, &QCheckBox::toggled, this, &CFiberWidget::emitDisplayOptions);
    }
    for (auto* spinBox : {_minLengthSpinBox, _sliceDistanceSpinBox, _lineWidthSpinBox}) {
        connect(spinBox, qOverload<double>(&QDoubleSpinBox::valueChanged),
                this, &CFiberWidget::emitDisplayOptions);
    }
    connect(_maxDisplayedSpinBox, qOverload<int>(&QSpinBox::valueChanged),
            this, &CFiberWidget::emitDisplayOptions);

    _totalLengthLabel = new QLabel(mainWidget);
    _totalLengthLabel->setObjectName(QStringLiteral("fiberTotalLengthLabel"));
    _totalLengthLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
    _totalLengthLabel->setToolTip(
        tr("Sum of the listed fibers' line lengths. Centimetres when the "
           "annotation frame's voxel size is known, otherwise voxels."));
    layout->addWidget(_totalLengthLabel);

    _calcMetricsCheckBox = new QCheckBox(tr("Calc metrics"), mainWidget);
    _calcMetricsCheckBox->setObjectName(QStringLiteral("fiberCalcMetricsCheckBox"));
    _calcMetricsCheckBox->setToolTip(
        tr("Sample Lasagna normal alignment errors for the listed fibers."));
    layout->addWidget(_calcMetricsCheckBox);
    connect(_calcMetricsCheckBox, &QCheckBox::toggled, this, [this](bool checked) {
        refreshMetricDisplays();
        if (checked) {
            emit metricsCalculationRequested(orderedFiberIds());
        }
    });

    _model = new QStandardItemModel(this);
    _model->setColumnCount(kColumnCount);
    _model->setHorizontalHeaderLabels(headerLabels());
    _treeView = new QTreeView(mainWidget);
    _treeView->setObjectName(QStringLiteral("fiberTreeView"));
    _treeView->setModel(_model);
    _treeView->setItemDelegateForColumn(kNameColumn, new FiberNameDelegate(_treeView));
    _treeView->setEditTriggers(QAbstractItemView::NoEditTriggers);
    _treeView->setSelectionBehavior(QAbstractItemView::SelectRows);
    _treeView->setSelectionMode(QAbstractItemView::ExtendedSelection);
    _treeView->setContextMenuPolicy(Qt::CustomContextMenu);
    _treeView->setAlternatingRowColors(true);
    _treeView->setRootIsDecorated(true);
    _treeView->setItemsExpandable(true);
    _treeView->header()->setSectionResizeMode(QHeaderView::Interactive);
    _treeView->header()->setStretchLastSection(false);
    _treeView->header()->setSectionsClickable(true);
    _treeView->header()->setSortIndicatorShown(true);
    _treeView->header()->setSortIndicator(_sortColumn, _sortOrder);
    _treeView->setColumnWidth(kNameColumn, 220);
    _treeView->setColumnWidth(kDirectionColumn, 42);
    _treeView->setColumnWidth(kLinkColumn, 42);
    _treeView->setColumnWidth(kPendingColumn, 56);
    _treeView->setColumnWidth(kLengthColumn, 72);
    _treeView->setColumnWidth(kControlPointsColumn, 48);
    _treeView->setColumnWidth(kLinePointsColumn, 48);
    _treeView->setColumnWidth(kTagsColumn, 110);
    _treeView->setColumnWidth(kInterpStatusColumn, 150);
    layout->addWidget(_treeView, 1);

    connect(_treeView->selectionModel(), &QItemSelectionModel::selectionChanged,
            this, &CFiberWidget::onSelectionChanged);
    connect(_treeView, &QTreeView::expanded,
            this, &CFiberWidget::ensureSpanChildrenBuilt);
    connect(_treeView, &QTreeView::doubleClicked,
            this, &CFiberWidget::onDoubleClicked);
    connect(_treeView, &QWidget::customContextMenuRequested,
            this, &CFiberWidget::showContextMenu);
    connect(_treeView->header(), &QHeaderView::sectionClicked,
            this, &CFiberWidget::onHeaderSectionClicked);

    _treeView->setMinimumHeight(160);
    _treeView->setToolTip(tr("Click a fiber to select it and focus the main views; "
                             "double-click to open it in Line Annotation."));
    connect(_treeView, &QTreeView::clicked, this, [this](const QModelIndex& index) {
        const auto selected = selectedFiberIds();
        auto* item = _model->itemFromIndex(index);
        if (item && selected.size() == 1 && item->data(kFiberIdRole).toULongLong() == selected.front()) {
            emit fiberActivated(selected.front());
        }
    });

    // Paging: the store lists fibers longest first, one page at a time.
    auto* pageLayout = new QHBoxLayout();
    _firstPageButton = new QPushButton(tr("First page"), mainWidget);
    _firstPageButton->setObjectName(QStringLiteral("fiberFirstPageButton"));
    _nextPageButton = new QPushButton(tr("Next page"), mainWidget);
    _nextPageButton->setObjectName(QStringLiteral("fiberNextPageButton"));
    _fiberIdEdit = new QLineEdit(mainWidget);
    _fiberIdEdit->setObjectName(QStringLiteral("fiberIdEdit"));
    _fiberIdEdit->setPlaceholderText(tr("Fiber ID · Enter to select"));
    pageLayout->addWidget(_firstPageButton);
    pageLayout->addWidget(_nextPageButton);
    pageLayout->addWidget(_fiberIdEdit, 1);
    layout->addLayout(pageLayout);
    connect(_firstPageButton, &QPushButton::clicked, this, &CFiberWidget::firstPageRequested);
    connect(_nextPageButton, &QPushButton::clicked, this, &CFiberWidget::nextPageRequested);
    connect(_fiberIdEdit, &QLineEdit::returnPressed, this, [this]() {
        bool ok = false;
        const uint64_t id = _fiberIdEdit->text().toULongLong(&ok);
        if (ok && id != 0) {
            emit fiberActivated(id);
        }
    });

    _nameLabel = new QLabel(mainWidget);
    _nameLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
    _nameLabel->setMinimumHeight(_nameLabel->fontMetrics().lineSpacing() * 2);
    layout->addWidget(_nameLabel);

    _scoreLabel = new QLabel(mainWidget);
    _scoreLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
    layout->addWidget(_scoreLabel);

    _autoLabel = new QLabel(mainWidget);
    _autoLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
    layout->addWidget(_autoLabel);

    // The selected fiber's stored trace: follow it, or open it in Line Annotation.
    _traceLabel = new QLabel(tr("Click a trace to select it in the list."), mainWidget);
    _traceLabel->setObjectName(QStringLiteral("fiberTraceLabel"));
    _traceLabel->setWordWrap(true);
    layout->addWidget(_traceLabel);
    _traceSlider = new QSlider(Qt::Horizontal, mainWidget);
    _traceSlider->setObjectName(QStringLiteral("fiberTraceSlider"));
    _traceSlider->setEnabled(false);
    layout->addWidget(_traceSlider);
    connect(_traceSlider, &QSlider::valueChanged, this, &CFiberWidget::traceIndexChanged);
    _openInLineAnnotationButton = new QPushButton(tr("Open in Line Annotation"), mainWidget);
    _openInLineAnnotationButton->setObjectName(QStringLiteral("fiberOpenInLineAnnotationButton"));
    _openInLineAnnotationButton->setEnabled(false);
    layout->addWidget(_openInLineAnnotationButton);
    connect(_openInLineAnnotationButton, &QPushButton::clicked, this, [this]() {
        if (_selectedFiberId != 0) {
            emit openInLineAnnotationRequested(_selectedFiberId);
        }
    });
    _annotationStatusLabel = new QLabel(mainWidget);
    _annotationStatusLabel->setObjectName(QStringLiteral("fiberAnnotationStatusLabel"));
    _annotationStatusLabel->setWordWrap(true);
    _annotationStatusLabel->hide();
    layout->addWidget(_annotationStatusLabel);

    layout->addWidget(new QLabel(tr("Tags:"), mainWidget));
    _tagListWidget = new QWidget(mainWidget);
    _tagListLayout = new QVBoxLayout(_tagListWidget);
    _tagListLayout->setContentsMargins(0, 0, 0, 0);
    _tagListLayout->setSpacing(2);
    layout->addWidget(_tagListWidget);

    auto* addTagLayout = new QHBoxLayout();
    _newTagEdit = new QLineEdit(mainWidget);
    _newTagEdit->setObjectName(QStringLiteral("fiberNewTagEdit"));
    _newTagEdit->setPlaceholderText(tr("New tag"));
    _addTagButton = new QPushButton(tr("Add"), mainWidget);
    _addTagButton->setObjectName(QStringLiteral("fiberAddTagButton"));
    addTagLayout->addWidget(_newTagEdit, 1);
    addTagLayout->addWidget(_addTagButton);
    layout->addLayout(addTagLayout);

    connect(_newTagEdit, &QLineEdit::returnPressed, this, &CFiberWidget::onAddTagClicked);
    connect(_addTagButton, &QPushButton::clicked, this, &CFiberWidget::onAddTagClicked);

    auto* manualLayout = new QHBoxLayout();
    manualLayout->addWidget(new QLabel(tr("Manual:"), mainWidget));
    _manualHButton = new QPushButton(tr("H"), mainWidget);
    _manualVButton = new QPushButton(tr("V"), mainWidget);
    _manualResetButton = new QPushButton(tr("Reset"), mainWidget);
    _manualHButton->setCheckable(true);
    _manualVButton->setCheckable(true);
    _manualHvGroup = new QButtonGroup(this);
    _manualHvGroup->setExclusive(true);
    _manualHvGroup->addButton(_manualHButton, 0);
    _manualHvGroup->addButton(_manualVButton, 1);
    manualLayout->addWidget(_manualHButton);
    manualLayout->addWidget(_manualVButton);
    manualLayout->addWidget(_manualResetButton);
    manualLayout->addStretch(1);
    layout->addLayout(manualLayout);

    connect(_manualHButton, &QPushButton::clicked, this, [this]() {
        onManualHvButtonClicked(0);
    });
    connect(_manualVButton, &QPushButton::clicked, this, [this]() {
        onManualHvButtonClicked(1);
    });
    connect(_manualResetButton, &QPushButton::clicked,
            this, &CFiberWidget::onManualHvResetClicked);

    _recalculateScoreButton = new QPushButton(tr("Recalc score"), mainWidget);
    layout->addWidget(_recalculateScoreButton);
    connect(_recalculateScoreButton, &QPushButton::clicked,
            this, &CFiberWidget::onRecalculateHvScoreClicked);

    auto* buttonLayout = new QHBoxLayout();
    _importButton = new QPushButton(tr("Import"), mainWidget);
    _importButton->setObjectName(QStringLiteral("fiberImportButton"));
    buttonLayout->addWidget(_importButton);
    _exportButton = new QPushButton(tr("Export"), mainWidget);
    _exportButton->setObjectName(QStringLiteral("fiberExportButton"));
    buttonLayout->addWidget(_exportButton);
    _deleteButton = new QPushButton(tr("Delete"), mainWidget);
    _deleteButton->setObjectName(QStringLiteral("fiberDeleteButton"));
    _deleteButton->setEnabled(false);
    buttonLayout->addWidget(_deleteButton);
    buttonLayout->addStretch(1);
    layout->addLayout(buttonLayout);

    connect(_importButton, &QPushButton::clicked, this, [this]() {
        emit importFibersRequested();
    });
    connect(_exportButton, &QPushButton::clicked, this, [this]() {
        emit exportFibersRequested();
    });
    connect(_deleteButton, &QPushButton::clicked, this, &CFiberWidget::onDeleteClicked);

    updateClassificationUi();
    setStoreStatus(false, tr("No object store attached."));
    scrollArea->setWidget(mainWidget);
    setWidget(scrollArea);
}

void CFiberWidget::setDisplayOptions(const FiberDisplayOptions& options)
{
    const QSignalBlocker horizontal(_horizontalCheckBox);
    const QSignalBlocker vertical(_verticalCheckBox);
    const QSignalBlocker showFibers(_showFibersCheckBox);
    const QSignalBlocker showLinked(_showLinkedCheckBox);
    const QSignalBlocker rainbow(_rainbowCheckBox);
    const QSignalBlocker minLength(_minLengthSpinBox);
    const QSignalBlocker sliceDistance(_sliceDistanceSpinBox);
    const QSignalBlocker maxDisplayed(_maxDisplayedSpinBox);
    const QSignalBlocker lineWidth(_lineWidthSpinBox);
    _horizontalCheckBox->setChecked(options.horizontal);
    _verticalCheckBox->setChecked(options.vertical);
    _showFibersCheckBox->setChecked(options.showFibers);
    _showLinkedCheckBox->setChecked(options.showLinked);
    _rainbowCheckBox->setChecked(options.rainbow);
    _minLengthSpinBox->setValue(options.minLengthMm);
    _sliceDistanceSpinBox->setValue(options.sliceDistance);
    _maxDisplayedSpinBox->setValue(options.maxDisplayed);
    _lineWidthSpinBox->setValue(options.lineWidth);
}

FiberDisplayOptions CFiberWidget::displayOptions() const
{
    FiberDisplayOptions options;
    options.horizontal = _horizontalCheckBox->isChecked();
    options.vertical = _verticalCheckBox->isChecked();
    options.showFibers = _showFibersCheckBox->isChecked();
    options.showLinked = _showLinkedCheckBox->isChecked();
    options.rainbow = _rainbowCheckBox->isChecked();
    options.minLengthMm = _minLengthSpinBox->value();
    options.sliceDistance = _sliceDistanceSpinBox->value();
    options.maxDisplayed = _maxDisplayedSpinBox->value();
    options.lineWidth = _lineWidthSpinBox->value();
    return options;
}

void CFiberWidget::emitDisplayOptions()
{
    emit displayOptionsChanged(displayOptions());
}

void CFiberWidget::setStoreStatus(bool attached, const QString& text)
{
    _storeStatusLabel->setText(text);
    _detachButton->setEnabled(attached);
    _firstPageButton->setEnabled(attached);
    _fiberIdEdit->setEnabled(attached);
    if (!attached) {
        _nextPageButton->setEnabled(false);
    }
}

void CFiberWidget::setNextPageAvailable(bool available)
{
    _nextPageButton->setEnabled(available);
}

void CFiberWidget::setTraceSelection(const QString& text,
                                     const QString& toolTip,
                                     int64_t pointCount,
                                     int index,
                                     bool openable)
{
    _traceLabel->setText(text.isEmpty() ? tr("Click a trace to select it in the list.") : text);
    _traceLabel->setToolTip(toolTip);
    const QSignalBlocker blocker(_traceSlider);
    const int last = static_cast<int>(std::clamp<int64_t>(pointCount - 1, 0, std::numeric_limits<int>::max()));
    _traceSlider->setRange(0, last);
    _traceSlider->setValue(std::clamp(index, 0, last));
    _traceSlider->setEnabled(pointCount > 0);
    _openInLineAnnotationButton->setEnabled(openable && pointCount > 0);
}

void CFiberWidget::setAnnotationStatus(const QString& text)
{
    _annotationStatusLabel->setText(text);
    _annotationStatusLabel->setVisible(!text.isEmpty());
}

void CFiberWidget::setLineAnnotationActive(bool active)
{
    _lineAnnotationActive = active;
    _treeView->setToolTip(active
        ? tr("Click a fiber to select it; double-click to open it in Line Annotation.")
        : tr("Click a fiber to select it and focus the main views; "
             "double-click to open it in Line Annotation."));
    _traceLabel->setVisible(!active);
    _traceSlider->setVisible(!active);
    _openInLineAnnotationButton->setVisible(!active);
}

QString CFiberWidget::displayNameForFiber(const FiberEntry& fiber)
{
    const QString name = vc3d::displayStemForFiberFile(QString::fromStdString(fiber.fileName));
    return name.isEmpty() ? tr("unnamed") : name;
}

QString CFiberWidget::directionForFiber(const FiberEntry& fiber)
{
    if (fiber.manualHvTag == "H" || fiber.manualHvTag == "V") {
        return QString::fromStdString(fiber.manualHvTag);
    }
    if (fiber.automaticHvTag == "H" || fiber.automaticHvTag == "V") {
        return QString::fromStdString(fiber.automaticHvTag);
    }
    return QStringLiteral("-");
}

QString CFiberWidget::statusTextForFiber(const FiberEntry& fiber)
{
    // Interpolation provenance only; the human review verdict lives in the
    // tags column as the ordinary 'reviewed' tag.
    switch (fiber.traceState) {
    case FiberEntry::TraceState::Predictions:
        return tr("predictions");
    case FiberEntry::TraceState::Mixed:
        return tr("mixed");
    case FiberEntry::TraceState::Legacy:
        break;
    }
    return tr("legacy");
}

std::vector<uint64_t> CFiberWidget::selectedFiberIds() const
{
    std::vector<uint64_t> ids;
    if (!_treeView || !_treeView->selectionModel()) {
        return ids;
    }

    const auto indexes = _treeView->selectionModel()->selectedRows(kNameColumn);
    ids.reserve(static_cast<size_t>(indexes.size()));
    for (const QModelIndex& index : indexes) {
        auto* item = _model->itemFromIndex(index);
        if (item) {
            const uint64_t id = item->data(kFiberIdRole).toULongLong();
            if (id != 0) {
                ids.push_back(id);
            }
        }
    }
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    return ids;
}

bool CFiberWidget::canDeleteSelection() const
{
    return !selectedFiberIds().empty();
}

bool CFiberWidget::canShowFiberSlice() const
{
    return selectedFiberIds().size() == 1;
}

bool CFiberWidget::canRenameFiberFile() const
{
    return selectedFiberIds().size() == 1;
}

std::vector<uint64_t> CFiberWidget::orderedFiberIds() const
{
    std::vector<uint64_t> ids;
    ids.reserve(_listedFibers.size());
    for (const auto& fiber : _listedFibers) {
        if (fiber.id != 0) {
            ids.push_back(fiber.id);
        }
    }
    return ids;
}

QAction* CFiberWidget::createShowFiberSliceAction(QObject* parent)
{
    auto* action = new QAction(tr("Show fiber slice"), parent);
    action->setObjectName(QStringLiteral("showFiberSliceAction"));
    action->setEnabled(canShowFiberSlice());
    connect(action, &QAction::triggered, this, [this]() {
        requestShowFiberSlice();
    });
    return action;
}

QAction* CFiberWidget::createRenameFiberFileAction(QObject* parent)
{
    auto* action = new QAction(tr("Rename fiber..."), parent);
    action->setObjectName(QStringLiteral("renameFiberFileAction"));
    action->setEnabled(canRenameFiberFile());
    connect(action, &QAction::triggered, this, [this]() {
        requestRenameFiberFile();
    });
    return action;
}

void CFiberWidget::setFibers(const std::vector<FiberEntry>& fibers)
{
    const std::vector<uint64_t> previousSelection = selectedFiberIds();
    _listedFibers = fibers;
    sortFibers();
    for (const auto& fiber : _listedFibers) {
        for (const auto& tag : fiber.tags) {
            addUniqueSorted(_knownTags, tag);
        }
    }

    rebuildModel();
    updateTotalLengthLabel();

    _selectedFiberId = 0;
    if (!previousSelection.empty()) {
        selectFibers(previousSelection);
    }
    _deleteButton->setEnabled(canDeleteSelection());
    updateClassificationUi();
    if (_calcMetricsCheckBox && _calcMetricsCheckBox->isChecked()) {
        emit metricsCalculationRequested(orderedFiberIds());
    }
}

void CFiberWidget::updateFiberDetails(const FiberEntry& fiber)
{
    auto it = std::find_if(_listedFibers.begin(), _listedFibers.end(), [&](const FiberEntry& listed) {
        return listed.id == fiber.id;
    });
    if (it == _listedFibers.end()) {
        return;
    }
    const std::vector<uint64_t> previousSelection = selectedFiberIds();
    *it = fiber;
    for (const auto& tag : fiber.tags) {
        addUniqueSorted(_knownTags, tag);
    }
    // Update the row in place: its index must survive, or a double click
    // whose first click requested these details would lose its press.
    if (QStandardItem* root = findFiberItem(fiber.id)) {
        fillFiberRow(rowItemsForNameItem(root), fiber);
        const bool expanded = _treeView->isExpanded(root->index());
        root->removeRows(0, root->rowCount());
        fillSpanChildren(root, fiber, expanded);
        if (expanded) {
            _treeView->setExpanded(root->index(), true);
        }
    }
    updateTotalLengthLabel();
    selectFibers(previousSelection);
    updateClassificationUi();
}

void CFiberWidget::revealFiber(const FiberEntry& fiber)
{
    if (std::none_of(_listedFibers.begin(), _listedFibers.end(), [&](const FiberEntry& listed) { return listed.id == fiber.id; })) {
        _listedFibers.insert(_listedFibers.begin(), fiber);
        sortFibers();
        rebuildModel();
        updateTotalLengthLabel();
    }
    selectFiber(fiber.id);
}

void CFiberWidget::setAnnotationVoxelSizeUm(std::optional<double> voxelSizeUm)
{
    if (!vc3d::fiber_length::hasPhysicalScale(voxelSizeUm)) {
        voxelSizeUm.reset();
    }
    if (voxelSizeUm == _voxelSizeUm) {
        return;
    }
    _voxelSizeUm = voxelSizeUm;
    refreshLengthDisplays();
}

double CFiberWidget::totalLengthVx() const
{
    double total = 0.0;
    for (const auto& fiber : _listedFibers) {
        if (std::isfinite(fiber.lengthVx)) {
            total += fiber.lengthVx;
        }
    }
    return total;
}

QStringList CFiberWidget::headerLabels() const
{
    return {
        tr("name"),
        tr("dir"),
        tr("link"),
        tr("pending"),
        vc3d::fiber_length::columnHeader(tr("len"), _voxelSizeUm),
        tr("cps"),
        tr("pts"),
        tr("tags"),
        tr("interp"),
    };
}

void CFiberWidget::refreshLengthDisplays()
{
    if (_model) {
        _model->setHorizontalHeaderLabels(headerLabels());
        // One lookup table for the pass, not a search of _listedFibers per row.
        std::unordered_map<uint64_t, const FiberEntry*> byId;
        byId.reserve(_listedFibers.size());
        for (const FiberEntry& fiber : _listedFibers) {
            byId.emplace(fiber.id, &fiber);
        }
        for (int row = 0; row < _model->rowCount(); ++row) {
            QStandardItem* root = _model->item(row, kNameColumn);
            if (!root) {
                continue;
            }
            const auto found = byId.find(root->data(kFiberIdRole).toULongLong());
            if (found == byId.end()) {
                continue;
            }
            const FiberEntry* it = found->second;
            if (QStandardItem* cell = _model->item(row, kLengthColumn)) {
                cell->setText(vc3d::fiber_length::formatValue(it->lengthVx, _voxelSizeUm));
            }
            if (spanChildrenArePlaceholder(root)) {
                continue;
            }
            for (int child = 0;
                 child < root->rowCount() && child < static_cast<int>(it->spans.size());
                 ++child) {
                if (QStandardItem* cell = root->child(child, kLengthColumn)) {
                    cell->setText(vc3d::fiber_length::formatValue(
                        it->spans[static_cast<size_t>(child)].lengthVx, _voxelSizeUm));
                }
            }
        }
    }
    updateTotalLengthLabel();
    updateClassificationUi();
}

void CFiberWidget::updateTotalLengthLabel()
{
    if (!_totalLengthLabel) {
        return;
    }
    _totalLengthLabel->setText(
        tr("Total fiber length: %1  (%2 fibers)")
            .arg(vc3d::fiber_length::formatLength(totalLengthVx(), _voxelSizeUm))
            .arg(_listedFibers.size()));
}

void CFiberWidget::updateAlignmentMetrics(
    uint64_t fiberId,
    const FiberEntry::AlignmentMetrics& alignment,
    const std::vector<FiberEntry::AlignmentMetrics>& spanAlignments)
{
    auto fiberIt = std::find_if(_listedFibers.begin(), _listedFibers.end(), [fiberId](const FiberEntry& fiber) {
        return fiber.id == fiberId;
    });
    if (fiberIt == _listedFibers.end()) {
        return;
    }

    fiberIt->alignment = alignment;
    const size_t spanCount = std::min(fiberIt->spans.size(), spanAlignments.size());
    for (size_t i = 0; i < spanCount; ++i) {
        fiberIt->spans[i].alignment = spanAlignments[i];
    }

    QStandardItem* root = findFiberItem(fiberId);
    if (!root) {
        return;
    }
    updateMetricDisplayForRow(root, alignment);
    if (spanChildrenArePlaceholder(root)) {
        return;
    }
    for (int row = 0; row < root->rowCount() && row < static_cast<int>(spanAlignments.size()); ++row) {
        updateMetricDisplayForRow(root->child(row, kNameColumn), spanAlignments[static_cast<size_t>(row)]);
    }
}

void CFiberWidget::rebuildModel()
{
    std::set<uint64_t> expandedFibers;
    if (_treeView) {
        for (int row = 0; row < _model->rowCount(); ++row) {
            QStandardItem* item = _model->item(row, kNameColumn);
            if (item && _treeView->isExpanded(item->index())) {
                expandedFibers.insert(item->data(kFiberIdRole).toULongLong());
            }
        }
    }

    _model->removeRows(0, _model->rowCount());
    _model->setColumnCount(kColumnCount);
    _model->setHorizontalHeaderLabels(headerLabels());
    if (_treeView && _treeView->header()) {
        _treeView->header()->setSortIndicator(_sortColumn, _sortOrder);
    }

    for (const auto& fiber : _listedFibers) {
        QList<QStandardItem*> row;
        for (int column = 0; column < kColumnCount; ++column) {
            row.push_back(readOnlyItem());
        }
        fillFiberRow(row, fiber);
        QStandardItem* root = row[kNameColumn];
        const bool wasExpanded =
            _treeView && expandedFibers.find(fiber.id) != expandedFibers.end();
        fillSpanChildren(root, fiber, wasExpanded);
        _model->appendRow(row);
        if (wasExpanded) {
            _treeView->setExpanded(root->index(), true);
        }
    }
}

void CFiberWidget::fillFiberRow(const QList<QStandardItem*>& row, const FiberEntry& fiber)
{
    const bool showMetrics = _calcMetricsCheckBox && _calcMetricsCheckBox->isChecked();
    row[kNameColumn]->setText(displayNameForFiber(fiber));
    row[kDirectionColumn]->setText(directionForFiber(fiber));
    row[kLinkColumn]->setText(fiber.linkedFiberCount > 0 ? QString::number(fiber.linkedFiberCount) : QString());
    row[kPendingColumn]->setText(fiber.pendingLinkCount > 0 ? QString::number(fiber.pendingLinkCount) : QString());
    row[kLengthColumn]->setText(vc3d::fiber_length::formatValue(fiber.lengthVx, _voxelSizeUm));
    row[kControlPointsColumn]->setText(QString::number(fiber.controlPointCount));
    row[kLinePointsColumn]->setText(QString::number(fiber.linePointCount));
    row[kTagsColumn]->setText(formatTags(fiber.tags));
    row[kInterpStatusColumn]->setText(statusTextForFiber(fiber));
    applyRowMetadata(row, fiber.id, false, fiber.alignment, showMetrics);
    // Predictions provenance: which fiber-inference manifests produced
    // the traced spans, shown on the interp status cell.
    QStringList tracedManifests;
    for (const auto& span : fiber.spans) {
        const QString manifest = QString::fromStdString(span.fiberManifest);
        if (!manifest.isEmpty() && !tracedManifests.contains(manifest)) {
            tracedManifests.push_back(manifest);
        }
    }
    row[kInterpStatusColumn]->setToolTip(tracedManifests.isEmpty()
        ? QString()
        : tr("Traced with:\n%1").arg(tracedManifests.join(QStringLiteral("\n"))));
}

void CFiberWidget::fillSpanChildren(QStandardItem* root, const FiberEntry& fiber, bool expanded)
{
    if (expanded && fiber.detailsLoaded) {
        buildSpanChildRows(root, fiber);
    } else if (!fiber.spans.empty() || (!fiber.detailsLoaded && fiber.controlPointCount >= 2)) {
        // Placeholder keeps the expand arrow; the real span rows (the
        // bulk of the model) are built on first expansion.
        QStandardItem* placeholder = readOnlyItem(QString());
        placeholder->setData(QVariant::fromValue(fiber.id), kFiberIdRole);
        placeholder->setData(true, kSpanPlaceholderRole);
        root->appendRow(placeholder);
    }
}

void CFiberWidget::buildSpanChildRows(QStandardItem* root, const FiberEntry& fiber)
{
    const bool showMetrics = _calcMetricsCheckBox && _calcMetricsCheckBox->isChecked();
    for (const auto& span : fiber.spans) {
        const QString spanName = tr("span %1  cp %2-%3")
            .arg(span.spanIndex + 1)
            .arg(span.firstControlIndex + 1)
            .arg(span.secondControlIndex + 1);
        QList<QStandardItem*> childRow{
            readOnlyItem(spanName),
            readOnlyItem(directionForFiber(fiber)),
            readOnlyItem(QString()),
            readOnlyItem(QString()),
            readOnlyItem(vc3d::fiber_length::formatValue(span.lengthVx, _voxelSizeUm)),
            readOnlyItem(QString::number(span.controlPointCount)),
            readOnlyItem(QString::number(span.linePointCount)),
            readOnlyItem(QString()),
            readOnlyItem(QString(QChar::fromLatin1(span.interpMarker))),
        };
        applyRowMetadata(childRow, fiber.id, true, span.alignment, showMetrics);
        applySpanMetadata(childRow, span.firstControlIndex, span.secondControlIndex);
        if (!span.fiberManifest.empty()) {
            childRow[kInterpStatusColumn]->setToolTip(
                tr("Traced with: %1")
                    .arg(QString::fromStdString(span.fiberManifest)));
        }
        root->appendRow(childRow);
    }
}

bool CFiberWidget::spanChildrenArePlaceholder(QStandardItem* root) const
{
    if (!root || root->rowCount() != 1) {
        return false;
    }
    QStandardItem* child = root->child(0, kNameColumn);
    return child && child->data(kSpanPlaceholderRole).toBool();
}

void CFiberWidget::ensureSpanChildrenBuilt(const QModelIndex& index)
{
    if (!_model || !index.isValid()) {
        return;
    }
    QStandardItem* root = _model->itemFromIndex(index.siblingAtColumn(kNameColumn));
    if (!root || root->parent() || !spanChildrenArePlaceholder(root)) {
        return;
    }
    const uint64_t fiberId = root->data(kFiberIdRole).toULongLong();
    const auto fiberIt = std::find_if(
        _listedFibers.begin(), _listedFibers.end(),
        [fiberId](const FiberEntry& fiber) { return fiber.id == fiberId; });
    if (fiberIt != _listedFibers.end() && !fiberIt->detailsLoaded) {
        emit fiberDetailsRequested(fiberId);
        return;
    }
    root->removeRows(0, 1);
    if (fiberIt != _listedFibers.end()) {
        buildSpanChildRows(root, *fiberIt);
    }
}

QList<QStandardItem*> CFiberWidget::rowItemsForNameItem(QStandardItem* nameItem) const
{
    QList<QStandardItem*> row;
    if (!nameItem || !_model) {
        return row;
    }
    row.reserve(kColumnCount);
    QStandardItem* parent = nameItem->parent();
    const int rowIndex = nameItem->row();
    for (int column = 0; column < kColumnCount; ++column) {
        row.push_back(parent ? parent->child(rowIndex, column)
                             : _model->item(rowIndex, column));
    }
    return row;
}

void CFiberWidget::updateMetricDisplayForRow(
    QStandardItem* nameItem,
    const FiberEntry::AlignmentMetrics& alignment)
{
    QList<QStandardItem*> row = rowItemsForNameItem(nameItem);
    if (row.size() < kColumnCount || !row[kNameColumn]) {
        return;
    }

    const bool showMetrics = _calcMetricsCheckBox && _calcMetricsCheckBox->isChecked();
    const uint64_t fiberId = row[kNameColumn]->data(kFiberIdRole).toULongLong();
    const bool isSpan = row[kNameColumn]->data(kIsSpanRole).toBool();
    applyRowMetadata(row, fiberId, isSpan, alignment, showMetrics);
}

void CFiberWidget::refreshMetricDisplays()
{
    for (const auto& fiber : _listedFibers) {
        QStandardItem* root = findFiberItem(fiber.id);
        if (!root) {
            continue;
        }
        updateMetricDisplayForRow(root, fiber.alignment);
        if (spanChildrenArePlaceholder(root)) {
            continue;
        }
        for (int row = 0; row < root->rowCount() && row < static_cast<int>(fiber.spans.size()); ++row) {
            updateMetricDisplayForRow(root->child(row, kNameColumn),
                                      fiber.spans[static_cast<size_t>(row)].alignment);
        }
    }
}

void CFiberWidget::sortFibers()
{
    const int column = std::clamp(_sortColumn, 0, kColumnCount - 1);
    const bool ascending = _sortOrder == Qt::AscendingOrder;
    auto compareText = [ascending](const QString& lhs, const QString& rhs) {
        const int cmp = QString::localeAwareCompare(lhs, rhs);
        return ascending ? cmp < 0 : cmp > 0;
    };
    auto compareNumber = [ascending](double lhs, double rhs) {
        if (lhs == rhs) {
            return false;
        }
        return ascending ? lhs < rhs : lhs > rhs;
    };
    std::stable_sort(_listedFibers.begin(), _listedFibers.end(), [&](const FiberEntry& lhs, const FiberEntry& rhs) {
        if (column == kLengthColumn) {
            const bool a = std::isfinite(lhs.lengthVx), b = std::isfinite(rhs.lengthVx);
            if (a != b) return a;
            if (!a) return lhs.id < rhs.id;
        }
        bool different = false;
        bool less = false;
        switch (column) {
        case kNameColumn: {
            const QString a = displayNameForFiber(lhs);
            const QString b = displayNameForFiber(rhs);
            different = QString::localeAwareCompare(a, b) != 0;
            less = compareText(a, b);
            break;
        }
        case kDirectionColumn: {
            const QString a = directionForFiber(lhs);
            const QString b = directionForFiber(rhs);
            different = QString::localeAwareCompare(a, b) != 0;
            less = compareText(a, b);
            break;
        }
        case kLinkColumn:
            different = lhs.linkedFiberCount != rhs.linkedFiberCount;
            less = compareNumber(lhs.linkedFiberCount, rhs.linkedFiberCount);
            break;
        case kPendingColumn:
            different = lhs.pendingLinkCount != rhs.pendingLinkCount;
            less = compareNumber(lhs.pendingLinkCount, rhs.pendingLinkCount);
            break;
        case kLengthColumn:
            different = lhs.lengthVx != rhs.lengthVx;
            less = compareNumber(lhs.lengthVx, rhs.lengthVx);
            break;
        case kControlPointsColumn:
            different = lhs.controlPointCount != rhs.controlPointCount;
            less = compareNumber(lhs.controlPointCount, rhs.controlPointCount);
            break;
        case kLinePointsColumn:
            different = lhs.linePointCount != rhs.linePointCount;
            less = compareNumber(lhs.linePointCount, rhs.linePointCount);
            break;
        case kTagsColumn: {
            const QString a = formatTags(lhs.tags);
            const QString b = formatTags(rhs.tags);
            different = QString::localeAwareCompare(a, b) != 0;
            less = compareText(a, b);
            break;
        }
        case kInterpStatusColumn: {
            const QString a = statusTextForFiber(lhs);
            const QString b = statusTextForFiber(rhs);
            different = QString::localeAwareCompare(a, b) != 0;
            less = compareText(a, b);
            break;
        }
        default:
            break;
        }
        if (different) {
            return less;
        }
        return ascending ? lhs.id < rhs.id : lhs.id > rhs.id;
    });
}

void CFiberWidget::setKnownTags(const std::vector<std::string>& tags)
{
    _knownTags.clear();
    for (const auto& tag : tags) {
        addUniqueSorted(_knownTags, tag);
    }
    for (const auto& fiber : _listedFibers) {
        for (const auto& tag : fiber.tags) {
            addUniqueSorted(_knownTags, tag);
        }
    }
    updateClassificationUi();
}

QStandardItem* CFiberWidget::findFiberItem(uint64_t fiberId)
{
    for (int i = 0; i < _model->rowCount(); ++i) {
        auto* item = _model->item(i, kNameColumn);
        if (item && item->data(kFiberIdRole).toULongLong() == fiberId) {
            return item;
        }
    }
    return nullptr;
}

void CFiberWidget::selectFiber(uint64_t fiberId)
{
    selectFibers(fiberId == 0 ? std::vector<uint64_t>{} : std::vector<uint64_t>{fiberId});
}

void CFiberWidget::selectFibers(const std::vector<uint64_t>& fiberIds)
{
    if (!_treeView || !_treeView->selectionModel()) {
        _selectedFiberId = 0;
        if (_deleteButton) {
            _deleteButton->setEnabled(false);
        }
        updateClassificationUi();
        return;
    }

    auto* verticalScrollBar = _treeView->verticalScrollBar();
    auto* horizontalScrollBar = _treeView->horizontalScrollBar();
    const int previousVerticalScroll = verticalScrollBar ? verticalScrollBar->value() : 0;
    const int previousHorizontalScroll = horizontalScrollBar ? horizontalScrollBar->value() : 0;

    _treeView->selectionModel()->clearSelection();
    for (uint64_t fiberId : fiberIds) {
        auto* item = findFiberItem(fiberId);
        if (!item) {
            continue;
        }
        _treeView->selectionModel()->select(item->index(),
                                            QItemSelectionModel::Select |
                                            QItemSelectionModel::Rows);
    }
    if (verticalScrollBar) {
        verticalScrollBar->setValue(previousVerticalScroll);
    }
    if (horizontalScrollBar) {
        horizontalScrollBar->setValue(previousHorizontalScroll);
    }

    const auto selected = selectedFiberIds();
    _selectedFiberId = selected.size() == 1 ? selected.front() : 0;
    for (const auto& fiber : _listedFibers) if (fiber.id == _selectedFiberId && !fiber.detailsLoaded)
        emit fiberDetailsRequested(fiber.id);
    if (_deleteButton) {
        _deleteButton->setEnabled(!selected.empty());
    }
    updateClassificationUi();
}

void CFiberWidget::setDeleteConfirmationForTesting(
    std::function<bool(const std::vector<uint64_t>&)> confirmer)
{
    _deleteConfirmationForTesting = std::move(confirmer);
}

void CFiberWidget::onSelectionChanged()
{
    const auto selected = selectedFiberIds();
    _selectedFiberId = selected.size() == 1 ? selected.front() : 0;
    for (const auto& fiber : _listedFibers) if (fiber.id == _selectedFiberId && !fiber.detailsLoaded)
        emit fiberDetailsRequested(fiber.id);
    _deleteButton->setEnabled(!selected.empty());
    updateClassificationUi();
}

void CFiberWidget::onDoubleClicked(const QModelIndex& index)
{
    auto* item = _model->itemFromIndex(index);
    if (!item) {
        return;
    }
    const uint64_t fiberId = item->data(kFiberIdRole).toULongLong();
    if (fiberId != 0) {
        if (item->data(kIsSpanRole).toBool()) {
            emit fiberSpanOpenRequested(
                fiberId,
                item->data(kSpanFirstControlIndexRole).toInt(),
                item->data(kSpanSecondControlIndexRole).toInt());
        } else {
            emit fiberOpenRequested(fiberId);
        }
    }
}

void CFiberWidget::onDeleteClicked()
{
    requestDeleteSelectedFibers();
}

void CFiberWidget::onManualHvButtonClicked(int id)
{
    if (_selectedFiberId == 0) {
        return;
    }
    emit manualHvTagChanged(_selectedFiberId, id == 0 ? QStringLiteral("H") : QStringLiteral("V"));
}

void CFiberWidget::onManualHvResetClicked()
{
    if (_selectedFiberId == 0) {
        return;
    }
    emit manualHvTagChanged(_selectedFiberId, QString());
}

void CFiberWidget::onRecalculateHvScoreClicked()
{
    if (_selectedFiberId == 0) {
        return;
    }
    emit hvScoreRecalculationRequested(_selectedFiberId);
}

void CFiberWidget::onAddTagClicked()
{
    if (_selectedFiberId == 0) {
        return;
    }
    const QString tag = _newTagEdit ? _newTagEdit->text().trimmed() : QString();
    if (tag.isEmpty()) {
        return;
    }
    const std::vector<uint64_t> previousSelection = selectedFiberIds();
    _newTagEdit->clear();
    addUniqueSorted(_knownTags, tag.toStdString());
    const uint64_t fiberId = _selectedFiberId;
    applyTagLocally(_selectedFiberId, tag.toStdString(), true);
    sortFibers();
    rebuildModel();
    selectFibers(previousSelection);
    rebuildTagList();
    QTimer::singleShot(0, this, [this, fiberId, tag]() {
        emit fiberTagChanged(fiberId, tag, true);
    });
}

void CFiberWidget::onHeaderSectionClicked(int section)
{
    if (section < 0 || section >= kColumnCount) {
        return;
    }
    const std::vector<uint64_t> previousSelection = selectedFiberIds();
    if (_sortColumn == section) {
        _sortOrder = _sortOrder == Qt::AscendingOrder
            ? Qt::DescendingOrder
            : Qt::AscendingOrder;
    } else {
        _sortColumn = section;
        _sortOrder = Qt::AscendingOrder;
    }
    sortFibers();
    rebuildModel();
    selectFibers(previousSelection);
}

const CFiberWidget::FiberEntry* CFiberWidget::selectedFiber() const
{
    const auto it = std::find_if(_listedFibers.begin(), _listedFibers.end(), [this](const FiberEntry& fiber) {
        return fiber.id == _selectedFiberId;
    });
    return it == _listedFibers.end() ? nullptr : &*it;
}

void CFiberWidget::rebuildTagList()
{
    if (!_tagListLayout || !_tagListWidget) {
        return;
    }

    // _knownTags is alphabetically sorted, but the review verdict tag is
    // pinned to the top of the checkbox list and is always offered, even in
    // a package where no fiber carries it yet.
    const std::string reviewedTag{vc3d::line_annotation::kReviewedTag};
    std::vector<std::string> orderedTags;
    orderedTags.reserve(_knownTags.size() + 1);
    orderedTags.push_back(reviewedTag);
    for (const auto& tag : _knownTags) {
        if (tag != reviewedTag) {
            orderedTags.push_back(tag);
        }
    }

    auto checkboxes = tagCheckboxesInLayout(_tagListLayout);
    bool canReuseCheckboxes = checkboxes.size() == orderedTags.size();
    if (canReuseCheckboxes) {
        for (size_t i = 0; i < orderedTags.size(); ++i) {
            if (checkboxes[i]->text() != QString::fromStdString(orderedTags[i])) {
                canReuseCheckboxes = false;
                break;
            }
        }
    }

    if (!canReuseCheckboxes) {
        while (auto* item = _tagListLayout->takeAt(0)) {
            delete item->widget();
            delete item;
        }

        for (const auto& tag : orderedTags) {
            const QString tagText = QString::fromStdString(tag);
            auto* checkbox = new QCheckBox(tagText, _tagListWidget);
            checkbox->setObjectName(QStringLiteral("fiberTagCheckBox"));
            connect(checkbox, &QCheckBox::toggled, this, [this, tagText](bool checked) {
                requestFiberTagChange(tagText, checked);
            });
            _tagListLayout->addWidget(checkbox);
        }
        _tagListLayout->addStretch(1);
        checkboxes = tagCheckboxesInLayout(_tagListLayout);
    }

    const FiberEntry* fiber = selectedFiber();
    const bool hasSelection = fiber != nullptr;
    for (size_t i = 0; i < checkboxes.size() && i < orderedTags.size(); ++i) {
        QCheckBox* checkbox = checkboxes[i];
        if (!checkbox) {
            continue;
        }
        const QSignalBlocker blocker(checkbox);
        checkbox->setEnabled(hasSelection);
        checkbox->setChecked(hasSelection && containsTag(fiber->tags, orderedTags[i]));
    }

    const bool canEditTags = hasSelection;
    if (_newTagEdit) {
        _newTagEdit->setEnabled(canEditTags);
    }
    if (_addTagButton) {
        _addTagButton->setEnabled(canEditTags);
    }
}

void CFiberWidget::requestFiberTagChange(const QString& tag, bool enabled)
{
    if (_selectedFiberId == 0) {
        return;
    }
    const std::vector<uint64_t> previousSelection = selectedFiberIds();
    const uint64_t fiberId = _selectedFiberId;
    applyTagLocally(_selectedFiberId, tag.toStdString(), enabled);
    sortFibers();
    rebuildModel();
    selectFibers(previousSelection);
    QTimer::singleShot(0, this, [this, fiberId, tag, enabled]() {
        emit fiberTagChanged(fiberId, tag, enabled);
    });
}

void CFiberWidget::applyTagLocally(uint64_t fiberId, const std::string& tag, bool enabled)
{
    auto it = std::find_if(_listedFibers.begin(), _listedFibers.end(), [fiberId](const FiberEntry& fiber) {
        return fiber.id == fiberId;
    });
    if (it == _listedFibers.end()) {
        return;
    }
    if (enabled) {
        addUniqueSorted(it->tags, tag);
    } else {
        it->tags.erase(std::remove(it->tags.begin(), it->tags.end(), tag), it->tags.end());
    }
}

void CFiberWidget::updateClassificationUi()
{
    const FiberEntry* fiber = selectedFiber();
    const bool hasSelection = fiber != nullptr;

    if (!hasSelection) {
        _nameLabel->setText(tr("No fiber selected"));
        _scoreLabel->setText(tr("z dist: -    control len: -\nH score: -    V score: -"));
        _autoLabel->setText(tr("Auto: -"));
    } else {
        const QString name = displayNameForFiber(*fiber);
        _nameLabel->setText(tr("%1\ncp=%2    pts=%3    len=%4")
                                .arg(name)
                                .arg(fiber->controlPointCount)
                                .arg(fiber->linePointCount)
                                .arg(vc3d::fiber_length::formatLength(fiber->lengthVx,
                                                                      _voxelSizeUm)));
        // The classification's distances stay in voxels whatever the len
        // line above shows, so they say so.
        _scoreLabel->setText(tr("z dist: %1 vx    control len: %2 vx\nH score: %3    V score: %4")
                                 .arg(fiber->hvZDistance, 0, 'f', 2)
                                 .arg(fiber->hvFiberLength, 0, 'f', 2)
                                 .arg(fiber->horizontalScore, 0, 'f', 2)
                                 .arg(fiber->verticalScore, 0, 'f', 2));
        _autoLabel->setText(tr("Auto: %1    certainty: %2")
                                .arg(QString::fromStdString(fiber->automaticHvTag))
                                .arg(fiber->automaticCertainty, 0, 'f', 2));
    }

    {
        const QSignalBlocker blockH(_manualHButton);
        const QSignalBlocker blockV(_manualVButton);
        _manualHvGroup->setExclusive(false);
        _manualHButton->setChecked(hasSelection && fiber->manualHvTag == "H");
        _manualVButton->setChecked(hasSelection && fiber->manualHvTag == "V");
        _manualHvGroup->setExclusive(true);
    }
    _manualHButton->setEnabled(hasSelection);
    _manualVButton->setEnabled(hasSelection);
    _manualResetButton->setEnabled(hasSelection && fiber->manualHvTag != "");
    _recalculateScoreButton->setEnabled(hasSelection);
    rebuildTagList();
}

void CFiberWidget::showContextMenu(const QPoint& pos)
{
    QModelIndex index = _treeView->indexAt(pos);
    if (index.isValid()) {
        if (auto* item = _model->itemFromIndex(index)) {
            const uint64_t clickedId = item->data(kFiberIdRole).toULongLong();
            const auto selected = selectedFiberIds();
            if (std::find(selected.begin(), selected.end(), clickedId) == selected.end()) {
                selectFiber(clickedId);
            }
        }
    }

    QMenu menu(this);
    auto* showFiberSliceAction = createShowFiberSliceAction(&menu);
    menu.addAction(showFiberSliceAction);
    menu.addSeparator();
    const auto selectedForCollection = selectedFiberIds();
    auto* addToCollectionAction = menu.addAction(
        selectedForCollection.size() > 1
            ? tr("Add %1 lines to point collections").arg(selectedForCollection.size())
            : tr("Add to point collection"));
    addToCollectionAction->setEnabled(!selectedForCollection.empty());
    connect(addToCollectionAction, &QAction::triggered, this, [this]() {
        const auto ids = selectedFiberIds();
        if (!ids.empty()) {
            emit addFibersToPointCollectionsRequested(ids);
        }
    });
    auto* addToSpiralAction = menu.addAction(
        selectedForCollection.size() > 1
            ? tr("Add %1 lines to current spiral fit").arg(selectedForCollection.size())
            : tr("Add to current spiral fit"));
    addToSpiralAction->setEnabled(_spiralFitAvailable && !selectedForCollection.empty());
    addToSpiralAction->setToolTip(_spiralFitAvailable
        ? tr("Upload the fiber(s) to the active Spiral session; they are used on the next run")
        : tr("No Spiral session is active on the connected service"));
    connect(addToSpiralAction, &QAction::triggered, this, [this]() {
        const auto ids = selectedFiberIds();
        if (!ids.empty()) {
            emit addFibersToSpiralFitRequested(ids);
        }
    });
    auto* renameAction = createRenameFiberFileAction(&menu);
    menu.addAction(renameAction);
    menu.addSeparator();
    auto* deleteAction = menu.addAction(tr("Delete"));
    deleteAction->setEnabled(canDeleteSelection());
    connect(deleteAction, &QAction::triggered, this, [this]() {
        requestDeleteSelectedFibers();
    });
    menu.exec(_treeView->viewport()->mapToGlobal(pos));
}

void CFiberWidget::requestDeleteSelectedFibers()
{
    const auto ids = selectedFiberIds();
    if (ids.empty() || !confirmDeleteFibers(ids)) {
        return;
    }

    emit deleteFibersRequested(ids);
}

void CFiberWidget::requestShowFiberSlice()
{
    if (_selectedFiberId != 0 && canShowFiberSlice()) {
        emit fiberSliceRequested(_selectedFiberId);
    }
}

void CFiberWidget::requestRenameFiberFile()
{
    if (_selectedFiberId != 0 && canRenameFiberFile()) {
        emit renameFiberFileRequested(_selectedFiberId);
    }
}

bool CFiberWidget::confirmDeleteFibers(const std::vector<uint64_t>& fiberIds)
{
    if (_deleteConfirmationForTesting) {
        return _deleteConfirmationForTesting(fiberIds);
    }

    const QString message = fiberIds.size() == 1
        ? tr("Delete fiber %1? This cannot be undone.").arg(fiberIds.front())
        : tr("Delete %1 selected fibers? This cannot be undone.").arg(fiberIds.size());
    return QMessageBox::question(this,
                                 tr("Delete Fibers"),
                                 message,
                                 QMessageBox::Yes | QMessageBox::Cancel,
                                 QMessageBox::Cancel) == QMessageBox::Yes;
}
