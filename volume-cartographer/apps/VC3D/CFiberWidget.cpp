#include "CFiberWidget.hpp"

#include "FiberNameDisplay.hpp"

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
#include <QScrollBar>
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
#include <set>
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
    kMeanAlignErrorColumn,
    kMaxAlignErrorColumn,
    kColumnCount,
};

constexpr int kFiberIdRole = Qt::UserRole + 1;
constexpr int kIsSpanRole = Qt::UserRole + 2;
constexpr int kSpanFirstControlIndexRole = Qt::UserRole + 3;
constexpr int kSpanSecondControlIndexRole = Qt::UserRole + 4;
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

QString formatDouble(double value, int precision = 1)
{
    if (!std::isfinite(value)) {
        return QStringLiteral("-");
    }
    return QString::number(value, 'f', precision);
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

QString formatMetric(const CFiberWidget::FiberEntry::AlignmentMetrics& metric,
                     bool showMetrics)
{
    if (!showMetrics) {
        return QStringLiteral("-");
    }
    if (metric.pending) {
        return QStringLiteral("...");
    }
    if (!metric.error.empty()) {
        return QStringLiteral("err");
    }
    if (!metric.available) {
        return QStringLiteral("-");
    }
    return formatDouble(metric.meanErrorDegrees, 1);
}

QString formatMaxMetric(const CFiberWidget::FiberEntry::AlignmentMetrics& metric,
                        bool showMetrics)
{
    if (!showMetrics) {
        return QStringLiteral("-");
    }
    if (metric.pending) {
        return QStringLiteral("...");
    }
    if (!metric.error.empty()) {
        return QStringLiteral("err");
    }
    if (!metric.available) {
        return QStringLiteral("-");
    }
    return formatDouble(metric.maxErrorDegrees, 1);
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
    for (QStandardItem* item : row) {
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
        item->setToolTip(tooltip);
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
    auto* mainWidget = new QWidget(this);
    auto* layout = new QVBoxLayout(mainWidget);

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

    auto* fiberDisplayLayout = new QHBoxLayout();
    _showFibersCheckBox = new QCheckBox(tr("Show fibers"), mainWidget);
    _showFibersCheckBox->setObjectName(QStringLiteral("fiberShowFibersCheckBox"));
    _showFibersCheckBox->setEnabled(false);
    _showFibersCheckBox->setToolTip(
        tr("Show all loaded fibers as control-point chains in the volume viewers."));
    fiberDisplayLayout->addWidget(_showFibersCheckBox);
    connect(_showFibersCheckBox, &QCheckBox::toggled,
            this, &CFiberWidget::showFibersToggled);

    auto* viewDistanceLabel = new QLabel(tr("View distance:"), mainWidget);
    fiberDisplayLayout->addWidget(viewDistanceLabel);
    _fiberViewDistanceSpinBox = new QDoubleSpinBox(mainWidget);
    _fiberViewDistanceSpinBox->setObjectName(QStringLiteral("fiberViewDistanceSpinBox"));
    _fiberViewDistanceSpinBox->setRange(0.0, 10000.0);
    _fiberViewDistanceSpinBox->setDecimals(1);
    _fiberViewDistanceSpinBox->setSingleStep(1.0);
    _fiberViewDistanceSpinBox->setValue(10.0);
    _fiberViewDistanceSpinBox->setSuffix(tr(" vx"));
    _fiberViewDistanceSpinBox->setMaximumWidth(100);
    _fiberViewDistanceSpinBox->setToolTip(
        tr("Maximum distance from the current plane or surface at which fibers remain visible."));
    viewDistanceLabel->setBuddy(_fiberViewDistanceSpinBox);
    fiberDisplayLayout->addWidget(_fiberViewDistanceSpinBox);
    fiberDisplayLayout->addStretch(1);
    layout->addLayout(fiberDisplayLayout);
    connect(_fiberViewDistanceSpinBox,
            qOverload<double>(&QDoubleSpinBox::valueChanged),
            this,
            &CFiberWidget::fiberViewDistanceChanged);

    _model = new QStandardItemModel(this);
    _model->setColumnCount(kColumnCount);
    _model->setHorizontalHeaderLabels({
        tr("name"),
        tr("dir"),
        tr("link"),
        tr("pending"),
        tr("len"),
        tr("cps"),
        tr("pts"),
        tr("tags"),
        tr("mean align deg"),
        tr("max align deg"),
    });
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
    _treeView->setColumnWidth(kMeanAlignErrorColumn, 110);
    _treeView->setColumnWidth(kMaxAlignErrorColumn, 105);
    layout->addWidget(_treeView, 1);

    connect(_treeView->selectionModel(), &QItemSelectionModel::selectionChanged,
            this, &CFiberWidget::onSelectionChanged);
    connect(_treeView, &QTreeView::doubleClicked,
            this, &CFiberWidget::onDoubleClicked);
    connect(_treeView, &QWidget::customContextMenuRequested,
            this, &CFiberWidget::showContextMenu);
    connect(_treeView->header(), &QHeaderView::sectionClicked,
            this, &CFiberWidget::onHeaderSectionClicked);

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
    setWidget(mainWidget);
}

void CFiberWidget::setShowFibersAvailable(bool available)
{
    _showFibersCheckBox->setEnabled(available);
    if (!available) {
        setShowFibersChecked(false);
    }
}

void CFiberWidget::setShowFibersChecked(bool checked)
{
    const QSignalBlocker blocker(_showFibersCheckBox);
    _showFibersCheckBox->setChecked(checked);
}

bool CFiberWidget::showFibersChecked() const
{
    return _showFibersCheckBox->isChecked();
}

void CFiberWidget::setFiberViewDistance(double distance)
{
    const QSignalBlocker blocker(_fiberViewDistanceSpinBox);
    _fiberViewDistanceSpinBox->setValue(distance);
}

double CFiberWidget::fiberViewDistance() const
{
    return _fiberViewDistanceSpinBox->value();
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

bool CFiberWidget::canCreateAtlasFromSelection() const
{
    return selectedFiberIds().size() == 1;
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
    ids.reserve(_fibers.size());
    for (const auto& fiber : _fibers) {
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
    auto* action = new QAction(tr("Rename JSON file..."), parent);
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
    _fibers = fibers;
    sortFibers();
    for (const auto& fiber : _fibers) {
        for (const auto& tag : fiber.tags) {
            addUniqueSorted(_knownTags, tag);
        }
    }

    rebuildModel();

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

void CFiberWidget::setAlignmentMetricsPending(bool pending)
{
    FiberEntry::AlignmentMetrics metric;
    metric.pending = pending;
    for (auto& fiber : _fibers) {
        fiber.alignment = metric;
        for (auto& span : fiber.spans) {
            span.alignment = metric;
        }
    }
    refreshMetricDisplays();
}

void CFiberWidget::updateAlignmentMetrics(
    uint64_t fiberId,
    const FiberEntry::AlignmentMetrics& alignment,
    const std::vector<FiberEntry::AlignmentMetrics>& spanAlignments)
{
    auto fiberIt = std::find_if(_fibers.begin(), _fibers.end(), [fiberId](const FiberEntry& fiber) {
        return fiber.id == fiberId;
    });
    if (fiberIt == _fibers.end()) {
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

    const bool showMetrics = _calcMetricsCheckBox && _calcMetricsCheckBox->isChecked();
    _model->removeRows(0, _model->rowCount());
    _model->setColumnCount(kColumnCount);
    _model->setHorizontalHeaderLabels({
        tr("name"),
        tr("dir"),
        tr("link"),
        tr("pending"),
        tr("len"),
        tr("cps"),
        tr("pts"),
        tr("tags"),
        tr("mean align deg"),
        tr("max align deg"),
    });
    if (_treeView && _treeView->header()) {
        _treeView->header()->setSortIndicator(_sortColumn, _sortOrder);
    }

    for (const auto& fiber : _fibers) {
        QList<QStandardItem*> row{
            readOnlyItem(displayNameForFiber(fiber)),
            readOnlyItem(directionForFiber(fiber)),
            readOnlyItem(fiber.linkedFiberCount > 0
                             ? QString::number(fiber.linkedFiberCount)
                             : QString()),
            readOnlyItem(fiber.pendingLinkCount > 0
                             ? QString::number(fiber.pendingLinkCount)
                             : QString()),
            readOnlyItem(formatDouble(fiber.lengthVx, 1)),
            readOnlyItem(QString::number(fiber.controlPointCount)),
            readOnlyItem(QString::number(fiber.linePointCount)),
            readOnlyItem(formatTags(fiber.tags)),
            readOnlyItem(formatMetric(fiber.alignment, showMetrics)),
            readOnlyItem(formatMaxMetric(fiber.alignment, showMetrics)),
        };
        applyRowMetadata(row, fiber.id, false, fiber.alignment, showMetrics);

        QStandardItem* root = row[kNameColumn];
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
                readOnlyItem(formatDouble(span.lengthVx, 1)),
                readOnlyItem(QString::number(span.controlPointCount)),
                readOnlyItem(QString::number(span.linePointCount)),
                readOnlyItem(QString()),
                readOnlyItem(formatMetric(span.alignment, showMetrics)),
                readOnlyItem(formatMaxMetric(span.alignment, showMetrics)),
            };
            applyRowMetadata(childRow, fiber.id, true, span.alignment, showMetrics);
            applySpanMetadata(childRow, span.firstControlIndex, span.secondControlIndex);
            root->appendRow(childRow);
        }

        _model->appendRow(row);
        if (_treeView && expandedFibers.find(fiber.id) != expandedFibers.end()) {
            _treeView->setExpanded(root->index(), true);
        }
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
    if (row[kMeanAlignErrorColumn]) {
        row[kMeanAlignErrorColumn]->setText(formatMetric(alignment, showMetrics));
    }
    if (row[kMaxAlignErrorColumn]) {
        row[kMaxAlignErrorColumn]->setText(formatMaxMetric(alignment, showMetrics));
    }

    const uint64_t fiberId = row[kNameColumn]->data(kFiberIdRole).toULongLong();
    const bool isSpan = row[kNameColumn]->data(kIsSpanRole).toBool();
    applyRowMetadata(row, fiberId, isSpan, alignment, showMetrics);
}

void CFiberWidget::refreshMetricDisplays()
{
    for (const auto& fiber : _fibers) {
        QStandardItem* root = findFiberItem(fiber.id);
        if (!root) {
            continue;
        }
        updateMetricDisplayForRow(root, fiber.alignment);
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
    auto metricValue = [](const FiberEntry::AlignmentMetrics& metric) {
        return metric.available && std::isfinite(metric.maxErrorDegrees)
            ? std::optional<double>(metric.maxErrorDegrees)
            : std::nullopt;
    };
    auto metricMeanValue = [](const FiberEntry::AlignmentMetrics& metric) {
        return metric.available && std::isfinite(metric.meanErrorDegrees)
            ? std::optional<double>(metric.meanErrorDegrees)
            : std::nullopt;
    };
    auto compareOptionalNumber = [ascending](std::optional<double> lhs,
                                             std::optional<double> rhs) {
        if (lhs && rhs) {
            if (*lhs == *rhs) {
                return false;
            }
            return ascending ? *lhs < *rhs : *lhs > *rhs;
        }
        if (lhs != rhs) {
            return lhs.has_value();
        }
        return false;
    };

    std::stable_sort(_fibers.begin(), _fibers.end(), [&](const FiberEntry& lhs, const FiberEntry& rhs) {
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
        case kMeanAlignErrorColumn: {
            const auto a = metricMeanValue(lhs.alignment);
            const auto b = metricMeanValue(rhs.alignment);
            different = a != b;
            less = compareOptionalNumber(a, b);
            break;
        }
        case kMaxAlignErrorColumn: {
            const auto a = metricValue(lhs.alignment);
            const auto b = metricValue(rhs.alignment);
            different = a != b;
            less = compareOptionalNumber(a, b);
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
    for (const auto& fiber : _fibers) {
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
    const auto it = std::find_if(_fibers.begin(), _fibers.end(), [this](const FiberEntry& fiber) {
        return fiber.id == _selectedFiberId;
    });
    return it == _fibers.end() ? nullptr : &*it;
}

void CFiberWidget::rebuildTagList()
{
    if (!_tagListLayout || !_tagListWidget) {
        return;
    }

    auto checkboxes = tagCheckboxesInLayout(_tagListLayout);
    bool canReuseCheckboxes = checkboxes.size() == _knownTags.size();
    if (canReuseCheckboxes) {
        for (size_t i = 0; i < _knownTags.size(); ++i) {
            if (checkboxes[i]->text() != QString::fromStdString(_knownTags[i])) {
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

        for (const auto& tag : _knownTags) {
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
    for (size_t i = 0; i < checkboxes.size() && i < _knownTags.size(); ++i) {
        QCheckBox* checkbox = checkboxes[i];
        if (!checkbox) {
            continue;
        }
        const QSignalBlocker blocker(checkbox);
        checkbox->setEnabled(hasSelection);
        checkbox->setChecked(hasSelection && containsTag(fiber->tags, _knownTags[i]));
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
    auto it = std::find_if(_fibers.begin(), _fibers.end(), [fiberId](const FiberEntry& fiber) {
        return fiber.id == fiberId;
    });
    if (it == _fibers.end()) {
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
        _nameLabel->setText(tr("%1\ncp=%2    pts=%3    len=%4 vx")
                                .arg(name)
                                .arg(fiber->controlPointCount)
                                .arg(fiber->linePointCount)
                                .arg(fiber->lengthVx, 0, 'f', 1));
        _scoreLabel->setText(tr("z dist: %1    control len: %2\nH score: %3    V score: %4")
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
    auto* newAtlasAction = menu.addAction(tr("New atlas from line"));
    newAtlasAction->setEnabled(canCreateAtlasFromSelection());
    connect(newAtlasAction, &QAction::triggered, this, [this]() {
        if (_selectedFiberId != 0) {
            emit newAtlasFromFiberRequested(_selectedFiberId);
        }
    });
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
