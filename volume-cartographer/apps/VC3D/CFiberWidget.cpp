#include "CFiberWidget.hpp"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>

CFiberWidget::CFiberWidget(VCCollection* collection, QWidget* parent)
    : QDockWidget(tr("Fibers"), parent), _collection(collection)
{
    setupUi();

    connect(_collection, &VCCollection::collectionsAdded, this, &CFiberWidget::onCollectionsAdded);
    connect(_collection, &VCCollection::collectionChanged, this, &CFiberWidget::onCollectionChanged);
    connect(_collection, &VCCollection::collectionRemoved, this, &CFiberWidget::onCollectionRemoved);
    connect(_collection, &VCCollection::pointAdded, this, &CFiberWidget::onPointAdded);
    connect(_collection, &VCCollection::pointRemoved, this, &CFiberWidget::onPointRemoved);

    refreshList();
}

CFiberWidget::~CFiberWidget() = default;

void CFiberWidget::setupUi()
{
    auto* mainWidget = new QWidget(this);
    auto* layout = new QVBoxLayout(mainWidget);

    _model = new QStandardItemModel(this);
    _listView = new QListView(mainWidget);
    _listView->setModel(_model);
    _listView->setEditTriggers(QAbstractItemView::NoEditTriggers);
    layout->addWidget(_listView);

    connect(_listView->selectionModel(), &QItemSelectionModel::selectionChanged,
            this, &CFiberWidget::onSelectionChanged);

    // Step toggle buttons
    auto* stepLayout = new QHBoxLayout();
    stepLayout->addWidget(new QLabel("Step:"));
    _stepGroup = new QButtonGroup(this);
    _stepGroup->setExclusive(true);
    for (int step : {5, 10, 25, 50, 100}) {
        auto* btn = new QPushButton(QString::number(step), mainWidget);
        btn->setCheckable(true);
        btn->setMinimumWidth(30);
        if (step == 50) btn->setChecked(true);
        _stepGroup->addButton(btn, step);
        stepLayout->addWidget(btn);
    }
    layout->addLayout(stepLayout);

    connect(_stepGroup, &QButtonGroup::idClicked, this, &CFiberWidget::onStepButtonClicked);

    // Buttons
    _newFiberButton = new QPushButton(tr("New Fiber"), mainWidget);
    _newFiberButton->setToolTip("Create a new fiber and start annotation (crosshair pick mode)");
    layout->addWidget(_newFiberButton);

    connect(_newFiberButton, &QPushButton::clicked, this, &CFiberWidget::onNewFiberClicked);

    layout->addStretch();
    setWidget(mainWidget);
}

void CFiberWidget::refreshList()
{
    _model->clear();

    if (!_collection) return;

    const auto& all = _collection->getAllCollections();

    // Collect fiber collections, sort by name
    std::vector<const VCCollection::Collection*> fibers;
    for (const auto& [id, col] : all) {
        if (isFiber(id)) {
            fibers.push_back(&col);
        }
    }
    std::sort(fibers.begin(), fibers.end(),
              [](const auto* a, const auto* b) { return a->name < b->name; });

    for (const auto* col : fibers) {
        auto* item = new QStandardItem(
            QString("%1 (%2 pts)").arg(QString::fromStdString(col->name)).arg(col->points.size()));
        item->setData(QVariant::fromValue(col->id));
        QColor color(col->color[0] * 255, col->color[1] * 255, col->color[2] * 255);
        item->setData(QBrush(color), Qt::DecorationRole);
        _model->appendRow(item);
    }

    // Re-select if still valid
    if (_selectedFiberId != 0) {
        auto* item = findFiberItem(_selectedFiberId);
        if (item) {
            _listView->selectionModel()->select(item->index(), QItemSelectionModel::Select);
        } else {
            _selectedFiberId = 0;
        }
    }
}

bool CFiberWidget::isFiber(uint64_t collectionId) const
{
    auto tag = _collection->getCollectionTag(collectionId, "fiber");
    return tag.has_value() && *tag == "true";
}

QStandardItem* CFiberWidget::findFiberItem(uint64_t fiberId)
{
    for (int i = 0; i < _model->rowCount(); ++i) {
        auto* item = _model->item(i);
        if (item && item->data().toULongLong() == fiberId) {
            return item;
        }
    }
    return nullptr;
}

void CFiberWidget::selectFiber(uint64_t fiberId)
{
    auto* item = findFiberItem(fiberId);
    if (item) {
        _listView->selectionModel()->clearSelection();
        _listView->selectionModel()->select(item->index(), QItemSelectionModel::Select);
        _listView->scrollTo(item->index());
    }
}

void CFiberWidget::onNewFiberClicked()
{
    emit newFiberRequested();
}

void CFiberWidget::onStepButtonClicked(int id)
{
    _currentStep = id;
    emit stepChanged(id);
}

void CFiberWidget::onSelectionChanged()
{
    _selectedFiberId = 0;

    auto indexes = _listView->selectionModel()->selectedIndexes();
    if (!indexes.isEmpty()) {
        auto* item = _model->itemFromIndex(indexes.first());
        if (item) {
            _selectedFiberId = item->data().toULongLong();
        }
    }

    emit fiberSelected(_selectedFiberId);
}

void CFiberWidget::onCollectionsAdded(const std::vector<uint64_t>& collectionIds)
{
    bool anyFiber = false;
    for (auto id : collectionIds) {
        if (isFiber(id)) { anyFiber = true; break; }
    }
    if (anyFiber) refreshList();
}

void CFiberWidget::onCollectionChanged(uint64_t collectionId)
{
    refreshList();
}

void CFiberWidget::onCollectionRemoved(uint64_t collectionId)
{
    if (_selectedFiberId == collectionId) {
        _selectedFiberId = 0;
    }
    refreshList();
}

void CFiberWidget::onPointAdded(const ColPoint& point)
{
    if (isFiber(point.collectionId)) {
        auto* item = findFiberItem(point.collectionId);
        if (item) {
            const auto& all = _collection->getAllCollections();
            if (all.count(point.collectionId)) {
                const auto& col = all.at(point.collectionId);
                item->setText(QString("%1 (%2 pts)")
                    .arg(QString::fromStdString(col.name))
                    .arg(col.points.size()));
            }
        }
    }
}

void CFiberWidget::onPointRemoved(uint64_t pointId)
{
    refreshList();
}
