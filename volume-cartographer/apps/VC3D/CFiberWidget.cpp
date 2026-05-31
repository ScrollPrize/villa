#include "CFiberWidget.hpp"

#include <QAbstractItemView>
#include <QHBoxLayout>
#include <QItemSelectionModel>
#include <QVBoxLayout>

#include <algorithm>

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

    _model = new QStandardItemModel(this);
    _listView = new QListView(mainWidget);
    _listView->setModel(_model);
    _listView->setEditTriggers(QAbstractItemView::NoEditTriggers);
    _listView->setSelectionMode(QAbstractItemView::SingleSelection);
    layout->addWidget(_listView, 1);

    connect(_listView->selectionModel(), &QItemSelectionModel::selectionChanged,
            this, &CFiberWidget::onSelectionChanged);
    connect(_listView, &QListView::doubleClicked,
            this, &CFiberWidget::onDoubleClicked);

    auto* buttonLayout = new QHBoxLayout();
    _deleteButton = new QPushButton(tr("Delete"), mainWidget);
    _deleteButton->setEnabled(false);
    buttonLayout->addWidget(_deleteButton);
    buttonLayout->addStretch(1);
    layout->addLayout(buttonLayout);

    connect(_deleteButton, &QPushButton::clicked, this, &CFiberWidget::onDeleteClicked);

    setWidget(mainWidget);
}

QString CFiberWidget::labelForFiber(const FiberEntry& fiber)
{
    return tr("Fiber %1  cp=%2  pts=%3  len=%4 vx")
        .arg(fiber.id)
        .arg(fiber.controlPointCount)
        .arg(fiber.linePointCount)
        .arg(fiber.lengthVx, 0, 'f', 1);
}

void CFiberWidget::setFibers(const std::vector<FiberEntry>& fibers)
{
    const uint64_t previousSelection = _selectedFiberId;
    _fibers = fibers;
    std::sort(_fibers.begin(), _fibers.end(), [](const FiberEntry& a, const FiberEntry& b) {
        return a.id < b.id;
    });

    _model->clear();
    for (const auto& fiber : _fibers) {
        auto* item = new QStandardItem(labelForFiber(fiber));
        item->setData(QVariant::fromValue(fiber.id));
        _model->appendRow(item);
    }

    _selectedFiberId = 0;
    if (previousSelection != 0) {
        selectFiber(previousSelection);
    }
    _deleteButton->setEnabled(_selectedFiberId != 0);
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
    if (!item) {
        _selectedFiberId = 0;
        if (_listView && _listView->selectionModel()) {
            _listView->selectionModel()->clearSelection();
        }
        if (_deleteButton) {
            _deleteButton->setEnabled(false);
        }
        return;
    }

    _listView->selectionModel()->clearSelection();
    _listView->selectionModel()->select(item->index(),
                                        QItemSelectionModel::ClearAndSelect);
    _listView->scrollTo(item->index());
    _selectedFiberId = fiberId;
    _deleteButton->setEnabled(true);
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

    _deleteButton->setEnabled(_selectedFiberId != 0);
}

void CFiberWidget::onDoubleClicked(const QModelIndex& index)
{
    auto* item = _model->itemFromIndex(index);
    if (!item) {
        return;
    }
    const uint64_t fiberId = item->data().toULongLong();
    if (fiberId != 0) {
        emit fiberOpenRequested(fiberId);
    }
}

void CFiberWidget::onDeleteClicked()
{
    if (_selectedFiberId != 0) {
        emit deleteFiberRequested(_selectedFiberId);
    }
}
