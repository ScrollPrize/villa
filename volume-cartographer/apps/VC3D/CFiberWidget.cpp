#include "CFiberWidget.hpp"

#include <QAbstractItemView>
#include <QButtonGroup>
#include <QHBoxLayout>
#include <QItemSelectionModel>
#include <QLabel>
#include <QMenu>
#include <QMessageBox>
#include <QSignalBlocker>
#include <QVBoxLayout>

#include <algorithm>
#include <utility>

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
    _listView->setSelectionMode(QAbstractItemView::ExtendedSelection);
    _listView->setContextMenuPolicy(Qt::CustomContextMenu);
    layout->addWidget(_listView, 1);

    connect(_listView->selectionModel(), &QItemSelectionModel::selectionChanged,
            this, &CFiberWidget::onSelectionChanged);
    connect(_listView, &QListView::doubleClicked,
            this, &CFiberWidget::onDoubleClicked);
    connect(_listView, &QWidget::customContextMenuRequested,
            this, &CFiberWidget::showContextMenu);

    _scoreLabel = new QLabel(mainWidget);
    _scoreLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
    layout->addWidget(_scoreLabel);

    _autoLabel = new QLabel(mainWidget);
    _autoLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
    layout->addWidget(_autoLabel);

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
    _deleteButton = new QPushButton(tr("Delete"), mainWidget);
    _deleteButton->setObjectName(QStringLiteral("fiberDeleteButton"));
    _deleteButton->setEnabled(false);
    buttonLayout->addWidget(_deleteButton);
    buttonLayout->addStretch(1);
    layout->addLayout(buttonLayout);

    connect(_deleteButton, &QPushButton::clicked, this, &CFiberWidget::onDeleteClicked);

    updateClassificationUi();
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

std::vector<uint64_t> CFiberWidget::selectedFiberIds() const
{
    std::vector<uint64_t> ids;
    if (!_listView || !_listView->selectionModel()) {
        return ids;
    }

    const auto indexes = _listView->selectionModel()->selectedIndexes();
    ids.reserve(static_cast<size_t>(indexes.size()));
    for (const QModelIndex& index : indexes) {
        auto* item = _model->itemFromIndex(index);
        if (item) {
            const uint64_t id = item->data().toULongLong();
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

void CFiberWidget::setFibers(const std::vector<FiberEntry>& fibers)
{
    const std::vector<uint64_t> previousSelection = selectedFiberIds();
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
    if (!previousSelection.empty()) {
        selectFibers(previousSelection);
    }
    _deleteButton->setEnabled(canDeleteSelection());
    updateClassificationUi();
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
    selectFibers(fiberId == 0 ? std::vector<uint64_t>{} : std::vector<uint64_t>{fiberId});
}

void CFiberWidget::selectFibers(const std::vector<uint64_t>& fiberIds)
{
    if (!_listView || !_listView->selectionModel()) {
        _selectedFiberId = 0;
        if (_deleteButton) {
            _deleteButton->setEnabled(false);
        }
        updateClassificationUi();
        return;
    }

    _listView->selectionModel()->clearSelection();
    QModelIndex firstSelectedIndex;
    for (uint64_t fiberId : fiberIds) {
        auto* item = findFiberItem(fiberId);
        if (!item) {
            continue;
        }
        _listView->selectionModel()->select(item->index(), QItemSelectionModel::Select);
        if (!firstSelectedIndex.isValid()) {
            firstSelectedIndex = item->index();
        }
    }
    if (firstSelectedIndex.isValid()) {
        _listView->scrollTo(firstSelectedIndex);
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
    const uint64_t fiberId = item->data().toULongLong();
    if (fiberId != 0) {
        emit fiberOpenRequested(fiberId);
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

const CFiberWidget::FiberEntry* CFiberWidget::selectedFiber() const
{
    const auto it = std::find_if(_fibers.begin(), _fibers.end(), [this](const FiberEntry& fiber) {
        return fiber.id == _selectedFiberId;
    });
    return it == _fibers.end() ? nullptr : &*it;
}

void CFiberWidget::updateClassificationUi()
{
    const FiberEntry* fiber = selectedFiber();
    const bool hasSelection = fiber != nullptr;

    if (!hasSelection) {
        _scoreLabel->setText(tr("z dist: -    control len: -\nH score: -    V score: -"));
        _autoLabel->setText(tr("Auto: -"));
    } else {
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
}

void CFiberWidget::showContextMenu(const QPoint& pos)
{
    QModelIndex index = _listView->indexAt(pos);
    if (index.isValid()) {
        if (auto* item = _model->itemFromIndex(index)) {
            const uint64_t clickedId = item->data().toULongLong();
            const auto selected = selectedFiberIds();
            if (std::find(selected.begin(), selected.end(), clickedId) == selected.end()) {
                selectFiber(clickedId);
            }
        }
    }

    QMenu menu(this);
    auto* newAtlasAction = menu.addAction(tr("New atlas from line"));
    newAtlasAction->setEnabled(canCreateAtlasFromSelection());
    connect(newAtlasAction, &QAction::triggered, this, [this]() {
        if (_selectedFiberId != 0) {
            emit newAtlasFromFiberRequested(_selectedFiberId);
        }
    });
    menu.addSeparator();
    auto* deleteAction = menu.addAction(tr("Delete"));
    deleteAction->setEnabled(canDeleteSelection());
    connect(deleteAction, &QAction::triggered, this, [this]() {
        requestDeleteSelectedFibers();
    });
    menu.exec(_listView->viewport()->mapToGlobal(pos));
}

void CFiberWidget::requestDeleteSelectedFibers()
{
    const auto ids = selectedFiberIds();
    if (ids.empty() || !confirmDeleteFibers(ids)) {
        return;
    }

    emit deleteFibersRequested(ids);
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
