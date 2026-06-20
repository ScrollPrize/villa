#include "CFiberWidget.hpp"

#include <QAbstractItemView>
#include <QAction>
#include <QButtonGroup>
#include <QCheckBox>
#include <QComboBox>
#include <QHBoxLayout>
#include <QItemSelectionModel>
#include <QLabel>
#include <QLineEdit>
#include <QMenu>
#include <QMessageBox>
#include <QSignalBlocker>
#include <QVBoxLayout>

#include <algorithm>
#include <utility>

namespace {

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

    auto* volumeScaleLayout = new QHBoxLayout();
    volumeScaleLayout->addWidget(new QLabel(tr("Volume scale:"), mainWidget));
    _volumeScaleCombo = new QComboBox(mainWidget);
    _volumeScaleCombo->setObjectName(QStringLiteral("fiberVolumeScaleCombo"));
    _volumeScaleCombo->setToolTip(tr("Working-volume scale relative to the full-resolution Lasagna "
                                     "data. 'Auto' detects it from the active volume shape."));
    _volumeScaleCombo->addItem(tr("Auto"), 0.0);
    _volumeScaleCombo->addItem(QStringLiteral("1"), 1.0);
    _volumeScaleCombo->addItem(QStringLiteral("0.5"), 0.5);
    _volumeScaleCombo->addItem(QStringLiteral("0.25"), 0.25);
    _volumeScaleCombo->addItem(QStringLiteral("0.125"), 0.125);
    volumeScaleLayout->addWidget(_volumeScaleCombo);
    _volumeScaleInfoLabel = new QLabel(mainWidget);
    volumeScaleLayout->addWidget(_volumeScaleInfoLabel, 1);
    layout->addLayout(volumeScaleLayout);

    connect(_volumeScaleCombo, qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int) {
        emit volumeScaleOverrideChanged(_volumeScaleCombo->currentData().toDouble());
    });

    _nameLabel = new QLabel(mainWidget);
    _nameLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
    layout->addWidget(_nameLabel);

    _scoreLabel = new QLabel(mainWidget);
    _scoreLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
    layout->addWidget(_scoreLabel);

    _autoLabel = new QLabel(mainWidget);
    _autoLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
    layout->addWidget(_autoLabel);

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
    const QString fileName = QString::fromStdString(fiber.fileName);
    const QString nameText = fileName.isEmpty()
        ? tr("Fiber %1").arg(fiber.id)
        : tr("Fiber %1  %2").arg(fiber.id).arg(fileName);
    return tr("%1  cp=%2  pts=%3  len=%4 vx")
        .arg(nameText)
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

bool CFiberWidget::canShowFiberSlice() const
{
    return selectedFiberIds().size() == 1;
}

bool CFiberWidget::canRenameFiberFile() const
{
    return selectedFiberIds().size() == 1;
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
    std::sort(_fibers.begin(), _fibers.end(), [](const FiberEntry& a, const FiberEntry& b) {
        return a.id < b.id;
    });
    for (const auto& fiber : _fibers) {
        for (const auto& tag : fiber.tags) {
            addUniqueSorted(_knownTags, tag);
        }
    }

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

void CFiberWidget::setVolumeScaleDisplay(double effectiveScale, bool autoDetected)
{
    if (!_volumeScaleInfoLabel) {
        return;
    }
    _volumeScaleInfoLabel->setText(
        autoDetected
            ? tr("detected: %1").arg(effectiveScale, 0, 'g', 4)
            : tr("override: %1").arg(effectiveScale, 0, 'g', 4));
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

void CFiberWidget::onAddTagClicked()
{
    if (_selectedFiberId == 0) {
        return;
    }
    const QString tag = _newTagEdit ? _newTagEdit->text().trimmed() : QString();
    if (tag.isEmpty()) {
        return;
    }
    _newTagEdit->clear();
    addUniqueSorted(_knownTags, tag.toStdString());
    applyTagLocally(_selectedFiberId, tag.toStdString(), true);
    emit fiberTagChanged(_selectedFiberId, tag, true);
    rebuildTagList();
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

    while (auto* item = _tagListLayout->takeAt(0)) {
        delete item->widget();
        delete item;
    }

    const FiberEntry* fiber = selectedFiber();
    const bool hasSelection = fiber != nullptr;
    for (const auto& tag : _knownTags) {
        auto* checkbox = new QCheckBox(QString::fromStdString(tag), _tagListWidget);
        checkbox->setObjectName(QStringLiteral("fiberTagCheckBox"));
        checkbox->setEnabled(hasSelection);
        checkbox->setChecked(hasSelection && containsTag(fiber->tags, tag));
        connect(checkbox, &QCheckBox::toggled, this, [this, tagText = QString::fromStdString(tag)](bool checked) {
            requestFiberTagChange(tagText, checked);
        });
        _tagListLayout->addWidget(checkbox);
    }
    _tagListLayout->addStretch(1);

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
    applyTagLocally(_selectedFiberId, tag.toStdString(), enabled);
    emit fiberTagChanged(_selectedFiberId, tag, enabled);
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
        const QString name = QString::fromStdString(fiber->fileName).isEmpty()
            ? tr("Fiber %1").arg(fiber->id)
            : QString::fromStdString(fiber->fileName);
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
    auto* renameAction = createRenameFiberFileAction(&menu);
    menu.addAction(renameAction);
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
