#include "WrapAnnotationWidget.hpp"

#include <algorithm>

#include <QAction>
#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QItemSelectionModel>
#include <QLabel>
#include <QMenu>
#include <QPushButton>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QStandardItem>
#include <QStandardItemModel>
#include <QTreeView>
#include <QVBoxLayout>
#include <QWidget>

namespace {
enum SameWrapAnnotationColumn {
    kNameColumn = 0,
    kCountColumn = 1,
    kDirectionColumn = 2,
    kPositionColumn = 3
};

bool isSameWrapCollection(const VCCollection::Collection& collection)
{
    return collection.name.rfind("same_wrap", 0) == 0;
}

QString sameWrapDirectionText(const VCCollection::Collection& collection)
{
    const auto it = collection.tags.find("same_wrap_direction");
    if (it == collection.tags.end()) {
        return {};
    }
    return QString::fromStdString(it->second);
}

QString pointPositionText(const cv::Vec3f& point)
{
    return QString("{%1, %2, %3}").arg(point[0]).arg(point[1]).arg(point[2]);
}

QStandardItem* readOnlyItem(const QString& text = {})
{
    QStandardItem* item = new QStandardItem(text);
    item->setFlags(item->flags() & ~Qt::ItemIsEditable);
    return item;
}
} // namespace

WrapAnnotationWidget::WrapAnnotationWidget(VCCollection* collection, QWidget* parent)
    : QDockWidget("Wrap Annotation", parent)
    , _pointCollection(collection)
{
    setupUi();

    if (_pointCollection) {
        connect(_pointCollection, &VCCollection::collectionsAdded,
                this, &WrapAnnotationWidget::onCollectionsAdded);
        connect(_pointCollection, &VCCollection::collectionChanged,
                this, &WrapAnnotationWidget::onCollectionChanged);
        connect(_pointCollection, &VCCollection::collectionRemoved,
                this, &WrapAnnotationWidget::onCollectionRemoved);
        connect(_pointCollection, &VCCollection::pointAdded,
                this, &WrapAnnotationWidget::onPointAdded);
        connect(_pointCollection, &VCCollection::pointsAdded,
                this, &WrapAnnotationWidget::onPointsAdded);
        connect(_pointCollection, &VCCollection::pointChanged,
                this, &WrapAnnotationWidget::onPointChanged);
        connect(_pointCollection, &VCCollection::pointRemoved,
                this, &WrapAnnotationWidget::onPointRemoved);
    }

    refreshSameWrapTree();
}

void WrapAnnotationWidget::setupUi()
{
    auto* mainWidget = new QWidget(this);
    auto* layout = new QVBoxLayout(mainWidget);

    auto* sameWrapGroup = new QGroupBox("Same-wrap Annotation", mainWidget);
    auto* sameWrapLayout = new QVBoxLayout(sameWrapGroup);

    _chkSameWrapAnnotation = new QCheckBox("Same-wrap annotation mode", sameWrapGroup);
    _chkSameWrapAnnotation->setToolTip("Shift-click or Shift-drag in an active volume viewer to preview same-wrap annotation points. Shift+E commits; Ctrl+Z clears the preview or undoes the last committed collection.");
    sameWrapLayout->addWidget(_chkSameWrapAnnotation);

    _chkSameWrapMerge = new QCheckBox("Manual same-wrap merge", sameWrapGroup);
    _chkSameWrapMerge->setToolTip("Shift-right-click a point in one same-wrap collection, then Shift-right-click a point in another same-wrap collection to merge them.");
    sameWrapLayout->addWidget(_chkSameWrapMerge);

    auto* sameWrapPathTypeLayout = new QHBoxLayout();
    sameWrapPathTypeLayout->addWidget(new QLabel("Path type:"));
    _sameWrapPathTypeCombo = new QComboBox(sameWrapGroup);
    _sameWrapPathTypeCombo->addItem("Connected components", 0);
    _sameWrapPathTypeCombo->addItem("Shortest path", 1);
    _sameWrapPathTypeCombo->addItem("Manual", 2);
    _sameWrapPathTypeCombo->setToolTip("Choose whether Shift input selects a skeleton component, two shortest-path endpoints, or a manually drawn path.");
    sameWrapPathTypeLayout->addWidget(_sameWrapPathTypeCombo);
    sameWrapPathTypeLayout->addStretch();
    sameWrapLayout->addLayout(sameWrapPathTypeLayout);

    auto* sameWrapFilterLayout = new QHBoxLayout();
    sameWrapFilterLayout->addWidget(new QLabel("Filter:"));
    _sameWrapFilterTypeCombo = new QComboBox(sameWrapGroup);
    _sameWrapFilterTypeCombo->addItem("None", 0);
    _sameWrapFilterTypeCombo->addItem("Median", 1);
    _sameWrapFilterTypeCombo->addItem("Gaussian", 2);
    _sameWrapFilterTypeCombo->setToolTip("Optionally filter the source image before thresholding and skeleton tracing.");
    sameWrapFilterLayout->addWidget(_sameWrapFilterTypeCombo);
    sameWrapFilterLayout->addWidget(new QLabel("Kernel:"));
    _sameWrapFilterKernelSpinbox = new QSpinBox(sameWrapGroup);
    _sameWrapFilterKernelSpinbox->setRange(3, 99);
    _sameWrapFilterKernelSpinbox->setSingleStep(2);
    _sameWrapFilterKernelSpinbox->setValue(3);
    _sameWrapFilterKernelSpinbox->setSuffix(" px");
    _sameWrapFilterKernelSpinbox->setMaximumWidth(80);
    _sameWrapFilterKernelSpinbox->setEnabled(false);
    _sameWrapFilterKernelSpinbox->setToolTip("Odd blur kernel size applied before connected components or shortest-path tracing.");
    sameWrapFilterLayout->addWidget(_sameWrapFilterKernelSpinbox);
    sameWrapFilterLayout->addStretch();
    sameWrapLayout->addLayout(sameWrapFilterLayout);

    auto* sameWrapSpacingLayout = new QHBoxLayout();
    sameWrapSpacingLayout->addWidget(new QLabel("Spacing:"));
    _sameWrapSpacingSpinbox = new QDoubleSpinBox(sameWrapGroup);
    _sameWrapSpacingSpinbox->setRange(1.0, 1000.0);
    _sameWrapSpacingSpinbox->setDecimals(1);
    _sameWrapSpacingSpinbox->setSingleStep(1.0);
    _sameWrapSpacingSpinbox->setValue(20.0);
    _sameWrapSpacingSpinbox->setSuffix(" vx");
    _sameWrapSpacingSpinbox->setMaximumWidth(90);
    _sameWrapSpacingSpinbox->setToolTip("Distance between generated same-wrap annotation points in surface voxels.");
    sameWrapSpacingLayout->addWidget(_sameWrapSpacingSpinbox);
    sameWrapSpacingLayout->addStretch();
    sameWrapLayout->addLayout(sameWrapSpacingLayout);

    auto* sameWrapPolylineLayout = new QHBoxLayout();
    sameWrapPolylineLayout->addWidget(new QLabel("Polyline opacity:"));
    _sameWrapPolylineOpacitySpinbox = new QDoubleSpinBox(sameWrapGroup);
    _sameWrapPolylineOpacitySpinbox->setRange(0.0, 1.0);
    _sameWrapPolylineOpacitySpinbox->setDecimals(2);
    _sameWrapPolylineOpacitySpinbox->setSingleStep(0.05);
    _sameWrapPolylineOpacitySpinbox->setValue(0.75);
    _sameWrapPolylineOpacitySpinbox->setMaximumWidth(90);
    _sameWrapPolylineOpacitySpinbox->setToolTip("Opacity of same-wrap point collection guide polylines.");
    sameWrapPolylineLayout->addWidget(_sameWrapPolylineOpacitySpinbox);
    sameWrapPolylineLayout->addStretch();
    sameWrapLayout->addLayout(sameWrapPolylineLayout);

    _clearSameWrapAnnotationButton = new QPushButton("Clear Same-wrap Preview", sameWrapGroup);
    _clearSameWrapAnnotationButton->setToolTip("Clear the current same-wrap annotation preview without committing it.");
    sameWrapLayout->addWidget(_clearSameWrapAnnotationButton);

    sameWrapLayout->addWidget(new QLabel("Current annotations:", sameWrapGroup));
    _sameWrapTreeView = new QTreeView(sameWrapGroup);
    _sameWrapModel = new QStandardItemModel(this);
    _sameWrapTreeView->setModel(_sameWrapModel);
    _sameWrapTreeView->setSelectionBehavior(QAbstractItemView::SelectRows);
    _sameWrapTreeView->setSelectionMode(QAbstractItemView::SingleSelection);
    _sameWrapTreeView->setEditTriggers(QAbstractItemView::NoEditTriggers);
    _sameWrapTreeView->setContextMenuPolicy(Qt::CustomContextMenu);
    _sameWrapTreeView->setToolTip("Committed same-wrap annotation collections.");
    sameWrapLayout->addWidget(_sameWrapTreeView);
    connect(_sameWrapTreeView->selectionModel(), &QItemSelectionModel::selectionChanged,
            this, &WrapAnnotationWidget::onSelectionChanged);
    connect(_sameWrapTreeView, &QWidget::customContextMenuRequested,
            this, &WrapAnnotationWidget::showContextMenu);
    connect(_sameWrapTreeView, &QTreeView::doubleClicked, this, [this](const QModelIndex& index) {
        QStandardItem* item = _sameWrapModel ? _sameWrapModel->itemFromIndex(index.sibling(index.row(), kNameColumn)) : nullptr;
        if (item && item->parent() && item->parent() != _sameWrapModel->invisibleRootItem()) {
            emit pointDoubleClicked(item->data().toULongLong());
        }
    });

    layout->addWidget(sameWrapGroup);

    auto* relWindingGroup = new QGroupBox("Rel Winding Annotation", mainWidget);
    auto* relWindingLayout = new QVBoxLayout(relWindingGroup);

    auto* relWindingSourceLayout = new QHBoxLayout();
    relWindingSourceLayout->addWidget(new QLabel("Intersections:"));
    _relWindingIntersectionSourceCombo = new QComboBox(relWindingGroup);
    _relWindingIntersectionSourceCombo->addItem("Current volume", 0);
    _relWindingIntersectionSourceCombo->addItem("Patches", 1);
    _relWindingIntersectionSourceCombo->setToolTip("Choose whether relative winding points come from thresholded current-volume bands or loaded surface patches.");
    relWindingSourceLayout->addWidget(_relWindingIntersectionSourceCombo);
    relWindingSourceLayout->addStretch();
    relWindingLayout->addLayout(relWindingSourceLayout);

    auto* relWindingToleranceLayout = new QHBoxLayout();
    relWindingToleranceLayout->addWidget(new QLabel("Patch tol.:"));
    _relWindingPatchToleranceSpinbox = new QDoubleSpinBox(relWindingGroup);
    _relWindingPatchToleranceSpinbox->setRange(0.0, 1000.0);
    _relWindingPatchToleranceSpinbox->setDecimals(2);
    _relWindingPatchToleranceSpinbox->setSingleStep(0.25);
    _relWindingPatchToleranceSpinbox->setValue(1.0);
    _relWindingPatchToleranceSpinbox->setSuffix(" vx");
    _relWindingPatchToleranceSpinbox->setMaximumWidth(90);
    _relWindingPatchToleranceSpinbox->setEnabled(false);
    _relWindingPatchToleranceSpinbox->setToolTip("Maximum distance between the drawn line and a loaded surface patch intersection.");
    relWindingToleranceLayout->addWidget(_relWindingPatchToleranceSpinbox);
    relWindingToleranceLayout->addStretch();
    relWindingLayout->addLayout(relWindingToleranceLayout);

    _relWindingAnnotationButton = new QPushButton("Start Rel Winding Annotation", relWindingGroup);
    _relWindingAnnotationButton->setCheckable(true);
    _relWindingAnnotationButton->setToolTip("Draw a line to auto-create a new collection with relative winding labels. Hold Shift for decreasing order.");
    relWindingLayout->addWidget(_relWindingAnnotationButton);

    layout->addWidget(relWindingGroup);

    connect(_chkSameWrapAnnotation, &QCheckBox::toggled, this, &WrapAnnotationWidget::sameWrapAnnotationToggled);
    connect(_chkSameWrapMerge, &QCheckBox::toggled, this, &WrapAnnotationWidget::sameWrapAnnotationMergeToggled);
    connect(_sameWrapPathTypeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int) {
        emit sameWrapAnnotationPathTypeChanged(sameWrapAnnotationPathType());
    });
    connect(_sameWrapFilterTypeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int) {
        const bool filterEnabled = sameWrapAnnotationFilterType() != 0;
        _sameWrapFilterKernelSpinbox->setEnabled(filterEnabled);
        emit sameWrapAnnotationFilterTypeChanged(sameWrapAnnotationFilterType());
    });
    connect(_sameWrapFilterKernelSpinbox, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        if ((value % 2) == 0) {
            const QSignalBlocker blocker(_sameWrapFilterKernelSpinbox);
            _sameWrapFilterKernelSpinbox->setValue(value + 1);
        }
        emit sameWrapAnnotationFilterKernelSizeChanged(sameWrapAnnotationFilterKernelSize());
    });
    connect(_sameWrapSpacingSpinbox, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &WrapAnnotationWidget::sameWrapAnnotationSpacingChanged);
    connect(_sameWrapPolylineOpacitySpinbox, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &WrapAnnotationWidget::sameWrapAnnotationPolylineOpacityChanged);
    connect(_clearSameWrapAnnotationButton, &QPushButton::clicked,
            this, &WrapAnnotationWidget::sameWrapAnnotationClearRequested);
    connect(_relWindingIntersectionSourceCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int) {
        const bool patchSource = relWindingIntersectionSource() == 1;
        if (_relWindingPatchToleranceSpinbox) {
            _relWindingPatchToleranceSpinbox->setEnabled(patchSource);
        }
        emit relWindingIntersectionSourceChanged(relWindingIntersectionSource());
    });
    connect(_relWindingPatchToleranceSpinbox, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &WrapAnnotationWidget::relWindingPatchToleranceChanged);
    connect(_relWindingAnnotationButton, &QPushButton::toggled, this, [this](bool checked) {
        _relWindingAnnotationButton->setText(checked ? "Stop Rel Winding Annotation" : "Start Rel Winding Annotation");
        emit relWindingAnnotationToggled(checked);
    });

    layout->addStretch();
    setWidget(mainWidget);
}

void WrapAnnotationWidget::setRelWindingAnnotationChecked(bool checked)
{
    if (!_relWindingAnnotationButton || _relWindingAnnotationButton->isChecked() == checked) {
        return;
    }

    const QSignalBlocker blocker(_relWindingAnnotationButton);
    _relWindingAnnotationButton->setChecked(checked);
    _relWindingAnnotationButton->setText(checked ? "Stop Rel Winding Annotation" : "Start Rel Winding Annotation");
}

bool WrapAnnotationWidget::sameWrapAnnotationEnabled() const
{
    return _chkSameWrapAnnotation && _chkSameWrapAnnotation->isChecked();
}

void WrapAnnotationWidget::setSameWrapAnnotationEnabled(bool enabled)
{
    if (!_chkSameWrapAnnotation)
        return;
    // setChecked() only emits toggled() when the state actually changes, which
    // is exactly what we want: setting the current state is an inert no-op.
    _chkSameWrapAnnotation->setChecked(enabled);
}

double WrapAnnotationWidget::sameWrapAnnotationSpacing() const
{
    return _sameWrapSpacingSpinbox ? _sameWrapSpacingSpinbox->value() : 20.0;
}

double WrapAnnotationWidget::sameWrapAnnotationPolylineOpacity() const
{
    return _sameWrapPolylineOpacitySpinbox ? _sameWrapPolylineOpacitySpinbox->value() : 0.75;
}

bool WrapAnnotationWidget::sameWrapAnnotationMergeEnabled() const
{
    return _chkSameWrapMerge && _chkSameWrapMerge->isChecked();
}

int WrapAnnotationWidget::sameWrapAnnotationPathType() const
{
    return _sameWrapPathTypeCombo ? _sameWrapPathTypeCombo->currentData().toInt() : 0;
}

int WrapAnnotationWidget::sameWrapAnnotationFilterType() const
{
    return _sameWrapFilterTypeCombo ? _sameWrapFilterTypeCombo->currentData().toInt() : 0;
}

int WrapAnnotationWidget::sameWrapAnnotationFilterKernelSize() const
{
    const int kernelSize = _sameWrapFilterKernelSpinbox ? _sameWrapFilterKernelSpinbox->value() : 3;
    return std::max(3, kernelSize | 1);
}

int WrapAnnotationWidget::relWindingIntersectionSource() const
{
    return _relWindingIntersectionSourceCombo ? _relWindingIntersectionSourceCombo->currentData().toInt() : 0;
}

double WrapAnnotationWidget::relWindingPatchTolerance() const
{
    return _relWindingPatchToleranceSpinbox ? _relWindingPatchToleranceSpinbox->value() : 1.0;
}

void WrapAnnotationWidget::refreshSameWrapTree()
{
    if (!_sameWrapModel) {
        return;
    }

    _sameWrapModel->blockSignals(true);
    _sameWrapModel->clear();
    _sameWrapModel->setHorizontalHeaderLabels({"Name", "Points", "Direction", "Position"});
    _pointItems.clear();

    if (_pointCollection) {
        std::vector<VCCollection::Collection> collections;
        for (const auto& [_, collection] : _pointCollection->getAllCollections()) {
            if (isSameWrapCollection(collection)) {
                collections.push_back(collection);
            }
        }

        std::sort(collections.begin(), collections.end(),
                  [](const VCCollection::Collection& a, const VCCollection::Collection& b) {
                      return a.name < b.name;
                  });

        for (const VCCollection::Collection& collection : collections) {
            appendCollectionRow(collection);
        }
    }

    _sameWrapModel->blockSignals(false);

    if (_sameWrapTreeView) {
        _sameWrapTreeView->expandAll();
        _sameWrapTreeView->resizeColumnToContents(kNameColumn);
        _sameWrapTreeView->resizeColumnToContents(kCountColumn);
        _sameWrapTreeView->resizeColumnToContents(kDirectionColumn);
    }
}

void WrapAnnotationWidget::onCollectionsAdded(const std::vector<uint64_t>& collectionIds)
{
    if (!_pointCollection || !_sameWrapModel) {
        return;
    }

    const auto& collections = _pointCollection->getAllCollections();
    for (uint64_t collectionId : collectionIds) {
        if (findCollectionItem(collectionId)) {
            continue;
        }
        const auto it = collections.find(collectionId);
        if (it != collections.end() && isSameWrapCollection(it->second)) {
            appendCollectionRow(it->second);
        }
    }
}

void WrapAnnotationWidget::onCollectionChanged(uint64_t collectionId)
{
    if (!_pointCollection || !_sameWrapModel) {
        return;
    }

    const auto& collections = _pointCollection->getAllCollections();
    const auto it = collections.find(collectionId);
    QStandardItem* item = findCollectionItem(collectionId);
    if (it == collections.end() || !isSameWrapCollection(it->second)) {
        if (item) {
            for (int row = 0; row < item->rowCount(); ++row) {
                if (QStandardItem* pointItem = item->child(row, kNameColumn)) {
                    _pointItems.erase(pointItem->data().toULongLong());
                }
            }
            _sameWrapModel->removeRow(item->row());
        }
        return;
    }

    const VCCollection::Collection& collection = it->second;
    if (!item) {
        appendCollectionRow(collection);
        return;
    }

    item->setText(QString::fromStdString(collection.name));
    const QColor color(collection.color[0] * 255,
                       collection.color[1] * 255,
                       collection.color[2] * 255);
    item->setData(QBrush(color), Qt::DecorationRole);
    const QString direction = sameWrapDirectionText(collection);
    if (QStandardItem* directionItem = _sameWrapModel->item(item->row(), kDirectionColumn)) {
        directionItem->setText(direction);
    }
    for (int row = 0; row < item->rowCount(); ++row) {
        if (QStandardItem* pointDirectionItem = item->child(row, kDirectionColumn)) {
            pointDirectionItem->setText(direction);
        }
    }
    updateCollectionCount(item);
}

void WrapAnnotationWidget::onCollectionRemoved(uint64_t collectionId)
{
    if (!_sameWrapModel) {
        return;
    }

    if (collectionId == static_cast<uint64_t>(-1)) {
        _pointItems.clear();
        refreshSameWrapTree();
        return;
    }

    QStandardItem* item = findCollectionItem(collectionId);
    if (item) {
        for (int row = 0; row < item->rowCount(); ++row) {
            if (QStandardItem* pointItem = item->child(row, kNameColumn)) {
                _pointItems.erase(pointItem->data().toULongLong());
            }
        }
        _sameWrapModel->removeRow(item->row());
    }
}

void WrapAnnotationWidget::onPointAdded(const ColPoint& point)
{
    if (!_pointCollection || !_sameWrapModel) {
        return;
    }

    const auto& collections = _pointCollection->getAllCollections();
    const auto it = collections.find(point.collectionId);
    if (it == collections.end() || !isSameWrapCollection(it->second)) {
        return;
    }

    QStandardItem* collectionItem = findCollectionItem(point.collectionId);
    if (!collectionItem) {
        appendCollectionRow(it->second);
        return;
    }

    appendPointRow(collectionItem, it->second, point);
    updateCollectionCount(collectionItem);
}

void WrapAnnotationWidget::onPointsAdded(const std::vector<ColPoint>& points)
{
    if (!_pointCollection || !_sameWrapModel || points.empty()) {
        return;
    }

    const QSignalBlocker modelBlocker(_sameWrapModel);
    std::unordered_map<uint64_t, QStandardItem*> changedCollections;
    const auto& collections = _pointCollection->getAllCollections();
    for (const ColPoint& point : points) {
        const auto it = collections.find(point.collectionId);
        if (it == collections.end() || !isSameWrapCollection(it->second)) {
            continue;
        }

        QStandardItem* collectionItem = findCollectionItem(point.collectionId);
        if (!collectionItem) {
            appendCollectionRow(it->second);
            continue;
        }
        if (findPointItem(point.id)) {
            continue;
        }

        appendPointRow(collectionItem, it->second, point);
        changedCollections[point.collectionId] = collectionItem;
    }

    for (const auto& [collectionId, collectionItem] : changedCollections) {
        (void)collectionId;
        updateCollectionCount(collectionItem);
    }
}

void WrapAnnotationWidget::onPointChanged(const ColPoint& point)
{
    QStandardItem* pointItem = findPointItem(point.id);
    if (!pointItem) {
        onPointAdded(point);
        return;
    }

    QStandardItem* collectionItem = pointItem->parent();
    if (!collectionItem) {
        return;
    }
    if (QStandardItem* positionItem = collectionItem->child(pointItem->row(), kPositionColumn)) {
        positionItem->setText(pointPositionText(point.p));
    }
}

void WrapAnnotationWidget::onPointRemoved(uint64_t pointId)
{
    QStandardItem* pointItem = findPointItem(pointId);
    if (!pointItem) {
        return;
    }

    QStandardItem* collectionItem = pointItem->parent();
    if (!collectionItem) {
        return;
    }

    collectionItem->removeRow(pointItem->row());
    _pointItems.erase(pointId);
    updateCollectionCount(collectionItem);
}

void WrapAnnotationWidget::onSelectionChanged(const QItemSelection&, const QItemSelection&)
{
    _selectedCollectionId = 0;
    _selectedPointId = 0;

    if (!_sameWrapTreeView || !_sameWrapModel) {
        emit collectionSelected(0);
        return;
    }

    const QModelIndexList selectedIndexes = _sameWrapTreeView->selectionModel()->selectedIndexes();
    if (!selectedIndexes.isEmpty()) {
        QStandardItem* item = _sameWrapModel->itemFromIndex(selectedIndexes.first().sibling(selectedIndexes.first().row(), kNameColumn));
        if (item) {
            if (!item->parent() || item->parent() == _sameWrapModel->invisibleRootItem()) {
                _selectedCollectionId = item->data().toULongLong();
            } else {
                _selectedPointId = item->data().toULongLong();
                _selectedCollectionId = item->parent()->data().toULongLong();
            }
        }
    }

    emit collectionSelected(_selectedCollectionId);
    if (_selectedPointId != 0) {
        emit pointSelected(_selectedPointId);
    }
}

void WrapAnnotationWidget::showContextMenu(const QPoint& pos)
{
    if (!_sameWrapTreeView || _selectedCollectionId == 0) {
        return;
    }

    QMenu menu(tr("Context Menu"), _sameWrapTreeView);
    QAction* focusAction = menu.addAction(tr("Focus && Align View"));
    QAction* chosen = menu.exec(_sameWrapTreeView->viewport()->mapToGlobal(pos));
    if (chosen == focusAction) {
        emit focusViewsRequested(_selectedCollectionId, _selectedPointId);
    }
}

void WrapAnnotationWidget::selectCollection(uint64_t collectionId)
{
    if (!_sameWrapTreeView || !_sameWrapModel) {
        return;
    }

    if (collectionId == 0) {
        _sameWrapTreeView->selectionModel()->clearSelection();
        return;
    }

    QStandardItem* item = findCollectionItem(collectionId);
    if (!item) {
        return;
    }

    _sameWrapTreeView->selectionModel()->clearSelection();
    _sameWrapTreeView->selectionModel()->select(item->index(),
                                                QItemSelectionModel::Select | QItemSelectionModel::Rows);
    _sameWrapTreeView->scrollTo(item->index());
}

void WrapAnnotationWidget::selectPoint(uint64_t pointId)
{
    if (!_sameWrapTreeView || !_sameWrapModel) {
        return;
    }

    QStandardItem* pointItem = findPointItem(pointId);
    if (!pointItem) {
        return;
    }
    _sameWrapTreeView->selectionModel()->clearSelection();
    _sameWrapTreeView->selectionModel()->select(pointItem->index(),
                                                QItemSelectionModel::Select | QItemSelectionModel::Rows);
    _sameWrapTreeView->scrollTo(pointItem->index());
    _sameWrapTreeView->setFocus();
}

QStandardItem* WrapAnnotationWidget::findCollectionItem(uint64_t collectionId) const
{
    if (!_sameWrapModel) {
        return nullptr;
    }

    for (int row = 0; row < _sameWrapModel->rowCount(); ++row) {
        QStandardItem* item = _sameWrapModel->item(row, kNameColumn);
        if (item && item->data().toULongLong() == collectionId) {
            return item;
        }
    }
    return nullptr;
}

QStandardItem* WrapAnnotationWidget::findPointItem(uint64_t pointId) const
{
    if (!_sameWrapModel) {
        return nullptr;
    }

    const auto cachedIt = _pointItems.find(pointId);
    if (cachedIt != _pointItems.end() && cachedIt->second.isValid()) {
        return _sameWrapModel->itemFromIndex(cachedIt->second);
    }

    for (int row = 0; row < _sameWrapModel->rowCount(); ++row) {
        QStandardItem* collectionItem = _sameWrapModel->item(row, kNameColumn);
        if (!collectionItem) {
            continue;
        }
        for (int pointRow = 0; pointRow < collectionItem->rowCount(); ++pointRow) {
            QStandardItem* pointItem = collectionItem->child(pointRow, kNameColumn);
            if (pointItem && pointItem->data().toULongLong() == pointId) {
                return pointItem;
            }
        }
    }
    return nullptr;
}

void WrapAnnotationWidget::appendCollectionRow(const VCCollection::Collection& collection)
{
    if (!_sameWrapModel) {
        return;
    }

    QStandardItem* nameItem = readOnlyItem(QString::fromStdString(collection.name));
    const QColor color(collection.color[0] * 255,
                       collection.color[1] * 255,
                       collection.color[2] * 255);
    nameItem->setData(QBrush(color), Qt::DecorationRole);
    nameItem->setData(QVariant::fromValue(collection.id));

    QStandardItem* countItem = readOnlyItem(QString::number(collection.points.size()));
    QStandardItem* directionItem = readOnlyItem(sameWrapDirectionText(collection));
    QStandardItem* positionItem = readOnlyItem();

    int insertRow = _sameWrapModel->rowCount();
    for (int row = 0; row < _sameWrapModel->rowCount(); ++row) {
        QStandardItem* existing = _sameWrapModel->item(row, kNameColumn);
        if (existing && existing->text() > nameItem->text()) {
            insertRow = row;
            break;
        }
    }
    _sameWrapModel->insertRow(insertRow, {nameItem, countItem, directionItem, positionItem});

    std::vector<ColPoint> points;
    points.reserve(collection.points.size());
    for (const auto& [_, point] : collection.points) {
        points.push_back(point);
    }
    std::sort(points.begin(), points.end(),
              [](const ColPoint& a, const ColPoint& b) {
                  return a.id < b.id;
              });

    for (const ColPoint& point : points) {
        appendPointRow(nameItem, collection, point);
    }
}

void WrapAnnotationWidget::appendPointRow(QStandardItem* collectionItem,
                                          const VCCollection::Collection& collection,
                                          const ColPoint& point)
{
    if (!collectionItem) {
        return;
    }

    QStandardItem* pointItem = readOnlyItem(QString::number(point.id));
    pointItem->setData(QVariant::fromValue(point.id));
    QStandardItem* emptyCountItem = readOnlyItem();
    QStandardItem* pointDirectionItem = readOnlyItem(sameWrapDirectionText(collection));
    QStandardItem* pointPositionItem = readOnlyItem(pointPositionText(point.p));

    int insertRow = collectionItem->rowCount();
    if (insertRow > 0) {
        QStandardItem* last = collectionItem->child(insertRow - 1, kNameColumn);
        if (last && last->data().toULongLong() <= point.id) {
            collectionItem->appendRow({pointItem, emptyCountItem, pointDirectionItem, pointPositionItem});
            _pointItems[point.id] = pointItem->index();
            return;
        }
    }
    for (int row = 0; row < insertRow; ++row) {
        QStandardItem* existing = collectionItem->child(row, kNameColumn);
        if (existing && existing->data().toULongLong() > point.id) {
            insertRow = row;
            break;
        }
    }
    collectionItem->insertRow(insertRow, {pointItem, emptyCountItem, pointDirectionItem, pointPositionItem});
    _pointItems[point.id] = pointItem->index();
}

void WrapAnnotationWidget::updateCollectionCount(QStandardItem* collectionItem)
{
    if (!_sameWrapModel || !collectionItem) {
        return;
    }

    QStandardItem* countItem = _sameWrapModel->item(collectionItem->row(), kCountColumn);
    if (countItem) {
        countItem->setText(QString::number(collectionItem->rowCount()));
    }
}
