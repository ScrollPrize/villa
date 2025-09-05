#include "CPointCollectionWidget.hpp"

#include <QStandardItem>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <QColorDialog>
#include <QFileDialog>
#include <QKeyEvent>
#include <QMessageBox>
#include <QLabel>
#include <QVBoxLayout>
#include <QHBoxLayout>
 
#include "vc/ui/VCCollection.hpp"
 


CPointCollectionWidget::CPointCollectionWidget(VCCollection *collection, QWidget *parent)
    : QDockWidget("Point Collections", parent), _point_collection(collection)
{
    if (!_point_collection) {
        throw std::invalid_argument("CPointCollectionWidget requires a valid VCCollection.");
    }

    setupUi();

    connect(_point_collection, &VCCollection::collectionAdded, this, &CPointCollectionWidget::onCollectionAdded);
    connect(_point_collection, &VCCollection::collectionChanged, this, &CPointCollectionWidget::onCollectionChanged);
    connect(_point_collection, &VCCollection::collectionRemoved, this, &CPointCollectionWidget::onCollectionRemoved);
    connect(_point_collection, &VCCollection::pointAdded, this, &CPointCollectionWidget::onPointAdded);
    connect(_point_collection, &VCCollection::pointChanged, this, &CPointCollectionWidget::onPointChanged);
    connect(_point_collection, &VCCollection::pointRemoved, this, &CPointCollectionWidget::onPointRemoved);

    refreshTree();
}

void CPointCollectionWidget::setupUi()
{
    QWidget *main_widget = new QWidget(this);
    QVBoxLayout *layout = new QVBoxLayout(main_widget);

    _tree_view = new QTreeView(main_widget);
    _model = new QStandardItemModel(this);
    _tree_view->setModel(_model);
    _tree_view->setSelectionBehavior(QAbstractItemView::SelectRows);
    _tree_view->setSelectionMode(QAbstractItemView::SingleSelection);
    layout->addWidget(_tree_view);

    connect(_tree_view->selectionModel(), &QItemSelectionModel::selectionChanged, this, &CPointCollectionWidget::onSelectionChanged);
    connect(_tree_view, &QTreeView::doubleClicked, this, [this](const QModelIndex &index) {
        // Get the index for the first column in the same row
        QModelIndex id_index = index.sibling(index.row(), 0);
        QStandardItem *item = _model->itemFromIndex(id_index);
        // Check if it's a point item (i.e., it has a parent)
        if (item && (item->parent() != nullptr && item->parent() != _model->invisibleRootItem())) {
            uint64_t pointId = item->data().toULongLong();
            emit pointDoubleClicked(pointId);
        }
    });

    // Collection Metadata
    _collection_metadata_group = new QGroupBox("Collection Metadata");
    QVBoxLayout *collection_layout = new QVBoxLayout(_collection_metadata_group);
    
    QHBoxLayout *rename_layout = new QHBoxLayout();
    _collection_name_edit = new QLineEdit();
    rename_layout->addWidget(_collection_name_edit);
    _new_name_button = new QPushButton("New Collection");
    rename_layout->addWidget(_new_name_button);
    collection_layout->addLayout(rename_layout);

    connect(_collection_name_edit, &QLineEdit::textEdited, this, &CPointCollectionWidget::onNameEdited);
    connect(_new_name_button, &QPushButton::clicked, this, &CPointCollectionWidget::onNewNameClicked);

    _absolute_winding_checkbox = new QCheckBox("Absolute Winding Number");
    collection_layout->addWidget(_absolute_winding_checkbox);

    _color_button = new QPushButton("Change Color");
    collection_layout->addWidget(_color_button);

    QHBoxLayout *fill_layout = new QHBoxLayout();
    _fill_winding_plus_button = new QPushButton("Fill +");
    _fill_winding_minus_button = new QPushButton("Fill -");
    _fill_winding_equals_button = new QPushButton("Fill =");
    fill_layout->addWidget(_fill_winding_plus_button);
    fill_layout->addWidget(_fill_winding_minus_button);
    fill_layout->addWidget(_fill_winding_equals_button);
    collection_layout->addLayout(fill_layout);

    layout->addWidget(_collection_metadata_group);
 
    connect(_absolute_winding_checkbox, &QCheckBox::stateChanged, this, &CPointCollectionWidget::onAbsoluteWindingChanged);
    connect(_color_button, &QPushButton::clicked, this, &CPointCollectionWidget::onColorButtonClicked);
    connect(_fill_winding_plus_button, &QPushButton::clicked, this, &CPointCollectionWidget::onFillWindingPlusClicked);
    connect(_fill_winding_minus_button, &QPushButton::clicked, this, &CPointCollectionWidget::onFillWindingMinusClicked);
    connect(_fill_winding_equals_button, &QPushButton::clicked, this, &CPointCollectionWidget::onFillWindingEqualsClicked);

    // Point Metadata
    _point_metadata_group = new QGroupBox("Point Metadata");
    QVBoxLayout *point_layout = new QVBoxLayout(_point_metadata_group);
    
    QHBoxLayout *winding_layout = new QHBoxLayout();
    _winding_enabled_checkbox = new QCheckBox("Enabled");
    winding_layout->addWidget(_winding_enabled_checkbox);
    winding_layout->addWidget(new QLabel("Winding:"));
    _winding_spinbox = new QDoubleSpinBox();
    _winding_spinbox->setRange(-1000, 1000);
    _winding_spinbox->setDecimals(2);
    _winding_spinbox->setSingleStep(0.1);
    winding_layout->addWidget(_winding_spinbox);
    point_layout->addLayout(winding_layout);

    layout->addWidget(_point_metadata_group);
 
    connect(_winding_enabled_checkbox, &QCheckBox::stateChanged, this, &CPointCollectionWidget::onWindingEnabledChanged);
    connect(_winding_spinbox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &CPointCollectionWidget::onWindingEdited);
 
    layout->addStretch();
 
    QHBoxLayout *file_layout = new QHBoxLayout();
    _load_button = new QPushButton("Load");
    file_layout->addWidget(_load_button);
    _save_button = new QPushButton("Save");
    file_layout->addWidget(_save_button);
    _reset_button = new QPushButton("Clear All Points");
    file_layout->addWidget(_reset_button);
    layout->addLayout(file_layout);
 
    connect(_load_button, &QPushButton::clicked, this, &CPointCollectionWidget::onLoadClicked);
    connect(_save_button, &QPushButton::clicked, this, &CPointCollectionWidget::onSaveClicked);
    connect(_reset_button, &QPushButton::clicked, this, &CPointCollectionWidget::onResetClicked);
 
    setWidget(main_widget);

    updateMetadataWidgets();
}


void CPointCollectionWidget::refreshTree()
{
    if (_model) {
        _model->blockSignals(true);

        // Remove all rows before clearing
        if (_model->rowCount() > 0) {
            _model->removeRows(0, _model->rowCount());
        }

        _model->clear();
        _model->setHorizontalHeaderLabels({"Name", "Points"});
        _model->blockSignals(false);
    }

    if (!_point_collection) {
        return;
    }

    // Get collections and sort them by name
    const auto& all_collections_map = _point_collection->getAllCollections();
    std::vector<VCCollection::Collection> sorted_collections;
    sorted_collections.reserve(all_collections_map.size());
    for (const auto& pair : all_collections_map) {
        sorted_collections.push_back(pair.second);
    }
    std::sort(sorted_collections.begin(), sorted_collections.end(),
              [](const VCCollection::Collection& a, const VCCollection::Collection& b) {
        return a.name < b.name;
    });

    // Iterate through sorted collections and add to tree
    for (const auto& collection : sorted_collections) {
        QStandardItem *name_item = new QStandardItem(QString::fromStdString(collection.name));
        QColor color(collection.color[0] * 255, collection.color[1] * 255, collection.color[2] * 255);
        name_item->setData(QBrush(color), Qt::DecorationRole);
        name_item->setData(QVariant::fromValue(collection.id));
        name_item->setFlags(name_item->flags() & ~Qt::ItemIsEditable);
        
        QStandardItem *count_item = new QStandardItem(QString::number(collection.points.size()));
        count_item->setFlags(count_item->flags() & ~Qt::ItemIsEditable);
        
        _model->appendRow({name_item, count_item});

        // Get points and sort them by ID
        std::vector<ColPoint> sorted_points;
        sorted_points.reserve(collection.points.size());
        for (const auto& pair : collection.points) {
            sorted_points.push_back(pair.second);
        }
        std::sort(sorted_points.begin(), sorted_points.end(),
                  [](const ColPoint& a, const ColPoint& b) {
            return a.id < b.id;
        });

        // Add sorted points to the collection item
        for (const auto& point : sorted_points) {
            QStandardItem *id_item = new QStandardItem(QString::number(point.id));
            id_item->setData(QVariant::fromValue(point.id));
            id_item->setFlags(id_item->flags() & ~Qt::ItemIsEditable);
            
            QStandardItem *pos_item = new QStandardItem(QString("{%1, %2, %3}").arg(point.p[0]).arg(point.p[1]).arg(point.p[2]));
            pos_item->setFlags(pos_item->flags() & ~Qt::ItemIsEditable);
            
            name_item->appendRow({id_item, pos_item});
        }
    }

    _tree_view->expandAll();
}

void CPointCollectionWidget::onResetClicked()
{
    if (_point_collection) {
        _tree_view->selectionModel()->clear();
        _point_collection->clearAll();
    }
}

void CPointCollectionWidget::onCollectionAdded(uint64_t collectionId)
{
    const auto& collection = _point_collection->getAllCollections().at(collectionId);
    QStandardItem *name_item = new QStandardItem(QString::fromStdString(collection.name));
    QColor color(collection.color[0] * 255, collection.color[1] * 255, collection.color[2] * 255);
    name_item->setData(QBrush(color), Qt::DecorationRole);
    name_item->setData(QVariant::fromValue(collection.id));
    name_item->setFlags(name_item->flags() & ~Qt::ItemIsEditable);
    
    QStandardItem *count_item = new QStandardItem(QString::number(collection.points.size()));
    count_item->setFlags(count_item->flags() & ~Qt::ItemIsEditable);
    
    _model->appendRow({name_item, count_item});

    for(const auto& point_pair : collection.points) {
        onPointAdded(point_pair.second);
    }
}

void CPointCollectionWidget::onCollectionChanged(uint64_t collectionId)
{
    QStandardItem* item = findCollectionItem(collectionId);
    if (item) {
        const auto& collection = _point_collection->getAllCollections().at(collectionId);
        if (item->text() != QString::fromStdString(collection.name)) {
            item->setText(QString::fromStdString(collection.name));
        }
        QColor color(collection.color[0] * 255, collection.color[1] * 255, collection.color[2] * 255);
        item->setData(QBrush(color), Qt::DecorationRole);
        // Also update metadata display if it's the selected collection
        if (collectionId == _selected_collection_id) {
            updateMetadataWidgets();
        }
    }
}

void CPointCollectionWidget::onCollectionRemoved(uint64_t collectionId)
{
    if (collectionId == static_cast<uint64_t>(-1)) { // Clear all
        // Properly clear the model
        if (_model) {
            _model->blockSignals(true);
            _model->removeRows(0, _model->rowCount());
            _model->clear();
            _model->setHorizontalHeaderLabels({"Name", "Points"});
            _model->blockSignals(false);
        }
        return;
    }

    QStandardItem* item = findCollectionItem(collectionId);
    if (item) {
        _model->removeRow(item->row());
    }
}

void CPointCollectionWidget::onPointAdded(const ColPoint& point)
{
    QStandardItem* collection_item = findCollectionItem(point.collectionId);
    if (collection_item) {
        QStandardItem *id_item = new QStandardItem(QString::number(point.id));
        id_item->setData(QVariant::fromValue(point.id));
        id_item->setFlags(id_item->flags() & ~Qt::ItemIsEditable);
        
        QStandardItem *pos_item = new QStandardItem(QString("{%1, %2, %3}").arg(point.p[0]).arg(point.p[1]).arg(point.p[2]));
        pos_item->setFlags(pos_item->flags() & ~Qt::ItemIsEditable);
        
        collection_item->appendRow({id_item, pos_item});
        
        // Update count
        QStandardItem* count_item = _model->item(collection_item->row(), 1);
        if(count_item) {
            count_item->setText(QString::number(collection_item->rowCount()));
        }
    }
}

void CPointCollectionWidget::onPointChanged(const ColPoint& point)
{
    // For now, just update the metadata if it's the selected point
    if (point.id == _selected_point_id) {
        updateMetadataWidgets();
    }
}

void CPointCollectionWidget::onPointRemoved(uint64_t pointId)
{
    // Find the item corresponding to the pointId and remove it
    for (int i = 0; i < _model->rowCount(); ++i) {
        QStandardItem *collection_item = _model->item(i);
        if (collection_item) {
            for (int j = 0; j < collection_item->rowCount(); ++j) {
                QStandardItem *point_item = collection_item->child(j);
                if (point_item && point_item->data().toULongLong() == pointId) {
                    collection_item->removeRow(j);
                    // Update count
                    QStandardItem* count_item = _model->item(collection_item->row(), 1);
                    if(count_item) {
                        count_item->setText(QString::number(collection_item->rowCount()));
                    }
                    return;
                }
            }
        }
    }
}

void CPointCollectionWidget::onSelectionChanged(const QItemSelection &selected, const QItemSelection &deselected)
{
    _selected_collection_id = 0;
    _selected_point_id = 0;

    QModelIndexList selected_indexes = _tree_view->selectionModel()->selectedIndexes();
    if (!selected_indexes.isEmpty()) {
        QModelIndex selected_index = selected_indexes.first();
        QStandardItem *item = _model->itemFromIndex(selected_index);
        if (item) {
            if (item->parent() == nullptr || item->parent() == _model->invisibleRootItem()) {
                _selected_collection_id = item->data().toULongLong();
            } else {
                _selected_point_id = item->data().toULongLong();
                QStandardItem* parent_item = item->parent();
                if (parent_item) {
                    _selected_collection_id = parent_item->data().toULongLong();
                }
            }
        }
    }
    updateMetadataWidgets();
    emit collectionSelected(_selected_collection_id);
    if (_selected_point_id != 0) {
        emit pointSelected(_selected_point_id);
    }
}

void CPointCollectionWidget::updateMetadataWidgets()
{
    bool collection_selected = (_selected_collection_id != 0);
    bool point_selected = (_selected_point_id != 0);

    _collection_metadata_group->setEnabled(collection_selected);
    _point_metadata_group->setEnabled(point_selected);

    if (collection_selected) {
        const auto& collections = _point_collection->getAllCollections();
        if (collections.count(_selected_collection_id)) {
            const auto& collection = collections.at(_selected_collection_id);
            
            // Temporarily block signals to prevent feedback loop
            _collection_name_edit->blockSignals(true);
            _collection_name_edit->setText(QString::fromStdString(collection.name));
            _collection_name_edit->blockSignals(false);

            _absolute_winding_checkbox->blockSignals(true);
            _absolute_winding_checkbox->setChecked(collection.metadata.absolute_winding_number);
            _absolute_winding_checkbox->blockSignals(false);

            QPalette pal = _color_button->palette();
            QColor q_color(collection.color[0] * 255, collection.color[1] * 255, collection.color[2] * 255);
            pal.setColor(QPalette::Button, q_color);
            _color_button->setAutoFillBackground(true);
            _color_button->setPalette(pal);
            _color_button->update();
        }
    } else {
        _collection_name_edit->clear();
        _absolute_winding_checkbox->setChecked(false);
        _color_button->setAutoFillBackground(false);
    }

    if (point_selected) {
        auto point_opt = _point_collection->getPoint(_selected_point_id);
        if (point_opt) {
            _winding_spinbox->blockSignals(true);
            _winding_enabled_checkbox->blockSignals(true);

            bool winding_enabled = !std::isnan(point_opt->winding_annotation);
            _winding_enabled_checkbox->setChecked(winding_enabled);
            _winding_spinbox->setEnabled(winding_enabled);
            if (winding_enabled) {
                _winding_spinbox->setValue(point_opt->winding_annotation);
            } else {
                _winding_spinbox->setValue(0);
            }

            _winding_spinbox->blockSignals(false);
            _winding_enabled_checkbox->blockSignals(false);
        }
    } else {
        _winding_spinbox->blockSignals(true);
        _winding_enabled_checkbox->blockSignals(true);

        _winding_enabled_checkbox->setChecked(false);
        _winding_spinbox->setEnabled(false);
        _winding_spinbox->setValue(0);
        
        _winding_spinbox->blockSignals(false);
        _winding_enabled_checkbox->blockSignals(false);
    }
}

void CPointCollectionWidget::onNameEdited(const QString &name)
{
    if (_selected_collection_id != 0) {
        std::string new_name = name.toStdString();
        if (!new_name.empty()) {
            _point_collection->renameCollection(_selected_collection_id, new_name);
        }
    }
}

void CPointCollectionWidget::onNewNameClicked()
{
    std::string new_name = _point_collection->generateNewCollectionName("col");
    uint64_t new_id = _point_collection->addCollection(new_name);
    selectCollection(new_id);
}

void CPointCollectionWidget::onAbsoluteWindingChanged(int state)
{
    if (_selected_collection_id != 0) {
        const auto& collections = _point_collection->getAllCollections();
        if (collections.count(_selected_collection_id)) {
            auto metadata = collections.at(_selected_collection_id).metadata;
            metadata.absolute_winding_number = (state == Qt::Checked);
            _point_collection->setCollectionMetadata(_selected_collection_id, metadata);
        }
    }
}

void CPointCollectionWidget::onColorButtonClicked()
{
    if (_selected_collection_id == 0) return;

    const auto& collection = _point_collection->getAllCollections().at(_selected_collection_id);
    QColor initial_color(collection.color[0] * 255, collection.color[1] * 255, collection.color[2] * 255);

    QColor color = QColorDialog::getColor(initial_color, this, "Select Collection Color");

    if (color.isValid()) {
        _point_collection->setCollectionColor(_selected_collection_id, { (float)color.redF(), (float)color.greenF(), (float)color.blueF() });
    }
}

void CPointCollectionWidget::onWindingEdited(double value)
{
    if (_selected_point_id != 0) {
        auto point_opt = _point_collection->getPoint(_selected_point_id);
        if (point_opt) {
            ColPoint updated_point = *point_opt;
            updated_point.winding_annotation = value;
            _point_collection->updatePoint(updated_point);
        }
    }
}

void CPointCollectionWidget::onWindingEnabledChanged(int state)
{
    if (_selected_point_id != 0) {
        auto point_opt = _point_collection->getPoint(_selected_point_id);
        if (point_opt) {
            ColPoint updated_point = *point_opt;
            if (state == Qt::Checked) {
                updated_point.winding_annotation = _winding_spinbox->value();
            } else {
                updated_point.winding_annotation = std::nan("");
            }
            _point_collection->updatePoint(updated_point);
        }
    }
}

void CPointCollectionWidget::onFillWindingPlusClicked()
{
    if (_selected_collection_id != 0) {
        _point_collection->autoFillWindingNumbers(_selected_collection_id, VCCollection::WindingFillMode::Incremental);
    }
}

void CPointCollectionWidget::onFillWindingMinusClicked()
{
    if (_selected_collection_id != 0) {
        _point_collection->autoFillWindingNumbers(_selected_collection_id, VCCollection::WindingFillMode::Decremental);
    }
}

void CPointCollectionWidget::onFillWindingEqualsClicked()
{
    if (_selected_collection_id != 0) {
        _point_collection->autoFillWindingNumbers(_selected_collection_id, VCCollection::WindingFillMode::Constant);
    }
}
 
void CPointCollectionWidget::onSaveClicked()
{
    QString fileName = QFileDialog::getSaveFileName(this, tr("Save Point Collection"), "", tr("JSON Files (*.json)"));
    if (fileName.isEmpty()) {
        return;
    }
 
    if (_point_collection) {
        _point_collection->saveToJSON(fileName.toStdString());
    }
}
 
void CPointCollectionWidget::onLoadClicked()
{
    QString fileName = QFileDialog::getOpenFileName(this, tr("Load Point Collection"), "", tr("JSON Files (*.json)"));
    if (fileName.isEmpty()) {
        return;
    }
 
    if (_point_collection) {
       try {
           if (_point_collection->loadFromJSON(fileName.toStdString())) {
               refreshTree();
           }
       } catch (const std::exception& e) {
           QMessageBox::critical(this, "Error Loading File", e.what());
       }
    }
}
 
void CPointCollectionWidget::selectCollection(uint64_t collectionId)
{
    QStandardItem* item = findCollectionItem(collectionId);
    if (item) {
        _tree_view->selectionModel()->clearSelection();
        _tree_view->selectionModel()->select(item->index(), QItemSelectionModel::Select | QItemSelectionModel::Rows);
        _tree_view->scrollTo(item->index());
    }
}

QStandardItem* CPointCollectionWidget::findCollectionItem(uint64_t collectionId)
{
    for (int i = 0; i < _model->rowCount(); ++i) {
        QStandardItem *item = _model->item(i);
        if (item && item->data().toULongLong() == collectionId) {
            return item;
        }
    }
    return nullptr;
}

void CPointCollectionWidget::selectPoint(uint64_t pointId)
{
    if (_selected_point_id == pointId) {
        return;
    }

    // Find the item corresponding to the pointId
    for (int i = 0; i < _model->rowCount(); ++i) {
        QStandardItem *collection_item = _model->item(i);
        if (collection_item) {
            for (int j = 0; j < collection_item->rowCount(); ++j) {
                QStandardItem *point_item = collection_item->child(j);
                if (point_item && point_item->data().toULongLong() == pointId) {
                    _tree_view->selectionModel()->clearSelection();
                    _tree_view->selectionModel()->select(point_item->index(), QItemSelectionModel::Select | QItemSelectionModel::Rows);
                    _tree_view->scrollTo(point_item->index());
                    _tree_view->setFocus();
                    return;
                }
            }
        }
    }
}

void CPointCollectionWidget::keyPressEvent(QKeyEvent *event)
{
    if (event->key() == Qt::Key_Delete && _selected_point_id != 0) {
        _point_collection->removePoint(_selected_point_id);
        event->accept();
    } else {
        QDockWidget::keyPressEvent(event);
    }
}

CPointCollectionWidget::~CPointCollectionWidget() {
    if (_tree_view && _tree_view->selectionModel()) {
        disconnect(_tree_view->selectionModel(), nullptr, this, nullptr);
    }

    // Clear model safely
    if (_model) {
        _model->blockSignals(true);
        _model->clear();
    }
}


