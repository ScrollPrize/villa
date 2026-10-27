#include "SurfacePanelController.hpp"
#include "BindCoordinatesPrompt.hpp"

#include "SurfaceTreeWidget.hpp"
#include "ViewerManager.hpp"
#include "CState.hpp"
#include <vc_object_store/objects/GeometryCoordinates.hpp>
#include <vc_object_store/store/GeometryViews.hpp>
#include "ObjectStoreRegions.hpp"
#include "volume_viewers/CChunkedVolumeViewer.hpp"
#include "elements/DropdownChecklistButton.hpp"
#include "VCSettings.hpp"

#include "vc/core/types/Volume.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/ui/VCCollection.hpp"

#include <nlohmann/json.hpp>

#include <QAction>
#include <QApplication>
#include <QBrush>
#include <QCheckBox>
#include <QClipboard>
#include <QComboBox>
#include <QDateTime>
#include <QDialog>
#include <QDialogButtonBox>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QInputDialog>
#include <QLabel>
#include <QLineEdit>
#include <QMenu>
#include <QMessageBox>
#include <QModelIndex>
#include <QPushButton>
#include <QSettings>
#include <QSignalBlocker>
#include <QSizePolicy>
#include <QStandardItem>
#include <QStandardItemModel>
#include <QStyle>
#include <QToolButton>
#include <QTreeWidget>
#include <QTreeWidgetItemIterator>
#include <QVBoxLayout>
#include <QVector>

#include <algorithm>
#include <limits>
#include <cmath>

namespace {

constexpr double kFocusPointFilterRadius = 10.0;
// Distance within which a point-set point lies on a surface.
constexpr double kPointSetTolerance = 2.0;
constexpr double kZRangeFilterLowerDefault = 0.0;
constexpr double kZRangeFilterUpperDefault = 1000000.0;
constexpr auto kSurfaceColumnSettingsGroup = "surface_panel/columns";
constexpr auto kCurrentOnlyFilterSettingsKey = "surface_panel/filters/current_only";
constexpr int kUnboundRole = Qt::UserRole + 1;

bool z_range_filter_active(QDoubleSpinBox* lower, QDoubleSpinBox* upper)
{
    if (!lower || !upper) {
        return false;
    }

    return lower->value() != kZRangeFilterLowerDefault ||
           upper->value() != kZRangeFilterUpperDefault;
}

QString surface_column_settings_key(int column)
{
    return QStringLiteral("%1/column_%2_visible").arg(kSurfaceColumnSettingsGroup).arg(column);
}

bool surface_column_default_visible(int column)
{
    return column != SURFACE_LONG_ID_COLUMN &&
           column != SURFACE_AVG_COST_COLUMN;
}

bool has_tag(const SurfaceCatalog::Row& row, const char* tag)
{
    return std::find(row.record.tags.begin(), row.record.tags.end(), tag) != row.record.tags.end();
}

QString display_name(const SurfaceCatalog::Row& row)
{
    return QString::fromStdString(row.record.name.empty() ? row.record.uuid : row.record.name);
}

nlohmann::json row_meta(const SurfaceCatalog::Row& row)
{
    auto meta = nlohmann::json::parse(row.record.meta, nullptr, false);
    return meta.is_object() ? meta : nlohmann::json::object();
}

QString long_id(const nlohmann::json& meta)
{
    const auto found = meta.find("segment_long_id");
    if (found != meta.end() && found->is_string()) return QString::fromStdString(found->get<std::string>());
    return {};
}

double number_or(const nlohmann::json& meta, const char* key, double fallback)
{
    const auto found = meta.find(key);
    return found != meta.end() && found->is_number() ? found->get<double>() : fallback;
}

std::string string_or(const nlohmann::json& meta, const char* key)
{
    const auto found = meta.find(key);
    return found != meta.end() && found->is_string() ? found->get<std::string>() : std::string{};
}

} // namespace

SurfacePanelController::SurfacePanelController(const UiRefs& ui,
                                               CState* state,
                                               SurfaceCatalog* catalog,
                                               ViewerManager* viewerManager,
                                               std::function<CChunkedVolumeViewer*()> segmentationViewerProvider,
                                               QObject* parent)
    : QObject(parent)
    , _ui(ui)
    , _state(state)
    , _catalog(catalog)
    , _viewerManager(viewerManager)
    , _segmentationViewerProvider(std::move(segmentationViewerProvider))
{
    if (_ui.treeWidget) {
        _ui.treeWidget->setContextMenuPolicy(Qt::CustomContextMenu);
        setupSurfaceColumnMenu();
        restoreSurfaceColumnVisibility();
        connect(_ui.treeWidget, &QTreeWidget::itemSelectionChanged,
                this, &SurfacePanelController::handleTreeSelectionChanged);
        connect(_ui.treeWidget, &QWidget::customContextMenuRequested,
                this, &SurfacePanelController::showContextMenu);
        connect(_ui.treeWidget, &QTreeWidget::itemDoubleClicked,
                this, [this](QTreeWidgetItem* item, int /*column*/) {
            if (!item) return;
            const QString segId = item->data(SURFACE_ID_COLUMN, Qt::UserRole).toString();
            if (!segId.isEmpty()) {
                emit focusSurfaceRequested(segId);
            }
        });
    }
    if (_catalog) {
        connect(_catalog, &SurfaceCatalog::rowsChanged, this, [this]() {
            syncTree();
            rebuildRecordTagFilterModel();
            refreshStoreFilters();
            applyFilters();
        });
        connect(_catalog, &SurfaceCatalog::errorChanged, this, [this](const QString& error) {
            if (!error.isEmpty()) {
                emit statusMessageRequested(tr("Cannot list surfaces: %1").arg(error), 8000);
            }
        });
    }
}

SurfaceTreeWidgetItem* SurfacePanelController::itemFor(const std::string& id) const
{
    if (!_ui.treeWidget || id.empty()) {
        return nullptr;
    }
    const QString idQString = QString::fromStdString(id);
    QTreeWidgetItemIterator it(_ui.treeWidget);
    while (*it) {
        if ((*it)->data(SURFACE_ID_COLUMN, Qt::UserRole).toString() == idQString) {
            return static_cast<SurfaceTreeWidgetItem*>(*it);
        }
        ++it;
    }
    return nullptr;
}

void SurfacePanelController::setItemRow(SurfaceTreeWidgetItem* item, const SurfaceCatalog::Row& row)
{
    const auto meta = row_meta(row);
    const QString name = display_name(row);
    const QString longId = long_id(meta);
    const double avgCost = number_or(meta, "avg_cost", -1.0);
    item->setData(SURFACE_ID_COLUMN, Qt::UserRole, QString::fromStdString(row.record.uuid));
    item->setData(SURFACE_ID_COLUMN, kUnboundRole, !row.bound());
    item->setText(SURFACE_ID_COLUMN, name);
    const auto coordinates=row_meta(row).value("coordinates",nlohmann::json::object());
    const auto source=coordinates.value("volume_uuid",nlohmann::json(nullptr));
    item->setToolTip(SURFACE_ID_COLUMN,tr("Source volume: %1\n%2")
        .arg(source.is_string()?QString::fromStdString(source.get<std::string>()):tr("Unknown"))
        .arg(row.mapping?tr("Available in the open volume"):tr("No resolved mapping to the open volume")));
    item->setText(SURFACE_LONG_ID_COLUMN, longId);
    // The store's area (stored units squared) in the open volume, in cm²: "-"
    // without geometry, a mapping or a voxel size.
    const auto volume = _catalog ? _catalog->volume() : nullptr;
    const double voxelUm = volume ? volume->voxelSize() : 0.0;
    const bool areaKnown = row.record.surface && row.record.surface->area > 0 && row.mapping && voxelUm > 0;
    const double areaCm2 = areaKnown
        ? row.record.surface->area * std::pow(row.mapping->lengthScale(), 2) * voxelUm * voxelUm * 1e-8 : 0.0;
    item->setText(SURFACE_AREA_COLUMN, areaKnown ? QString::number(areaCm2, 'f', 3) : QStringLiteral("-"));
    item->setText(SURFACE_AVG_COST_COLUMN, avgCost >= 0.0 ? QString::number(avgCost, 'f', 3) : QStringLiteral("-"));
    item->setText(TIMESTAMP_COLUMN, QString::fromStdString(row.record.changedAt));
    item->updateItemIcon(has_tag(row, "approved"), has_tag(row, "defective"));
    applyBindingStyle(item);
}

void SurfacePanelController::syncTree()
{
    if (!_ui.treeWidget || !_catalog) {
        return;
    }
    const QSignalBlocker blocker{_ui.treeWidget};
    const bool sorting = _ui.treeWidget->isSortingEnabled();
    _ui.treeWidget->setSortingEnabled(false);
    const auto& rows = _catalog->rows();
    std::set<std::string> present;
    for (int i = _ui.treeWidget->topLevelItemCount() - 1; i >= 0; --i) {
        auto* item = static_cast<SurfaceTreeWidgetItem*>(_ui.treeWidget->topLevelItem(i));
        const auto id = item->data(SURFACE_ID_COLUMN, Qt::UserRole).toString().toStdString();
        const auto row = rows.find(id);
        if (row == rows.end()) {
            delete item;
            continue;
        }
        setItemRow(item, row->second);
        present.insert(id);
    }
    for (const auto& [id, row] : rows) {
        if (!present.contains(id)) {
            setItemRow(new SurfaceTreeWidgetItem(_ui.treeWidget), row);
        }
    }
    _ui.treeWidget->setSortingEnabled(sorting);
    for (int column = 0; column <= TIMESTAMP_COLUMN; ++column) {
        _ui.treeWidget->resizeColumnToContents(column);
    }
    if (!_currentSurfaceId.empty() && !rows.contains(_currentSurfaceId)) {
        // The open surface left the store (retired) or this volume.
        _currentSurfaceId.clear();
        _opened.reset();
        resetTagUi();
        emit surfaceSelectionCleared();
    } else {
        updateTagCheckboxStates(_catalog->row(_currentSurfaceId));
    }
}

void SurfacePanelController::activate(const std::string& id, bool preserveEditing,
                                      std::optional<SurfaceCatalog::Opened> opened)
{
    const QString idQString = QString::fromStdString(id);
    std::shared_ptr<QuadSurface> surface;
    _opened.reset();
    try {
        _opened = opened ? std::move(opened) : std::optional(_catalog->open(id));
        surface = _opened->surface;
    } catch (const SurfaceCatalog::NoGeometry&) {
        const auto* row = _catalog->row(id);
        if (row && row->catalog) {
            fetchThenActivate(id, preserveEditing);
            return;
        }
        emit statusMessageRequested(tr("Cannot open surface %1: it has no geometry").arg(idQString), 8000);
    } catch (const std::exception& error) {
        emit statusMessageRequested(tr("Cannot open surface %1: %2").arg(idQString, QString::fromUtf8(error.what())), 8000);
    }
    _currentSurfaceId = id;
    updateTagCheckboxStates(_catalog->row(id));
    if (_segmentationViewerProvider) {
        if (auto* viewer = _segmentationViewerProvider()) {
            const auto* row = _catalog->row(id);
            viewer->setWindowTitle(surface && row ? tr("Surface %1").arg(display_name(*row)) : tr("Surface"));
        }
    }
    if (preserveEditing) {
        emit surfaceActivatedPreserveEditing(idQString, surface);
    } else {
        emit surfaceActivated(idQString, surface);
    }
    if (isCurrentOnlyFilterEnabled()) {
        applyFilters();
    }
}

// A catalog surface still remote: fetch its geometry in the background, then
// open it if it is still the one asked for.
void SurfacePanelController::fetchThenActivate(const std::string& id, bool preserveEditing)
{
    _fetchingSurfaceId = id;
    emit statusMessageRequested(tr("Fetching surface %1 from the catalog...").arg(QString::fromStdString(id)), 0);
    const auto* row = _catalog->row(id);
    if (!row) {
        _fetchingSurfaceId.clear();
        return;
    }
    vc3d::store::background<bool>(this,
        [session = _catalog->session(), attachment = row->source, id]() {
            session->ensureGeometry(attachment, id);
            return true;
        },
        [this, id, preserveEditing](vc3d::store::Outcome<bool> result) {
            if (_fetchingSurfaceId != id) return;
            _fetchingSurfaceId.clear();
            if (!result.error.isEmpty()) {
                emit statusMessageRequested(tr("Cannot fetch surface %1: %2").arg(QString::fromStdString(id), result.error), 8000);
                return;
            }
            emit statusMessageRequested(tr("Fetched surface %1").arg(QString::fromStdString(id)), 3000);
            activate(id, preserveEditing);
        });
}

void SurfacePanelController::handleTreeSelectionChanged()
{
    if (!_ui.treeWidget) {
        return;
    }

    const QList<QTreeWidgetItem*> selectedItems = _ui.treeWidget->selectedItems();

    if (_selectionLocked) {
        QStringList currentIds;
        currentIds.reserve(selectedItems.size());
        for (auto* item : selectedItems) {
            const QString id = item ? item->data(SURFACE_ID_COLUMN, Qt::UserRole).toString() : QString();
            if (!id.isEmpty()) {
                currentIds.append(id);
            }
        }

        QStringList normalizedCurrent = currentIds;
        QStringList normalizedLocked = _lockedSelectionIds;
        std::sort(normalizedCurrent.begin(), normalizedCurrent.end());
        std::sort(normalizedLocked.begin(), normalizedLocked.end());

        if (normalizedCurrent != normalizedLocked) {
            const QSignalBlocker blocker{_ui.treeWidget};
            _ui.treeWidget->clearSelection();
            for (const QString& id : _lockedSelectionIds) {
                if (auto* item = itemFor(id.toStdString())) {
                    item->setSelected(true);
                }
            }
            if (!_selectionLockNotified) {
                _selectionLockNotified = true;
                constexpr int kLockNoticeMs = 3000;
                emit statusMessageRequested(tr("Surface selection is locked while growth runs."), kLockNoticeMs);
            }
        }
        return;
    }

    if (selectedItems.isEmpty()) {
        _currentSurfaceId.clear();
        _opened.reset();
        resetTagUi();
        if (_segmentationViewerProvider) {
            if (auto* viewer = _segmentationViewerProvider()) {
                viewer->setWindowTitle(tr("Surface"));
            }
        }
        emit surfaceSelectionCleared();
        return;
    }

    const auto id = selectedItems.first()->data(SURFACE_ID_COLUMN, Qt::UserRole).toString().toStdString();
    activate(id, false);
}

void SurfacePanelController::showContextMenu(const QPoint& pos)
{
    if (!_ui.treeWidget || !_catalog) {
        return;
    }

    QTreeWidgetItem* item = _ui.treeWidget->itemAt(pos);
    if (!item) {
        return;
    }

    QStringList selectedSegmentIds;
    for (auto* selectedItem : _ui.treeWidget->selectedItems()) {
        selectedSegmentIds << selectedItem->data(SURFACE_ID_COLUMN, Qt::UserRole).toString();
    }
    const QString segmentId = selectedSegmentIds.isEmpty() ?
        item->data(SURFACE_ID_COLUMN, Qt::UserRole).toString() :
        selectedSegmentIds.front();
    const std::string segmentIdStd = segmentId.toStdString();
    QStringList targets = selectedSegmentIds;
    if (targets.isEmpty()) {
        targets << segmentId;
    }

    // These tool actions target the user's store. Interactive editing also
    // supports sample stores through their views; external tools read stored
    // coordinates, so they take user surfaces stored in this volume's voxels.
    const bool storeWritable = _catalog->attachment() && _catalog->session()->writable(*_catalog->attachment());
    const bool writable = storeWritable &&
        std::none_of(targets.begin(), targets.end(), [this](const QString& id) {
            const auto* row = _catalog->row(id.toStdString());
            return row && row->catalog;
        });
    auto bound = [this](const QString& id) {
        const auto* row = _catalog->row(id.toStdString());
        return row && row->bound();
    };
    // Patch edits the parent in place: it takes surfaces stored in the volume's voxels.
    auto toolInput = [this](const QString& id) {
        const auto* row = _catalog->row(id.toStdString());
        return row && row->toolInput();
    };
    const bool isBound = bound(segmentId);
    const bool allBound = std::all_of(targets.begin(), targets.end(), bound);
    const bool usable = isBound && writable;
    const bool allUsable = allBound && writable;
    const bool allToolUsable = std::all_of(targets.begin(), targets.end(), toolInput) && writable;

    QMenu contextMenu(tr("Context Menu"), _ui.treeWidget);

    if (selectedSegmentIds.size() <= 1) {
        QAction* renameAction = contextMenu.addAction(tr("Rename Surface"));
        renameAction->setEnabled(writable);
        connect(renameAction, &QAction::triggered, this, [this, segmentIdStd]() { renameSurface(segmentIdStd); });
        QAction* historyAction = contextMenu.addAction(tr("History..."));
        connect(historyAction, &QAction::triggered, this, [this, segmentIdStd]() { showHistory(segmentIdStd); });
        const auto* row = _catalog->row(segmentIdStd);
        if (row && row->imported()) {
            // Its first revision in your store is the catalog version.
            QAction* resetAction = contextMenu.addAction(tr("Reset to Open Data Version"));
            resetAction->setEnabled(storeWritable);
            connect(resetAction, &QAction::triggered, this, [this, segmentIdStd]() {
                QString error;
                if (!resetToCatalogVersion(segmentIdStd, &error)) {
                    QMessageBox::critical(_ui.treeWidget, tr("Reset Failed"), error);
                }
            });
        }
        QAction* focusAction = contextMenu.addAction(tr("Focus"));
        focusAction->setEnabled(isBound);
        connect(focusAction, &QAction::triggered, this, [this, segmentId]() {
            emit focusSurfaceRequested(segmentId);
        });
    }

    QAction* copyUuidAction = contextMenu.addAction(tr("Copy UUID"));
    connect(copyUuidAction, &QAction::triggered, this, [this, targets]() {
        QApplication::clipboard()->setText(targets.join(QLatin1Char('\n')));
        emit statusMessageRequested(tr("Copied %n surface UUID(s) to the clipboard", nullptr, int(targets.size())), 3000);
    });

    if (!isBound) {
        QAction* bindAction = contextMenu.addAction(tr("Bind to Current Volume..."));
        bindAction->setEnabled(writable && _catalog->volume());
        connect(bindAction, &QAction::triggered, this, [this, segmentIdStd]() { bindSurface(segmentIdStd); });
    }

    QAction* addToSpiralAction = contextMenu.addAction(tr("Add to current spiral fit"));
    addToSpiralAction->setEnabled(_spiralFitAvailable && isBound);
    addToSpiralAction->setToolTip(_spiralFitAvailable
        ? tr("Upload this surface to the active Spiral session; it is used on the next run")
        : tr("No Spiral session is active on the connected service"));
    connect(addToSpiralAction, &QAction::triggered, this, [this, segmentId]() {
        emit addSurfaceToSpiralFitRequested(segmentId);
    });

    contextMenu.addSeparator();

    QAction* growSegmentAction = contextMenu.addAction(tr("Run Trace"));
    growSegmentAction->setEnabled(usable);
    connect(growSegmentAction, &QAction::triggered, this, [this, segmentId]() {
        emit growSegmentRequested(segmentId);
    });

    QMenu* copySurfaceMenu = contextMenu.addMenu(tr("Copy Surface"));
    copySurfaceMenu->setEnabled(storeWritable);
    if (selectedSegmentIds.size() <= 1) {
        // Catalog surfaces are edited in place (in your store), never copied.
        QAction* copySurfaceAction = copySurfaceMenu->addAction(tr("Copy Surface..."));
        copySurfaceAction->setEnabled(writable);
        connect(copySurfaceAction, &QAction::triggered, this, [this, segmentIdStd]() { copySurface(segmentIdStd); });
        copySurfaceMenu->addSeparator();
    }
    QAction* copyOutAction = copySurfaceMenu->addAction(tr("Out"));
    copyOutAction->setEnabled(usable);
    connect(copyOutAction, &QAction::triggered, this, [this, segmentId]() {
        emit neighborCopyRequested(segmentId, true);
    });
    QAction* copyInAction = copySurfaceMenu->addAction(tr("In"));
    copyInAction->setEnabled(usable);
    connect(copyInAction, &QAction::triggered, this, [this, segmentId]() {
        emit neighborCopyRequested(segmentId, false);
    });
    QAction* resumeLocalAction = contextMenu.addAction(tr("Reoptimize Surface"));
    resumeLocalAction->setEnabled(usable);
    connect(resumeLocalAction, &QAction::triggered, this, [this, segmentId]() {
        emit resumeLocalGrowPatchRequested(segmentId);
    });

    contextMenu.addSeparator();

    auto addToolAction = [&](const QString& text, bool enabled, auto signal) {
        QAction* action = contextMenu.addAction(text);
        action->setEnabled(enabled);
        connect(action, &QAction::triggered, this, [this, segmentId, signal]() { emit (this->*signal)(segmentId); });
        return action;
    };
    addToolAction(tr("Render segment"), usable, &SurfacePanelController::renderSegmentRequested);
    addToolAction(tr("Convert to OBJ"), usable, &SurfacePanelController::convertToObjRequested);
    addToolAction(tr("Lasagna Vis as OBJ"), usable, &SurfacePanelController::visLasagnaObjRequested);
    addToolAction(tr("Crop bounds to valid region"), usable, &SurfacePanelController::cropBoundsRequested);

    QMenu* flipMenu = contextMenu.addMenu(tr("Flip Surface"));
    flipMenu->setEnabled(usable);
    QAction* flipUAction = flipMenu->addAction(tr("V Flip"));
    connect(flipUAction, &QAction::triggered, this, [this, segmentId]() {
        emit flipURequested(segmentId);
    });
    QAction* flipVAction = flipMenu->addAction(tr("H Flip"));
    connect(flipVAction, &QAction::triggered, this, [this, segmentId]() {
        emit flipVRequested(segmentId);
    });
    QAction* flipNormalsAction = flipMenu->addAction(tr("Normals (H Flip)"));
    connect(flipNormalsAction, &QAction::triggered, this, [this, segmentId]() {
        emit flipVRequested(segmentId);
    });

    addToolAction(tr("Rotate Surface 90° CW"), usable, &SurfacePanelController::rotateSurfaceRequested);
    addToolAction(tr("Refine (Alpha-comp)"), usable, &SurfacePanelController::alphaCompRefineRequested);
    addToolAction(tr("SLIM-flatten"), usable, &SurfacePanelController::slimFlattenRequested);
    addToolAction(tr("Straighten (vc_straighten)"), usable, &SurfacePanelController::straightenRequested);

    contextMenu.addSeparator();

    QAction* rasterizeAction = contextMenu.addAction(tr("Rasterize"));
    rasterizeAction->setEnabled(allUsable);
    connect(rasterizeAction, &QAction::triggered, this, [this, targets]() {
        emit rasterizeSegmentsRequested(targets);
    });

    // Merge only makes sense for >=2 surfaces; gate on selection so it
    // doesn't appear when the user right-clicks a single surface.
    if (selectedSegmentIds.size() >= 2 && allUsable) {
        QAction* mergeAction = contextMenu.addAction(tr("Merge surfaces..."));
        connect(mergeAction, &QAction::triggered, this, [this, selectedSegmentIds]() {
            emit mergeSurfacesRequested(selectedSegmentIds);
        });
    }
    // Patch is exactly-2-input; gate to exactly 2 selected surfaces so the
    // dialog opens with both pickers pre-filled.
    if (selectedSegmentIds.size() == 2 && allToolUsable) {
        QAction* patchAction = contextMenu.addAction(tr("Patch surface..."));
        connect(patchAction, &QAction::triggered, this, [this, selectedSegmentIds]() {
            emit mergePatchRequested(selectedSegmentIds);
        });
    }

    QAction* addIgnoreLabelAction = contextMenu.addAction(tr("Add ignore label"));
    addIgnoreLabelAction->setEnabled(usable);
    connect(addIgnoreLabelAction, &QAction::triggered, this, [this]() {
        emit addIgnoreLabelRequested();
    });

    contextMenu.addSeparator();

    const QString deleteText = targets.size() > 1 ?
        tr("Delete %1 Surfaces").arg(targets.size()) :
        tr("Delete Surface");
    QAction* deleteAction = contextMenu.addAction(deleteText);
    deleteAction->setIcon(_ui.treeWidget->style()->standardIcon(QStyle::SP_TrashIcon));
    deleteAction->setEnabled(writable);
    connect(deleteAction, &QAction::triggered, this, [this, targets]() {
        handleRetireSurfaces(targets);
    });

    contextMenu.addSeparator();

    QAction* highlightAction = contextMenu.addAction(tr("Highlight in slice views"));
    highlightAction->setCheckable(true);
    highlightAction->setChecked(_highlightedSurfaceIds.count(segmentIdStd) > 0);
    connect(highlightAction, &QAction::toggled, this, [this, segmentIdStd](bool checked) {
        applyHighlightSelection(segmentIdStd, checked);
    });

    contextMenu.exec(_ui.treeWidget->mapToGlobal(pos));
}

void SurfacePanelController::renameSurface(const std::string& id)
{
    const auto* row = _catalog->row(id);
    if (!row) {
        return;
    }
    bool accepted = false;
    const QString name = QInputDialog::getText(_ui.treeWidget, tr("Rename Surface"), tr("Name:"),
                                               QLineEdit::Normal, display_name(*row), &accepted).trimmed();
    if (!accepted || name.isEmpty() || name.toStdString() == row->record.name) {
        return;
    }
    try {
        _catalog->session()->editRecords(*_catalog->attachment(), {id},
                                         [&](vc::objects::ObjectRecord& record) { record.name = name.toStdString(); });
    } catch (const std::exception& error) {
        QMessageBox::critical(_ui.treeWidget, tr("Rename Failed"), QString::fromUtf8(error.what()));
    }
}

void SurfacePanelController::copySurface(const std::string& id)
{
    const auto* row = _catalog->row(id);
    if (!row) {
        return;
    }
    bool accepted = false;
    const QString name = QInputDialog::getText(_ui.treeWidget, tr("Copy Surface"), tr("Name of the copy:"),
                                               QLineEdit::Normal, display_name(*row) + tr("_copy"), &accepted).trimmed();
    if (!accepted || name.isEmpty()) {
        return;
    }
    try {
        const auto copied =
            _catalog->session()->copyObject(*_catalog->attachment(), *_catalog->attachment(), id, name.toStdString());
        emit statusMessageRequested(tr("Copied surface to %1").arg(QString::fromStdString(copied)), 5000);
    } catch (const std::exception& error) {
        QMessageBox::critical(_ui.treeWidget, tr("Copy Failed"), QString::fromUtf8(error.what()));
    }
}

void SurfacePanelController::showHistory(const std::string& id)
{
    const auto* row = _catalog->row(id);
    if (!row) {
        return;
    }
    const auto attachment = row->source;
    std::vector<vc::objects::ObjectRevision> history;
    try {
        _catalog->session()->read(attachment, [&](const auto& db, const auto& txn, const auto&) {
            history = db.objectHistory(txn, id);
        });
    } catch (const std::exception& error) {
        QMessageBox::critical(_ui.treeWidget, tr("Surface History"), QString::fromUtf8(error.what()));
        return;
    }

    QDialog dialog(_ui.treeWidget);
    dialog.setWindowTitle(tr("History of %1").arg(display_name(*row)));
    auto* layout = new QVBoxLayout(&dialog);
    auto* list = new QTreeWidget(&dialog);
    list->setHeaderLabels({tr("Published"), tr("Author"), tr("Name"), tr("Geometry")});
    list->setRootIsDecorated(false);
    std::optional<std::string> previousGeometry;
    for (const auto& revision : history) {
        if (revision.deleted) {
            continue;
        }
        const auto published = revision.publishedAtUs
            ? QDateTime::fromMSecsSinceEpoch(*revision.publishedAtUs / 1000).toString(Qt::ISODate)
            : tr("(not published)");
        auto* item = new QTreeWidgetItem(list, {published, QString::fromStdString(revision.record.changedBy),
                                                QString::fromStdString(revision.record.name),
                                                revision.geometryRevision == previousGeometry ? QString() : tr("changed")});
        item->setData(0, Qt::UserRole, QString::fromStdString(revision.uuid));
        previousGeometry = revision.geometryRevision;
    }
    if (list->topLevelItemCount() > 0) {
        list->setCurrentItem(list->topLevelItem(list->topLevelItemCount() - 1));
    }
    layout->addWidget(list);
    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Close, &dialog);
    auto* view = buttons->addButton(tr("View"), QDialogButtonBox::ActionRole);
    auto* restore = buttons->addButton(tr("Restore"), QDialogButtonBox::ActionRole);
    restore->setEnabled(_catalog->session()->writable(attachment));
    auto selected = [list]() {
        const auto* item = list->currentItem();
        return item ? item->data(0, Qt::UserRole).toString().toStdString() : std::string();
    };
    connect(view, &QPushButton::clicked, &dialog, [&]() {
        const auto revision = selected();
        if (revision.empty() || !selectSurfaceById(id)) {
            return;
        }
        try {
            activate(id, false, _catalog->open(id, revision));
        } catch (const std::exception& error) {
            QMessageBox::critical(&dialog, tr("Surface History"), QString::fromUtf8(error.what()));
        }
    });
    connect(restore, &QPushButton::clicked, &dialog, [&]() {
        const auto revision = selected();
        if (revision.empty()) {
            return;
        }
        QString error;
        if (restoreRevision(id, revision, &error)) {
            dialog.accept();
        } else {
            QMessageBox::critical(&dialog, tr("Restore Failed"), error);
        }
    });
    connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    layout->addWidget(buttons);
    dialog.resize(640, 360);
    dialog.exec();
}

bool SurfacePanelController::restoreRevision(const std::string& id, const std::string& revision, QString* errorMessage)
{
    try {
        const auto* row = _catalog->row(id);
        if (!row || !_catalog->session()->writable(row->source)) {
            throw std::runtime_error("The surface requires a writable object store to restore a revision");
        }
        const auto attachment = row->source;
        std::optional<std::string> head;
        _catalog->session()->read(attachment, [&](const auto& db, const auto& txn, const auto&) {
            if (const auto current = db.currentObjectRevision(txn, id)) head = current->uuid;
        });
        if (!head) throw std::runtime_error("The surface is no longer in the object store");
        if (*head == revision) return true;
        _catalog->session()->editCurrent(attachment, {id}, [&](auto& db, auto& txn, const auto& uuid) {
            db.restoreObject(txn, uuid, revision);
        });
    } catch (const std::exception& error) {
        if (errorMessage) *errorMessage = QString::fromUtf8(error.what());
        return false;
    }
    return true;
}

bool SurfacePanelController::resetToCatalogVersion(const std::string& id, QString* errorMessage)
{
    const auto* row = _catalog->row(id);
    if (!row || !row->imported()) {
        if (errorMessage) *errorMessage = tr("The surface has no catalog version");
        return false;
    }
    try {
        _catalog->session()->restoreImported(row->source, id);
    } catch (const std::exception& error) {
        if (errorMessage) *errorMessage = QString::fromUtf8(error.what());
        return false;
    }
    return true;
}

bool SurfacePanelController::prepareSurfaceEditing(const std::string& id, QString* errorMessage)
{
    auto fail = [&](const QString& message) {
        if (errorMessage) *errorMessage = message;
        return false;
    };
    if (!_opened || _opened->surface->id != id || !_opened->current) {
        return fail(tr("Open the surface's current version to edit it"));
    }
    if (!_opened->editable || !_catalog->session()->writable(_opened->pin.attachment)) {
        return fail(tr("The surface's object store is attached read-only"));
    }
    return true;
}

void SurfacePanelController::bindSurface(const std::string& id)
{
    const auto volume = _catalog->volume();
    const auto* row = _catalog->row(id);
    if (!volume || !row || !_catalog->attachment()) {
        return;
    }
    const auto scale = vc3d::promptStoredUnitsPerVoxel(_ui.treeWidget, tr("Bind Surface Coordinates"), tr("surface"), *volume,
                                                       row_meta(*row).value("coordinates", nlohmann::json::object()));
    if (!scale) {
        return;
    }
    try {
        _catalog->session()->editCurrent(*_catalog->attachment(), {id}, [&](auto& db, auto& txn, const auto& uuid) {
            db.bindGeometryCoordinates(txn, uuid, volume->id(), *scale);
        });
    } catch (const std::exception& error) {
        QMessageBox::critical(_ui.treeWidget, tr("Bind Failed"), QString::fromUtf8(error.what()));
    }
}

void SurfacePanelController::handleRetireSurfaces(const QStringList& surfaceIds)
{
    if (surfaceIds.isEmpty()) {
        return;
    }

    const QString message = surfaceIds.size() == 1
        ? tr("Delete surface '%1'?\n\nIts revisions stay in the object store history.").arg(
              _catalog->row(surfaceIds.first().toStdString()) ? display_name(*_catalog->row(surfaceIds.first().toStdString()))
                                                               : surfaceIds.first())
        : tr("Delete %1 surfaces?\n\nTheir revisions stay in the object store history.").arg(surfaceIds.size());
    if (QMessageBox::question(_ui.treeWidget, tr("Confirm Deletion"), message,
                              QMessageBox::Yes | QMessageBox::No, QMessageBox::No) != QMessageBox::Yes) {
        return;
    }

    QString err;
    if (retireSurfaces(surfaceIds, &err)) {
        emit statusMessageRequested(tr("Deleted %n surface(s)", nullptr, int(surfaceIds.size())), 5000);
    } else {
        QMessageBox::critical(_ui.treeWidget, tr("Deletion Failed"), err);
    }
}

bool SurfacePanelController::retireSurfaces(const QStringList& surfaceIds, QString* err)
{
    if (surfaceIds.isEmpty() || !_catalog) {
        if (err) *err = tr("no surfaces specified");
        return false;
    }
    std::vector<std::string> uuids;
    for (const auto& id : surfaceIds) {
        uuids.push_back(id.toStdString());
    }
    if (!_catalog->attachment()) {
        if (err) *err = tr("Attach an object store first");
        return false;
    }
    try {
        _catalog->session()->editCurrent(*_catalog->attachment(), uuids,
                                         [](auto& db, auto& txn, const auto& uuid) { db.retireObject(txn, uuid); });
        return true;
    } catch (const std::exception& error) {
        if (err) *err = QString::fromUtf8(error.what());
        return false;
    }
}

void SurfacePanelController::configureFilters(const FilterUiRefs& filters, VCCollection* pointCollection)
{
    _filters = filters;
    _pointCollection = pointCollection;

    if (_filters.dropdown) {
        _filters.dropdown->clearOptions();
        _filters.dropdown->setText(tr("Filters"));
        _filters.dropdown->setPopupMode(QToolButton::DelayedPopup);
        _filters.dropdown->setMenu(nullptr);
        _filters.dropdown->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
        _filters.dropdown->setMinimumWidth(
            _filters.dropdown->fontMetrics().horizontalAdvance(_filters.dropdown->text()) + 32);
        connect(_filters.dropdown, &QToolButton::clicked,
                this, &SurfacePanelController::showFilterDialog);
    }

    if (_filters.focusPointDistance) {
        _filters.focusPointDistance->setRange(0.0, 1000000.0);
        _filters.focusPointDistance->setDecimals(2);
        _filters.focusPointDistance->setSingleStep(1.0);
        _filters.focusPointDistance->setSuffix(tr(" vox"));
        if (_filters.focusPointDistance->value() <= 0.0) {
            _filters.focusPointDistance->setValue(kFocusPointFilterRadius);
        }
        _filters.focusPointDistance->setToolTip(tr("Maximum distance from the focus point for the Focus Point filter."));
    }

    if (_filters.zLowerBound) {
        _filters.zLowerBound->setRange(kZRangeFilterLowerDefault, kZRangeFilterUpperDefault);
        _filters.zLowerBound->setDecimals(2);
        _filters.zLowerBound->setSingleStep(1.0);
        _filters.zLowerBound->setPrefix(tr("Lower "));
        _filters.zLowerBound->setSuffix(tr(" z"));
        _filters.zLowerBound->setValue(kZRangeFilterLowerDefault);
        _filters.zLowerBound->setToolTip(tr("Lower Z bound for visible surfaces."));
    }

    if (_filters.zUpperBound) {
        _filters.zUpperBound->setRange(kZRangeFilterLowerDefault, kZRangeFilterUpperDefault);
        _filters.zUpperBound->setDecimals(2);
        _filters.zUpperBound->setSingleStep(1.0);
        _filters.zUpperBound->setPrefix(tr("Upper "));
        _filters.zUpperBound->setSuffix(tr(" z"));
        _filters.zUpperBound->setValue(kZRangeFilterUpperDefault);
        _filters.zUpperBound->setToolTip(tr("Upper Z bound for visible surfaces."));
    }

    const auto addFilterOption = [this](QCheckBox*& target, const QString& text, const QString& objectName) {
        if (!target) {
            target = new QCheckBox(text, _ui.treeWidget);
            if (!objectName.isEmpty()) {
                target->setObjectName(objectName);
            }
        } else {
            target->setText(text);
        }
        target->hide();
    };

    addFilterOption(_filters.focusPoints, tr("Focus Point"), QStringLiteral("chkFilterFocusPoints"));
    addFilterOption(_filters.unreviewed, tr("Unreviewed"), QStringLiteral("chkFilterUnreviewed"));
    addFilterOption(_filters.hideUnapproved, tr("Hide Unapproved"), QStringLiteral("chkFilterHideUnapproved"));
    addFilterOption(_filters.noExpansion, tr("Hide Expansion"), QStringLiteral("chkFilterNoExpansion"));
    addFilterOption(_filters.noDefective, tr("Hide Defective"), QStringLiteral("chkFilterNoDefective"));
    addFilterOption(_filters.partialReview, tr("Hide Partial Review"), QStringLiteral("chkFilterPartialReview"));
    addFilterOption(_filters.showPartialReview, tr("Show Partial Review"), QStringLiteral("chkFilterShowPartialReview"));
    addFilterOption(_filters.inspectOnly, tr("Inspect Only"), QStringLiteral("chkFilterInspectOnly"));

    if (_filters.currentOnly) {
        _filters.currentOnly->setText(tr("Current Segment Only"));
        _filters.currentOnly->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
        _filters.currentOnly->setMinimumWidth(_filters.currentOnly->sizeHint().width());
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        _filters.currentOnly->setChecked(settings.value(kCurrentOnlyFilterSettingsKey, false).toBool());
        _filters.currentOnly->show();
    }

    if (_filters.focusPointDistance && _filters.focusPoints) {
        _filters.focusPointDistance->setEnabled(_filters.focusPoints->isChecked());
    }

    _recordTagFilter = new QComboBox(_ui.treeWidget);
    _recordTagFilter->setObjectName(QStringLiteral("cmbFilterRecordTags"));
    _recordTagFilter->setToolTip(tr("Show only surfaces with any of the checked tags."));
    _recordTagFilter->hide();

    buildFilterDialog();
    connectFilterSignals();
    rebuildPointSetFilterModel();
    rebuildRecordTagFilterModel();
    applyFilters();
    updateFilterSummary();
}

void SurfacePanelController::configureTags(const TagUiRefs& tags)
{
    _tags = tags;
    if (_tags.dropdown) {
        _tags.dropdown->setText(tr("Tags"));
        _tags.dropdown->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
    }
    connectTagSignals();
    resetTagUi();
    updateTagSummary();
}

void SurfacePanelController::refreshPointSetFilterOptions()
{
    rebuildPointSetFilterModel();
    refreshStoreFilters();
    applyFilters();
}

void SurfacePanelController::applyFilters()
{
    if (_configuringFilters) {
        return;
    }
    applyFiltersInternal();
    updateFilterSummary();
}

bool SurfacePanelController::selectSurfaceById(const std::string& surfaceId)
{
    auto* targetItem = itemFor(surfaceId);
    if (!targetItem) {
        return false;
    }

    {
        const QSignalBlocker blocker{_ui.treeWidget};
        _ui.treeWidget->clearSelection();
        targetItem->setSelected(true);
        _ui.treeWidget->scrollToItem(targetItem);
    }

    _currentSurfaceId = surfaceId;
    updateTagCheckboxStates(_catalog->row(surfaceId));
    if (isCurrentOnlyFilterEnabled()) {
        applyFilters();
    }

    if (_segmentationViewerProvider) {
        if (auto* viewer = _segmentationViewerProvider()) {
            viewer->setWindowTitle(tr("Surface %1").arg(targetItem->text(SURFACE_ID_COLUMN)));
        }
    }

    return true;
}

bool SurfacePanelController::activateSurfaceById(const std::string& surfaceId,
                                                 QString* errorMessage)
{
    auto fail = [errorMessage](const QString& msg) {
        if (errorMessage) {
            *errorMessage = msg;
        }
        return false;
    };

    // Selection lock (held during growth): refuse headlessly.
    if (_selectionLocked) {
        return fail(tr("surface selection is locked while growth runs"));
    }
    const QString idQString = QString::fromStdString(surfaceId);
    if (!_catalog || !_catalog->row(surfaceId)) {
        return fail(tr("unknown surface: %1").arg(idQString));
    }
    std::optional<SurfaceCatalog::Opened> opened;
    try {
        try {
            opened = _catalog->open(surfaceId);
        } catch (const SurfaceCatalog::NoGeometry&) {
            if (!_catalog->row(surfaceId)->catalog) throw;
            // Headless callers wait for a catalog surface to be fetched.
            _catalog->session()->ensureGeometry(_catalog->row(surfaceId)->source, surfaceId);
            opened = _catalog->open(surfaceId);
        }
    } catch (const std::exception& error) {
        return fail(tr("surface %1 could not be opened: %2").arg(idQString, QString::fromUtf8(error.what())));
    }
    // The blocked tree selection keeps the tree signal from activating twice.
    if (!selectSurfaceById(surfaceId)) {
        return fail(tr("surface %1 could not be selected").arg(idQString));
    }
    activate(surfaceId, false, std::move(opened));
    return _opened.has_value();
}

void SurfacePanelController::resetTagUi()
{
    auto resetBox = [](QCheckBox* box) {
        if (!box) {
            return;
        }
        const QSignalBlocker blocker{box};
        box->setCheckState(Qt::Unchecked);
        box->setEnabled(false);
    };

    resetBox(_tags.approved);
    resetBox(_tags.defective);
    resetBox(_tags.reviewed);
    resetBox(_tags.inspect);
    updateTagSummary();
}

bool SurfacePanelController::isCurrentOnlyFilterEnabled() const
{
    return _filters.currentOnly && _filters.currentOnly->isChecked();
}

bool SurfacePanelController::setTagChecked(Tag tag, bool checked)
{
    QCheckBox* target = nullptr;
    switch (tag) {
        case Tag::Approved: target = _tags.approved; break;
        case Tag::Defective: target = _tags.defective; break;
        case Tag::Reviewed: target = _tags.reviewed; break;
        case Tag::Inspect: target = _tags.inspect; break;
    }

    if (!target || !target->isEnabled()) {
        return false;
    }

    const Qt::CheckState desiredState = checked ? Qt::Checked : Qt::Unchecked;
    if (target->checkState() == desiredState) {
        onTagCheckboxToggled();
    } else {
        target->setCheckState(desiredState);
    }
    return true;
}

void SurfacePanelController::setSelectionLocked(bool locked)
{
    if (_selectionLocked == locked) {
        return;
    }

    _selectionLocked = locked;
    _lockedSelectionIds.clear();
    _selectionLockNotified = false;

    if (!_ui.treeWidget) {
        return;
    }

    _ui.treeWidget->setDisabled(locked);

    if (locked) {
        for (auto* item : _ui.treeWidget->selectedItems()) {
            const QString id = item ? item->data(SURFACE_ID_COLUMN, Qt::UserRole).toString() : QString();
            if (!id.isEmpty()) {
                _lockedSelectionIds.append(id);
            }
        }
    }
}

void SurfacePanelController::applyBindingStyle(SurfaceTreeWidgetItem* item)
{
    if (!item) {
        return;
    }

    const bool unbound = item->data(SURFACE_ID_COLUMN, kUnboundRole).toBool();
    const QBrush foreground = unbound ? QBrush(QColor(Qt::gray)) : QBrush();
    for (int column = SURFACE_ID_COLUMN; column <= TIMESTAMP_COLUMN; ++column) {
        item->setForeground(column, foreground);
        if (unbound) {
            item->setToolTip(column, tr("Not bound to the current volume"));
        } else if (column == SURFACE_ID_COLUMN || column == SURFACE_LONG_ID_COLUMN) {
            item->setToolTip(column, item->text(column));
        } else {
            item->setToolTip(column, QString());
        }
    }
}

void SurfacePanelController::setupSurfaceColumnMenu()
{
    if (!_ui.treeWidget || !_ui.treeWidget->header()) {
        return;
    }

    auto* header = _ui.treeWidget->header();
    header->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(header, &QWidget::customContextMenuRequested,
            this, &SurfacePanelController::showSurfaceColumnMenu);
}

void SurfacePanelController::restoreSurfaceColumnVisibility()
{
    if (!_ui.treeWidget) {
        return;
    }

    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    for (int column = 0; column < _ui.treeWidget->columnCount(); ++column) {
        if (column == SURFACE_ID_COLUMN) {
            _ui.treeWidget->setColumnHidden(column, false);
            continue;
        }
        const QString key = surface_column_settings_key(column);
        const bool visible = settings.contains(key)
            ? settings.value(key).toBool()
            : surface_column_default_visible(column);
        _ui.treeWidget->setColumnHidden(column, !visible);
    }
}

void SurfacePanelController::showSurfaceColumnMenu(const QPoint& pos)
{
    if (!_ui.treeWidget || !_ui.treeWidget->header()) {
        return;
    }

    QMenu menu(_ui.treeWidget);
    for (int column = 0; column < _ui.treeWidget->columnCount(); ++column) {
        QString label = _ui.treeWidget->headerItem()
            ? _ui.treeWidget->headerItem()->text(column)
            : QString();
        if (label.isEmpty()) {
            label = tr("Status");
        }

        QAction* action = menu.addAction(label);
        action->setCheckable(true);
        action->setChecked(!_ui.treeWidget->isColumnHidden(column));
        action->setEnabled(column != SURFACE_ID_COLUMN);
        connect(action, &QAction::toggled, this, [this, column](bool checked) {
            if (!_ui.treeWidget || column == SURFACE_ID_COLUMN) {
                return;
            }
            _ui.treeWidget->setColumnHidden(column, !checked);
            QSettings columnSettings(vc3d::settingsFilePath(), QSettings::IniFormat);
            columnSettings.setValue(surface_column_settings_key(column), checked);
        });
    }

    menu.exec(_ui.treeWidget->header()->mapToGlobal(pos));
}

void SurfacePanelController::buildFilterDialog()
{
    if (_filterDialog) {
        return;
    }

    _filterDialog = new QDialog(_ui.treeWidget);
    _filterDialog->setObjectName(QStringLiteral("surfaceFilterDialog"));
    _filterDialog->setWindowTitle(tr("Surface Filters"));
    _filterDialog->setModal(false);

    auto* mainLayout = new QVBoxLayout(_filterDialog);
    mainLayout->setContentsMargins(12, 12, 12, 12);
    mainLayout->setSpacing(10);

    auto* filterGrid = new QGridLayout();
    filterGrid->setHorizontalSpacing(12);
    filterGrid->setVerticalSpacing(4);
    const QList<QCheckBox*> filterBoxes = {
        _filters.focusPoints,
        _filters.unreviewed,
        _filters.hideUnapproved,
        _filters.noExpansion,
        _filters.noDefective,
        _filters.partialReview,
        _filters.showPartialReview,
        _filters.inspectOnly,
    };
    int row = 0;
    int col = 0;
    for (auto* box : filterBoxes) {
        if (!box) {
            continue;
        }
        box->setParent(_filterDialog);
        box->show();
        filterGrid->addWidget(box, row, col);
        col = (col + 1) % 2;
        if (col == 0) {
            ++row;
        }
    }
    mainLayout->addLayout(filterGrid);

    auto* form = new QFormLayout();
    form->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    if (_filters.surfaceIdFilter) {
        _filters.surfaceIdFilter->setParent(_filterDialog);
        _filters.surfaceIdFilter->show();
        form->addRow(tr("Name or UUID"), _filters.surfaceIdFilter);
    }
    if (_recordTagFilter) {
        _recordTagFilter->setParent(_filterDialog);
        _recordTagFilter->show();
        form->addRow(tr("Tags"), _recordTagFilter);
    }
    if (_filters.focusPointDistance) {
        _filters.focusPointDistance->setParent(_filterDialog);
        _filters.focusPointDistance->show();
        form->addRow(tr("Focus radius"), _filters.focusPointDistance);
    }
    if (_filters.zLowerBound && _filters.zUpperBound) {
        auto* zRangeWidget = new QWidget(_filterDialog);
        auto* zRangeLayout = new QHBoxLayout(zRangeWidget);
        zRangeLayout->setContentsMargins(0, 0, 0, 0);
        zRangeLayout->setSpacing(4);
        _filters.zLowerBound->setParent(zRangeWidget);
        _filters.zUpperBound->setParent(zRangeWidget);
        _filters.zLowerBound->show();
        _filters.zUpperBound->show();
        zRangeLayout->addWidget(_filters.zLowerBound);
        zRangeLayout->addWidget(_filters.zUpperBound);
        form->addRow(tr("Z range"), zRangeWidget);
    }
    mainLayout->addLayout(form);

    if (_filters.pointSet || _filters.pointSetMode || _filters.pointSetAll || _filters.pointSetNone) {
        auto* pointSetLayout = new QHBoxLayout();
        pointSetLayout->setSpacing(4);
        pointSetLayout->addWidget(new QLabel(tr("Point sets"), _filterDialog));
        for (QWidget* widget : {static_cast<QWidget*>(_filters.pointSet), static_cast<QWidget*>(_filters.pointSetMode),
                                static_cast<QWidget*>(_filters.pointSetAll), static_cast<QWidget*>(_filters.pointSetNone)}) {
            if (!widget) {
                continue;
            }
            widget->setParent(_filterDialog);
            widget->show();
            pointSetLayout->addWidget(widget, widget == _filters.pointSet ? 1 : 0);
        }
        mainLayout->addLayout(pointSetLayout);
    }

    auto* closeButton = new QPushButton(tr("Close"), _filterDialog);
    connect(closeButton, &QPushButton::clicked, _filterDialog, &QDialog::hide);
    auto* buttonLayout = new QHBoxLayout();
    buttonLayout->addStretch();
    buttonLayout->addWidget(closeButton);
    mainLayout->addLayout(buttonLayout);
}

void SurfacePanelController::showFilterDialog()
{
    buildFilterDialog();
    if (!_filterDialog) {
        return;
    }

    _filterDialog->show();
    _filterDialog->raise();
    _filterDialog->activateWindow();
}

void SurfacePanelController::connectFilterSignals()
{
    auto connectToggle = [this](QCheckBox* box) {
        if (!box) {
            return;
        }
        connect(box, &QCheckBox::toggled, this, [this]() { applyFilters(); });
    };
    // Tag filters are a store selection (tagQuery).
    auto connectTagToggle = [this](QCheckBox* box) {
        if (!box) {
            return;
        }
        connect(box, &QCheckBox::toggled, this, [this]() { refreshStoreFilters(); applyFilters(); });
    };

    connectTagToggle(_filters.unreviewed);
    connectToggle(_filters.noExpansion);
    connectTagToggle(_filters.noDefective);
    connectTagToggle(_filters.hideUnapproved);
    connectTagToggle(_filters.inspectOnly);
    if (_filters.currentOnly) {
        connect(_filters.currentOnly, &QCheckBox::toggled, this, [this](bool checked) {
            QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
            settings.setValue(kCurrentOnlyFilterSettingsKey, checked);
            applyFilters();
        });
    }

    if (_filters.partialReview) {
        connect(_filters.partialReview, &QCheckBox::toggled, this, [this](bool checked) {
            if (checked && _filters.showPartialReview) {
                const QSignalBlocker blocker(_filters.showPartialReview);
                _filters.showPartialReview->setChecked(false);
            }
            refreshStoreFilters();
            applyFilters();
        });
    }

    if (_filters.focusPoints) {
        connect(_filters.focusPoints, &QCheckBox::toggled, this, [this]() {
            refreshStoreFilters();
            applyFilters();
        });
        if (_filters.focusPointDistance) {
            connect(_filters.focusPoints, &QCheckBox::toggled,
                    _filters.focusPointDistance, &QDoubleSpinBox::setEnabled);
        }
    }

    if (_filters.focusPointDistance) {
        connect(_filters.focusPointDistance,
                QOverload<double>::of(&QDoubleSpinBox::valueChanged),
                this,
                [this](double) { refreshStoreFilters(); applyFilters(); });
    }

    if (_state) {
        connect(_state, &CState::poiChanged, this, [this](const std::string& name, POI*) {
            if (name == "focus" && _filters.focusPoints && _filters.focusPoints->isChecked()) {
                refreshStoreFilters();
            }
        });
    }

    if (_filters.zLowerBound) {
        connect(_filters.zLowerBound,
                QOverload<double>::of(&QDoubleSpinBox::valueChanged),
                this,
                [this](double value) {
                    if (_filters.zUpperBound && value > _filters.zUpperBound->value()) {
                        const QSignalBlocker blocker(_filters.zUpperBound);
                        _filters.zUpperBound->setValue(value);
                    }
                    refreshStoreFilters();
                    applyFilters();
                });
    }

    if (_filters.zUpperBound) {
        connect(_filters.zUpperBound,
                QOverload<double>::of(&QDoubleSpinBox::valueChanged),
                this,
                [this](double value) {
                    if (_filters.zLowerBound && value < _filters.zLowerBound->value()) {
                        const QSignalBlocker blocker(_filters.zLowerBound);
                        _filters.zLowerBound->setValue(value);
                    }
                    refreshStoreFilters();
                    applyFilters();
                });
    }

    if (_filters.showPartialReview) {
        connect(_filters.showPartialReview, &QCheckBox::toggled, this, [this](bool checked) {
            if (checked && _filters.partialReview) {
                const QSignalBlocker blocker(_filters.partialReview);
                _filters.partialReview->setChecked(false);
            }
            refreshStoreFilters();
            applyFilters();
        });
    }

    if (_filters.pointSetMode) {
        connect(_filters.pointSetMode, &QComboBox::currentIndexChanged, this, [this]() { applyFilters(); });
    }

    auto setAllPointSets = [this](Qt::CheckState state) {
        auto* model = qobject_cast<QStandardItemModel*>(_filters.pointSet ? _filters.pointSet->model() : nullptr);
        if (!model) {
            return;
        }
        {
            QSignalBlocker blocker(model);
            for (int row = 0; row < model->rowCount(); ++row) {
                model->setData(model->index(row, 0), state, Qt::CheckStateRole);
            }
        }
        refreshStoreFilters();
        applyFilters();
    };
    if (_filters.pointSetAll) {
        connect(_filters.pointSetAll, &QPushButton::clicked, this, [setAllPointSets]() { setAllPointSets(Qt::Checked); });
    }
    if (_filters.pointSetNone) {
        connect(_filters.pointSetNone, &QPushButton::clicked, this, [setAllPointSets]() { setAllPointSets(Qt::Unchecked); });
    }

    if (_pointCollection) {
        auto pointSetsChanged = [this]() {
            refreshStoreFilters();
            applyFilters();
        };
        connect(_pointCollection, &VCCollection::collectionsAdded, this, [this](const std::vector<uint64_t>&) {
            rebuildPointSetFilterModel();
            applyFilters();
        });
        connect(_pointCollection, &VCCollection::collectionRemoved, this, [this](uint64_t) {
            rebuildPointSetFilterModel();
            refreshStoreFilters();
            applyFilters();
        });
        connect(_pointCollection, &VCCollection::collectionChanged, this, [this](uint64_t) {
            rebuildPointSetFilterModel();
            refreshStoreFilters();
            applyFilters();
        });
        connect(_pointCollection, &VCCollection::pointAdded, this, [pointSetsChanged](const ColPoint&) { pointSetsChanged(); });
        connect(_pointCollection, &VCCollection::pointChanged, this, [pointSetsChanged](const ColPoint&) { pointSetsChanged(); });
        connect(_pointCollection, &VCCollection::pointRemoved, this, [pointSetsChanged](vc::PointRef) { pointSetsChanged(); });
    }

    if (_filters.surfaceIdFilter) {
        connect(_filters.surfaceIdFilter, &QLineEdit::textChanged, this, [this]() { applyFilters(); });
    }
}

void SurfacePanelController::connectTagSignals()
{
    auto connectBox = [this](QCheckBox* box) {
        if (!box) {
            return;
        }
#if QT_VERSION < QT_VERSION_CHECK(6, 8, 0)
        connect(box, &QCheckBox::stateChanged, this, [this](int) { onTagCheckboxToggled(); });
#else
        connect(box, &QCheckBox::checkStateChanged, this, [this](Qt::CheckState) { onTagCheckboxToggled(); });
#endif
    };

    connectBox(_tags.approved);
    connectBox(_tags.defective);
    connectBox(_tags.reviewed);
    connectBox(_tags.inspect);
}

void SurfacePanelController::rebuildPointSetFilterModel()
{
    if (!_filters.pointSet) {
        return;
    }

    _configuringFilters = true;

    // Keep the checked sets across rebuilds.
    std::set<QString> checked;
    if (auto* existing = qobject_cast<QStandardItemModel*>(_filters.pointSet->model())) {
        for (int row = 0; row < existing->rowCount(); ++row) {
            if (existing->data(existing->index(row, 0), Qt::CheckStateRole) == Qt::Checked) {
                checked.insert(existing->data(existing->index(row, 0), Qt::DisplayRole).toString());
            }
        }
    }

    auto* model = new QStandardItemModel(_filters.pointSet);
    if (_pointSetModelConnection) {
        disconnect(_pointSetModelConnection);
        _pointSetModelConnection = QMetaObject::Connection{};
    }
    if (auto* existingModel = _filters.pointSet->model()) {
        existingModel->deleteLater();
    }
    _filters.pointSet->setModel(model);

    if (_pointCollection) {
        for (const auto& pair : _pointCollection->getAllCollections()) {
            const auto name = QString::fromStdString(pair.second.name);
            auto* item = new QStandardItem(name);
            item->setFlags(Qt::ItemIsUserCheckable | Qt::ItemIsEnabled);
            item->setData(checked.contains(name) ? Qt::Checked : Qt::Unchecked, Qt::CheckStateRole);
            model->appendRow(item);
        }
    }

    _pointSetModelConnection = connect(model, &QStandardItemModel::dataChanged,
        this,
        [this](const QModelIndex&, const QModelIndex&, const QVector<int>& roles) {
            if (roles.contains(Qt::CheckStateRole)) {
                refreshStoreFilters();
                applyFilters();
            }
        });

    _configuringFilters = false;
    updateFilterSummary();
}

void SurfacePanelController::rebuildRecordTagFilterModel()
{
    if (!_recordTagFilter || !_catalog) {
        return;
    }
    std::set<QString> checked;
    if (auto* existing = qobject_cast<QStandardItemModel*>(_recordTagFilter->model())) {
        for (int row = 0; row < existing->rowCount(); ++row) {
            if (existing->data(existing->index(row, 0), Qt::CheckStateRole) == Qt::Checked) {
                checked.insert(existing->data(existing->index(row, 0), Qt::DisplayRole).toString());
            }
        }
    }
    std::set<std::string> tags;
    for (const auto& source : _catalog->sources()) {
        try {
            _catalog->session()->read(source, [&](const auto& db, const auto& txn, const auto&) {
                for (auto& tag : db.objectTags(txn, std::string("surface"))) tags.insert(std::move(tag));
            });
        } catch (const std::exception&) {
            // A detached source lists no tags.
        }
    }
    _configuringFilters = true;
    auto* model = new QStandardItemModel(_recordTagFilter);
    for (const auto& tag : tags) {
        const auto name = QString::fromStdString(tag);
        auto* item = new QStandardItem(name);
        item->setFlags(Qt::ItemIsUserCheckable | Qt::ItemIsEnabled);
        item->setData(checked.contains(name) ? Qt::Checked : Qt::Unchecked, Qt::CheckStateRole);
        model->appendRow(item);
    }
    if (auto* existingModel = _recordTagFilter->model()) {
        existingModel->deleteLater();
    }
    _recordTagFilter->setModel(model);
    connect(model, &QStandardItemModel::dataChanged, this,
            [this](const QModelIndex&, const QModelIndex&, const QVector<int>& roles) {
                if (roles.contains(Qt::CheckStateRole)) {
                    refreshStoreFilters();
                    applyFilters();
                }
            });
    _configuringFilters = false;
}

namespace {
std::vector<QString> checkedNames(QComboBox* box)
{
    std::vector<QString> names;
    auto* model = qobject_cast<QStandardItemModel*>(box ? box->model() : nullptr);
    if (!model) {
        return names;
    }
    for (int row = 0; row < model->rowCount(); ++row) {
        if (model->data(model->index(row, 0), Qt::CheckStateRole) == Qt::Checked) {
            names.push_back(model->data(model->index(row, 0), Qt::DisplayRole).toString());
        }
    }
    return names;
}
}

SurfacePanelController::TagQuery SurfacePanelController::tagQuery() const
{
    auto checked = [](QCheckBox* box) { return box && box->isChecked(); };
    TagQuery query;
    std::vector<std::string> any;
    for (const auto& name : checkedNames(_recordTagFilter)) any.push_back(name.toStdString());
    if (!any.empty()) query.groups.push_back(std::move(any));
    if (checked(_filters.hideUnapproved)) query.groups.push_back({"approved"});
    if (checked(_filters.inspectOnly)) query.groups.push_back({"inspect"});
    if (checked(_filters.showPartialReview)) query.groups.push_back({"partial_review", "reviewed"});
    if (checked(_filters.unreviewed)) query.without.push_back("reviewed");
    if (checked(_filters.noDefective)) query.without.push_back("defective");
    if (checked(_filters.partialReview)) {
        query.without.push_back("partial_review");
        query.without.push_back("reviewed");
    }
    return query;
}

void SurfacePanelController::refreshStoreFilters()
{
    const auto generation = ++_storeFilterGeneration;
    if (!_catalog || _catalog->sources().empty() || !_catalog->volume()) {
        _focusAnswer.reset();
        _pointSetAnswer.reset();
        _zRangeAnswer.reset();
        _tagAnswer.reset();
        return;
    }
    const auto tags = tagQuery();
    // The Z-range filter: the surfaces whose bounds in the open volume reach the range.
    std::optional<std::array<double, 2>> zRange;
    if (z_range_filter_active(_filters.zLowerBound, _filters.zUpperBound)) {
        zRange = std::array<double, 2>{_filters.zLowerBound->value(), _filters.zUpperBound->value()};
    }
    // The questions: one point set for the focus point, one per checked collection.
    std::vector<std::vector<std::array<double, 3>>> focus, pointSets;
    double focusRadius = _filters.focusPointDistance ? _filters.focusPointDistance->value() : kFocusPointFilterRadius;
    if (POI* poi = (_filters.focusPoints && _filters.focusPoints->isChecked() && _state) ? _state->poi("focus") : nullptr) {
        focus.push_back({{poi->p[0], poi->p[1], poi->p[2]}});
    }
    if (_pointCollection) {
        for (const auto& name : checkedNames(_filters.pointSet)) {
            std::vector<std::array<double, 3>> points;
            for (const auto& point : _pointCollection->getPoints(name.toStdString())) {
                points.push_back({point.p[0], point.p[1], point.p[2]});
            }
            pointSets.push_back(std::move(points));
        }
    }
    struct Answers { ProximityAnswer focus, pointSets; std::optional<ZRangeAnswer> zRange; std::optional<TagAnswer> tags; };
    vc3d::store::background<Answers>(this,
        [session = _catalog->session(), sources = _catalog->sources(), volume = _catalog->volume()->id(), focus, pointSets,
         focusRadius, zRange, tags]() {
            auto answer = [&](const auto& sets, double tolerance) {
                ProximityAnswer result;
                result.pointSets = sets;
                result.tolerance = tolerance;
                vc::objects::SurfaceProximityQuery query;
                query.tolerance = tolerance;
                query.volume = volume;
                for (const auto& points : sets) {
                    std::set<std::string> any, all;
                    bool first = true;
                    for (const auto& hits : session->nearestSurfaces(sources, query, points)) {
                        std::set<std::string> near;
                        for (const auto& [attachment, hit] : hits) near.insert(hit.objectUuid);
                        any.insert(near.begin(), near.end());
                        if (first) all = near;
                        else std::erase_if(all, [&](const std::string& uuid) { return !near.contains(uuid); });
                        first = false;
                    }
                    result.any.push_back(std::move(any));
                    result.all.push_back(std::move(all));
                }
                return result;
            };
            Answers answers;
            answers.focus = answer(focus, focusRadius);
            answers.pointSets = answer(pointSets, kPointSetTolerance);
            if (zRange) {
                constexpr double kUnbounded = std::numeric_limits<double>::max();
                vc::objects::ObjectSelection selection;
                selection.kind = "surface";
                selection.volume = volume;
                selection.bbox = std::array<double, 6>{-kUnbounded, -kUnbounded, (*zRange)[0],
                                                       kUnbounded, kUnbounded, (*zRange)[1]};
                ZRangeAnswer reached{*zRange, {}};
                for (const auto& [attachment, record] : session->selectObjects(sources, selection))
                    reached.surfaces.insert(record.uuid);
                answers.zRange = std::move(reached);
            }
            if (!tags.empty()) {
                vc::objects::ObjectSelection selection;
                selection.kind = "surface";
                selection.tagGroups = tags.groups;
                selection.withoutTags = tags.without;
                TagAnswer tagged{tags, {}};
                for (const auto& [attachment, record] : session->selectObjects(sources, selection))
                    tagged.surfaces.insert(record.uuid);
                answers.tags = std::move(tagged);
            }
            return answers;
        },
        [this, generation](vc3d::store::Outcome<Answers> result) {
            if (generation != _storeFilterGeneration) return;
            if (!result.error.isEmpty()) {
                emit statusMessageRequested(tr("Surface filter query failed: %1").arg(result.error), 8000);
                return;
            }
            _focusAnswer = std::move(result.value.focus);
            _pointSetAnswer = std::move(result.value.pointSets);
            _zRangeAnswer = std::move(result.value.zRange);
            _tagAnswer = std::move(result.value.tags);
            applyFilters();
        });
}

void SurfacePanelController::updateFilterSummary()
{
    if (!_filters.dropdown) {
        return;
    }

    int activeFilters = 0;
    const auto countIfChecked = [&activeFilters](QCheckBox* box) {
        if (box && box->isChecked()) {
            ++activeFilters;
        }
    };

    countIfChecked(_filters.focusPoints);
    countIfChecked(_filters.unreviewed);
    countIfChecked(_filters.hideUnapproved);
    countIfChecked(_filters.noExpansion);
    countIfChecked(_filters.noDefective);
    countIfChecked(_filters.partialReview);
    countIfChecked(_filters.showPartialReview);
    countIfChecked(_filters.inspectOnly);
    countIfChecked(_filters.currentOnly);
    if (_filters.surfaceIdFilter && !_filters.surfaceIdFilter->text().trimmed().isEmpty()) {
        ++activeFilters;
    }
    if (z_range_filter_active(_filters.zLowerBound, _filters.zUpperBound)) {
        ++activeFilters;
    }
    if (!checkedNames(_filters.pointSet).empty()) {
        ++activeFilters;
    }
    if (!checkedNames(_recordTagFilter).empty()) {
        ++activeFilters;
    }

    QString label = tr("Filters");
    if (activeFilters > 0) {
        label += tr(" (%1)").arg(activeFilters);
    }
    _filters.dropdown->setText(label);
    _filters.dropdown->setMinimumWidth(
        _filters.dropdown->fontMetrics().horizontalAdvance(label) + 32);
}

void SurfacePanelController::updateTagSummary()
{
    if (!_tags.dropdown) {
        return;
    }

    int activeTags = 0;
    for (auto* box : {_tags.approved, _tags.defective, _tags.reviewed, _tags.inspect}) {
        if (box && box->isChecked()) {
            ++activeTags;
        }
    }

    QString label = tr("Tags");
    if (activeTags > 0) {
        label += tr(" (%1)").arg(activeTags);
    }
    _tags.dropdown->setText(label);
    _tags.dropdown->setMinimumWidth(
        _tags.dropdown->fontMetrics().horizontalAdvance(label) + 32);
}

void SurfacePanelController::onTagCheckboxToggled()
{
    if (!_ui.treeWidget || !_catalog) {
        return;
    }

    const std::vector<std::pair<QCheckBox*, const char*>> boxes{
        {_tags.approved, "approved"}, {_tags.defective, "defective"}, {_tags.reviewed, "reviewed"}, {_tags.inspect, "inspect"}};
    std::vector<std::string> add, remove;
    for (const auto& [box, name] : boxes) {
        if (box) (box->checkState() == Qt::Checked ? add : remove).push_back(name);
    }
    // The selected surfaces whose tags change; the store applies the change
    // to their current records.
    std::vector<std::string> changed;
    for (auto* item : _ui.treeWidget->selectedItems()) {
        const std::string id = item->data(SURFACE_ID_COLUMN, Qt::UserRole).toString().toStdString();
        const auto* row = _catalog->row(id);
        if (!row) {
            continue;
        }
        const auto has = [&](const std::string& tag) {
            return std::find(row->record.tags.begin(), row->record.tags.end(), tag) != row->record.tags.end();
        };
        if (std::any_of(add.begin(), add.end(), [&](const auto& tag) { return !has(tag); }) ||
            std::any_of(remove.begin(), remove.end(), has)) {
            changed.push_back(id);
        }
    }
    if (!changed.empty() && _catalog->attachment()) {
        try {
            _catalog->session()->tagObjects(*_catalog->attachment(), changed, add, remove);
        } catch (const std::exception& error) {
            emit statusMessageRequested(tr("Cannot tag surface: %1").arg(QString::fromUtf8(error.what())), 8000);
        }
    }

    updateTagSummary();
}

void SurfacePanelController::applyFiltersInternal()
{
    if (!_ui.treeWidget || !_catalog) {
        emit filtersApplied(0);
        return;
    }

    auto isChecked = [](QCheckBox* box) {
        return box && box->isChecked();
    };

    const QString idFilter = _filters.surfaceIdFilter ? _filters.surfaceIdFilter->text().trimmed() : QString{};
    const bool hasZRangeFilter = z_range_filter_active(_filters.zLowerBound, _filters.zUpperBound);
    const double zLowerBound = _filters.zLowerBound ? _filters.zLowerBound->value() : kZRangeFilterLowerDefault;
    const double zUpperBound = _filters.zUpperBound ? _filters.zUpperBound->value() : kZRangeFilterUpperDefault;
    const bool currentOnly = isChecked(_filters.currentOnly);
    const bool focusFilter = isChecked(_filters.focusPoints);
    const bool pointSetFilter = !checkedNames(_filters.pointSet).empty();
    const bool pointSetAny = !_filters.pointSetMode || _filters.pointSetMode->currentIndex() == 0;
    const auto tags = tagQuery();

    int filterCounter = 0;
    std::vector<std::string> visible;
    QTreeWidgetItemIterator it(_ui.treeWidget);
    while (*it) {
        auto* item = *it;
        ++it;
        const std::string id = item->data(SURFACE_ID_COLUMN, Qt::UserRole).toString().toStdString();
        const auto* row = _catalog->row(id);
        bool show = row != nullptr;
        if (show && !idFilter.isEmpty()) {
            show = QString::fromStdString(id).contains(idFilter, Qt::CaseInsensitive) ||
                   item->text(SURFACE_ID_COLUMN).contains(idFilter, Qt::CaseInsensitive) ||
                   item->text(SURFACE_LONG_ID_COLUMN).contains(idFilter, Qt::CaseInsensitive);
        }
        if (show && !tags.empty()) {
            // Until the store answers these tag filters, they show nothing.
            show = _tagAnswer && _tagAnswer->query == tags && _tagAnswer->surfaces.contains(id);
        }
        if (show && focusFilter) {
            // Until the store answers, the focus filter shows nothing rather than everything.
            show = _focusAnswer && !_focusAnswer->any.empty() && _focusAnswer->any.front().contains(id);
        }
        if (show && pointSetFilter) {
            show = _pointSetAnswer && !_pointSetAnswer->any.empty();
            if (show) {
                const auto& sets = pointSetAny ? _pointSetAnswer->any : _pointSetAnswer->all;
                show = pointSetAny ? std::any_of(sets.begin(), sets.end(), [&](const auto& set) { return set.contains(id); })
                                   : std::all_of(sets.begin(), sets.end(), [&](const auto& set) { return set.contains(id); });
            }
        }
        if (show && hasZRangeFilter) {
            // Until the store answers this range, the filter shows nothing.
            show = _zRangeAnswer && _zRangeAnswer->range == std::array<double, 2>{zLowerBound, zUpperBound} &&
                   _zRangeAnswer->surfaces.contains(id);
        }
        if (show && isChecked(_filters.noExpansion)) {
            show = string_or(row_meta(*row), "vc_gsfs_mode") != "expansion";
        }

        item->setHidden(!show);
        if (!show) {
            ++filterCounter;
        } else if (row->bound() && (!currentOnly || id == _currentSurfaceId)) {
            visible.push_back(id);
        }
    }

    std::sort(visible.begin(), visible.end());
    if (visible != _visibleSurfaceIds) {
        _visibleSurfaceIds = std::move(visible);
        emit visibleSurfacesChanged(_visibleSurfaceIds);
    }


    emit filtersApplied(filterCounter);
}

void SurfacePanelController::updateTagCheckboxStates(const SurfaceCatalog::Row* row)
{
    for (auto* box : {_tags.approved, _tags.defective, _tags.reviewed, _tags.inspect}) {
        if (box) {
            const QSignalBlocker blocker{box};
            box->setCheckState(Qt::Unchecked);
        }
    }

    const bool writable = _catalog && _catalog->attachment() && _catalog->session()->writable(*_catalog->attachment());
    setTagCheckboxEnabled(row != nullptr && !row->catalog && writable);
    if (row) {
        for (const auto& [box, name] : std::vector<std::pair<QCheckBox*, const char*>>{
                 {_tags.approved, "approved"}, {_tags.defective, "defective"},
                 {_tags.reviewed, "reviewed"}, {_tags.inspect, "inspect"}}) {
            if (box && has_tag(*row, name)) {
                const QSignalBlocker blocker{box};
                box->setCheckState(Qt::Checked);
            }
        }
    }
    updateTagSummary();
}

void SurfacePanelController::setTagCheckboxEnabled(bool enabled)
{
    for (auto* box : {_tags.approved, _tags.defective, _tags.reviewed, _tags.inspect}) {
        if (box) {
            box->setEnabled(enabled);
        }
    }
}

void SurfacePanelController::applyHighlightSelection(const std::string& id, bool enabled)
{
    if (id.empty()) {
        return;
    }

    if (enabled) {
        _highlightedSurfaceIds.insert(id);
    } else {
        _highlightedSurfaceIds.erase(id);
    }

    emit highlightedSurfacesChanged({_highlightedSurfaceIds.begin(), _highlightedSurfaceIds.end()});
}

void SurfacePanelController::setHighlightedSurfaceIds(const std::vector<std::string>& ids)
{
    _highlightedSurfaceIds.clear();
    for (const auto& id : ids) {
        if (!id.empty()) {
            _highlightedSurfaceIds.insert(id);
        }
    }

    emit highlightedSurfacesChanged({_highlightedSurfaceIds.begin(), _highlightedSurfaceIds.end()});
}

std::vector<std::string> SurfacePanelController::highlightedSurfaceIds() const
{
    return std::vector<std::string>(_highlightedSurfaceIds.begin(), _highlightedSurfaceIds.end());
}

bool SurfacePanelController::cycleToNextVisibleSegment()
{
    return cycleVisibleSegment(1);
}

bool SurfacePanelController::cycleToPreviousVisibleSegment()
{
    return cycleVisibleSegment(-1);
}

bool SurfacePanelController::cycleVisibleSegment(int direction)
{
    if (!_ui.treeWidget || !_catalog) {
        return false;
    }

    // Visible surfaces that open in this volume, in tree order.
    std::vector<QTreeWidgetItem*> visibleItems;
    QTreeWidgetItemIterator it(_ui.treeWidget);
    while (*it) {
        if (!(*it)->isHidden() && !(*it)->data(SURFACE_ID_COLUMN, kUnboundRole).toBool()) {
            visibleItems.push_back(*it);
        }
        ++it;
    }

    if (visibleItems.empty()) {
        return false;
    }

    int currentIndex = -1;
    const QList<QTreeWidgetItem*> selectedItems = _ui.treeWidget->selectedItems();
    if (!selectedItems.isEmpty()) {
        const auto found = std::find(visibleItems.begin(), visibleItems.end(), selectedItems.first());
        if (found != visibleItems.end()) {
            currentIndex = int(found - visibleItems.begin());
        }
    }

    const int count = int(visibleItems.size());
    const int nextIndex = currentIndex < 0 ? (direction > 0 ? 0 : count - 1)
                                           : ((currentIndex + direction) % count + count) % count;
    QTreeWidgetItem* nextItem = visibleItems[nextIndex];

    {
        // Blocked: activation below replaces handleTreeSelectionChanged.
        const QSignalBlocker blocker{_ui.treeWidget};
        _ui.treeWidget->clearSelection();
        nextItem->setSelected(true);
        _ui.treeWidget->scrollToItem(nextItem);
    }

    activate(nextItem->data(SURFACE_ID_COLUMN, Qt::UserRole).toString().toStdString(), true);
    return true;
}

#ifdef VC_TESTING
#include "SurfacePanelStoreTest.inc"
#endif
