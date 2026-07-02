#include "SurfacePanelController.hpp"

#include "SurfaceTreeWidget.hpp"
#include "ViewerManager.hpp"
#include "CState.hpp"
#include "volume_viewers/CChunkedVolumeViewer.hpp"
#include "elements/DropdownChecklistButton.hpp"
#include "VCSettings.hpp"

#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/DateTime.hpp"
#include "vc/core/util/LoadJson.hpp"
#include "utils/Json.hpp"
#include "vc/ui/VCCollection.hpp"

#include <QCheckBox>
#include <QComboBox>
#include <QDialog>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QGridLayout>
#include <QHeaderView>
#include <QHBoxLayout>
#include <QLineEdit>
#include <QAction>
#include <QLabel>
#include <QMenu>
#include <QMessageBox>
#include <QModelIndex>
#include <QPushButton>
#include <QSettings>
#include <QSignalBlocker>
#include <QStandardItem>
#include <QStandardItemModel>
#include <QStyle>
#include <QWidget>
#include <QString>
#include <QTreeWidget>
#include <QTreeWidgetItemIterator>
#include <QToolButton>
#include <QVBoxLayout>
#include <QVector>

#include <iostream>
#include <algorithm>
#include <fstream>
#include <optional>
#include <unordered_set>
#include <set>
#include <filesystem>

namespace {

constexpr double kFocusPointFilterRadius = 10.0;
constexpr double kZRangeFilterLowerDefault = 0.0;
constexpr double kZRangeFilterUpperDefault = 1000000.0;
constexpr auto kSurfaceColumnSettingsGroup = "surface_panel/columns";

bool z_range_filter_active(QDoubleSpinBox* lower, QDoubleSpinBox* upper)
{
    if (!lower || !upper) {
        return false;
    }

    return lower->value() != kZRangeFilterLowerDefault ||
           upper->value() != kZRangeFilterUpperDefault;
}

QString surface_long_id(QuadSurface* surf)
{
    if (!surf || surf->meta.is_null()) {
        return {};
    }

    for (const char* key : {
             "vc_open_data_segment_long_id",
             "segment_long_id",
             "long_id",
             "longId",
         }) {
        const auto value = vc::json::string_or(surf->meta, key, std::string{});
        if (!value.empty()) {
            return QString::fromStdString(value);
        }
    }

    return {};
}

QString surface_timestamp(QuadSurface* surf)
{
    if (!surf || surf->meta.is_null() || !surf->meta.contains("date_last_modified")) {
        return {};
    }

    return QString::fromStdString(surf->meta["date_last_modified"].get_string());
}

QString surface_column_settings_key(int column)
{
    return QStringLiteral("%1/column_%2_visible").arg(kSurfaceColumnSettingsGroup).arg(column);
}

void set_surface_tree_item_text(SurfaceTreeWidgetItem* item,
                                const std::string& id,
                                QuadSurface* surf)
{
    if (!item || !surf) {
        return;
    }

    const QString idText = QString::fromStdString(id);
    const QString longIdText = surface_long_id(surf);
    const double areaCm2 = vc::json::number_or(surf->meta, "area_cm2", -1.0);
    const double avgCost = vc::json::number_or(surf->meta, "avg_cost", -1.0);

    item->setText(SURFACE_ID_COLUMN, idText);
    item->setToolTip(SURFACE_ID_COLUMN, idText);
    item->setText(SURFACE_LONG_ID_COLUMN, longIdText);
    item->setToolTip(SURFACE_LONG_ID_COLUMN, longIdText);
    item->setText(SURFACE_AREA_COLUMN, QString::number(areaCm2, 'f', 3));
    item->setText(SURFACE_AVG_COST_COLUMN, QString::number(avgCost, 'f', 3));
    item->setText(SURFACE_OVERLAPS_COLUMN, QString::number(surf->overlappingIds().size()));
    item->setText(TIMESTAMP_COLUMN, surface_timestamp(surf));
}

void sync_tag(utils::Json& dict, bool checked, const std::string& name, const std::string& username = {})
{
    if (checked && !dict.count(name)) {
        dict[name] = utils::Json::object();
        if (!username.empty()) {
            dict[name]["user"] = username;
        }
        dict[name]["date"] = get_surface_time_str();
        if (name == "approved") {
            dict["date_last_modified"] = get_surface_time_str();
        }
    }

    if (!checked && dict.count(name)) {
        dict.erase(name);
        if (name == "approved") {
            dict["date_last_modified"] = get_surface_time_str();
        }
    }
}

} // namespace

SurfacePanelController::SurfacePanelController(const UiRefs& ui,
                                               CState* state,
                                               ViewerManager* viewerManager,
                                               std::function<CChunkedVolumeViewer*()> segmentationViewerProvider,
                                               std::function<void()> filtersUpdated,
                                               QObject* parent)
    : QObject(parent)
    , _ui(ui)
    , _state(state)
    , _viewerManager(viewerManager)
    , _segmentationViewerProvider(std::move(segmentationViewerProvider))
    , _filtersUpdated(std::move(filtersUpdated))
{
    if (_ui.reloadButton) {
        connect(_ui.reloadButton, &QPushButton::clicked, this, &SurfacePanelController::loadSurfacesIncremental);
    }

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
}

void SurfacePanelController::setVolumePkg(const std::shared_ptr<VolumePkg>& pkg)
{
    _volumePkg = pkg;
}

void SurfacePanelController::clear()
{
    if (_ui.treeWidget) {
        const QSignalBlocker blocker{_ui.treeWidget};
        _ui.treeWidget->clear();
    }
}

bool SurfacePanelController::hasSurfaces() const
{
    return _ui.treeWidget && _ui.treeWidget->topLevelItemCount() > 0;
}

void SurfacePanelController::loadSurfaces(bool reload)
{
    if (!_volumePkg) {
        return;
    }

    if (reload) {
        // Clear all surfaces from collection BEFORE unloading to prevent dangling pointers
        if (_state) {
            auto names = _state->surfaceNames();
            for (const auto& name : names) {
                _state->setSurface(name, nullptr, true, false);
            }
        }
        _volumePkg->unloadAllSurfaces();
    }

    auto segIds = _volumePkg->segmentationIDs();
    _volumePkg->loadSurfacesBatch(segIds);

    if (_state) {
        for (const auto& id : segIds) {
            auto surf = _volumePkg->getSurface(id);
            if (surf) {
                _state->setSurface(id, surf, true, false);
            }
        }
    }

    populateSurfaceTree();
    applyFilters();
    logSurfaceLoadSummary();
    if (_filtersUpdated) {
        _filtersUpdated();
    }
    if (_viewerManager) {
        _viewerManager->primeSurfacePatchIndicesAsync();
    }
    emit surfacesLoaded();
}

void SurfacePanelController::refreshSurfaceList()
{
    if (!_volumePkg) return;
    populateSurfaceTree();
    applyFilters();
    if (_filtersUpdated) _filtersUpdated();
}

void SurfacePanelController::loadSurfacesIncremental()
{
    if (!_volumePkg) {
        return;
    }

    std::cout << "Starting incremental surface load..." << std::endl;
    _volumePkg->refreshSegmentations();
    auto changes = detectSurfaceChanges();

    // Suppress signals during batch removal to avoid dangling pointer crashes
    if (_ui.treeWidget) {
        const QSignalBlocker blocker{_ui.treeWidget};
        // Perform UI mutations without emitting per-item signals.
        for (const auto& id : changes.toRemove) {
            removeSingleSegmentation(id, true);
        }
        for (const auto& id : changes.toAdd) {
            addSingleSegmentation(id);
        }
    } else {
        for (const auto& id : changes.toRemove) {
            removeSingleSegmentation(id, true);
        }
        for (const auto& id : changes.toAdd) {
            addSingleSegmentation(id);
        }
    }
    // Emit a single signal after batch removal
    if (!changes.toRemove.empty() && _state) {
        _state->emitSurfacesChanged();
    }

    if (!changes.toReload.empty()) {
        std::vector<std::string> reloadedIds;
        reloadedIds.reserve(changes.toReload.size());

        for (const auto& id : changes.toReload) {
            std::cout << "Queueing for reload: " << id << std::endl;
            auto currentSurface = _state ? _state->surface(id) : nullptr;
            auto activeSegSurface = _state ? _state->surface("segmentation") : nullptr;
            const bool wasActiveSeg = (currentSurface != nullptr && activeSegSurface.get() == currentSurface.get());

            if (_state) {
                _state->setSurface(id, nullptr, true, false);
                if (wasActiveSeg) {
                    _state->setSurface("segmentation", nullptr, false, false);
                }
            }

            _volumePkg->unloadSurface(id);
            reloadedIds.push_back(id);
        }

        _volumePkg->loadSurfacesBatch(reloadedIds);

        for (const auto& id : reloadedIds) {
            auto reloadedSurface = _volumePkg->getSurface(id);
            if (!reloadedSurface) {
                continue;
            }

            if (_state) {
                _state->setSurface(id, reloadedSurface, true, false);
                auto activeSegSurface = _state ? _state->surface("segmentation") : nullptr;
                if (activeSegSurface == nullptr) {
                    _state->setSurface("segmentation", reloadedSurface, false, false);
                }
            }

            refreshSurfaceMetrics(id);
            if (_currentSurfaceId == id) {
                syncSelectionUi(id, reloadedSurface.get());
            }
        }
    }

    std::cout << "Incremental delta: add=" << changes.toAdd.size()
              << " remove=" << changes.toRemove.size()
              << " reload=" << changes.toReload.size() << std::endl;

    applyFilters();
    logSurfaceLoadSummary();
    if (_filtersUpdated) {
        _filtersUpdated();
    }
    if (_viewerManager) {
        _viewerManager->primeSurfacePatchIndicesAsync();
    }
    emit surfacesLoaded();
    std::cout << "Incremental surface load completed." << std::endl;
}

SurfacePanelController::SurfaceChanges SurfacePanelController::detectSurfaceChanges() const
{
    SurfaceChanges changes;
    if (!_volumePkg) {
        return changes;
    }

    // Build the set of segmentation IDs currently present on disk.
    std::unordered_set<std::string> diskIds;
    for (const auto& id : _volumePkg->segmentationIDs()) {
        diskIds.insert(id);
    }

    // Build the set of IDs that the UI currently knows about (tree contents).
    std::unordered_set<std::string> uiIds;
    if (_ui.treeWidget) {
        QTreeWidgetItemIterator it(_ui.treeWidget);
        while (*it) {
            const auto qid = (*it)->data(SURFACE_ID_COLUMN, Qt::UserRole).toString();
            if (!qid.isEmpty()) {
                uiIds.insert(qid.toStdString());
            }
            ++it;
        }
    } else {
        // Fallback: if no UI is present, best-effort use currently loaded surfaces
        // (legacy behavior), but note this may be over-inclusive.
        for (const auto& id : _volumePkg->getLoadedSurfaceIDs()) {
            uiIds.insert(id);
        }
    }

    // toAdd: present on disk but not yet in the UI tree
    changes.toAdd.reserve(diskIds.size());
    for (const auto& id : diskIds) {
        if (!uiIds.contains(id)) {
            changes.toAdd.push_back(id);
        }
    }

    // toRemove: present in the UI tree but no longer on disk
    changes.toRemove.reserve(uiIds.size());
    for (const auto& uiId : uiIds) {
        if (!diskIds.contains(uiId)) {
            changes.toRemove.push_back(uiId);
        }
    }

    std::unordered_set<std::string> addedIds(
        changes.toAdd.begin(), changes.toAdd.end());
    if (_volumePkg) {
        for (const auto& uiId : uiIds) {
            if (!diskIds.contains(uiId)) {
                continue;
            }
            if (addedIds.find(uiId) != addedIds.end()) {
                continue;
            }
            if (!_volumePkg->isSurfaceLoaded(uiId)) {
                changes.toReload.push_back(uiId);
                continue;
            }
            auto surf = _volumePkg->getSurface(uiId);
            if (!surf) {
                continue;
            }
            const auto storedTs = surf->maskTimestamp();
            const auto currentTs = QuadSurface::readMaskTimestamp(surf->path);
            if (storedTs != currentTs) {
                changes.toReload.push_back(uiId);
            }
        }
    }

    std::cout << "detectSurfaceChanges: disk=" << diskIds.size()
              << " ui=" << uiIds.size()
              << " add=" << changes.toAdd.size()
              << " remove=" << changes.toRemove.size()
              << " reload=" << changes.toReload.size() << std::endl;
    return changes;
}

void SurfacePanelController::populateSurfaceTree()
{
    if (!_ui.treeWidget || !_volumePkg) {
        return;
    }

    const QSignalBlocker blocker{_ui.treeWidget};
    _ui.treeWidget->clear();

    for (const auto& id : _volumePkg->segmentationIDs()) {
        auto surf = _volumePkg->getSurface(id);

        auto* item = new SurfaceTreeWidgetItem(_ui.treeWidget);
        item->setData(SURFACE_ID_COLUMN, Qt::UserRole, QString::fromStdString(id));

        if (surf) {
            set_surface_tree_item_text(item, id, surf.get());
            updateTreeItemIcon(item);
        } else {
            delete item;
        }
    }

    for (int column = 0; column <= TIMESTAMP_COLUMN; ++column) {
        _ui.treeWidget->resizeColumnToContents(column);
    }
}

void SurfacePanelController::refreshSurfaceMetrics(const std::string& surfaceId)
{
    if (!_ui.treeWidget) {
        return;
    }

    SurfaceTreeWidgetItem* targetItem = nullptr;
    const QString idQString = QString::fromStdString(surfaceId);
    QTreeWidgetItemIterator iterator(_ui.treeWidget);
    while (*iterator) {
        if ((*iterator)->data(SURFACE_ID_COLUMN, Qt::UserRole).toString() == idQString) {
            targetItem = static_cast<SurfaceTreeWidgetItem*>(*iterator);
            break;
        }
        ++iterator;
    }

    auto surf = getSurfaceById(surfaceId);
    double areaCm2 = -1.0;
    double avgCost = -1.0;
    int overlapCount = 0;
    QString longId;
    QString timestamp;

    if (surf) {
        areaCm2 = vc::json::number_or(surf->meta, "area_cm2", -1.0);
        avgCost = vc::json::number_or(surf->meta, "avg_cost", -1.0);
        overlapCount = static_cast<int>(surf->overlappingIds().size());
        longId = surface_long_id(surf.get());
        timestamp = surface_timestamp(surf.get());
    }

    if (targetItem) {
        const QString areaText = areaCm2 >= 0.0 ? QString::number(areaCm2, 'f', 3) : QStringLiteral("-");
        const QString costText = avgCost >= 0.0 ? QString::number(avgCost, 'f', 3) : QStringLiteral("-");
        targetItem->setText(SURFACE_LONG_ID_COLUMN, longId);
        targetItem->setToolTip(SURFACE_LONG_ID_COLUMN, longId);
        targetItem->setText(SURFACE_AREA_COLUMN, areaText);
        targetItem->setText(SURFACE_AVG_COST_COLUMN, costText);
        targetItem->setText(SURFACE_OVERLAPS_COLUMN, QString::number(overlapCount));
        targetItem->setText(TIMESTAMP_COLUMN, timestamp);
        updateTreeItemIcon(targetItem);
    }
}

void SurfacePanelController::updateTreeItemIcon(SurfaceTreeWidgetItem* item)
{
    if (!item) {
        return;
    }

    const auto id = item->data(SURFACE_ID_COLUMN, Qt::UserRole).toString().toStdString();
    auto surf = getSurfaceById(id);
    if (!surf || surf->meta.is_null()) {
        return;
    }

    const auto tags = vc::json::tags_or_empty(surf->meta);
    item->updateItemIcon(tags.contains("approved"), tags.contains("defective"));
}

void SurfacePanelController::addSingleSegmentation(const std::string& segId)
{
    if (!_volumePkg) {
        return;
    }

    std::cout << "Adding segmentation: " << segId << std::endl;
    try {
        (void)_volumePkg->addSingleSegmentation(segId);
        auto surf = _volumePkg->loadSurface(segId);
        if (!surf) {
            return;
        }
        if (_state) {
            _state->setSurface(segId, surf, true, false);
        }
        if (_ui.treeWidget) {
            SurfaceTreeWidgetItem* item = nullptr;
            QTreeWidgetItemIterator it(_ui.treeWidget);
            while (*it) {
                if ((*it)->data(SURFACE_ID_COLUMN, Qt::UserRole).toString().toStdString() == segId) {
                    item = static_cast<SurfaceTreeWidgetItem*>(*it);
                    break;
                }
                ++it;
            }
            if (!item) {
                item = new SurfaceTreeWidgetItem(_ui.treeWidget);
            }
            item->setData(SURFACE_ID_COLUMN, Qt::UserRole, QString::fromStdString(segId));
            set_surface_tree_item_text(item, segId, surf.get());
            updateTreeItemIcon(item);
        }
    } catch (const std::exception& e) {
        std::cout << "Failed to add segmentation " << segId << ": " << e.what() << std::endl;
    }
}

void SurfacePanelController::removeSingleSegmentation(const std::string& segId, bool suppressSignals)
{
    std::cout << "Removing segmentation: " << segId << std::endl;

    std::shared_ptr<Surface> removedSurface;
    std::shared_ptr<Surface> activeSegSurface;

    if (_state) {
        removedSurface = _state->surface(segId);
        activeSegSurface = _state->surface("segmentation");
    }

    if (_state) {
        if (removedSurface && activeSegSurface.get() == removedSurface.get()) {
            _state->setSurface("segmentation", nullptr, suppressSignals);
        }
        _state->setSurface(segId, nullptr, suppressSignals);
    }

    if (_volumePkg) {
        _volumePkg->unloadSurface(segId);
    }

    if (_ui.treeWidget) {
        // When suppressing signals, also block tree widget signals to prevent
        // handleTreeSelectionChanged from running during batch deletion.
        // This avoids accessing surfaces that may have been deleted.
        std::optional<QSignalBlocker> blocker;
        if (suppressSignals) {
            blocker.emplace(_ui.treeWidget);
        }

        QTreeWidgetItemIterator it(_ui.treeWidget);
        while (*it) {
            if ((*it)->data(SURFACE_ID_COLUMN, Qt::UserRole).toString().toStdString() == segId) {
                const bool wasSelected = (*it)->isSelected();
                delete *it;
                if (wasSelected && !suppressSignals) {
                    emit surfaceSelectionCleared();
                }
                break;
            }
            ++it;
        }
    }
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
            if (!item) {
                continue;
            }
            const QString id = item->data(SURFACE_ID_COLUMN, Qt::UserRole).toString();
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
                if (id.isEmpty()) {
                    continue;
                }
                QTreeWidgetItemIterator it(_ui.treeWidget);
                while (*it) {
                    if ((*it)->data(SURFACE_ID_COLUMN, Qt::UserRole).toString() == id) {
                        (*it)->setSelected(true);
                        break;
                    }
                    ++it;
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
        resetTagUi();
        if (_segmentationViewerProvider) {
            if (auto* viewer = _segmentationViewerProvider()) {
                viewer->setWindowTitle(tr("Surface"));
            }
        }
        emit surfaceSelectionCleared();
        return;
    }

    auto* firstSelected = selectedItems.first();
    const QString idQString = firstSelected->data(SURFACE_ID_COLUMN, Qt::UserRole).toString();
    const std::string id = idQString.toStdString();

    std::shared_ptr<QuadSurface> surface = getSurfaceById(id);
    bool surfaceJustLoaded = (surface != nullptr);

    if (surface && _state) {
        // Keep the named entry in sync so intersection viewers can retain this mesh
        if (surfaceJustLoaded || !_state->surface(id)) {
            _state->setSurface(id, surface, true, false);
        }
    }

    syncSelectionUi(id, surface.get());

    if (_segmentationViewerProvider) {
        if (auto* viewer = _segmentationViewerProvider()) {
            viewer->setWindowTitle(surface ? tr("Surface %1").arg(idQString)
                                           : tr("Surface"));
        }
    }

    emit surfaceActivated(idQString, surface.get());

    if (surfaceJustLoaded) {
        applyFilters();
    }
}

void SurfacePanelController::showContextMenu(const QPoint& pos)
{
    if (!_ui.treeWidget) {
        return;
    }

    QTreeWidgetItem* item = _ui.treeWidget->itemAt(pos);
    if (!item) {
        return;
    }

    const QList<QTreeWidgetItem*> selectedItems = _ui.treeWidget->selectedItems();
    QStringList selectedSegmentIds;
    selectedSegmentIds.reserve(selectedItems.size());
    for (auto* selectedItem : selectedItems) {
        selectedSegmentIds << selectedItem->data(SURFACE_ID_COLUMN, Qt::UserRole).toString();
    }

    const QString segmentId = selectedSegmentIds.isEmpty() ?
        item->data(SURFACE_ID_COLUMN, Qt::UserRole).toString() :
        selectedSegmentIds.front();

    QMenu contextMenu(tr("Context Menu"), _ui.treeWidget);
    const bool isLocal = (_volumePkg != nullptr);

    if (isLocal) {
        std::string currentDir = _volumePkg->getSegmentationDirectory();
        if (currentDir == "traces") {
            QAction* moveToPathsAction = contextMenu.addAction(tr("Move to Paths"));
            moveToPathsAction->setIcon(_ui.treeWidget->style()->standardIcon(QStyle::SP_FileDialogDetailedView));
            connect(moveToPathsAction, &QAction::triggered, this, [this, segmentId]() {
                emit moveToPathsRequested(segmentId);
            });
            contextMenu.addSeparator();
        }

        // Rename only for single selection
        if (selectedSegmentIds.size() == 1) {
            QAction* renameAction = contextMenu.addAction(tr("Rename Surface"));
            connect(renameAction, &QAction::triggered, this, [this, segmentId]() {
                emit renameSurfaceRequested(segmentId);
            });
        }
    }

    if (selectedSegmentIds.size() == 1) {
        QAction* focusAction = contextMenu.addAction(tr("Focus"));
        connect(focusAction, &QAction::triggered, this, [this, segmentId]() {
            emit focusSurfaceRequested(segmentId);
        });
    }

    QAction* copyPathAction = contextMenu.addAction(tr("Copy Segment Path"));
    connect(copyPathAction, &QAction::triggered, this, [this, segmentId]() {
        emit copySegmentPathRequested(segmentId);
    });

    QMenu* maskMenu = contextMenu.addMenu(tr("Mask"));
    QAction* generateMaskAction = maskMenu->addAction(tr("Generate Segment Mask"));
    connect(generateMaskAction, &QAction::triggered, this, [this, segmentId]() {
        emit generateSegmentMaskRequested(segmentId);
    });
    QAction* appendMaskAction = maskMenu->addAction(tr("Append Surface Mask"));
    connect(appendMaskAction, &QAction::triggered, this, [this, segmentId]() {
        emit appendSegmentMaskRequested(segmentId);
    });

    contextMenu.addSeparator();

    QAction* growSegmentAction = contextMenu.addAction(tr("Run Trace"));
    connect(growSegmentAction, &QAction::triggered, this, [this, segmentId]() {
        emit growSegmentRequested(segmentId);
    });

    if (_volumePkg) {
        QMenu* copySurfaceMenu = contextMenu.addMenu(tr("Copy Surface"));
        if (selectedSegmentIds.size() == 1) {
            QAction* copySurfaceAction = copySurfaceMenu->addAction(tr("Copy Surface..."));
            connect(copySurfaceAction, &QAction::triggered, this, [this, segmentId]() {
                emit copySurfaceRequested(segmentId);
            });
            copySurfaceMenu->addSeparator();
        }
        QAction* copyOutAction = copySurfaceMenu->addAction(tr("Out"));
        connect(copyOutAction, &QAction::triggered, this, [this, segmentId]() {
            emit neighborCopyRequested(segmentId, true);
        });
        QAction* copyInAction = copySurfaceMenu->addAction(tr("In"));
        connect(copyInAction, &QAction::triggered, this, [this, segmentId]() {
            emit neighborCopyRequested(segmentId, false);
        });
        QAction* resumeLocalAction = contextMenu.addAction(tr("Reoptimize Surface"));
        connect(resumeLocalAction, &QAction::triggered, this, [this, segmentId]() {
            emit resumeLocalGrowPatchRequested(segmentId);
        });

        // Reload from Backup submenu. Backups live under
        // <backupRoot>/backups/<id> (backupRoot is the volpkg.json's directory),
        // matching QuadSurface::saveSnapshot().
        auto backupSurf = getSurfaceById(segmentId.toStdString());
        std::filesystem::path backupsDir;
        if (backupSurf && !backupSurf->path.empty()) {
            std::filesystem::path root = backupSurf->backupRoot.empty()
                ? backupSurf->path.parent_path() : backupSurf->backupRoot;
            backupsDir = root / "backups" / backupSurf->path.filename();
        }
        if (!backupsDir.empty() && std::filesystem::exists(backupsDir) && std::filesystem::is_directory(backupsDir)) {
            std::vector<int> availableBackups;
            for (const auto& entry : std::filesystem::directory_iterator(backupsDir)) {
                if (entry.is_directory()) {
                    try {
                        int idx = std::stoi(entry.path().filename().string());
                        if (idx >= 0 && idx <= 9) {
                            availableBackups.push_back(idx);
                        }
                    } catch (...) {
                        // Not a numeric directory, skip
                    }
                }
            }
            if (!availableBackups.empty()) {
                std::sort(availableBackups.begin(), availableBackups.end());
                QMenu* backupMenu = contextMenu.addMenu(tr("Reload from Backup"));
                for (int idx : availableBackups) {
                    std::filesystem::path backupPath = backupsDir / std::to_string(idx);
                    QString label = tr("Backup %1").arg(idx);

                    // Try to get timestamp from meta.json
                    std::filesystem::path metaPath = backupPath / "meta.json";
                    if (std::filesystem::exists(metaPath)) {
                        try {
                            auto meta = utils::Json::parse_file(metaPath);
                            if (meta.contains("backup_timestamp")) {
                                label = tr("Backup %1 - %2").arg(idx).arg(
                                    QString::fromStdString(meta["backup_timestamp"].get_string()));
                            }
                        } catch (...) {
                            // Couldn't read meta, use simple label
                        }
                    }

                    QAction* backupAction = backupMenu->addAction(label);
                    connect(backupAction, &QAction::triggered, this, [this, segmentId, idx]() {
                        emit reloadFromBackupRequested(segmentId, idx);
                    });
                }
            }
        }
    }

    contextMenu.addSeparator();

    QAction* renderAction = contextMenu.addAction(tr("Render segment"));
    connect(renderAction, &QAction::triggered, this, [this, segmentId]() {
        emit renderSegmentRequested(segmentId);
    });

    QAction* convertToObjAction = contextMenu.addAction(tr("Convert to OBJ"));
    connect(convertToObjAction, &QAction::triggered, this, [this, segmentId]() {
        emit convertToObjRequested(segmentId);
    });
    QAction* visObjAction = contextMenu.addAction(tr("Lasagna Vis as OBJ"));
    connect(visObjAction, &QAction::triggered, this, [this, segmentId]() {
        emit visLasagnaObjRequested(segmentId);
    });
    QAction* cropBoundsAction = contextMenu.addAction(tr("Crop bounds to valid region"));
    connect(cropBoundsAction, &QAction::triggered, this, [this, segmentId]() {
        emit cropBoundsRequested(segmentId);
    });

    QMenu* flipMenu = contextMenu.addMenu(tr("Flip Surface"));
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

    QAction* rotateAction = contextMenu.addAction(tr("Rotate Surface 90° CW"));
    connect(rotateAction, &QAction::triggered, this, [this, segmentId]() {
        emit rotateSurfaceRequested(segmentId);
    });

    QAction* refineAlphaCompAction = contextMenu.addAction(tr("Refine (Alpha-comp)"));
    connect(refineAlphaCompAction, &QAction::triggered, this, [this, segmentId]() {
        emit alphaCompRefineRequested(segmentId);
    });

    QAction* slimFlattenAction = contextMenu.addAction(tr("SLIM-flatten"));
    connect(slimFlattenAction, &QAction::triggered, this, [this, segmentId]() {
        emit slimFlattenRequested(segmentId);
    });

    QAction* straightenAction = contextMenu.addAction(tr("Straighten (vc_straighten)"));
    connect(straightenAction, &QAction::triggered, this, [this, segmentId]() {
        emit straightenRequested(segmentId);
    });

    QAction* abfFlattenAction = contextMenu.addAction(tr("ABF++ flatten"));
    connect(abfFlattenAction, &QAction::triggered, this, [this, segmentId]() {
        emit abfFlattenRequested(segmentId);
    });

    contextMenu.addSeparator();

    QAction* exportChunksAction = contextMenu.addAction(tr("Export width-chunks (40k px)"));
    connect(exportChunksAction, &QAction::triggered, this, [this, segmentId]() {
        emit exportTifxyzChunksRequested(segmentId);
    });

    QAction* rasterizeAction = contextMenu.addAction(tr("Rasterize"));
    connect(rasterizeAction, &QAction::triggered, this, [this, selectedSegmentIds, segmentId]() {
        QStringList rasterizeTargets = selectedSegmentIds;
        if (rasterizeTargets.isEmpty()) {
            rasterizeTargets << segmentId;
        }
        emit rasterizeSegmentsRequested(rasterizeTargets);
    });

    // Merge tifxyz only makes sense for >=2 surfaces; gate on selection
    // so it doesn't appear when the user right-clicks a single segment.
    if (selectedSegmentIds.size() >= 2) {
        QAction* mergeAction = contextMenu.addAction(tr("Merge tifxyz..."));
        connect(mergeAction, &QAction::triggered, this, [this, selectedSegmentIds]() {
            emit mergeTifxyzRequested(selectedSegmentIds);
        });
    }
    // Patch tifxyz is exactly-2-input; gate to exactly 2 selected segments
    // so the dialog opens with both pickers pre-filled and the user can
    // dial in border / blend without picking inputs.
    if (selectedSegmentIds.size() == 2) {
        QAction* patchAction = contextMenu.addAction(tr("Patch tifxyz..."));
        connect(patchAction, &QAction::triggered, this, [this, selectedSegmentIds]() {
            emit mergePatchRequested(selectedSegmentIds);
        });
    }

    QAction* addIgnoreLabelAction = contextMenu.addAction(tr("Add ignore label"));
    connect(addIgnoreLabelAction, &QAction::triggered, this, [this]() {
        emit addIgnoreLabelRequested();
    });

    QStringList recalcTargets = selectedSegmentIds;
    if (recalcTargets.isEmpty()) {
        recalcTargets << segmentId;
    }

    contextMenu.addSeparator();

    QAction* recalcAreaAction = contextMenu.addAction(tr("Recalculate Area from Mask"));
    connect(recalcAreaAction, &QAction::triggered, this, [this, recalcTargets]() {
        emit recalcAreaRequested(recalcTargets);
    });

    QStringList deletionTargets = selectedSegmentIds;
    if (deletionTargets.isEmpty()) {
        deletionTargets << segmentId;
    }

    QString deleteText = deletionTargets.size() > 1 ?
        tr("Delete %1 Segments").arg(deletionTargets.size()) :
        tr("Delete Segment");
    QAction* deleteAction = contextMenu.addAction(deleteText);
    deleteAction->setIcon(_ui.treeWidget->style()->standardIcon(QStyle::SP_TrashIcon));
    deleteAction->setEnabled(isLocal);
    connect(deleteAction, &QAction::triggered, this, [this, deletionTargets]() {
        handleDeleteSegments(deletionTargets);
    });

    contextMenu.addSeparator();

    const std::string segmentIdStd = segmentId.toStdString();
    QAction* highlightAction = contextMenu.addAction(tr("Highlight in slice views"));
    highlightAction->setCheckable(true);
    highlightAction->setChecked(_highlightedSurfaceIds.count(segmentIdStd) > 0);
    connect(highlightAction, &QAction::toggled, this, [this, segmentIdStd](bool checked) {
        applyHighlightSelection(segmentIdStd, checked);
    });

    contextMenu.exec(_ui.treeWidget->mapToGlobal(pos));
}

void SurfacePanelController::handleDeleteSegments(const QStringList& segmentIds)
{
    if (segmentIds.isEmpty() || !_volumePkg) {
        return;
    }

    QString message;
    if (segmentIds.size() == 1) {
        message = tr("Are you sure you want to delete segment '%1'?\n\nThis action cannot be undone.")
                      .arg(segmentIds.first());
    } else {
        message = tr("Are you sure you want to delete %1 segments?\n\nThis action cannot be undone.")
                      .arg(segmentIds.size());
    }

    QWidget* parentWidget = _ui.treeWidget ? static_cast<QWidget*>(_ui.treeWidget) : nullptr;
    QMessageBox::StandardButton reply = QMessageBox::question(
        parentWidget,
        tr("Confirm Deletion"),
        message,
        QMessageBox::Yes | QMessageBox::No,
        QMessageBox::No);

    if (reply != QMessageBox::Yes) {
        return;
    }

    int successCount = 0;
    QStringList failedSegments;
    bool anyChanges = false;

    for (const auto& id : segmentIds) {
        const std::string idStd = id.toStdString();
        try {
            // Must clean up CState before destroying the Surface
            // to avoid dangling pointers in signal handlers.
            // Suppress signals during batch deletion to prevent handlers from
            // iterating over surfaces while we're in the middle of deleting them.
            removeSingleSegmentation(idStd, true);
            _volumePkg->removeSegmentation(idStd);
            ++successCount;
            anyChanges = true;
        } catch (const std::filesystem::filesystem_error& e) {
            if (e.code() == std::errc::permission_denied) {
                failedSegments << id + tr(" (permission denied)");
            } else {
                failedSegments << id + tr(" (filesystem error)");
            }
            std::cerr << "Failed to delete segment " << idStd << ": " << e.what() << std::endl;
        } catch (const std::exception& e) {
            failedSegments << id;
            std::cerr << "Failed to delete segment " << idStd << ": " << e.what() << std::endl;
        }
    }

    // After all deletions are done, emit a single signal to trigger surface index rebuild
    if (anyChanges && _state) {
        _state->emitSurfacesChanged();
    }

    if (anyChanges) {
        try {
            _volumePkg->refreshSegmentations();
        } catch (const std::exception& e) {
            std::cerr << "Error refreshing segmentations after deletion: " << e.what() << std::endl;
        }
        applyFilters();
        if (_filtersUpdated) {
            _filtersUpdated();
        }
        emit surfacesLoaded();
    }

    if (successCount == segmentIds.size()) {
        emit statusMessageRequested(tr("Successfully deleted %1 segment(s)").arg(successCount), 5000);
    } else if (successCount > 0) {
        QMessageBox::warning(parentWidget,
                             tr("Partial Success"),
                             tr("Deleted %1 segment(s), but failed to delete: %2\n\n"
                                "Note: Permission errors may require manual deletion or running with elevated privileges.")
                                 .arg(successCount)
                                 .arg(failedSegments.join(", ")));
    } else {
        QMessageBox::critical(parentWidget,
                              tr("Deletion Failed"),
                              tr("Failed to delete any segments.\n\n"
                                 "Failed segments: %1\n\n"
                                 "This may be due to insufficient permissions. "
                                 "Try running the application with elevated privileges or manually delete the folders.")
                                  .arg(failedSegments.join(", ")));
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

    if (!_filters.currentOnly) {
        _filters.currentOnly = new QCheckBox(tr("Current Segment Only"),
                                             _filters.dropdown ? _filters.dropdown->parentWidget() : _ui.treeWidget);
        _filters.currentOnly->setObjectName(QStringLiteral("chkFilterCurrentOnly"));
    } else {
        _filters.currentOnly->setText(tr("Current Segment Only"));
    }
    _filters.currentOnly->show();
    if (_filters.dropdown && _filters.currentOnly->parentWidget() == _filters.dropdown->parentWidget()) {
        if (auto* grid = qobject_cast<QGridLayout*>(_filters.dropdown->parentWidget()->layout())) {
            grid->addWidget(_filters.currentOnly, 0, 2);
        }
    }

    if (_filters.focusPointDistance && _filters.focusPoints) {
        _filters.focusPointDistance->setEnabled(_filters.focusPoints->isChecked());
    }

    buildFilterDialog();
    connectFilterSignals();
    rebuildPointSetFilterModel();
    applyFilters();
    updateFilterSummary();
}

void SurfacePanelController::configureTags(const TagUiRefs& tags)
{
    _tags = tags;
    connectTagSignals();
    resetTagUi();
}

void SurfacePanelController::refreshPointSetFilterOptions()
{
    rebuildPointSetFilterModel();
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

void SurfacePanelController::syncSelectionUi(const std::string& surfaceId, QuadSurface* surface)
{
    _currentSurfaceId = surfaceId;
    updateTagCheckboxStatesForSurface(surface);
    if (isCurrentOnlyFilterEnabled()) {
        applyFilters();
    }
}

void SurfacePanelController::resetTagUi()
{
    _currentSurfaceId.clear();

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
}

bool SurfacePanelController::isCurrentOnlyFilterEnabled() const
{
    return _filters.currentOnly && _filters.currentOnly->isChecked();
}

bool SurfacePanelController::toggleTag(Tag tag)
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

    target->setCheckState(target->checkState() == Qt::Checked ? Qt::Unchecked : Qt::Checked);
    return true;
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

void SurfacePanelController::reloadSurfacesFromDisk()
{
    loadSurfacesIncremental();
}

void SurfacePanelController::refreshFiltersOnly()
{
    applyFilters();
}

void SurfacePanelController::setSelectionLocked(bool locked)
{
    if (_selectionLocked == locked) {
        return;
    }

    _selectionLocked = locked;
    _lockedSelectionIds.clear();
    _selectionLockNotified = false;

    if (_ui.reloadButton) {
        _ui.reloadButton->setDisabled(locked);
    }

    if (!_ui.treeWidget) {
        return;
    }

    _ui.treeWidget->setDisabled(locked);

    if (locked) {
        const QList<QTreeWidgetItem*> selectedItems = _ui.treeWidget->selectedItems();
        _lockedSelectionIds.reserve(selectedItems.size());
        for (auto* item : selectedItems) {
            if (!item) {
                continue;
            }
            const QString id = item->data(SURFACE_ID_COLUMN, Qt::UserRole).toString();
            if (!id.isEmpty()) {
                _lockedSelectionIds.append(id);
            }
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
        const bool visible = settings.value(surface_column_settings_key(column), true).toBool();
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
        form->addRow(tr("Surface ID"), _filters.surfaceIdFilter);
    }
    if (_filters.focusPointDistance) {
        if (auto* parent = _filters.focusPointDistance->parentWidget()) {
            const auto labels = parent->findChildren<QLabel*>();
            for (auto* label : labels) {
                if (label->text() == tr("Focus radius")) {
                    label->hide();
                }
            }
        }
        _filters.focusPointDistance->setParent(_filterDialog);
        _filters.focusPointDistance->show();
        form->addRow(tr("Focus radius"), _filters.focusPointDistance);
    }
    if (_filters.zLowerBound && _filters.zUpperBound) {
        if (auto* parent = _filters.zLowerBound->parentWidget()) {
            const auto labels = parent->findChildren<QLabel*>();
            for (auto* label : labels) {
                if (label->text() == tr("Z range")) {
                    label->hide();
                }
            }
        }
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
        if (_filters.pointSet) {
            _filters.pointSet->setParent(_filterDialog);
            _filters.pointSet->show();
            pointSetLayout->addWidget(_filters.pointSet, 1);
        }
        if (_filters.pointSetMode) {
            _filters.pointSetMode->setParent(_filterDialog);
            _filters.pointSetMode->show();
            pointSetLayout->addWidget(_filters.pointSetMode);
        }
        if (_filters.pointSetAll) {
            _filters.pointSetAll->setParent(_filterDialog);
            _filters.pointSetAll->show();
            pointSetLayout->addWidget(_filters.pointSetAll);
        }
        if (_filters.pointSetNone) {
            _filters.pointSetNone->setParent(_filterDialog);
            _filters.pointSetNone->show();
            pointSetLayout->addWidget(_filters.pointSetNone);
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

    connectToggle(_filters.focusPoints);
    connectToggle(_filters.unreviewed);
    connectToggle(_filters.noExpansion);
    connectToggle(_filters.noDefective);
    connectToggle(_filters.hideUnapproved);
    connectToggle(_filters.inspectOnly);
    connectToggle(_filters.currentOnly);

    if (_filters.partialReview) {
        connect(_filters.partialReview, &QCheckBox::toggled, this, [this](bool checked) {
            if (checked && _filters.showPartialReview) {
                const QSignalBlocker blocker(_filters.showPartialReview);
                _filters.showPartialReview->setChecked(false);
            }
            applyFilters();
        });
    }

    if (_filters.focusPoints && _filters.focusPointDistance) {
        connect(_filters.focusPoints, &QCheckBox::toggled,
                _filters.focusPointDistance, &QDoubleSpinBox::setEnabled);
    }

    if (_filters.focusPointDistance) {
        connect(_filters.focusPointDistance,
                QOverload<double>::of(&QDoubleSpinBox::valueChanged),
                this,
                [this](double) { applyFilters(); });
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
                    applyFilters();
                });
    }

    if (_filters.showPartialReview) {
        connect(_filters.showPartialReview, &QCheckBox::toggled, this, [this](bool checked) {
            if (checked && _filters.partialReview) {
                const QSignalBlocker blocker(_filters.partialReview);
                _filters.partialReview->setChecked(false);
            }
            applyFilters();
        });
    }

    if (_filters.pointSetMode) {
        connect(_filters.pointSetMode, &QComboBox::currentIndexChanged, this, [this]() { applyFilters(); });
    }

    if (_filters.pointSetAll) {
        connect(_filters.pointSetAll, &QPushButton::clicked, this, [this]() {
            auto* model = qobject_cast<QStandardItemModel*>(_filters.pointSet ? _filters.pointSet->model() : nullptr);
            if (!model) {
                return;
            }
            QSignalBlocker blocker(model);
            for (int row = 0; row < model->rowCount(); ++row) {
                model->setData(model->index(row, 0), Qt::Checked, Qt::CheckStateRole);
            }
            applyFilters();
        });
    }

    if (_filters.pointSetNone) {
        connect(_filters.pointSetNone, &QPushButton::clicked, this, [this]() {
            auto* model = qobject_cast<QStandardItemModel*>(_filters.pointSet ? _filters.pointSet->model() : nullptr);
            if (!model) {
                return;
            }
            QSignalBlocker blocker(model);
            for (int row = 0; row < model->rowCount(); ++row) {
                model->setData(model->index(row, 0), Qt::Unchecked, Qt::CheckStateRole);
            }
            applyFilters();
        });
    }

    if (_pointCollection) {
        connect(_pointCollection, &VCCollection::collectionsAdded, this, [this](const std::vector<uint64_t>&) {
            rebuildPointSetFilterModel();
            applyFilters();
        });
        connect(_pointCollection, &VCCollection::collectionRemoved, this, [this](uint64_t) {
            rebuildPointSetFilterModel();
            applyFilters();
        });
        connect(_pointCollection, &VCCollection::collectionChanged, this, [this](uint64_t) {
            rebuildPointSetFilterModel();
            applyFilters();
        });
        connect(_pointCollection, &VCCollection::pointAdded, this, [this](const ColPoint&) {
            applyFilters();
        });
        connect(_pointCollection, &VCCollection::pointChanged, this, [this](const ColPoint&) {
            applyFilters();
        });
        connect(_pointCollection, &VCCollection::pointRemoved, this, [this](uint64_t) {
            applyFilters();
        });
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
            auto* item = new QStandardItem(QString::fromStdString(pair.second.name));
            item->setFlags(Qt::ItemIsUserCheckable | Qt::ItemIsEnabled);
            item->setData(Qt::Unchecked, Qt::CheckStateRole);
            model->appendRow(item);
        }
    }

    _pointSetModelConnection = connect(model, &QStandardItemModel::dataChanged,
        this,
        [this](const QModelIndex&, const QModelIndex&, const QVector<int>& roles) {
            if (roles.contains(Qt::CheckStateRole)) {
                applyFilters();
            }
        });

    _configuringFilters = false;
    updateFilterSummary();
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
    if (auto* model = qobject_cast<QStandardItemModel*>(_filters.pointSet ? _filters.pointSet->model() : nullptr)) {
        bool hasPointSetFilter = false;
        for (int row = 0; row < model->rowCount(); ++row) {
            if (model->data(model->index(row, 0), Qt::CheckStateRole) == Qt::Checked) {
                hasPointSetFilter = true;
                break;
            }
        }
        if (hasPointSetFilter) {
            ++activeFilters;
        }
    }

    QString label = tr("Filters");
    if (activeFilters > 0) {
        label += tr(" (%1)").arg(activeFilters);
    }
    _filters.dropdown->setText(label);
}

void SurfacePanelController::onTagCheckboxToggled()
{
    if (!_ui.treeWidget) {
        return;
    }

    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    const std::string username = settings.value(vc3d::settings::viewer::USERNAME, vc3d::settings::viewer::USERNAME_DEFAULT).toString().toStdString();

    const auto selectedItems = _ui.treeWidget->selectedItems();
    for (auto* item : selectedItems) {
        if (!item) {
            continue;
        }

        const std::string id = item->data(SURFACE_ID_COLUMN, Qt::UserRole).toString().toStdString();
        auto surface = _volumePkg ? _volumePkg->getSurface(id) : nullptr;

        if (!surface || surface->meta.is_null()) {
            continue;
        }

        if (surface->meta.contains("tags")) {
            auto& tags = surface->meta.at("tags");
            sync_tag(tags, _tags.approved && _tags.approved->checkState() == Qt::Checked, "approved", username);
            sync_tag(tags, _tags.defective && _tags.defective->checkState() == Qt::Checked, "defective", username);
            sync_tag(tags, _tags.reviewed && _tags.reviewed->checkState() == Qt::Checked, "reviewed", username);
            sync_tag(tags, _tags.inspect && _tags.inspect->checkState() == Qt::Checked, "inspect", username);
            surface->save_meta();
        } else if ((_tags.approved && _tags.approved->checkState() == Qt::Checked) ||
                   (_tags.defective && _tags.defective->checkState() == Qt::Checked) ||
                   (_tags.reviewed && _tags.reviewed->checkState() == Qt::Checked) ||
                   (_tags.inspect && _tags.inspect->checkState() == Qt::Checked)) {
            surface->meta["tags"] = utils::Json::object();
            auto& tags = surface->meta["tags"];

            if (_tags.approved && _tags.approved->checkState() == Qt::Checked) {
                tags["approved"] = utils::Json::object();
                if (!username.empty()) {
                    tags["approved"]["user"] = username;
                }
            }
            if (_tags.defective && _tags.defective->checkState() == Qt::Checked) {
                tags["defective"] = utils::Json::object();
                if (!username.empty()) {
                    tags["defective"]["user"] = username;
                }
            }
            if (_tags.reviewed && _tags.reviewed->checkState() == Qt::Checked) {
                tags["reviewed"] = utils::Json::object();
                if (!username.empty()) {
                    tags["reviewed"]["user"] = username;
                }
            }
            if (_tags.inspect && _tags.inspect->checkState() == Qt::Checked) {
                tags["inspect"] = utils::Json::object();
                if (!username.empty()) {
                    tags["inspect"]["user"] = username;
                }
            }

            surface->save_meta();
        }

        if (auto* treeItem = dynamic_cast<SurfaceTreeWidgetItem*>(item)) {
            updateTreeItemIcon(treeItem);
        }
    }

    applyFilters();
}

void SurfacePanelController::applyFiltersInternal()
{
    if (!_ui.treeWidget) {
        emit filtersApplied(0);
        return;
    }

    auto isChecked = [](QCheckBox* box) {
        return box && box->isChecked();
    };

    const QString surfaceIdFilterText = _filters.surfaceIdFilter ? _filters.surfaceIdFilter->text().trimmed() : QString{};
    const bool hasSurfaceIdFilter = !surfaceIdFilterText.isEmpty();
    const bool hasZRangeFilter = z_range_filter_active(_filters.zLowerBound, _filters.zUpperBound);

    bool hasActiveFilters = isChecked(_filters.focusPoints) ||
                            isChecked(_filters.unreviewed) ||
                            isChecked(_filters.noExpansion) ||
                            isChecked(_filters.noDefective) ||
                            isChecked(_filters.partialReview) ||
                            isChecked(_filters.showPartialReview) ||
                            isChecked(_filters.currentOnly) ||
                            isChecked(_filters.hideUnapproved) ||
                            isChecked(_filters.inspectOnly) ||
                            hasSurfaceIdFilter ||
                            hasZRangeFilter;

    auto* model = qobject_cast<QStandardItemModel*>(_filters.pointSet ? _filters.pointSet->model() : nullptr);
    if (!hasActiveFilters && model) {
        for (int row = 0; row < model->rowCount(); ++row) {
            if (model->data(model->index(row, 0), Qt::CheckStateRole) == Qt::Checked) {
                hasActiveFilters = true;
                break;
            }
        }
    }

    auto collectVisibleSurfaces = [&](std::set<std::string>& out) {
        if (!_ui.treeWidget) {
            return;
        }
        QTreeWidgetItemIterator visIt(_ui.treeWidget);
        while (*visIt) {
            auto* item = *visIt;
            const auto idStr = item->data(SURFACE_ID_COLUMN, Qt::UserRole).toString();
            std::string id = idStr.toStdString();
            if (!id.empty() && !item->isHidden()) {
                // Only use already-loaded surfaces; never trigger TIFF I/O from filters.
                auto surf = getSurfaceById(id);
                if (surf) {
                    out.insert(id);
                    if (_state && !_state->surface(id)) {
                        _state->setSurface(id, surf, true, false);
                    }
                }
            }
            ++visIt;
        }
    };

    if (!hasActiveFilters) {
        QTreeWidgetItemIterator it(_ui.treeWidget);
        while (*it) {
            (*it)->setHidden(false);
            ++it;
        }

        std::set<std::string> intersects = {"segmentation"};
        collectVisibleSurfaces(intersects);

        if (_viewerManager) {
            _viewerManager->forEachBaseViewer([&intersects](VolumeViewerBase* viewer) {
                if (viewer && viewer->surfName() != "segmentation") {
                    viewer->setIntersects(intersects);
                }
            });
        }

        emit filtersApplied(0);
        return;
    }

    std::set<std::string> intersects = {"segmentation"};
    POI* poi = _state ? _state->poi("focus") : nullptr;
    int filterCounter = 0;
    const bool currentOnly = isChecked(_filters.currentOnly);
    const bool focusPointFilter = isChecked(_filters.focusPoints) && poi;
    const double focusPointFilterRadius = _filters.focusPointDistance
        ? _filters.focusPointDistance->value()
        : kFocusPointFilterRadius;
    const double zLowerBound = _filters.zLowerBound
        ? _filters.zLowerBound->value()
        : kZRangeFilterLowerDefault;
    const double zUpperBound = _filters.zUpperBound
        ? _filters.zUpperBound->value()
        : kZRangeFilterUpperDefault;
    std::unordered_set<QuadSurface*> focusPointSurfaces;
    if (focusPointFilter) {
        if (auto* patchIndex = _viewerManager ? _viewerManager->surfacePatchIndex() : nullptr) {
            SurfacePatchIndex::PointQuery query;
            query.worldPoint = poi->p;
            query.tolerance = static_cast<float>(focusPointFilterRadius);
            for (const auto& hit : patchIndex->locateAll(query)) {
                if (hit.surface) {
                    focusPointSurfaces.insert(hit.surface.get());
                }
            }
        }
    }

    QTreeWidgetItemIterator it(_ui.treeWidget);
    while (*it) {
        auto* item = *it;
        std::string id = item->data(SURFACE_ID_COLUMN, Qt::UserRole).toString().toStdString();

        bool show = true;
        // Only use already-loaded surfaces; never trigger TIFF I/O from filters.
        auto surf = getSurfaceById(id);

        if (hasSurfaceIdFilter && !id.empty()) {
            const bool idMatches =
                QString::fromStdString(id).contains(surfaceIdFilterText, Qt::CaseInsensitive);
            const bool longIdMatches =
                item->text(SURFACE_LONG_ID_COLUMN).contains(surfaceIdFilterText, Qt::CaseInsensitive);
            show = show && (idMatches || longIdMatches);
        }

        if (focusPointFilter) {
            show = show && surf && focusPointSurfaces.find(surf.get()) != focusPointSurfaces.end();
        }

        if (hasZRangeFilter && !surf) {
            show = false;
        }

        if (surf) {
            if (hasZRangeFilter) {
                const auto bbox = surf->bbox();
                show = show && bbox.low[0] != -1.0f &&
                       bbox.low[2] <= zUpperBound &&
                       bbox.high[2] >= zLowerBound;
            }

            if (model) {
                bool anyChecked = false;
                bool anyMatches = false;
                bool allMatch = true;
                for (int row = 0; row < model->rowCount(); ++row) {
                    if (model->data(model->index(row, 0), Qt::CheckStateRole) == Qt::Checked) {
                        anyChecked = true;
                        const auto collectionName = model->data(model->index(row, 0), Qt::DisplayRole).toString().toStdString();
                        std::vector<cv::Vec3f> points;
                        if (_pointCollection) {
                            auto collection = _pointCollection->getPoints(collectionName);
                            points.reserve(collection.size());
                            for (const auto& p : collection) {
                                points.push_back(p.p);
                            }
                        }
                        if (allMatch && !contains(*surf, points)) {
                            allMatch = false;
                        }
                        if (!anyMatches && contains_any(*surf, points)) {
                            anyMatches = true;
                        }
                    }
                }

                if (anyChecked) {
                    if (_filters.pointSetMode && _filters.pointSetMode->currentIndex() == 0) {
                        show = show && anyMatches;
                    } else {
                        show = show && allMatch;
                    }
                }
            }

            if (isChecked(_filters.unreviewed)) {
                if (!surf->meta.is_null()) {
                    const auto tags = vc::json::tags_or_empty(surf->meta);
                    show = show && !tags.contains("reviewed");
                }
            }

            if (isChecked(_filters.noExpansion)) {
                if (!surf->meta.is_null()) {
                    const auto mode = vc::json::string_or(surf->meta, "vc_gsfs_mode", std::string{});
                    show = show && (mode != "expansion");
                }
            }

            if (isChecked(_filters.noDefective)) {
                if (!surf->meta.is_null()) {
                    const auto tags = vc::json::tags_or_empty(surf->meta);
                    show = show && !tags.contains("defective");
                }
            }

            if (isChecked(_filters.partialReview)) {
                if (!surf->meta.is_null()) {
                    const auto tags = vc::json::tags_or_empty(surf->meta);
                    const bool hasPartialReview = tags.contains("partial_review") || tags.contains("reviewed");
                    show = show && !hasPartialReview;
                }
            }

            if (isChecked(_filters.showPartialReview)) {
                if (!surf->meta.is_null()) {
                    const auto tags = vc::json::tags_or_empty(surf->meta);
                    const bool hasPartialReview = tags.contains("partial_review") || tags.contains("reviewed");
                    show = show && hasPartialReview;
                } else {
                    show = false;
                }
            }

            if (isChecked(_filters.hideUnapproved)) {
                if (!surf->meta.is_null()) {
                    const auto tags = vc::json::tags_or_empty(surf->meta);
                    show = show && tags.contains("approved");
                } else {
                    show = false;
                }
            }

            if (isChecked(_filters.inspectOnly)) {
                if (!surf->meta.is_null()) {
                    const auto tags = vc::json::tags_or_empty(surf->meta);
                    show = show && tags.contains("inspect");
                } else {
                    show = false;
                }
            }
        }

        item->setHidden(!show);

        if (!show) {
            filterCounter++;
        }

        ++it;
    }

    intersects.clear();
    intersects.insert("segmentation");
    if (currentOnly) {
        // Only show the current segment's intersection line.
        // If no segment is selected, just show "segmentation" alone.
        if (!_currentSurfaceId.empty() && getSurfaceById(_currentSurfaceId)) {
            intersects.insert(_currentSurfaceId);
        }
    } else {
        collectVisibleSurfaces(intersects);
    }

    if (_viewerManager) {
        _viewerManager->forEachBaseViewer([&intersects](VolumeViewerBase* viewer) {
            if (viewer && viewer->surfName() != "segmentation") {
                viewer->setIntersects(intersects);
            }
        });
    }

    emit filtersApplied(filterCounter);
}

void SurfacePanelController::updateTagCheckboxStatesForSurface(QuadSurface* surface)
{
    auto resetState = [](QCheckBox* box) {
        if (!box) {
            return;
        }
        const QSignalBlocker blocker{box};
        box->setCheckState(Qt::Unchecked);
    };

    resetState(_tags.approved);
    resetState(_tags.defective);
    resetState(_tags.reviewed);
    resetState(_tags.inspect);

    if (!surface) {
        setTagCheckboxEnabled(false, false, false, false);
        return;
    }

    setTagCheckboxEnabled(true, true, true, true);

    if (surface->meta.is_null()) {
        setTagCheckboxEnabled(false, false, true, true);
        return;
    }

    const auto tags = vc::json::tags_or_empty(surface->meta);

    auto applyTag = [&tags](QCheckBox* box, const char* name) {
        if (!box) {
            return;
        }
        const QSignalBlocker blocker{box};
        if (tags.contains(name)) {
            box->setCheckState(Qt::Checked);
        }
    };

    applyTag(_tags.approved, "approved");
    applyTag(_tags.defective, "defective");
    applyTag(_tags.reviewed, "reviewed");
    applyTag(_tags.inspect, "inspect");
}

void SurfacePanelController::setTagCheckboxEnabled(bool enabledApproved,
                                                   bool enabledDefective,
                                                   bool enabledReviewed,
                                                   bool enabledInspect)
{
    if (_tags.approved) {
        _tags.approved->setEnabled(enabledApproved);
    }
    if (_tags.defective) {
        _tags.defective->setEnabled(enabledDefective);
    }
    if (_tags.reviewed) {
        _tags.reviewed->setEnabled(enabledReviewed);
    }
    if (_tags.inspect) {
        _tags.inspect->setEnabled(enabledInspect);
    }
}

void SurfacePanelController::logSurfaceLoadSummary() const
{
    if (!_volumePkg) {
        std::cout << "[SurfacePanel] No volume package set; skipping surface load summary." << std::endl;
        return;
    }

    const auto segIds = _volumePkg->segmentationIDs();
    if (segIds.empty()) {
        std::cout << "[SurfacePanel] No segmentation IDs available." << std::endl;
        return;
    }

    size_t loadedCount = 0;
    std::vector<std::string> missing;
    missing.reserve(segIds.size());

    for (const auto& id : segIds) {
        bool hasSurface = false;
        if (_state) {
            if (_state->surface(id)) {
                hasSurface = true;
            }
        } else {
            hasSurface = static_cast<bool>(_volumePkg->getSurface(id));
        }

        if (hasSurface) {
            ++loadedCount;
        } else {
            missing.push_back(id);
        }
    }

    std::cout << "[SurfacePanel] Loaded " << loadedCount << " / " << segIds.size()
              << " surfaces into memory." << std::endl;
    if (!missing.empty()) {
        const size_t previewCount = std::min<size_t>(missing.size(), 10);
        std::cout << "[SurfacePanel] Missing (" << missing.size() << ") IDs: ";
        for (size_t i = 0; i < previewCount; ++i) {
            std::cout << missing[i];
            if (i + 1 < previewCount) {
                std::cout << ", ";
            }
        }
        if (missing.size() > previewCount) {
            std::cout << ", ...";
        }
        std::cout << std::endl;
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

    if (_viewerManager) {
        std::vector<std::string> ids(_highlightedSurfaceIds.begin(), _highlightedSurfaceIds.end());
        _viewerManager->setHighlightedSurfaceIds(ids);
    }
}

std::shared_ptr<QuadSurface> SurfacePanelController::getSurfaceById(const std::string& id) const
{
    // Try volumePkg first (local mode)
    if (_volumePkg) {
        auto surf = _volumePkg->getSurface(id);
        if (surf) {
            return surf;
        }
    }
    // Fall back to CState (remote mode)
    if (_state) {
        auto surf = _state->surface(id);
        if (surf) {
            return std::dynamic_pointer_cast<QuadSurface>(surf);
        }
    }
    return nullptr;
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
    if (!_ui.treeWidget) {
        return false;
    }

    // Collect all visible (non-hidden) items in tree order
    std::vector<QTreeWidgetItem*> visibleItems;
    QTreeWidgetItemIterator it(_ui.treeWidget);
    while (*it) {
        if (!(*it)->isHidden()) {
            visibleItems.push_back(*it);
        }
        ++it;
    }

    if (visibleItems.empty()) {
        return false;
    }

    // Find current selection index
    int currentIndex = -1;
    const QList<QTreeWidgetItem*> selectedItems = _ui.treeWidget->selectedItems();
    if (!selectedItems.isEmpty()) {
        QTreeWidgetItem* currentItem = selectedItems.first();
        for (size_t i = 0; i < visibleItems.size(); ++i) {
            if (visibleItems[i] == currentItem) {
                currentIndex = static_cast<int>(i);
                break;
            }
        }
    }

    // Calculate next index with wraparound
    int nextIndex;
    if (currentIndex < 0) {
        nextIndex = (direction > 0) ? 0 : static_cast<int>(visibleItems.size()) - 1;
    } else {
        nextIndex = currentIndex + direction;
        if (nextIndex < 0) {
            nextIndex = static_cast<int>(visibleItems.size()) - 1;
        } else if (nextIndex >= static_cast<int>(visibleItems.size())) {
            nextIndex = 0;
        }
    }

    QTreeWidgetItem* nextItem = visibleItems[nextIndex];

    // Block signals to prevent normal handleTreeSelectionChanged
    const QSignalBlocker blocker{_ui.treeWidget};
    _ui.treeWidget->clearSelection();
    nextItem->setSelected(true);
    _ui.treeWidget->scrollToItem(nextItem);

    // Get surface and update state
    const QString idQString = nextItem->data(SURFACE_ID_COLUMN, Qt::UserRole).toString();
    const std::string id = idQString.toStdString();

    std::shared_ptr<QuadSurface> surface = getSurfaceById(id);

    if (surface && _state) {
        if (!_state->surface(id)) {
            _state->setSurface(id, surface, true, false);
        }
    }

    _currentSurfaceId = id;
    syncSelectionUi(id, surface.get());

    if (_segmentationViewerProvider) {
        if (auto* viewer = _segmentationViewerProvider()) {
            viewer->setWindowTitle(surface ? tr("Surface %1").arg(idQString) : tr("Surface"));
        }
    }

    emit surfaceActivatedPreserveEditing(idQString, surface.get());
    return true;
}
