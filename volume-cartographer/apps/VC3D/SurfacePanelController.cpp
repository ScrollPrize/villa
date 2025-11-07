#include "SurfacePanelController.hpp"

#include "SurfaceTreeWidget.hpp"
#include "ViewerManager.hpp"
#include "CSurfaceCollection.hpp"
#include "OpChain.hpp"
#include "CVolumeViewer.hpp"
#include "elements/DropdownChecklistButton.hpp"
#include "VCSettings.hpp"

#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/DateTime.hpp"
#include "vc/core/util/JsonSafe.hpp"
#include "vc/ui/VCCollection.hpp"

#include <QCheckBox>
#include <QComboBox>
#include <QAction>
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
#include <QVector>

#include <iostream>
#include <algorithm>
#include <unordered_set>
#include <set>
#include <filesystem>

namespace {

void sync_tag(nlohmann::json& dict, bool checked, const std::string& name, const std::string& username = {})
{
    if (checked && !dict.count(name)) {
        dict[name] = nlohmann::json::object();
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
                                               CSurfaceCollection* surfaces,
                                               ViewerManager* viewerManager,
                                               std::unordered_map<std::string, OpChain*>* opchains,
                                               std::function<CVolumeViewer*()> segmentationViewerProvider,
                                               std::function<void()> filtersUpdated,
                                               QObject* parent)
    : QObject(parent)
    , _ui(ui)
    , _surfaces(surfaces)
    , _viewerManager(viewerManager)
    , _opchains(opchains)
    , _segmentationViewerProvider(std::move(segmentationViewerProvider))
    , _filtersUpdated(std::move(filtersUpdated))
{
    if (_ui.reloadButton) {
        connect(_ui.reloadButton, &QPushButton::clicked, this, &SurfacePanelController::loadSurfacesIncremental);
    }

    if (_ui.treeWidget) {
        _ui.treeWidget->setContextMenuPolicy(Qt::CustomContextMenu);
        connect(_ui.treeWidget, &QTreeWidget::itemSelectionChanged,
                this, &SurfacePanelController::handleTreeSelectionChanged);
        connect(_ui.treeWidget, &QWidget::customContextMenuRequested,
                this, &SurfacePanelController::showContextMenu);
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

void SurfacePanelController::loadSurfaces(bool reload)
{
    if (!_volumePkg) {
        return;
    }

    if (reload) {
        if (_opchains) {
            for (auto& pair : *_opchains) {
                delete pair.second;
            }
            _opchains->clear();
        }
        _volumePkg->unloadAllSurfaces();
    }

    auto segIds = _volumePkg->segmentationIDs();
    _volumePkg->loadSurfacesBatch(segIds);

    if (_surfaces) {
        for (const auto& id : segIds) {
            auto surfMeta = _volumePkg->getSurface(id);
            if (surfMeta) {
                _surfaces->setSurface(id, surfMeta->surface(), true, false);
            }
        }
    }

    populateSurfaceTree();
    applyFilters();
    if (_filtersUpdated) {
        _filtersUpdated();
    }
    emit surfacesLoaded();
}

void SurfacePanelController::loadSurfacesIncremental()
{
    if (!_volumePkg) {
        return;
    }

    std::cout << "Starting incremental surface load..." << std::endl;
    _volumePkg->refreshSegmentations();
    auto changes = detectSurfaceChanges();

    if (_ui.treeWidget) {
        const QSignalBlocker blocker{_ui.treeWidget};
        // Perform UI mutations without emitting per-item signals.
        for (const auto& id : changes.toRemove) {
            removeSingleSegmentation(id);
        }
        for (const auto& id : changes.toAdd) {
            addSingleSegmentation(id);
        }
    } else {
        for (const auto& id : changes.toRemove) {
            removeSingleSegmentation(id);
        }
        for (const auto& id : changes.toAdd) {
            addSingleSegmentation(id);
        }
    }

    if (!changes.toReload.empty()) {
        std::vector<std::string> reloadedIds;
        reloadedIds.reserve(changes.toReload.size());

        for (const auto& id : changes.toReload) {
            std::cout << "Queueing for reload: " << id << std::endl;
            Surface* currentSurface = _surfaces ? _surfaces->surface(id) : nullptr;
            Surface* activeSegSurface = _surfaces ? _surfaces->surface("segmentation") : nullptr;
            const bool wasActiveSeg = (currentSurface != nullptr && activeSegSurface == currentSurface);

            if (_surfaces) {
                _surfaces->setSurface(id, nullptr, true, false);
                if (wasActiveSeg) {
                    _surfaces->setSurface("segmentation", nullptr, false, false);
                }
            }

            if (_opchains && _opchains->count(id)) {
                delete (*_opchains)[id];
                _opchains->erase(id);
            }
            _volumePkg->unloadSurface(id);
            reloadedIds.push_back(id);
        }

        _volumePkg->loadSurfacesBatch(reloadedIds);

        for (const auto& id : reloadedIds) {
            auto surfMeta = _volumePkg->getSurface(id);
            if (!surfMeta) {
                continue;
            }

            QuadSurface* reloadedSurface = surfMeta->surface();
            if (_surfaces) {
                _surfaces->setSurface(id, reloadedSurface, true, false);
                Surface* activeSegSurface = _surfaces ? _surfaces->surface("segmentation") : nullptr;
                if (activeSegSurface == nullptr) {
                    _surfaces->setSurface("segmentation", reloadedSurface, false, false);
                }
            }

            refreshSurfaceMetrics(id);
            if (_currentSurfaceId == id) {
                syncSelectionUi(id, reloadedSurface);
            }
        }
    }

    std::cout << "Incremental delta: add=" << changes.toAdd.size()
              << " remove=" << changes.toRemove.size()
              << " reload=" << changes.toReload.size() << std::endl;

    applyFilters();
    if (_filtersUpdated) {
        _filtersUpdated();
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
            // Only check timestamps for surfaces that are actually loaded in memory.
            // If not loaded, we'll get fresh data when we eventually load it.
            if (!_volumePkg->isSurfaceLoaded(uiId)) {
                continue;
            }
            auto surfMeta = _volumePkg->getSurface(uiId);
            if (!surfMeta) {
                continue;
            }
            const auto storedTs = surfMeta->maskTimestamp();
            const auto currentTs = SurfaceMeta::readMaskTimestamp(surfMeta->path);
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
        auto surfMeta = _volumePkg->getSurface(id);
        if (!surfMeta) {
            continue;
        }

        auto* item = new SurfaceTreeWidgetItem(_ui.treeWidget);
        item->setText(SURFACE_ID_COLUMN, QString::fromStdString(id));
        item->setData(SURFACE_ID_COLUMN, Qt::UserRole, QString::fromStdString(id));
        const double areaCm2 = vc::json_safe::number_or(surfMeta->meta, "area_cm2", -1.0);
        const double avgCost = vc::json_safe::number_or(surfMeta->meta, "avg_cost", -1.0);
        item->setText(2, QString::number(areaCm2, 'f', 3));
        item->setText(3, QString::number(avgCost, 'f', 3));
        item->setText(4, QString::number(surfMeta->overlapping_str.size()));
        QString timestamp;
        if (surfMeta->meta && surfMeta->meta->contains("date_last_modified")) {
            timestamp = QString::fromStdString((*surfMeta->meta)["date_last_modified"].get<std::string>());
        }
        item->setText(5, timestamp);
        updateTreeItemIcon(item);
    }

    _ui.treeWidget->resizeColumnToContents(0);
    _ui.treeWidget->resizeColumnToContents(1);
    _ui.treeWidget->resizeColumnToContents(2);
    _ui.treeWidget->resizeColumnToContents(3);
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

    auto surfMeta = _volumePkg ? _volumePkg->getSurface(surfaceId) : nullptr;
    double areaCm2 = -1.0;
    double avgCost = -1.0;
    int overlapCount = 0;
    QString timestamp;

    if (surfMeta) {
        areaCm2 = vc::json_safe::number_or(surfMeta->meta, "area_cm2", -1.0);
        avgCost = vc::json_safe::number_or(surfMeta->meta, "avg_cost", -1.0);
        overlapCount = static_cast<int>(surfMeta->overlapping_str.size());
        if (surfMeta->meta && surfMeta->meta->contains("date_last_modified")) {
            timestamp = QString::fromStdString((*surfMeta->meta)["date_last_modified"].get<std::string>());
        }
    }

    if (targetItem) {
        const QString areaText = areaCm2 >= 0.0 ? QString::number(areaCm2, 'f', 3) : QStringLiteral("-");
        const QString costText = avgCost >= 0.0 ? QString::number(avgCost, 'f', 3) : QStringLiteral("-");
        targetItem->setText(2, areaText);
        targetItem->setText(3, costText);
        targetItem->setText(4, QString::number(overlapCount));
        targetItem->setText(TIMESTAMP_COLUMN, timestamp);
        updateTreeItemIcon(targetItem);
    }
}

void SurfacePanelController::updateTreeItemIcon(SurfaceTreeWidgetItem* item)
{
    if (!item || !_volumePkg) {
        return;
    }

    const auto id = item->data(SURFACE_ID_COLUMN, Qt::UserRole).toString().toStdString();
    auto surfMeta = _volumePkg->getSurface(id);
    if (!surfMeta || !surfMeta->surface() || !surfMeta->surface()->meta) {
        return;
    }

    const auto tags = vc::json_safe::tags_or_empty(surfMeta->surface()->meta);
    item->updateItemIcon(tags.contains("approved"), tags.contains("defective"));
}

void SurfacePanelController::addSingleSegmentation(const std::string& segId)
{
    if (!_volumePkg) {
        return;
    }

    std::cout << "Adding segmentation: " << segId << std::endl;
    try {
        auto surfMeta = _volumePkg->loadSurface(segId);
        if (!surfMeta) {
            return;
        }
        if (_surfaces) {
            _surfaces->setSurface(segId, surfMeta->surface(), true, false);
        }
        if (_ui.treeWidget) {
            auto* item = new SurfaceTreeWidgetItem(_ui.treeWidget);
            item->setText(SURFACE_ID_COLUMN, QString::fromStdString(segId));
            item->setData(SURFACE_ID_COLUMN, Qt::UserRole, QString::fromStdString(segId));
            const double areaCm2 = vc::json_safe::number_or(surfMeta->meta, "area_cm2", -1.0);
            const double avgCost = vc::json_safe::number_or(surfMeta->meta, "avg_cost", -1.0);
            item->setText(2, QString::number(areaCm2, 'f', 3));
            item->setText(3, QString::number(avgCost, 'f', 3));
            item->setText(4, QString::number(surfMeta->overlapping_str.size()));
            QString timestamp;
            if (surfMeta->meta && surfMeta->meta->contains("date_last_modified")) {
                timestamp = QString::fromStdString((*surfMeta->meta)["date_last_modified"].get<std::string>());
            }
            item->setText(5, timestamp);
            updateTreeItemIcon(item);
        }
    } catch (const std::exception& e) {
        std::cout << "Failed to add segmentation " << segId << ": " << e.what() << std::endl;
    }
}

void SurfacePanelController::removeSingleSegmentation(const std::string& segId)
{
    std::cout << "Removing segmentation: " << segId << std::endl;

    Surface* removedSurface = nullptr;
    Surface* activeSegSurface = nullptr;

    if (_surfaces) {
        removedSurface = _surfaces->surface(segId);
        activeSegSurface = _surfaces->surface("segmentation");
    }

    if (_surfaces) {
        if (removedSurface && activeSegSurface == removedSurface) {
            _surfaces->setSurface("segmentation", nullptr);
        }
        _surfaces->setSurface(segId, nullptr, false);
    }

    if (_volumePkg) {
        _volumePkg->unloadSurface(segId);
    }

    if (_opchains && _opchains->count(segId)) {
        delete (*_opchains)[segId];
        _opchains->erase(segId);
    }

    if (_ui.treeWidget) {
        QTreeWidgetItemIterator it(_ui.treeWidget);
        while (*it) {
            if ((*it)->data(SURFACE_ID_COLUMN, Qt::UserRole).toString().toStdString() == segId) {
                const bool wasSelected = (*it)->isSelected();
                delete *it;
                if (wasSelected) {
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

    OpChain* chain = ensureOpChainFor(id);
    QuadSurface* surface = chain ? chain->src() : nullptr;

    if (!surface && _volumePkg) {
        if (auto surfMeta = _volumePkg->getSurface(id)) {
            surface = surfMeta->surface();
        }
    }

    if (surface && _surfaces) {
        _surfaces->setSurface("segmentation", surface, false, false);
    }

    syncSelectionUi(id, surface);

    if (_segmentationViewerProvider) {
        if (auto* viewer = _segmentationViewerProvider()) {
            viewer->setWindowTitle(surface ? tr("Surface %1").arg(idQString)
                                           : tr("Surface"));
        }
    }

    emit surfaceActivated(idQString, surface, chain);
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

    std::string currentDir = _volumePkg->getSegmentationDirectory();
    if (currentDir == "traces") {
        QAction* moveToPathsAction = contextMenu.addAction(tr("Move to Paths"));
        moveToPathsAction->setIcon(_ui.treeWidget->style()->standardIcon(QStyle::SP_FileDialogDetailedView));
        connect(moveToPathsAction, &QAction::triggered, this, [this, segmentId]() {
            emit moveToPathsRequested(segmentId);
        });
        contextMenu.addSeparator();
    }

    QAction* copyPathAction = contextMenu.addAction(tr("Copy Segment Path"));
    connect(copyPathAction, &QAction::triggered, this, [this, segmentId]() {
        emit copySegmentPathRequested(segmentId);
    });

    contextMenu.addSeparator();

    QMenu* seedMenu = contextMenu.addMenu(tr("Run Seed"));
    QAction* seedWithSeedAction = seedMenu->addAction(tr("Seed from Focus Point"));
    connect(seedWithSeedAction, &QAction::triggered, this, [this, segmentId]() {
        emit growSeedsRequested(segmentId, false, false);
    });
    QAction* seedWithRandomAction = seedMenu->addAction(tr("Random Seed"));
    connect(seedWithRandomAction, &QAction::triggered, this, [this, segmentId]() {
        emit growSeedsRequested(segmentId, false, true);
    });
    QAction* seedWithExpandAction = seedMenu->addAction(tr("Expand Seed"));
    connect(seedWithExpandAction, &QAction::triggered, this, [this, segmentId]() {
        emit growSeedsRequested(segmentId, true, false);
    });

    QAction* growSegmentAction = contextMenu.addAction(tr("Run Trace"));
    connect(growSegmentAction, &QAction::triggered, this, [this, segmentId]() {
        emit growSegmentRequested(segmentId);
    });

    QAction* addOverlapAction = contextMenu.addAction(tr("Add overlap"));
    connect(addOverlapAction, &QAction::triggered, this, [this, segmentId]() {
        emit addOverlapRequested(segmentId);
    });

    if (_volumePkg) {
        QAction* copyOutAction = contextMenu.addAction(tr("Copy Out"));
        connect(copyOutAction, &QAction::triggered, this, [this, segmentId]() {
            emit neighborCopyRequested(segmentId, true);
        });
        QAction* copyInAction = contextMenu.addAction(tr("Copy In"));
        connect(copyInAction, &QAction::triggered, this, [this, segmentId]() {
            emit neighborCopyRequested(segmentId, false);
        });
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

    QAction* refineAlphaCompAction = contextMenu.addAction(tr("Refine (Alpha-comp)"));
    connect(refineAlphaCompAction, &QAction::triggered, this, [this, segmentId]() {
        emit alphaCompRefineRequested(segmentId);
    });

    QAction* slimFlattenAction = contextMenu.addAction(tr("SLIM-flatten"));
    connect(slimFlattenAction, &QAction::triggered, this, [this, segmentId]() {
        emit slimFlattenRequested(segmentId);
    });

    QAction* awsUploadAction = contextMenu.addAction(tr("Upload artifacts to AWS"));
    connect(awsUploadAction, &QAction::triggered, this, [this, segmentId]() {
        emit awsUploadRequested(segmentId);
    });

    contextMenu.addSeparator();

    QAction* exportChunksAction = contextMenu.addAction(tr("Export width-chunks (40k px)"));
    connect(exportChunksAction, &QAction::triggered, this, [this, segmentId]() {
        emit exportTifxyzChunksRequested(segmentId);
    });

    contextMenu.addSeparator();

    QAction* inpaintTeleaAction = contextMenu.addAction(tr("Inpaint (Telea) && Rebuild Segment"));
    connect(inpaintTeleaAction, &QAction::triggered, this, [this]() {
        emit teleaInpaintRequested();
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
    connect(deleteAction, &QAction::triggered, this, [this, deletionTargets]() {
        handleDeleteSegments(deletionTargets);
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
            _volumePkg->removeSegmentation(idStd);
            ++successCount;
            anyChanges = true;
            removeSingleSegmentation(idStd);
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
        if (auto* menu = _filters.dropdown->menu()) {
            menu->setObjectName(QStringLiteral("menuFilters"));
        }
    }

    _filters.focusPoints = nullptr;
    _filters.unreviewed = nullptr;
    _filters.revisit = nullptr;
    _filters.hideUnapproved = nullptr;
    _filters.noExpansion = nullptr;
    _filters.noDefective = nullptr;
    _filters.partialReview = nullptr;
    _filters.inspectOnly = nullptr;
    _filters.currentOnly = nullptr;

    const auto addFilterOption = [&](QCheckBox*& target, const QString& text, const QString& objectName) {
        if (_filters.dropdown) {
            target = _filters.dropdown->addOption(text, objectName);
            return;
        }

        if (!target) {
            target = new QCheckBox(text);
            if (!objectName.isEmpty()) {
                target->setObjectName(objectName);
            }
        } else {
            target->setText(text);
        }
        target->hide();
    };

    const auto addSeparator = [&]() {
        if (_filters.dropdown) {
            _filters.dropdown->addSeparator();
        }
    };

    addFilterOption(_filters.focusPoints, tr("Focus Point"), QStringLiteral("chkFilterFocusPoints"));
    addSeparator();
    addFilterOption(_filters.unreviewed, tr("Unreviewed"), QStringLiteral("chkFilterUnreviewed"));
    addFilterOption(_filters.revisit, tr("Revisit"), QStringLiteral("chkFilterRevisit"));
    addFilterOption(_filters.hideUnapproved, tr("Hide Unapproved"), QStringLiteral("chkFilterHideUnapproved"));
    addSeparator();
    addFilterOption(_filters.noExpansion, tr("Hide Expansion"), QStringLiteral("chkFilterNoExpansion"));
    addFilterOption(_filters.noDefective, tr("Hide Defective"), QStringLiteral("chkFilterNoDefective"));
    addFilterOption(_filters.partialReview, tr("Hide Partial Review"), QStringLiteral("chkFilterPartialReview"));
    addFilterOption(_filters.inspectOnly, tr("Inspect Only"), QStringLiteral("chkFilterInspectOnly"));
    addSeparator();
    addFilterOption(_filters.currentOnly, tr("Current Segment Only"), QStringLiteral("chkFilterCurrentOnly"));

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
    resetBox(_tags.revisit);
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
        case Tag::Revisit: target = _tags.revisit; break;
        case Tag::Inspect: target = _tags.inspect; break;
    }

    if (!target || !target->isEnabled()) {
        return false;
    }

    target->setCheckState(target->checkState() == Qt::Checked ? Qt::Unchecked : Qt::Checked);
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
    connectToggle(_filters.revisit);
    connectToggle(_filters.noExpansion);
    connectToggle(_filters.noDefective);
    connectToggle(_filters.partialReview);
    connectToggle(_filters.hideUnapproved);
    connectToggle(_filters.inspectOnly);
    connectToggle(_filters.currentOnly);

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
        connect(_pointCollection, &VCCollection::collectionAdded, this, [this](uint64_t) {
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
    connectBox(_tags.revisit);
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
    countIfChecked(_filters.revisit);
    countIfChecked(_filters.hideUnapproved);
    countIfChecked(_filters.noExpansion);
    countIfChecked(_filters.noDefective);
    countIfChecked(_filters.partialReview);
    countIfChecked(_filters.inspectOnly);
    countIfChecked(_filters.currentOnly);

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
    const std::string username = settings.value("viewer/username", "").toString().toStdString();

    const auto selectedItems = _ui.treeWidget->selectedItems();
    for (auto* item : selectedItems) {
        if (!item) {
            continue;
        }

        const std::string id = item->data(SURFACE_ID_COLUMN, Qt::UserRole).toString().toStdString();
        QuadSurface* surface = nullptr;

        if (_opchains && _opchains->count(id) && (*_opchains)[id]) {
            surface = (*_opchains)[id]->src();
        } else if (_volumePkg) {
            auto meta = _volumePkg->getSurface(id);
            if (meta) {
                surface = meta->surface();
            }
        }

        if (!surface || !surface->meta) {
            continue;
        }

        const bool wasReviewed = surface->meta->contains("tags") && surface->meta->at("tags").contains("reviewed");
        const bool isNowReviewed = _tags.reviewed && _tags.reviewed->checkState() == Qt::Checked;
        const bool reviewedJustAdded = !wasReviewed && isNowReviewed;

        if (surface->meta->contains("tags")) {
            auto& tags = surface->meta->at("tags");
            sync_tag(tags, _tags.approved && _tags.approved->checkState() == Qt::Checked, "approved", username);
            sync_tag(tags, _tags.defective && _tags.defective->checkState() == Qt::Checked, "defective", username);
            sync_tag(tags, _tags.reviewed && _tags.reviewed->checkState() == Qt::Checked, "reviewed", username);
            sync_tag(tags, _tags.revisit && _tags.revisit->checkState() == Qt::Checked, "revisit", username);
            sync_tag(tags, _tags.inspect && _tags.inspect->checkState() == Qt::Checked, "inspect", username);
            surface->save_meta();
        } else if ((_tags.approved && _tags.approved->checkState() == Qt::Checked) ||
                   (_tags.defective && _tags.defective->checkState() == Qt::Checked) ||
                   (_tags.reviewed && _tags.reviewed->checkState() == Qt::Checked) ||
                   (_tags.revisit && _tags.revisit->checkState() == Qt::Checked) ||
                   (_tags.inspect && _tags.inspect->checkState() == Qt::Checked)) {
            (*surface->meta)["tags"] = nlohmann::json::object();
            auto& tags = (*surface->meta)["tags"];

            if (_tags.approved && _tags.approved->checkState() == Qt::Checked) {
                tags["approved"] = nlohmann::json::object();
                if (!username.empty()) {
                    tags["approved"]["user"] = username;
                }
            }
            if (_tags.defective && _tags.defective->checkState() == Qt::Checked) {
                tags["defective"] = nlohmann::json::object();
                if (!username.empty()) {
                    tags["defective"]["user"] = username;
                }
            }
            if (_tags.reviewed && _tags.reviewed->checkState() == Qt::Checked) {
                tags["reviewed"] = nlohmann::json::object();
                if (!username.empty()) {
                    tags["reviewed"]["user"] = username;
                }
            }
            if (_tags.revisit && _tags.revisit->checkState() == Qt::Checked) {
                tags["revisit"] = nlohmann::json::object();
                if (!username.empty()) {
                    tags["revisit"]["user"] = username;
                }
            }
            if (_tags.inspect && _tags.inspect->checkState() == Qt::Checked) {
                tags["inspect"] = nlohmann::json::object();
                if (!username.empty()) {
                    tags["inspect"]["user"] = username;
                }
            }

            surface->save_meta();
        }

        if (reviewedJustAdded && _volumePkg) {
            auto surfMeta = _volumePkg->getSurface(id);
            if (surfMeta) {
                for (const auto& overlapId : surfMeta->overlapping_str) {
                    auto overlapMeta = _volumePkg->getSurface(overlapId);
                    if (!overlapMeta) {
                        continue;
                    }
                    QuadSurface* overlapSurf = overlapMeta->surface();
                    if (!overlapSurf || !overlapSurf->meta) {
                        continue;
                    }

                    const bool alreadyReviewed = overlapSurf->meta->contains("tags") &&
                                                 overlapSurf->meta->at("tags").contains("reviewed");
                    if (alreadyReviewed) {
                        continue;
                    }

                    if (!overlapSurf->meta->contains("tags")) {
                        (*overlapSurf->meta)["tags"] = nlohmann::json::object();
                    }

                    auto& overlapTags = (*overlapSurf->meta)["tags"];
                    overlapTags["partial_review"] = nlohmann::json::object();
                    if (!username.empty()) {
                        overlapTags["partial_review"]["user"] = username;
                    }
                    overlapTags["partial_review"]["source"] = id;
                    overlapSurf->save_meta();
                }
            }
        }

        if (auto* treeItem = dynamic_cast<SurfaceTreeWidgetItem*>(item)) {
            updateTreeItemIcon(treeItem);
        }
    }

    applyFilters();
}

void SurfacePanelController::applyFiltersInternal()
{
    if (!_ui.treeWidget || !_volumePkg) {
        emit filtersApplied(0);
        return;
    }

    auto isChecked = [](QCheckBox* box) {
        return box && box->isChecked();
    };

    bool hasActiveFilters = isChecked(_filters.focusPoints) ||
                            isChecked(_filters.unreviewed) ||
                            isChecked(_filters.revisit) ||
                            isChecked(_filters.noExpansion) ||
                            isChecked(_filters.noDefective) ||
                            isChecked(_filters.partialReview) ||
                            isChecked(_filters.currentOnly) ||
                            isChecked(_filters.hideUnapproved) ||
                            isChecked(_filters.inspectOnly);

    auto* model = qobject_cast<QStandardItemModel*>(_filters.pointSet ? _filters.pointSet->model() : nullptr);
    if (!hasActiveFilters && model) {
        for (int row = 0; row < model->rowCount(); ++row) {
            if (model->data(model->index(row, 0), Qt::CheckStateRole) == Qt::Checked) {
                hasActiveFilters = true;
                break;
            }
        }
    }

    if (!hasActiveFilters) {
        QTreeWidgetItemIterator it(_ui.treeWidget);
        while (*it) {
            (*it)->setHidden(false);
            ++it;
        }

        std::set<std::string> intersects = {"segmentation"};
        for (const auto& id : _volumePkg->getLoadedSurfaceIDs()) {
            intersects.insert(id);
        }

        if (_viewerManager) {
            _viewerManager->forEachViewer([&intersects](CVolumeViewer* viewer) {
                if (viewer && viewer->surfName() != "segmentation") {
                    viewer->setIntersects(intersects);
                }
            });
        }

        emit filtersApplied(0);
        return;
    }

    std::set<std::string> intersects = {"segmentation"};
    POI* poi = _surfaces ? _surfaces->poi("focus") : nullptr;
    int filterCounter = 0;
    const bool currentOnly = isChecked(_filters.currentOnly);

    if (currentOnly && !_currentSurfaceId.empty() && _volumePkg->getSurface(_currentSurfaceId)) {
        intersects.insert(_currentSurfaceId);
    }

    QTreeWidgetItemIterator it(_ui.treeWidget);
    while (*it) {
        auto* item = *it;
        std::string id = item->data(SURFACE_ID_COLUMN, Qt::UserRole).toString().toStdString();

        bool show = true;
        auto surfMeta = _volumePkg->getSurface(id);

        if (surfMeta) {
            if (isChecked(_filters.focusPoints) && poi) {
                show = show && contains(*surfMeta, poi->p);
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
                        if (allMatch && !contains(*surfMeta, points)) {
                            allMatch = false;
                        }
                        if (!anyMatches && contains_any(*surfMeta, points)) {
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
                const auto* surface = surfMeta->surface();
                if (surface && surface->meta) {
                    const auto tags = vc::json_safe::tags_or_empty(surface->meta);
                    show = show && !tags.contains("reviewed");
                }
            }

            if (isChecked(_filters.revisit)) {
                const auto* surface = surfMeta->surface();
                if (surface && surface->meta) {
                    const auto tags = vc::json_safe::tags_or_empty(surface->meta);
                    show = show && tags.contains("revisit");
                } else {
                    show = false;
                }
            }

            if (isChecked(_filters.noExpansion)) {
                const auto* surface = surfMeta->surface();
                if (surface && surface->meta) {
                    const auto mode = vc::json_safe::string_or(surface->meta, "vc_gsfs_mode", std::string{});
                    show = show && (mode != "expansion");
                }
            }

            if (isChecked(_filters.noDefective)) {
                const auto* surface = surfMeta->surface();
                if (surface && surface->meta) {
                    const auto tags = vc::json_safe::tags_or_empty(surface->meta);
                    show = show && !tags.contains("defective");
                }
            }

            if (isChecked(_filters.partialReview)) {
                const auto* surface = surfMeta->surface();
                if (surface && surface->meta) {
                    const auto tags = vc::json_safe::tags_or_empty(surface->meta);
                    show = show && !tags.contains("partial_review");
                }
            }

            if (isChecked(_filters.hideUnapproved)) {
                const auto* surface = surfMeta->surface();
                if (surface && surface->meta) {
                    const auto tags = vc::json_safe::tags_or_empty(surface->meta);
                    show = show && tags.contains("approved");
                } else {
                    show = false;
                }
            }

            if (isChecked(_filters.inspectOnly)) {
                const auto* surface = surfMeta->surface();
                if (surface && surface->meta) {
                    const auto tags = vc::json_safe::tags_or_empty(surface->meta);
                    show = show && tags.contains("inspect");
                } else {
                    show = false;
                }
            }
        }

        item->setHidden(!show);

        if (show && !currentOnly && surfMeta) {
            intersects.insert(id);
        } else if (!show) {
            filterCounter++;
        }

        ++it;
    }

    if (_viewerManager) {
        _viewerManager->forEachViewer([&intersects](CVolumeViewer* viewer) {
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
    resetState(_tags.revisit);
    resetState(_tags.inspect);

    if (!surface) {
        setTagCheckboxEnabled(false, false, false, false, false);
        return;
    }

    setTagCheckboxEnabled(true, true, true, true, true);

    if (!surface->meta) {
        setTagCheckboxEnabled(false, false, true, true, true);
        return;
    }

    const auto tags = vc::json_safe::tags_or_empty(surface->meta);

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
    applyTag(_tags.revisit, "revisit");
    applyTag(_tags.inspect, "inspect");
}

void SurfacePanelController::setTagCheckboxEnabled(bool enabledApproved,
                                                   bool enabledDefective,
                                                   bool enabledReviewed,
                                                   bool enabledRevisit,
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
    if (_tags.revisit) {
        _tags.revisit->setEnabled(enabledRevisit);
    }
    if (_tags.inspect) {
        _tags.inspect->setEnabled(enabledInspect);
    }
}

OpChain* SurfacePanelController::ensureOpChainFor(const std::string& id)
{
    if (!_opchains) {
        return nullptr;
    }

    auto it = _opchains->find(id);
    if (it == _opchains->end()) {
        if (_volumePkg) {
            if (auto meta = _volumePkg->getSurface(id)) {
                (*_opchains)[id] = new OpChain(meta->surface());
            }
        }
        it = _opchains->find(id);
    }
    return it != _opchains->end() ? it->second : nullptr;
}
