#pragma once

#include <QFutureWatcher>
#include <QHash>
#include <QString>
#include <QWidget>

#include <cstdint>
#include <optional>
#include <vector>

#include "vc_object_store/objects/ObjectStoreSession.hpp"

class CState;
class MenuActionController;
class QComboBox;
class QFormLayout;
class QLabel;
class QLineEdit;
class QProgressBar;
class QPushButton;
class QStackedWidget;
class QTimer;
class QTreeWidget;
class QTreeWidgetItem;

// One line on a hosted replica's sync state ("synced 12:03:04", "2 pending", ...).
QString hubStatusSummary(const vc::objects::HubSessionStatus& status);

// The Project tab: an overview of the open project. Its store and the stores
// attached beside it (paths, access, hosting, object counts) and which store
// each kind is kept in; its volumes, normal grids and Lasagna datasets (where
// they are, who sees them, their disk cache); and the chunk and surface caches'
// use against their limits. Attaching and detaching goes through the File
// menu's actions; store routes and sync through CState and the session.
class ProjectWorkspace : public QWidget
{
    Q_OBJECT

public:
    ProjectWorkspace(CState* state, MenuActionController* menu, QWidget* parent = nullptr);
    ~ProjectWorkspace() override;

    // Re-read the project now when shown, else when next shown.
    void refresh();

signals:
    // View this project volume in the main workspace.
    void volumeActivationRequested(const QString& volumeId);
    void projectRenamed();
    // A hosted store sync was started from the tab.
    void hubSyncRequested();

protected:
    void showEvent(QShowEvent* event) override;
    void hideEvent(QHideEvent* event) override;

private:
    void rebuild();
    void rebuildProject();
    void rebuildStores();
    void rebuildRoutes();
    void rebuildResources();
    void updateCaches();
    void updateResourceButtons();
    void updateStoreButtons();
    void scheduleRefresh();
    void measureVolumeCaches();
    std::optional<vc::objects::AttachmentIdentity> selectedStore() const;
    QString storeDisplayName(const vc::objects::AttachmentIdentity& attachment) const;

    CState* _state;
    MenuActionController* _menu;
    bool _dirty = true;
    QTimer* _refreshTimer = nullptr;
    QTimer* _cacheTimer = nullptr;

    QStackedWidget* _pages = nullptr;

    // Project
    QLineEdit* _name = nullptr;
    QLabel* _storePath = nullptr;
    QLabel* _directory = nullptr;
    QLabel* _author = nullptr;
    QLabel* _access = nullptr;
    QLabel* _hosting = nullptr;
    QLabel* _sample = nullptr;
    QPushButton* _resetOpenData = nullptr;

    // Stores
    QTreeWidget* _stores = nullptr;
    QPushButton* _detachStore = nullptr;
    QPushButton* _syncStores = nullptr;
    QPushButton* _revealStore = nullptr;
    QFormLayout* _routesForm = nullptr;
    std::vector<QComboBox*> _routes;  // CState::routableKinds() order

    // Volumes, normal grids, Lasagna datasets
    QTreeWidget* _resources = nullptr;
    QPushButton* _detachResource = nullptr;
    QPushButton* _viewVolume = nullptr;
    QPushButton* _revealResource = nullptr;
    QPushButton* _measureCaches = nullptr;
    QHash<QString, quint64> _cacheBytes;  // by remote volume cache directory
    QFutureWatcher<QHash<QString, quint64>>* _measure = nullptr;

    // Caches
    QProgressBar* _ramBar = nullptr;
    QLabel* _ramText = nullptr;
    QProgressBar* _diskBar = nullptr;
    QLabel* _diskText = nullptr;
    QLabel* _diskPath = nullptr;
    QProgressBar* _surfaceBar = nullptr;
    QLabel* _surfaceText = nullptr;
};
