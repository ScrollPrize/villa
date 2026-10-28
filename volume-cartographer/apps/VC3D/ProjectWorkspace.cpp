#include "ProjectWorkspace.hpp"

#include "CState.hpp"
#include "MenuActionController.hpp"
#include "OpenDataSampleProject.hpp"
#include "VCSettings.hpp"

#include "vc/core/render/ChunkCache.hpp"
#include "vc/core/render/DecodedChunkCacheBudget.hpp"
#include "vc/core/render/PersistentZarrCacheBudget.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc_object_store/objects/SurfaceTileCache.hpp"

#include <QApplication>
#include <QClipboard>
#include <QComboBox>
#include <QDateTime>
#include <QDesktopServices>
#include <QFileInfo>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QLineEdit>
#include <QLocale>
#include <QMessageBox>
#include <QProgressBar>
#include <QPushButton>
#include <QScrollArea>
#include <QSignalBlocker>
#include <QStackedWidget>
#include <QTimer>
#include <QTreeWidget>
#include <QUrl>
#include <QVBoxLayout>
#include <QtConcurrent/QtConcurrent>

#include <algorithm>
#include <filesystem>
#include <map>

namespace {

enum StoreColumn { StoreName, StoreRole, StoreAccess, StoreHosting, StoreSurfaces, StoreFibers,
                   StorePointCollections, StoreSize, StorePath, StoreColumns };
enum ResourceColumn { ResName, ResLocation, ResVisibility, ResTags, ResShape, ResVoxel, ResLevels,
                      ResType, ResCache, ResColumns };
// Item data: what a resource row is.
constexpr int kLocationRole = Qt::UserRole;      // project entry location (detachable rows)
constexpr int kVolumeIdRole = Qt::UserRole + 1;  // volume id (viewable rows)
constexpr int kRevealRole = Qt::UserRole + 2;    // local path or URL
constexpr int kStoreIndexRole = Qt::UserRole;

QString bytes(quint64 n)
{
    return QLocale().formattedDataSize(qint64(n), 1, QLocale::DataSizeIecFormat);
}

QString kindTitle(const std::string& kind)
{
    if (kind == "surface") return QObject::tr("Surfaces");
    if (kind == "fiber") return QObject::tr("Fibers");
    if (kind == "point_collection") return QObject::tr("Point collections");
    return QString::fromStdString(kind);
}

QString trimmedLocation(QString location)
{
    while (location.size() > 1 && location.endsWith(QLatin1Char('/'))) location.chop(1);
    return location;
}

// Where a volume was loaded from: its URL, or its local path.
QString volumeLocation(const Volume& volume)
{
    return trimmedLocation(QString::fromStdString(volume.isRemote() ? volume.remoteUrl() : volume.path().string()));
}

QString dtypeName(vc::render::ChunkDtype dtype)
{
    switch (dtype) {
    case vc::render::ChunkDtype::UInt8: return QStringLiteral("uint8");
    case vc::render::ChunkDtype::UInt16: return QStringLiteral("uint16");
    case vc::render::ChunkDtype::Float32: return QStringLiteral("float32");
    case vc::render::ChunkDtype::Opaque: return QStringLiteral("opaque");
    }
    return {};
}

void setBar(QProgressBar* bar, QLabel* text, quint64 used, std::optional<quint64> maximum, const QString& extra = {})
{
    bar->setRange(0, 1000);
    bar->setValue(maximum && *maximum ? int(std::min<quint64>(1000, used * 1000 / *maximum)) : 0);
    bar->setEnabled(maximum.has_value());
    QString line = maximum ? QObject::tr("%1 of %2").arg(bytes(used), bytes(*maximum))
                           : QObject::tr("%1 (no limit)").arg(bytes(used));
    if (!extra.isEmpty()) line += QStringLiteral(" · ") + extra;
    text->setText(line);
}

QPushButton* button(const QString& text, QWidget* parent, QHBoxLayout* row)
{
    auto* b = new QPushButton(text, parent);
    row->addWidget(b);
    return b;
}

void reveal(const QString& location)
{
    if (location.isEmpty()) return;
    if (vc::project::isLocationRemote(location.toStdString())) {
        QApplication::clipboard()->setText(location);
        return;
    }
    const QFileInfo info(location);
    QDesktopServices::openUrl(QUrl::fromLocalFile(info.isDir() ? info.absoluteFilePath() : info.absolutePath()));
}

} // namespace

QString hubStatusSummary(const vc::objects::HubSessionStatus& status)
{
    if (!status.conflicts.empty())
        return QObject::tr("%n conflict(s)", nullptr, int(status.conflicts.size()));
    if (!status.error.empty())
        return QObject::tr("offline (%1)").arg(QString::fromStdString(status.error));
    if (!status.pending.empty())
        return QObject::tr("%n pending", nullptr, int(status.pending.size()));
    if (status.syncedAtUs)
        return QObject::tr("synced %1").arg(
            QDateTime::fromMSecsSinceEpoch(*status.syncedAtUs / 1000).toString(QStringLiteral("HH:mm:ss")));
    return QObject::tr("connecting");
}

ProjectWorkspace::ProjectWorkspace(CState* state, MenuActionController* menu, QWidget* parent)
    : QWidget(parent), _state(state), _menu(menu)
{
    setObjectName(QStringLiteral("projectWorkspace"));
    auto* outer = new QVBoxLayout(this);
    outer->setContentsMargins(0, 0, 0, 0);
    _pages = new QStackedWidget(this);
    outer->addWidget(_pages);

    // No project open.
    auto* empty = new QWidget(_pages);
    auto* emptyLayout = new QVBoxLayout(empty);
    emptyLayout->addStretch(1);
    auto* emptyText = new QLabel(tr("No project is open.\n\nA project is an object store (.vcobj): it lists the "
                                    "volumes and other data you work with and holds your surfaces, fibers "
                                    "and point collections."), empty);
    emptyText->setAlignment(Qt::AlignCenter);
    emptyText->setWordWrap(true);
    emptyLayout->addWidget(emptyText);
    auto* emptyButtons = new QHBoxLayout;
    emptyButtons->addStretch(1);
    connect(button(tr("New Project..."), empty, emptyButtons), &QPushButton::clicked, _menu, &MenuActionController::newProject);
    connect(button(tr("Open Project..."), empty, emptyButtons), &QPushButton::clicked, _menu, &MenuActionController::openVolpkg);
    connect(button(tr("Open Data Catalog..."), empty, emptyButtons), &QPushButton::clicked, _menu,
            &MenuActionController::showOpenDataCatalog);
    emptyButtons->addStretch(1);
    emptyLayout->addLayout(emptyButtons);
    emptyLayout->addStretch(2);
    _pages->addWidget(empty);

    // The overview.
    auto* scroll = new QScrollArea(_pages);
    scroll->setWidgetResizable(true);
    scroll->setFrameShape(QFrame::NoFrame);
    auto* page = new QWidget(scroll);
    auto* layout = new QVBoxLayout(page);
    scroll->setWidget(page);
    _pages->addWidget(scroll);

    auto* top = new QHBoxLayout;
    layout->addLayout(top);

    // Project
    auto* projectBox = new QGroupBox(tr("Project"), page);
    auto* projectLayout = new QVBoxLayout(projectBox);
    auto* projectForm = new QFormLayout;
    projectLayout->addLayout(projectForm);
    _name = new QLineEdit(projectBox);
    _name->setObjectName(QStringLiteral("projectName"));
    connect(_name, &QLineEdit::editingFinished, this, [this]() {
        const auto pkg = _state->vpkg();
        const auto name = _name->text().trimmed().toStdString();
        if (!pkg || name.empty() || name == pkg->name()) return;
        try {
            pkg->setName(name);
            emit projectRenamed();
        } catch (const std::exception& error) {
            QMessageBox::warning(this, tr("Rename Project"), QString::fromUtf8(error.what()));
            _name->setText(QString::fromStdString(pkg->name()));
        }
    });
    projectForm->addRow(tr("Name:"), _name);
    const auto addLabel = [&](const QString& title) {
        auto* label = new QLabel(projectBox);
        label->setTextInteractionFlags(Qt::TextSelectableByMouse);
        label->setWordWrap(true);
        projectForm->addRow(title, label);
        return label;
    };
    _storePath = addLabel(tr("Store:"));
    _directory = addLabel(tr("Folder:"));
    _access = addLabel(tr("Access:"));
    _hosting = addLabel(tr("Hosting:"));
    _author = addLabel(tr("You are:"));
    _sample = addLabel(tr("Open Data:"));
    auto* projectButtons = new QHBoxLayout;
    connect(button(tr("Save As..."), projectBox, projectButtons), &QPushButton::clicked, _menu,
            &MenuActionController::saveProjectAs);
    _resetOpenData = button(tr("Reset to Open Data..."), projectBox, projectButtons);
    connect(_resetOpenData, &QPushButton::clicked, _menu, &MenuActionController::resetOpenDataProject);
    connect(button(tr("Open Folder"), projectBox, projectButtons), &QPushButton::clicked, this, [this]() {
        reveal(_state->projectDirectory());
    });
    projectButtons->addStretch(1);
    projectLayout->addLayout(projectButtons);
    projectLayout->addStretch(1);
    top->addWidget(projectBox, 1);

    // Caches
    auto* cacheBox = new QGroupBox(tr("Caches"), page);
    auto* cacheLayout = new QVBoxLayout(cacheBox);
    auto* cacheForm = new QFormLayout;
    cacheLayout->addLayout(cacheForm);
    const auto addBar = [&](const QString& title, const QString& tip, QProgressBar*& bar, QLabel*& text) {
        auto* cell = new QWidget(cacheBox);
        auto* cellLayout = new QVBoxLayout(cell);
        cellLayout->setContentsMargins(0, 0, 0, 0);
        cellLayout->setSpacing(2);
        bar = new QProgressBar(cell);
        bar->setTextVisible(false);
        bar->setMaximumHeight(10);
        text = new QLabel(cell);
        cellLayout->addWidget(bar);
        cellLayout->addWidget(text);
        auto* label = new QLabel(title, cacheBox);
        label->setToolTip(tip);
        cell->setToolTip(tip);
        cacheForm->addRow(label, cell);
    };
    addBar(tr("Volume chunks (RAM):"), tr("Decoded volume chunks held in memory, shared by every viewer."),
           _ramBar, _ramText);
    addBar(tr("Volume chunks (disk):"),
           tr("Chunks of remote volumes downloaded to disk. Older chunks are removed past the limit, "
              "or when the disk runs low."),
           _diskBar, _diskText);
    _diskPath = new QLabel(cacheBox);
    _diskPath->setTextInteractionFlags(Qt::TextSelectableByMouse);
    _diskPath->setWordWrap(true);
    cacheForm->addRow(tr("Disk cache folder:"), _diskPath);
    addBar(tr("Surface geometry (RAM):"), tr("Decoded surface tiles kept for every open store."),
           _surfaceBar, _surfaceText);
    auto* cacheButtons = new QHBoxLayout;
    connect(button(tr("Cache Settings..."), cacheBox, cacheButtons), &QPushButton::clicked, _menu,
            &MenuActionController::showSettingsDialog);
    connect(button(tr("Open Cache Folder"), cacheBox, cacheButtons), &QPushButton::clicked, this, []() {
        reveal(vc3d::remoteCachePath());
    });
    cacheButtons->addStretch(1);
    cacheLayout->addLayout(cacheButtons);
    cacheLayout->addStretch(1);
    top->addWidget(cacheBox, 1);

    // Stores
    auto* storeBox = new QGroupBox(tr("Stores"), page);
    auto* storeLayout = new QVBoxLayout(storeBox);
    _stores = new QTreeWidget(storeBox);
    _stores->setObjectName(QStringLiteral("projectStores"));
    _stores->setRootIsDecorated(false);
    _stores->setColumnCount(StoreColumns);
    _stores->setHeaderLabels({tr("Store"), tr("Role"), tr("Access"), tr("Hosting"), tr("Surfaces"), tr("Fibers"),
                              tr("Point collections"), tr("Size"), tr("Path")});
    _stores->header()->setStretchLastSection(true);
    connect(_stores, &QTreeWidget::itemSelectionChanged, this, &ProjectWorkspace::updateStoreButtons);
    storeLayout->addWidget(_stores);
    auto* storeButtons = new QHBoxLayout;
    connect(button(tr("Attach Store..."), storeBox, storeButtons), &QPushButton::clicked, this,
            [this]() { _menu->attachStore(false); });
    connect(button(tr("Attach Read-only..."), storeBox, storeButtons), &QPushButton::clicked, this,
            [this]() { _menu->attachStore(true); });
    connect(button(tr("Clone Hosted Store..."), storeBox, storeButtons), &QPushButton::clicked, this,
            [this]() { _menu->cloneHostedStore(false); });
    connect(button(tr("Import from Folder..."), storeBox, storeButtons), &QPushButton::clicked, _menu,
            &MenuActionController::importIntoObjectStore);
    _syncStores = button(tr("Sync Hosted Now"), storeBox, storeButtons);
    connect(_syncStores, &QPushButton::clicked, this, [this]() {
        const auto session = _state->objectStoreSession();
        std::vector<vc::objects::AttachmentIdentity> stores = _state->attachedStores();
        if (const auto project = _state->projectAttachment()) stores.insert(stores.begin(), *project);
        for (const auto& store : stores)
            if (session->hubStatus(store)) session->syncHub(store);
        emit hubSyncRequested();
    });
    _revealStore = button(tr("Open Folder"), storeBox, storeButtons);
    connect(_revealStore, &QPushButton::clicked, this, [this]() {
        if (const auto store = selectedStore())
            reveal(QString::fromStdString(_state->objectStoreSession()->path(*store).string()));
    });
    _detachStore = button(tr("Detach"), storeBox, storeButtons);
    connect(_detachStore, &QPushButton::clicked, this, [this]() {
        const auto store = selectedStore();
        if (!store || store == _state->projectAttachment()) return;
        if (QMessageBox::question(this, tr("Detach Store"),
                tr("Detach %1 from this project?\n\nThe store file is kept; objects routed to it are "
                   "shown from the project's store again.").arg(storeDisplayName(*store)),
                QMessageBox::Yes | QMessageBox::No, QMessageBox::No) != QMessageBox::Yes)
            return;
        try { _state->detachStore(*store); }
        catch (const std::exception& error) {
            QMessageBox::critical(this, tr("Cannot detach store"), QString::fromUtf8(error.what()));
        }
    });
    storeButtons->addStretch(1);
    storeLayout->addLayout(storeButtons);
    auto* routesTitle = new QLabel(tr("Where new and listed objects are kept:"), storeBox);
    storeLayout->addWidget(routesTitle);
    _routesForm = new QFormLayout;
    _routesForm->setFieldGrowthPolicy(QFormLayout::FieldsStayAtSizeHint);
    for (const auto& kind : CState::routableKinds()) {
        auto* combo = new QComboBox(storeBox);
        combo->setObjectName(QStringLiteral("projectRoute_%1").arg(QString::fromStdString(kind)));
        combo->setSizeAdjustPolicy(QComboBox::AdjustToContents);
        connect(combo, &QComboBox::activated, this, [this, kind, combo](int index) {
            std::optional<vc::objects::AttachmentIdentity> store;
            if (index > 0) {
                const auto attached = _state->attachedStores();
                const int at = combo->itemData(index).toInt();
                if (at < 0 || at >= int(attached.size())) return;
                store = attached[size_t(at)];
            }
            try { _state->setStoreFor(kind, store); }
            catch (const std::exception& error) {
                QMessageBox::warning(this, tr("Stores"), QString::fromUtf8(error.what()));
                rebuildRoutes();
            }
        });
        _routes.push_back(combo);
        _routesForm->addRow(kindTitle(kind) + QLatin1Char(':'), combo);
    }
    storeLayout->addLayout(_routesForm);
    layout->addWidget(storeBox);

    // Data
    auto* dataBox = new QGroupBox(tr("Data"), page);
    auto* dataLayout = new QVBoxLayout(dataBox);
    _resources = new QTreeWidget(dataBox);
    _resources->setObjectName(QStringLiteral("projectResources"));
    _resources->setColumnCount(ResColumns);
    _resources->setHeaderLabels({tr("Name"), tr("Location"), tr("Seen by"), tr("Tags"), tr("Size (x × y × z)"),
                                 tr("Voxel"), tr("Levels"), tr("Type"), tr("Disk cache")});
    _resources->setMinimumHeight(240);
    _resources->setUniformRowHeights(true);
    connect(_resources, &QTreeWidget::itemSelectionChanged, this, &ProjectWorkspace::updateResourceButtons);
    connect(_resources, &QTreeWidget::itemDoubleClicked, this, [this](QTreeWidgetItem* item) {
        const auto id = item ? item->data(0, kVolumeIdRole).toString() : QString();
        if (!id.isEmpty()) emit volumeActivationRequested(id);
    });
    dataLayout->addWidget(_resources, 1);
    auto* attachButtons = new QHBoxLayout;
    connect(button(tr("Attach Volume..."), dataBox, attachButtons), &QPushButton::clicked, _menu,
            &MenuActionController::attachVolume);
    connect(button(tr("Attach Remote Zarr..."), dataBox, attachButtons), &QPushButton::clicked, _menu,
            &MenuActionController::attachRemoteZarr);
    connect(button(tr("Attach Normal Grid..."), dataBox, attachButtons), &QPushButton::clicked, _menu,
            &MenuActionController::attachNormalGrid);
    connect(button(tr("Attach Lasagna..."), dataBox, attachButtons), &QPushButton::clicked, _menu,
            &MenuActionController::attachLasagnaManifest);
    connect(button(tr("Attach Remote Lasagna..."), dataBox, attachButtons), &QPushButton::clicked, _menu,
            &MenuActionController::attachRemoteLasagnaManifest);
    connect(button(tr("Open Data Catalog..."), dataBox, attachButtons), &QPushButton::clicked, _menu,
            &MenuActionController::showOpenDataCatalog);
    attachButtons->addStretch(1);
    dataLayout->addLayout(attachButtons);
    auto* itemButtons = new QHBoxLayout;
    _viewVolume = button(tr("View Volume"), dataBox, itemButtons);
    connect(_viewVolume, &QPushButton::clicked, this, [this]() {
        if (auto* item = _resources->currentItem()) {
            const auto id = item->data(0, kVolumeIdRole).toString();
            if (!id.isEmpty()) emit volumeActivationRequested(id);
        }
    });
    _revealResource = button(tr("Open Folder / Copy URL"), dataBox, itemButtons);
    connect(_revealResource, &QPushButton::clicked, this, [this]() {
        if (auto* item = _resources->currentItem()) reveal(item->data(0, kRevealRole).toString());
    });
    _detachResource = button(tr("Detach"), dataBox, itemButtons);
    connect(_detachResource, &QPushButton::clicked, this, [this]() {
        if (auto* item = _resources->currentItem()) {
            const auto location = item->data(0, kLocationRole).toString();
            if (!location.isEmpty()) _menu->detachEntryAt(location);
        }
    });
    _measureCaches = button(tr("Measure Disk Cache"), dataBox, itemButtons);
    _measureCaches->setToolTip(tr("Add up the downloaded chunks of each remote volume (reads the cache folders)."));
    connect(_measureCaches, &QPushButton::clicked, this, &ProjectWorkspace::measureVolumeCaches);
    itemButtons->addStretch(1);
    dataLayout->addLayout(itemButtons);
    layout->addWidget(dataBox, 1);

    _measure = new QFutureWatcher<QHash<QString, quint64>>(this);
    connect(_measure, &QFutureWatcher<QHash<QString, quint64>>::finished, this, [this]() {
        for (auto [path, size] : _measure->result().asKeyValueRange()) _cacheBytes[path] = size;
        _measureCaches->setEnabled(true);
        _measureCaches->setText(tr("Measure Disk Cache"));
        rebuildResources();
    });

    // Many store events arrive together (an import, a sync); refresh once.
    _refreshTimer = new QTimer(this);
    _refreshTimer->setSingleShot(true);
    _refreshTimer->setInterval(250);
    connect(_refreshTimer, &QTimer::timeout, this, &ProjectWorkspace::refresh);
    _cacheTimer = new QTimer(this);
    _cacheTimer->setInterval(2000);
    connect(_cacheTimer, &QTimer::timeout, this, &ProjectWorkspace::updateCaches);

    connect(_state, &CState::vpkgChanged, this, &ProjectWorkspace::scheduleRefresh);
    connect(_state, &CState::volumeChanged, this, &ProjectWorkspace::scheduleRefresh);
    connect(_state, &CState::storeRoutesChanged, this, &ProjectWorkspace::scheduleRefresh);
    connect(_state, &CState::objectStoreChanged, this, [this](const vc::objects::StoreSessionEvent& event) {
        if (event.kind != vc::objects::StoreSessionEvent::Kind::Draft) scheduleRefresh();
    });
}

ProjectWorkspace::~ProjectWorkspace()
{
    if (_measure) _measure->waitForFinished();
}

void ProjectWorkspace::scheduleRefresh()
{
    if (isVisible()) _refreshTimer->start();
    else _dirty = true;
}

void ProjectWorkspace::refresh()
{
    if (!isVisible()) {
        _dirty = true;
        return;
    }
    _dirty = false;
    rebuild();
}

void ProjectWorkspace::showEvent(QShowEvent* event)
{
    QWidget::showEvent(event);
    if (_dirty) refresh();
    updateCaches();
    _cacheTimer->start();
}

void ProjectWorkspace::hideEvent(QHideEvent* event)
{
    QWidget::hideEvent(event);
    _cacheTimer->stop();
}

void ProjectWorkspace::rebuild()
{
    const bool open = _state->vpkg() && _state->projectAttachment();
    _pages->setCurrentIndex(open ? 1 : 0);
    if (!open) return;
    rebuildProject();
    rebuildStores();
    rebuildRoutes();
    rebuildResources();
    updateCaches();
}

QString ProjectWorkspace::storeDisplayName(const vc::objects::AttachmentIdentity& attachment) const
{
    return QString::fromStdString(_state->objectStoreSession()->path(attachment).filename().string());
}

void ProjectWorkspace::rebuildProject()
{
    const auto pkg = _state->vpkg();
    const auto project = *_state->projectAttachment();
    const auto session = _state->objectStoreSession();
    if (!_name->hasFocus()) _name->setText(QString::fromStdString(pkg->name()));
    _name->setReadOnly(!pkg->writable());
    _storePath->setText(QString::fromStdString(pkg->path().string()));
    _directory->setText(_state->projectDirectory());
    _access->setText(pkg->writable() ? tr("Read-write") : tr("Read-only"));
    const auto hub = session->hubStatus(project);
    _hosting->setText(hub ? tr("Hosted replica, %1").arg(hubStatusSummary(*hub)) : tr("Local store"));
    _author->setText(CState::storeAuthor());
    const auto sample = vc3d::opendata::openDataProjectSample(*session, project);
    _sample->setText(sample ? tr("Made from sample %1").arg(QString::fromStdString(*sample)) : tr("—"));
    _resetOpenData->setVisible(sample.has_value());
    _resetOpenData->setEnabled(sample && !hub);
}

void ProjectWorkspace::rebuildStores()
{
    const auto session = _state->objectStoreSession();
    const auto project = *_state->projectAttachment();
    const auto attached = _state->attachedStores();
    const auto selected = selectedStore();
    QSignalBlocker block(_stores);
    _stores->clear();
    std::vector<vc::objects::AttachmentIdentity> stores{project};
    stores.insert(stores.end(), attached.begin(), attached.end());
    for (size_t i = 0; i < stores.size(); ++i) {
        const auto& store = stores[i];
        auto* item = new QTreeWidgetItem(_stores);
        item->setData(0, kStoreIndexRole, int(i) - 1);  // -1: the project's store
        const auto path = session->path(store);
        item->setText(StoreName, QString::fromStdString(path.filename().string()));
        item->setToolTip(StoreName, tr("Store UUID %1").arg(QString::fromStdString(store.storeUuid)));
        item->setText(StoreRole, i == 0 ? tr("Project") : tr("Attached"));
        item->setText(StoreAccess, session->writable(store) ? tr("Read-write") : tr("Read-only"));
        const auto hub = session->hubStatus(store);
        item->setText(StoreHosting, hub ? hubStatusSummary(*hub) : tr("Local"));
        std::map<std::string, int64_t> counts;
        try {
            session->read(store, [&](const vc::objects::ObjectStore& s, const vc::objects::ReadTxn& txn,
                                     const std::atomic_bool&) {
                for (const auto& kind : CState::routableKinds()) counts[kind] = s.objectCount(txn, kind);
            });
        } catch (const std::exception& error) {
            item->setToolTip(StoreSurfaces, QString::fromUtf8(error.what()));
        }
        const auto count = [&](const char* kind) {
            const auto it = counts.find(kind);
            return it == counts.end() ? QStringLiteral("?") : QLocale().toString(qlonglong(it->second));
        };
        item->setText(StoreSurfaces, count("surface"));
        item->setText(StoreFibers, count("fiber"));
        item->setText(StorePointCollections, count("point_collection"));
        for (int c : {StoreSurfaces, StoreFibers, StorePointCollections, StoreSize})
            item->setTextAlignment(c, Qt::AlignRight | Qt::AlignVCenter);
        std::error_code ec;
        const auto size = std::filesystem::file_size(path, ec);
        item->setText(StoreSize, ec ? QString() : bytes(size));
        item->setText(StorePath, QString::fromStdString(path.string()));
        item->setToolTip(StorePath, item->text(StorePath));
        if (selected == store) _stores->setCurrentItem(item);
    }
    for (int c = 0; c < StorePath; ++c) _stores->resizeColumnToContents(c);
    // As tall as its rows, so the data list below gets the room.
    const int rowHeight = _stores->topLevelItemCount() ? _stores->visualItemRect(_stores->topLevelItem(0)).height() : 0;
    _stores->setFixedHeight(_stores->header()->sizeHint().height() + 2 * _stores->frameWidth() +
                            std::max(1, _stores->topLevelItemCount()) * std::max(rowHeight, fontMetrics().height() + 4) + 4);
    updateStoreButtons();
}

void ProjectWorkspace::rebuildRoutes()
{
    const auto project = *_state->projectAttachment();
    const auto attached = _state->attachedStores();
    const auto& kinds = CState::routableKinds();
    for (size_t k = 0; k < kinds.size(); ++k) {
        auto* combo = _routes[k];
        QSignalBlocker block(combo);
        combo->clear();
        combo->addItem(tr("Project (%1)").arg(storeDisplayName(project)), -1);
        const auto current = _state->storeFor(kinds[k]);
        for (size_t i = 0; i < attached.size(); ++i) {
            QString name = storeDisplayName(attached[i]);
            if (!_state->objectStoreSession()->writable(attached[i])) name += tr(" [read-only]");
            combo->addItem(name, int(i));
            if (current == attached[i]) combo->setCurrentIndex(combo->count() - 1);
        }
        combo->setEnabled(!attached.empty());
    }
}

void ProjectWorkspace::rebuildResources()
{
    const auto pkg = _state->vpkg();
    const auto currentId = QString::fromStdString(_state->currentVolumeId());
    const auto selectedLocation = _resources->currentItem() ? _resources->currentItem()->data(0, kLocationRole) : QVariant();
    const auto selectedVolume = _resources->currentItem() ? _resources->currentItem()->data(0, kVolumeIdRole) : QVariant();
    QSignalBlocker block(_resources);
    _resources->clear();

    const auto resolved = [&](const std::string& location) {
        return trimmedLocation(vc::project::isLocationRemote(location)
            ? QString::fromStdString(location)
            : QString::fromStdString(vc::project::resolveLocalPath(location, pkg->directory()).string()));
    };
    const auto visibility = [&](const std::string& location) {
        return pkg->entryOwner(location).empty() ? tr("Everyone") : tr("Only you");
    };
    const auto entryItem = [&](QTreeWidgetItem* parent, const vc::project::Entry& entry) {
        auto* item = new QTreeWidgetItem(parent);
        const auto location = QString::fromStdString(entry.location);
        item->setText(ResName, QFileInfo(trimmedLocation(location)).fileName());
        item->setText(ResLocation, location);
        item->setToolTip(ResLocation, resolved(entry.location));
        item->setText(ResVisibility, visibility(entry.location));
        item->setToolTip(ResVisibility, pkg->entryOwner(entry.location).empty()
            ? tr("A remote location: every user of this project sees it.")
            : tr("A path on this computer: only you see it in this project."));
        QStringList tags;
        for (const auto& tag : entry.tags) tags << QString::fromStdString(tag);
        item->setText(ResTags, tags.join(QStringLiteral(", ")));
        item->setData(0, kLocationRole, location);
        item->setData(0, kRevealRole, resolved(entry.location));
        if (location == selectedLocation.toString() && selectedVolume.toString().isEmpty())
            _resources->setCurrentItem(item);
        return item;
    };
    const auto describeVolume = [&](QTreeWidgetItem* item, const std::string& id, const Volume& volume) {
        const auto qid = QString::fromStdString(id);
        const auto name = QString::fromStdString(volume.name());
        item->setText(ResName, name.isEmpty() || name == qid ? qid : QStringLiteral("%1 (%2)").arg(name, qid));
        const auto shape = volume.shapeXyz();
        item->setText(ResShape, QStringLiteral("%1 × %2 × %3").arg(shape[0]).arg(shape[1]).arg(shape[2]));
        if (volume.voxelSize() > 0) item->setText(ResVoxel, tr("%1 µm").arg(volume.voxelSize()));
        item->setText(ResLevels, QString::number(volume.numScales()));
        item->setText(ResType, dtypeName(volume.dtype()));
        if (volume.isRemote()) {
            const auto cache = QString::fromStdString(volume.remotePersistentCachePath().string());
            const auto measured = _cacheBytes.find(cache);
            item->setText(ResCache, measured != _cacheBytes.end() ? bytes(*measured) : tr("not measured"));
            item->setToolTip(ResCache, cache);
        } else {
            item->setText(ResCache, tr("local"));
        }
        item->setData(0, kVolumeIdRole, qid);
        item->setData(0, kRevealRole, volumeLocation(volume));
        if (qid == currentId) {
            for (int c = 0; c < ResColumns; ++c) {
                auto font = item->font(c);
                font.setBold(true);
                item->setFont(c, font);
            }
            item->setToolTip(ResName, tr("Open in the viewers"));
        }
        if (qid == selectedVolume.toString()) _resources->setCurrentItem(item);
    };
    const auto group = [&](const QString& title, int count) {
        auto* item = new QTreeWidgetItem(_resources);
        item->setText(ResName, QStringLiteral("%1 (%2)").arg(title).arg(count));
        item->setFirstColumnSpanned(true);
        auto font = item->font(0);
        font.setBold(true);
        item->setFont(0, font);
        item->setFlags(Qt::ItemIsEnabled);
        return item;
    };

    // Volumes under the entry they were loaded from (the longest matching
    // location: an entry may be a folder of zarrs).
    const auto& volumeEntries = pkg->volumeEntries();
    std::vector<QString> entryLocations;
    for (const auto& entry : volumeEntries) entryLocations.push_back(resolved(entry.location));
    std::vector<std::vector<std::string>> volumesOf(volumeEntries.size());
    std::vector<std::string> unlisted;
    for (const auto& id : pkg->volumeIDs()) {
        const auto volume = pkg->volume(id);
        if (!volume) continue;
        const auto location = volumeLocation(*volume);
        int best = -1;
        for (size_t e = 0; e < entryLocations.size(); ++e) {
            const auto& at = entryLocations[e];
            if ((location == at || location.startsWith(at + QLatin1Char('/'))) &&
                (best < 0 || at.size() > entryLocations[size_t(best)].size()))
                best = int(e);
        }
        if (best < 0) unlisted.push_back(id);
        else volumesOf[size_t(best)].push_back(id);
    }
    auto* volumes = group(tr("Volumes"), int(pkg->numberOfVolumes()));
    for (size_t e = 0; e < volumeEntries.size(); ++e) {
        auto* item = entryItem(volumes, volumeEntries[e]);
        const auto& ids = volumesOf[e];
        // A zarr entry is its volume's row; a folder lists its volumes.
        if (ids.size() == 1 && volumeLocation(*pkg->volume(ids[0])) == entryLocations[e]) {
            describeVolume(item, ids[0], *pkg->volume(ids[0]));
            continue;
        }
        if (ids.empty())
            item->setText(ResShape, pkg->entryResolutionDeferred() ? tr("loading...") : tr("not loaded"));
        for (const auto& id : ids) describeVolume(new QTreeWidgetItem(item), id, *pkg->volume(id));
        item->setExpanded(true);
    }
    // Volumes not backed by a volume entry (e.g. a Lasagna dataset's).
    for (const auto& id : unlisted) {
        auto* item = new QTreeWidgetItem(volumes);
        const auto volume = pkg->volume(id);
        describeVolume(item, id, *volume);
        item->setText(ResLocation, volumeLocation(*volume));
        item->setText(ResVisibility, tr("via dataset"));
    }
    volumes->setExpanded(true);

    const auto& grids = pkg->normalGridEntries();
    auto* gridGroup = group(tr("Normal grids"), int(grids.size()));
    for (const auto& entry : grids) entryItem(gridGroup, entry)->setText(ResType, tr("normal grid"));
    gridGroup->setExpanded(true);

    const auto& lasagna = pkg->allLasagnaDatasetEntries();
    auto* lasagnaGroup = group(tr("Lasagna datasets"), int(lasagna.size()));
    for (const auto& entry : lasagna) {
        auto* item = entryItem(lasagnaGroup, entry);
        const bool fiber = vc::project::isFiberLasagnaEntry(entry);
        const bool selected = entry.location == (fiber ? pkg->selectedFiberInferenceDataset()
                                                       : pkg->selectedLasagnaDataset());
        QString type = fiber ? tr("fiber inference") : tr("Lasagna");
        if (selected) type += tr(" (selected)");
        item->setText(ResType, type);
    }
    lasagnaGroup->setExpanded(true);

    for (int c = 0; c < ResColumns; ++c) _resources->resizeColumnToContents(c);
    _resources->setColumnWidth(ResLocation, std::clamp(_resources->columnWidth(ResLocation), 160, 520));
    updateResourceButtons();
}

void ProjectWorkspace::updateCaches()
{
    if (!isVisible()) return;
    if (const auto budget = vc::render::processChunkCacheService()->decodedByteBudget()) {
        const auto stats = budget->stats();
        setBar(_ramBar, _ramText, stats.decodedBytes, stats.maximumBytes);
    }
    const auto root = vc3d::remoteCachePathFs();
    _diskPath->setText(QString::fromStdString(root.string()));
    if (const auto budget = vc::render::PersistentZarrCacheBudget::findForPath(root)) {
        const auto stats = budget->stats();
        // Free space is measured with the first trim; until then it reads 0.
        QStringList extra;
        if (stats.freeBytes || stats.lowSpace) extra << tr("%1 free on disk").arg(bytes(stats.freeBytes));
        if (stats.minimumFreeBytes) extra << tr("keeps %1 free").arg(bytes(stats.minimumFreeBytes));
        if (stats.scanInFlight) extra << tr("counting...");
        setBar(_diskBar, _diskText, stats.managedBytes,
               stats.maximumBytes ? std::optional<quint64>(*stats.maximumBytes) : std::nullopt,
               extra.join(QStringLiteral(" · ")));
        _diskText->setStyleSheet(stats.lowSpace ? QStringLiteral("color: #d28b00; font-weight: 600;") : QString());
    } else {
        _diskBar->setEnabled(false);
        _diskText->setText(tr("not in use"));
    }
    const auto surfaces = CState::surfaceGeometryCache();
    setBar(_surfaceBar, _surfaceText, surfaces->size(), surfaces->capacity());
}

std::optional<vc::objects::AttachmentIdentity> ProjectWorkspace::selectedStore() const
{
    const auto* item = _stores->currentItem();
    if (!item) return std::nullopt;
    const int index = item->data(0, kStoreIndexRole).toInt();
    if (index < 0) return _state->projectAttachment();
    const auto attached = _state->attachedStores();
    if (index >= int(attached.size())) return std::nullopt;
    return attached[size_t(index)];
}

void ProjectWorkspace::updateStoreButtons()
{
    const auto store = selectedStore();
    _detachStore->setEnabled(store && store != _state->projectAttachment());
    _revealStore->setEnabled(store.has_value());
    const auto session = _state->objectStoreSession();
    bool hosted = false;
    if (const auto project = _state->projectAttachment()) hosted = session->hubStatus(*project).has_value();
    for (const auto& attached : _state->attachedStores()) hosted = hosted || session->hubStatus(attached);
    _syncStores->setEnabled(hosted);
}

void ProjectWorkspace::updateResourceButtons()
{
    const auto* item = _resources->currentItem();
    _viewVolume->setEnabled(item && !item->data(0, kVolumeIdRole).toString().isEmpty());
    _detachResource->setEnabled(item && !item->data(0, kLocationRole).toString().isEmpty() &&
                                _state->vpkg() && _state->vpkg()->writable());
    _revealResource->setEnabled(item && !item->data(0, kRevealRole).toString().isEmpty());
}

void ProjectWorkspace::measureVolumeCaches()
{
    const auto pkg = _state->vpkg();
    if (!pkg || _measure->isRunning()) return;
    QStringList paths;
    for (const auto& id : pkg->volumeIDs())
        if (const auto volume = pkg->volume(id); volume && volume->isRemote())
            paths << QString::fromStdString(volume->remotePersistentCachePath().string());
    if (paths.isEmpty()) return;
    _measureCaches->setEnabled(false);
    _measureCaches->setText(tr("Measuring..."));
    _measure->setFuture(QtConcurrent::run([paths]() {
        QHash<QString, quint64> sizes;
        for (const auto& path : paths) {
            quint64 total = 0;
            std::error_code ec;
            for (std::filesystem::recursive_directory_iterator it(
                     path.toStdString(), std::filesystem::directory_options::skip_permission_denied, ec), end;
                 !ec && it != end; it.increment(ec)) {
                std::error_code sizeError;
                if (it->is_regular_file(sizeError)) {
                    const auto size = it->file_size(sizeError);
                    if (!sizeError) total += size;
                }
            }
            sizes[path] = total;
        }
        return sizes;
    }));
}
