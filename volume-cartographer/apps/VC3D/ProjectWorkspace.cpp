#include "ProjectWorkspace.hpp"

#include "CState.hpp"
#include "MenuActionController.hpp"
#include "OpenDataSampleProject.hpp"
#include "VCSettings.hpp"

#include "vc/core/util/RemoteCacheSettings.hpp"
#include "OpenedVolumes.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc_object_store/objects/SurfaceTileCache.hpp"

#include <QApplication>
#include <QClipboard>
#include <QComboBox>
#include <QDateTime>
#include <QDesktopServices>
#include <QFileInfo>
#include <QFileDialog>
#include <QFormLayout>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QLineEdit>
#include <QLocale>
#include <QMenu>
#include <QMessageBox>
#include <QProgressBar>
#include <QPushButton>
#include <QScrollArea>
#include <QSignalBlocker>
#include <QStackedWidget>
#include <QTimer>
#include <QTreeWidget>
#include <QTreeWidgetItemIterator>
#include <QUrl>
#include <QVBoxLayout>
#include <QtConcurrent/QtConcurrent>

#include <algorithm>
#include <filesystem>
#include <map>

namespace {

enum StoreColumn { StoreName, StoreAccess, StoreSurfaces, StoreFibers, StorePointCollections, StoreSize,
                   StoreFolder, StoreColumns };
enum ResourceColumn { ResName, ResLocation, ResTags, ResShape, ResVoxel, ResLevels, ResType, ResCache,
                      ResColumns };
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
    if (kind == "umbilicus") return QObject::tr("Umbilici");
    if (kind == "volume_transform") return QObject::tr("Registrations");
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

QString dtypeName(const std::string& dtype)
{
    if (dtype == "|u1") return QStringLiteral("uint8");
    if (dtype.ends_with("u2")) return QStringLiteral("uint16");
    if (dtype.ends_with("f4")) return QStringLiteral("float32");
    return QString::fromStdString(dtype);
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

// A button whose menu holds the actions; returns the menu to fill.
QMenu* menuButton(const QString& text, QWidget* parent, QHBoxLayout* row)
{
    auto* menu = new QMenu(parent);
    button(text, parent, row)->setMenu(menu);
    return menu;
}

// Secondary text: details, hints, column notes.
void dim(QWidget* widget)
{
    auto palette = widget->palette();
    palette.setColor(QPalette::WindowText, palette.color(QPalette::PlaceholderText));
    palette.setColor(QPalette::Text, palette.color(QPalette::PlaceholderText));
    widget->setPalette(palette);
}

// A section's title, with its actions to the right on the same line.
QHBoxLayout* section(const QString& title, QWidget* parent, QVBoxLayout* layout)
{
    auto* row = new QHBoxLayout;
    auto* label = new QLabel(title, parent);
    auto font = label->font();
    font.setBold(true);
    font.setPointSizeF(font.pointSizeF() * 1.15);
    label->setFont(font);
    row->addWidget(label);
    row->addStretch(1);
    layout->addSpacing(14);
    layout->addLayout(row);
    return row;
}

void addGap(QHBoxLayout* row)
{
    row->addSpacing(12);
}

// Hide columns that no row fills (e.g. Tags when nothing is tagged).
void hideEmptyColumns(QTreeWidget* tree, int first)
{
    for (int c = first; c < tree->columnCount(); ++c) {
        bool used = false;
        for (QTreeWidgetItemIterator it(tree); *it && !used; ++it) used = !(*it)->text(c).isEmpty();
        tree->setColumnHidden(c, !used);
    }
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
    auto* emptyText = new QLabel(tr("No project is open.\n\nA project file (.vcproj) lists the "
                                    "volumes and stores you work with. Stores hold your surfaces, fibers "
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

    // The overview: the project, then its data, its stores and the caches.
    auto* scroll = new QScrollArea(_pages);
    scroll->setWidgetResizable(true);
    scroll->setFrameShape(QFrame::NoFrame);
    auto* page = new QWidget(scroll);
    auto* layout = new QVBoxLayout(page);
    layout->setContentsMargins(16, 12, 16, 12);
    layout->setSpacing(6);
    scroll->setWidget(page);
    _pages->addWidget(scroll);

    // Project: its name (edited in place), one line of details, its actions.
    auto* title = new QHBoxLayout;
    layout->addLayout(title);
    _name = new QLineEdit(page);
    _name->setObjectName(QStringLiteral("projectName"));
    _name->setToolTip(tr("Project name; click to rename."));
    _name->setFrame(false);
    _name->setStyleSheet(QStringLiteral("QLineEdit#projectName { background: transparent; }"
                                        "QLineEdit#projectName:focus { background: palette(base); }"));
    auto nameFont = _name->font();
    nameFont.setBold(true);
    nameFont.setPointSizeF(nameFont.pointSizeF() * 1.5);
    _name->setFont(nameFont);
    _name->setMaximumWidth(520);
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
    title->addWidget(_name, 1);
    _resetOpenData = button(tr("Reset to Open Data..."), page, title);
    connect(_resetOpenData, &QPushButton::clicked, _menu, &MenuActionController::resetOpenDataProject);
    connect(button(tr("Save As..."), page, title), &QPushButton::clicked, _menu, &MenuActionController::saveProjectAs);
    _revealProject = button(tr("Open Folder"), page, title);
    connect(_revealProject, &QPushButton::clicked, this, [this]() { reveal(_state->projectDirectory()); });
    _details = new QLabel(page);
    _details->setTextInteractionFlags(Qt::TextSelectableByMouse);
    _details->setWordWrap(true);
    _details->setContentsMargins(_name->textMargins().left() + 2, 0, 0, 0);
    dim(_details);
    layout->addWidget(_details);

    // Data
    auto* dataHeader = section(tr("Data"), page, layout);
    auto* add = menuButton(tr("Add"), page, dataHeader);
    add->addAction(tr("Volume..."), _menu, &MenuActionController::attachVolume);
    add->addAction(tr("Remote Zarr..."), _menu, &MenuActionController::attachRemoteZarr);
    add->addSeparator();
    add->addAction(tr("Normal Grid..."), _menu, &MenuActionController::attachNormalGrid);
    add->addSeparator();
    add->addAction(tr("Lasagna Dataset..."), _menu, &MenuActionController::attachLasagnaManifest);
    add->addAction(tr("Remote Lasagna Dataset..."), _menu, &MenuActionController::attachRemoteLasagnaManifest);
    connect(button(tr("Open Data Catalog..."), page, dataHeader), &QPushButton::clicked, _menu,
            &MenuActionController::showOpenDataCatalog);
    addGap(dataHeader);
    _viewVolume = button(tr("View"), page, dataHeader);
    _viewVolume->setToolTip(tr("Show this volume in the viewers (or double-click it)."));
    connect(_viewVolume, &QPushButton::clicked, this, [this]() {
        if (auto* item = _resources->currentItem()) {
            const auto id = item->data(0, kVolumeIdRole).toString();
            if (!id.isEmpty()) emit volumeActivationRequested(id);
        }
    });
    _revealResource = button(tr("Open Folder"), page, dataHeader);
    connect(_revealResource, &QPushButton::clicked, this, [this]() {
        if (auto* item = _resources->currentItem()) reveal(item->data(0, kRevealRole).toString());
    });
    _detachResource = button(tr("Detach"), page, dataHeader);
    connect(_detachResource, &QPushButton::clicked, this, [this]() {
        if (auto* item = _resources->currentItem()) {
            const auto location = item->data(0, kLocationRole).toString();
            if (!location.isEmpty()) _menu->detachEntryAt(location);
        }
    });
    _resources = new QTreeWidget(page);
    _resources->setObjectName(QStringLiteral("projectResources"));
    _resources->setColumnCount(ResColumns);
    _resources->setHeaderLabels({tr("Name"), tr("Location"), tr("Tags"), tr("Size"), tr("Voxel"), tr("Levels"),
                                 tr("Type"), tr("Disk cache")});
    _resources->headerItem()->setToolTip(ResShape, tr("x × y × z voxels"));
    _resources->headerItem()->setToolTip(ResCache, tr("Downloaded chunks of a remote volume (Caches › Measure)."));
    _resources->setMinimumHeight(220);
    _resources->setUniformRowHeights(true);
    _resources->setAlternatingRowColors(true);
    connect(_resources, &QTreeWidget::itemSelectionChanged, this, &ProjectWorkspace::updateResourceButtons);
    connect(_resources, &QTreeWidget::itemDoubleClicked, this, [this](QTreeWidgetItem* item) {
        const auto id = item ? item->data(0, kVolumeIdRole).toString() : QString();
        if (!id.isEmpty()) emit volumeActivationRequested(id);
    });
    layout->addWidget(_resources, 1);

    // Stores, and which one each kind is kept in.
    auto* storeHeader = section(tr("Stores"), page, layout);
    auto* addStore = menuButton(tr("Add"), page, storeHeader);
    addStore->addAction(tr("New Store..."), _menu, &MenuActionController::newStore);
    addStore->addAction(tr("Attach Store..."), this, [this]() { _menu->attachStore(false); });
    addStore->addAction(tr("Attach Read-only..."), this, [this]() { _menu->attachStore(true); });
    addStore->addAction(tr("Clone Hosted Store..."), this, [this]() { _menu->cloneHostedStore(false); });
    connect(button(tr("Import from Folder..."), page, storeHeader), &QPushButton::clicked, _menu,
            &MenuActionController::importIntoObjectStore);
    _syncStores = button(tr("Sync Now"), page, storeHeader);
    _syncStores->setToolTip(tr("Sync every hosted store now."));
    connect(_syncStores, &QPushButton::clicked, this, [this]() {
        const auto session = _state->objectStoreSession();
        std::vector<vc::objects::AttachmentIdentity> stores = _state->attachedStores();
        for (const auto& store : stores)
            if (session->hubStatus(store)) session->syncHub(store);
        emit hubSyncRequested();
    });
    addGap(storeHeader);
    _revealStore = button(tr("Open Folder"), page, storeHeader);
    connect(_revealStore, &QPushButton::clicked, this, [this]() {
        if (const auto store = selectedStore())
            reveal(QString::fromStdString(_state->objectStoreSession()->path(*store).string()));
    });
    _replaceStore = button(tr("Locate..."), page, storeHeader);
    _replaceStore->setToolTip(tr("Point this store reference at another .vcobj file."));
    connect(_replaceStore, &QPushButton::clicked, this, [this]() {
        const auto* item = _stores->currentItem();
        if (!item || !_state->vpkg() || !_state->vpkg()->writable()) return;
        const auto id = item->data(0, kStoreIndexRole).toString().toStdString();
        const auto path = QFileDialog::getOpenFileName(this, tr("Locate Store"), _state->projectDirectory(), tr("Object store (*.vcobj)"));
        if (path.isEmpty()) return;
        try { _state->replaceStoreReference(id, path.toStdString()); }
        catch (const std::exception& error) { QMessageBox::warning(this, tr("Cannot replace store"), QString::fromUtf8(error.what())); }
    });
    _detachStore = button(tr("Detach"), page, storeHeader);
    connect(_detachStore, &QPushButton::clicked, this, [this]() {
        const auto* item = _stores->currentItem();
        if (!item) return;
        try { _state->detachStoreReference(item->data(0, kStoreIndexRole).toString().toStdString()); }
        catch (const std::exception& error) {
            QMessageBox::critical(this, tr("Cannot detach store"), QString::fromUtf8(error.what()));
        }
    });
    auto* storeRow = new QHBoxLayout;
    storeRow->setSpacing(16);
    layout->addLayout(storeRow);
    _stores = new QTreeWidget(page);
    _stores->setObjectName(QStringLiteral("projectStores"));
    _stores->setRootIsDecorated(false);
    _stores->setAlternatingRowColors(true);
    _stores->setColumnCount(StoreColumns);
    _stores->setHeaderLabels({tr("Store"), tr("Access"), tr("Surfaces"), tr("Fibers"), tr("Point collections"),
                              tr("Size"), tr("Folder")});
    _stores->header()->setStretchLastSection(true);
    connect(_stores, &QTreeWidget::itemSelectionChanged, this, &ProjectWorkspace::updateStoreButtons);
    storeRow->addWidget(_stores, 1);
    _routesPanel = new QWidget(page);
    auto* routesLayout = new QVBoxLayout(_routesPanel);
    routesLayout->setContentsMargins(0, 0, 0, 0);
    auto* routesTitle = new QLabel(tr("Each kind is kept in"), _routesPanel);
    routesTitle->setToolTip(tr("The store new objects of each kind are saved to, and whose objects are listed."));
    dim(routesTitle);
    routesLayout->addWidget(routesTitle);
    auto* routesForm = new QFormLayout;
    routesForm->setFieldGrowthPolicy(QFormLayout::FieldsStayAtSizeHint);
    routesForm->setLabelAlignment(Qt::AlignRight | Qt::AlignVCenter);
    for (const auto& kind : CState::routableKinds()) {
        auto* combo = new QComboBox(_routesPanel);
        combo->setObjectName(QStringLiteral("projectRoute_%1").arg(QString::fromStdString(kind)));
        combo->setSizeAdjustPolicy(QComboBox::AdjustToContents);
        connect(combo, &QComboBox::activated, this, [this, kind, combo](int index) {
            try { _state->setStoreSelection(kind, combo->itemData(index).toString().toStdString()); }
            catch (const std::exception& error) {
                QMessageBox::warning(this, tr("Stores"), QString::fromUtf8(error.what()));
                rebuildRoutes();
            }
        });
        _routes.push_back(combo);
        routesForm->addRow(kindTitle(kind), combo);
    }
    routesLayout->addLayout(routesForm);
    routesLayout->addStretch(1);
    storeRow->addWidget(_routesPanel, 0, Qt::AlignTop);

    // Caches: one meter each, side by side.
    auto* cacheHeader = section(tr("Caches"), page, layout);
    _measureCaches = button(tr("Measure Volume Caches"), page, cacheHeader);
    _measureCaches->setToolTip(tr("Add up the downloaded chunks of each remote volume (reads the cache folders)."));
    connect(_measureCaches, &QPushButton::clicked, this, &ProjectWorkspace::measureVolumeCaches);
    connect(button(tr("Settings..."), page, cacheHeader), &QPushButton::clicked, _menu,
            &MenuActionController::showSettingsDialog);
    _revealCache = button(tr("Open Folder"), page, cacheHeader);
    connect(_revealCache, &QPushButton::clicked, this, []() { reveal(vc3d::remoteCachePath()); });
    auto* caches = new QGridLayout;
    caches->setHorizontalSpacing(24);
    caches->setVerticalSpacing(3);
    layout->addLayout(caches);
    const auto addBar = [&](int column, const QString& name, const QString& tip, QProgressBar*& bar, QLabel*& text) {
        auto* label = new QLabel(name, page);
        bar = new QProgressBar(page);
        bar->setTextVisible(false);
        bar->setMaximumHeight(8);
        text = new QLabel(page);
        text->setWordWrap(true);
        dim(text);
        for (QWidget* w : {static_cast<QWidget*>(label), static_cast<QWidget*>(bar), static_cast<QWidget*>(text)})
            w->setToolTip(tip);
        caches->addWidget(label, 0, column);
        caches->addWidget(bar, 1, column);
        caches->addWidget(text, 2, column, Qt::AlignTop);
        caches->setColumnStretch(column, 1);
        return label;
    };
    addBar(0, tr("Volume chunks in memory"), tr("Decoded volume chunks held in memory, shared by every viewer."),
           _ramBar, _ramText);
    _diskCell = addBar(1, tr("Volume chunks on disk"),
                       tr("Chunks of remote volumes downloaded to disk. Older chunks are removed past the limit; "
                          "when the disk runs low, caching stops and nothing is removed."),
                       _diskBar, _diskText);
    addBar(2, tr("Surface geometry in memory"), tr("Decoded surface tiles kept for every open store."),
           _surfaceBar, _surfaceText);

    _measure = new QFutureWatcher<QHash<QString, quint64>>(this);
    connect(_measure, &QFutureWatcher<QHash<QString, quint64>>::finished, this, [this]() {
        for (auto [path, size] : _measure->result().asKeyValueRange()) _cacheBytes[path] = size;
        _measureCaches->setEnabled(true);
        _measureCaches->setText(tr("Measure Volume Caches"));
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
    const bool open = bool(_state->vpkg());
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
    const auto session = _state->objectStoreSession();
    if (!_name->hasFocus()) _name->setText(QString::fromStdString(pkg->name()));
    _name->setReadOnly(!pkg->writable());
    QStringList details{QString::fromStdString(pkg->path().string()),
                        pkg->writable() ? tr("read-write") : tr("read-only"),
                        tr("editing as %1").arg(CState::storeAuthor())};
    const auto sample = vc3d::opendata::openDataProjectSample(*pkg);
    if (sample) details << tr("made from Open Data sample %1").arg(QString::fromStdString(*sample));
    _details->setText(details.join(QStringLiteral("  ·  ")));
    _revealProject->setToolTip(_state->projectDirectory());
    _resetOpenData->setVisible(sample.has_value());
    _resetOpenData->setEnabled(sample && pkg->writable());
}

void ProjectWorkspace::rebuildStores()
{
    const auto session = _state->objectStoreSession();
    const auto attached = _state->attachedStores();
    const auto selectedId = _stores->currentItem() ? _stores->currentItem()->data(0, kStoreIndexRole).toString() : QString();
    QSignalBlocker block(_stores);
    _stores->clear();
    const auto projectDirectory = QFileInfo(_state->projectDirectory()).absoluteFilePath();
    // A store's folder, or "project folder" for the usual case.
    const auto folder = [&](const std::filesystem::path& path) {
        const auto directory = QFileInfo(QString::fromStdString(path.string())).absolutePath();
        return directory == projectDirectory ? tr("project folder") : directory;
    };
    for (const auto& reference : _state->vpkg()->document().data().at("stores")) {
        const auto id = reference.at("id").get<std::string>();
        const auto available = _state->storeReference(id);
        auto* item = new QTreeWidgetItem(_stores);
        item->setData(0, kStoreIndexRole, QString::fromStdString(id));
        if (selectedId == QString::fromStdString(id)) _stores->setCurrentItem(item);
        for (int c : {StoreSurfaces, StoreFibers, StorePointCollections, StoreSize})
            item->setTextAlignment(c, Qt::AlignRight | Qt::AlignVCenter);
        if (!available) {
            const auto path = _state->vpkg()->document().resolve(reference.at("location"));
            item->setText(StoreName, QString::fromStdString(path.filename().string()));
            item->setText(StoreAccess, tr("Missing"));
            item->setToolTip(StoreAccess, tr("Not found at %1. Use Locate... to point at it.")
                                              .arg(QString::fromStdString(path.string())));
            item->setForeground(StoreAccess, QColor(0xd2, 0x8b, 0x00));
            item->setText(StoreFolder, folder(path));
            item->setToolTip(StoreFolder, QString::fromStdString(path.string()));
            continue;
        }
        const auto& store = *available;
        const auto path = session->path(store);
        item->setText(StoreName, QString::fromStdString(path.filename().string()));
        item->setToolTip(StoreName, tr("Store UUID %1").arg(QString::fromStdString(store.storeUuid)));
        QString access = session->writable(store) ? tr("Read-write") : tr("Read-only");
        if (const auto hub = session->hubStatus(store)) access += tr(" · hosted, %1").arg(hubStatusSummary(*hub));
        item->setText(StoreAccess, access);
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
        std::error_code ec;
        const auto size = std::filesystem::file_size(path, ec);
        item->setText(StoreSize, ec ? QString() : bytes(size));
        item->setText(StoreFolder, folder(path));
        item->setToolTip(StoreFolder, QString::fromStdString(path.string()));
    }
    for (int c = 0; c < StoreFolder; ++c) _stores->resizeColumnToContents(c);
    // As tall as its rows (or the routes beside it), so the data list above gets the room.
    const int rowHeight = _stores->topLevelItemCount() ? _stores->visualItemRect(_stores->topLevelItem(0)).height() : 0;
    _stores->setFixedHeight(std::max(_routesPanel->sizeHint().height(),
                                     _stores->header()->sizeHint().height() + 2 * _stores->frameWidth() +
                                         std::max(1, _stores->topLevelItemCount()) *
                                             std::max(rowHeight, fontMetrics().height() + 4) + 4));
    updateStoreButtons();
}

void ProjectWorkspace::rebuildRoutes()
{
    const auto& data = _state->vpkg()->document().data();
    const auto& kinds = CState::routableKinds();
    for (size_t k = 0; k < kinds.size(); ++k) {
        auto* combo = _routes[k];
        QSignalBlocker block(combo);
        combo->clear();
        combo->addItem(tr("None"), QString());
        for (const auto& reference : data.at("stores")) {
            const auto id = reference.at("id").get<std::string>();
            const auto store = _state->storeReference(id);
            QString name = QString::fromStdString(reference.at("location").get<std::string>());
            if (!store) name += tr(" [missing]");
            else if (!_state->objectStoreSession()->writable(*store)) name += tr(" [read-only]");
            combo->addItem(name, QString::fromStdString(id));
            if (data.at("selections").at(kinds[k]) == id) combo->setCurrentIndex(combo->count() - 1);
        }
        combo->setEnabled(_state->vpkg()->writable());
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
    const auto entryItem = [&](QTreeWidgetItem* parent, const vc::project::Entry& entry) {
        auto* item = new QTreeWidgetItem(parent);
        const auto location = QString::fromStdString(entry.location);
        item->setText(ResName, QFileInfo(trimmedLocation(location)).fileName());
        item->setText(ResLocation, location);
        item->setToolTip(ResLocation, resolved(entry.location));
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
        try {
            const auto& pyramid = _state->openedVolumes()->pyramid(volume);
            item->setText(ResLevels, QString::number(pyramid.levels.size()));
            item->setText(ResType, dtypeName(pyramid.dtype));
        } catch (const std::exception& error) {
            item->setToolTip(ResLevels, QString::fromUtf8(error.what()));
        }
        if (volume.isRemote()) {
            const auto cache = QString::fromStdString(volume.remotePersistentCachePath().string());
            const auto measured = _cacheBytes.find(cache);
            item->setText(ResCache, measured != _cacheBytes.end() ? bytes(*measured) : tr("—"));
            item->setToolTip(ResCache, measured != _cacheBytes.end() ? cache : tr("Not measured yet: Caches › Measure."));
            item->setTextAlignment(ResCache, Qt::AlignRight | Qt::AlignVCenter);
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
        item->setText(ResName, QStringLiteral("%1  %2").arg(title).arg(count));
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
        item->setText(ResLocation, tr("%1 (via dataset)").arg(volumeLocation(*volume)));
        item->setToolTip(ResLocation, tr("Loaded by a dataset; not listed in the project file."));
    }
    volumes->setExpanded(true);

    // Groups with nothing in them are left out (Add lists what can be attached).
    const auto& grids = pkg->normalGridEntries();
    if (!grids.empty()) {
        auto* gridGroup = group(tr("Normal grids"), int(grids.size()));
        for (const auto& entry : grids) entryItem(gridGroup, entry)->setText(ResType, tr("normal grid"));
        gridGroup->setExpanded(true);
    }

    const auto& lasagna = pkg->allLasagnaDatasetEntries();
    auto* lasagnaGroup = lasagna.empty() ? nullptr : group(tr("Lasagna datasets"), int(lasagna.size()));
    for (const auto& entry : lasagna) {
        auto* item = entryItem(lasagnaGroup, entry);
        const bool fiber = vc::project::isFiberLasagnaEntry(entry);
        const bool selected = entry.location == (fiber ? pkg->selectedFiberInferenceDataset()
                                                       : pkg->selectedLasagnaDataset());
        QString type = fiber ? tr("fiber inference") : tr("Lasagna");
        if (selected) type += tr(" (selected)");
        item->setText(ResType, type);
    }
    if (lasagnaGroup) lasagnaGroup->setExpanded(true);

    hideEmptyColumns(_resources, ResLocation);
    for (int c = 0; c < ResColumns; ++c) _resources->resizeColumnToContents(c);
    _resources->setColumnWidth(ResLocation, std::clamp(_resources->columnWidth(ResLocation), 160, 520));
    updateResourceButtons();
}

void ProjectWorkspace::updateCaches()
{
    if (!isVisible()) return;
    const auto ram = _state->openedVolumes()->cacheStats();
    setBar(_ramBar, _ramText, ram.bytes, ram.capacity);
    const auto root = vc3d::remoteCachePathFs();
    const auto rootText = QString::fromStdString(root.string());
    _revealCache->setToolTip(rootText);
    _diskCell->setToolTip(tr("Chunks of remote volumes downloaded to %1. Older chunks are removed past the limit; "
                             "when the disk runs low, caching stops and nothing is removed.").arg(rootText));
    std::error_code rootError;
    if (std::filesystem::is_regular_file(root / ".vc_reader_cache.sqlite3", rootError)) {
        const auto budget = vcr::zarr::diskBudget(root.string(), std::nullopt);
        const auto limit = vc::settings::remoteCacheMaxBytes();
        const auto minimumFree = static_cast<std::uint64_t>(vc::settings::remoteCacheMinFreeBytes());
        const auto space = std::filesystem::space(root, rootError);
        const std::uint64_t freeBytes = rootError ? 0 : space.available;
        const bool lowSpace = !rootError && freeBytes < minimumFree;
        QStringList extra;
        if (!rootError) extra << tr("%1 free").arg(bytes(freeBytes));
        if (minimumFree) extra << tr("stops caching under %1 free").arg(bytes(minimumFree));
        setBar(_diskBar, _diskText, static_cast<quint64>(budget->total_bytes()),
               limit ? std::optional<quint64>(static_cast<quint64>(*limit)) : std::nullopt,
               extra.join(QStringLiteral(" · ")));
        // Overrides the dimmed palette while space is low.
        _diskText->setStyleSheet(lowSpace ? QStringLiteral("color: #d28b00; font-weight: 600;") : QString());
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
    return _state->storeReference(item->data(0, kStoreIndexRole).toString().toStdString());
}

void ProjectWorkspace::updateStoreButtons()
{
    const auto store = selectedStore();
    _detachStore->setEnabled(_stores->currentItem() && _state->vpkg() && _state->vpkg()->writable() && _state->vpkg()->document().data().at("stores").size() > 1);
    _revealStore->setEnabled(store.has_value());
    _replaceStore->setEnabled(_stores->currentItem() && _state->vpkg() && _state->vpkg()->writable());
    const auto session = _state->objectStoreSession();
    bool hosted = false;
    for (const auto& attached : _state->attachedStores()) hosted = hosted || session->hubStatus(attached);
    _syncStores->setVisible(hosted);
}

void ProjectWorkspace::updateResourceButtons()
{
    const auto* item = _resources->currentItem();
    _viewVolume->setEnabled(item && !item->data(0, kVolumeIdRole).toString().isEmpty());
    _detachResource->setEnabled(item && !item->data(0, kLocationRole).toString().isEmpty() &&
                                _state->vpkg() && _state->vpkg()->writable());
    const auto location = item ? item->data(0, kRevealRole).toString() : QString();
    _revealResource->setEnabled(!location.isEmpty());
    _revealResource->setText(vc::project::isLocationRemote(location.toStdString()) ? tr("Copy URL") : tr("Open Folder"));
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
