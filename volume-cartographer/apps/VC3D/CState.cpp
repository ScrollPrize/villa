#include "CState.hpp"
#include "PointCollectionSync.hpp"

#include "OpenDataManifest.hpp"
#include "VCSettings.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <exception>
#include <thread>
#include <optional>
#include <QSettings>
#include <QTimer>

#include "vc/core/types/VolumePkg.hpp"
#include <vc_object_store/geometry/Geometry.hpp>
#include <vc_object_store/objects/CatalogStore.hpp>
#include <vc_object_store/Uuid.hpp>
#include "vc/core/types/Volume.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/ui/VCCollection.hpp"
#include "vc/core/util/PlaneSurface.hpp"

namespace {

std::shared_ptr<const vcr::catalog::RegistrationGraph> catalogTransformSnapshot()
{
    const auto catalog=vc3d::opendata::cachedCatalog();
    if(!catalog)return {};
    auto graph=std::make_shared<vcr::catalog::RegistrationGraph>();
    graph->addCatalog(*catalog);
    return graph;
}

// Viewers re-query regions as the view moves: keep verified geometry per read
// connection well beyond the store's default (16 MB) so warm queries skip
// reloading and re-hashing chunks.
vc::objects::OpenOptions viewerOpenOptions()
{
    vc::objects::OpenOptions options;
    options.catalogTransforms = catalogTransformSnapshot();
    options.geometryCacheBytes = size_t(256) << 20;
    options.surfaceTiles = CState::surfaceGeometryCache();
    return options;
}

size_t surfaceGeometryCacheBytes()
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    const auto gb = settings.value(vc3d::settings::viewer_cache::SURFACE_GEOMETRY_CACHE_GB,
                                   vc3d::settings::viewer_cache::SURFACE_GEOMETRY_CACHE_GB_DEFAULT).toLongLong();
    return size_t(std::max<qlonglong>(0, gb)) << 30;
}

// The box spanned by `box`'s corners through `m`.
Rect3D mappedBox(const Rect3D& box, const vcr::catalog::Affine& m)
{
    Rect3D result;
    result.low = cv::Vec3f::all(std::numeric_limits<float>::infinity());
    result.high = cv::Vec3f::all(-std::numeric_limits<float>::infinity());
    for (int corner = 0; corner < 8; ++corner) {
        const auto p = m.apply({(corner & 1 ? box.high : box.low)[0], (corner & 2 ? box.high : box.low)[1],
                                (corner & 4 ? box.high : box.low)[2]});
        for (int axis = 0; axis < 3; ++axis) {
            result.low[axis] = std::min(result.low[axis], float(p[axis]));
            result.high[axis] = std::max(result.high[axis], float(p[axis]));
        }
    }
    return result;
}

struct SurfaceFocusPoint {
    cv::Vec3f world{0, 0, 0};
    cv::Vec3f ptr{0, 0, 0};
    int row = -1;
    int col = -1;
};

std::optional<SurfaceFocusPoint> focusPointAtGrid(QuadSurface& surface, int row, int col)
{
    const cv::Mat_<cv::Vec3f>* points = surface.rawPointsPtr();
    if (!points || row < 0 || col < 0 || row >= points->rows || col >= points->cols) {
        return std::nullopt;
    }
    if (row <= 0 || col <= 0 || row >= points->rows - 1 || col >= points->cols - 1) {
        return std::nullopt;
    }
    if (!surface.isQuadValid(row, col)) {
        return std::nullopt;
    }
    const cv::Vec3f normal = surface.gridNormal(row, col);
    if (!std::isfinite(normal[0]) || !std::isfinite(normal[1]) || !std::isfinite(normal[2])) {
        return std::nullopt;
    }

    const cv::Vec3f point = (*points)(row, col);
    if (!vc::geometry::validSurfacePoint(point)) {
        return std::nullopt;
    }

    const cv::Vec3f center = surface.center();
    const cv::Vec2f scale = surface.scale();
    return SurfaceFocusPoint{
        point,
        cv::Vec3f(static_cast<float>(col) - center[0] * scale[0],
                  static_cast<float>(row) - center[1] * scale[1],
                  0.0f),
        row,
        col,
    };
}

std::optional<SurfaceFocusPoint> findSegmentFocusPoint(QuadSurface& surface)
{
    const cv::Mat_<cv::Vec3f>* points = surface.rawPointsPtr();
    if (!points || points->empty()) {
        return std::nullopt;
    }

    const cv::Vec2f centerGrid = surface.ptrToGrid({0, 0, 0});
    const int centerRow = std::clamp(static_cast<int>(std::lround(centerGrid[1])), 0, points->rows - 1);
    const int centerCol = std::clamp(static_cast<int>(std::lround(centerGrid[0])), 0, points->cols - 1);

    if (auto focus = focusPointAtGrid(surface, centerRow, centerCol)) {
        return focus;
    }

    const int maxHorizontalRadius = std::max(centerCol, points->cols - 1 - centerCol);
    for (int radius = 1; radius <= maxHorizontalRadius; ++radius) {
        if (auto focus = focusPointAtGrid(surface, centerRow, centerCol - radius)) {
            return focus;
        }
        if (auto focus = focusPointAtGrid(surface, centerRow, centerCol + radius)) {
            return focus;
        }
    }

    const int maxRadius = std::max(std::max(centerRow, centerCol),
                                   std::max(points->rows - 1 - centerRow, points->cols - 1 - centerCol));
    for (int radius = 1; radius <= maxRadius; ++radius) {
        const int rowMin = std::max(0, centerRow - radius);
        const int rowMax = std::min(points->rows - 1, centerRow + radius);
        const int colMin = std::max(0, centerCol - radius);
        const int colMax = std::min(points->cols - 1, centerCol + radius);

        for (int col = colMin; col <= colMax; ++col) {
            if (auto focus = focusPointAtGrid(surface, rowMin, col)) {
                return focus;
            }
            if (rowMax != rowMin) {
                if (auto focus = focusPointAtGrid(surface, rowMax, col)) {
                    return focus;
                }
            }
        }

        for (int row = rowMin + 1; row < rowMax; ++row) {
            if (auto focus = focusPointAtGrid(surface, row, colMin)) {
                return focus;
            }
            if (colMax != colMin) {
                if (auto focus = focusPointAtGrid(surface, row, colMax)) {
                    return focus;
                }
            }
        }
    }

    return std::nullopt;
}

std::string segmentationSurfaceIdForFocus(CState* state, QuadSurface& surface)
{
    if (!state) {
        return surface.id;
    }
    if (auto active = state->activeSurface().lock(); active.get() == &surface) {
        return state->activeSurfaceId();
    }
    if (!surface.id.empty()) {
        return surface.id;
    }
    return "segmentation";
}

bool resetViewOnSurfaceChangeEnabled()
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    return settings.value(vc3d::settings::viewer::RESET_VIEW_ON_SURFACE_CHANGE,
                          vc3d::settings::viewer::RESET_VIEW_ON_SURFACE_CHANGE_DEFAULT).toBool();
}

std::unique_ptr<POI> createSegmentationFocusPoi(CState* state, QuadSurface& surface)
{
    if (!state) {
        return nullptr;
    }

    const auto focusPoint = findSegmentFocusPoint(surface);
    if (!focusPoint) {
        return nullptr;
    }

    auto vol = state->currentVolume();
    if (!vol) {
        return nullptr;
    }

    auto [w, h, d] = vol->shapeXyz();
    cv::Vec3f clamped = focusPoint->world;
    clamped[0] = std::clamp(clamped[0], 0.0f, static_cast<float>(w - 1));
    clamped[1] = std::clamp(clamped[1], 0.0f, static_cast<float>(h - 1));
    clamped[2] = std::clamp(clamped[2], 0.0f, static_cast<float>(d - 1));

    auto poi = std::make_unique<POI>();
    poi->p = clamped;
    poi->n = cv::Vec3f(0, 0, 0);
    poi->surfaceId = segmentationSurfaceIdForFocus(state, surface);
    poi->surfacePtr = focusPoint->ptr;
    poi->suppressViewerRecenter = false;

    return poi;
}

} // namespace

CState::CState(QObject* parent, bool debugDownloadQueue, CState* objectStoreOwner)
    : QObject(parent)
    , _debugDownloadQueue(debugDownloadQueue)
{
    _objectStoreOwner = objectStoreOwner;
    _objectStoreSession = objectStoreOwner ? objectStoreOwner->objectStoreSession()
                                          : std::make_shared<vc::objects::ObjectStoreSession>();
    // Open Data surfaces are fetched from the catalog on the first read reaching them.
    if (!objectStoreOwner)
        _objectStoreSession->setGeometryFetcher(vc::objects::kCatalogMetaKey, vc::objects::fetchCatalogSurface);
    qRegisterMetaType<vc::objects::StoreSessionEvent>();
    if(objectStoreOwner) {
        connect(objectStoreOwner, &CState::objectStoreChanged, this, &CState::objectStoreChanged);
        connect(objectStoreOwner, &CState::objectStoreObservationFailed, this, &CState::objectStoreObservationFailed);
        connect(objectStoreOwner, &CState::storeRoutesChanged, this, &CState::storeRoutesChanged);
    } else {
        auto* journalTimer = new QTimer(this);
        journalTimer->setInterval(250);
        connect(journalTimer, &QTimer::timeout, this, &CState::pollObjectStoreChanges);
        journalTimer->start();
    }
    _pointCollection = new VCCollection(this);
    if (!objectStoreOwner) _pointCollectionSync = new vc3d::PointCollectionSync(this);

    setSurface("xy plane",
        std::make_shared<PlaneSurface>(cv::Vec3f{2000,2000,2000}, cv::Vec3f{0,0,1}));
    setSurface("xz plane",
        std::make_shared<PlaneSurface>(cv::Vec3f{2000,2000,2000}, cv::Vec3f{0,1,0}));
    setSurface("yz plane",
        std::make_shared<PlaneSurface>(cv::Vec3f{2000,2000,2000}, cv::Vec3f{1,0,0}));
    connect(this, &CState::objectStoreChanged, this, [this](const vc::objects::StoreSessionEvent& event) {
        // Volumes another replica (or a teammate) added or removed.
        using Kind = vc::objects::StoreSessionEvent::Kind;
        if (_vpkg && event.kind == Kind::Committed && event.attachment == _vpkg->attachment() &&
            event.touches({"volume", "normal_grid", "lasagna_dataset"})) {
            try {
                if (_vpkg->reload()) emit vpkgChanged(_vpkg);
            } catch (const std::exception& error) {
                emit objectStoreObservationFailed(QString::fromUtf8(error.what()));
            }
        }
    });
    connect(this, &CState::objectStoreChanged, this, [this](const vc::objects::StoreSessionEvent& event) {
        // The umbilicus is read through the store's volume mapping, so a
        // registration change moves it too.
        if (event.kind == vc::objects::StoreSessionEvent::Kind::Hub ||
            !event.touches({"umbilicus", "volume_transform", "volume_transform_choice"}))
            return;
        emit umbilicusChanged();
    });
}

CState::~CState()
{
    if (_objectStoreOwner) return;
    for (const auto& attachment : _attachedStores) {
        try { _objectStoreSession->detach(attachment); } catch (...) {}
    }
    if (_projectAttachment) {
        try { _objectStoreSession->detach(*_projectAttachment); } catch (...) {}
    }
}

QString CState::storeAuthor()
{
    const QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    auto author = settings.value(vc3d::settings::viewer::USERNAME).toString().trimmed();
    if (author.isEmpty() || author == QStringLiteral("anon"))
        author = qEnvironmentVariable("USER", qEnvironmentVariable("USERNAME", "vc3d"));
    return author;
}

vc::objects::OpenOptions CState::storeOpenOptions(const QString& author, bool readOnly)
{
    auto options = viewerOpenOptions();
    options.author = author.toStdString();
    options.readOnly = readOnly;
    return options;
}

std::optional<vc::objects::AttachmentIdentity> CState::projectAttachment() const
{
    return _objectStoreOwner ? _objectStoreOwner->projectAttachment() : _projectAttachment;
}

std::optional<vc::objects::AttachmentIdentity> CState::attachedAt(const std::filesystem::path& path) const
{
    for (const auto& attachment : _objectStoreSession->attachments()) {
        std::error_code error;
        if (std::filesystem::equivalent(_objectStoreSession->path(attachment), path, error)) return attachment;
    }
    return std::nullopt;
}

vc::objects::AttachmentIdentity CState::attachProjectStore(const std::filesystem::path& path, const QString& author,
                                                           bool readOnly)
{
    if (_objectStoreOwner) return _objectStoreOwner->attachProjectStore(path, author, readOnly);
    if (const auto attached = attachedAt(path)) {
        // A store beside the project becomes the project: it is no longer beside it.
        if (std::erase(_attachedStores, *attached)) {
            std::erase_if(_storeRoutes, [&](const auto& route) { return route.second == *attached; });
            emit storeRoutesChanged();
        }
        return *attached;
    }
    const auto attachment = _objectStoreSession->attach(path, storeOpenOptions(author, readOnly));
    pollObjectStoreChanges();
    return attachment;
}

void CState::releaseProjectStore(const vc::objects::AttachmentIdentity& attachment)
{
    if (_objectStoreOwner) { _objectStoreOwner->releaseProjectStore(attachment); return; }
    if (attachment == _projectAttachment ||
        std::find(_attachedStores.begin(), _attachedStores.end(), attachment) != _attachedStores.end())
        return;
    try { _objectStoreSession->detach(attachment); } catch (...) {}
    pollObjectStoreChanges();
}

void CState::closeProject()
{
    if (_objectStoreOwner) { _objectStoreOwner->closeProject(); return; }
    setVpkg(nullptr);
    switchProject(std::nullopt);
}

std::vector<vc::objects::AttachmentIdentity> CState::attachedStores() const
{
    return _objectStoreOwner ? _objectStoreOwner->attachedStores() : _attachedStores;
}

vc::objects::AttachmentIdentity CState::attachStore(const std::filesystem::path& path, const QString& author,
                                                    bool readOnly)
{
    if (_objectStoreOwner) return _objectStoreOwner->attachStore(path, author, readOnly);
    if (const auto attached = attachedAt(path)) return *attached;
    const auto attachment = _objectStoreSession->attach(path, storeOpenOptions(author, readOnly));
    _attachedStores.push_back(attachment);
    saveStoreRoutes();
    pollObjectStoreChanges();
    emit storeRoutesChanged();
    return attachment;
}

void CState::detachStore(const vc::objects::AttachmentIdentity& attachment)
{
    if (_objectStoreOwner) { _objectStoreOwner->detachStore(attachment); return; }
    if (!std::erase(_attachedStores, attachment)) return;
    std::erase_if(_storeRoutes, [&](const auto& route) { return route.second == attachment; });
    if (_pointCollectionSync) _pointCollectionSync->flush();
    _objectStoreSession->detach(attachment);
    saveStoreRoutes();
    pollObjectStoreChanges();
    emit storeRoutesChanged();
}

const std::vector<std::string>& CState::routableKinds()
{
    static const std::vector<std::string> kinds{"surface", "fiber", "point_collection"};
    return kinds;
}

std::optional<vc::objects::AttachmentIdentity> CState::storeFor(std::string_view kind) const
{
    if (_objectStoreOwner) return _objectStoreOwner->storeFor(kind);
    if (const auto route = _storeRoutes.find(std::string(kind)); route != _storeRoutes.end()) return route->second;
    return _projectAttachment;
}

std::vector<vc::objects::AttachmentIdentity> CState::surfaceSources() const
{
    std::vector<vc::objects::AttachmentIdentity> sources;
    if (const auto store = storeFor("surface"); store && _currentVolume) sources.push_back(*store);
    return sources;
}

void CState::setStoreFor(const std::string& kind, const std::optional<vc::objects::AttachmentIdentity>& store)
{
    if (_objectStoreOwner) { _objectStoreOwner->setStoreFor(kind, store); return; }
    if (std::find(routableKinds().begin(), routableKinds().end(), kind) == routableKinds().end())
        throw std::invalid_argument("objects of kind " + kind + " are kept in the project's store");
    if (store && std::find(_attachedStores.begin(), _attachedStores.end(), *store) == _attachedStores.end() &&
        store != _projectAttachment)
        throw std::invalid_argument("not a store of this project");
    if (storeFor(kind) == store || (!store && !_storeRoutes.contains(kind))) return;
    if (kind == "point_collection" && _pointCollectionSync) _pointCollectionSync->flush();
    if (store && store != _projectAttachment) _storeRoutes[kind] = *store;
    else _storeRoutes.erase(kind);
    saveStoreRoutes();
    emit storeRoutesChanged();
}

namespace {
QString projectSettingsGroup(const vc::objects::AttachmentIdentity& project)
{
    return QStringLiteral("project_stores/") + QString::fromStdString(project.storeUuid);
}
}

void CState::saveStoreRoutes() const
{
    if (!_projectAttachment) return;
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.beginGroup(projectSettingsGroup(*_projectAttachment));
    settings.remove(QString());
    settings.beginWriteArray(QStringLiteral("stores"), int(_attachedStores.size()));
    for (int i = 0; i < int(_attachedStores.size()); ++i) {
        settings.setArrayIndex(i);
        settings.setValue("path", QString::fromStdString(_objectStoreSession->path(_attachedStores[i]).string()));
        settings.setValue("read_only", !_objectStoreSession->writable(_attachedStores[i]));
    }
    settings.endArray();
    for (const auto& [kind, store] : _storeRoutes)
        settings.setValue(QStringLiteral("routes/") + QString::fromStdString(kind), QString::fromStdString(store.storeUuid));
    settings.endGroup();
}

void CState::restoreStoreRoutes()
{
    if (!_projectAttachment) return;
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.beginGroup(projectSettingsGroup(*_projectAttachment));
    const int count = settings.beginReadArray(QStringLiteral("stores"));
    QStringList failures;
    for (int i = 0; i < count; ++i) {
        settings.setArrayIndex(i);
        const auto path = settings.value("path").toString();
        try {
            _attachedStores.push_back(_objectStoreSession->attach(
                path.toStdString(), storeOpenOptions(storeAuthor(), settings.value("read_only").toBool())));
        } catch (const std::exception& error) {
            failures << QStringLiteral("%1: %2").arg(path, QString::fromUtf8(error.what()));
        }
    }
    settings.endArray();
    for (const auto& kind : routableKinds()) {
        const auto store = settings.value(QStringLiteral("routes/") + QString::fromStdString(kind)).toString().toStdString();
        for (const auto& attachment : _attachedStores)
            if (attachment.storeUuid == store) _storeRoutes[kind] = attachment;
    }
    settings.endGroup();
    if (!failures.isEmpty())
        emit objectStoreObservationFailed(tr("Stores attached to this project could not be reattached:\n%1")
                                              .arg(failures.join(QLatin1Char('\n'))));
}

void CState::switchProject(const std::optional<vc::objects::AttachmentIdentity>& next)
{
    if (next == _projectAttachment) return;
    if (_pointCollectionSync) _pointCollectionSync->flush();
    for (const auto& attachment : _attachedStores) {
        try { _objectStoreSession->detach(attachment); } catch (...) {}
    }
    _attachedStores.clear();
    _storeRoutes.clear();
    if (_projectAttachment) {
        try { _objectStoreSession->detach(*_projectAttachment); } catch (...) {}
    }
    _projectAttachment = next;
    restoreStoreRoutes();
    pollObjectStoreChanges();
    emit storeRoutesChanged();
}

std::shared_ptr<vc::objects::SurfaceTileCache> CState::surfaceGeometryCache()
{
    static const auto cache = std::make_shared<vc::objects::SurfaceTileCache>(surfaceGeometryCacheBytes());
    return cache;
}

void CState::applySurfaceGeometryCacheSettings()
{
    surfaceGeometryCache()->setCapacity(surfaceGeometryCacheBytes());
}

void CState::pollObjectStoreChanges()
{
    if(_objectStoreOwner) { _objectStoreOwner->pollObjectStoreChanges(); return; }
    try {
        const auto events = _objectStoreSession->takeEvents();
        _objectStoreObservationError.clear();
        for (const auto& event : events) emit objectStoreChanged(event);
    } catch (const std::exception& error) {
        const auto message = QString::fromUtf8(error.what());
        if (_objectStoreObservationError != message) {
            _objectStoreObservationError = message;
            emit objectStoreObservationFailed(message);
        }
    }
}

std::shared_ptr<VolumePkg> CState::vpkg() const { return _vpkg; }

void CState::setVpkg(std::shared_ptr<VolumePkg> pkg)
{
    const bool changed = _vpkg != pkg;
    if (pkg && !_objectStoreOwner) switchProject(pkg->attachment());
    _vpkg = std::move(pkg);
    updateAnnotationVolume();
    if (changed) {
        // Restore without invoking setters: switching projects must not persist
        // the previous project's bounds into either project.
        _focusBounds = _vpkg ? _vpkg->focusBounds() : std::nullopt;
        _focusBoundsEnabled = _vpkg && _vpkg->focusBoundsEnabled();
        ++_focusBoundsRevision;
        emit focusBoundsChanged();
    }
    emit vpkgChanged(_vpkg);
}

QString CState::projectDirectory() const
{
    return _vpkg ? QString::fromStdString(_vpkg->directory().string()) : QString();
}

bool CState::hasVpkg() const { return _vpkg != nullptr; }

std::shared_ptr<Volume> CState::currentVolume() const { return _currentVolume; }

std::string CState::currentVolumeId() const { return _currentVolume ? _currentVolume->id() : std::string{}; }

void CState::setCurrentVolume(std::shared_ptr<Volume> vol)
{
    if (_currentVolume == vol) {
        ++_focusBoundsRevision;
        emit volumeChanged(_currentVolume, currentVolumeId());
        emit focusBoundsChanged();
        return;
    }
    _currentVolume = std::move(vol);
    updateAnnotationVolume();
    // Reopening the project returns to it.
    if (_vpkg && _currentVolume && !_objectStoreOwner) {
        try { _vpkg->setCurrentVolumeId(_currentVolume->id()); } catch (const std::exception&) {}
    }
    // The open surface's draft is declared in the volume's voxels.
    draftActiveSurface(std::dynamic_pointer_cast<QuadSurface>(surface("segmentation")));
    const auto volumeId = currentVolumeId();
    ++_focusBoundsRevision;
    emit volumeChanged(_currentVolume, volumeId);
    emit focusBoundsChanged();
}

void CState::updateAnnotationVolume()
{
    _annotationVolume.reset();
    if (_currentVolume) _annotationVolume = vc::project::annotationVolume(_vpkg.get(), *_currentVolume);
}

std::optional<Rect3D> CState::focusBounds() const
{
    if (!_focusBounds || !_annotationVolume) return std::nullopt;
    return mappedBox(*_focusBounds, _annotationVolume->fromVolume.inverse());
}

std::optional<Rect3D> CState::activeFocusBounds() const
{
    return _focusBoundsEnabled ? focusBounds() : std::nullopt;
}

std::optional<Rect3D> CState::activeAnnotationFocusBounds() const
{
    return _focusBoundsEnabled && _annotationVolume ? _focusBounds : std::nullopt;
}

bool CState::focusBoundsEnabled() const { return _focusBoundsEnabled; }

uint64_t CState::focusBoundsRevision() const { return _focusBoundsRevision; }

void CState::setFocusBounds(const Rect3D& bounds)
{
    if (!_annotationVolume)
        return;
    const auto normalized = mappedBox(bounds, _annotationVolume->fromVolume);
    for (int axis = 0; axis < 3; ++axis)
        if (!std::isfinite(normalized.low[axis]) || !std::isfinite(normalized.high[axis]))
            return;
    if (_focusBounds && _focusBounds->low == normalized.low &&
        _focusBounds->high == normalized.high) {
        return;
    }
    _focusBounds = normalized;
    if (_vpkg) _vpkg->setFocusBounds(_focusBounds, _focusBoundsEnabled);
    ++_focusBoundsRevision;
    emit focusBoundsChanged();
}

void CState::setFocusBoundsEnabled(bool enabled)
{
    if (_focusBoundsEnabled == enabled) {
        return;
    }
    _focusBoundsEnabled = enabled;
    if (_vpkg) _vpkg->setFocusBounds(_focusBounds, _focusBoundsEnabled);
    ++_focusBoundsRevision;
    emit focusBoundsChanged();
}

void CState::clearFocusBounds()
{
    if (!_focusBounds && !_focusBoundsEnabled) {
        return;
    }
    _focusBounds.reset();
    _focusBoundsEnabled = false;
    if (_vpkg) _vpkg->setFocusBounds(_focusBounds, false);
    ++_focusBoundsRevision;
    emit focusBoundsChanged();
}

std::string CState::segmentationGrowthVolumeId() const { return _segmentationGrowthVolumeId; }

void CState::setSegmentationGrowthVolumeId(const std::string& id)
{
    _segmentationGrowthVolumeId = id;
}

std::weak_ptr<QuadSurface> CState::activeSurface() const { return _activeSurface; }

std::string CState::activeSurfaceId() const { return _activeSurfaceId; }

void CState::setActiveSurface(const std::string& id, std::shared_ptr<QuadSurface> surf)
{
    // Drop derived caches on whichever surface was active before — we only
    // keep them populated for the segment the user is currently editing.
    auto prev = _activeSurface.lock();
    if (prev && prev != surf) {
        prev->unloadCaches();
    }
    _activeSurfaceId = id;
    _activeSurface = surf;
}

void CState::clearActiveSurface()
{
    if (auto prev = _activeSurface.lock()) {
        prev->unloadCaches();
    }
    _activeSurface.reset();
    _activeSurfaceId.clear();
}

VCCollection* CState::pointCollection() const { return _pointCollection; }

void CState::savePointCollections()
{
    if (_pointCollectionSync) _pointCollectionSync->flush();
    emit pointCollectionsSaveRequested();
}

void CState::closeAll()
{
    emit volumeClosing();

    clearActiveSurface();

    // Every surface except the view planes belongs to the closed project.
    std::vector<std::string> closed;
    for (const auto& [name, surf] : _surfs)
        if (!std::dynamic_pointer_cast<PlaneSurface>(surf)) closed.push_back(name);
    std::sort(closed.begin(), closed.end());
    for (const auto& name : closed)
        setSurface(name, nullptr, true);

    _currentVolume = nullptr;
    _segmentationGrowthVolumeId.clear();

    _pois.clear();
    if (_pointCollectionSync) _pointCollectionSync->unload();
    _pointCollection->clearAll();

    setVpkg(nullptr);
}

// --- Surface methods (from CSurfaceCollection) ---

void CState::setSurface(const std::string& name, std::shared_ptr<Surface> surf, bool noSignalSend, bool isEditUpdate)
{
    auto it = _surfs.find(name);
    const bool sameSurface = it != _surfs.end() && it->second == surf;
    if (sameSurface && !isEditUpdate && surf != nullptr) {
        return;
    }
    if (_surfaceBatchDepth == 0 && it != _surfs.end() && it->second &&
        it->second != surf) {
        emit surfaceWillBeDeleted(name, it->second);
    }

    POI* delayedFocusPoi = nullptr;
    if (name == "segmentation" && surf != nullptr && !isEditUpdate &&
        resetViewOnSurfaceChangeEnabled()) {
        if (auto quad = std::dynamic_pointer_cast<QuadSurface>(surf)) {
            try {
                auto focusPoi = createSurfaceFocusPoi(*quad);
                if (focusPoi) {
                    delayedFocusPoi = focusPoi.get();
                    _pois["focus"] = std::move(focusPoi);
                }
            } catch (const std::exception&) {
                // Reset recentering is optional; activation/orientation paths
                // handle and report lazy-load failures after the surface is set.
            } catch (...) {
                // Keep surface activation alive even if focus initialization fails.
            }
        }
    }

    _surfs[name] = surf;
    if (name == "segmentation" && (!sameSurface || (isEditUpdate && !_activeSurfaceEditing))) {
        draftActiveSurface(std::dynamic_pointer_cast<QuadSurface>(surf));
    }
    if (!sameSurface) {
        // Edit updates re-set the same pointer every frame; only a mapping
        // change should invalidate cached views of the surface map.
        ++_surfacesVersion;
        if (_surfaceBatchDepth > 0) {
            _surfaceBatchChanged = true;
        }
    }

    if (_surfaceBatchDepth == 0 && (!noSignalSend || surf == nullptr)) {
        emit surfaceChanged(name, surf, isEditUpdate);
    }

    if (delayedFocusPoi) {
        auto poiIt = _pois.find("focus");
        if (poiIt != _pois.end() && poiIt->second.get() == delayedFocusPoi) {
            emit poiChanged("focus", delayedFocusPoi);
            delayedFocusPoi->suppressViewerRecenter = false;
        }
    }
}

void CState::draftActiveSurface(const std::shared_ptr<QuadSurface>& surface)
{
    // Workspaces sharing the main session draft their own surfaces.
    if (_objectStoreOwner || !_objectStoreSession) {
        return;
    }
    if (!_activeSurfaceDraft.empty() && (!surface || surface->id != _activeSurfaceDraft)) {
        _objectStoreSession->clearDraft(_activeSurfaceDraft);
        _activeSurfaceDraft.clear();
    }
    if (!surface || !_currentVolume || !vc::objects::util::isUuid(surface->id)) {
        return;
    }
    _objectStoreSession->setDraft(surface->id, quad_surface_grid(*surface),
                                  vc::objects::volumeCoordinates(currentVolumeId(), 1));
    _activeSurfaceDraft = surface->id;
}

void CState::setSurfacesBatch(
    const std::vector<std::pair<std::string, std::shared_ptr<Surface>>>& updates)
{
    const bool outermost = _surfaceBatchDepth == 0;
    if (outermost) {
        _surfaceBatchChanged = false;
    }
    ++_surfaceBatchDepth;

    auto finishBatch = [this, outermost]() {
        --_surfaceBatchDepth;
        if (outermost) {
            const bool changed = _surfaceBatchChanged;
            _surfaceBatchChanged = false;
            if (changed) {
                emit surfaceChanged("", nullptr, false);
            }
        }
    };

    try {
        for (const auto& [name, surface] : updates) {
            setSurface(name, surface, true, false);
        }
    } catch (...) {
        finishBatch();
        throw;
    }
    finishBatch();
}

std::shared_ptr<Surface> CState::surface(const std::string& name)
{
    auto it = _surfs.find(name);
    if (it == _surfs.end())
        return nullptr;
    return it->second;
}

Surface* CState::surfaceRaw(const std::string& name)
{
    auto it = _surfs.find(name);
    if (it == _surfs.end())
        return nullptr;
    return it->second.get();
}

std::string CState::findSurfaceId(Surface* surf)
{
    if (!surf) return {};
    for (const auto& [name, s] : _surfs) {
        if (s.get() == surf) {
            return name;
        }
    }
    return {};
}

std::vector<std::shared_ptr<Surface>> CState::surfaces()
{
    std::vector<std::shared_ptr<Surface>> result;
    result.reserve(_surfs.size());

    for (auto& surface : _surfs) {
        result.push_back(surface.second);
    }

    return result;
}

// --- POI methods (from CSurfaceCollection) ---

std::unique_ptr<POI> CState::createSurfaceFocusPoi(QuadSurface& surface)
{
    return createSegmentationFocusPoi(this, surface);
}

void CState::setPOI(const std::string& name, POI* poi)
{
    auto it = _pois.find(name);
    if (it != _pois.end() && it->second.get() == poi) {
        // Same pointer re-submitted (caller mutated in place) - just signal
        emit poiChanged(name, poi);
        poi->suppressViewerRecenter = false;
        return;
    }
    _pois[name] = std::unique_ptr<POI>(poi);
    emit poiChanged(name, poi);
    poi->suppressViewerRecenter = false;
}

POI* CState::poi(const std::string& name)
{
    auto it = _pois.find(name);
    if (it == _pois.end())
        return nullptr;
    return it->second.get();
}
