#include "CState.hpp"
#include <vc_object_store/Uuid.hpp>
#include "PointCollectionSync.hpp"

#include "OpenDataManifest.hpp"
#include "VCSettings.hpp"
#include "OpenedVolumes.hpp"

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

// The box spanned by `box`'s corners scaled by `scale` (positive).
Rect3D scaledBox(const Rect3D& box, double scale)
{
    Rect3D result;
    for (int axis = 0; axis < 3; ++axis) {
        result.low[axis] = std::min(box.low[axis], box.high[axis]) * float(scale);
        result.high[axis] = std::max(box.low[axis], box.high[axis]) * float(scale);
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
    if (objectStoreOwner) {
        _openedVolumes = objectStoreOwner->_openedVolumes;
    } else {
        using namespace vc3d::settings;
        const QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        const auto gib = settings.value(perf::RAM_CACHE_SIZE_GB, perf::RAM_CACHE_SIZE_GB_DEFAULT).toULongLong();
        _openedVolumes = std::make_shared<OpenedVolumes>(std::size_t(gib) << 30);
    }
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

std::optional<vc::objects::AttachmentIdentity> CState::storeReference(const std::string& id) const
{
    if (_objectStoreOwner) return _objectStoreOwner->storeReference(id);
    const auto found = _storeReferences.find(id);
    return found == _storeReferences.end() ? std::nullopt : std::optional(found->second);
}
std::optional<vc::objects::AttachmentIdentity> CState::sampleStore() const
{
    if (_objectStoreOwner) return _objectStoreOwner->sampleStore();
    if (!_vpkg || !_vpkg->document().data().contains("open_data")) return std::nullopt;
    return storeReference(_vpkg->document().data().at("open_data").at("store"));
}
std::optional<vc::objects::AttachmentIdentity> CState::attachedAt(const std::filesystem::path& path) const
{
    for (const auto& attachment : _objectStoreSession->attachments()) {
        std::error_code error;
        if (std::filesystem::equivalent(_objectStoreSession->path(attachment), path, error)) return attachment;
    }
    return std::nullopt;
}
vc::objects::AttachmentIdentity CState::prepareStore(const std::filesystem::path& path, const QString& author, bool readOnly)
{
    if (_objectStoreOwner) return _objectStoreOwner->prepareStore(path, author, readOnly);
    if (const auto existing = attachedAt(path)) return *existing;
    return _objectStoreSession->attach(path, storeOpenOptions(author, readOnly));
}
void CState::releasePreparedStore(const vc::objects::AttachmentIdentity& attachment)
{
    if (_objectStoreOwner) { _objectStoreOwner->releasePreparedStore(attachment); return; }
    if (std::find(_attachedStores.begin(), _attachedStores.end(), attachment) == _attachedStores.end())
        _objectStoreSession->detach(attachment);
}
void CState::closeProject() { setVpkg(nullptr); }
std::vector<vc::objects::AttachmentIdentity> CState::attachedStores() const
{
    return _objectStoreOwner ? _objectStoreOwner->attachedStores() : _attachedStores;
}
vc::objects::AttachmentIdentity CState::attachStore(const std::filesystem::path& path, const QString& author, bool readOnly)
{
    if (_objectStoreOwner) return _objectStoreOwner->attachStore(path, author, readOnly);
    if (!_vpkg) throw std::runtime_error("Open a project first");
    if (const auto existing = attachedAt(path)) {
        if (std::find(_attachedStores.begin(), _attachedStores.end(), *existing) != _attachedStores.end()) return *existing;
    }
    const auto attachment = prepareStore(path, author, readOnly);
    const auto id = vc::objects::util::newUuid();
    auto data = _vpkg->document().data();
    data["stores"].push_back({{"id", id}, {"location", _vpkg->document().location(path)}, {"read_only", readOnly}});
    try { _vpkg->document().replace(std::move(data)); }
    catch (...) { releasePreparedStore(attachment); throw; }
    _attachedStores.push_back(attachment);
    _storeReferences.emplace(id, attachment);
    pollObjectStoreChanges();
    emit storeRoutesChanged();
    return attachment;
}
void CState::detachStore(const vc::objects::AttachmentIdentity& attachment)
{
    if (_objectStoreOwner) { _objectStoreOwner->detachStore(attachment); return; }
    for (const auto& [id, store] : _storeReferences)
        if (store == attachment) { const auto copy = id; detachStoreReference(copy); return; }
}
void CState::detachStoreReference(const std::string& id)
{
    if (_objectStoreOwner) { _objectStoreOwner->detachStoreReference(id); return; }
    if (!_vpkg) return;
    auto data = _vpkg->document().data();
    auto& stores = data["stores"];
    if (stores.size() <= 1) throw std::runtime_error("A project must reference at least one store");
    if (prepareStoreSwitch && !prepareStoreSwitch()) throw std::runtime_error("Store switch cancelled: pending edits could not be saved");
    if (_pointCollectionSync && !_pointCollectionSync->flush())
        throw std::runtime_error("Pending point-collection edits could not be saved");
    stores.erase(std::remove_if(stores.begin(), stores.end(), [&](const auto& entry) { return entry.at("id") == id; }), stores.end());
    for (auto& selection : data["selections"]) if (selection == id) selection = nullptr;
    if (data.contains("open_data") && data["open_data"]["store"] == id) data.erase("open_data");
    _vpkg->document().replace(std::move(data));
    if (const auto attachment = storeReference(id)) {
        _objectStoreSession->detach(*attachment);
        std::erase(_attachedStores, *attachment);
    }
    _storeReferences.erase(id);
    pollObjectStoreChanges();
    emit storeRoutesChanged();
    emit umbilicusChanged();
}
void CState::replaceStoreReference(const std::string& id, const std::filesystem::path& path)
{
    if (_objectStoreOwner) { _objectStoreOwner->replaceStoreReference(id, path); return; }
    if (!_vpkg) return;
    auto data = _vpkg->document().data();
    auto entry = std::find_if(data["stores"].begin(), data["stores"].end(), [&](const auto& e) { return e.at("id") == id; });
    if (entry == data["stores"].end()) throw std::invalid_argument("Unknown store reference");
    if (prepareStoreSwitch && !prepareStoreSwitch()) throw std::runtime_error("Store switch cancelled");
    if (_pointCollectionSync && !_pointCollectionSync->flush())
        throw std::runtime_error("Pending point-collection edits could not be saved");
    const auto existing = attachedAt(path);
    if (existing && existing != storeReference(id)) throw std::invalid_argument("Store is already attached");
    const auto replacement = prepareStore(path, storeAuthor(), entry->value("read_only", false));
    (*entry)["location"] = _vpkg->document().location(path);
    try { _vpkg->document().replace(std::move(data)); }
    catch (...) { releasePreparedStore(replacement); throw; }
    if (const auto old = storeReference(id); old && old != replacement) {
        _objectStoreSession->detach(*old);
        std::erase(_attachedStores, *old);
    }
    if (std::find(_attachedStores.begin(), _attachedStores.end(), replacement) == _attachedStores.end()) _attachedStores.push_back(replacement);
    _storeReferences.insert_or_assign(id, replacement);
    pollObjectStoreChanges();
    emit storeRoutesChanged();
    emit umbilicusChanged();
}
const std::vector<std::string>& CState::routableKinds() { return vc::project::ProjectDocument::kinds(); }
std::optional<vc::objects::AttachmentIdentity> CState::storeFor(std::string_view kind) const
{
    if (_objectStoreOwner) return _objectStoreOwner->storeFor(kind);
    if (!_vpkg) return std::nullopt;
    const auto& selections = _vpkg->document().data().at("selections");
    const auto found = selections.find(std::string(kind));
    return found == selections.end() || found->is_null() ? std::nullopt : storeReference(found->get<std::string>());
}
std::vector<vc::objects::AttachmentIdentity> CState::surfaceSources() const
{
    if (const auto store = storeFor("surface"); store && _currentVolume) return {*store};
    return {};
}
void CState::setStoreFor(const std::string& kind, const std::optional<vc::objects::AttachmentIdentity>& store)
{
    if (_objectStoreOwner) { _objectStoreOwner->setStoreFor(kind, store); return; }
    if (!store) { setStoreSelection(kind, {}); return; }
    for (const auto& [id, attachment] : _storeReferences)
        if (attachment == store) { setStoreSelection(kind, id); return; }
    throw std::invalid_argument("Store is not attached to this project");
}
void CState::setStoreSelection(const std::string& kind, const std::string& id)
{
    if (_objectStoreOwner) { _objectStoreOwner->setStoreSelection(kind, id); return; }
    if (!_vpkg) throw std::runtime_error("Open a project first");
    auto data = _vpkg->document().data();
    if (!data["selections"].contains(kind)) throw std::invalid_argument("Unknown object kind");
    const nlohmann::json selected = id.empty() ? nlohmann::json(nullptr) : nlohmann::json(id);
    if (data["selections"][kind] == selected) return;
    if (prepareStoreSwitch && !prepareStoreSwitch()) throw std::runtime_error("Store switch cancelled: pending edits could not be saved");
    if (_pointCollectionSync && !_pointCollectionSync->flush())
        throw std::runtime_error("Pending point-collection edits could not be saved");
    data["selections"][kind] = selected;
    _vpkg->document().replace(std::move(data));
    emit storeRoutesChanged();
    emit umbilicusChanged();
}
void CState::restoreStoreRoutes()
{
    if (!_vpkg) return;
    QStringList failures;
    for (const auto& entry : _vpkg->document().data().at("stores")) {
        try {
            const auto path = _vpkg->document().resolve(entry.at("location"));
            const auto attachment = prepareStore(path, storeAuthor(), entry.value("read_only", false));
            if (std::find(_attachedStores.begin(), _attachedStores.end(), attachment) != _attachedStores.end())
                throw std::runtime_error("Store is referenced more than once");
            _attachedStores.push_back(attachment);
            _storeReferences.emplace(entry.at("id").get<std::string>(), attachment);
        } catch (const std::exception& error) {
            failures << QString::fromStdString(entry.at("location").get<std::string>()) + ": " + QString::fromUtf8(error.what());
        }
    }
    if (!failures.empty()) emit objectStoreObservationFailed(tr("Unavailable project stores:\n%1").arg(failures.join('\n')));
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
    if (changed && !_objectStoreOwner) {
        if (_pointCollectionSync && !_pointCollectionSync->flush())
            throw std::runtime_error("Pending point-collection edits could not be saved");
        for (const auto& store : _attachedStores) _objectStoreSession->detach(store);
        _attachedStores.clear();
        _storeReferences.clear();
    }
    _vpkg = std::move(pkg);
    if (changed && !_objectStoreOwner) {
        restoreStoreRoutes();
        pollObjectStoreChanges();
        emit storeRoutesChanged();
    }
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
    return scaledBox(*_focusBounds, _annotationVolume->scale);
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
    const auto normalized = scaledBox(bounds, 1.0 / _annotationVolume->scale);
    for (int axis = 0; axis < 3; ++axis)
        if (!std::isfinite(normalized.low[axis]) || !std::isfinite(normalized.high[axis]))
            return;
    if (_focusBounds && _focusBounds->low == normalized.low &&
        _focusBounds->high == normalized.high) {
        return;
    }
    if (_vpkg) _vpkg->setFocusBounds(normalized, _focusBoundsEnabled);
    _focusBounds = normalized;
    ++_focusBoundsRevision;
    emit focusBoundsChanged();
}

void CState::setFocusBoundsEnabled(bool enabled)
{
    if (_focusBoundsEnabled == enabled) {
        return;
    }
    if (_vpkg) _vpkg->setFocusBounds(_focusBounds, enabled);
    _focusBoundsEnabled = enabled;
    ++_focusBoundsRevision;
    emit focusBoundsChanged();
}

void CState::clearFocusBounds()
{
    if (!_focusBounds && !_focusBoundsEnabled) {
        return;
    }
    if (_vpkg) _vpkg->setFocusBounds(std::nullopt, false);
    _focusBounds.reset();
    _focusBoundsEnabled = false;
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
    if (_pointCollectionSync && !_pointCollectionSync->flush())
        throw std::runtime_error("Pending point-collection edits could not be saved");
    emit pointCollectionsSaveRequested();
}

void CState::closeAll()
{
    savePointCollections();
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
    if (!surface || !_annotationVolume || !vc::objects::util::isUuid(surface->id)) {
        return;
    }
    _objectStoreSession->setDraft(surface->id, quad_surface_grid(*surface),
                                  vc::objects::volumeCoordinates(*_annotationVolume));
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
