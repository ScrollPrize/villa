#pragma once

#include <QObject>
#include <QString>
#include <stdexcept>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <set>
#include <vector>

#include <opencv2/core/mat.hpp>
#include <vc_reader/catalog.hpp>

#include "vc/core/util/Rect3D.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc_object_store/objects/ObjectStoreSession.hpp"

Q_DECLARE_METATYPE(vc::objects::StoreSessionEvent)

class VolumePkg;
class Volume;
class OpenedVolumes;
class QuadSurface;
class Surface;
class VCCollection;
namespace vc3d { class PointCollectionSync; }

struct POI
{
    cv::Vec3f p = {0,0,0};
    std::string surfaceId;  // ID of the source surface (for lookup, not ownership)
    std::optional<cv::Vec3f> surfacePtr;  // Nominal source-surface pointer for exact viewer recentering.
    cv::Vec3f n = {0,0,0};
    bool suppressViewerRecenter = false;
};

class SessionDrafts;
// A store switch (detach, route, replace) was refused because pending edits
// could not be saved or an edit operation is running; nothing changed.
struct StoreSwitchRefused : std::runtime_error {
    using std::runtime_error::runtime_error;
};

class CState : public QObject
{
    Q_OBJECT

public:
    // cacheBytes sizes the opened volumes' memory cache (0: the RAM cache setting).
    explicit CState(QObject* parent = nullptr, CState* objectStoreOwner = nullptr, std::size_t cacheBytes = 0);
    ~CState();

    // One session for all object kinds; controllers share this handle.
    std::shared_ptr<vc::objects::ObjectStoreSession> objectStoreSession() const { return _objectStoreSession; }
    // The opened volume levels the tools sample, shared with the object-store owner's windows.
    std::shared_ptr<OpenedVolumes> openedVolumes() const { return _openedVolumes; }
    // Decoded surface geometry kept for every store VC3D opens (all workspaces),
    // up to the surface geometry cache setting; re-read after Settings change.
    static std::shared_ptr<vc::objects::SurfaceTileCache> surfaceGeometryCache();
    static void applySurfaceGeometryCacheSettings();
    // The author VC3D writes as: the username setting, else the login name.
    static QString storeAuthor();
    // Whether the username setting names an author (set, and not "anon").
    static bool storeAuthorConfigured();
    // Open Options for attaching stores to the viewer's session.
    static vc::objects::OpenOptions storeOpenOptions(const QString& author, bool readOnly);
    static vc::objects::HostedSessionOptions hostedOptions(const QString& author);

    // Project references are stable IDs; attachments exist only for available stores.
    std::optional<vc::objects::AttachmentIdentity> storeReference(const std::string& id) const;
    // The project's reference id of an attached store.
    std::optional<std::string> storeReferenceId(const vc::objects::AttachmentIdentity&) const;
    std::optional<vc::objects::AttachmentIdentity> sampleStore() const;
    vc::objects::AttachmentIdentity prepareStore(const std::filesystem::path&, const QString& author, bool readOnly = false);
    void releasePreparedStore(const vc::objects::AttachmentIdentity&);
    void closeProject();
    std::vector<vc::objects::AttachmentIdentity> attachedStores() const;
    vc::objects::AttachmentIdentity attachStore(const std::filesystem::path&, const QString& author, bool readOnly = false);
    // A store served by vc_object_server (e.g. http://127.0.0.1:8750 through
    // an SSH tunnel), added to the project as a hosted entry.
    vc::objects::AttachmentIdentity attachHostedStore(const std::string& endpoint, const QString& author);
    void detachStore(const vc::objects::AttachmentIdentity&);
    void detachStoreReference(const std::string& id);
    void replaceStoreReference(const std::string& id, const std::filesystem::path& path);
    static const std::vector<std::string>& routableKinds();
    std::optional<vc::objects::AttachmentIdentity> storeFor(std::string_view kind) const;
    std::vector<vc::objects::AttachmentIdentity> surfaceSources() const;
    void setStoreFor(const std::string& kind, const std::optional<vc::objects::AttachmentIdentity>& store);
    void setStoreSelection(const std::string& kind, const std::string& id);
    // Called before a source is changed or detached; failure leaves it unchanged.
    std::function<bool()> prepareStoreSwitch;
    // The same for a change that only affects the routes of these object kinds
    // (a selection change, or a store reference removed or replaced that these
    // kinds select); unset: prepareStoreSwitch.
    std::function<bool(const std::set<std::string>& kinds)> prepareStoreKindsSwitch;
    bool prepareStoreSwitchForKinds(const std::set<std::string>& kinds);
    // The object kinds whose selection names this store reference.
    std::set<std::string> kindsSelecting(const std::string& id) const;
    void pollObjectStoreChanges();
    // Crash recovery of the session's drafts (not started until the
    // application gives it a directory; the object-store owner's).
    SessionDrafts* sessionDrafts() const;

    // --- VolumePkg ---
    std::shared_ptr<VolumePkg> vpkg() const;
    // Replace the project and its store attachments; nullptr closes it.
    void setVpkg(std::shared_ptr<VolumePkg> pkg);
    // The project file directory (relative locations resolve against it).
    QString projectDirectory() const;
    bool hasVpkg() const;

    // --- Current Volume ---
    std::shared_ptr<Volume> currentVolume() const;
    std::string currentVolumeId() const;
    void setCurrentVolume(std::shared_ptr<Volume> vol);
    // The volume the current volume's annotations are kept in
    // (vc::project::annotationVolume); nothing without a volume.
    const std::optional<vc::project::AnnotationVolume>& annotationVolume() const { return _annotationVolume; }

    // Public bounds use current-volume XYZ; storage uses annotation-volume XYZ.
    std::optional<Rect3D> focusBounds() const;
    std::optional<Rect3D> activeFocusBounds() const;
    // The active bounds in the annotation volume's voxels.
    std::optional<Rect3D> activeAnnotationFocusBounds() const;
    bool focusBoundsEnabled() const;
    uint64_t focusBoundsRevision() const;
    void setFocusBounds(const Rect3D& bounds);
    void setFocusBoundsEnabled(bool enabled);
    void clearFocusBounds();

    // --- Growth Volume ---
    std::string segmentationGrowthVolumeId() const;
    void setSegmentationGrowthVolumeId(const std::string& id);

    // --- Active Surface ---
    std::weak_ptr<QuadSurface> activeSurface() const;
    std::string activeSurfaceId() const;
    void setActiveSurface(const std::string& id, std::shared_ptr<QuadSurface> surf);
    void clearActiveSurface();

    // --- Collections ---
    VCCollection* pointCollection() const;
    // Write pending point-collection edits to the store now: the volume's,
    // and the open surface's corrections. Failure vetoes close/switch.
    void savePointCollections();
    std::function<bool()> saveSurfaceCorrections;
    // The store one of the volume's point collections is kept in; nullopt for
    // a surface's corrections and in windows without their own collections.
    std::optional<vc::objects::AttachmentIdentity> pointCollectionStore(uint64_t collectionId) const;

    // --- Teardown ---
    void closeAll(const std::function<void()>& afterSave = {});

    // --- Surfaces (inlined from CSurfaceCollection) ---
    void setSurface(const std::string& name, std::shared_ptr<Surface> surf, bool noSignalSend = false, bool isEditUpdate = false);
    // While an edit session publishes the open surface's changed rectangles to
    // its draft itself, edit updates do not re-draft the whole surface.
    void setActiveSurfaceEditing(bool editing) { _activeSurfaceEditing = editing; }
    // The open surface's draft after a recovered draft of it was resolved:
    // `restored` (the session's draft now holds the surface) or discarded
    // (draft the open surface again).
    void recoveredDraftResolved(const std::string& objectUuid, bool restored);
    // Apply catalog-style surface additions/replacements/removals atomically
    // from observers' perspective. Individual surface signals are suppressed
    // and one empty-name surfaceChanged signal is emitted after the map is final.
    void setSurfacesBatch(
        const std::vector<std::pair<std::string, std::shared_ptr<Surface>>>& updates);
    std::shared_ptr<Surface> surface(const std::string& name);
    Surface* surfaceRaw(const std::string& name);
    std::string findSurfaceId(Surface* surf);
    std::vector<std::shared_ptr<Surface>> surfaces();
    // Bumped on every setSurface() that changes the map (including
    // noSignalSend calls), so callers can cheaply detect that a cached view
    // of the surface map is stale.
    uint64_t surfacesVersion() const { return _surfacesVersion; }

    // --- POIs (inlined from CSurfaceCollection) ---
    std::unique_ptr<POI> createSurfaceFocusPoi(QuadSurface& surface);
    void setPOI(const std::string& name, POI* poi);
    POI* poi(const std::string& name);

signals:
    void objectStoreChanged(const vc::objects::StoreSessionEvent& event);
    void objectStoreObservationFailed(const QString& error);
    // The stores beside the project, or the store a kind is routed to, changed.
    void storeRoutesChanged();
    void vpkgChanged(std::shared_ptr<VolumePkg> vpkg);
    // An umbilicus was committed, or a store was attached or detached.
    void umbilicusChanged();
    // The open surface has a draft recovered from an earlier session; it
    // stays undrafted until recoveredDraftResolved().
    void recoveredDraftFound(const QString& objectUuid);
    void volumeChanged(std::shared_ptr<Volume> volume, const std::string& volumeId);
    void focusBoundsChanged();
    void surfacesLoaded();
    void volumeClosing();

    // Surface/POI signals (formerly on CSurfaceCollection)
    void surfaceChanged(std::string name, std::shared_ptr<Surface> surf, bool isEditUpdate = false);
    void surfaceWillBeDeleted(std::string name, std::shared_ptr<Surface> surf);
    void poiChanged(std::string, POI*);

private:
    void updateAnnotationVolume();
    // The open ("segmentation") surface is in memory: while open it is a
    // draft of the session (in the current volume's voxels), so store
    // queries read it there rather than from its store.
    // `saved`: the surface holds its committed geometry (just opened), so its
    // draft needs no checkpoint until it changes.
    void draftActiveSurface(const std::shared_ptr<QuadSurface>& surface, bool saved = false);
    std::string _activeSurfaceDraft;
    SessionDrafts* _sessionDrafts = nullptr;
    bool _activeSurfaceEditing = false;

    void restoreStoreRoutes();
    std::optional<vc::objects::AttachmentIdentity> attachedAt(const std::filesystem::path& path) const;
    CState* _objectStoreOwner = nullptr;
    std::shared_ptr<vc::objects::ObjectStoreSession> _objectStoreSession;
    std::shared_ptr<OpenedVolumes> _openedVolumes;
    std::vector<vc::objects::AttachmentIdentity> _attachedStores;
    std::map<std::string, vc::objects::AttachmentIdentity> _storeReferences;
    QString _objectStoreObservationError;
    std::shared_ptr<VolumePkg> _vpkg;
    std::shared_ptr<Volume> _currentVolume;

    std::optional<vc::project::AnnotationVolume> _annotationVolume;
    std::optional<Rect3D> _focusBounds; // In the annotation volume, independent of pyramid level.
    bool _focusBoundsEnabled = false;
    uint64_t _focusBoundsRevision = 0;
    std::string _segmentationGrowthVolumeId;
    std::weak_ptr<QuadSurface> _activeSurface;
    std::string _activeSurfaceId;

    VCCollection* _pointCollection;
    // Keeps the volume's point collections in the store (the main state only).
    vc3d::PointCollectionSync* _pointCollectionSync = nullptr;

    // Surface/POI data (formerly in CSurfaceCollection)
    std::unordered_map<std::string, std::shared_ptr<Surface>> _surfs;
    std::unordered_map<std::string, std::unique_ptr<POI>> _pois;
    uint64_t _surfacesVersion = 0;
    int _surfaceBatchDepth = 0;
    bool _surfaceBatchChanged = false;
};
