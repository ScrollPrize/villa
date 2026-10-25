#pragma once

#include <QObject>
#include <QString>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <opencv2/core/mat.hpp>

#include "vc/core/util/Rect3D.hpp"
#include "AnnotationFrame.hpp"
#include "vc_object_store/objects/ObjectStoreSession.hpp"

Q_DECLARE_METATYPE(vc::objects::StoreSessionEvent)

class VolumePkg;
class Volume;
class QuadSurface;
class Surface;
class VCCollection;

struct POI
{
    cv::Vec3f p = {0,0,0};
    std::string surfaceId;  // ID of the source surface (for lookup, not ownership)
    std::optional<cv::Vec3f> surfacePtr;  // Nominal source-surface pointer for exact viewer recentering.
    cv::Vec3f n = {0,0,0};
    bool suppressViewerRecenter = false;
    bool suppressTransientPlaneIntersections = false;
};

class CState : public QObject
{
    Q_OBJECT

public:
    explicit CState(QObject* parent = nullptr,
                    bool debugDownloadQueue = false, CState* objectStoreOwner = nullptr);
    ~CState();

    // One session for all object kinds; controllers share this handle.
    std::shared_ptr<vc::objects::ObjectStoreSession> objectStoreSession() const { return _objectStoreSession; }
    // The user's store in that session: the attachment edits go to.
    std::optional<vc::objects::AttachmentIdentity> objectStoreAttachment() const;
    void attachObjectStore(const QString& path, const QString& author, bool readOnly = false);
    void detachObjectStore();
    // A read-only catalog store (one per Open Data sample) beside the user's,
    // attached on first use; only the catalog importer writes to it.
    vc::objects::AttachmentIdentity catalogAttachment(const std::filesystem::path& path);
    void pollObjectStoreChanges();

    // --- VolumePkg ---
    std::shared_ptr<VolumePkg> vpkg() const;
    void setVpkg(std::shared_ptr<VolumePkg> pkg);
    QString vpkgPath() const;
    bool hasVpkg() const;

    // --- Current Volume ---
    std::shared_ptr<Volume> currentVolume() const;
    std::string currentVolumeId() const;
    void setCurrentVolume(std::shared_ptr<Volume> vol);
    vc3d::annotation::AnnotationFrame annotationFrame() const;

    // Public bounds use current-volume XYZ; storage uses annotation-frame XYZ.
    std::optional<Rect3D> focusBounds() const;
    std::optional<Rect3D> activeFocusBounds() const;
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

    bool debugDownloadQueueEnabled() const { return _debugDownloadQueue; }

    // --- Teardown ---
    void closeAll();

    // --- Surfaces (inlined from CSurfaceCollection) ---
    void setSurface(const std::string& name, std::shared_ptr<Surface> surf, bool noSignalSend = false, bool isEditUpdate = false);
    // Apply catalog-style surface additions/replacements/removals atomically
    // from observers' perspective. Individual surface signals are suppressed
    // and one empty-name surfaceChanged signal is emitted after the map is final.
    void setSurfacesBatch(
        const std::vector<std::pair<std::string, std::shared_ptr<Surface>>>& updates);
    std::shared_ptr<Surface> surface(const std::string& name);
    Surface* surfaceRaw(const std::string& name);
    std::string findSurfaceId(Surface* surf);
    std::vector<std::shared_ptr<Surface>> surfaces();
    std::vector<std::string> surfaceNames();
    void emitSurfacesChanged();
    // Bumped on every setSurface() that changes the map (including
    // noSignalSend calls), so callers can cheaply detect that a cached view
    // of the surface map is stale.
    uint64_t surfacesVersion() const { return _surfacesVersion; }

    // --- POIs (inlined from CSurfaceCollection) ---
    std::unique_ptr<POI> createSurfaceFocusPoi(QuadSurface& surface);
    void setPOI(const std::string& name, POI* poi);
    POI* poi(const std::string& name);
    std::vector<POI*> pois();
    std::vector<std::string> poiNames();

    // The umbilicus of the current volume: the one live umbilicus object in the
    // attached store bound to it, with its points in the volume's voxels. Its frame
    // was decided when it was stored. `error` says why there is none (nothing
    // attached or bound, or several bound). `token` (uuid@revision, else the
    // error) changes whenever the answer does.
    struct StoredUmbilicus {
        std::string uuid;
        std::string name;
        std::vector<cv::Vec3f> points;
        std::string error;
        std::string token;
    };
    StoredUmbilicus storedUmbilicus() const;

signals:
    void objectStoreChanged(const vc::objects::StoreSessionEvent& event);
    void objectStoreObservationFailed(const QString& error);
    void vpkgChanged(std::shared_ptr<VolumePkg> vpkg);
    // An umbilicus was committed, or a store was attached or detached.
    void umbilicusChanged();
    void volumeChanged(std::shared_ptr<Volume> volume, const std::string& volumeId);
    void focusBoundsChanged();
    void surfacesLoaded();
    void volumeClosing();

    // Surface/POI signals (formerly on CSurfaceCollection)
    void surfaceChanged(std::string name, std::shared_ptr<Surface> surf, bool isEditUpdate = false);
    void surfaceWillBeDeleted(std::string name, std::shared_ptr<Surface> surf);
    void poiChanged(std::string, POI*);

private:
    void resolveCurrentVolumeId();

    CState* _objectStoreOwner = nullptr;
    std::shared_ptr<vc::objects::ObjectStoreSession> _objectStoreSession;
    std::optional<vc::objects::AttachmentIdentity> _objectStoreAttachment;
    std::map<std::filesystem::path, vc::objects::AttachmentIdentity> _catalogAttachments;
    QString _objectStoreObservationError;
    std::shared_ptr<VolumePkg> _vpkg;
    std::shared_ptr<Volume> _currentVolume;
    std::string _currentVolumeId;
    std::optional<Rect3D> _focusBounds; // Annotation frame, independent of pyramid level.
    bool _focusBoundsEnabled = false;
    uint64_t _focusBoundsRevision = 0;
    std::string _segmentationGrowthVolumeId;
    std::weak_ptr<QuadSurface> _activeSurface;
    std::string _activeSurfaceId;

    VCCollection* _pointCollection;

    bool _debugDownloadQueue = false;

    // Surface/POI data (formerly in CSurfaceCollection)
    std::unordered_map<std::string, std::shared_ptr<Surface>> _surfs;
    std::unordered_map<std::string, std::unique_ptr<POI>> _pois;
    uint64_t _surfacesVersion = 0;
    int _surfaceBatchDepth = 0;
    bool _surfaceBatchChanged = false;
};
