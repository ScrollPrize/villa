#pragma once

#include <QObject>
#include <QString>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <opencv2/core.hpp>

#include "vc/core/cache/HttpMetadataFetcher.hpp"
#include "vc/core/types/Project.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/util/RemoteScroll.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/ui/VCCollection.hpp"
#include "SurfaceLRU.hpp"

struct POI
{
    cv::Vec3f p = {0,0,0};
    std::string surfaceId;  // ID of the source surface (for lookup, not ownership)
    cv::Vec3f n = {0,0,0};
};

class CState : public QObject
{
    Q_OBJECT

public:
    explicit CState(size_t cacheSizeBytes, QObject* parent = nullptr);
    ~CState();

    // --- VolumePkg ---
    std::shared_ptr<VolumePkg> vpkg() const;
    void setVpkg(std::shared_ptr<VolumePkg> pkg);
    QString vpkgPath() const;
    bool hasVpkg() const;

    // --- Project (flexible JSON-backed descriptor; shadows vpkg for now) ---
    std::shared_ptr<vc::Project> project() const;
    void setProject(std::shared_ptr<vc::Project> proj);
    bool hasProject() const;

    // Path lookups that prefer the active Project, falling back to vpkg.
    // Empty return means "not resolvable" — callers must check.
    std::filesystem::path segmentsPath(const std::string& idOrName) const;
    std::filesystem::path activeSegmentsPath() const;
    std::filesystem::path outputSegmentsPath() const;
    std::filesystem::path volumesPath() const;
    std::vector<std::filesystem::path> allSegmentsPaths() const;
    // Sibling file (e.g. seed.json, trace_params.json). Empty if nothing
    // can resolve it (no project and no vpkg).
    std::filesystem::path supportFilePath(const std::string& filename) const;

    // --- Remote segment registry (for lazy list-then-click downloads) ---
    // Populated when a remote SegmentsDir is loaded with metadata only.
    // Looked up by CWindow::downloadRemoteSegmentOnDemand so the correct
    // base URL / auth / cache / source layout is used per segment.
    struct RemoteSegmentInfo {
        std::string baseUrl;
        std::string cachePath;
        vc::cache::HttpAuth auth;
        vc::RemoteSegmentSource source = vc::RemoteSegmentSource::Direct;
    };
    void registerRemoteSegment(const std::string& segId, RemoteSegmentInfo info);
    void clearRemoteSegmentRegistry();
    bool hasRemoteSegmentInfo(const std::string& segId) const;
    RemoteSegmentInfo remoteSegmentInfo(const std::string& segId) const;

    // --- Current Volume ---
    std::shared_ptr<Volume> currentVolume() const;
    std::string currentVolumeId() const;
    void setCurrentVolume(std::shared_ptr<Volume> vol);

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

    // --- Cache budget ---
    size_t cacheSizeBytes() const;

    // --- Teardown ---
    void closeAll();

    // --- Surface LRU (point-grid eviction) ---
    SurfaceLRU& surfaceLRU() { return _surfaceLRU; }

    // --- Surfaces (inlined from CSurfaceCollection) ---
    void setSurface(const std::string& name, std::shared_ptr<Surface> surf, bool noSignalSend = false, bool isEditUpdate = false);
    std::shared_ptr<Surface> surface(const std::string& name);
    Surface* surfaceRaw(const std::string& name);
    std::string findSurfaceId(Surface* surf);
    std::vector<std::shared_ptr<Surface>> surfaces();
    std::vector<std::string> surfaceNames();
    void emitSurfacesChanged();

    // --- POIs (inlined from CSurfaceCollection) ---
    void setPOI(const std::string& name, POI* poi);
    POI* poi(const std::string& name);
    std::vector<POI*> pois();
    std::vector<std::string> poiNames();

signals:
    void vpkgChanged(std::shared_ptr<VolumePkg> vpkg);
    void projectChanged(std::shared_ptr<vc::Project> project);
    void volumeChanged(std::shared_ptr<Volume> volume, const std::string& volumeId);
    void surfacesLoaded();
    void volumeClosing();

    // Surface/POI signals (formerly on CSurfaceCollection)
    void surfaceChanged(std::string name, std::shared_ptr<Surface> surf, bool isEditUpdate = false);
    void surfaceWillBeDeleted(std::string name, std::shared_ptr<Surface> surf);
    void poiChanged(std::string, POI*);

private:
    void applyCacheBudget(const std::shared_ptr<Volume>& vol) const;
    void resolveCurrentVolumeId();

    std::shared_ptr<VolumePkg> _vpkg;
    std::shared_ptr<vc::Project> _project;
    std::unordered_map<std::string, RemoteSegmentInfo> _remoteSegmentInfo;
    std::shared_ptr<Volume> _currentVolume;
    std::string _currentVolumeId;
    std::string _segmentationGrowthVolumeId;
    std::weak_ptr<QuadSurface> _activeSurface;
    std::string _activeSurfaceId;

    VCCollection* _pointCollection;

    size_t _cacheSizeBytes;

    // Surface/POI data (formerly in CSurfaceCollection)
    std::unordered_map<std::string, std::shared_ptr<Surface>> _surfs;
    std::unordered_map<std::string, std::unique_ptr<POI>> _pois;

    // See SurfaceLRU default — 4 resident × ~170 MiB/surface cap.
    SurfaceLRU _surfaceLRU{4};
};
