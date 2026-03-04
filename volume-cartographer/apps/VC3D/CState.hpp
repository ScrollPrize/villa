#pragma once

#include <QObject>
#include <QString>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <opencv2/core.hpp>

#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/ui/VCCollection.hpp"

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
    explicit CState(size_t cacheSizeBytes, size_t diskCacheSizeBytes = 100ULL << 30, QObject* parent = nullptr);
    ~CState();

    // --- VolumePkg ---
    std::shared_ptr<VolumePkg> vpkg() const;
    void setVpkg(std::shared_ptr<VolumePkg> pkg);
    QString vpkgPath() const;
    bool hasVpkg() const;

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
    std::shared_ptr<Volume> _currentVolume;
    std::string _currentVolumeId;
    std::string _segmentationGrowthVolumeId;
    std::weak_ptr<QuadSurface> _activeSurface;
    std::string _activeSurfaceId;

    VCCollection* _pointCollection;

    size_t _cacheSizeBytes;
    size_t _diskCacheSizeBytes;

    // Surface/POI data (formerly in CSurfaceCollection)
    std::unordered_map<std::string, std::shared_ptr<Surface>> _surfs;
    std::unordered_map<std::string, std::unique_ptr<POI>> _pois;
};
