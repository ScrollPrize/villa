#pragma once

#include <array>
#include <memory>

#include <QObject>
#include <opencv2/core.hpp>

#include "vc/core/util/Surface.hpp"


struct POI
{
    cv::Vec3f p = {0,0,0};
    std::string surfaceId;  // ID of the source surface (for lookup, not ownership)
    cv::Vec3f n = {0,0,0};
};

struct IntersectionLine
{
    std::array<cv::Vec3f, 2> world{};         // 3D points in volume space
    std::array<cv::Vec3f, 2> surfaceParams{}; // QuadSurface ptr-space samples aligned with `world`
};

struct Intersection
{
    std::vector<IntersectionLine> lines;
};



// This class shall handle all the (GUI) interactions for its stored objects but does not itself provide the GUI
// Slices: all the defined slices of all kinds
// Segmentators: segmentations and interactions with segments
// POIs : e.g. active constrol points or slicing focus points
class CSurfaceCollection : public QObject
{
    Q_OBJECT

public:
    ~CSurfaceCollection();

    // Surface management with shared_ptr
    void setSurface(const std::string &name, std::shared_ptr<Surface> surf, bool noSignalSend = false, bool isEditUpdate = false);
    std::shared_ptr<Surface> surface(const std::string &name);
    std::vector<std::shared_ptr<Surface>> surfaces();
    std::vector<std::string> surfaceNames();

    // Convenience for raw pointer access (for gradual migration)
    Surface* surfaceRaw(const std::string &name);

    // Find the ID for a given surface pointer
    std::string findSurfaceId(Surface* surf);

    void emitSurfacesChanged();  // Emit signal to notify listeners of batch surface changes
    void setPOI(const std::string &name, POI *poi);
    POI *poi(const std::string &name);
    std::vector<POI*> pois();
    std::vector<std::string> poiNames();

signals:
    void sendSurfaceChanged(std::string name, std::shared_ptr<Surface> surf, bool isEditUpdate = false);
    void sendSurfaceWillBeDeleted(std::string name, std::shared_ptr<Surface> surf);
    void sendPOIChanged(std::string, POI*);

protected:
    bool _regular_pan = false;
    std::unordered_map<std::string, std::shared_ptr<Surface>> _surfs;
    std::unordered_map<std::string, POI*> _pois;
};
