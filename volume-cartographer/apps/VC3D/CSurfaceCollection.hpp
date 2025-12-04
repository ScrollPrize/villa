#pragma once

#include <array>

#include <QObject>
#include <opencv2/core.hpp>

#include "vc/core/util/Surface.hpp"


struct POI
{
    cv::Vec3f p = {0,0,0};
    Surface *src = nullptr;
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
    void setSurface(const std::string &name, Surface*, bool noSignalSend = false, bool takeOwnership = true, bool isEditUpdate = false);
    void emitSurfacesChanged();  // Emit signal to notify listeners of batch surface changes
    void setPOI(const std::string &name, POI *poi);
    Surface *surface(const std::string &name);
    POI *poi(const std::string &name);
    std::vector<Surface*> surfaces();
    std::vector<POI*> pois();
    std::vector<std::string> surfaceNames();
    std::vector<std::string> poiNames();
    
signals:
    void sendSurfaceChanged(std::string, Surface*, bool isEditUpdate = false);
    void sendPOIChanged(std::string, POI*);
    
protected:
    struct SurfaceEntry {
        Surface* ptr = nullptr;
        bool owns = true;
    };

    bool _regular_pan = false;
    std::unordered_map<std::string, SurfaceEntry> _surfs;
    std::unordered_map<std::string, POI*> _pois;
};
