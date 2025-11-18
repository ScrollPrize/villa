#pragma once

#include <QObject>
#include <opencv2/core.hpp>

#include <vc/core/util/HashFunctions.hpp>

#include "vc/core/util/Surface.hpp"


struct POI
{
    cv::Vec3f p = {0,0,0};
    Surface *src = nullptr;
    cv::Vec3f n = {0,0,0};
};

struct IntersectionLine
{
    std::vector<cv::Vec3f> world;         // 3D points in volume space
    std::vector<cv::Vec3f> surfaceParams; // QuadSurface ptr-space samples aligned with `world`
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
    void setSurface(const std::string &name, Surface*, bool noSignalSend = false, bool takeOwnership = true);
    void setPOI(const std::string &name, POI *poi);
    void setIntersection(const std::string &a, const std::string &b, Intersection *intersect);
    Surface *surface(const std::string &name);
    Intersection *intersection(const std::string &a, const std::string &b);
    POI *poi(const std::string &name);
    std::vector<Surface*> surfaces();
    std::vector<POI*> pois();
    std::vector<std::string> surfaceNames();
    std::vector<std::string> poiNames();
    std::vector<std::pair<std::string,std::string>> intersections(const std::string &a = "");
    
signals:
    void sendSurfaceChanged(std::string, Surface*);
    void sendPOIChanged(std::string, POI*);
    void sendIntersectionChanged(std::string, std::string, Intersection*);
    
protected:
    struct SurfaceEntry {
        Surface* ptr = nullptr;
        bool owns = true;
    };

    bool _regular_pan = false;
    std::unordered_map<std::string, SurfaceEntry> _surfs;
    std::unordered_map<std::string, POI*> _pois;
    std::unordered_map<std::pair<std::string,std::string>, Intersection*, string_pair_hash> _intersections;
};
