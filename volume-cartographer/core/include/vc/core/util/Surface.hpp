#pragma once

#include <filesystem>
#include <memory>
#include <string>

#include <opencv2/core/mat.hpp>
#include "utils/Json.hpp"

//base surface class
class Surface
{
public:
    virtual ~Surface();

    // Returns a default starting point for surface iteration.
    // Subclasses may override but default returns origin.
    virtual cv::Vec3f pointer() { return {0, 0, 0}; }


    //move pointer within internal coordinate system
    virtual void move(cv::Vec3f &ptr, const cv::Vec3f &offset) = 0;
    //does the pointer location contain valid surface data
    virtual bool valid(const cv::Vec3f &ptr, const cv::Vec3f &offset = {0,0,0}) = 0;
    //nominal pointer coordinates (in "output" coordinates)
    virtual cv::Vec3f loc(const cv::Vec3f &ptr, const cv::Vec3f &offset = {0,0,0}) = 0;
    //read coord at pointer location, potentially with (3) offset
    virtual cv::Vec3f coord(const cv::Vec3f &ptr, const cv::Vec3f &offset = {0,0,0}) const = 0;
    virtual cv::Vec3f normal(const cv::Vec3f &ptr, const cv::Vec3f &offset = {0,0,0}) = 0;
    virtual float pointTo(cv::Vec3f &ptr, const cv::Vec3f &coord, float th, int max_iters = 1000,
                          class SurfacePatchIndex* surfaceIndex = nullptr, class PointIndex* pointIndex = nullptr) = 0;
    //coordgenerator relative to ptr&offset
    //needs to be deleted after use
    virtual void gen(cv::Mat_<cv::Vec3f> *coords, cv::Mat_<cv::Vec3f> *normals, cv::Size size, const cv::Vec3f &ptr, float scale, const cv::Vec3f &offset) const = 0;
    utils::Json meta;
    std::filesystem::path path;
    std::string id;
    // Directory under which rotating backups are written, as
    // <backupRoot>/backups/<id>/{0..N-1}/. VolumePkg sets this to the directory
    // holding the volpkg.json so backups stay a sibling of the project file no
    // matter where the segment itself lives (a segments dir, an explicit path,
    // etc.). When empty, saveSnapshot() falls back to the segment's parent dir.
    std::filesystem::path backupRoot;
};
