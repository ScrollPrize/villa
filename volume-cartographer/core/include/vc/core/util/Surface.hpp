#pragma once

#include <filesystem>
#include <memory>
#include <string>

#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>

//base surface class
class Surface
{
public:
    virtual ~Surface() = default;

    // get a central location point
    virtual cv::Vec3f pointer() = 0;

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
    std::unique_ptr<nlohmann::json> meta;
    std::filesystem::path path;
    std::string id;
};
