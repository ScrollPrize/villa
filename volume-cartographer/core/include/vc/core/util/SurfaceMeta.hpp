#pragma once

#include <filesystem>
#include <set>
#include <optional>
#include <vector>

#include <opencv2/core.hpp>
#include <nlohmann/json_fwd.hpp>

#include "Surface.hpp"
#include "QuadSurface.hpp"

class SurfaceMeta
{
public:
    SurfaceMeta() {};
    SurfaceMeta(const std::filesystem::path &path_, const nlohmann::json &json);
    SurfaceMeta(const std::filesystem::path &path_);
    ~SurfaceMeta();
    void readOverlapping();
    QuadSurface *surface();
    void setSurface(QuadSurface *surf, bool takeOwnership = true);
    std::string name();
    std::filesystem::path path;
    QuadSurface *_surf = nullptr;
    bool _ownsSurface = false;
    Rect3D bbox;
    nlohmann::json *meta = nullptr;
    std::set<std::string> overlapping_str;
    std::set<SurfaceMeta*> overlapping;
    std::optional<std::filesystem::file_time_type> maskTimestamp() const { return maskTimestamp_; }
    static std::optional<std::filesystem::file_time_type> readMaskTimestamp(const std::filesystem::path& dir);

private:
    void cacheMaskTimestamp();
    std::optional<std::filesystem::file_time_type> maskTimestamp_;
};

bool overlap(SurfaceMeta &a, SurfaceMeta &b, int max_iters = 1000);
bool contains(SurfaceMeta &a, const cv::Vec3f &loc, int max_iters = 1000);
bool contains(SurfaceMeta &a, const std::vector<cv::Vec3f> &locs);
bool contains_any(SurfaceMeta &a, const std::vector<cv::Vec3f> &locs);

void write_overlapping_json(const std::filesystem::path& seg_path, const std::set<std::string>& overlapping_names);
std::set<std::string> read_overlapping_json(const std::filesystem::path& seg_path);
