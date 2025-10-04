#pragma once

#include <QString>
#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace volume_viewer_cmaps
{

enum class OverlayColormapKind { OpenCv, Tint };

struct OverlayColormapSpec
{
    std::string id;
    QString label;
    OverlayColormapKind kind;
    int opencvCode;
    cv::Vec3f tint; // B, G, R in [0,1]
};

struct OverlayColormapEntry
{
    QString label;
    std::string id;
};

const std::vector<OverlayColormapSpec>& specs();
const OverlayColormapSpec& resolve(const std::string& id);
cv::Mat makeColors(const cv::Mat_<uint8_t>& values, const OverlayColormapSpec& spec);
const std::vector<OverlayColormapEntry>& entries();

} // namespace volume_viewer_cmaps
