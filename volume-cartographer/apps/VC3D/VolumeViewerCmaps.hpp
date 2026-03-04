#pragma once

#include <QImage>
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
    cv::Vec3f tint; // R, G, B in [0,1]
};

struct OverlayColormapEntry
{
    QString label;
    std::string id;
};

const std::vector<OverlayColormapSpec>& specs();
const OverlayColormapSpec& resolve(const std::string& id);

// Apply colormap and write directly into a QImage::Format_RGB32 buffer.
// Avoids intermediate cv::Mat BGR and eliminates cvtColor conversions.
QImage makeColors(const cv::Mat_<uint8_t>& values, const OverlayColormapSpec& spec);

const std::vector<OverlayColormapEntry>& entries();

} // namespace volume_viewer_cmaps
