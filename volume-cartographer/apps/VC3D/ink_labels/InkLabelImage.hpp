#pragma once

#include <vc_reader/source_reader.hpp>
#include <opencv2/core.hpp>
#include <algorithm>
#include <filesystem>
#include <limits>
#include <stdexcept>

namespace vc3d::ink_labels {

struct LabelImageInfo {
    int width, height, bitsPerSample;
};

// Stream the first image page through the upstream reader. Retain storage sample
// order (RGB, not OpenCV's BGR): label imports deliberately use sample zero.
// Memory is bounded by a reader decode budget and one 1024-square crop.
template <typename Visitor>
bool forEachLabelImageBlock(const std::filesystem::path& path, Visitor&& visit, std::string* error)
{
    try {
        vcr::sources::SourceReader::Settings settings;
        settings.decodedCacheBytes = 0;
        settings.maxDecodeBytes = 256 * 1024 * 1024;
        vcr::sources::SourceReader reader(vcr::sources::imageSource(path.string(), {}, "tiff"), settings);
        const auto& axes = reader.axes();
        const auto& shape = reader.shape();
        const auto dimension = [&](const char* name) {
            auto it = std::find(axes.begin(), axes.end(), name);
            if (it == axes.end()) throw std::invalid_argument("label image needs x and y axes");
            auto n = shape.at(std::size_t(it - axes.begin()));
            if (n <= 0 || n > std::numeric_limits<int>::max()) throw std::invalid_argument("label image dimension out of range");
            return int(n);
        };
        const auto dtype = reader.dtype();
        const bool boolean = dtype == "|b1";
        const int depth = dtype == "|u1" || boolean ? CV_8U : dtype == "<u2" ? CV_16U : -1;
        if (depth < 0) throw std::invalid_argument("label TIFF must have unsigned 8-bit or 16-bit samples");
        LabelImageInfo info{dimension("x"), dimension("y"), depth == CV_16U ? 16 : 8};
        for (int y = 0; y < info.height; y += 1024) {
            for (int x = 0; x < info.width; x += 1024) {
                const int h = std::min(1024, info.height - y), w = std::min(1024, info.width - x);
                std::vector<std::int64_t> start, size;
                for (const auto& axis : reader.spatialAxes()) {
                    if (axis != "x" && axis != "y" && axis != "z") throw std::invalid_argument("unsupported label image axes");
                    start.push_back(axis == "y" ? y : axis == "x" ? x : 0);
                    size.push_back(axis == "y" ? h : axis == "x" ? w : 1);
                }
                auto block = reader.read(start, size);
                const auto z = std::find(axes.begin(), axes.end(), "z");
                if (z != axes.end()) block.shape.erase(block.shape.begin() + (z - axes.begin()));
                if (block.shape.size() < 2 || block.shape.size() > 3 || block.shape[0] != h || block.shape[1] != w)
                    throw std::invalid_argument("label image axes must be rows, columns, then samples");
                const int channels = block.shape.size() == 3 ? int(block.shape[2]) : 1;
                if (channels < 1 || channels > CV_CN_MAX) throw std::invalid_argument("unsupported label sample count");
                cv::Mat pixels(h, w, CV_MAKETYPE(depth, channels), block.data.get());
                if (boolean) pixels *= 255;
                if (!visit(info, cv::Point(x, y), pixels)) return false;
            }
        }
        return true;
    } catch (const std::exception& e) {
        if (error) *error = e.what();
        return false;
    }
}
} // namespace vc3d::ink_labels
