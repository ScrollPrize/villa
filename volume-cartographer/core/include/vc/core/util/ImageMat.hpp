#pragma once
// OpenCV Mats of image files read and written through vc_reader: the buffer conversion at
// VC's boundary. TIFF and PNG store RGB(A); Mats hold BGR(A).

#include <vc_reader/image_write.hpp>
#include <vc_reader/source_reader.hpp>

#include <opencv2/core/mat.hpp>

#include <cstdint>
#include <filesystem>

namespace vcr::sources {
struct Block;
}

namespace vc::core::util {

// The typestr of an OpenCV depth; nullptr for one TIFF cannot hold.
[[nodiscard]] const char* tiffDtype(int depth);

// A 2-D or rows x cols x channels block as a Mat in OpenCV's colour order; 1-bit samples
// become 0 and 255. Empty when the block's dtype has no OpenCV depth.
[[nodiscard]] cv::Mat imageMat(const vcr::sources::Block& block);

// Owns the contiguous RGB(A) buffer passed to native image writers.
struct ImageBuffer {
    cv::Mat pixels;
    [[nodiscard]] vcr::sources::ImageView view() const;
};
[[nodiscard]] ImageBuffer imageBuffer(const cv::Mat& mat);

}  // namespace vc::core::util
