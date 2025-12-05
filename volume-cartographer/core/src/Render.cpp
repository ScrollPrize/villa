#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/Slicing.hpp"
#include <opencv2/imgproc.hpp>


void render_binary_mask(QuadSurface* surf,
                         cv::Mat_<uint8_t>& mask,
                         cv::Mat_<cv::Vec3f>& coords_out,
                         float scale) {

    // Get raw mesh vertices - this is the actual defined surface
    cv::Mat_<cv::Vec3f> rawPts = surf->rawPoints();
    cv::Size rawSize = rawPts.size();

    // Calculate target size: scale the raw size by the user's requested scale
    cv::Size targetSize(
        static_cast<int>(std::round(rawSize.width * scale)),
        static_cast<int>(std::round(rawSize.height * scale))
    );

    std::cout << "render_binary_mask: rawSize=" << rawSize
              << " targetSize=" << targetSize
              << " scale=" << scale << std::endl;

    // Create mask from raw points at their native resolution
    cv::Mat_<uint8_t> rawMask = surf->validMask();
    int rawValid = cv::countNonZero(rawMask);

    // Upscale the mask using nearest neighbor to target resolution
    cv::resize(rawMask, mask, targetSize, 0, 0, cv::INTER_NEAREST);

    // Generate coords at target resolution for rendering
    // Use surface's scale divided by user scale, so when scale=1.0:
    // genScale = surf->_scale (e.g. 0.05), and sx = 0.05/0.05 = 1.0
    // This samples 1:1 from the raw points grid
    cv::Vec3f ptr = surf->pointer();
    cv::Vec3f offset(-rawSize.width/2.0f, -rawSize.height/2.0f, 0);
    float genScale = surf->_scale[0] / scale;
    surf->gen(&coords_out, nullptr, targetSize, ptr, genScale, offset);

    int finalValid = cv::countNonZero(mask);
    std::cout << "  rawValid=" << rawValid << "/" << (rawSize.width * rawSize.height)
              << " (" << (100.0 * rawValid / (rawSize.width * rawSize.height)) << "%)"
              << " targetValid=" << finalValid << "/" << (targetSize.width * targetSize.height)
              << " (" << (100.0 * finalValid / (targetSize.width * targetSize.height)) << "%)" << std::endl;

    std::cout << "  ptr=" << ptr << ", offset=" << offset << ", genScale=" << genScale
              << ", surf->_scale=" << surf->_scale << std::endl;

    // Log coordinate bounds
    if (coords_out.rows > 8 && coords_out.cols > 8) {
        cv::Vec3f minCoord(FLT_MAX, FLT_MAX, FLT_MAX);
        cv::Vec3f maxCoord(-FLT_MAX, -FLT_MAX, -FLT_MAX);
        int validCoords = 0;
        for (int y = 4; y < coords_out.rows - 4; y++) {
            for (int x = 4; x < coords_out.cols - 4; x++) {
                const cv::Vec3f& c = coords_out(y, x);
                if (std::isfinite(c[0]) && std::isfinite(c[1]) && std::isfinite(c[2])) {
                    minCoord[0] = std::min(minCoord[0], c[0]);
                    minCoord[1] = std::min(minCoord[1], c[1]);
                    minCoord[2] = std::min(minCoord[2], c[2]);
                    maxCoord[0] = std::max(maxCoord[0], c[0]);
                    maxCoord[1] = std::max(maxCoord[1], c[1]);
                    maxCoord[2] = std::max(maxCoord[2], c[2]);
                    validCoords++;
                }
            }
        }
        cv::Vec3f span = maxCoord - minCoord;
        std::cout << "  coords bounds: min=" << minCoord << ", max=" << maxCoord
                  << ", span=" << span << ", validCoords=" << validCoords << std::endl;
    }
}

void render_image_from_coords(const cv::Mat_<cv::Vec3f>& coords,
                              cv::Mat_<uint8_t>& img,
                              z5::Dataset* ds,
                              ChunkCache<uint8_t>* cache) {
    if (!ds || !cache) {
        throw std::runtime_error("Dataset or cache is null in render_image_from_coords");
    }

    readInterpolated3D(img, ds, coords, cache);
    std::cout << "render_image_from_coords: completed" << std::endl;
}

// Render surface - generates both mask and image
void render_surface_image(QuadSurface* surf,
                         cv::Mat_<uint8_t>& mask,
                         cv::Mat_<uint8_t>& img,
                         z5::Dataset* ds,
                         ChunkCache<uint8_t>* cache,
                         float scale) {

    cv::Mat_<cv::Vec3f> coords;
    render_binary_mask(surf, mask, coords, scale);
    render_image_from_coords(coords, img, ds, cache);

    std::cout << "render_surface_image: completed" << std::endl;
}